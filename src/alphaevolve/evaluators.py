"""Generic evaluator modules and optional sandbox execution backends."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any

from alphaevolve.errors import CapabilityUnavailableError
from alphaevolve.llm import AsyncInferenceClient
from alphaevolve.models import (
    Artifact,
    ArtifactRecord,
    EvaluationResult,
    EvaluatorConfig,
    ExecutionResult,
    Program,
    SandboxConfig,
    StageResult,
)


class SandboxExecutor(ABC):
    """Executes a candidate program inside a configured runtime."""

    @abstractmethod
    async def execute(
        self,
        program_path: Path,
        work_dir: Path,
        *,
        primary_metric: str,
    ) -> tuple[dict[str, float], ExecutionResult]:
        """Run the candidate and return parsed metrics plus execution metadata."""

    async def aclose(self) -> None:
        """Optional cleanup hook."""


class PythonSubprocessSandbox(SandboxExecutor):
    """Execute candidates as local Python subprocesses."""

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config

    async def execute(
        self,
        program_path: Path,
        work_dir: Path,
        *,
        primary_metric: str,
    ) -> tuple[dict[str, float], ExecutionResult]:
        started = asyncio.get_running_loop().time()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(program_path.name),
            cwd=str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self._config.timeout_seconds,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            execution = ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode,
                duration_ms=(asyncio.get_running_loop().time() - started) * 1_000,
                status="success" if process.returncode == 0 else "error",
            )
            if process.returncode != 0:
                return {primary_metric: 0.0}, execution
            return _metrics_from_stdout(stdout, primary_metric), execution
        except TimeoutError:
            process.kill()
            await process.communicate()
            return (
                {primary_metric: 0.0},
                ExecutionResult(
                    exit_code=None,
                    duration_ms=(asyncio.get_running_loop().time() - started) * 1_000,
                    status="timeout",
                ),
            )


class DockerGVisorSandbox(SandboxExecutor):
    """Evaluate programs inside a locked-down Docker container."""

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._docker_client = None

    async def execute(
        self,
        program_path: Path,
        work_dir: Path,
        *,
        primary_metric: str,
    ) -> tuple[dict[str, float], ExecutionResult]:
        return await asyncio.to_thread(
            self._execute_blocking,
            program_path,
            work_dir,
            primary_metric,
        )

    def _execute_blocking(
        self,
        program_path: Path,
        work_dir: Path,
        primary_metric: str,
    ) -> tuple[dict[str, float], ExecutionResult]:
        docker = self._docker_module()
        client = self._client(docker)
        started = perf_counter()
        container = None
        try:
            container = client.containers.run(
                self._config.image,
                command=["python", f"{self._config.working_dir}/{program_path.name}"],
                detach=True,
                network_mode=self._config.network_mode,
                read_only=self._config.read_only_root,
                runtime=self._config.runtime,
                mem_limit=self._config.mem_limit,
                cpu_quota=self._config.cpu_quota,
                working_dir=self._config.working_dir,
                tmpfs={self._config.tmpfs_path: f"rw,size={self._config.tmpfs_size_bytes}"},
                volumes={
                    str(work_dir): {
                        "bind": self._config.working_dir,
                        "mode": "rw",
                    }
                },
            )
            wait_result, wait_status = self._wait_for_container(container)
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            execution = ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=wait_result,
                duration_ms=(perf_counter() - started) * 1_000,
                status=wait_status,
            )
            if wait_status != "success":
                return {primary_metric: 0.0}, execution
            return _metrics_from_stdout(stdout, primary_metric), execution
        except Exception as exc:
            return (
                {primary_metric: 0.0},
                ExecutionResult(
                    stderr=str(exc),
                    exit_code=1,
                    duration_ms=(perf_counter() - started) * 1_000,
                    status="error",
                    exception=str(exc),
                ),
            )
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _wait_for_container(self, container) -> tuple[int | None, str]:
        async def _wait_once():
            return await asyncio.wait_for(
                asyncio.to_thread(container.wait),
                timeout=self._config.timeout_seconds,
            )

        try:
            result = asyncio.run(_wait_once())
            exit_code = int(result.get("StatusCode", 1))
            status = "success" if exit_code == 0 else "error"
            return exit_code, status
        except TimeoutError:
            container.kill()
            return None, "timeout"

    def _docker_module(self):
        try:
            import docker  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise CapabilityUnavailableError("Install the docker extra to use the Docker sandbox.") from exc
        return docker

    def _client(self, docker):
        if self._docker_client is None:
            self._docker_client = docker.from_env()
        return self._docker_client

    async def aclose(self) -> None:
        if self._docker_client is not None:
            await asyncio.to_thread(self._docker_client.close)


class EvaluationContext:
    """Runtime context passed to evaluator modules."""

    def __init__(
        self,
        *,
        program: Program,
        program_path: Path,
        work_dir: Path,
        primary_metric: str,
        sandbox: SandboxExecutor,
    ) -> None:
        self.program = program
        self.program_path = program_path
        self.work_dir = work_dir
        self.primary_metric = primary_metric
        self._sandbox = sandbox

    async def execute_candidate(self) -> tuple[dict[str, float], ExecutionResult]:
        """Run the candidate program via the configured sandbox."""
        return await self._sandbox.execute(
            self.program_path,
            self.work_dir,
            primary_metric=self.primary_metric,
        )


class Evaluator(ABC):
    """Abstract evaluator interface."""

    @abstractmethod
    async def evaluate(
        self,
        program: Program,
        *,
        primary_metric: str,
        artifact_dir: Path,
    ) -> EvaluationResult:
        """Evaluate a program and return a structured result."""

    async def aclose(self) -> None:
        """Optional cleanup hook."""


class ModuleEvaluator(Evaluator):
    """Load evaluator functions from a Python module."""

    def __init__(
        self,
        config: EvaluatorConfig,
        sandbox: SandboxExecutor,
        *,
        feedback_client: AsyncInferenceClient | None = None,
    ) -> None:
        self._config = config
        self._sandbox = sandbox
        self._feedback_client = feedback_client
        self._module = self._load_module(config.module)

    async def evaluate(
        self,
        program: Program,
        *,
        primary_metric: str,
        artifact_dir: Path,
    ) -> EvaluationResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{program.id}_") as temp_dir:
            work_dir = Path(temp_dir)
            program_path = work_dir / "program.py"
            program_path.write_text(program.code, encoding="utf-8")
            context = EvaluationContext(
                program=program,
                program_path=program_path,
                work_dir=work_dir,
                primary_metric=primary_metric,
                sandbox=self._sandbox,
            )

            if self._has_cascade():
                result = await self._evaluate_stages(context, artifact_dir, primary_metric=primary_metric)
            else:
                raw_result = await self._invoke_function(
                    getattr(self._module, "evaluate"),
                    context=context,
                    previous_result=None,
                    stage_results=(),
                )
                result = self._normalize_evaluation_result(raw_result, primary_metric=primary_metric)
                stage_artifacts = tuple(
                    artifact
                    for stage in result.stage_results
                    for artifact in stage.artifacts
                )
                persisted_artifacts = self._persist_artifacts(
                    stage_artifacts,
                    artifact_dir,
                    prefix="evaluate",
                )
                if persisted_artifacts or result.artifacts:
                    result = EvaluationResult(
                        status=result.status,
                        primary_score=result.primary_score,
                        metrics=result.metrics,
                        features=result.features,
                        execution=result.execution,
                        stage_results=result.stage_results,
                        artifacts=persisted_artifacts or result.artifacts,
                        feedback=result.feedback,
                        rejection_reason=result.rejection_reason,
                    )

        feedback = await self._maybe_generate_feedback(program, result)
        if feedback is not None:
            result = EvaluationResult(
                status=result.status,
                primary_score=result.primary_score,
                metrics=result.metrics,
                features=result.features,
                execution=result.execution,
                stage_results=result.stage_results,
                artifacts=result.artifacts,
                feedback=feedback,
                rejection_reason=result.rejection_reason,
            )
        return result

    async def aclose(self) -> None:
        await self._sandbox.aclose()

    def _load_module(self, module_path: Path) -> ModuleType:
        module_dir = str(module_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        module_name = f"alphaevolve_eval_{module_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise CapabilityUnavailableError(f"Unable to load evaluator module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _has_cascade(self) -> bool:
        if not self._config.stages.enabled:
            return False
        return hasattr(self._module, "evaluate_stage1")

    async def _evaluate_stages(
        self,
        context: EvaluationContext,
        artifact_dir: Path,
        *,
        primary_metric: str,
    ) -> EvaluationResult:
        stage_results: list[StageResult] = []
        cumulative_metrics: dict[str, float] = {}
        cumulative_features: dict[str, float] = {}
        persisted_artifacts: list[ArtifactRecord] = []
        last_execution = ExecutionResult()
        last_score = float("-inf")
        final_status = "success"
        rejection_reason: str | None = None
        previous_result: StageResult | None = None

        for index in range(1, self._config.stages.max_stages + 1):
            function_name = f"evaluate_stage{index}"
            if not hasattr(self._module, function_name):
                break
            raw_result = await self._invoke_function(
                getattr(self._module, function_name),
                context=context,
                previous_result=previous_result,
                stage_results=tuple(stage_results),
            )
            stage_result = self._normalize_stage_result(
                raw_result,
                stage_name=function_name,
                primary_metric=primary_metric,
            )
            stage_results.append(stage_result)
            cumulative_metrics.update(stage_result.metrics)
            cumulative_features.update(stage_result.features)
            if stage_result.primary_score is not None:
                last_score = stage_result.primary_score
            elif primary_metric in stage_result.metrics:
                last_score = stage_result.metrics[primary_metric]
            if stage_result.execution.status != "pending" or stage_result.execution.duration_ms:
                last_execution = stage_result.execution
            persisted_artifacts.extend(
                self._persist_artifacts(stage_result.artifacts, artifact_dir, prefix=function_name)
            )
            previous_result = stage_result
            if stage_result.status != "success" or not stage_result.should_continue:
                final_status = stage_result.status
                rejection_reason = stage_result.rejection_reason
                break

        if last_score == float("-inf"):
            last_score = cumulative_metrics.get(primary_metric, float("-inf"))
        if final_status == "success" and stage_results and stage_results[-1].status != "success":
            final_status = stage_results[-1].status
        return EvaluationResult(
            status=final_status,
            primary_score=last_score,
            metrics=cumulative_metrics,
            features=cumulative_features,
            execution=last_execution,
            stage_results=tuple(stage_results),
            artifacts=tuple(persisted_artifacts),
            feedback=None,
            rejection_reason=rejection_reason,
        )

    async def _invoke_function(
        self,
        function,
        *,
        context: EvaluationContext,
        previous_result: StageResult | None,
        stage_results: tuple[StageResult, ...],
    ) -> Any:
        signature = inspect.signature(function)
        positional_params = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            result = function(
                context=context,
                previous_result=previous_result,
                stage_results=stage_results,
            )
        elif len(positional_params) >= 3:
            result = function(context, previous_result, stage_results)
        elif len(positional_params) == 2:
            result = function(context, previous_result)
        elif len(positional_params) == 1:
            result = function(context)
        else:
            result = function()
        if inspect.isawaitable(result):
            return await result
        return result

    def _normalize_evaluation_result(
        self,
        raw_result: Any,
        *,
        primary_metric: str,
    ) -> EvaluationResult:
        if isinstance(raw_result, EvaluationResult):
            return raw_result
        if isinstance(raw_result, StageResult):
            stage = raw_result
            score = (
                stage.primary_score
                if stage.primary_score is not None
                else stage.metrics.get(primary_metric, float("-inf"))
            )
            return EvaluationResult(
                status=stage.status,
                primary_score=score,
                metrics=stage.metrics,
                features=stage.features,
                execution=stage.execution,
                stage_results=(stage,),
                feedback=stage.feedback,
                rejection_reason=stage.rejection_reason,
            )
        if isinstance(raw_result, dict):
            stage = self._normalize_stage_result(
                raw_result,
                stage_name="evaluate",
                primary_metric=primary_metric,
            )
            score = (
                stage.primary_score
                if stage.primary_score is not None
                else stage.metrics.get(primary_metric, float("-inf"))
            )
            return EvaluationResult(
                status=stage.status,
                primary_score=score,
                metrics=stage.metrics,
                features=stage.features,
                execution=stage.execution,
                stage_results=(stage,),
                feedback=stage.feedback,
                rejection_reason=stage.rejection_reason,
            )
        raise CapabilityUnavailableError(
            f"Unsupported evaluator result type: {type(raw_result).__name__}"
        )

    def _normalize_stage_result(
        self,
        raw_result: Any,
        *,
        stage_name: str,
        primary_metric: str,
    ) -> StageResult:
        if isinstance(raw_result, StageResult):
            return raw_result
        if isinstance(raw_result, EvaluationResult):
            return StageResult(
                name=stage_name,
                status=raw_result.status,
                metrics=raw_result.metrics,
                features=raw_result.features,
                primary_score=raw_result.primary_score,
                should_continue=raw_result.status == "success",
                feedback=raw_result.feedback,
                rejection_reason=raw_result.rejection_reason,
                artifacts=tuple(),
                execution=raw_result.execution,
            )
        if not isinstance(raw_result, dict):
            raise CapabilityUnavailableError(
                f"Unsupported stage result type: {type(raw_result).__name__}"
            )
        metrics = _coerce_numeric_mapping(raw_result.get("metrics", {}))
        features = _coerce_numeric_mapping(raw_result.get("features", {}))
        execution = ExecutionResult.from_dict(raw_result.get("execution"))
        artifacts = tuple(
            _normalize_artifact(raw_artifact) for raw_artifact in raw_result.get("artifacts", [])
        )
        primary_score = (
            float(raw_result["primary_score"])
            if raw_result.get("primary_score") is not None
            else metrics.get(primary_metric)
        )
        return StageResult(
            name=str(raw_result.get("name", stage_name)),
            status=str(raw_result.get("status", "success")),
            metrics=metrics,
            features=features,
            primary_score=primary_score,
            should_continue=bool(raw_result.get("should_continue", True)),
            feedback=str(raw_result["feedback"]) if raw_result.get("feedback") is not None else None,
            rejection_reason=(
                str(raw_result["rejection_reason"])
                if raw_result.get("rejection_reason") is not None
                else None
            ),
            artifacts=artifacts,
            execution=execution,
        )

    def _persist_artifacts(
        self,
        artifacts: tuple[Artifact, ...],
        artifact_dir: Path,
        *,
        prefix: str,
    ) -> tuple[ArtifactRecord, ...]:
        if not self._config.artifacts.enabled or not artifacts:
            return ()
        records: list[ArtifactRecord] = []
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for index, artifact in enumerate(artifacts, start=1):
            slug = _slugify(f"{prefix}_{index}_{artifact.name}")
            if artifact.type == "json":
                path = artifact_dir / f"{slug}.json"
                content = artifact.content if isinstance(artifact.content, dict) else {"value": artifact.content}
                path.write_text(json.dumps(content, indent=2), encoding="utf-8")
                summary = artifact.summary or _truncate(json.dumps(content))
            elif artifact.type == "file":
                if artifact.source_path is None:
                    continue
                suffix = artifact.source_path.suffix or ".bin"
                path = artifact_dir / f"{slug}{suffix}"
                shutil.copy2(artifact.source_path, path)
                summary = artifact.summary
            else:
                path = artifact_dir / f"{slug}.txt"
                text = "" if artifact.content is None else str(artifact.content)
                path.write_text(text, encoding="utf-8")
                summary = artifact.summary or _truncate(text, limit=self._config.artifacts.max_inline_bytes)
            records.append(
                ArtifactRecord(
                    name=artifact.name,
                    type=artifact.type,
                    path=path.resolve(),
                    summary=summary,
                )
            )
        return tuple(records)

    async def _maybe_generate_feedback(
        self,
        program: Program,
        result: EvaluationResult,
    ) -> str | None:
        feedback_config = self._config.feedback
        if not feedback_config.enabled or self._feedback_client is None:
            return None
        should_request = False
        if result.status != "success":
            should_request = feedback_config.on_failure
        elif feedback_config.on_success:
            should_request = True
        elif (
            feedback_config.borderline_score_threshold is not None
            and result.primary_score <= feedback_config.borderline_score_threshold
        ):
            should_request = True
        if not should_request:
            return None
        stderr_excerpt = ""
        if result.execution.stderr:
            stderr_excerpt = result.execution.stderr[-500:]
        prompt = (
            "You are reviewing an evaluated candidate program.\n"
            "Write a short paragraph explaining the likely problem and one concrete next-step idea.\n\n"
            f"Program ID: {program.id}\n"
            f"Status: {result.status}\n"
            f"Primary score: {result.primary_score}\n"
            f"Metrics: {json.dumps(result.metrics, sort_keys=True)}\n"
            f"Rejection reason: {result.rejection_reason or 'none'}\n"
            f"Stderr excerpt: {stderr_excerpt or '<empty>'}\n"
        )
        feedback = await self._feedback_client.generate_text(prompt, attempt=1)
        return _truncate(feedback.strip(), limit=feedback_config.max_feedback_chars)


def _normalize_artifact(raw_artifact: Any) -> Artifact:
    if isinstance(raw_artifact, Artifact):
        return raw_artifact
    if isinstance(raw_artifact, dict):
        return Artifact.from_dict(raw_artifact)
    raise CapabilityUnavailableError(
        f"Unsupported artifact type: {type(raw_artifact).__name__}"
    )


def _coerce_numeric_mapping(raw_mapping: Any) -> dict[str, float]:
    mapping = dict(raw_mapping or {})
    coerced: dict[str, float] = {}
    for key, value in mapping.items():
        if isinstance(value, bool):
            coerced[str(key)] = float(value)
        else:
            coerced[str(key)] = float(value)
    return coerced


def _metrics_from_stdout(stdout: str, primary_metric: str) -> dict[str, float]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {primary_metric: 0.0}
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        return {primary_metric: 0.0}
    return _coerce_numeric_mapping(payload)


def _slugify(value: str) -> str:
    output = []
    for char in value.lower():
        if char.isalnum():
            output.append(char)
        else:
            output.append("_")
    return "".join(output).strip("_") or "artifact"


def _truncate(value: str, *, limit: int = 600) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def docker_environment_status(runtime_name: str = "runsc") -> tuple[bool, str]:
    """Return whether Docker and the required gVisor runtime appear available."""
    if runtime_name != "runsc":
        return False, f"Sandbox runtime must be 'runsc', got {runtime_name!r}"
    try:
        import docker  # type: ignore
    except ImportError:
        return False, "docker Python package is not installed"
    client = None
    try:
        client = docker.from_env()
        client.ping()
        info = client.info()
    except Exception as exc:
        return False, str(exc)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    runtimes = info.get("Runtimes", {})
    if isinstance(runtimes, dict):
        available_runtimes = set(runtimes)
    elif isinstance(runtimes, list):
        available_runtimes = {str(item) for item in runtimes}
    else:
        available_runtimes = set()
    if "runsc" not in available_runtimes:
        return False, "Docker daemon does not report the 'runsc' runtime"
    return True, "ok"


def build_evaluator(
    config: EvaluatorConfig,
    sandbox_config: SandboxConfig,
    *,
    feedback_client: AsyncInferenceClient | None = None,
) -> Evaluator:
    """Factory for evaluator adapters."""
    backend = sandbox_config.backend.lower()
    if backend == "docker":
        sandbox = DockerGVisorSandbox(sandbox_config)
    elif backend == "python":
        sandbox = PythonSubprocessSandbox(sandbox_config)
    else:
        raise CapabilityUnavailableError(f"Unsupported evaluator backend: {sandbox_config.backend}")
    return ModuleEvaluator(config, sandbox, feedback_client=feedback_client)
