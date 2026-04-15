"""Evaluators for local fake execution and Docker/gVisor sandboxing."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import traceback
from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter

from alphaevolve.errors import CapabilityUnavailableError
from alphaevolve.models import ExecutionResult, Program, SandboxConfig


class Evaluator(ABC):
    """Abstract evaluator interface."""

    @abstractmethod
    async def evaluate(self, program: Program, *, primary_metric: str) -> tuple[dict[str, float], ExecutionResult]:
        """Evaluate a program and return metrics plus execution metadata."""

    async def aclose(self) -> None:
        """Optional cleanup hook."""


class FakeEvaluator(Evaluator):
    """Execute trusted benchmark code locally for tests and offline development."""

    async def evaluate(self, program: Program, *, primary_metric: str) -> tuple[dict[str, float], ExecutionResult]:
        return await asyncio.to_thread(self._run, program.code, primary_metric)

    def _run(self, code: str, primary_metric: str) -> tuple[dict[str, float], ExecutionResult]:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        started = perf_counter()
        namespace = {"__name__": "__alphaevolve__", "__builtins__": __builtins__}

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(compile(code, "<alphaevolve>", "exec"), namespace, namespace)
                evaluate = namespace["evaluate"]
                raw_metrics = evaluate()
            metrics = {name: float(value) for name, value in dict(raw_metrics).items()}
            execution = ExecutionResult(
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                exit_code=0,
                duration_ms=(perf_counter() - started) * 1_000,
                status="success",
            )
            return metrics, execution
        except Exception as exc:
            traceback.print_exc(file=stderr_buffer)
            execution = ExecutionResult(
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                exit_code=1,
                duration_ms=(perf_counter() - started) * 1_000,
                status="error",
                exception=str(exc),
            )
            return {primary_metric: 0.0}, execution


class DockerGVisorSandbox(Evaluator):
    """Evaluate programs inside a locked-down Docker container."""

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._docker_client = None

    async def evaluate(self, program: Program, *, primary_metric: str) -> tuple[dict[str, float], ExecutionResult]:
        return await asyncio.to_thread(self._evaluate_blocking, program.code, primary_metric)

    def _evaluate_blocking(self, code: str, primary_metric: str) -> tuple[dict[str, float], ExecutionResult]:
        docker = self._docker_module()
        client = self._client(docker)
        started = perf_counter()
        container = None
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            program_path = workspace / "program.py"
            program_path.write_text(code, encoding="utf-8")
            try:
                container = client.containers.run(
                    self._config.image,
                    command=["python", f"{self._config.working_dir}/program.py"],
                    detach=True,
                    network_mode=self._config.network_mode,
                    read_only=self._config.read_only_root,
                    runtime=self._config.runtime,
                    mem_limit=self._config.mem_limit,
                    cpu_quota=self._config.cpu_quota,
                    working_dir=self._config.working_dir,
                    tmpfs={self._config.tmpfs_path: f"rw,size={self._config.tmpfs_size_bytes}"},
                    volumes={
                        str(workspace): {
                            "bind": self._config.working_dir,
                            "mode": "ro",
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
                execution = ExecutionResult(
                    stdout="",
                    stderr=str(exc),
                    exit_code=1,
                    duration_ms=(perf_counter() - started) * 1_000,
                    status="error",
                    exception=str(exc),
                )
                return {primary_metric: 0.0}, execution
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


def _metrics_from_stdout(stdout: str, primary_metric: str) -> dict[str, float]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {primary_metric: 0.0}
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        return {primary_metric: 0.0}
    return {name: float(value) for name, value in payload.items()}


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


def build_evaluator(config: SandboxConfig) -> Evaluator:
    """Factory for evaluator adapters."""
    backend = config.backend.lower()
    if backend == "fake":
        return FakeEvaluator()
    if backend == "docker":
        return DockerGVisorSandbox(config)
    raise CapabilityUnavailableError(f"Unsupported evaluator backend: {config.backend}")
