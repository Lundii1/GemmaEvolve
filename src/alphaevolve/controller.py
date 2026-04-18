"""Bounded async orchestration loop with islands, checkpoints, and prompt logs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from alphaevolve.archive import ProgramDatabase
from alphaevolve.diffing import apply_diff, parse_diff
from alphaevolve.errors import DiffApplyError, DiffParseError, PromptTooLargeError
from alphaevolve.evaluators import Evaluator
from alphaevolve.llm import AsyncInferenceClient
from alphaevolve.logging_utils import PipelineStats
from alphaevolve.models import (
    CheckpointState,
    EvaluationResult,
    ExecutionResult,
    ExperimentConfig,
    Program,
    PromptArtifactContext,
    PromptLog,
    decode_random_state,
    encode_random_state,
    make_program_id,
    make_prompt_log_id,
)
from alphaevolve.prompts import PromptBuilder, locate_edit_window


@dataclass(frozen=True, slots=True)
class ControllerResult:
    """Summary of a completed controller run."""

    best_program: Program
    stop_reason: str
    total_evaluated: int
    total_generation_jobs: int
    diff_parse_failures: int
    diff_apply_failures: int
    retry_count: int
    sandbox_failures: int
    latest_llm_latency_ms: float | None
    latest_evaluation_latency_ms: float | None
    sample_parse_failure_preview: str | None
    sample_apply_failure_preview: str | None
    run_dir: Path


class EvolutionController:
    """Run the AlphaEvolve sample/generate/evaluate loop with bounded parallelism."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        database: ProgramDatabase,
        inference_client: AsyncInferenceClient,
        evaluator: Evaluator,
        prompt_builder: PromptBuilder,
        run_dir: str | Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._database = database
        self._inference_client = inference_client
        self._evaluator = evaluator
        self._prompt_builder = prompt_builder
        self._run_dir = Path(run_dir).resolve()
        self._checkpoint_path = self._run_dir / "checkpoint.json"
        self._logger = logger or logging.getLogger("alphaevolve.controller")
        self._stats = PipelineStats()
        self._best_program: Program | None = None
        self._stop_reason = "generation_limit"
        self._generation = 0
        self._next_island_index = 0
        self._stagnation_generations = 0
        self._last_metrics_log_at = perf_counter()
        self._stop_requested = False
        self._signal_stop_reason = "signal"
        self._random = random.Random()
        self._state_lock = asyncio.Lock()
        self._checkpoint_lock = asyncio.Lock()
        self._active_prompts = 0
        self._active_evaluations = 0

    async def run(self, seed_program: Program) -> ControllerResult:
        """Evaluate the seed and drive the mutation/evaluation pipeline."""
        self._run_dir.mkdir(parents=True, exist_ok=True)
        await self._restore_checkpoint_if_present()
        in_flight: set[asyncio.Task[None]] = set()
        with _signal_guard(self._request_stop):
            try:
                if await self._database.best_program() is None:
                    await self._evaluate_seed(seed_program)
                    async with self._state_lock:
                        self._best_program = await self._database.best_program()
                else:
                    async with self._state_lock:
                        self._best_program = await self._database.best_program()

                semaphore = asyncio.Semaphore(self._config.controller.max_inflight)

                async def _dispatch_one(island_id: int, generation_number: int) -> None:
                    async with semaphore:
                        await self._run_generation(
                            island_id=island_id,
                            generation_number=generation_number,
                        )

                if not await self._target_reached():
                    while self._generation < self._config.controller.max_generations:
                        stop_reason = await self._current_stop_reason()
                        if stop_reason is not None:
                            self._stop_reason = stop_reason
                            break
                        while (
                            len(in_flight) < self._config.controller.max_inflight
                            and self._generation < self._config.controller.max_generations
                        ):
                            stop_reason = await self._current_stop_reason()
                            if stop_reason is not None:
                                self._stop_reason = stop_reason
                                break
                            island_id = self._next_island_index
                            self._next_island_index = (self._next_island_index + 1) % self._config.database.islands
                            generation_number = self._generation + 1
                            self._generation = generation_number
                            in_flight.add(asyncio.create_task(_dispatch_one(island_id, generation_number)))
                        if not in_flight:
                            break
                        done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                        in_flight = set(pending)
                        await self._consume_completed_tasks(done)
                        await self._maybe_log_metrics()
                        await self._maybe_checkpoint(force=False)
                    else:
                        self._stop_reason = "generation_limit"
            finally:
                if in_flight:
                    results = await asyncio.gather(*in_flight, return_exceptions=True)
                    await self._consume_completed_results(results)
                await self._finalize()

        stop_reason = await self._current_stop_reason()
        if stop_reason is not None:
            self._stop_reason = stop_reason
        elif self._generation >= self._config.controller.max_generations:
            self._stop_reason = "generation_limit"

        best_program = self._best_program or await self._database.best_program() or seed_program
        return self._build_result(best_program)

    async def _evaluate_seed(self, seed_program: Program) -> None:
        async with self._state_lock:
            self._active_evaluations += 1
        try:
            result = await self._evaluator.evaluate(
                seed_program,
                primary_metric=self._config.primary_metric,
                artifact_dir=self._database.root_dir / "artifacts" / seed_program.id,
            )
        finally:
            async with self._state_lock:
                self._active_evaluations -= 1
        async with self._state_lock:
            self._stats.record_evaluation_latency(result.execution.duration_ms)
            self._stats.evaluation_count += 1
            if result.status not in {"success", "pending"}:
                self._stats.sandbox_failures += 1
        seed_program.metrics = result.metrics
        seed_program.features = result.features
        seed_program.execution = result.execution
        seed_program.evaluation = result
        seed_program.primary_score = result.primary_score
        seed_program.artifact_records = result.artifacts
        seed_program.accepted = result.status == "success" and result.rejection_reason is None
        seed_program.rejection_reason = result.rejection_reason
        await self._database.record_seed_across_islands(seed_program)
        async with self._state_lock:
            self._stats.best_score = seed_program.primary_score
        await self._maybe_checkpoint(force=True)

    async def _run_generation(self, *, island_id: int, generation_number: int) -> None:
        parent = await self._database.sample(island_id)
        history = await self._database.promising_programs(
            self._config.model.prompt_budget.max_history_programs,
            island_id=island_id,
            exclude_id=parent.id,
        )
        parent_view = Program.from_dict(parent.to_dict())
        parent_view.island_id = island_id
        artifact_context = self._artifact_context(parent, history)
        evaluator_feedback = parent.evaluation.feedback if parent.evaluation else None

        try:
            rendered_prompt = self._prompt_builder.build(
                system_instructions=self._config.system_instructions,
                task_contract=f"{self._config.task_description}\n\n{self._config.evaluation_contract}",
                current_program=parent_view,
                history=history,
                artifact_context=artifact_context,
                evaluator_feedback=evaluator_feedback,
            )
        except PromptTooLargeError as exc:
            self._logger.error("Prompt budget exceeded for program %s: %s", parent.id, exc)
            async with self._state_lock:
                self._stagnation_generations += 1
            return

        prompt_log = await self._database.record_prompt_log(
            PromptLog(
                id=make_prompt_log_id(),
                parent_id=parent.id,
                island_id=island_id,
                prompt_text=rendered_prompt.text,
                estimated_tokens=rendered_prompt.estimated_tokens,
                included_program_ids=rendered_prompt.included_program_ids,
                summarized_program_ids=rendered_prompt.summarized_program_ids,
                artifact_context=rendered_prompt.artifact_context,
                evaluator_feedback_used=rendered_prompt.evaluator_feedback_used,
            )
        )
        async with self._state_lock:
            self._stats.mutation_jobs_started += 1
        child = await self._propose_child(
            parent=parent,
            island_id=island_id,
            generation_number=generation_number,
            prompt_text=rendered_prompt.text,
            prompt_log_id=prompt_log.id,
        )
        if child is None:
            async with self._state_lock:
                self._stagnation_generations += 1
            return

        novelty_rejection = await self._database.preflight_candidate(child, island_id=island_id)
        if novelty_rejection is not None:
            child.accepted = False
            child.rejection_reason = novelty_rejection
            child.evaluation = EvaluationResult(
                status="rejected",
                primary_score=float("-inf"),
                metrics={self._config.primary_metric: 0.0},
                execution=ExecutionResult(status="rejected"),
                feedback=None,
                rejection_reason=novelty_rejection,
            )
            child.execution = child.evaluation.execution
            child.metrics = child.evaluation.metrics
            child.primary_score = child.evaluation.primary_score
            await self._database.record(child)
            async with self._state_lock:
                self._stagnation_generations += 1
            return

        evaluated = await self._evaluate_candidate(child)
        improved = False
        made_progress = False
        async with self._state_lock:
            if evaluated.accepted and (
                self._best_program is None or evaluated.primary_score > self._best_program.primary_score
            ):
                self._best_program = evaluated
                self._stats.best_score = evaluated.primary_score
                self._stagnation_generations = 0
                improved = True
                made_progress = True
            elif evaluated.accepted and evaluated.behaviorally_novel:
                self._stagnation_generations = 0
                made_progress = True
            else:
                self._stagnation_generations += 1
        if improved or made_progress:
            await self._maybe_checkpoint(force=True)
        migration = await self._database.maybe_migrate(
            generation=generation_number,
            source_island_id=island_id,
        )
        if migration is not None:
            target_island, program_id = migration
            self._logger.info(
                "migration_complete source_island=%s target_island=%s program_id=%s",
                island_id,
                target_island,
                program_id,
            )

    async def _propose_child(
        self,
        *,
        parent: Program,
        island_id: int,
        generation_number: int,
        prompt_text: str,
        prompt_log_id: str,
    ) -> Program | None:
        raw_preview: str | None = None
        for attempt in range(1, self._config.controller.max_retries + 1):
            try:
                started = perf_counter()
                async with self._state_lock:
                    self._active_prompts += 1
                raw_diff = await self._inference_client.generate_diff(prompt_text, attempt=attempt)
                async with self._state_lock:
                    self._stats.record_llm_latency((perf_counter() - started) * 1_000)
                raw_preview = _sanitize_response_preview(raw_diff)
                parsed = parse_diff(raw_diff)
                child_code = apply_diff(parent.code, parsed)
                _enforce_edit_window(
                    parent.code,
                    child_code,
                    mutation_scope=self._config.mutation_scope,
                )
                child = Program(
                    id=make_program_id(),
                    code=child_code,
                    parent_id=parent.id,
                    generation=generation_number,
                    island_id=island_id,
                    prompt_log_id=prompt_log_id,
                    lineage_depth=parent.lineage_depth + 1,
                )
                await self._database.update_prompt_log(
                    prompt_log_id,
                    model_response=raw_diff,
                    child_id=child.id,
                )
                await self._database.record_parent_mutation_outcome(parent.id, success=True)
                return child
            except DiffParseError as exc:
                async with self._state_lock:
                    self._stats.record_diff_parse_failure(raw_preview)
                    if attempt < self._config.controller.max_retries:
                        self._stats.retry_count += 1
                self._logger.warning(
                    "Diff parse attempt %s/%s failed for parent %s: %s preview=%s",
                    attempt,
                    self._config.controller.max_retries,
                    parent.id,
                    exc,
                    raw_preview or "<empty>",
                )
            except DiffApplyError as exc:
                async with self._state_lock:
                    self._stats.record_diff_apply_failure(raw_preview)
                    if attempt < self._config.controller.max_retries:
                        self._stats.retry_count += 1
                self._logger.warning(
                    "Diff apply attempt %s/%s failed for parent %s: %s preview=%s",
                    attempt,
                    self._config.controller.max_retries,
                    parent.id,
                    exc,
                    raw_preview or "<empty>",
                )
            finally:
                async with self._state_lock:
                    if self._active_prompts > 0:
                        self._active_prompts -= 1
        if raw_preview is not None:
            await self._database.update_prompt_log(
                prompt_log_id,
                model_response=raw_preview,
                child_id=None,
            )
        await self._database.record_parent_mutation_outcome(parent.id, success=False)
        return None

    async def _evaluate_candidate(self, program: Program) -> Program:
        started = perf_counter()
        async with self._state_lock:
            self._active_evaluations += 1
        try:
            result = await self._evaluator.evaluate(
                program,
                primary_metric=self._config.primary_metric,
                artifact_dir=self._database.root_dir / "artifacts" / program.id,
            )
        finally:
            async with self._state_lock:
                self._active_evaluations -= 1
        async with self._state_lock:
            self._stats.record_evaluation_latency((perf_counter() - started) * 1_000)
            self._stats.evaluation_count += 1
            if result.status not in {"success", "pending"}:
                self._stats.sandbox_failures += 1
        program.metrics = result.metrics
        program.features = result.features
        program.execution = result.execution
        program.evaluation = result
        program.primary_score = result.primary_score
        program.artifact_records = result.artifacts
        program.accepted = result.status == "success" and result.rejection_reason is None
        program.rejection_reason = result.rejection_reason
        await self._database.record(program)
        async with self._state_lock:
            if self._best_program is None:
                self._best_program = await self._database.best_program()
        return program

    async def _restore_checkpoint_if_present(self) -> None:
        resume_from = self._config.checkpoint.resume_from
        if resume_from is not None:
            checkpoint_path = resume_from / "checkpoint.json"
        else:
            checkpoint_path = self._checkpoint_path
        if not checkpoint_path.exists():
            self._best_program = await self._database.best_program()
            return
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        state = CheckpointState.from_dict(payload)
        self._generation = state.generation
        self._next_island_index = state.next_island_index
        self._stagnation_generations = state.stagnation_generations
        self._stop_reason = state.stop_reason or self._stop_reason
        self._random.setstate(decode_random_state(state.random_state))
        if state.best_program_id is not None:
            self._best_program = self._database.get_program(state.best_program_id)
        else:
            self._best_program = await self._database.best_program()

    async def _maybe_checkpoint(self, *, force: bool) -> None:
        if not self._config.checkpoint.enabled:
            return
        interval = max(self._config.checkpoint.interval_generations, 1)
        if not force and self._generation > 0 and self._generation % interval != 0:
            return
        async with self._checkpoint_lock:
            async with self._state_lock:
                state = CheckpointState(
                    generation=self._generation,
                    best_program_id=self._best_program.id if self._best_program is not None else None,
                    next_island_index=self._next_island_index,
                    stagnation_generations=self._stagnation_generations,
                    stop_reason=self._stop_reason,
                    random_state=encode_random_state(self._random.getstate()),
                )
            tmp_path = self._checkpoint_path.with_name(f"{self._checkpoint_path.name}.tmp")
            tmp_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
            os.replace(tmp_path, self._checkpoint_path)

    async def _finalize(self) -> None:
        await self._maybe_checkpoint(force=True)
        await self._inference_client.aclose()
        await self._evaluator.aclose()

    async def _target_reached(self) -> bool:
        async with self._state_lock:
            return self._target_reached_unlocked()

    def _target_reached_unlocked(self) -> bool:
        return bool(self._best_program and self._best_program.primary_score >= self._config.target_score)

    async def _current_stop_reason(self) -> str | None:
        async with self._state_lock:
            if self._target_reached_unlocked():
                return "target_score"
            if self._stagnation_generations >= self._config.controller.stagnation_patience:
                return "stagnation"
        if self._stop_requested:
            return self._signal_stop_reason
        return None

    def _build_result(self, best_program: Program) -> ControllerResult:
        return ControllerResult(
            best_program=best_program,
            stop_reason=self._stop_reason,
            total_evaluated=self._stats.evaluation_count,
            total_generation_jobs=self._stats.mutation_jobs_started,
            diff_parse_failures=self._stats.diff_parse_failures,
            diff_apply_failures=self._stats.diff_apply_failures,
            retry_count=self._stats.retry_count,
            sandbox_failures=self._stats.sandbox_failures,
            latest_llm_latency_ms=self._stats.latest_llm_latency_ms,
            latest_evaluation_latency_ms=self._stats.latest_evaluation_latency_ms,
            sample_parse_failure_preview=self._stats.sample_parse_failure_preview,
            sample_apply_failure_preview=self._stats.sample_apply_failure_preview,
            run_dir=self._run_dir,
        )

    def _artifact_context(
        self,
        parent: Program,
        history: list[Program],
    ) -> tuple[PromptArtifactContext, ...]:
        contexts: list[PromptArtifactContext] = []
        seen: set[tuple[str, str]] = set()
        for program in [parent, *history]:
            for artifact in program.artifact_records:
                if not artifact.summary:
                    continue
                key = (artifact.name, artifact.summary)
                if key in seen:
                    continue
                contexts.append(
                    PromptArtifactContext(
                        name=artifact.name,
                        type=artifact.type,
                        summary=artifact.summary,
                    )
                )
                seen.add(key)
                if len(contexts) >= self._config.model.prompt_budget.max_artifact_context:
                    return tuple(contexts)
        return tuple(contexts)

    async def _maybe_log_metrics(self) -> None:
        if perf_counter() - self._last_metrics_log_at < self._config.controller.metrics_interval_seconds:
            return
        self._last_metrics_log_at = perf_counter()
        async with self._state_lock:
            snapshot = self._stats.snapshot(
                pending_prompts=self._active_prompts,
                pending_evaluations=self._active_evaluations,
            )
        self._logger.info("pipeline_stats=%s database=%s", snapshot, self._database.stats())

    def _request_stop(self) -> None:
        self._stop_requested = True

    async def _consume_completed_tasks(self, tasks: set[asyncio.Task[None]]) -> None:
        results: list[BaseException] = []
        for task in tasks:
            exc = task.exception()
            if exc is not None:
                results.append(exc)
        if results:
            await self._consume_completed_results(results)

    async def _consume_completed_results(self, results: list[object]) -> None:
        failures = 0
        for result in results:
            if isinstance(result, BaseException):
                failures += 1
                self._logger.error(
                    "generation task failed",
                    exc_info=(type(result), result, result.__traceback__),
                )
        if failures:
            async with self._state_lock:
                self._stagnation_generations += failures


def _sanitize_response_preview(raw_text: str, limit: int = 240) -> str:
    preview = raw_text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit]}..."


def _enforce_edit_window(
    source: str,
    updated: str,
    *,
    mutation_scope: str = "evolve_block",
) -> None:
    if mutation_scope != "evolve_block":
        return
    edit_window = locate_edit_window(source)
    if edit_window is None:
        return
    prefix = source[: edit_window.start]
    suffix = source[edit_window.end :]
    if not updated.startswith(prefix):
        raise DiffApplyError("Diff modified code before the EVOLVE-BLOCK start marker.")
    if suffix and not updated.endswith(suffix):
        raise DiffApplyError("Diff modified code after the EVOLVE-BLOCK end marker.")


class _signal_guard:
    """Temporarily request a graceful stop on SIGINT/SIGTERM."""

    def __init__(self, callback) -> None:
        self._callback = callback
        self._previous: dict[int, Any] = {}

    def __enter__(self):
        for signame in ("SIGINT", "SIGTERM"):
            signum = getattr(signal, signame, None)
            if signum is None:
                continue
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
            except (ValueError, OSError):
                continue
        return self

    def __exit__(self, exc_type, exc, tb):
        for signum, previous in self._previous.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError):
                continue
        return False

    def _handle(self, signum, frame) -> None:
        self._callback()
