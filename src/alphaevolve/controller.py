"""Asynchronous orchestration loop."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter

from alphaevolve.archive import ProgramDatabase
from alphaevolve.diffing import apply_diff, parse_diff
from alphaevolve.errors import DiffApplyError, DiffParseError, PromptTooLargeError
from alphaevolve.evaluators import Evaluator
from alphaevolve.llm import AsyncInferenceClient
from alphaevolve.logging_utils import PipelineStats
from alphaevolve.models import ExperimentConfig, Program, make_program_id
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


class EvolutionController:
    """Run the AlphaEvolve sample/generate/evaluate loop asynchronously."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        archive: ProgramDatabase,
        inference_client: AsyncInferenceClient,
        evaluator: Evaluator,
        prompt_builder: PromptBuilder,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._archive = archive
        self._inference_client = inference_client
        self._evaluator = evaluator
        self._prompt_builder = prompt_builder
        self._logger = logger or logging.getLogger("alphaevolve.controller")
        self._stats = PipelineStats()
        self._llm_semaphore = asyncio.Semaphore(config.controller.llm_concurrency)
        self._evaluation_semaphore = asyncio.Semaphore(config.controller.evaluation_concurrency)
        self._archive_lock = asyncio.Lock()
        self._best_program: Program | None = None
        self._stop_reason = "generation_limit"

    async def run(self, seed_program: Program) -> ControllerResult:
        """Evaluate the seed and drive the async mutation/evaluation pipeline."""
        await self._evaluate_and_record(seed_program)
        if self._target_reached():
            self._stop_reason = "target_score"
            return self._build_result(self._best_program or seed_program)

        stop_event = asyncio.Event()
        prompt_queue: asyncio.Queue[Program | None] = asyncio.Queue(
            maxsize=self._config.controller.max_pending_prompts
        )
        evaluation_queue: asyncio.Queue[Program | None] = asyncio.Queue(
            maxsize=self._config.controller.max_pending_evaluations
        )

        stats_task = asyncio.create_task(self._stats_loop(stop_event, prompt_queue, evaluation_queue))
        evaluation_tasks = [
            asyncio.create_task(self._evaluation_worker(evaluation_queue, stop_event))
            for _ in range(self._config.controller.evaluation_concurrency)
        ]
        llm_tasks = [
            asyncio.create_task(self._llm_worker(prompt_queue, evaluation_queue, stop_event))
            for _ in range(self._config.controller.llm_concurrency)
        ]
        sampler_task = asyncio.create_task(self._sampler(prompt_queue, stop_event))

        try:
            await sampler_task
            await asyncio.gather(*llm_tasks)
            for _ in evaluation_tasks:
                await evaluation_queue.put(None)
            await asyncio.gather(*evaluation_tasks)
        finally:
            stop_event.set()
            stats_task.cancel()
            await asyncio.gather(stats_task, return_exceptions=True)
            await self._inference_client.aclose()
            await self._evaluator.aclose()

        best_program = self._best_program or seed_program
        if self._target_reached():
            self._stop_reason = "target_score"
        return self._build_result(best_program)

    async def _sampler(
        self,
        prompt_queue: asyncio.Queue[Program | None],
        stop_event: asyncio.Event,
    ) -> None:
        try:
            while not stop_event.is_set():
                async with self._archive_lock:
                    if self._stats.mutation_jobs_started >= self._config.controller.max_generations:
                        break
                    parent = self._archive.sample()
                    self._stats.mutation_jobs_started += 1
                await prompt_queue.put(parent)
        finally:
            for _ in range(self._config.controller.llm_concurrency):
                await prompt_queue.put(None)

    async def _llm_worker(
        self,
        prompt_queue: asyncio.Queue[Program | None],
        evaluation_queue: asyncio.Queue[Program | None],
        stop_event: asyncio.Event,
    ) -> None:
        while True:
            parent = await prompt_queue.get()
            try:
                if parent is None:
                    return
                if stop_event.is_set():
                    continue
                child = await self._propose_child(parent)
                if child is not None:
                    await evaluation_queue.put(child)
            finally:
                prompt_queue.task_done()

    async def _propose_child(self, parent: Program) -> Program | None:
        async with self._archive_lock:
            history = self._archive.promising_programs(
                self._config.model.prompt_budget.max_history_programs,
                exclude_id=parent.id,
            )

        try:
            rendered_prompt = self._prompt_builder.build(
                system_instructions=self._config.system_instructions,
                task_contract=f"{self._config.task_description}\n\n{self._config.evaluation_contract}",
                current_program=parent,
                history=history,
            )
        except PromptTooLargeError as exc:
            self._logger.error("Prompt budget exceeded for program %s: %s", parent.id, exc)
            return None

        for attempt in range(1, self._config.controller.max_retries + 1):
            raw_preview: str | None = None
            try:
                started = perf_counter()
                async with self._llm_semaphore:
                    raw_diff = await self._inference_client.generate_diff(rendered_prompt.text, attempt=attempt)
                self._stats.record_llm_latency((perf_counter() - started) * 1_000)
                raw_preview = _sanitize_response_preview(raw_diff)
                parsed = parse_diff(raw_diff)
                child_code = apply_diff(parent.code, parsed)
                _enforce_edit_window(parent.code, child_code)
                return Program(
                    id=make_program_id(),
                    code=child_code,
                    parent_id=parent.id,
                )
            except DiffParseError as exc:
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
        return None

    async def _evaluation_worker(
        self,
        evaluation_queue: asyncio.Queue[Program | None],
        stop_event: asyncio.Event,
    ) -> None:
        while True:
            candidate = await evaluation_queue.get()
            try:
                if candidate is None:
                    return
                await self._evaluate_and_record(candidate)
                if self._target_reached():
                    stop_event.set()
            finally:
                evaluation_queue.task_done()

    async def _evaluate_and_record(self, program: Program) -> Program:
        async with self._evaluation_semaphore:
            started = perf_counter()
            metrics, execution = await self._evaluator.evaluate(
                program,
                primary_metric=self._config.primary_metric,
            )
            self._stats.record_evaluation_latency((perf_counter() - started) * 1_000)
        program.metrics = metrics
        program.execution = execution
        program.primary_score = metrics.get(self._config.primary_metric, float("-inf"))
        async with self._archive_lock:
            self._archive.record(program)
            self._best_program = self._archive.best_program()
            self._stats.evaluation_count += 1
            self._stats.best_score = (
                self._best_program.primary_score if self._best_program else self._stats.best_score
            )
            if execution.status not in {"success", "pending"}:
                self._stats.sandbox_failures += 1
        return program

    async def _stats_loop(
        self,
        stop_event: asyncio.Event,
        prompt_queue: asyncio.Queue[Program | None],
        evaluation_queue: asyncio.Queue[Program | None],
    ) -> None:
        try:
            while not stop_event.is_set():
                await asyncio.sleep(self._config.controller.metrics_interval_seconds)
                snapshot = self._stats.snapshot(
                    pending_prompts=prompt_queue.qsize(),
                    pending_evaluations=evaluation_queue.qsize(),
                )
                self._logger.info("pipeline_stats=%s archive=%s", snapshot, self._archive.stats())
        except asyncio.CancelledError:
            return

    def _target_reached(self) -> bool:
        return bool(self._best_program and self._best_program.primary_score >= self._config.target_score)

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
        )


def _sanitize_response_preview(raw_text: str, limit: int = 240) -> str:
    preview = raw_text.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit]}..."


def _enforce_edit_window(source: str, updated: str) -> None:
    edit_window = locate_edit_window(source)
    if edit_window is None:
        return
    prefix = source[: edit_window.start]
    suffix = source[edit_window.end :]
    if not updated.startswith(prefix):
        raise DiffApplyError("Diff modified code before the EVOLVE-BLOCK start marker.")
    if suffix and not updated.endswith(suffix):
        raise DiffApplyError("Diff modified code after the EVOLVE-BLOCK end marker.")
