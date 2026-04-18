from __future__ import annotations

import asyncio
import random
from pathlib import Path

from alphaevolve.archive import ProgramDatabase
from alphaevolve.controller import EvolutionController
from alphaevolve.errors import CapabilityUnavailableError
from alphaevolve.evaluators import Evaluator, build_evaluator
from alphaevolve.llm import AsyncInferenceClient
from alphaevolve.models import (
    CheckpointConfig,
    EvaluationResult,
    ExecutionResult,
    ControllerConfig,
    DatabaseConfig,
    EvaluatorConfig,
    ExperimentConfig,
    FeatureAxisConfig,
    ModelConfig,
    Program,
    PromptBudget,
    RetentionConfig,
    SandboxConfig,
)
from alphaevolve.prompts import PromptBuilder

DEFAULT_KNAPSACK_DIFF = """<<<<<<< SEARCH
    return value
=======
    return value / max(weight, 1e-9)
>>>>>>> REPLACE"""

PARALLEL_KNAPSACK_DIFFS = [
    """<<<<<<< SEARCH
    return value
=======
    return value / max(weight, 1e-9) + 0.0
>>>>>>> REPLACE""",
    """<<<<<<< SEARCH
    return value
=======
    return (1.0 * value) / max(weight, 1e-9)
>>>>>>> REPLACE""",
    """<<<<<<< SEARCH
    return value
=======
    return value / max(weight, 1e-9) + weight * 0.0
>>>>>>> REPLACE""",
    """<<<<<<< SEARCH
    return value
=======
    return (value + 0.0) / max(weight, 1e-9)
>>>>>>> REPLACE""",
]


class StubInferenceClient(AsyncInferenceClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate_text(self, prompt: str, *, attempt: int = 1) -> str:
        if not self._responses:
            raise AssertionError("No more stub responses available.")
        return self._responses.pop(0)


class FailingInferenceClient(AsyncInferenceClient):
    async def generate_text(self, prompt: str, *, attempt: int = 1) -> str:
        raise CapabilityUnavailableError("rate limited")


class TrackingEvaluator(Evaluator):
    def __init__(self, *, delay_seconds: float = 0.05) -> None:
        self._delay_seconds = delay_seconds
        self._lock = asyncio.Lock()
        self.current_evaluations = 0
        self.max_concurrent_evaluations = 0

    async def evaluate(
        self,
        program: Program,
        *,
        primary_metric: str,
        artifact_dir: Path,
    ) -> EvaluationResult:
        async with self._lock:
            self.current_evaluations += 1
            self.max_concurrent_evaluations = max(
                self.max_concurrent_evaluations,
                self.current_evaluations,
            )
        try:
            await asyncio.sleep(self._delay_seconds)
        finally:
            async with self._lock:
                self.current_evaluations -= 1

        score = 1.0 if program.parent_id is None else 2.0
        execution = ExecutionResult(
            duration_ms=self._delay_seconds * 1_000,
            status="success",
            exit_code=0,
        )
        return EvaluationResult(
            status="success",
            primary_score=score,
            metrics={primary_metric: score},
            execution=execution,
        )


class NoveltyEvaluator(Evaluator):
    def __init__(self) -> None:
        self._variant = 0

    async def evaluate(
        self,
        program: Program,
        *,
        primary_metric: str,
        artifact_dir: Path,
    ) -> EvaluationResult:
        del artifact_dir
        execution = ExecutionResult(duration_ms=5.0, status="success", exit_code=0)
        if program.parent_id is None:
            return EvaluationResult(
                status="success",
                primary_score=1.0,
                metrics={primary_metric: 1.0, "variant_metric": 0.0},
                features={"behavior_axis": 0.0},
                execution=execution,
            )
        self._variant += 1
        return EvaluationResult(
            status="success",
            primary_score=1.0,
            metrics={primary_metric: 1.0, "variant_metric": float(self._variant)},
            features={"behavior_axis": float(self._variant)},
            execution=execution,
        )


def _config(
    tmp_path,
    *,
    max_generations: int,
    max_retries: int = 2,
    stagnation_patience: int = 4,
    resume_from=None,
    max_inflight: int = 1,
    target_score: float = 275.0,
    database_config: DatabaseConfig | None = None,
    mutation_scope: str = "evolve_block",
):
    root = Path("experiments").resolve()
    return ExperimentConfig(
        name="knapsack",
        description="test",
        seed_program_path=(root / "knapsack_seed.py").resolve(),
        system_instructions="Improve the provided code.",
        task_description="Maximize the knapsack score.",
        evaluation_contract="Return SEARCH/REPLACE blocks only.",
        primary_metric="score",
        target_score=target_score,
        mutation_scope=mutation_scope,
        model=ModelConfig(
            provider="ollama",
            prompt_budget=PromptBudget(
                max_prompt_tokens=12_000,
                reserved_completion_tokens=2_048,
                max_history_programs=4,
            ),
        ),
        sandbox=SandboxConfig(backend="python", timeout_seconds=5.0),
        database=database_config or DatabaseConfig(islands=2),
        evaluator=EvaluatorConfig(module=(root / "knapsack_evaluator.py").resolve()),
        checkpoint=CheckpointConfig(enabled=True, interval_generations=1, resume_from=resume_from),
        controller=ControllerConfig(
            max_generations=max_generations,
            max_retries=max_retries,
            metrics_interval_seconds=30.0,
            stagnation_patience=stagnation_patience,
            max_inflight=max_inflight,
        ),
    )


def test_controller_reaches_knapsack_optimum_with_python_backend(tmp_path) -> None:
    run_dir = tmp_path / "run"
    config = _config(tmp_path, max_generations=3)
    database = ProgramDatabase(config.database, run_dir / "database", random_source=random.Random(0))
    inference_client = StubInferenceClient([DEFAULT_KNAPSACK_DIFF])
    evaluator = build_evaluator(config.evaluator, config.sandbox, feedback_client=inference_client)
    controller = EvolutionController(
        config=config,
        database=database,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=PromptBuilder(config.model.prompt_budget),
        run_dir=run_dir,
    )

    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    result = asyncio.run(controller.run(Program.seed(seed_code)))

    assert result.best_program.primary_score == 275.0
    assert result.stop_reason == "target_score"
    assert result.total_evaluated >= 2
    assert (run_dir / "checkpoint.json").exists()
    assert database.list_prompt_logs()


def test_controller_tracks_parse_failures_and_retries(tmp_path) -> None:
    run_dir = tmp_path / "run"
    malformed = "I will explain the change before the diff."
    config = _config(tmp_path, max_generations=1, max_retries=2, stagnation_patience=2)
    database = ProgramDatabase(config.database, run_dir / "database", random_source=random.Random(0))
    inference_client = StubInferenceClient([malformed, DEFAULT_KNAPSACK_DIFF])
    evaluator = build_evaluator(config.evaluator, config.sandbox, feedback_client=inference_client)
    controller = EvolutionController(
        config=config,
        database=database,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=PromptBuilder(config.model.prompt_budget),
        run_dir=run_dir,
    )

    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    result = asyncio.run(controller.run(Program.seed(seed_code)))

    assert result.total_generation_jobs == 1
    assert result.diff_parse_failures == 1
    assert result.diff_apply_failures == 0
    assert result.retry_count == 1
    assert result.sample_parse_failure_preview == malformed


def test_controller_records_parent_penalty_after_terminal_parse_failure(tmp_path) -> None:
    run_dir = tmp_path / "run"
    malformed = "not a diff"
    config = _config(
        tmp_path,
        max_generations=1,
        max_retries=1,
        stagnation_patience=2,
        database_config=DatabaseConfig(
            islands=2,
            retention=RetentionConfig(
                mutation_failure_streak_threshold=1,
                mutation_failure_penalty_samples=5,
            ),
        ),
    )
    database = ProgramDatabase(config.database, run_dir / "database", random_source=random.Random(0))
    inference_client = StubInferenceClient([malformed])
    evaluator = build_evaluator(config.evaluator, config.sandbox, feedback_client=inference_client)
    controller = EvolutionController(
        config=config,
        database=database,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=PromptBuilder(config.model.prompt_budget),
        run_dir=run_dir,
    )

    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    seed_program = Program.seed(seed_code)
    asyncio.run(controller.run(seed_program))

    assert database._parent_penalty_until_sample == {seed_program.id: 6}
    assert database._parent_mutation_failure_streaks == {}


def test_controller_tracks_apply_failures_for_edits_outside_evolve_block(tmp_path) -> None:
    run_dir = tmp_path / "run"
    outside_boundary_diff = """<<<<<<< SEARCH
    return {"score": total_value}
=======
    return {"score": total_value + 1.0}
>>>>>>> REPLACE"""
    config = _config(tmp_path, max_generations=1, max_retries=1, stagnation_patience=2)
    database = ProgramDatabase(config.database, run_dir / "database", random_source=random.Random(0))
    inference_client = StubInferenceClient([outside_boundary_diff])
    evaluator = build_evaluator(config.evaluator, config.sandbox, feedback_client=inference_client)
    controller = EvolutionController(
        config=config,
        database=database,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=PromptBuilder(config.model.prompt_budget),
        run_dir=run_dir,
    )

    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    result = asyncio.run(controller.run(Program.seed(seed_code)))

    assert result.total_generation_jobs == 1
    assert result.total_evaluated == 1
    assert result.diff_parse_failures == 0
    assert result.diff_apply_failures == 1
    assert result.retry_count == 0
    assert result.sample_apply_failure_preview is not None


def test_controller_allows_full_file_edits_outside_evolve_block(tmp_path) -> None:
    run_dir = tmp_path / "run"
    outside_boundary_diff = """<<<<<<< SEARCH
    return {"score": total_value}
=======
    return {"score": total_value + 1.0}
>>>>>>> REPLACE"""
    config = _config(
        tmp_path,
        max_generations=1,
        max_retries=1,
        stagnation_patience=2,
        mutation_scope="full_file",
    )
    database = ProgramDatabase(config.database, run_dir / "database", random_source=random.Random(0))
    inference_client = StubInferenceClient([outside_boundary_diff])
    evaluator = build_evaluator(config.evaluator, config.sandbox, feedback_client=inference_client)
    controller = EvolutionController(
        config=config,
        database=database,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=PromptBuilder(
            config.model.prompt_budget,
            mutation_scope=config.mutation_scope,
        ),
        run_dir=run_dir,
    )

    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    result = asyncio.run(controller.run(Program.seed(seed_code)))

    assert result.total_generation_jobs == 1
    assert result.diff_parse_failures == 0
    assert result.diff_apply_failures == 0
    assert result.total_evaluated >= 2
    assert result.best_program.primary_score == 276.0
    assert result.stop_reason == "target_score"


def test_controller_stops_on_stagnation(tmp_path) -> None:
    run_dir = tmp_path / "run"
    config = _config(tmp_path, max_generations=5, max_retries=1, stagnation_patience=1)
    database = ProgramDatabase(config.database, run_dir / "database", random_source=random.Random(0))
    inference_client = StubInferenceClient(["nonsense"])
    evaluator = build_evaluator(config.evaluator, config.sandbox, feedback_client=inference_client)
    controller = EvolutionController(
        config=config,
        database=database,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=PromptBuilder(config.model.prompt_budget),
        run_dir=run_dir,
    )

    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    result = asyncio.run(controller.run(Program.seed(seed_code)))

    assert result.stop_reason == "stagnation"
    assert result.total_generation_jobs == 1


def test_controller_does_not_penalize_parent_for_provider_failures(tmp_path) -> None:
    run_dir = tmp_path / "run"
    config = _config(
        tmp_path,
        max_generations=1,
        max_retries=1,
        stagnation_patience=1,
        database_config=DatabaseConfig(
            islands=2,
            retention=RetentionConfig(
                mutation_failure_streak_threshold=1,
                mutation_failure_penalty_samples=5,
            ),
        ),
    )
    database = ProgramDatabase(config.database, run_dir / "database", random_source=random.Random(0))
    inference_client = FailingInferenceClient()
    evaluator = build_evaluator(config.evaluator, config.sandbox, feedback_client=inference_client)
    controller = EvolutionController(
        config=config,
        database=database,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=PromptBuilder(config.model.prompt_budget),
        run_dir=run_dir,
    )

    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    seed_program = Program.seed(seed_code)
    result = asyncio.run(controller.run(seed_program))

    assert result.stop_reason == "stagnation"
    assert database._parent_penalty_until_sample == {}
    assert database._parent_mutation_failure_streaks == {}


def test_controller_resume_uses_checkpoint_and_reaches_target(tmp_path) -> None:
    run_dir = tmp_path / "run"
    first_config = _config(tmp_path, max_generations=1, max_retries=1, stagnation_patience=5)
    first_database = ProgramDatabase(first_config.database, run_dir / "database", random_source=random.Random(0))
    first_inference = StubInferenceClient(["nonsense", DEFAULT_KNAPSACK_DIFF])
    first_evaluator = build_evaluator(first_config.evaluator, first_config.sandbox, feedback_client=first_inference)
    first_controller = EvolutionController(
        config=first_config,
        database=first_database,
        inference_client=first_inference,
        evaluator=first_evaluator,
        prompt_builder=PromptBuilder(first_config.model.prompt_budget),
        run_dir=run_dir,
    )
    seed_code = first_config.seed_program_path.read_text(encoding="utf-8")
    first_result = asyncio.run(first_controller.run(Program.seed(seed_code)))

    assert first_result.stop_reason == "generation_limit"
    assert (run_dir / "checkpoint.json").exists()

    resumed_config = _config(tmp_path, max_generations=2, stagnation_patience=5, resume_from=run_dir)
    resumed_database = ProgramDatabase(resumed_config.database, run_dir / "database", random_source=random.Random(0))
    resumed_inference = StubInferenceClient([DEFAULT_KNAPSACK_DIFF])
    resumed_evaluator = build_evaluator(
        resumed_config.evaluator,
        resumed_config.sandbox,
        feedback_client=resumed_inference,
    )
    resumed_controller = EvolutionController(
        config=resumed_config,
        database=resumed_database,
        inference_client=resumed_inference,
        evaluator=resumed_evaluator,
        prompt_builder=PromptBuilder(resumed_config.model.prompt_budget),
        run_dir=run_dir,
    )

    resumed_result = asyncio.run(resumed_controller.run(Program.seed(seed_code)))

    assert resumed_result.best_program.primary_score == 275.0
    assert resumed_result.stop_reason == "target_score"


def test_controller_runs_multiple_evaluations_in_parallel(tmp_path) -> None:
    run_dir = tmp_path / "run"
    config = _config(
        tmp_path,
        max_generations=4,
        max_retries=1,
        stagnation_patience=10,
        max_inflight=4,
        target_score=999.0,
    )
    database = ProgramDatabase(config.database, run_dir / "database", random_source=random.Random(0))
    inference_client = StubInferenceClient(PARALLEL_KNAPSACK_DIFFS.copy())
    evaluator = TrackingEvaluator(delay_seconds=0.05)
    controller = EvolutionController(
        config=config,
        database=database,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=PromptBuilder(config.model.prompt_budget),
        run_dir=run_dir,
    )

    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    result = asyncio.run(controller.run(Program.seed(seed_code)))

    assert evaluator.max_concurrent_evaluations >= 2
    assert result.total_generation_jobs == 4
    assert result.total_evaluated >= 3
    assert result.stop_reason == "generation_limit"


def test_controller_resets_stagnation_on_behavioral_novelty(tmp_path) -> None:
    run_dir = tmp_path / "run"
    database_config = DatabaseConfig(
        islands=2,
        feature_axes=(
            FeatureAxisConfig(
                name="behavior_axis",
                source="behavior_axis",
                scale="identity",
                bins=(0.0, 0.5, 1.5, 2.5, 3.5),
            ),
        ),
    )
    config = _config(
        tmp_path,
        max_generations=2,
        max_retries=1,
        stagnation_patience=1,
        max_inflight=1,
        target_score=999.0,
        database_config=database_config,
    )
    database = ProgramDatabase(config.database, run_dir / "database", random_source=random.Random(0))
    inference_client = StubInferenceClient([DEFAULT_KNAPSACK_DIFF, PARALLEL_KNAPSACK_DIFFS[0]])
    evaluator = NoveltyEvaluator()
    controller = EvolutionController(
        config=config,
        database=database,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=PromptBuilder(config.model.prompt_budget),
        run_dir=run_dir,
    )

    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    result = asyncio.run(controller.run(Program.seed(seed_code)))

    assert result.total_generation_jobs == 2
    assert result.stop_reason == "generation_limit"
