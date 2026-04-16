from __future__ import annotations

import asyncio
import random
from pathlib import Path

from alphaevolve.archive import ProgramDatabase
from alphaevolve.controller import EvolutionController
from alphaevolve.evaluators import build_evaluator
from alphaevolve.llm import AsyncInferenceClient
from alphaevolve.models import (
    CheckpointConfig,
    ControllerConfig,
    DatabaseConfig,
    EvaluatorConfig,
    ExperimentConfig,
    ModelConfig,
    Program,
    PromptBudget,
    SandboxConfig,
)
from alphaevolve.prompts import PromptBuilder

DEFAULT_KNAPSACK_DIFF = """<<<<<<< SEARCH
    return value
=======
    return value / max(weight, 1e-9)
>>>>>>> REPLACE"""


class StubInferenceClient(AsyncInferenceClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate_text(self, prompt: str, *, attempt: int = 1) -> str:
        if not self._responses:
            raise AssertionError("No more stub responses available.")
        return self._responses.pop(0)


def _config(tmp_path, *, max_generations: int, max_retries: int = 2, stagnation_patience: int = 4, resume_from=None):
    root = Path("experiments").resolve()
    return ExperimentConfig(
        name="knapsack",
        description="test",
        seed_program_path=(root / "knapsack_seed.py").resolve(),
        system_instructions="Improve the provided code.",
        task_description="Maximize the knapsack score.",
        evaluation_contract="Return SEARCH/REPLACE blocks only.",
        primary_metric="score",
        target_score=275.0,
        model=ModelConfig(
            provider="ollama",
            prompt_budget=PromptBudget(
                max_prompt_tokens=12_000,
                reserved_completion_tokens=2_048,
                max_history_programs=4,
            ),
        ),
        sandbox=SandboxConfig(backend="python", timeout_seconds=5.0),
        database=DatabaseConfig(islands=2),
        evaluator=EvaluatorConfig(module=(root / "knapsack_evaluator.py").resolve()),
        checkpoint=CheckpointConfig(enabled=True, interval_generations=1, resume_from=resume_from),
        controller=ControllerConfig(
            max_generations=max_generations,
            max_retries=max_retries,
            metrics_interval_seconds=30.0,
            stagnation_patience=stagnation_patience,
            max_inflight=1,
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
