from __future__ import annotations

import asyncio
from pathlib import Path

from alphaevolve.archive import ProgramDatabase
from alphaevolve.controller import EvolutionController
from alphaevolve.evaluators import FakeEvaluator
from alphaevolve.llm import DEFAULT_KNAPSACK_DIFF, FakeInferenceClient
from alphaevolve.models import ArchiveConfig, ControllerConfig, ExperimentConfig, ModelConfig, Program, PromptBudget, SandboxConfig
from alphaevolve.prompts import PromptBuilder


def test_controller_reaches_knapsack_optimum_with_fake_adapters() -> None:
    seed_code = Path("experiments/knapsack_seed.py").read_text(encoding="utf-8")
    config = ExperimentConfig(
        name="knapsack",
        description="test",
        seed_program_path=Path("experiments/knapsack_seed.py"),
        system_instructions="Improve the provided code.",
        task_description="Maximize the knapsack score.",
        evaluation_contract="Return SEARCH/REPLACE blocks only.",
        primary_metric="score",
        target_score=275.0,
        model=ModelConfig(prompt_budget=PromptBudget(max_prompt_tokens=12_000, reserved_completion_tokens=2_048, max_history_programs=4)),
        sandbox=SandboxConfig(backend="fake"),
        archive=ArchiveConfig(),
        controller=ControllerConfig(max_generations=3, metrics_interval_seconds=30.0),
    )
    controller = EvolutionController(
        config=config,
        archive=ProgramDatabase(config.archive),
        inference_client=FakeInferenceClient([DEFAULT_KNAPSACK_DIFF]),
        evaluator=FakeEvaluator(),
        prompt_builder=PromptBuilder(config.model.prompt_budget),
    )

    result = asyncio.run(controller.run(Program.seed(seed_code)))

    assert result.best_program.primary_score == 275.0
    assert result.stop_reason == "target_score"
    assert result.total_evaluated >= 2


def test_controller_tracks_parse_failures_and_retries() -> None:
    seed_code = Path("experiments/knapsack_seed.py").read_text(encoding="utf-8")
    malformed = "I will explain the change before the diff."
    config = ExperimentConfig(
        name="knapsack",
        description="test",
        seed_program_path=Path("experiments/knapsack_seed.py"),
        system_instructions="Improve the provided code.",
        task_description="Maximize the knapsack score.",
        evaluation_contract="Return SEARCH/REPLACE blocks only.",
        primary_metric="score",
        target_score=float("inf"),
        model=ModelConfig(prompt_budget=PromptBudget(max_prompt_tokens=12_000, reserved_completion_tokens=2_048, max_history_programs=4)),
        sandbox=SandboxConfig(backend="fake"),
        archive=ArchiveConfig(),
        controller=ControllerConfig(max_generations=1, max_retries=2, metrics_interval_seconds=30.0),
    )
    controller = EvolutionController(
        config=config,
        archive=ProgramDatabase(config.archive),
        inference_client=FakeInferenceClient([malformed, DEFAULT_KNAPSACK_DIFF]),
        evaluator=FakeEvaluator(),
        prompt_builder=PromptBuilder(config.model.prompt_budget),
    )

    result = asyncio.run(controller.run(Program.seed(seed_code)))

    assert result.total_generation_jobs == 1
    assert result.diff_parse_failures == 1
    assert result.diff_apply_failures == 0
    assert result.retry_count == 1
    assert result.sample_parse_failure_preview == malformed


def test_controller_tracks_apply_failures_for_edits_outside_evolve_block() -> None:
    seed_code = Path("experiments/knapsack_seed.py").read_text(encoding="utf-8")
    outside_boundary_diff = """<<<<<<< SEARCH
    return {"score": total_value}
=======
    return {"score": total_value + 1.0}
>>>>>>> REPLACE"""
    config = ExperimentConfig(
        name="knapsack",
        description="test",
        seed_program_path=Path("experiments/knapsack_seed.py"),
        system_instructions="Improve the provided code.",
        task_description="Maximize the knapsack score.",
        evaluation_contract="Return SEARCH/REPLACE blocks only.",
        primary_metric="score",
        target_score=float("inf"),
        model=ModelConfig(prompt_budget=PromptBudget(max_prompt_tokens=12_000, reserved_completion_tokens=2_048, max_history_programs=4)),
        sandbox=SandboxConfig(backend="fake"),
        archive=ArchiveConfig(),
        controller=ControllerConfig(max_generations=1, max_retries=1, metrics_interval_seconds=30.0),
    )
    controller = EvolutionController(
        config=config,
        archive=ProgramDatabase(config.archive),
        inference_client=FakeInferenceClient([outside_boundary_diff]),
        evaluator=FakeEvaluator(),
        prompt_builder=PromptBuilder(config.model.prompt_budget),
    )

    result = asyncio.run(controller.run(Program.seed(seed_code)))

    assert result.total_generation_jobs == 1
    assert result.total_evaluated == 1
    assert result.diff_parse_failures == 0
    assert result.diff_apply_failures == 1
    assert result.retry_count == 0
    assert result.sample_apply_failure_preview is not None
