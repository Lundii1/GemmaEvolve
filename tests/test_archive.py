from __future__ import annotations

import random

from alphaevolve.archive import ProgramDatabase
from alphaevolve.models import (
    DatabaseConfig,
    EvaluationResult,
    ExecutionResult,
    FeatureAxisConfig,
    MigrationConfig,
    NoveltyConfig,
    Program,
    PromptLog,
    RetentionConfig,
)


def _program(identifier: str, code: str, score: float, duration_ms: float, *, island_id: int = 0) -> Program:
    execution = ExecutionResult(duration_ms=duration_ms, status="success", exit_code=0)
    evaluation = EvaluationResult(
        status="success",
        primary_score=score,
        metrics={"score": score},
        features={"custom_feature": 5.0},
        execution=execution,
    )
    return Program(
        id=identifier,
        code=code,
        primary_score=score,
        metrics={"score": score},
        execution=execution,
        evaluation=evaluation,
        island_id=island_id,
        accepted=True,
    )


def test_database_bins_programs_by_scaled_features_and_persists(tmp_path) -> None:
    config = DatabaseConfig(
        islands=2,
        feature_axes=(
            FeatureAxisConfig(name="code_bytes", source="code_bytes", scale="log1p", bins=(0.0, 2.0, 3.0, 6.0)),
            FeatureAxisConfig(name="duration_ms", source="duration_ms", scale="identity", bins=(0.0, 5.0, 10.0)),
        ),
    )
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))

    program = _program("a", "print('hello')", 10.0, 7.0)
    database.record(program)

    assert program.archive_cell == (1, 1)
    assert database.best_program() is not None
    assert database.best_program().id == "a"

    reloaded = ProgramDatabase(config, tmp_path / "db")
    assert reloaded.best_program() is not None
    assert reloaded.best_program().id == "a"


def test_database_rejects_exact_and_near_duplicates(tmp_path) -> None:
    config = DatabaseConfig(
        novelty=NoveltyConfig(enabled=True, exact_dedupe=True, similarity_threshold=0.9, recent_program_window=10),
    )
    database = ProgramDatabase(config, tmp_path / "db")
    seed = _program("seed", "print('hello world')\n", 1.0, 1.0)
    database.record_seed_across_islands(seed)

    exact_duplicate = Program(id="dup", code="print('hello world')\n")
    near_duplicate = Program(id="near", code="print('hello worlds')\n")

    assert database.preflight_candidate(exact_duplicate, island_id=0) == "exact_duplicate"
    assert database.preflight_candidate(near_duplicate, island_id=0) == "near_duplicate"


def test_database_migrates_elites_and_persists_prompt_logs(tmp_path) -> None:
    config = DatabaseConfig(
        islands=2,
        migration=MigrationConfig(enabled=True, interval_generations=2, strategy="best"),
        retention=RetentionConfig(hall_of_fame_size=5, recent_per_cell=3, recent_success_window=5),
    )
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    seed = _program("seed", "print('seed')\n", 1.0, 1.0)
    database.record_seed_across_islands(seed)
    candidate = _program("p1", "print('candidate')\n", 10.0, 1.0, island_id=0)
    database.record(candidate)

    prompt_log = PromptLog(
        id="prompt_1",
        parent_id="seed",
        island_id=0,
        prompt_text="prompt",
        estimated_tokens=10,
    )
    database.record_prompt_log(prompt_log)
    migration = database.maybe_migrate(generation=2, source_island_id=0)

    assert migration == (1, "p1")
    assert database.get_prompt_log("prompt_1") is not None
    assert any(
        island_id == 1 and cell.elite_id == "p1"
        for island_id, _, cell in database.iter_cells()
    )
