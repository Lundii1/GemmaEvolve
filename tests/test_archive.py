from __future__ import annotations

import asyncio
import json
import random
from collections import deque

import pytest

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


def _program_with_behavior(
    identifier: str,
    code: str,
    score: float,
    *,
    signature_111_ratio: float,
    signature_210_ratio: float,
    signature_300_ratio: float,
    island_id: int = 0,
) -> Program:
    program = _program(identifier, code, score, 1.0, island_id=island_id)
    program.evaluation = EvaluationResult(
        status="success",
        primary_score=score,
        metrics={"score": score},
        features={
            "signature_111_ratio": signature_111_ratio,
            "signature_210_ratio": signature_210_ratio,
            "signature_300_ratio": signature_300_ratio,
        },
        execution=program.execution,
    )
    return program


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
    asyncio.run(database.record(program))

    assert program.archive_cell == (1, 1)
    assert asyncio.run(database.best_program()) is not None
    assert asyncio.run(database.best_program()).id == "a"

    reloaded = ProgramDatabase(config, tmp_path / "db")
    assert asyncio.run(reloaded.best_program()) is not None
    assert asyncio.run(reloaded.best_program()).id == "a"


def test_database_rejects_exact_and_near_duplicates(tmp_path) -> None:
    config = DatabaseConfig(
        novelty=NoveltyConfig(enabled=True, exact_dedupe=True, similarity_threshold=0.9, recent_program_window=10),
    )
    database = ProgramDatabase(config, tmp_path / "db")
    seed = _program("seed", "print('hello world')\n", 1.0, 1.0)
    asyncio.run(database.record_seed_across_islands(seed))

    exact_duplicate = Program(id="dup", code="print('hello world')\n")
    near_duplicate = Program(id="near", code="print('hello worlds')\n")

    assert asyncio.run(database.preflight_candidate(exact_duplicate, island_id=0)) == "exact_duplicate"
    assert asyncio.run(database.preflight_candidate(near_duplicate, island_id=0)) == "near_duplicate"


def test_database_migrates_elites_and_persists_prompt_logs(tmp_path) -> None:
    config = DatabaseConfig(
        islands=2,
        migration=MigrationConfig(enabled=True, interval_generations=2, strategy="best"),
        retention=RetentionConfig(hall_of_fame_size=5, recent_per_cell=3, recent_success_window=5),
    )
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    seed = _program("seed", "print('seed')\n", 1.0, 1.0)
    asyncio.run(database.record_seed_across_islands(seed))
    candidate = _program("p1", "print('candidate')\n", 10.0, 1.0, island_id=0)
    asyncio.run(database.record(candidate))

    prompt_log = PromptLog(
        id="prompt_1",
        parent_id="seed",
        island_id=0,
        prompt_text="prompt",
        estimated_tokens=10,
    )
    asyncio.run(database.record_prompt_log(prompt_log))
    migration = asyncio.run(database.maybe_migrate(generation=2, source_island_id=0))

    assert migration == (1, "p1")
    assert database.get_prompt_log("prompt_1") is not None
    assert any(
        island_id == 1 and cell.elite_id == "p1"
        for island_id, _, cell in database.iter_cells()
    )


def test_database_serializes_concurrent_record_calls(tmp_path) -> None:
    config = DatabaseConfig(
        islands=2,
        retention=RetentionConfig(hall_of_fame_size=20, recent_per_cell=5, recent_success_window=10),
    )
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))

    async def _record_many() -> None:
        await database.record_seed_across_islands(_program("seed", "print('seed')\n", 1.0, 1.0))
        await asyncio.gather(
            *(
                database.record(
                    _program(
                        f"prog_{index}",
                        f"print('candidate {index}')\n",
                        float(index + 2),
                        1.0,
                        island_id=index % config.islands,
                    )
                )
                for index in range(10)
            )
        )

    asyncio.run(_record_many())

    state_path = tmp_path / "db" / "state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(payload["program_ids"]) == 11
    assert len(database.list_programs()) == 11
    assert asyncio.run(database.best_program()) is not None


def test_database_bins_turan_programs_by_behavior_features(tmp_path) -> None:
    config = DatabaseConfig(
        feature_axes=(
            FeatureAxisConfig(
                name="signature_210_ratio",
                source="signature_210_ratio",
                scale="identity",
                bins=(0.0, 0.1, 0.2, 0.3, 0.4),
            ),
            FeatureAxisConfig(
                name="late_case_ratio",
                source="late_case_ratio",
                scale="identity",
                bins=(0.0, 0.5, 0.6, 0.7, 0.8),
            ),
        ),
    )
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    low_mix = _program("low_mix", "print('a')\n", 10.0, 1.0)
    low_mix.features = {"signature_210_ratio": 0.0, "late_case_ratio": 0.45}
    high_mix = _program("high_mix", "print('b')\n", 10.0, 1.0)
    high_mix.features = {"signature_210_ratio": 0.35, "late_case_ratio": 0.72}

    asyncio.run(database.record(low_mix))
    asyncio.run(database.record(high_mix))

    assert low_mix.archive_cell != high_mix.archive_cell


def test_database_rejects_behavior_duplicates_with_cosmetic_code_changes(tmp_path) -> None:
    config = DatabaseConfig()
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    original = _program("original", "print('first')\n", 10.0, 1.0)
    original.metrics = {
        "score": 10.0,
        "n13_edges": 5.0,
        "n14_edges": 5.0,
        "signature_111_edges": 8.0,
        "signature_210_edges": 2.0,
        "signature_300_edges": 0.0,
    }
    original.evaluation = EvaluationResult(
        status="success",
        primary_score=10.0,
        metrics=original.metrics,
        execution=original.execution,
    )
    duplicate = _program("duplicate", "print('second')\n", 10.0, 1.0)
    duplicate.metrics = dict(original.metrics)
    duplicate.evaluation = EvaluationResult(
        status="success",
        primary_score=10.0,
        metrics=duplicate.metrics,
        execution=duplicate.execution,
    )

    asyncio.run(database.record(original))
    asyncio.run(database.record(duplicate))

    assert original.accepted is True
    assert duplicate.accepted is False
    assert duplicate.rejection_reason == "behavior_duplicate"


def test_promising_programs_include_signature_family_specialists(tmp_path) -> None:
    config = DatabaseConfig(
        retention=RetentionConfig(hall_of_fame_size=10, recent_per_cell=5, recent_success_window=10),
    )
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    seed = _program("seed", "print('seed')\n", 1.0, 1.0)
    asyncio.run(database.record_seed_across_islands(seed))

    mix111 = _program_with_behavior(
        "mix111",
        "print('111')\n",
        24.0,
        signature_111_ratio=0.92,
        signature_210_ratio=0.08,
        signature_300_ratio=0.0,
    )
    mix210 = _program_with_behavior(
        "mix210",
        "print('210')\n",
        30.0,
        signature_111_ratio=0.30,
        signature_210_ratio=0.60,
        signature_300_ratio=0.10,
    )
    mix300 = _program_with_behavior(
        "mix300",
        "print('300')\n",
        26.0,
        signature_111_ratio=0.20,
        signature_210_ratio=0.05,
        signature_300_ratio=0.75,
    )

    asyncio.run(database.record(mix111))
    asyncio.run(database.record(mix210))
    asyncio.run(database.record(mix300))

    selected = asyncio.run(database.promising_programs(limit=3, island_id=0, exclude_id="seed"))

    assert [program.id for program in selected] == ["mix210", "mix111", "mix300"]


def test_parent_tracking_state_persists_and_reloads(tmp_path) -> None:
    config = DatabaseConfig()
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    seed = _program("seed", "print('seed')\n", 10.0, 1.0)
    asyncio.run(database.record_seed_across_islands(seed))

    database._record_parent_sample_unlocked("seed")
    database._parent_mutation_failure_streaks["seed"] = 2
    database._parent_penalty_until_sample["seed"] = 7
    database.save()

    reloaded = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))

    assert reloaded._parent_sample_counter == 1
    assert list(reloaded._recent_parent_ids) == ["seed"]
    assert reloaded._parent_mutation_failure_streaks == {"seed": 2}
    assert reloaded._parent_penalty_until_sample == {"seed": 7}


def test_database_loads_v1_state_with_parent_tracking_defaults(tmp_path) -> None:
    config = DatabaseConfig()
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    seed = _program("seed", "print('seed')\n", 10.0, 1.0)
    asyncio.run(database.record_seed_across_islands(seed))

    state_path = tmp_path / "db" / "state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["version"] = 1
    payload.pop("parent_sample_counter", None)
    payload.pop("recent_parent_ids", None)
    payload.pop("parent_mutation_failure_streaks", None)
    payload.pop("parent_penalty_until_sample", None)
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    reloaded = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))

    assert reloaded._parent_sample_counter == 0
    assert list(reloaded._recent_parent_ids) == []
    assert reloaded._parent_mutation_failure_streaks == {}
    assert reloaded._parent_penalty_until_sample == {}


def test_database_prunes_stale_parent_tracking_state_on_save(tmp_path) -> None:
    config = DatabaseConfig()
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    seed = _program("seed", "print('seed')\n", 10.0, 1.0)
    asyncio.run(database.record_seed_across_islands(seed))

    database._parent_sample_counter = 2
    database._parent_mutation_failure_streaks = {"seed": 1, "stale": 4}
    database._parent_penalty_until_sample = {"seed": 8, "stale": 9}
    database.save()

    payload = json.loads((tmp_path / "db" / "state.json").read_text(encoding="utf-8"))

    assert payload["parent_mutation_failure_streaks"] == {"seed": 1}
    assert payload["parent_penalty_until_sample"] == {"seed": 8}


def test_sample_attempts_lanes_in_deterministic_fallback_order(tmp_path) -> None:
    config = DatabaseConfig()
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    attempts: list[tuple[str, str]] = []

    class StubRandom:
        def choices(self, population, weights=None, k=1):
            return ["recent"]

    database._random = StubRandom()

    def _fake_sample_lane(island_id: int, lane: str, *, relaxation: str):
        attempts.append((relaxation, lane))
        return None

    database._sample_lane = _fake_sample_lane  # type: ignore[method-assign]

    with pytest.raises(ValueError):
        asyncio.run(database.sample(0))

    assert attempts == [
        ("STRICT", "recent"),
        ("STRICT", "elite"),
        ("STRICT", "hall_of_fame"),
        ("NO_SHARE_CAP", "recent"),
        ("NO_SHARE_CAP", "elite"),
        ("NO_SHARE_CAP", "hall_of_fame"),
        ("NO_COOLDOWN", "recent"),
        ("NO_COOLDOWN", "elite"),
        ("NO_COOLDOWN", "hall_of_fame"),
    ]


def test_share_cap_blocks_dominant_parent_when_alternatives_exist(tmp_path) -> None:
    config = DatabaseConfig(
        retention=RetentionConfig(parent_share_window=4, parent_share_cap=0.5),
    )
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    seed = _program("seed", "print('seed')\n", 10.0, 1.0)
    alt = _program("alt", "print('alt')\n", 5.0, 1.0)
    asyncio.run(database.record_seed_across_islands(seed))
    asyncio.run(database.record(alt))

    database._recent_parent_ids = deque(["seed", "seed", "seed", "alt"], maxlen=4)

    sampled = database._sample_lane(0, "hall_of_fame", relaxation="STRICT")

    assert sampled is not None
    assert sampled.id == "alt"


def test_best_parent_cooldown_expires_after_recent_window_moves_past_best(tmp_path) -> None:
    config = DatabaseConfig(
        retention=RetentionConfig(best_parent_cooldown_samples=2),
    )
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    seed = _program("seed", "print('seed')\n", 10.0, 1.0)
    alt = _program("alt", "print('alt')\n", 5.0, 1.0)
    asyncio.run(database.record_seed_across_islands(seed))
    asyncio.run(database.record(alt))

    database._recent_parent_ids = deque(["seed", "alt"], maxlen=4)
    assert database._is_best_parent_on_cooldown("seed") is True

    database._recent_parent_ids = deque(["seed", "alt", "alt"], maxlen=4)
    assert database._is_best_parent_on_cooldown("seed") is False


def test_recent_lane_applies_novelty_bonus_only_to_behaviorally_novel_programs(tmp_path) -> None:
    config = DatabaseConfig()
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    program = _program("recent", "print('recent')\n", 3.0, 1.0)
    database._programs[program.id] = program
    database._islands[0].recent_success_ids.append(program.id)

    program.behaviorally_novel = False
    without_bonus = database._lane_candidates(0, "recent")[0][1]

    program.behaviorally_novel = True
    with_bonus = database._lane_candidates(0, "recent")[0][1]

    assert without_bonus == 1.0
    assert with_bonus == 1.5


def test_recent_lane_downweights_recently_oversampled_parents(tmp_path) -> None:
    config = DatabaseConfig()
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    program = _program("recent", "print('recent')\n", 3.0, 1.0)
    database._programs[program.id] = program
    database._islands[0].recent_success_ids.append(program.id)
    database._recent_parent_ids = deque(["recent", "recent", "recent"], maxlen=64)

    weight = database._lane_candidates(0, "recent")[0][1]

    assert weight == pytest.approx(0.25)


def test_hall_of_fame_weighting_handles_negative_scores(tmp_path) -> None:
    config = DatabaseConfig()
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    program = _program("negative", "print('negative')\n", -5.0, 1.0)
    database._programs[program.id] = program
    database._hall_of_fame = [program.id]

    weight = database._lane_candidates(0, "hall_of_fame")[0][1]

    assert weight == pytest.approx(1.0)


def test_parent_mutation_penalty_activates_after_terminal_failures_and_clears_on_success(tmp_path) -> None:
    config = DatabaseConfig(
        retention=RetentionConfig(
            mutation_failure_streak_threshold=3,
            mutation_failure_penalty_samples=4,
        ),
    )
    database = ProgramDatabase(config, tmp_path / "db", random_source=random.Random(0))
    seed = _program("seed", "print('seed')\n", 10.0, 1.0)
    asyncio.run(database.record_seed_across_islands(seed))

    asyncio.run(database.record_parent_mutation_outcome("seed", success=False))
    asyncio.run(database.record_parent_mutation_outcome("seed", success=False))

    assert database._parent_mutation_failure_streaks == {"seed": 2}
    assert database._parent_penalty_until_sample == {}

    asyncio.run(database.record_parent_mutation_outcome("seed", success=False))

    assert database._parent_mutation_failure_streaks == {}
    assert database._parent_penalty_until_sample == {"seed": 4}

    asyncio.run(database.record_parent_mutation_outcome("seed", success=True))

    assert database._parent_mutation_failure_streaks == {}
    assert database._parent_penalty_until_sample == {}
