from __future__ import annotations

import random
from unittest.mock import patch

from alphaevolve.archive import ProgramDatabase
from alphaevolve.models import ArchiveConfig, ExecutionResult, Program


def _program(identifier: str, code: str, score: float, duration_ms: float) -> Program:
    return Program(
        id=identifier,
        code=code,
        primary_score=score,
        metrics={"score": score},
        execution=ExecutionResult(duration_ms=duration_ms, status="success", exit_code=0),
    )


def test_archive_bins_programs_by_code_length_and_eval_time() -> None:
    config = ArchiveConfig(
        code_length_buckets=(0, 10, 20, 30),
        eval_time_buckets_ms=(0, 5, 10, 20),
        hall_of_fame_size=5,
        recent_per_cell=3,
    )
    database = ProgramDatabase(config)

    program = _program("a", "print('hello')", 10.0, 7.0)
    database.record(program)

    assert program.archive_cell == (1, 1)
    assert database.best_program().id == "a"


def test_archive_sample_supports_each_sampling_lane() -> None:
    database = ProgramDatabase(ArchiveConfig(), random_source=random.Random(0))
    p1 = database.record(_program("p1", "a" * 200, 100.0, 20.0))
    p2 = database.record(_program("p2", "b" * 400, 90.0, 30.0))
    database.record(_program("p3", "c" * 600, 80.0, 5.0))

    with patch.object(database._random, "random", return_value=0.1):
        assert database.sample().id in {p1.id, p2.id}
    with patch.object(database._random, "random", return_value=0.6):
        assert database.sample().id in {"p1", "p2", "p3"}
    with patch.object(database._random, "random", return_value=0.95):
        assert database.sample().id in {"p1", "p2", "p3"}
