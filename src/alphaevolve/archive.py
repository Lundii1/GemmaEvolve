"""Lightweight MAP-Elites style archive."""

from __future__ import annotations

import bisect
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from alphaevolve.models import ArchiveConfig, Program


@dataclass(slots=True)
class ArchiveCell:
    """Per-cell state in the MAP-Elites archive."""

    elite: Program | None = None
    recent: deque[Program] = field(default_factory=deque)
    visits: int = 0


class ProgramDatabase:
    """In-memory archive keyed by code length and evaluation time buckets."""

    def __init__(self, config: ArchiveConfig, *, random_source: random.Random | None = None) -> None:
        self._config = config
        self._cells: dict[tuple[int, int], ArchiveCell] = {}
        self._hall_of_fame: list[Program] = []
        self._random = random_source or random.Random()

    def record(self, program: Program) -> Program:
        """Insert a program into the archive and update elites."""
        key = self.cell_key_for(program)
        cell = self._cells.setdefault(
            key,
            ArchiveCell(recent=deque(maxlen=self._config.recent_per_cell)),
        )
        cell.visits += 1
        program.archive_cell = key
        cell.recent.append(program)
        if cell.elite is None or program.primary_score >= cell.elite.primary_score:
            cell.elite = program
        self._update_hall_of_fame(program)
        return program

    def sample(self) -> Program:
        """Sample a parent with a fixed exploration/exploitation mix."""
        if not self._hall_of_fame:
            raise ValueError("Cannot sample from an empty archive.")

        roll = self._random.random()
        if roll < 0.5:
            return self._sample_hall_of_fame()
        if roll < 0.8:
            sparse = self._sample_sparse_cell()
            if sparse is not None:
                return sparse
        recent = self._sample_recent_program()
        if recent is not None:
            return recent
        return self._sample_hall_of_fame()

    def best_program(self) -> Program | None:
        """Return the current best program if present."""
        return self._hall_of_fame[0] if self._hall_of_fame else None

    def promising_programs(self, limit: int, *, exclude_id: str | None = None) -> list[Program]:
        """Return top programs suitable for prompt history."""
        programs = [
            program for program in self._hall_of_fame if exclude_id is None or program.id != exclude_id
        ]
        return programs[:limit]

    def cell_key_for(self, program: Program) -> tuple[int, int]:
        """Compute the 2D MAP-Elites bin for a program."""
        code_length = len(program.code.encode("utf-8"))
        duration_ms = program.execution.duration_ms if program.execution else 0.0
        return (
            self._bucket_index(self._config.code_length_buckets, code_length),
            self._bucket_index(self._config.eval_time_buckets_ms, duration_ms),
        )

    def stats(self) -> dict[str, int | float]:
        """Return basic archive statistics for logging."""
        best = self.best_program()
        return {
            "cells": len(self._cells),
            "hall_of_fame": len(self._hall_of_fame),
            "best_score": best.primary_score if best else float("-inf"),
        }

    def iter_cells(self) -> Iterable[tuple[tuple[int, int], ArchiveCell]]:
        """Yield cell contents for tests and debugging."""
        return self._cells.items()

    def _bucket_index(self, boundaries: tuple[int, ...], value: float) -> int:
        return max(0, bisect.bisect_right(boundaries, value) - 1)

    def _sample_hall_of_fame(self) -> Program:
        scores = [max(program.primary_score, 0.0) + 1.0 for program in self._hall_of_fame]
        return self._random.choices(self._hall_of_fame, weights=scores, k=1)[0]

    def _sample_sparse_cell(self) -> Program | None:
        candidates = [
            (key, cell) for key, cell in self._cells.items() if cell.elite is not None
        ]
        if not candidates:
            return None
        weights = [1.0 / (1 + cell.visits) for _, cell in candidates]
        _, cell = self._random.choices(candidates, weights=weights, k=1)[0]
        return cell.elite

    def _sample_recent_program(self) -> Program | None:
        recent_programs = [program for cell in self._cells.values() for program in cell.recent]
        if not recent_programs:
            return None
        return self._random.choice(recent_programs)

    def _update_hall_of_fame(self, program: Program) -> None:
        by_id = {existing.id: existing for existing in self._hall_of_fame}
        by_id[program.id] = program
        ranked = sorted(by_id.values(), key=lambda item: item.primary_score, reverse=True)
        self._hall_of_fame = ranked[: self._config.hall_of_fame_size]
