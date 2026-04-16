"""Filesystem-backed program database with island-aware MAP-Elites cells."""

from __future__ import annotations

import bisect
import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from alphaevolve.models import (
    ArtifactRecord,
    DatabaseConfig,
    Program,
    PromptLog,
    decode_random_state,
    encode_random_state,
)


@dataclass(slots=True)
class ArchiveCell:
    """Per-cell state in the MAP-Elites archive."""

    elite_id: str | None = None
    recent_ids: deque[str] = field(default_factory=deque)
    visits: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "elite_id": self.elite_id,
            "recent_ids": list(self.recent_ids),
            "visits": self.visits,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, recent_per_cell: int) -> "ArchiveCell":
        return cls(
            elite_id=str(data["elite_id"]) if data.get("elite_id") is not None else None,
            recent_ids=deque(
                (str(item) for item in data.get("recent_ids", [])),
                maxlen=recent_per_cell,
            ),
            visits=int(data.get("visits", 0)),
        )


@dataclass(slots=True)
class IslandState:
    """Per-island database state."""

    island_id: int
    cells: dict[tuple[int, ...], ArchiveCell] = field(default_factory=dict)
    recent_success_ids: deque[str] = field(default_factory=deque)

    def to_dict(self) -> dict[str, Any]:
        return {
            "island_id": self.island_id,
            "cells": {
                ",".join(str(part) for part in key): cell.to_dict()
                for key, cell in self.cells.items()
            },
            "recent_success_ids": list(self.recent_success_ids),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        recent_per_cell: int,
        recent_success_window: int,
    ) -> "IslandState":
        cells = {
            tuple(int(part) for part in key.split(",")): ArchiveCell.from_dict(
                value,
                recent_per_cell=recent_per_cell,
            )
            for key, value in dict(data.get("cells", {})).items()
        }
        return cls(
            island_id=int(data["island_id"]),
            cells=cells,
            recent_success_ids=deque(
                (str(item) for item in data.get("recent_success_ids", [])),
                maxlen=recent_success_window,
            ),
        )


class ProgramDatabase:
    """Persisted program database keyed by feature-map cells within islands."""

    def __init__(
        self,
        config: DatabaseConfig,
        root_dir: str | Path,
        *,
        random_source: random.Random | None = None,
    ) -> None:
        self._config = config
        self._root_dir = Path(root_dir).resolve()
        self._programs_dir = self._root_dir / "programs"
        self._prompts_dir = self._root_dir / "prompts"
        self._artifacts_dir = self._root_dir / "artifacts"
        self._state_path = self._root_dir / "state.json"
        self._programs: dict[str, Program] = {}
        self._prompt_logs: dict[str, PromptLog] = {}
        self._hall_of_fame: list[str] = []
        self._best_program_id: str | None = None
        self._accepted_code_hashes: dict[str, str] = {}
        self._recent_code_ids: dict[int, deque[str]] = {
            island_id: deque(maxlen=config.novelty.recent_program_window)
            for island_id in range(config.islands)
        }
        self._islands: dict[int, IslandState] = {
            island_id: IslandState(
                island_id=island_id,
                recent_success_ids=deque(maxlen=config.retention.recent_success_window),
            )
            for island_id in range(config.islands)
        }
        self._migration_events = 0
        self._random = random_source or random.Random()
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._programs_dir.mkdir(parents=True, exist_ok=True)
        self._prompts_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        if self._state_path.exists():
            self.load()

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def save(self) -> None:
        """Persist metadata and lightweight indexes to disk."""
        payload = {
            "version": 1,
            "best_program_id": self._best_program_id,
            "hall_of_fame": self._hall_of_fame,
            "accepted_code_hashes": self._accepted_code_hashes,
            "migration_events": self._migration_events,
            "database_random_state": encode_random_state(self._random.getstate()),
            "program_ids": sorted(self._programs),
            "prompt_log_ids": sorted(self._prompt_logs),
            "recent_code_ids": {
                str(island_id): list(ids)
                for island_id, ids in self._recent_code_ids.items()
            },
            "islands": {
                str(island_id): state.to_dict()
                for island_id, state in self._islands.items()
            },
        }
        self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Load persisted database state from disk."""
        if not self._state_path.exists():
            return
        payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        self._programs = {}
        for program_id in payload.get("program_ids", []):
            program_path = self._programs_dir / f"{program_id}.json"
            if program_path.exists():
                self._programs[program_id] = Program.from_dict(
                    json.loads(program_path.read_text(encoding="utf-8"))
                )
        self._prompt_logs = {}
        for prompt_log_id in payload.get("prompt_log_ids", []):
            prompt_path = self._prompts_dir / f"{prompt_log_id}.json"
            if prompt_path.exists():
                self._prompt_logs[prompt_log_id] = PromptLog.from_dict(
                    json.loads(prompt_path.read_text(encoding="utf-8"))
                )
        self._best_program_id = (
            str(payload["best_program_id"]) if payload.get("best_program_id") is not None else None
        )
        self._hall_of_fame = [str(item) for item in payload.get("hall_of_fame", [])]
        self._accepted_code_hashes = {
            str(key): str(value)
            for key, value in dict(payload.get("accepted_code_hashes", {})).items()
        }
        self._migration_events = int(payload.get("migration_events", 0))
        if payload.get("database_random_state"):
            self._random.setstate(decode_random_state(str(payload["database_random_state"])))
        self._recent_code_ids = {
            island_id: deque(
                (
                    str(item)
                    for item in dict(payload.get("recent_code_ids", {})).get(str(island_id), [])
                ),
                maxlen=self._config.novelty.recent_program_window,
            )
            for island_id in range(self._config.islands)
        }
        self._islands = {
            island_id: IslandState.from_dict(
                dict(payload.get("islands", {})).get(str(island_id), {"island_id": island_id}),
                recent_per_cell=self._config.retention.recent_per_cell,
                recent_success_window=self._config.retention.recent_success_window,
            )
            for island_id in range(self._config.islands)
        }

    def record_prompt_log(self, prompt_log: PromptLog) -> PromptLog:
        """Persist a prompt log."""
        self._prompt_logs[prompt_log.id] = prompt_log
        self._write_json(self._prompts_dir / f"{prompt_log.id}.json", prompt_log.to_dict())
        self.save()
        return prompt_log

    def update_prompt_log(
        self,
        prompt_log_id: str,
        *,
        model_response: str | None = None,
        child_id: str | None = None,
    ) -> PromptLog:
        """Update a previously recorded prompt log."""
        prompt_log = self._prompt_logs[prompt_log_id]
        updated = PromptLog(
            id=prompt_log.id,
            parent_id=prompt_log.parent_id,
            island_id=prompt_log.island_id,
            prompt_text=prompt_log.prompt_text,
            estimated_tokens=prompt_log.estimated_tokens,
            included_program_ids=prompt_log.included_program_ids,
            summarized_program_ids=prompt_log.summarized_program_ids,
            artifact_context=prompt_log.artifact_context,
            evaluator_feedback_used=prompt_log.evaluator_feedback_used,
            model_response=model_response if model_response is not None else prompt_log.model_response,
            child_id=child_id if child_id is not None else prompt_log.child_id,
            created_at=prompt_log.created_at,
        )
        self._prompt_logs[prompt_log_id] = updated
        self._write_json(self._prompts_dir / f"{prompt_log_id}.json", updated.to_dict())
        self.save()
        return updated

    def get_prompt_log(self, prompt_log_id: str) -> PromptLog | None:
        return self._prompt_logs.get(prompt_log_id)

    def get_program(self, program_id: str) -> Program | None:
        return self._programs.get(program_id)

    def record_seed_across_islands(self, seed_program: Program) -> Program:
        """Record the seed once and share it across all islands."""
        seed_program.island_id = 0
        seed_program.accepted = True
        seed_program.features = self._raw_feature_values(seed_program)
        seed_program.archive_cell = self.cell_key_for(seed_program)
        self._record_program(seed_program)
        self._accepted_code_hashes[seed_program.code_hash] = seed_program.id
        self._register_program_in_island(seed_program.id, island_id=0)
        self._recent_code_ids[0].append(seed_program.id)
        for island_id in range(1, self._config.islands):
            self._register_program_in_island(seed_program.id, island_id=island_id)
            self._recent_code_ids[island_id].append(seed_program.id)
        self._update_hall_of_fame(seed_program.id)
        self.save()
        return seed_program

    def preflight_candidate(self, program: Program, *, island_id: int) -> str | None:
        """Return a rejection reason if the candidate is not novel enough."""
        if not self._config.novelty.enabled:
            return None
        if self._config.novelty.exact_dedupe and program.code_hash in self._accepted_code_hashes:
            return "exact_duplicate"
        for recent_program_id in self._recent_code_ids[island_id]:
            prior = self._programs.get(recent_program_id)
            if prior is None:
                continue
            similarity = SequenceMatcher(a=program.code, b=prior.code).ratio()
            if similarity >= self._config.novelty.similarity_threshold:
                return "near_duplicate"
        return None

    def record(self, program: Program) -> Program:
        """Persist a program and, when accepted, update island state."""
        program.features = self._raw_feature_values(program)
        program.archive_cell = self.cell_key_for(program)
        self._record_program(program)
        if program.accepted:
            self._accepted_code_hashes[program.code_hash] = program.id
            self._recent_code_ids[program.island_id].append(program.id)
            self._register_program_in_island(program.id, island_id=program.island_id)
            self._update_hall_of_fame(program.id)
        self.save()
        return program

    def maybe_migrate(self, *, generation: int, source_island_id: int) -> tuple[int, str] | None:
        """Migrate an elite across the ring topology when the interval is reached."""
        migration = self._config.migration
        if not migration.enabled:
            return None
        if generation == 0 or generation % max(migration.interval_generations, 1) != 0:
            return None
        source_island = self._islands[source_island_id]
        candidate_ids = [
            cell.elite_id
            for cell in source_island.cells.values()
            if cell.elite_id is not None and self._programs.get(cell.elite_id) is not None
        ]
        if not candidate_ids:
            return None
        if migration.strategy == "diverse":
            program_id = max(
                candidate_ids,
                key=lambda item: (
                    self._diversity_score(self._programs[item].code, source_island_id),
                    self._programs[item].primary_score,
                ),
            )
        else:
            program_id = max(candidate_ids, key=lambda item: self._programs[item].primary_score)
        target_island_id = (source_island_id + 1) % self._config.islands
        program = self._programs[program_id]
        if self.preflight_candidate(program, island_id=target_island_id) == "near_duplicate":
            return None
        self._register_program_in_island(program_id, island_id=target_island_id)
        self._migration_events += 1
        self.save()
        return target_island_id, program_id

    def sample(self, island_id: int) -> Program:
        """Sample a parent from a chosen island."""
        island = self._islands[island_id]
        lane_names = ["elite", "recent", "hall_of_fame"]
        lane_weights = [
            self._config.retention.sample_elite_weight,
            self._config.retention.sample_recent_weight,
            self._config.retention.sample_hall_of_fame_weight,
        ]
        lane = self._random.choices(lane_names, weights=lane_weights, k=1)[0]
        program = self._sample_lane(island_id, lane)
        if program is not None:
            return program
        for fallback_lane in lane_names:
            program = self._sample_lane(island_id, fallback_lane)
            if program is not None:
                return program
        raise ValueError("Cannot sample from an empty database island.")

    def best_program(self) -> Program | None:
        """Return the current best program if present."""
        return self._programs.get(self._best_program_id) if self._best_program_id else None

    def promising_programs(
        self,
        limit: int,
        *,
        island_id: int,
        exclude_id: str | None = None,
    ) -> list[Program]:
        """Return high-quality programs suitable for prompt history."""
        seen: set[str] = set()
        candidates: list[Program] = []
        island = self._islands[island_id]
        for program_id in island.recent_success_ids:
            if program_id == exclude_id or program_id in seen:
                continue
            program = self._programs.get(program_id)
            if program is None or not program.accepted:
                continue
            seen.add(program_id)
            candidates.append(program)
        for program_id in self._hall_of_fame:
            if program_id == exclude_id or program_id in seen:
                continue
            program = self._programs.get(program_id)
            if program is None or not program.accepted:
                continue
            seen.add(program_id)
            candidates.append(program)
        ranked = sorted(candidates, key=lambda item: item.primary_score, reverse=True)
        return ranked[:limit]

    def cell_key_for(self, program: Program) -> tuple[int, ...]:
        """Compute the MAP-Elites bin for a program using configured feature axes."""
        features = self._raw_feature_values(program)
        key: list[int] = []
        for axis in self._config.feature_axes:
            raw_value = float(features.get(axis.source, 0.0))
            scaled_value = self._scale_value(raw_value, axis.scale)
            key.append(self._bucket_index(axis.bins, scaled_value))
        return tuple(key)

    def stats(self) -> dict[str, int | float]:
        """Return basic database statistics for logging."""
        best = self.best_program()
        populated_cells = sum(len(island.cells) for island in self._islands.values())
        return {
            "programs": len(self._programs),
            "prompt_logs": len(self._prompt_logs),
            "islands": self._config.islands,
            "cells": populated_cells,
            "hall_of_fame": len(self._hall_of_fame),
            "best_score": best.primary_score if best else float("-inf"),
            "migrations": self._migration_events,
        }

    def iter_cells(self) -> Iterable[tuple[int, tuple[int, ...], ArchiveCell]]:
        """Yield cells for tests and debugging."""
        for island_id, island in self._islands.items():
            for cell_key, cell in island.cells.items():
                yield island_id, cell_key, cell

    def list_programs(self) -> list[Program]:
        return list(self._programs.values())

    def list_prompt_logs(self) -> list[PromptLog]:
        return list(self._prompt_logs.values())

    def _sample_lane(self, island_id: int, lane: str) -> Program | None:
        island = self._islands[island_id]
        if lane == "elite":
            elite_ids = [
                cell.elite_id
                for cell in island.cells.values()
                if cell.elite_id is not None and self._programs.get(cell.elite_id) is not None
            ]
            if not elite_ids:
                return None
            return self._programs[self._random.choice(elite_ids)]
        if lane == "recent":
            recent_ids = [
                program_id
                for program_id in island.recent_success_ids
                if self._programs.get(program_id) is not None
            ]
            if not recent_ids:
                return None
            return self._programs[self._random.choice(recent_ids)]
        if not self._hall_of_fame:
            return None
        weighted_ids = [
            program_id
            for program_id in self._hall_of_fame
            if self._programs.get(program_id) is not None
        ]
        if not weighted_ids:
            return None
        weights = [max(self._programs[program_id].primary_score, 0.0) + 1.0 for program_id in weighted_ids]
        return self._programs[self._random.choices(weighted_ids, weights=weights, k=1)[0]]

    def _register_program_in_island(self, program_id: str, *, island_id: int) -> None:
        program = self._programs[program_id]
        cell_key = self.cell_key_for(program)
        island = self._islands[island_id]
        cell = island.cells.setdefault(
            cell_key,
            ArchiveCell(recent_ids=deque(maxlen=self._config.retention.recent_per_cell)),
        )
        cell.visits += 1
        cell.recent_ids.append(program_id)
        island.recent_success_ids.append(program_id)
        current_elite = self._programs.get(cell.elite_id) if cell.elite_id is not None else None
        if current_elite is None or program.primary_score >= current_elite.primary_score:
            cell.elite_id = program_id

    def _record_program(self, program: Program) -> None:
        self._programs[program.id] = program
        self._write_json(self._programs_dir / f"{program.id}.json", program.to_dict())

    def _update_hall_of_fame(self, program_id: str) -> None:
        ranked_ids = {item: item for item in self._hall_of_fame}
        ranked_ids[program_id] = program_id
        scored = sorted(
            ranked_ids.values(),
            key=lambda item: self._programs[item].primary_score,
            reverse=True,
        )
        self._hall_of_fame = scored[: self._config.retention.hall_of_fame_size]
        if self._hall_of_fame:
            self._best_program_id = self._hall_of_fame[0]

    def _raw_feature_values(self, program: Program) -> dict[str, float]:
        execution_duration_ms = 0.0
        if program.execution is not None:
            execution_duration_ms = program.execution.duration_ms
        values = {
            "code_bytes": float(len(program.code.encode("utf-8"))),
            "duration_ms": float(execution_duration_ms),
            "primary_score": float(program.primary_score if math.isfinite(program.primary_score) else 0.0),
        }
        values.update({name: float(value) for name, value in program.features.items()})
        if program.evaluation is not None:
            values.update({name: float(value) for name, value in program.evaluation.features.items()})
        return values

    def _bucket_index(self, boundaries: tuple[float, ...], value: float) -> int:
        return max(0, bisect.bisect_right(boundaries, value) - 1)

    def _scale_value(self, value: float, scale: str) -> float:
        if scale == "log1p":
            return math.log1p(max(value, 0.0))
        return value

    def _diversity_score(self, code: str, island_id: int) -> float:
        scores: list[float] = []
        for program_id in self._recent_code_ids[island_id]:
            prior = self._programs.get(program_id)
            if prior is None:
                continue
            scores.append(1.0 - SequenceMatcher(a=code, b=prior.code).ratio())
        return sum(scores) / len(scores) if scores else 1.0

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
