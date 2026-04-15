"""Structured logging helpers for the async controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter


def configure_logging(level: str = "INFO") -> None:
    """Configure a concise log format for CLI runs."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@dataclass(slots=True)
class PipelineStats:
    """Counters emitted periodically during controller execution."""

    started_at: float = field(default_factory=perf_counter)
    mutation_jobs_started: int = 0
    diff_failures: int = 0
    retry_count: int = 0
    evaluation_count: int = 0
    sandbox_failures: int = 0
    best_score: float = float("-inf")

    def snapshot(self, *, pending_prompts: int, pending_evaluations: int) -> dict[str, int | float]:
        elapsed = max(perf_counter() - self.started_at, 1e-6)
        return {
            "elapsed_seconds": round(elapsed, 3),
            "mutation_jobs_started": self.mutation_jobs_started,
            "diff_failures": self.diff_failures,
            "retry_count": self.retry_count,
            "evaluation_count": self.evaluation_count,
            "sandbox_failures": self.sandbox_failures,
            "best_score": round(self.best_score, 3) if self.best_score != float("-inf") else self.best_score,
            "generation_throughput": round(self.mutation_jobs_started / elapsed, 3),
            "evaluation_throughput": round(self.evaluation_count / elapsed, 3),
            "pending_prompts": pending_prompts,
            "pending_evaluations": pending_evaluations,
        }
