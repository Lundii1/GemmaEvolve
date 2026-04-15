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
    diff_parse_failures: int = 0
    diff_apply_failures: int = 0
    retry_count: int = 0
    evaluation_count: int = 0
    sandbox_failures: int = 0
    best_score: float = float("-inf")
    latest_llm_latency_ms: float | None = None
    latest_evaluation_latency_ms: float | None = None
    sample_parse_failure_preview: str | None = None
    sample_apply_failure_preview: str | None = None

    def record_diff_parse_failure(self, preview: str | None = None) -> None:
        self.diff_parse_failures += 1
        if preview and self.sample_parse_failure_preview is None:
            self.sample_parse_failure_preview = preview

    def record_diff_apply_failure(self, preview: str | None = None) -> None:
        self.diff_apply_failures += 1
        if preview and self.sample_apply_failure_preview is None:
            self.sample_apply_failure_preview = preview

    def record_llm_latency(self, duration_ms: float) -> None:
        self.latest_llm_latency_ms = duration_ms

    def record_evaluation_latency(self, duration_ms: float) -> None:
        self.latest_evaluation_latency_ms = duration_ms

    def snapshot(
        self,
        *,
        pending_prompts: int,
        pending_evaluations: int,
    ) -> dict[str, int | float | None]:
        elapsed = max(perf_counter() - self.started_at, 1e-6)
        return {
            "elapsed_seconds": round(elapsed, 3),
            "mutation_jobs_started": self.mutation_jobs_started,
            "diff_parse_failures": self.diff_parse_failures,
            "diff_apply_failures": self.diff_apply_failures,
            "retry_count": self.retry_count,
            "evaluation_count": self.evaluation_count,
            "sandbox_failures": self.sandbox_failures,
            "best_score": round(self.best_score, 3) if self.best_score != float("-inf") else self.best_score,
            "latest_llm_latency_ms": (
                round(self.latest_llm_latency_ms, 3)
                if self.latest_llm_latency_ms is not None
                else None
            ),
            "latest_evaluation_latency_ms": (
                round(self.latest_evaluation_latency_ms, 3)
                if self.latest_evaluation_latency_ms is not None
                else None
            ),
            "generation_throughput": round(self.mutation_jobs_started / elapsed, 3),
            "evaluation_throughput": round(self.evaluation_count / elapsed, 3),
            "pending_prompts": pending_prompts,
            "pending_evaluations": pending_evaluations,
        }

    def summary(self) -> dict[str, int | float | str | None]:
        return {
            "mutation_jobs_started": self.mutation_jobs_started,
            "diff_parse_failures": self.diff_parse_failures,
            "diff_apply_failures": self.diff_apply_failures,
            "retry_count": self.retry_count,
            "evaluation_count": self.evaluation_count,
            "sandbox_failures": self.sandbox_failures,
            "best_score": round(self.best_score, 3) if self.best_score != float("-inf") else self.best_score,
            "latest_llm_latency_ms": (
                round(self.latest_llm_latency_ms, 3)
                if self.latest_llm_latency_ms is not None
                else None
            ),
            "latest_evaluation_latency_ms": (
                round(self.latest_evaluation_latency_ms, 3)
                if self.latest_evaluation_latency_ms is not None
                else None
            ),
            "sample_parse_failure_preview": self.sample_parse_failure_preview,
            "sample_apply_failure_preview": self.sample_apply_failure_preview,
        }
