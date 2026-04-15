"""Core dataclasses used across the AlphaEvolve pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def make_program_id(prefix: str = "prog") -> str:
    """Return a compact identifier for a program."""
    return f"{prefix}_{uuid4().hex[:10]}"


@dataclass(slots=True)
class ExecutionResult:
    """Captured output and metadata from an evaluation run."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float = 0.0
    status: str = "pending"
    exception: str | None = None


@dataclass(frozen=True, slots=True)
class DiffBlock:
    """One SEARCH/REPLACE block emitted by a model."""

    search: str
    replace: str


@dataclass(frozen=True, slots=True)
class Diff:
    """Parsed diff response containing one or more SEARCH/REPLACE blocks."""

    raw_text: str
    blocks: tuple[DiffBlock, ...]


@dataclass(frozen=True, slots=True)
class PromptBudget:
    """Boundaries for prompt construction."""

    max_prompt_tokens: int = 12_000
    reserved_completion_tokens: int = 2_048
    max_history_programs: int = 4

    @property
    def usable_prompt_tokens(self) -> int:
        """Prompt tokens available for input after reserving completion space."""
        return self.max_prompt_tokens - self.reserved_completion_tokens


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Inference configuration."""

    provider: str = "fake"
    model: str = "gemma4:26b"
    base_url: str = "http://localhost:11434"
    request_timeout_seconds: float = 120.0
    temperature: float = 0.2
    prompt_budget: PromptBudget = field(default_factory=PromptBudget)
    scripted_responses: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Sandbox runtime configuration."""

    backend: str = "fake"
    image: str = "python:3.12-slim"
    runtime: str = "runsc"
    network_mode: str = "none"
    read_only_root: bool = True
    tmpfs_path: str = "/tmp/eval"
    tmpfs_size_bytes: int = 5 * 1024 * 1024
    mem_limit: str = "256m"
    cpu_quota: int = 50_000
    timeout_seconds: float = 10.0
    working_dir: str = "/workspace"


@dataclass(frozen=True, slots=True)
class ArchiveConfig:
    """Archive binning and retention policy."""

    code_length_buckets: tuple[int, ...] = (0, 512, 1_024, 2_048, 4_096, 8_192, 16_384)
    eval_time_buckets_ms: tuple[int, ...] = (0, 10, 50, 100, 250, 500, 1_000, 5_000, 10_000)
    hall_of_fame_size: int = 20
    recent_per_cell: int = 5


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Controller concurrency and retry controls."""

    llm_concurrency: int = 2
    evaluation_concurrency: int = 2
    max_generations: int = 12
    max_pending_prompts: int = 4
    max_pending_evaluations: int = 4
    max_retries: int = 3
    metrics_interval_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level experiment configuration loaded from TOML."""

    name: str
    description: str
    seed_program_path: Path
    system_instructions: str
    task_description: str
    evaluation_contract: str
    primary_metric: str
    target_score: float
    model: ModelConfig = field(default_factory=ModelConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


@dataclass(slots=True)
class Program:
    """A concrete candidate program tracked by the archive."""

    id: str
    code: str
    metrics: dict[str, float] = field(default_factory=dict)
    primary_score: float = float("-inf")
    execution: ExecutionResult | None = None
    parent_id: str | None = None
    archive_cell: tuple[int, int] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def seed(cls, code: str) -> "Program":
        """Create a seed program with a stable seed prefix."""
        return cls(id=make_program_id("seed"), code=code)
