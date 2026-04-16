"""Core dataclasses used across the AlphaEvolve pipeline."""

from __future__ import annotations

import base64
import pickle
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


def make_program_id(prefix: str = "prog") -> str:
    """Return a compact identifier for a program."""
    return f"{prefix}_{uuid4().hex[:10]}"


def make_prompt_log_id(prefix: str = "prompt") -> str:
    """Return a compact identifier for a prompt log."""
    return f"{prefix}_{uuid4().hex[:10]}"


def encode_random_state(state: object) -> str:
    """Serialize a random state into a JSON-safe string."""
    return base64.b64encode(pickle.dumps(state)).decode("ascii")


def decode_random_state(value: str) -> object:
    """Deserialize a random state produced by encode_random_state."""
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def _serialize_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _deserialize_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _serialize_datetime(value: datetime) -> str:
    return value.isoformat()


def _deserialize_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


@dataclass(slots=True)
class ExecutionResult:
    """Captured output and metadata from an evaluation run."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float = 0.0
    status: str = "pending"
    exception: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExecutionResult":
        if not data:
            return cls()
        return cls(
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            exit_code=int(data["exit_code"]) if data.get("exit_code") is not None else None,
            duration_ms=float(data.get("duration_ms", 0.0)),
            status=str(data.get("status", "pending")),
            exception=str(data["exception"]) if data.get("exception") is not None else None,
        )


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
    max_artifact_context: int = 3

    @property
    def usable_prompt_tokens(self) -> int:
        """Prompt tokens available for input after reserving completion space."""
        return self.max_prompt_tokens - self.reserved_completion_tokens


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Inference configuration."""

    provider: str = "ollama"
    model: str = "gemma4:26b"
    base_url: str = "http://localhost:11434"
    request_timeout_seconds: float = 120.0
    temperature: float = 0.2
    prompt_budget: PromptBudget = field(default_factory=PromptBudget)


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    """Sandbox runtime configuration."""

    backend: str = "python"
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
class FeatureAxisConfig:
    """One MAP-Elites feature dimension."""

    name: str
    source: str
    scale: Literal["identity", "log1p"] = "identity"
    bins: tuple[float, ...] = (0.0, 1.0)


@dataclass(frozen=True, slots=True)
class MigrationConfig:
    """Island migration controls."""

    enabled: bool = True
    interval_generations: int = 5
    strategy: Literal["best", "diverse"] = "best"


@dataclass(frozen=True, slots=True)
class NoveltyConfig:
    """Novelty filtering controls."""

    enabled: bool = True
    exact_dedupe: bool = True
    similarity_threshold: float = 0.999
    recent_program_window: int = 20


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    """Database retention and sampling policy."""

    hall_of_fame_size: int = 20
    recent_per_cell: int = 5
    recent_success_window: int = 12
    sample_elite_weight: float = 0.55
    sample_recent_weight: float = 0.25
    sample_hall_of_fame_weight: float = 0.20


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Program database settings."""

    islands: int = 3
    feature_axes: tuple[FeatureAxisConfig, ...] = field(
        default_factory=lambda: (
            FeatureAxisConfig(
                name="code_bytes",
                source="code_bytes",
                scale="log1p",
                bins=(0.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0),
            ),
            FeatureAxisConfig(
                name="duration_ms",
                source="duration_ms",
                scale="log1p",
                bins=(0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1000.0, 5000.0),
            ),
        )
    )
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)


@dataclass(frozen=True, slots=True)
class EvaluatorStageConfig:
    """Evaluator stage behavior settings."""

    enabled: bool = True
    max_stages: int = 3


@dataclass(frozen=True, slots=True)
class EvaluatorFeedbackConfig:
    """Evaluator feedback settings."""

    enabled: bool = False
    on_success: bool = False
    on_failure: bool = True
    borderline_score_threshold: float | None = None
    max_feedback_chars: int = 600


@dataclass(frozen=True, slots=True)
class EvaluatorArtifactsConfig:
    """Artifact capture settings."""

    enabled: bool = True
    max_inline_bytes: int = 4_096


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    """Generic evaluator module configuration."""

    module: Path
    stages: EvaluatorStageConfig = field(default_factory=EvaluatorStageConfig)
    feedback: EvaluatorFeedbackConfig = field(default_factory=EvaluatorFeedbackConfig)
    artifacts: EvaluatorArtifactsConfig = field(default_factory=EvaluatorArtifactsConfig)


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint settings."""

    enabled: bool = True
    interval_generations: int = 5
    resume_from: Path | None = None


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Controller concurrency and retry controls."""

    max_generations: int = 12
    max_retries: int = 3
    metrics_interval_seconds: float = 5.0
    stagnation_patience: int = 8
    max_inflight: int = 1


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
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(
        default_factory=lambda: EvaluatorConfig(module=Path("experiments/knapsack_evaluator.py"))
    )
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


@dataclass(frozen=True, slots=True)
class Artifact:
    """Structured artifact emitted during evaluation."""

    name: str
    type: Literal["text", "json", "file"] = "text"
    content: str | dict[str, Any] | None = None
    source_path: Path | None = None
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "content": self.content,
            "source_path": _serialize_path(self.source_path),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            name=str(data["name"]),
            type=str(data.get("type", "text")),
            content=data.get("content"),
            source_path=_deserialize_path(data.get("source_path")),
            summary=str(data["summary"]) if data.get("summary") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Persisted artifact reference."""

    name: str
    type: str
    path: Path
    summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "path": str(self.path),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRecord":
        return cls(
            name=str(data["name"]),
            type=str(data["type"]),
            path=Path(str(data["path"])),
            summary=str(data["summary"]) if data.get("summary") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class StageResult:
    """Normalized result of one evaluation stage."""

    name: str
    status: str = "success"
    metrics: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    primary_score: float | None = None
    should_continue: bool = True
    feedback: str | None = None
    rejection_reason: str | None = None
    artifacts: tuple[Artifact, ...] = ()
    execution: ExecutionResult = field(default_factory=ExecutionResult)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "metrics": dict(self.metrics),
            "features": dict(self.features),
            "primary_score": self.primary_score,
            "should_continue": self.should_continue,
            "feedback": self.feedback,
            "rejection_reason": self.rejection_reason,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "execution": self.execution.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageResult":
        return cls(
            name=str(data.get("name", "stage")),
            status=str(data.get("status", "success")),
            metrics={str(key): float(value) for key, value in dict(data.get("metrics", {})).items()},
            features={str(key): float(value) for key, value in dict(data.get("features", {})).items()},
            primary_score=float(data["primary_score"]) if data.get("primary_score") is not None else None,
            should_continue=bool(data.get("should_continue", True)),
            feedback=str(data["feedback"]) if data.get("feedback") is not None else None,
            rejection_reason=(
                str(data["rejection_reason"]) if data.get("rejection_reason") is not None else None
            ),
            artifacts=tuple(Artifact.from_dict(item) for item in data.get("artifacts", [])),
            execution=ExecutionResult.from_dict(data.get("execution")),
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Structured evaluator output used across the pipeline."""

    status: str = "success"
    primary_score: float = float("-inf")
    metrics: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    execution: ExecutionResult = field(default_factory=ExecutionResult)
    stage_results: tuple[StageResult, ...] = ()
    artifacts: tuple[ArtifactRecord, ...] = ()
    feedback: str | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "primary_score": self.primary_score,
            "metrics": dict(self.metrics),
            "features": dict(self.features),
            "execution": self.execution.to_dict(),
            "stage_results": [stage.to_dict() for stage in self.stage_results],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "feedback": self.feedback,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvaluationResult":
        if not data:
            return cls()
        return cls(
            status=str(data.get("status", "success")),
            primary_score=float(data.get("primary_score", float("-inf"))),
            metrics={str(key): float(value) for key, value in dict(data.get("metrics", {})).items()},
            features={str(key): float(value) for key, value in dict(data.get("features", {})).items()},
            execution=ExecutionResult.from_dict(data.get("execution")),
            stage_results=tuple(StageResult.from_dict(item) for item in data.get("stage_results", [])),
            artifacts=tuple(ArtifactRecord.from_dict(item) for item in data.get("artifacts", [])),
            feedback=str(data["feedback"]) if data.get("feedback") is not None else None,
            rejection_reason=(
                str(data["rejection_reason"]) if data.get("rejection_reason") is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class PromptArtifactContext:
    """Prompt-safe summary of an artifact."""

    name: str
    type: str
    summary: str


@dataclass(frozen=True, slots=True)
class PromptLog:
    """Persisted prompt/model/evaluator trace."""

    id: str
    parent_id: str
    island_id: int
    prompt_text: str
    estimated_tokens: int
    included_program_ids: tuple[str, ...] = ()
    summarized_program_ids: tuple[str, ...] = ()
    artifact_context: tuple[PromptArtifactContext, ...] = ()
    evaluator_feedback_used: str | None = None
    model_response: str | None = None
    child_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "island_id": self.island_id,
            "prompt_text": self.prompt_text,
            "estimated_tokens": self.estimated_tokens,
            "included_program_ids": list(self.included_program_ids),
            "summarized_program_ids": list(self.summarized_program_ids),
            "artifact_context": [asdict(item) for item in self.artifact_context],
            "evaluator_feedback_used": self.evaluator_feedback_used,
            "model_response": self.model_response,
            "child_id": self.child_id,
            "created_at": _serialize_datetime(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptLog":
        return cls(
            id=str(data["id"]),
            parent_id=str(data["parent_id"]),
            island_id=int(data["island_id"]),
            prompt_text=str(data["prompt_text"]),
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            included_program_ids=tuple(str(item) for item in data.get("included_program_ids", [])),
            summarized_program_ids=tuple(str(item) for item in data.get("summarized_program_ids", [])),
            artifact_context=tuple(
                PromptArtifactContext(
                    name=str(item["name"]),
                    type=str(item["type"]),
                    summary=str(item["summary"]),
                )
                for item in data.get("artifact_context", [])
            ),
            evaluator_feedback_used=(
                str(data["evaluator_feedback_used"])
                if data.get("evaluator_feedback_used") is not None
                else None
            ),
            model_response=str(data["model_response"]) if data.get("model_response") is not None else None,
            child_id=str(data["child_id"]) if data.get("child_id") is not None else None,
            created_at=_deserialize_datetime(str(data["created_at"])),
        )


@dataclass(slots=True)
class Program:
    """A concrete candidate program tracked by the database."""

    id: str
    code: str
    metrics: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    primary_score: float = float("-inf")
    execution: ExecutionResult | None = None
    evaluation: EvaluationResult | None = None
    parent_id: str | None = None
    generation: int = 0
    island_id: int = 0
    archive_cell: tuple[int, ...] | None = None
    prompt_log_id: str | None = None
    artifact_records: tuple[ArtifactRecord, ...] = ()
    lineage_depth: int = 0
    accepted: bool = True
    rejection_reason: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def code_hash(self) -> str:
        import hashlib

        return hashlib.sha256(self.code.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "metrics": dict(self.metrics),
            "features": dict(self.features),
            "primary_score": self.primary_score,
            "execution": self.execution.to_dict() if self.execution is not None else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation is not None else None,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "island_id": self.island_id,
            "archive_cell": list(self.archive_cell) if self.archive_cell is not None else None,
            "prompt_log_id": self.prompt_log_id,
            "artifact_records": [record.to_dict() for record in self.artifact_records],
            "lineage_depth": self.lineage_depth,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "created_at": _serialize_datetime(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Program":
        return cls(
            id=str(data["id"]),
            code=str(data["code"]),
            metrics={str(key): float(value) for key, value in dict(data.get("metrics", {})).items()},
            features={str(key): float(value) for key, value in dict(data.get("features", {})).items()},
            primary_score=float(data.get("primary_score", float("-inf"))),
            execution=ExecutionResult.from_dict(data.get("execution")) if data.get("execution") else None,
            evaluation=EvaluationResult.from_dict(data.get("evaluation")) if data.get("evaluation") else None,
            parent_id=str(data["parent_id"]) if data.get("parent_id") is not None else None,
            generation=int(data.get("generation", 0)),
            island_id=int(data.get("island_id", 0)),
            archive_cell=tuple(int(item) for item in data.get("archive_cell", []))
            if data.get("archive_cell") is not None
            else None,
            prompt_log_id=str(data["prompt_log_id"]) if data.get("prompt_log_id") is not None else None,
            artifact_records=tuple(
                ArtifactRecord.from_dict(item) for item in data.get("artifact_records", [])
            ),
            lineage_depth=int(data.get("lineage_depth", 0)),
            accepted=bool(data.get("accepted", True)),
            rejection_reason=str(data["rejection_reason"]) if data.get("rejection_reason") is not None else None,
            created_at=_deserialize_datetime(str(data["created_at"])),
        )

    @classmethod
    def seed(cls, code: str) -> "Program":
        """Create a seed program with a stable seed prefix."""
        return cls(id=make_program_id("seed"), code=code)


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Controller checkpoint payload."""

    generation: int
    best_program_id: str | None
    next_island_index: int
    stagnation_generations: int
    stop_reason: str | None
    random_state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "best_program_id": self.best_program_id,
            "next_island_index": self.next_island_index,
            "stagnation_generations": self.stagnation_generations,
            "stop_reason": self.stop_reason,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointState":
        return cls(
            generation=int(data.get("generation", 0)),
            best_program_id=str(data["best_program_id"]) if data.get("best_program_id") is not None else None,
            next_island_index=int(data.get("next_island_index", 0)),
            stagnation_generations=int(data.get("stagnation_generations", 0)),
            stop_reason=str(data["stop_reason"]) if data.get("stop_reason") is not None else None,
            random_state=str(data["random_state"]),
        )
