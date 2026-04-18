"""Configuration loading and validation."""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

from alphaevolve.errors import ConfigError
from alphaevolve.models import (
    CheckpointConfig,
    ControllerConfig,
    DatabaseConfig,
    EvaluatorArtifactsConfig,
    EvaluatorConfig,
    EvaluatorFeedbackConfig,
    EvaluatorStageConfig,
    ExperimentConfig,
    FeatureAxisConfig,
    MigrationConfig,
    ModelConfig,
    NoveltyConfig,
    PromptBudget,
    RetentionConfig,
    SandboxConfig,
)

logger = logging.getLogger("alphaevolve.config")


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Expected table [{name}] in config.")
    return value


def _as_float_tuple(raw: Any, field_name: str) -> tuple[float, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"Expected non-empty numeric list for {field_name}.")
    try:
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid numeric value in {field_name}.") from exc
    if values != tuple(sorted(values)):
        raise ConfigError(f"{field_name} must be sorted in ascending order.")
    return values


def _resolve_path(base_dir: Path, raw: Any, field_name: str) -> Path:
    if raw is None:
        raise ConfigError(f"Missing required path for {field_name}.")
    path = (base_dir / str(raw)).resolve()
    return path


def _as_command_tuple(raw: Any, field_name: str, *, allow_none: bool = False) -> tuple[str, ...] | None:
    if raw is None:
        if allow_none:
            return None
        raise ConfigError(f"Missing required command list for {field_name}.")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{field_name} must be a non-empty list of strings.")
    if not all(isinstance(item, str) and item for item in raw):
        raise ConfigError(f"{field_name} must be a non-empty list of strings.")
    return tuple(raw)


def _reject_fake(section_name: str, field_name: str, value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "fake":
        raise ConfigError(f"{section_name}.{field_name}='fake' is no longer supported.")
    return value


def _load_feature_axes(raw_axes: Any) -> tuple[FeatureAxisConfig, ...]:
    if raw_axes is None:
        return DatabaseConfig().feature_axes
    if not isinstance(raw_axes, list) or not raw_axes:
        raise ConfigError("database.feature_axes must be a non-empty list of tables.")
    axes: list[FeatureAxisConfig] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw_axes, start=1):
        if not isinstance(item, dict):
            raise ConfigError(f"database.feature_axes[{index}] must be a table.")
        try:
            name = str(item["name"])
            source = str(item["source"])
        except KeyError as exc:
            raise ConfigError(
                f"database.feature_axes[{index}] is missing required key {exc.args[0]!r}."
            ) from exc
        if name in seen_names:
            raise ConfigError(f"database.feature_axes uses duplicate axis name {name!r}.")
        scale = str(item.get("scale", "identity"))
        if scale not in {"identity", "log1p"}:
            raise ConfigError(
                f"database.feature_axes[{index}].scale must be 'identity' or 'log1p'."
            )
        bins = _as_float_tuple(item.get("bins"), f"database.feature_axes[{index}].bins")
        axes.append(FeatureAxisConfig(name=name, source=source, scale=scale, bins=bins))
        seen_names.add(name)
    return tuple(axes)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment TOML file into strongly typed dataclasses."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    try:
        name = str(raw["name"])
        description = str(raw["description"])
        seed_program_path = (config_path.parent / str(raw["seed_program"])).resolve()
        system_instructions = str(raw["system_instructions"])
        task_description = str(raw["task_description"])
        evaluation_contract = str(raw["evaluation_contract"])
    except KeyError as exc:
        raise ConfigError(f"Missing required top-level config key: {exc.args[0]}") from exc

    if not seed_program_path.exists():
        raise ConfigError(f"Seed program does not exist: {seed_program_path}")
    mutation_scope = str(raw.get("mutation_scope", "evolve_block"))
    if mutation_scope not in {"evolve_block", "full_file"}:
        raise ConfigError("mutation_scope must be 'evolve_block' or 'full_file'.")

    model_section = _section(raw, "model")
    sandbox_section = _section(raw, "sandbox")
    database_section = raw.get("database")
    archive_section = raw.get("archive")
    if database_section is not None and archive_section is not None:
        raise ConfigError("Use either [database] or deprecated [archive], not both.")
    if database_section is None and archive_section is not None:
        if not isinstance(archive_section, dict):
            raise ConfigError("Expected table [archive] in config.")
        database_section = archive_section
    if database_section is None:
        database_section = {}
    if not isinstance(database_section, dict):
        raise ConfigError("Expected table [database] in config.")
    evaluator_section = _section(raw, "evaluator")
    checkpoint_section = _section(raw, "checkpoint")
    controller_section = _section(raw, "controller")

    prompt_budget = PromptBudget(
        max_prompt_tokens=int(model_section.get("max_prompt_tokens", 12_000)),
        reserved_completion_tokens=int(model_section.get("reserved_completion_tokens", 2_048)),
        max_history_programs=int(model_section.get("max_history_programs", 4)),
        max_artifact_context=int(model_section.get("max_artifact_context", 3)),
    )
    provider = _reject_fake("model", "provider", str(model_section.get("provider", "ollama")))
    model = ModelConfig(
        provider=provider,
        model=str(model_section.get("model", "gemma4:26b")),
        base_url=str(model_section.get("base_url", "http://localhost:11434")),
        api_key_env=(
            str(model_section["api_key_env"])
            if model_section.get("api_key_env") is not None
            else None
        ),
        request_timeout_seconds=float(model_section.get("request_timeout_seconds", 120.0)),
        temperature=float(model_section.get("temperature", 0.2)),
        prompt_budget=prompt_budget,
    )

    backend = _reject_fake("sandbox", "backend", str(sandbox_section.get("backend", "python")))
    sandbox = SandboxConfig(
        backend=backend,
        image=str(sandbox_section.get("image", "python:3.12-slim")),
        runtime=str(sandbox_section.get("runtime", "runsc")),
        network_mode=str(sandbox_section.get("network_mode", "none")),
        read_only_root=bool(sandbox_section.get("read_only_root", True)),
        tmpfs_path=str(sandbox_section.get("tmpfs_path", "/tmp/eval")),
        tmpfs_size_bytes=int(sandbox_section.get("tmpfs_size_bytes", 5 * 1024 * 1024)),
        mem_limit=str(sandbox_section.get("mem_limit", "256m")),
        cpu_quota=int(sandbox_section.get("cpu_quota", 50_000)),
        timeout_seconds=float(sandbox_section.get("timeout_seconds", 10.0)),
        working_dir=str(sandbox_section.get("working_dir", "/workspace")),
        program_filename=str(sandbox_section.get("program_filename", "program.py")),
        build_command=_as_command_tuple(
            sandbox_section.get("build_command"),
            "sandbox.build_command",
            allow_none=True,
        ),
        run_command=_as_command_tuple(
            sandbox_section.get("run_command", ["python", "{program}"]),
            "sandbox.run_command",
        ),
    )
    if not sandbox.program_filename.strip():
        raise ConfigError("sandbox.program_filename must not be empty.")

    migration_section = _section(database_section, "migration")
    novelty_section = _section(database_section, "novelty")
    retention_section = _section(database_section, "retention")
    database = DatabaseConfig(
        islands=int(database_section.get("islands", 3)),
        feature_axes=_load_feature_axes(database_section.get("feature_axes")),
        migration=MigrationConfig(
            enabled=bool(migration_section.get("enabled", True)),
            interval_generations=int(migration_section.get("interval_generations", 5)),
            strategy=str(migration_section.get("strategy", "best")),
        ),
        novelty=NoveltyConfig(
            enabled=bool(novelty_section.get("enabled", True)),
            exact_dedupe=bool(novelty_section.get("exact_dedupe", True)),
            similarity_threshold=float(novelty_section.get("similarity_threshold", 0.999)),
            recent_program_window=int(novelty_section.get("recent_program_window", 20)),
        ),
        retention=RetentionConfig(
            hall_of_fame_size=int(retention_section.get("hall_of_fame_size", 20)),
            recent_per_cell=int(retention_section.get("recent_per_cell", 5)),
            recent_success_window=int(retention_section.get("recent_success_window", 12)),
            sample_elite_weight=float(retention_section.get("sample_elite_weight", 0.55)),
            sample_recent_weight=float(retention_section.get("sample_recent_weight", 0.25)),
            sample_hall_of_fame_weight=float(
                retention_section.get("sample_hall_of_fame_weight", 0.20)
            ),
            parent_share_window=int(retention_section.get("parent_share_window", 64)),
            parent_share_cap=float(retention_section.get("parent_share_cap", 0.35)),
            best_parent_cooldown_samples=int(
                retention_section.get("best_parent_cooldown_samples", 8)
            ),
            mutation_failure_streak_threshold=int(
                retention_section.get("mutation_failure_streak_threshold", 3)
            ),
            mutation_failure_penalty_samples=int(
                retention_section.get("mutation_failure_penalty_samples", 12)
            ),
            mutation_failure_weight_multiplier=float(
                retention_section.get("mutation_failure_weight_multiplier", 0.20)
            ),
        ),
    )

    stages_section = _section(evaluator_section, "stages")
    feedback_section = _section(evaluator_section, "feedback")
    artifacts_section = _section(evaluator_section, "artifacts")
    evaluator_module = _resolve_path(
        config_path.parent,
        evaluator_section.get("module"),
        "evaluator.module",
    )
    if not evaluator_module.exists():
        raise ConfigError(f"Evaluator module does not exist: {evaluator_module}")
    evaluator = EvaluatorConfig(
        module=evaluator_module,
        stages=EvaluatorStageConfig(
            enabled=bool(stages_section.get("enabled", True)),
            max_stages=int(stages_section.get("max_stages", 3)),
        ),
        feedback=EvaluatorFeedbackConfig(
            enabled=bool(feedback_section.get("enabled", False)),
            on_success=bool(feedback_section.get("on_success", False)),
            on_failure=bool(feedback_section.get("on_failure", True)),
            borderline_score_threshold=float(feedback_section["borderline_score_threshold"])
            if feedback_section.get("borderline_score_threshold") is not None
            else None,
            max_feedback_chars=int(feedback_section.get("max_feedback_chars", 600)),
        ),
        artifacts=EvaluatorArtifactsConfig(
            enabled=bool(artifacts_section.get("enabled", True)),
            max_inline_bytes=int(artifacts_section.get("max_inline_bytes", 4_096)),
        ),
    )

    resume_from = checkpoint_section.get("resume_from")
    checkpoint = CheckpointConfig(
        enabled=bool(checkpoint_section.get("enabled", True)),
        interval_generations=int(checkpoint_section.get("interval_generations", 5)),
        resume_from=_resolve_path(config_path.parent, resume_from, "checkpoint.resume_from")
        if resume_from is not None
        else None,
    )

    controller = ControllerConfig(
        max_generations=int(controller_section.get("max_generations", 12)),
        max_retries=int(controller_section.get("max_retries", 3)),
        metrics_interval_seconds=float(controller_section.get("metrics_interval_seconds", 5.0)),
        stagnation_patience=int(controller_section.get("stagnation_patience", 8)),
        max_inflight=int(controller_section.get("max_inflight", 1)),
    )
    if controller.max_inflight > 32:
        logger.warning(
            "controller.max_inflight=%s is unusually high; Docker and the model endpoint may bottleneck above 32 concurrent jobs.",
            controller.max_inflight,
        )

    return ExperimentConfig(
        name=name,
        description=description,
        seed_program_path=seed_program_path,
        system_instructions=system_instructions,
        task_description=task_description,
        evaluation_contract=evaluation_contract,
        primary_metric=str(raw.get("primary_metric", "score")),
        target_score=float(raw.get("target_score", 275.0)),
        mutation_scope=mutation_scope,
        model=model,
        sandbox=sandbox,
        database=database,
        evaluator=evaluator,
        checkpoint=checkpoint,
        controller=controller,
    )
