"""Configuration loading and validation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from alphaevolve.errors import ConfigError
from alphaevolve.models import (
    ArchiveConfig,
    ControllerConfig,
    ExperimentConfig,
    ModelConfig,
    PromptBudget,
    SandboxConfig,
)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Expected table [{name}] in config.")
    return value


def _as_int_tuple(raw: Any, field_name: str) -> tuple[int, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"Expected non-empty integer list for {field_name}.")
    try:
        values = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid integer value in {field_name}.") from exc
    if values != tuple(sorted(values)):
        raise ConfigError(f"{field_name} must be sorted in ascending order.")
    return values


def _as_str_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError("scripted_responses must be a list of strings.")
    return tuple(str(item) for item in raw)


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

    model_section = _section(raw, "model")
    sandbox_section = _section(raw, "sandbox")
    archive_section = _section(raw, "archive")
    controller_section = _section(raw, "controller")

    prompt_budget = PromptBudget(
        max_prompt_tokens=int(model_section.get("max_prompt_tokens", 12_000)),
        reserved_completion_tokens=int(model_section.get("reserved_completion_tokens", 2_048)),
        max_history_programs=int(model_section.get("max_history_programs", 4)),
    )
    model = ModelConfig(
        provider=str(model_section.get("provider", "fake")),
        model=str(model_section.get("model", "gemma4:26b")),
        base_url=str(model_section.get("base_url", "http://localhost:11434")),
        request_timeout_seconds=float(model_section.get("request_timeout_seconds", 120.0)),
        temperature=float(model_section.get("temperature", 0.2)),
        prompt_budget=prompt_budget,
        scripted_responses=_as_str_tuple(model_section.get("scripted_responses")),
    )
    sandbox = SandboxConfig(
        backend=str(sandbox_section.get("backend", "fake")),
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
    )
    archive = ArchiveConfig(
        code_length_buckets=_as_int_tuple(
            archive_section.get("code_length_buckets", [0, 512, 1_024, 2_048, 4_096, 8_192, 16_384]),
            "archive.code_length_buckets",
        ),
        eval_time_buckets_ms=_as_int_tuple(
            archive_section.get("eval_time_buckets_ms", [0, 10, 50, 100, 250, 500, 1_000, 5_000, 10_000]),
            "archive.eval_time_buckets_ms",
        ),
        hall_of_fame_size=int(archive_section.get("hall_of_fame_size", 20)),
        recent_per_cell=int(archive_section.get("recent_per_cell", 5)),
    )
    controller = ControllerConfig(
        llm_concurrency=int(controller_section.get("llm_concurrency", 2)),
        evaluation_concurrency=int(controller_section.get("evaluation_concurrency", 2)),
        max_generations=int(controller_section.get("max_generations", 12)),
        max_pending_prompts=int(controller_section.get("max_pending_prompts", 4)),
        max_pending_evaluations=int(controller_section.get("max_pending_evaluations", 4)),
        max_retries=int(controller_section.get("max_retries", 3)),
        metrics_interval_seconds=float(controller_section.get("metrics_interval_seconds", 5.0)),
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
        model=model,
        sandbox=sandbox,
        archive=archive,
        controller=controller,
    )
