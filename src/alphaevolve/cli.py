"""Command-line entrypoints."""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from alphaevolve.archive import ProgramDatabase
from alphaevolve.errors import ConfigError
from alphaevolve.config import load_experiment_config
from alphaevolve.controller import EvolutionController
from alphaevolve.evaluators import build_evaluator, docker_environment_status
from alphaevolve.llm import build_inference_client, check_inference_availability
from alphaevolve.logging_utils import configure_logging
from alphaevolve.models import Program
from alphaevolve.prompts import PromptBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaEvolve runner")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment config.")
    run_group = run_parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--config", help="Path to an experiment TOML file.")
    run_group.add_argument("--resume", help="Path to a previous run directory.")
    run_parser.add_argument(
        "--output-dir",
        default="runs",
        help="Directory that stores fresh run outputs.",
    )

    model_parser = subparsers.add_parser(
        "smoke-model",
        help="Smoke-test the configured model endpoint.",
    )
    model_parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")

    ollama_parser = subparsers.add_parser(
        "smoke-ollama",
        help="Backward-compatible alias for smoke-model.",
    )
    ollama_parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")

    docker_parser = subparsers.add_parser("smoke-docker", help="Smoke-test the Docker/gVisor runtime.")
    docker_parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")

    clone_parser = subparsers.add_parser(
        "clone-best",
        help="Clone an experiment directory and replace the cloned seed with the run's best program.",
    )
    clone_parser.add_argument("--config", required=True, help="Path to the original experiment TOML file.")
    clone_parser.add_argument("--run", required=True, help="Path to a previous run directory.")
    clone_parser.add_argument(
        "--output",
        required=True,
        help="Destination directory for the cloned experiment bundle.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.command == "run":
        return asyncio.run(_run_command(args.config, args.resume, output_dir=Path(args.output_dir)))
    if args.command in {"smoke-model", "smoke-ollama"}:
        return asyncio.run(_smoke_model(Path(args.config)))
    if args.command == "smoke-docker":
        return asyncio.run(_smoke_docker(Path(args.config)))
    if args.command == "clone-best":
        return _clone_best_command(
            config_path=Path(args.config),
            run_dir=Path(args.run),
            output_dir=Path(args.output),
        )
    parser.error(f"Unsupported command: {args.command}")
    return 2


async def _run_command(config_arg: str | None, resume_arg: str | None, *, output_dir: Path) -> int:
    logger = logging.getLogger("alphaevolve.cli")
    if resume_arg is not None:
        run_dir = Path(resume_arg).expanduser().resolve()
        config_path = run_dir / "config.toml"
        if not config_path.exists():
            logger.error("Resume directory does not contain config.toml: %s", run_dir)
            return 2
    else:
        config_path = Path(config_arg).expanduser().resolve()
        run_dir = _create_run_dir(output_dir, stem=config_path.stem)
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, run_dir / "config.toml")

    config = load_experiment_config(config_path)
    result = await _run_controller(config, run_dir=run_dir, logger=logger)
    logger.info(
        "run_complete stop_reason=%s best_program=%s best_score=%.3f total_evaluated=%s run_dir=%s",
        result.stop_reason,
        result.best_program.id,
        result.best_program.primary_score,
        result.total_evaluated,
        result.run_dir,
    )
    return 0 if result.best_program.primary_score >= config.target_score else 1


async def _smoke_model(config_path: Path) -> int:
    config = load_experiment_config(config_path)
    available, reason = await check_inference_availability(
        config.model,
        timeout_seconds=min(config.model.request_timeout_seconds, 2.0),
    )
    logger = logging.getLogger("alphaevolve.cli")
    if not available:
        logger.info(
            "Skipping model smoke test for provider=%s: %s",
            config.model.provider,
            reason,
        )
        return 0
    logger.info(
        "Model provider %s is reachable at %s",
        config.model.provider,
        config.model.base_url,
    )
    return 0


async def _smoke_docker(config_path: Path) -> int:
    config = load_experiment_config(config_path)
    available, reason = docker_environment_status(config.sandbox.runtime)
    logger = logging.getLogger("alphaevolve.cli")
    if not available:
        logger.info("Skipping Docker smoke test: %s", reason)
        return 0
    logger.info("Docker and runsc appear available.")
    return 0


async def _run_controller(config, *, run_dir: Path, logger: logging.Logger):
    database = ProgramDatabase(config.database, run_dir / "database")
    inference_client = build_inference_client(config.model)
    evaluator = build_evaluator(config.evaluator, config.sandbox, feedback_client=inference_client)
    prompt_builder = PromptBuilder(
        config.model.prompt_budget,
        mutation_scope=config.mutation_scope,
    )
    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    controller = EvolutionController(
        config=config,
        database=database,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=prompt_builder,
        run_dir=run_dir,
        logger=logger,
    )
    return await controller.run(Program.seed(seed_code))


def _clone_best_command(*, config_path: Path, run_dir: Path, output_dir: Path) -> int:
    logger = logging.getLogger("alphaevolve.cli")
    resolved_config_path = config_path.expanduser().resolve()
    try:
        config = load_experiment_config(resolved_config_path)
    except ConfigError as exc:
        logger.error("Failed to load experiment config %s: %s", resolved_config_path, exc)
        return 2

    resolved_run_dir = run_dir.expanduser().resolve()
    database_dir = resolved_run_dir / "database"
    if not database_dir.exists():
        logger.error("Run directory does not contain a database: %s", resolved_run_dir)
        return 2

    database = ProgramDatabase(config.database, database_dir)
    best_program = asyncio.run(database.best_program())
    if best_program is None:
        logger.error("Run directory does not contain a best program: %s", resolved_run_dir)
        return 2

    source_dir = resolved_config_path.parent
    resolved_output_dir = output_dir.expanduser().resolve()
    if resolved_output_dir == source_dir or source_dir in resolved_output_dir.parents:
        logger.error("Output directory must not be inside the source experiment directory: %s", resolved_output_dir)
        return 2
    if resolved_output_dir.exists():
        shutil.rmtree(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = (
        resolved_config_path,
        config.seed_program_path,
        config.evaluator.module,
    )
    for source_file in files_to_copy:
        relative_path = source_file.relative_to(source_dir)
        destination = resolved_output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)

    cloned_config_path = resolved_output_dir / resolved_config_path.name
    cloned_seed_path = resolved_output_dir / config.seed_program_path.relative_to(source_dir)
    cloned_seed_path.parent.mkdir(parents=True, exist_ok=True)
    cloned_seed_path.write_text(best_program.code, encoding="utf-8")

    logger.info(
        "cloned_best_program run_dir=%s best_program=%s best_score=%.3f clone_dir=%s config_path=%s seed_path=%s",
        resolved_run_dir,
        best_program.id,
        best_program.primary_score,
        resolved_output_dir,
        cloned_config_path,
        cloned_seed_path,
    )
    return 0


def _create_run_dir(output_dir: Path, *, stem: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return output_dir.expanduser().resolve() / f"{timestamp}-{stem}"


if __name__ == "__main__":
    raise SystemExit(main())
