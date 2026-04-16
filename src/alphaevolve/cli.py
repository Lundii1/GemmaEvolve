"""Command-line entrypoints."""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from alphaevolve.archive import ProgramDatabase
from alphaevolve.config import load_experiment_config
from alphaevolve.controller import EvolutionController
from alphaevolve.evaluators import build_evaluator, docker_environment_status
from alphaevolve.llm import build_inference_client, check_ollama_availability
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

    ollama_parser = subparsers.add_parser("smoke-ollama", help="Smoke-test the Ollama endpoint.")
    ollama_parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")

    docker_parser = subparsers.add_parser("smoke-docker", help="Smoke-test the Docker/gVisor runtime.")
    docker_parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    if args.command == "run":
        return asyncio.run(_run_command(args.config, args.resume, output_dir=Path(args.output_dir)))
    if args.command == "smoke-ollama":
        return asyncio.run(_smoke_ollama(Path(args.config)))
    if args.command == "smoke-docker":
        return asyncio.run(_smoke_docker(Path(args.config)))
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


async def _smoke_ollama(config_path: Path) -> int:
    config = load_experiment_config(config_path)
    available, reason = await check_ollama_availability(
        config.model.base_url,
        timeout_seconds=min(config.model.request_timeout_seconds, 2.0),
    )
    logger = logging.getLogger("alphaevolve.cli")
    if not available:
        logger.info("Skipping Ollama smoke test: %s", reason)
        return 0
    logger.info("Ollama is reachable at %s", config.model.base_url)
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
    prompt_builder = PromptBuilder(config.model.prompt_budget)
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


def _create_run_dir(output_dir: Path, *, stem: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return output_dir.expanduser().resolve() / f"{timestamp}-{stem}"
