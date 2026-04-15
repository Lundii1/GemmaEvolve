"""Command-line entrypoints."""

from __future__ import annotations

import argparse
import asyncio
import logging
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
    parser = argparse.ArgumentParser(description="AlphaEvolve async benchmark runner")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment config.")
    run_parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")

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
        return asyncio.run(_run_command(Path(args.config)))
    if args.command == "smoke-ollama":
        return asyncio.run(_smoke_ollama(Path(args.config)))
    if args.command == "smoke-docker":
        return asyncio.run(_smoke_docker(Path(args.config)))
    parser.error(f"Unsupported command: {args.command}")
    return 2


async def _run_command(config_path: Path) -> int:
    config = load_experiment_config(config_path)
    logger = logging.getLogger("alphaevolve.cli")
    archive = ProgramDatabase(config.archive)
    inference_client = build_inference_client(config.model)
    evaluator = build_evaluator(config.sandbox)
    prompt_builder = PromptBuilder(config.model.prompt_budget)
    seed_code = config.seed_program_path.read_text(encoding="utf-8")
    controller = EvolutionController(
        config=config,
        archive=archive,
        inference_client=inference_client,
        evaluator=evaluator,
        prompt_builder=prompt_builder,
        logger=logger,
    )
    result = await controller.run(Program.seed(seed_code))
    logger.info(
        "run_complete stop_reason=%s best_program=%s best_score=%.3f total_evaluated=%s",
        result.stop_reason,
        result.best_program.id,
        result.best_program.primary_score,
        result.total_evaluated,
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
    _ = load_experiment_config(config_path)
    available, reason = docker_environment_status()
    logger = logging.getLogger("alphaevolve.cli")
    if not available:
        logger.info("Skipping Docker smoke test: %s", reason)
        return 0
    logger.info("Docker and runsc appear available.")
    return 0
