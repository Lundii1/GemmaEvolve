"""Command-line entrypoints."""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter

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

    probe_ollama_parser = subparsers.add_parser(
        "probe-ollama",
        help="Run the real Ollama generator with the fake evaluator for diff-format diagnostics.",
    )
    probe_ollama_parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")
    probe_ollama_parser.add_argument("--count", type=int, default=5, help="Number of generation jobs to attempt.")

    probe_docker_parser = subparsers.add_parser(
        "probe-docker",
        help="Benchmark the Docker/gVisor sandbox with success, network-blocked, and timeout fixtures.",
    )
    probe_docker_parser.add_argument("--config", required=True, help="Path to an experiment TOML file.")
    probe_docker_parser.add_argument("--repeat", type=int, default=3, help="Number of valid cold-start runs.")

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
    if args.command == "probe-ollama":
        return asyncio.run(_probe_ollama(Path(args.config), count=args.count))
    if args.command == "probe-docker":
        return asyncio.run(_probe_docker(Path(args.config), repeat=args.repeat))
    parser.error(f"Unsupported command: {args.command}")
    return 2


async def _run_command(config_path: Path) -> int:
    config = load_experiment_config(config_path)
    logger = logging.getLogger("alphaevolve.cli")
    result = await _run_controller(config, logger=logger)
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
    config = load_experiment_config(config_path)
    available, reason = docker_environment_status(config.sandbox.runtime)
    logger = logging.getLogger("alphaevolve.cli")
    if not available:
        logger.info("Skipping Docker smoke test: %s", reason)
        return 0
    logger.info("Docker and runsc appear available.")
    return 0


async def _probe_ollama(config_path: Path, *, count: int) -> int:
    config = load_experiment_config(config_path)
    logger = logging.getLogger("alphaevolve.cli")
    if config.model.provider.lower() != "ollama":
        logger.error("probe-ollama requires model.provider='ollama'.")
        return 2

    probe_config = replace(
        config,
        target_score=float("inf"),
        sandbox=replace(config.sandbox, backend="fake"),
        controller=replace(config.controller, max_generations=count),
    )
    result = await _run_controller(probe_config, logger=logger)
    logger.info(
        "probe_ollama_complete generation_jobs=%s evaluated=%s best_score=%.3f parse_failures=%s apply_failures=%s retries=%s",
        result.total_generation_jobs,
        result.total_evaluated,
        result.best_program.primary_score,
        result.diff_parse_failures,
        result.diff_apply_failures,
        result.retry_count,
    )
    logger.info(
        "probe_ollama_samples parse_preview=%s apply_preview=%s",
        result.sample_parse_failure_preview or "<none>",
        result.sample_apply_failure_preview or "<none>",
    )
    return 0


async def _probe_docker(config_path: Path, *, repeat: int) -> int:
    config = load_experiment_config(config_path)
    logger = logging.getLogger("alphaevolve.cli")
    if config.sandbox.backend.lower() != "docker":
        logger.error("probe-docker requires sandbox.backend='docker'.")
        return 2

    available, reason = docker_environment_status(config.sandbox.runtime)
    if not available:
        logger.error("Docker probe cannot run: %s", reason)
        return 1

    evaluator = build_evaluator(config.sandbox)
    valid_program = Program.seed(
        "import json\n"
        "print(json.dumps({'score': 1.0}))\n"
    )
    network_program = Program.seed(
        "import socket\n"
        "socket.create_connection(('1.1.1.1', 80), timeout=1)\n"
    )
    timeout_program = Program.seed("while True:\n    pass\n")
    valid_wall_clock_ms: list[float] = []

    try:
        for run_number in range(1, repeat + 1):
            started = perf_counter()
            metrics, execution = await evaluator.evaluate(valid_program, primary_metric="score")
            wall_clock_ms = (perf_counter() - started) * 1_000
            valid_wall_clock_ms.append(wall_clock_ms)
            logger.info(
                "probe_docker_valid run=%s wall_clock_ms=%.3f status=%s metric=%.3f",
                run_number,
                wall_clock_ms,
                execution.status,
                metrics.get("score", 0.0),
            )
            if execution.status != "success" or metrics.get("score") != 1.0:
                logger.error("Valid Docker probe fixture failed unexpectedly.")
                return 1

        started = perf_counter()
        _, network_execution = await evaluator.evaluate(network_program, primary_metric="score")
        network_wall_clock_ms = (perf_counter() - started) * 1_000
        logger.info(
            "probe_docker_network wall_clock_ms=%.3f status=%s stderr=%s",
            network_wall_clock_ms,
            network_execution.status,
            network_execution.stderr.strip() or "<empty>",
        )
        if network_execution.status == "success":
            logger.error("Network fixture unexpectedly succeeded inside the sandbox.")
            return 1

        started = perf_counter()
        _, timeout_execution = await evaluator.evaluate(timeout_program, primary_metric="score")
        timeout_wall_clock_ms = (perf_counter() - started) * 1_000
        logger.info(
            "probe_docker_timeout wall_clock_ms=%.3f status=%s",
            timeout_wall_clock_ms,
            timeout_execution.status,
        )
        if timeout_execution.status != "timeout":
            logger.error("Timeout fixture did not terminate with status='timeout'.")
            return 1
    finally:
        await evaluator.aclose()

    median_wall_clock_ms = median(valid_wall_clock_ms)
    if median_wall_clock_ms <= 3_000:
        logger.info(
            "probe_docker_complete median_valid_wall_clock_ms=%.3f health=healthy",
            median_wall_clock_ms,
        )
    else:
        logger.warning(
            "probe_docker_complete median_valid_wall_clock_ms=%.3f health=slow_threshold_exceeded",
            median_wall_clock_ms,
        )
    return 0


async def _run_controller(config, *, logger: logging.Logger):
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
    return await controller.run(Program.seed(seed_code))
