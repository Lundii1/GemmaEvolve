from __future__ import annotations

import asyncio

import pytest

from alphaevolve.evaluators import docker_environment_status
from alphaevolve.llm import check_ollama_availability


def test_ollama_smoke_hook_skips_cleanly_when_ollama_is_unavailable() -> None:
    available, reason = asyncio.run(check_ollama_availability("http://localhost:11434", timeout_seconds=0.2))
    if not available:
        pytest.skip(reason)
    assert available is True


def test_docker_smoke_hook_skips_cleanly_when_docker_is_unavailable() -> None:
    available, reason = docker_environment_status()
    if not available:
        pytest.skip(reason)
    assert available is True
