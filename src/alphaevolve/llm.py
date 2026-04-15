"""Inference clients for local models and deterministic tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Sequence

import httpx

from alphaevolve.errors import CapabilityUnavailableError
from alphaevolve.models import ModelConfig

DEFAULT_KNAPSACK_DIFF = """<<<<<<< SEARCH
    return value
=======
    return value / max(weight, 1e-9)
>>>>>>> REPLACE"""


class AsyncInferenceClient(ABC):
    """Abstract async inference interface."""

    @abstractmethod
    async def generate_diff(self, prompt: str, *, attempt: int = 1) -> str:
        """Generate a diff-like response for a given prompt."""

    async def aclose(self) -> None:
        """Optional async cleanup hook."""


class OllamaClient(AsyncInferenceClient):
    """Async HTTP client for a local Ollama instance."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
        )

    async def generate_diff(self, prompt: str, *, attempt: int = 1) -> str:
        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._config.temperature},
        }
        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        body = response.json()
        if "response" in body:
            return str(body["response"])
        message = body.get("message", {})
        if isinstance(message, dict) and "content" in message:
            return str(message["content"])
        raise CapabilityUnavailableError("Ollama response did not contain a generated diff.")

    async def aclose(self) -> None:
        await self._client.aclose()


class FakeInferenceClient(AsyncInferenceClient):
    """Deterministic fake client for tests and offline benchmarks."""

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        fallback: Callable[[str, int], str] | None = None,
    ) -> None:
        self._responses = deque(responses or [])
        self._fallback = fallback or (lambda prompt, attempt: DEFAULT_KNAPSACK_DIFF)

    async def generate_diff(self, prompt: str, *, attempt: int = 1) -> str:
        if self._responses:
            return self._responses.popleft()
        return self._fallback(prompt, attempt)


async def check_ollama_availability(base_url: str, timeout_seconds: float = 2.0) -> tuple[bool, str]:
    """Best-effort availability check for a local Ollama instance."""
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def build_inference_client(config: ModelConfig) -> AsyncInferenceClient:
    """Factory for inference adapters."""
    provider = config.provider.lower()
    if provider == "ollama":
        return OllamaClient(config)
    if provider == "fake":
        return FakeInferenceClient(config.scripted_responses)
    raise CapabilityUnavailableError(f"Unsupported inference provider: {config.provider}")
