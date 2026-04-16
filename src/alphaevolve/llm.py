"""Inference clients for local models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from alphaevolve.errors import CapabilityUnavailableError
from alphaevolve.models import ModelConfig


class AsyncInferenceClient(ABC):
    """Abstract async inference interface."""

    @abstractmethod
    async def generate_text(self, prompt: str, *, attempt: int = 1) -> str:
        """Generate text for a given prompt."""

    async def generate_diff(self, prompt: str, *, attempt: int = 1) -> str:
        """Backward-compatible helper for diff prompts."""
        return await self.generate_text(prompt, attempt=attempt)

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

    async def generate_text(self, prompt: str, *, attempt: int = 1) -> str:
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
        raise CapabilityUnavailableError("Ollama response did not contain generated text.")

    async def aclose(self) -> None:
        await self._client.aclose()


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
    raise CapabilityUnavailableError(f"Unsupported inference provider: {config.provider}")
