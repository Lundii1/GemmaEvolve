"""Inference clients for local and hosted models."""

from __future__ import annotations

import os
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


class LongCatClient(AsyncInferenceClient):
    """Async HTTP client for the LongCat chat completions API."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._api_key_env = config.api_key_env or "LONGCAT_API_KEY"
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise CapabilityUnavailableError(
                f"LongCat API key not found in environment variable {self._api_key_env!r}."
            )
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=config.request_timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def generate_text(self, prompt: str, *, attempt: int = 1) -> str:
        payload = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": self._config.temperature,
            "max_tokens": self._config.prompt_budget.reserved_completion_tokens,
        }
        response = await self._client.post("/v1/chat/completions", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()
            if detail:
                raise CapabilityUnavailableError(
                    f"LongCat API request failed with status {response.status_code}: {detail}"
                ) from exc
            raise
        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            raise CapabilityUnavailableError("LongCat response did not contain any choices.")
        message = choices[0].get("message", {})
        if "content" in message and message["content"] is not None:
            return str(message["content"])
        raise CapabilityUnavailableError("LongCat response did not contain generated text.")

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


async def check_longcat_availability(
    config: ModelConfig,
    timeout_seconds: float = 2.0,
) -> tuple[bool, str]:
    """Best-effort availability check for a LongCat API endpoint."""
    api_key_env = config.api_key_env or "LONGCAT_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return False, f"missing environment variable {api_key_env!r}"
    try:
        async with httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": config.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "max_tokens": 1,
                    "temperature": 0.1,
                },
            )
            response.raise_for_status()
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


async def check_inference_availability(
    config: ModelConfig,
    timeout_seconds: float = 2.0,
) -> tuple[bool, str]:
    """Best-effort availability check for the configured inference provider."""
    provider = config.provider.lower()
    if provider == "ollama":
        return await check_ollama_availability(config.base_url, timeout_seconds=timeout_seconds)
    if provider == "longcat":
        return await check_longcat_availability(config, timeout_seconds=timeout_seconds)
    return False, f"unsupported inference provider: {config.provider}"


def build_inference_client(config: ModelConfig) -> AsyncInferenceClient:
    """Factory for inference adapters."""
    provider = config.provider.lower()
    if provider == "ollama":
        return OllamaClient(config)
    if provider == "longcat":
        return LongCatClient(config)
    raise CapabilityUnavailableError(f"Unsupported inference provider: {config.provider}")
