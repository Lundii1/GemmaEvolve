from __future__ import annotations

import asyncio

from alphaevolve.llm import build_inference_client, check_inference_availability
from alphaevolve.models import ModelConfig, PromptBudget


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_build_inference_client_supports_longcat(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, *, base_url, timeout, headers=None):
            recorded["base_url"] = base_url
            recorded["timeout"] = timeout
            recorded["headers"] = headers

        async def post(self, path, json):
            recorded["path"] = path
            recorded["json"] = json
            return _FakeResponse({"choices": [{"message": {"content": "patched"}}]})

        async def aclose(self) -> None:
            return None

    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")
    monkeypatch.setattr("alphaevolve.llm.httpx.AsyncClient", _FakeAsyncClient)

    client = build_inference_client(
        ModelConfig(
            provider="longcat",
            model="LongCat-Flash-Chat",
            base_url="https://api.longcat.chat/openai",
            temperature=0.3,
            prompt_budget=PromptBudget(reserved_completion_tokens=321),
        )
    )

    text = asyncio.run(client.generate_text("hello"))

    assert text == "patched"
    assert recorded["base_url"] == "https://api.longcat.chat/openai"
    assert recorded["headers"] == {"Authorization": "Bearer test-key"}
    assert recorded["path"] == "/v1/chat/completions"
    assert recorded["json"] == {
        "model": "LongCat-Flash-Chat",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 321,
    }


def test_check_inference_availability_supports_longcat(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    class _FakeAsyncClient:
        def __init__(self, *, base_url, timeout, headers=None):
            recorded["base_url"] = base_url
            recorded["timeout"] = timeout
            recorded["headers"] = headers

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, path, json):
            recorded["path"] = path
            recorded["json"] = json
            return _FakeResponse({"choices": [{"message": {"content": "pong"}}]})

    monkeypatch.setenv("LONGCAT_API_KEY", "test-key")
    monkeypatch.setattr("alphaevolve.llm.httpx.AsyncClient", _FakeAsyncClient)

    available, reason = asyncio.run(
        check_inference_availability(
            ModelConfig(
                provider="longcat",
                model="LongCat-Flash-Chat",
                base_url="https://api.longcat.chat/openai",
            ),
            timeout_seconds=1.5,
        )
    )

    assert available is True
    assert reason == "ok"
    assert recorded["base_url"] == "https://api.longcat.chat/openai"
    assert recorded["headers"] == {"Authorization": "Bearer test-key"}
    assert recorded["path"] == "/v1/chat/completions"
    assert recorded["json"] == {
        "model": "LongCat-Flash-Chat",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "max_tokens": 1,
        "temperature": 0.1,
    }
