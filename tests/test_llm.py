from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import httpx
import pytest
from openai import APIConnectionError, AuthenticationError

from llm import (
    LLMAuthenticationError,
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
    LLMConnectionError,
    LLMRequestError,
    LLMResponse,
    LLMResponseError,
)
from llm.client import OpenAIProvider


class FakeProvider:
    def __init__(self, response: LLMResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        timeout: float,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "timeout": timeout,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def valid_config() -> LLMConfig:
    return LLMConfig(provider="openai", model="test-model", api_key="test-key", timeout=12.0)


def test_config_loads_valid_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_TIMEOUT", "15")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_TOKENS", "100")

    config = LLMConfig.from_env()

    assert config == LLMConfig("openai", "test-model", "test-key", 15.0, 0.2, 100)


@pytest.mark.parametrize("missing_name", ["LLM_MODEL", "LLM_API_KEY"])
def test_config_rejects_missing_required_value(monkeypatch, missing_name):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv(missing_name)

    with pytest.raises(LLMConfigurationError, match=f"Missing required {missing_name}") as error:
        LLMConfig.from_env()

    assert "test-key" not in str(error.value)


def test_config_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    with pytest.raises(LLMConfigurationError, match="Unsupported LLM provider"):
        LLMConfig.from_env()


def test_config_rejects_invalid_generation_settings(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_TIMEOUT", "0")

    with pytest.raises(LLMConfigurationError, match="LLM_TIMEOUT"):
        LLMConfig.from_env()


def test_client_generates_and_normalizes_response():
    provider = FakeProvider(LLMResponse("Mock productivity response.", "test-model"))
    client = LLMClient(valid_config(), provider=provider)

    response = client.generate("Explain what a task is in one sentence.", system_message="Be concise.")

    assert response.content == "Mock productivity response."
    assert response.model == "test-model"
    assert provider.calls[0]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Explain what a task is in one sentence."},
    ]
    assert provider.calls[0]["model"] == "test-model"


def test_client_rejects_empty_prompt():
    client = LLMClient(valid_config(), provider=FakeProvider(LLMResponse("ok", "test-model")))

    with pytest.raises(ValueError, match="non-empty"):
        client.generate("  ")


def test_client_propagates_provider_errors_without_secrets():
    provider = FakeProvider(LLMRequestError("The LLM provider rejected the request."))
    client = LLMClient(valid_config(), provider=provider)

    with pytest.raises(LLMRequestError) as error:
        client.generate("synthetic prompt")

    assert "test-key" not in str(error.value)


class FakeOpenAIClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )
        self.response = response
        self.error = error

    def create(self, **kwargs):
        if self.error is not None:
            raise self.error
        return self.response


def test_openai_provider_normalizes_successful_response():
    raw = SimpleNamespace(
        model="test-model",
        choices=[SimpleNamespace(message=SimpleNamespace(content="Mock response."))],
        usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 3}),
    )
    provider = OpenAIProvider("test-key", timeout=10.0, client=FakeOpenAIClient(raw))

    response = provider.generate([{"role": "user", "content": "Hello"}], model="test-model", timeout=10.0)

    assert response == LLMResponse("Mock response.", "test-model", {"usage": {"total_tokens": 3}})


def test_openai_provider_rejects_malformed_response():
    raw = SimpleNamespace(model="test-model", choices=[])
    provider = OpenAIProvider("test-key", timeout=10.0, client=FakeOpenAIClient(raw))

    with pytest.raises(LLMResponseError):
        provider.generate([{"role": "user", "content": "Hello"}], model="test-model", timeout=10.0)


def test_openai_provider_translates_authentication_failure():
    response = httpx.Response(401, request=httpx.Request("POST", "https://example.test"))
    provider = OpenAIProvider(
        "test-key",
        timeout=10.0,
        client=FakeOpenAIClient(error=AuthenticationError("unauthorized", response=response, body=None)),
    )

    with pytest.raises(LLMAuthenticationError, match="authentication failed") as error:
        provider.generate([{"role": "user", "content": "Hello"}], model="test-model", timeout=10.0)

    assert "test-key" not in str(error.value)


def test_openai_provider_translates_connection_failure():
    request = httpx.Request("POST", "https://example.test")
    provider = OpenAIProvider(
        "test-key",
        timeout=10.0,
        client=FakeOpenAIClient(error=APIConnectionError(request=request)),
    )

    with pytest.raises(LLMConnectionError):
        provider.generate([{"role": "user", "content": "Hello"}], model="test-model", timeout=10.0)
