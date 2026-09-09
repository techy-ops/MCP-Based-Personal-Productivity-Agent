from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, TypedDict

from openai import APIConnectionError, APIStatusError, AuthenticationError, BadRequestError, OpenAI, RateLimitError

from .config import LLMConfig
from .exceptions import LLMAuthenticationError, LLMConnectionError, LLMRequestError, LLMResponseError


class LLMMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral result returned to future agent layers."""

    content: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        timeout: float,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Generate a provider response for normalized chat messages."""


class OpenAIProvider:
    """Adapter around the OpenAI SDK; raw SDK objects stay inside this module."""

    def __init__(self, api_key: str, *, timeout: float, client: Any | None = None) -> None:
        self._client = client or OpenAI(api_key=api_key, timeout=timeout)

    def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        timeout: float,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {"model": model, "messages": list(messages), "timeout": timeout}
        if temperature is not None:
            request["temperature"] = temperature
        if max_tokens is not None:
            request["max_tokens"] = max_tokens

        try:
            raw_response = self._client.chat.completions.create(**request)
        except AuthenticationError as exc:
            raise LLMAuthenticationError("LLM provider authentication failed.") from exc
        except (APIConnectionError, TimeoutError) as exc:
            raise LLMConnectionError("Unable to connect to the LLM provider.") from exc
        except (BadRequestError, RateLimitError, APIStatusError) as exc:
            raise LLMRequestError("The LLM provider rejected the request.") from exc
        except Exception as exc:
            raise LLMRequestError("The LLM provider request failed.") from exc

        try:
            choice = raw_response.choices[0]
            content = choice.message.content
            response_model = raw_response.model
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise LLMResponseError("The LLM provider returned an invalid response.") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("The LLM provider returned empty content.")
        if not isinstance(response_model, str) or not response_model.strip():
            raise LLMResponseError("The LLM provider response did not include a model.")

        usage = getattr(raw_response, "usage", None)
        metadata: dict[str, Any] = {}
        if usage is not None:
            metadata["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else usage
        return LLMResponse(content=content, model=response_model, metadata=metadata)


class LLMClient:
    """Small provider-neutral generation interface for the future agent layer."""

    def __init__(self, config: LLMConfig, *, provider: LLMProvider | None = None) -> None:
        self.config = config
        self._provider = provider or self._build_provider(config)

    @classmethod
    def from_env(cls, *, provider: LLMProvider | None = None) -> "LLMClient":
        return cls(LLMConfig.from_env(), provider=provider)

    @staticmethod
    def _build_provider(config: LLMConfig) -> LLMProvider:
        if config.provider == "openai":
            return OpenAIProvider(config.api_key, timeout=config.timeout)
        raise ValueError(f"Unsupported LLM provider '{config.provider}'.")

    def generate(self, prompt: str | Sequence[LLMMessage], *, system_message: str | None = None) -> LLMResponse:
        messages = self._normalize_messages(prompt, system_message=system_message)
        return self._provider.generate(
            messages,
            model=self.config.model,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    @staticmethod
    def _normalize_messages(
        prompt: str | Sequence[LLMMessage], *, system_message: str | None
    ) -> list[LLMMessage]:
        if isinstance(prompt, str):
            if not prompt.strip():
                raise ValueError("Prompt must be a non-empty string.")
            messages: list[LLMMessage] = [{"role": "user", "content": prompt}]
        else:
            messages = [dict(message) for message in prompt]
            if not messages or any(
                not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in messages
            ):
                raise ValueError("Messages must contain non-empty role and content strings.")

        if system_message is not None:
            if not system_message.strip():
                raise ValueError("System message must be a non-empty string.")
            messages.insert(0, {"role": "system", "content": system_message})
        return messages
