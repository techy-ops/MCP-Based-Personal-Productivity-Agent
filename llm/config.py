from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .exceptions import LLMConfigurationError

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    """Validated provider settings used by the LLM client."""

    provider: str
    model: str
    api_key: str
    timeout: float = 60.0
    temperature: float | None = None
    max_tokens: int | None = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()

        if provider != "openai":
            raise LLMConfigurationError(f"Unsupported LLM provider '{provider}'.")
        if not model:
            raise LLMConfigurationError("Missing required LLM_MODEL configuration.")
        if not api_key:
            raise LLMConfigurationError("Missing required LLM_API_KEY configuration.")

        timeout = _read_float("LLM_TIMEOUT", default=60.0, minimum=0.0)
        temperature = _read_optional_float("LLM_TEMPERATURE", minimum=0.0, maximum=2.0)
        max_tokens = _read_optional_int("LLM_MAX_TOKENS", minimum=1)
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def _read_float(name: str, *, default: float, minimum: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be a number.") from exc
    if parsed <= minimum:
        raise LLMConfigurationError(f"{name} must be greater than {minimum}.")
    return parsed


def _read_optional_float(name: str, *, minimum: float, maximum: float) -> float | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be a number.") from exc
    if not minimum <= parsed <= maximum:
        raise LLMConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _read_optional_int(name: str, *, minimum: int) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be an integer.") from exc
    if parsed < minimum:
        raise LLMConfigurationError(f"{name} must be at least {minimum}.")
    return parsed
