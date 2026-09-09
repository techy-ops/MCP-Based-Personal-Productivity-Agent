class LLMError(Exception):
    """Base exception for LLM configuration, transport, and response errors."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM configuration is missing or invalid."""


class LLMAuthenticationError(LLMError):
    """Raised when the provider rejects the configured credentials."""


class LLMConnectionError(LLMError):
    """Raised when a provider request cannot reach the service."""


class LLMRequestError(LLMError):
    """Raised when the provider rejects a validly formed request."""


class LLMResponseError(LLMError):
    """Raised when the provider response cannot be normalized."""
