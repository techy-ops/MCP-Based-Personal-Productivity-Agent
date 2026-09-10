from __future__ import annotations


class AgentError(Exception):
    """Base class for agent-layer failures."""


class AgentConfigurationError(AgentError):
    """Raised when the agent cannot be configured safely."""


class AgentExecutionError(AgentError):
    """Raised when the graph executes but the LLM step fails."""
