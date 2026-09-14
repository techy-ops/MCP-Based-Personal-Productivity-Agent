from __future__ import annotations


class AgentError(Exception):
    """Base class for agent-layer failures."""


class AgentConfigurationError(AgentError):
    """Raised when the agent cannot be configured safely."""


class AgentExecutionError(AgentError):
    """Raised when the graph executes but a node or workflow step fails."""


class AgentToolDiscoveryError(AgentExecutionError):
    """Raised when discovering tools from the MCP server fails."""


class AgentToolValidationError(AgentExecutionError):
    """Raised when a proposed tool call fails schema or argument validation."""


class AgentToolInvocationError(AgentExecutionError):
    """Raised when executing a tool call through MCP fails."""
