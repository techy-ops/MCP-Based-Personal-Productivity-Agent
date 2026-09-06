"""MCP client foundation for lifecycle management and generic tool invocation."""

from .client import MCPClient
from .exceptions import (
    MCPClientError,
    MCPClientStateError,
    MCPConnectionError,
    MCPToolDiscoveryError,
    MCPToolInvocationError,
)

__all__ = [
    "MCPClient",
    "MCPClientError",
    "MCPClientStateError",
    "MCPConnectionError",
    "MCPToolDiscoveryError",
    "MCPToolInvocationError",
]
