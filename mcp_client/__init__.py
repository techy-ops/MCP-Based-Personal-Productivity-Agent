"""MCP client foundation for lifecycle management against the Unified MCP server."""

from .client import MCPClient
from .exceptions import MCPClientError, MCPClientStateError, MCPConnectionError

__all__ = [
    "MCPClient",
    "MCPClientError",
    "MCPClientStateError",
    "MCPConnectionError",
]
