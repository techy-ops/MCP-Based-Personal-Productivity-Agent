class MCPClientError(Exception):
    """Base exception for MCP client lifecycle and protocol errors."""


class MCPConnectionError(MCPClientError):
    """Raised when the client cannot establish or maintain an MCP session."""


class MCPClientStateError(MCPClientError):
    """Raised when a client lifecycle method is used in an invalid state."""


class MCPToolDiscoveryError(MCPClientError):
    """Raised when the MCP server cannot provide its tool metadata."""
