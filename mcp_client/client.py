from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.config import BASE_DIR

from .exceptions import MCPClientStateError, MCPConnectionError, MCPToolDiscoveryError


class MCPClient:
    """Async client for Unified MCP server lifecycle and tool discovery.

    The client is intentionally scoped to protocol/session management only. Later
    phases can build tool invocation on top of this foundation.
    """

    def __init__(
        self,
        server_path: str | Path | None = None,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self.server_path = Path(server_path) if server_path is not None else BASE_DIR / "mcp_servers" / "unified_server.py"
        self.server_path = self.server_path.resolve()
        self.command = command or sys.executable
        self.args = list(args) if args is not None else [str(self.server_path)]
        self.cwd = Path(cwd).resolve() if cwd is not None else BASE_DIR

        base_env = {**os.environ, **(env or {})}
        project_root = str(self.cwd)
        pythonpath = base_env.get("PYTHONPATH")
        if pythonpath:
            base_env["PYTHONPATH"] = os.pathsep.join(filter(None, [project_root, pythonpath]))
        else:
            base_env["PYTHONPATH"] = project_root
        self.env = base_env

        self.session: ClientSession | None = None
        self._transport_cm: Any | None = None
        self._session_cm: Any | None = None

    @property
    def is_connected(self) -> bool:
        return self.session is not None and self._transport_cm is not None and self._session_cm is not None

    async def connect(self) -> "MCPClient":
        """Connect to the Unified MCP server and initialize the MCP session."""
        if self.is_connected:
            return self

        if self.session is not None or self._session_cm is not None or self._transport_cm is not None:
            await self.close()

        transport_server = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
            cwd=self.cwd,
        )

        try:
            self._transport_cm = stdio_client(transport_server)
            read_stream, write_stream = await self._transport_cm.__aenter__()
            self._session_cm = ClientSession(read_stream, write_stream)
            await self._session_cm.__aenter__()
            self.session = self._session_cm
            await self.session.initialize()
            return self
        except Exception as exc:
            await self.close()
            raise MCPConnectionError(f"Unable to connect to MCP server at {self.server_path}") from exc

    async def close(self) -> None:
        """Shut down the current MCP session and associated transport."""
        if self._session_cm is None and self.session is None and self._transport_cm is None:
            return

        session_cm = self._session_cm
        transport_cm = self._transport_cm

        self._session_cm = None
        self.session = None
        self._transport_cm = None

        try:
            if session_cm is not None:
                await session_cm.__aexit__(None, None, None)
        except Exception:
            pass

        try:
            if transport_cm is not None:
                await transport_cm.__aexit__(None, None, None)
        except Exception:
            pass

    async def list_tools(self) -> list[Any]:
        """Return native MCP tool metadata discovered from the active session.

        Each SDK tool object exposes its name, description, and input schema.
        Metadata is retrieved through ``ClientSession.list_tools()``; no local
        tool registry or server-module inspection is used.
        """
        if not self.is_connected or self.session is None:
            raise MCPClientStateError("Connect to the MCP server before listing tools.")

        try:
            result = await self.session.list_tools()
        except Exception as exc:
            raise MCPToolDiscoveryError("Unable to discover MCP server tools.") from exc

        return result.tools

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"MCPClient(server_path={self.server_path!s}, connected={self.is_connected})"
