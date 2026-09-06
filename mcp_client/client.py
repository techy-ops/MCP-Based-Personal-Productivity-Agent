from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from app.config import BASE_DIR

from .exceptions import (
    MCPClientStateError,
    MCPConnectionError,
    MCPToolDiscoveryError,
    MCPToolInvocationError,
)


class MCPClient:
    """Async client for Unified MCP server lifecycle, discovery, and invocation."""

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
        self._known_tool_names: set[str] = set()

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
            self._known_tool_names.clear()
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
        self._known_tool_names.clear()

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
        """Return native MCP tool metadata discovered from the active session."""
        if not self.is_connected or self.session is None:
            raise MCPClientStateError("Connect to the MCP server before listing tools.")

        try:
            result = await self.session.list_tools()
        except Exception as exc:
            raise MCPToolDiscoveryError("Unable to discover MCP server tools.") from exc

        tool_metadata = result.tools
        self._known_tool_names = {tool.name for tool in tool_metadata}
        return tool_metadata

    @staticmethod
    def _extract_text_content(result: Any) -> str | None:
        if result is None:
            return None

        if getattr(result, "structuredContent", None) is not None:
            structured = result.structuredContent
            if isinstance(structured, dict):
                if structured.get("success") is False:
                    error = structured.get("error")
                    if error:
                        return str(error)
                return json.dumps(structured)

        content = getattr(result, "content", None)
        if not content:
            return None

        for item in content:
            text = getattr(item, "text", None)
            if text:
                return str(text)
        return None

    @classmethod
    def _coerce_tool_error_message(cls, result: Any) -> str:
        text = cls._extract_text_content(result)
        if text is None:
            return "MCP tool invocation failed."

        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return text

        if isinstance(payload, dict):
            if payload.get("success") is False:
                message = payload.get("error") or payload.get("message")
                if message:
                    return str(message)
            if payload.get("error"):
                return str(payload["error"])

        return text

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Invoke any MCP tool discovered from the active server session.

        The client validates the connection and optional discovered tool names, then
        forwards the request to the active ``ClientSession`` without hard-coded domain
        routing.
        """
        if not self.is_connected or self.session is None:
            raise MCPClientStateError("Connect to the MCP server before invoking a tool.")

        if self._known_tool_names and name not in self._known_tool_names:
            raise MCPToolInvocationError(f"Tool '{name}' is not available on the connected MCP server.")

        payload = arguments if arguments is not None else {}
        if not isinstance(payload, dict):
            raise MCPToolInvocationError("Tool arguments must be provided as a dictionary.")

        try:
            result = await self.session.call_tool(name, payload)
        except Exception as exc:
            raise MCPToolInvocationError(f"MCP invocation failed for tool '{name}'.") from exc

        if getattr(result, "isError", False):
            message = self._coerce_tool_error_message(result)
            raise MCPToolInvocationError(message, result=result)

        if getattr(result, "structuredContent", None) is not None:
            structured = result.structuredContent
            if isinstance(structured, dict) and structured.get("success") is False:
                error_message = structured.get("error") or structured.get("message") or "Tool execution failed."
                raise MCPToolInvocationError(str(error_message), result=result)

        if getattr(result, "content", None):
            try:
                content = result.content[0].text
                payload = json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("success") is False:
                error_message = payload.get("error") or payload.get("message") or "Tool execution failed."
                raise MCPToolInvocationError(str(error_message), result=result)

        return result

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"MCPClient(server_path={self.server_path!s}, connected={self.is_connected})"
