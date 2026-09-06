import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.config import BASE_DIR
from mcp_client import MCPClient
from mcp_client.exceptions import MCPClientStateError, MCPConnectionError, MCPToolDiscoveryError


EXPECTED_TOOL_NAMES = {
    "create_task", "get_task", "list_tasks", "update_task", "complete_task", "delete_task",
    "create_event", "get_event", "list_events", "update_event", "delete_event",
    "create_note", "get_note", "list_notes", "update_note", "delete_note", "search_notes",
}


@pytest.fixture
def client_factory():
    def _factory(**kwargs):
        env = {**os.environ, "DATABASE_URL": "sqlite:///:memory:"}
        return MCPClient(
            server_path=BASE_DIR / "mcp_servers" / "unified_server.py",
            env=env,
            **kwargs,
        )

    return _factory


def test_client_creation(client_factory):
    client = client_factory()
    assert client is not None
    assert client.is_connected is False
    assert client.session is None


def test_client_initial_state(client_factory):
    client = client_factory()
    assert client.is_connected is False
    assert client.session is None


@pytest.mark.asyncio
async def test_successful_connection(client_factory):
    client = client_factory()
    await client.connect()
    assert client.is_connected is True
    assert client.session is not None
    await client.close()
    assert client.is_connected is False


@pytest.mark.asyncio
async def test_successful_close(client_factory):
    client = client_factory()
    await client.connect()
    await client.close()
    assert client.is_connected is False
    assert client.session is None


@pytest.mark.asyncio
async def test_repeated_close(client_factory):
    client = client_factory()
    await client.connect()
    await client.close()
    await client.close()
    assert client.is_connected is False
    assert client.session is None


@pytest.mark.asyncio
async def test_repeated_connect_is_idempotent(client_factory):
    client = client_factory()
    await client.connect()
    await client.connect()
    assert client.is_connected is True
    assert client.session is not None
    await client.close()


@pytest.mark.asyncio
async def test_connection_failure_raises_client_exception():
    client = MCPClient(server_path=BASE_DIR / "mcp_servers" / "missing_server.py")
    with pytest.raises(MCPConnectionError):
        await client.connect()


@pytest.mark.asyncio
async def test_context_manager_lifecycle(client_factory):
    client = client_factory()
    async with client:
        assert client.is_connected is True
        assert client.session is not None
    assert client.is_connected is False
    assert client.session is None


@pytest.mark.asyncio
async def test_list_tools_requires_connection(client_factory):
    with pytest.raises(MCPClientStateError, match="Connect to the MCP server"):
        await client_factory().list_tools()


@pytest.mark.asyncio
async def test_successful_tool_discovery(client_factory):
    client = client_factory()
    await client.connect()
    tools = await client.list_tools()
    assert tools
    await client.close()


@pytest.mark.asyncio
async def test_discovery_returns_exactly_17_tools(client_factory):
    client = client_factory()
    await client.connect()
    tools = await client.list_tools()
    assert len(tools) == 17
    await client.close()


@pytest.mark.asyncio
async def test_discovered_tool_names_match_unified_server_contract(client_factory):
    client = client_factory()
    await client.connect()
    names = {tool.name for tool in await client.list_tools()}
    assert names == EXPECTED_TOOL_NAMES
    await client.close()


@pytest.mark.asyncio
async def test_discovered_tool_names_are_unique(client_factory):
    client = client_factory()
    await client.connect()
    names = [tool.name for tool in await client.list_tools()]
    assert len(names) == len(set(names))
    await client.close()


@pytest.mark.asyncio
async def test_discovered_tool_metadata_is_accessible(client_factory):
    client = client_factory()
    await client.connect()
    tools_by_name = {tool.name: tool for tool in await client.list_tools()}
    tool = tools_by_name["create_task"]
    assert isinstance(tool.name, str)
    assert hasattr(tool, "description")
    await client.close()


@pytest.mark.asyncio
async def test_discovered_tool_input_schemas_are_accessible(client_factory):
    client = client_factory()
    await client.connect()
    tools_by_name = {tool.name: tool for tool in await client.list_tools()}
    for name in ("create_task", "create_event", "create_note"):
        tool = tools_by_name[name]
        assert isinstance(tool.inputSchema, dict)
    await client.close()


@pytest.mark.asyncio
async def test_repeated_tool_discovery_uses_active_session(client_factory):
    client = client_factory()
    await client.connect()
    tools1 = await client.list_tools()
    tools2 = await client.list_tools()
    assert {tool.name for tool in tools1} == {tool.name for tool in tools2}
    assert len(tools1) == len(tools2)
    await client.close()


@pytest.mark.asyncio
async def test_tool_discovery_after_reconnect(client_factory):
    client = client_factory()
    await client.connect()
    first_names = {tool.name for tool in await client.list_tools()}
    await client.close()
    await client.connect()
    second_names = {tool.name for tool in await client.list_tools()}
    await client.close()
    assert first_names == second_names == EXPECTED_TOOL_NAMES


@pytest.mark.asyncio
async def test_close_after_tool_discovery(client_factory):
    client = client_factory()
    await client.connect()
    await client.list_tools()
    await client.close()
    assert client.is_connected is False


@pytest.mark.asyncio
async def test_tool_discovery_failure_raises_client_exception(client_factory):
    client = client_factory()
    await client.connect()
    assert client.session is not None
    client.session.list_tools = AsyncMock(side_effect=RuntimeError("protocol failure"))
    with pytest.raises(MCPToolDiscoveryError, match="Unable to discover MCP server tools") as error:
        await client.list_tools()
    assert isinstance(error.value.__cause__, RuntimeError)
    await client.close()
