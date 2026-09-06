import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.config import BASE_DIR
from mcp_client import MCPClient
from mcp_client.exceptions import (
    MCPClientStateError,
    MCPConnectionError,
    MCPToolDiscoveryError,
    MCPToolInvocationError,
)


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


def _result_payload(result):
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    if getattr(result, "content", None):
        first = result.content[0]
        if hasattr(first, "text"):
            return json.loads(first.text)
    return result


@pytest.mark.asyncio
async def test_call_tool_requires_connection(client_factory):
    with pytest.raises(MCPClientStateError, match="Connect to the MCP server"):
        await client_factory().call_tool("create_task", {"title": "No connection"})


@pytest.mark.asyncio
async def test_call_tool_invokes_task_tool_via_mcp(client_factory):
    client = client_factory()
    await client.connect()
    try:
        result = await client.call_tool("create_task", {"title": "Client task", "priority": "high"})
        payload = _result_payload(result)
        assert result.isError is False
        assert payload["success"] is True
        assert payload["data"]["title"] == "Client task"
        task_id = payload["data"]["id"]
        retrieved = await client.call_tool("get_task", {"task_id": task_id})
        assert _result_payload(retrieved)["data"]["id"] == task_id
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_call_tool_invokes_calendar_tool_via_mcp(client_factory):
    client = client_factory()
    await client.connect()
    try:
        result = await client.call_tool(
            "create_event",
            {
                "title": "Client event",
                "start_time": "2026-09-20T09:00:00",
                "end_time": "2026-09-20T10:00:00",
                "location": "Workspace",
            },
        )
        payload = _result_payload(result)
        assert result.isError is False
        assert payload["success"] is True
        assert payload["data"]["title"] == "Client event"
        event_id = payload["data"]["id"]
        listed = await client.call_tool("list_events", {"start_date": "2026-09-20T00:00:00", "end_date": "2026-09-20T23:59:59"})
        assert any(item["id"] == event_id for item in _result_payload(listed)["data"])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_call_tool_invokes_note_tool_via_mcp(client_factory):
    client = client_factory()
    await client.connect()
    try:
        result = await client.call_tool("create_note", {"title": "Client note", "content": "Created through generic client call."})
        payload = _result_payload(result)
        assert result.isError is False
        assert payload["success"] is True
        assert payload["data"]["title"] == "Client note"
        note_id = payload["data"]["id"]
        retrieved = await client.call_tool("get_note", {"note_id": note_id})
        assert _result_payload(retrieved)["data"]["title"] == "Client note"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_call_tool_supports_cross_domain_single_session_workflow(client_factory):
    client = client_factory()
    await client.connect()
    try:
        task_result = await client.call_tool("create_task", {"title": "Cross task", "priority": "medium"})
        task_id = _result_payload(task_result)["data"]["id"]

        event_result = await client.call_tool(
            "create_event",
            {
                "title": "Cross event",
                "start_time": "2026-09-21T08:00:00",
                "end_time": "2026-09-21T09:00:00",
            },
        )
        event_id = _result_payload(event_result)["data"]["id"]

        note_result = await client.call_tool("create_note", {"title": "Cross note", "content": "Stored through a shared MCP session."})
        note_id = _result_payload(note_result)["data"]["id"]

        assert _result_payload(await client.call_tool("get_task", {"task_id": task_id}))["data"]["title"] == "Cross task"
        assert _result_payload(await client.call_tool("get_event", {"event_id": event_id}))["data"]["title"] == "Cross event"
        assert _result_payload(await client.call_tool("get_note", {"note_id": note_id}))["data"]["title"] == "Cross note"

        await client.call_tool("delete_task", {"task_id": task_id})
        await client.call_tool("delete_event", {"event_id": event_id})
        await client.call_tool("delete_note", {"note_id": note_id})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_call_tool_unknown_tool_raises_client_exception(client_factory):
    client = client_factory()
    await client.connect()
    try:
        with pytest.raises(MCPToolInvocationError, match="not available|does not exist|unknown"):
            await client.call_tool("nonexistent_tool", {})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_call_tool_invalid_arguments_propagate_server_errors(client_factory):
    client = client_factory()
    await client.connect()
    try:
        with pytest.raises(MCPToolInvocationError, match="title|empty"):
            await client.call_tool("create_task", {"title": ""})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_call_tool_uses_discovered_tool_names_from_session(client_factory):
    client = client_factory()
    await client.connect()
    try:
        discovered = {tool.name for tool in await client.list_tools()}
        assert "create_task" in discovered
        result = await client.call_tool("create_task", {"title": "Discovery invoked task"})
        assert result.isError is False
    finally:
        await client.close()
