import asyncio
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import connection as db_connection
import app.services.calendar_service as calendar_service


@pytest.fixture
def calendar_mcp_server(monkeypatch):
    test_engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
    from app.database.models import Base

    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    monkeypatch.setattr(db_connection, "engine", test_engine)
    monkeypatch.setattr(db_connection, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(calendar_service, "SessionLocal", TestSessionLocal)

    from mcp_servers.calendar_server import server
    yield server


async def invoke_tool(server, tool_name, **kwargs):
    result = await server.call_tool(tool_name, kwargs)
    assert result and len(result) > 0
    payload = json.loads(result[0].text)
    return payload


@pytest.mark.asyncio
async def test_calendar_mcp_server_imports_and_initializes(calendar_mcp_server):
    assert calendar_mcp_server is not None
    tools = await calendar_mcp_server.list_tools()
    assert isinstance(tools, list)


@pytest.mark.asyncio
async def test_calendar_mcp_server_registers_expected_tools(calendar_mcp_server):
    tools = await calendar_mcp_server.list_tools()
    names = {tool.name for tool in tools}
    assert {"create_event", "get_event"}.issubset(names)


@pytest.mark.asyncio
async def test_create_event_tool_success(calendar_mcp_server):
    payload = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Quarterly planning",
        description="Plan the next milestone",
        start_time="2026-09-10T09:00:00",
        end_time="2026-09-10T10:00:00",
        location="Conference room",
    )
    assert payload["success"] is True
    assert payload["data"]["title"] == "Quarterly planning"
    assert payload["data"]["location"] == "Conference room"


@pytest.mark.asyncio
async def test_get_event_tool_success(calendar_mcp_server):
    created = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Design review",
        start_time="2026-09-10T12:00:00",
        end_time="2026-09-10T13:00:00",
    )
    payload = await invoke_tool(calendar_mcp_server, "get_event", event_id=created["data"]["id"])
    assert payload["success"] is True
    assert payload["data"]["title"] == "Design review"


@pytest.mark.asyncio
async def test_get_event_tool_not_found(calendar_mcp_server):
    payload = await invoke_tool(calendar_mcp_server, "get_event", event_id=999)
    assert payload["success"] is False
    assert "not found" in payload["error"].lower()


@pytest.mark.asyncio
async def test_create_event_invalid_input_handling(calendar_mcp_server):
    payload = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="",
        start_time="2026-09-10T15:00:00",
        end_time="2026-09-10T14:00:00",
    )
    assert payload["success"] is False
    assert "title" in payload["error"].lower() or "end_time" in payload["error"].lower()


@pytest.mark.asyncio
async def test_create_event_conflict_detection(calendar_mcp_server):
    await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Existing meeting",
        start_time="2026-09-10T10:00:00",
        end_time="2026-09-10T11:00:00",
    )
    payload = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Overlap meeting",
        start_time="2026-09-10T10:30:00",
        end_time="2026-09-10T11:30:00",
    )
    assert payload["success"] is False
    assert "conflict" in payload["error"].lower()


@pytest.mark.asyncio
async def test_create_event_persists_through_service_layer(calendar_mcp_server):
    created = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Persistence check",
        start_time="2026-09-10T08:00:00",
        end_time="2026-09-10T09:00:00",
    )
    event_id = created["data"]["id"]
    fetched = await invoke_tool(calendar_mcp_server, "get_event", event_id=event_id)
    assert fetched["success"] is True
    assert fetched["data"]["id"] == event_id
    assert fetched["data"]["title"] == "Persistence check"


@pytest.mark.asyncio
async def test_calendar_mcp_tool_descriptions_and_schemas(calendar_mcp_server):
    tools = await calendar_mcp_server.list_tools()
    tool_map = {tool.name: tool for tool in tools}
    assert "create_event" in tool_map
    assert "get_event" in tool_map
    assert "calendar" in (tool_map["create_event"].description or "").lower()
    assert "event" in (tool_map["get_event"].description or "").lower()
    assert "inputSchema" in tool_map["create_event"].__dict__
    assert "inputSchema" in tool_map["get_event"].__dict__
