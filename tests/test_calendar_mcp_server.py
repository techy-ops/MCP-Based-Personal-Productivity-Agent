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
MCP    assert {"create_event", "get_event", "list_events", "update_event", "delete_event"}.issubset(names)


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
async def test_list_events_tool_success(calendar_mcp_server):
    await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Morning standup",
        start_time="2026-09-10T09:00:00",
        end_time="2026-09-10T09:30:00",
    )
    payload = await invoke_tool(calendar_mcp_server, "list_events")
    assert payload["success"] is True
    assert len(payload["data"]) >= 1


@pytest.mark.asyncio
async def test_list_events_tool_with_date_range_filter(calendar_mcp_server):
    await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Day A event",
        start_time="2026-09-10T09:00:00",
        end_time="2026-09-10T10:00:00",
    )
    await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Day B event",
        start_time="2026-09-11T09:00:00",
        end_time="2026-09-11T10:00:00",
    )
    payload = await invoke_tool(
        calendar_mcp_server,
        "list_events",
        start_date="2026-09-10T00:00:00",
        end_date="2026-09-10T23:59:59",
    )
    assert payload["success"] is True
    assert all(item["title"] == "Day A event" for item in payload["data"])


@pytest.mark.asyncio
async def test_update_event_tool_success(calendar_mcp_server):
    created = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Original event",
        start_time="2026-09-10T14:00:00",
        end_time="2026-09-10T15:00:00",
    )
    payload = await invoke_tool(
        calendar_mcp_server,
        "update_event",
        event_id=created["data"]["id"],
        title="Updated event",
        end_time="2026-09-10T16:00:00",
    )
    assert payload["success"] is True
    assert payload["data"]["title"] == "Updated event"


@pytest.mark.asyncio
async def test_update_event_conflict_detection(calendar_mcp_server):
    first = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="First slot",
        start_time="2026-09-10T09:00:00",
        end_time="2026-09-10T10:00:00",
    )
    second = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Second slot",
        start_time="2026-09-10T11:00:00",
        end_time="2026-09-10T12:00:00",
    )
    payload = await invoke_tool(
        calendar_mcp_server,
        "update_event",
        event_id=second["data"]["id"],
        start_time="2026-09-10T09:30:00",
        end_time="2026-09-10T10:30:00",
    )
    assert payload["success"] is False
    assert "conflict" in payload["error"].lower()


@pytest.mark.asyncio
async def test_delete_event_tool_success(calendar_mcp_server):
    created = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="Delete me",
        start_time="2026-09-10T16:00:00",
        end_time="2026-09-10T17:00:00",
    )
    payload = await invoke_tool(calendar_mcp_server, "delete_event", event_id=created["data"]["id"])
    assert payload["success"] is True
    assert payload["data"] is True


@pytest.mark.asyncio
async def test_delete_event_missing_event(calendar_mcp_server):
    payload = await invoke_tool(calendar_mcp_server, "delete_event", event_id=999)
    assert payload["success"] is False
    assert "not found" in payload["error"].lower()


@pytest.mark.asyncio
async def test_calendar_mcp_end_to_end_workflow(calendar_mcp_server):
    created_a = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="A event",
        start_time="2026-09-12T09:00:00",
        end_time="2026-09-12T10:00:00",
    )
    created_b = await invoke_tool(
        calendar_mcp_server,
        "create_event",
        title="B event",
        start_time="2026-09-12T11:00:00",
        end_time="2026-09-12T12:00:00",
    )

    all_events = await invoke_tool(calendar_mcp_server, "list_events")
    assert all_events["success"] is True
    assert len(all_events["data"]) >= 2

    filtered = await invoke_tool(
        calendar_mcp_server,
        "list_events",
        start_date="2026-09-12T00:00:00",
        end_date="2026-09-12T23:59:59",
    )
    assert filtered["success"] is True
    assert any(item["id"] == created_a["data"]["id"] for item in filtered["data"])

    retrieved = await invoke_tool(calendar_mcp_server, "get_event", event_id=created_a["data"]["id"])
    assert retrieved["success"] is True

    updated = await invoke_tool(
        calendar_mcp_server,
        "update_event",
        event_id=created_a["data"]["id"],
        title="A event updated",
        end_time="2026-09-12T10:30:00",
    )
    assert updated["success"] is True
    assert updated["data"]["title"] == "A event updated"

    deleted = await invoke_tool(calendar_mcp_server, "delete_event", event_id=created_b["data"]["id"])
    assert deleted["success"] is True

    missing_after_delete = await invoke_tool(calendar_mcp_server, "get_event", event_id=created_b["data"]["id"])
    assert missing_after_delete["success"] is False


@pytest.mark.asyncio
async def test_calendar_mcp_tool_descriptions_and_schemas(calendar_mcp_server):
    tools = await calendar_mcp_server.list_tools()
    tool_map = {tool.name: tool for tool in tools}
    for name in ["create_event", "get_event", "list_events", "update_event", "delete_event"]:
        assert name in tool_map
        assert tool_map[name].description
        assert "inputSchema" in tool_map[name].__dict__

    assert "calendar" in (tool_map["create_event"].description or "").lower()
    assert "date" in (tool_map["list_events"].description or "").lower()
    assert "update" in (tool_map["update_event"].description or "").lower()
    assert "delete" in (tool_map["delete_event"].description or "").lower()
