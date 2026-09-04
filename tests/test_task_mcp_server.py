import asyncio
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import connection as db_connection
import app.services.task_service as task_service
from app.utils.validators import NotFoundError, ValidationError


@pytest.fixture
def task_mcp_server(monkeypatch):
    test_engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
    from app.database.models import Base

    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    monkeypatch.setattr(db_connection, "engine", test_engine)
    monkeypatch.setattr(db_connection, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(task_service, "SessionLocal", TestSessionLocal)

    from mcp_servers.task_server import server
    yield server


async def invoke_tool(server, tool_name, **kwargs):
    result = await server.call_tool(tool_name, kwargs)
    assert result and len(result) > 0
    payload = json.loads(result[0].text)
    return payload


@pytest.mark.asyncio
async def test_task_mcp_server_registers_expected_tools(task_mcp_server):
    tools = await task_mcp_server.list_tools()
    names = {tool.name for tool in tools}
    assert {"create_task", "get_task", "list_tasks", "update_task", "complete_task", "delete_task"}.issubset(names)


@pytest.mark.asyncio
async def test_create_task_tool_success(task_mcp_server):
    payload = await invoke_tool(
        task_mcp_server,
        "create_task",
        title="Write report",
        description="Finish the final draft",
        priority="high",
    )
    assert payload["success"] is True
    assert payload["data"]["title"] == "Write report"
    assert payload["data"]["priority"] == "high"


@pytest.mark.asyncio
async def test_get_task_tool_success(task_mcp_server):
    created = await invoke_tool(task_mcp_server, "create_task", title="Review code", priority="medium")
    task_id = created["data"]["id"]
    payload = await invoke_tool(task_mcp_server, "get_task", task_id=task_id)
    assert payload["success"] is True
    assert payload["data"]["title"] == "Review code"


@pytest.mark.asyncio
async def test_get_task_tool_not_found(task_mcp_server):
    payload = await invoke_tool(task_mcp_server, "get_task", task_id=999)
    assert payload["success"] is False
    assert "not found" in payload["error"].lower()


@pytest.mark.asyncio
async def test_list_tasks_tool(task_mcp_server):
    await invoke_tool(task_mcp_server, "create_task", title="Alpha", priority="low")
    await invoke_tool(task_mcp_server, "create_task", title="Beta", priority="high")
    payload = await invoke_tool(task_mcp_server, "list_tasks")
    assert payload["success"] is True
    assert len(payload["data"]) >= 2


@pytest.mark.asyncio
async def test_list_tasks_tool_with_status_filter(task_mcp_server):
    await invoke_tool(task_mcp_server, "create_task", title="A", status="pending")
    await invoke_tool(task_mcp_server, "create_task", title="B", status="in_progress")
    payload = await invoke_tool(task_mcp_server, "list_tasks", status="in_progress")
    assert payload["success"] is True
    assert all(item["status"] == "in_progress" for item in payload["data"])


@pytest.mark.asyncio
async def test_list_tasks_tool_with_priority_filter(task_mcp_server):
    await invoke_tool(task_mcp_server, "create_task", title="High task", priority="high")
    await invoke_tool(task_mcp_server, "create_task", title="Low task", priority="low")
    payload = await invoke_tool(task_mcp_server, "list_tasks", priority="high")
    assert payload["success"] is True
    assert all(item["priority"] == "high" for item in payload["data"])


@pytest.mark.asyncio
async def test_update_task_tool_success(task_mcp_server):
    created = await invoke_tool(task_mcp_server, "create_task", title="Old task", priority="low")
    task_id = created["data"]["id"]
    payload = await invoke_tool(
        task_mcp_server,
        "update_task",
        task_id=task_id,
        title="New task",
        priority="high",
    )
    assert payload["success"] is True
    assert payload["data"]["title"] == "New task"
    assert payload["data"]["priority"] == "high"


@pytest.mark.asyncio
async def test_complete_task_tool_success(task_mcp_server):
    created = await invoke_tool(task_mcp_server, "create_task", title="Finish task", priority="medium")
    task_id = created["data"]["id"]
    payload = await invoke_tool(task_mcp_server, "complete_task", task_id=task_id)
    assert payload["success"] is True
    assert payload["data"]["status"] == "completed"


@pytest.mark.asyncio
async def test_delete_task_tool_success(task_mcp_server):
    created = await invoke_tool(task_mcp_server, "create_task", title="Delete me", priority="low")
    payload = await invoke_tool(task_mcp_server, "delete_task", task_id=created["data"]["id"])
    assert payload["success"] is True
    assert payload["data"] is True


@pytest.mark.asyncio
async def test_task_mcp_invalid_input_handling(task_mcp_server):
    payload = await invoke_tool(task_mcp_server, "create_task", title="", priority="high")
    assert payload["success"] is False
    assert "title" in payload["error"].lower()


@pytest.mark.asyncio
async def test_task_mcp_invalid_priority_handling(task_mcp_server):
    payload = await invoke_tool(task_mcp_server, "create_task", title="Bad priority", priority="urgent")
    assert payload["success"] is False
    assert "priority" in payload["error"].lower()
