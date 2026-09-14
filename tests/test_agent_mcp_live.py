from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import pytest

from agent import Agent, build_agent
from app.config import BASE_DIR
from llm import LLMClient, LLMConfig, LLMResponse
from mcp_client import MCPClient


class LiveFakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        model: str,
        timeout: float,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": list(messages), "model": model})
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse("Default live response", model)


def make_live_client(responses: list[LLMResponse]) -> LLMClient:
    provider = LiveFakeProvider(responses)
    config = LLMConfig(provider="openai", model="live-test-model", api_key="test-key", timeout=15.0)
    return LLMClient(config, provider=provider)


@pytest.fixture
def isolated_mcp_client(tmp_path):
    db_path = tmp_path / "live_agent_verification.db"
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path.as_posix()}"}
    client = MCPClient(
        server_path=BASE_DIR / "mcp_servers" / "unified_server.py",
        env=env,
    )
    return client


@pytest.mark.asyncio
async def test_live_dynamic_discovery_finds_exactly_17_tools(isolated_mcp_client):
    """Verify live dynamic discovery finds exactly 17 Unified MCP tools."""
    llm_client = make_live_client([
        LLMResponse("Direct answer", "live-test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=isolated_mcp_client)

    try:
        result = await agent.ainvoke("Check tools")
        discovered = result["discovered_tools"]
        assert len(discovered) == 17

        names = {t["name"] for t in discovered}
        # Check Task domain
        assert {"create_task", "get_task", "list_tasks", "update_task", "complete_task", "delete_task"}.issubset(names)
        # Check Calendar domain
        assert {"create_event", "get_event", "list_events", "update_event", "delete_event"}.issubset(names)
        # Check Notes domain
        assert {"create_note", "get_note", "list_notes", "update_note", "delete_note", "search_notes"}.issubset(names)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_live_agent_invokes_task_tool(isolated_mcp_client):
    """Verify live agent invocation of Task domain tool create_task."""
    llm_client = make_live_client([
        LLMResponse(json.dumps({
            "tool_name": "create_task",
            "arguments": {
                "title": "Live Task Verification",
                "description": "Created via Phase 3.3 live test",
                "priority": "high",
            },
        }), "live-test-model"),
        LLMResponse("Successfully created the live verification task.", "live-test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=isolated_mcp_client)

    try:
        result = await agent.ainvoke("Create a high priority task titled Live Task Verification")
        assert result["selected_tool"] == "create_task"
        assert result["tool_status"] == "success"
        assert result["tool_result"]["success"] is True
        assert result["tool_result"]["data"]["title"] == "Live Task Verification"
        assert result["tool_result"]["data"]["priority"] == "high"
        assert "Successfully created" in result["final_response"]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_live_agent_invokes_calendar_tool(isolated_mcp_client):
    """Verify live agent invocation of Calendar domain tool create_event."""
    llm_client = make_live_client([
        LLMResponse(json.dumps({
            "tool_name": "create_event",
            "arguments": {
                "title": "Live Standup",
                "start_time": "2026-09-25T09:00:00",
                "end_time": "2026-09-25T09:30:00",
                "location": "Virtual Room",
            },
        }), "live-test-model"),
        LLMResponse("Scheduled Live Standup on 2026-09-25.", "live-test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=isolated_mcp_client)

    try:
        result = await agent.ainvoke("Schedule Live Standup on Sep 25 from 9:00 to 9:30 AM")
        assert result["selected_tool"] == "create_event"
        assert result["tool_status"] == "success"
        assert result["tool_result"]["success"] is True
        assert result["tool_result"]["data"]["title"] == "Live Standup"
        assert result["tool_result"]["data"]["location"] == "Virtual Room"
        assert "Live Standup" in result["final_response"]
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_live_agent_invokes_notes_tool(isolated_mcp_client):
    """Verify live agent invocation of Notes domain tool create_note."""
    llm_client = make_live_client([
        LLMResponse(json.dumps({
            "tool_name": "create_note",
            "arguments": {
                "title": "Live Release Notes",
                "content": "Phase 3.3 live verification completed successfully.",
            },
        }), "live-test-model"),
        LLMResponse("Saved Live Release Notes note.", "live-test-model"),
    ])
    agent = build_agent(llm_client, mcp_client=isolated_mcp_client)

    try:
        result = await agent.ainvoke("Write note Live Release Notes with content Phase 3.3 live verification.")
        assert result["selected_tool"] == "create_note"
        assert result["tool_status"] == "success"
        assert result["tool_result"]["success"] is True
        assert result["tool_result"]["data"]["title"] == "Live Release Notes"
        assert "Phase 3.3 live verification" in result["tool_result"]["data"]["content"]
    finally:
        await agent.close()
