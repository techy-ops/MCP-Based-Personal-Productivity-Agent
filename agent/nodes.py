from __future__ import annotations

from datetime import datetime
import json
import re
from typing import Any

from llm import LLMClient
from mcp_client import MCPClient
from mcp_client.exceptions import (
    MCPClientError,
    MCPConnectionError,
    MCPToolDiscoveryError,
    MCPToolInvocationError,
)

from .exceptions import (
    AgentConfigurationError,
    AgentExecutionError,
    AgentToolDiscoveryError,
    AgentToolInvocationError,
    AgentToolValidationError,
)
from .state import AgentState
from .validation import validate_tool_call

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful personal productivity assistant. Be concise, practical, and do not claim "
    "actions were performed unless they were."
)


def extract_tool_result_payload(result: Any) -> Any:
    """Extract structured dictionary or text payload from MCP CallToolResult."""
    if result is None:
        return None
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    if getattr(result, "content", None):
        first = result.content[0]
        text = getattr(first, "text", None)
        if text:
            try:
                return json.loads(text)
            except Exception:
                return text
    return result


def parse_llm_tool_selection(content: str) -> tuple[str | None, Any, str | None]:
    """Parse tool proposal from LLM output.

    Returns (tool_name, arguments, direct_response).
    """
    cleaned = content.strip()
    # Handle markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
            if cleaned.lower().startswith("json\n"):
                cleaned = cleaned[5:].strip()

    data = None
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback: search for JSON object block {...}
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError, ValueError):
                data = None

    if isinstance(data, dict):
        tool_name = data.get("tool_name") or data.get("tool")
        if tool_name and isinstance(tool_name, str) and tool_name.strip():
            args = data.get("arguments", {})
            if args is None:
                args = {}
            return tool_name.strip(), args, None
        direct = data.get("response") or data.get("message")
        return None, {}, str(direct) if direct else content

    # Free-form conversational response (no tool selected)
    return None, {}, content


def _normalize_tool_arguments(
    tool_name: str,
    arguments: Any,
    discovered_tools: list[dict[str, Any]],
) -> Any:
    """Safely coerce argument types such as numeric string IDs to integers."""
    if not isinstance(arguments, dict):
        return arguments

    matching_tools = [t for t in discovered_tools if t.get("name") == tool_name]
    if not matching_tools:
        return arguments

    tool_def = matching_tools[0]
    input_schema = tool_def.get("input_schema", {})
    properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}

    normalized = dict(arguments)
    for key, val in arguments.items():
        if key in ("task_id", "event_id", "note_id") and isinstance(val, str) and val.strip().lstrip("-").isdigit():
            try:
                normalized[key] = int(val.strip())
            except ValueError:
                pass
        elif key in properties and isinstance(properties[key], dict):
            prop_type = properties[key].get("type")
            if prop_type == "integer" and isinstance(val, str) and val.strip().lstrip("-").isdigit():
                try:
                    normalized[key] = int(val.strip())
                except ValueError:
                    pass
    return normalized


async def discovery_node(state: AgentState, *, mcp_client: MCPClient | None) -> AgentState:
    """Discover available tools from connected MCP server and populate state."""
    # If tools were pre-populated, retain them
    if state.get("discovered_tools"):
        return state

    if mcp_client is None:
        return {**state, "discovered_tools": []}

    try:
        if not mcp_client.is_connected:
            await mcp_client.connect()
        raw_tools = await mcp_client.list_tools()
    except Exception as exc:
        raise AgentToolDiscoveryError(f"Failed to discover tools from MCP server: {exc}") from exc

    discovered: list[dict[str, Any]] = []
    for tool in raw_tools:
        name = getattr(tool, "name", "")
        description = getattr(tool, "description", "") or ""
        input_schema = getattr(tool, "inputSchema", {}) or {}
        discovered.append({
            "name": name,
            "description": description,
            "input_schema": input_schema,
        })

    return {**state, "discovered_tools": discovered}


def selection_node(
    state: AgentState,
    *,
    llm_client: LLMClient,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> AgentState:
    """Prompt the LLM to either propose an MCP tool call or provide a direct response."""
    if llm_client is None:
        raise AgentConfigurationError("LLM client is required for tool selection.")

    user_message = (state.get("user_request") or state.get("user_message", "")).strip()
    if not user_message:
        raise ValueError("User message must be a non-empty string.")

    discovered_tools = state.get("discovered_tools", [])
    if not discovered_tools:
        # No tools available; standard single-turn LLM generation
        try:
            response = llm_client.generate(user_message, system_message=system_prompt)
        except Exception as exc:
            raise AgentExecutionError("The agent LLM node failed while generating a response.") from exc

        return {
            **state,
            "selected_tool": None,
            "tool_arguments": {},
            "tool_status": "no_tool",
            "final_response": response.content,
            "metadata": {
                **state.get("metadata", {}),
                "model": response.model,
                "provider": llm_client.config.provider,
            },
        }

    # Format tools schema for prompt
    tools_summary = []
    for t in discovered_tools:
        tools_summary.append({
            "name": t.get("name"),
            "description": t.get("description"),
            "parameters": t.get("input_schema", {}),
        })

    reference_time = state.get("metadata", {}).get("reference_time")
    if not reference_time:
        ref_dt = datetime.now()
        reference_time_str = f"{ref_dt.isoformat()} ({ref_dt.strftime('%A')})"
    else:
        reference_time_str = str(reference_time)

    prompt = (
        f"{system_prompt}\n\n"
        f"Reference Context:\n"
        f"- Current Date & Time: {reference_time_str}\n\n"
        "You have access to the following productivity tools:\n"
        f"{json.dumps(tools_summary, indent=2)}\n\n"
        "Instructions for Natural Language Requests:\n"
        "1. Understand the user's intent. If the request corresponds to any available tool, select that tool and extract all required/optional arguments.\n"
        "2. Argument format:\n"
        '   Respond ONLY with a JSON object in the exact format: {"tool_name": "<name>", "arguments": {<args>}}\n'
        "   - Use exact parameter names from the tool parameters schema.\n"
        "   - For task/event/note IDs, provide positive integers (e.g. task_id: 12).\n"
        "   - For dates/times (due_date, start_time, end_time, start_date, end_date), resolve relative expressions like 'today', 'tomorrow', 'Friday' into ISO 8601 strings (YYYY-MM-DDTHH:MM:SS) using the Current Date & Time.\n"
        "   - If scheduling an event (create_event) without an explicit duration, default end_time to 1 hour after start_time.\n"
        "   - Do not invent unsupported arguments.\n"
        "3. If no tool is needed (such as greetings, general questions, or if critical required information like title or date cannot be safely determined), respond conversationally without calling any tool."
    )

    try:
        response = llm_client.generate(user_message, system_message=prompt)
    except Exception as exc:
        raise AgentExecutionError("The agent tool-selection LLM call failed.") from exc

    tool_name, raw_tool_args, direct_resp = parse_llm_tool_selection(response.content)
    tool_args = _normalize_tool_arguments(tool_name, raw_tool_args, discovered_tools) if tool_name else raw_tool_args

    new_metadata = dict(state.get("metadata", {}))
    new_metadata.update({"model": response.model, "provider": llm_client.config.provider})

    if tool_name:
        return {
            **state,
            "selected_tool": tool_name,
            "tool_arguments": tool_args,
            "tool_status": "proposed",
            "tool_error": None,
            "metadata": new_metadata,
        }

    return {
        **state,
        "selected_tool": None,
        "tool_arguments": {},
        "tool_status": "no_tool",
        "final_response": direct_resp or response.content,
        "metadata": new_metadata,
    }


def validation_node(state: AgentState) -> AgentState:
    """Validate proposed tool call against discovered schemas; fail closed if invalid."""
    tool_name = state.get("selected_tool")
    if not tool_name:
        return state

    tool_args = state.get("tool_arguments", {})
    discovered_tools = state.get("discovered_tools", [])

    is_valid, error_msg = validate_tool_call(tool_name, tool_args, discovered_tools)
    if is_valid:
        return {
            **state,
            "tool_status": "validated",
            "tool_error": None,
        }

    return {
        **state,
        "tool_status": "validation_error",
        "tool_error": error_msg,
    }


async def mcp_invocation_node(state: AgentState, *, mcp_client: MCPClient | None) -> AgentState:
    """Execute validated tool call via standard MCP Client."""
    if state.get("tool_status") != "validated":
        return state

    tool_name = state.get("selected_tool")
    tool_args = state.get("tool_arguments", {})

    if not tool_name:
        return state

    if mcp_client is None:
        return {
            **state,
            "tool_status": "connection_error",
            "tool_error": "MCP client is not configured.",
        }

    try:
        if not mcp_client.is_connected:
            await mcp_client.connect()
        result = await mcp_client.call_tool(tool_name, tool_args)
        payload = extract_tool_result_payload(result)
        return {
            **state,
            "tool_result": payload,
            "tool_status": "success",
            "tool_error": None,
        }
    except MCPToolInvocationError as exc:
        msg = str(exc).lower()
        if "not found" in msg:
            status = "not_found"
        elif "conflict" in msg:
            status = "conflict_error"
        elif "validation" in msg or "invalid" in msg:
            status = "validation_error"
        else:
            status = "server_error"
        payload = extract_tool_result_payload(exc.result) if exc.result else None
        return {
            **state,
            "tool_result": payload,
            "tool_status": status,
            "tool_error": str(exc),
        }
    except (MCPConnectionError, MCPClientError) as exc:
        return {
            **state,
            "tool_result": None,
            "tool_status": "connection_error",
            "tool_error": str(exc),
        }
    except Exception as exc:
        return {
            **state,
            "tool_result": None,
            "tool_status": "error",
            "tool_error": str(exc),
        }


def final_response_node(
    state: AgentState,
    *,
    llm_client: LLMClient,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> AgentState:
    """Synthesize final truthful natural-language response using LLM."""
    if llm_client is None:
        raise AgentConfigurationError("LLM client is required for generating final response.")

    user_message = state.get("user_request") or state.get("user_message", "")
    selected_tool = state.get("selected_tool")
    tool_status = state.get("tool_status")
    tool_result = state.get("tool_result")
    tool_error = state.get("tool_error")

    # If no tool was called and final_response was already produced, record message & return
    if not selected_tool and state.get("final_response"):
        conversation = list(state.get("messages", []))
        conversation.append({"role": "assistant", "content": state["final_response"]})
        return {**state, "messages": conversation}

    # Prepare context for synthesis
    status_summary = {
        "tool_requested": selected_tool,
        "arguments": state.get("tool_arguments", {}),
        "execution_status": tool_status,
        "result": tool_result,
        "error": tool_error,
    }

    synthesis_prompt = (
        f"{system_prompt}\n\n"
        "You just executed an action for the user request.\n"
        f"Execution Summary:\n{json.dumps(status_summary, indent=2)}\n\n"
        "Rules:\n"
        "- Provide a concise, clear, and user-friendly natural-language response directly addressing the user request.\n"
        "- NEVER claim an action succeeded if execution_status is not 'success'.\n"
        "- If an operation succeeded:\n"
        "  * For task/event/note creations, confirm creation with the item title and relevant details (e.g. date/time, priority).\n"
        "  * For completions and deletions, confirm the action clearly.\n"
        "  * For listings or searches, summarize returned items concisely in a readable format. If empty, clearly state no items were found.\n"
        "- If an operation failed (validation_error, not_found, conflict_error, server_error, connection_error):\n"
        "  * Clearly explain what failed using plain language (e.g. 'Task 12 was not found', 'A calendar conflict occurred', 'Missing required title').\n"
        "- NEVER expose raw technical stack traces, system paths, internal error codes, or secrets.\n"
        "- Do not hallucinate missing data."
    )

    try:
        response = llm_client.generate(user_message, system_message=synthesis_prompt)
    except Exception as exc:
        raise AgentExecutionError("The agent final response LLM call failed.") from exc

    conversation = list(state.get("messages", []))
    conversation.append({"role": "assistant", "content": response.content})

    metadata = dict(state.get("metadata", {}))
    metadata.update({
        "model": response.model,
        "provider": llm_client.config.provider,
        "tool_status": tool_status,
    })

    return {
        **state,
        "messages": conversation,
        "final_response": response.content,
        "metadata": metadata,
    }


def llm_node(state: AgentState, *, llm_client: LLMClient, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> AgentState:
    """Legacy single-node LLM step preserved for Phase 3.2 backwards compatibility."""
    if llm_client is None:
        raise AgentConfigurationError("LLM client is required for the agent workflow.")

    user_message = (state.get("user_request") or state.get("user_message", "")).strip()
    if not user_message:
        raise ValueError("User message must be a non-empty string.")

    try:
        response = llm_client.generate(user_message, system_message=system_prompt)
    except Exception as exc:
        raise AgentExecutionError("The agent LLM node failed while generating a response.") from exc

    conversation = list(state.get("messages", []))
    conversation.append({"role": "assistant", "content": response.content})

    metadata = dict(state.get("metadata", {}))
    metadata.update({"model": response.model, "provider": llm_client.config.provider})

    return {
        **state,
        "user_request": user_message,
        "user_message": user_message,
        "messages": conversation,
        "final_response": response.content,
        "metadata": metadata,
    }
