from __future__ import annotations

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


def parse_llm_tool_selection(content: str) -> tuple[str | None, dict[str, Any], str | None]:
    """Parse tool proposal from LLM output.

    Returns (tool_name, arguments, direct_response).
    """
    cleaned = content.strip()
    # Handle markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            tool_name = data.get("tool_name")
            if tool_name and isinstance(tool_name, str) and tool_name.strip():
                args = data.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                return tool_name.strip(), args, None
            # If tool_name is explicitly None or missing
            direct = data.get("response") or data.get("message")
            return None, {}, str(direct) if direct else content
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Free-form conversational response (no tool selected)
    return None, {}, content


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

    user_message = state.get("user_message", "").strip()
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

    prompt = (
        f"{system_prompt}\n\n"
        "You have access to the following tools:\n"
        f"{json.dumps(tools_summary, indent=2)}\n\n"
        "Instructions:\n"
        "- If an available tool should be called to address the user request, respond ONLY with a JSON object "
        'in the exact format: {"tool_name": "<name>", "arguments": {<args>}}\n'
        "- If no tool is needed (such as greetings, general questions, or if required info is completely missing), "
        "respond conversationally without calling any tool."
    )

    try:
        response = llm_client.generate(user_message, system_message=prompt)
    except Exception as exc:
        raise AgentExecutionError("The agent tool-selection LLM call failed.") from exc

    tool_name, tool_args, direct_resp = parse_llm_tool_selection(response.content)

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

    user_message = state.get("user_message", "")
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
        "- Clearly explain the outcome to the user.\n"
        "- NEVER claim an action succeeded if execution_status is not 'success'.\n"
        "- If an operation failed, explain the failure accurately based on the error.\n"
        "- Do not expose internal technical stack traces, system paths, or secrets.\n"
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

    user_message = state.get("user_message", "").strip()
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
        "user_message": user_message,
        "messages": conversation,
        "final_response": response.content,
        "metadata": metadata,
    }
