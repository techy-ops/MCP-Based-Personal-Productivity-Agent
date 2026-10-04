from __future__ import annotations

from datetime import datetime
from typing import Any

VALID_TASK_STATUSES = {"pending", "in_progress", "completed"}
VALID_TASK_PRIORITIES = {"low", "medium", "high"}


def validate_tool_call(
    tool_name: str | None,
    arguments: Any,
    discovered_tools: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    """Validate a proposed tool call against dynamically discovered MCP tool definitions.

    Enforces:
    1. Tool name exists among discovered tools.
    2. Arguments are a dictionary.
    3. Required schema parameters are provided.
    4. Argument types match the JSON schema definitions (including anyOf unions).
    5. Disallows unknown arguments when the tool schema defines strict properties.
    6. Ensures non-empty strings for required text parameters.
    7. Validates positive integers for ID fields.
    8. Validates ISO 8601 formatting and chronological order for date/time parameters.
    9. Validates task status and priority enum values.
    """
    if not tool_name or not isinstance(tool_name, str) or not tool_name.strip():
        return False, "Tool name must be a non-empty string."

    matching_tools = [t for t in discovered_tools if t.get("name") == tool_name]
    if not matching_tools:
        return False, f"Tool '{tool_name}' is not among the available discovered tools."

    tool_def = matching_tools[0]
    input_schema = tool_def.get("input_schema", {})
    if not isinstance(input_schema, dict):
        input_schema = {}

    if not isinstance(arguments, dict):
        return False, f"Tool arguments for '{tool_name}' must be provided as a dictionary/object."

    # Check required properties from schema
    required_fields = list(input_schema.get("required", []))
    # Domain-specific essential required parameters
    if tool_name == "create_event":
        for req_prop in ("start_time", "end_time"):
            if req_prop not in required_fields:
                required_fields.append(req_prop)
    elif tool_name == "create_note":
        for req_prop in ("title", "content"):
            if req_prop not in required_fields:
                required_fields.append(req_prop)
    elif tool_name == "search_notes":
        if "query" not in required_fields:
            required_fields.append("query")

    for req in required_fields:
        if req not in arguments:
            return False, f"Missing required argument '{req}' for tool '{tool_name}'."
        val = arguments[req]
        if val is None:
            return False, f"Required argument '{req}' cannot be null for tool '{tool_name}'."
        if isinstance(val, str) and not val.strip():
            return False, f"Required argument '{req}' cannot be empty for tool '{tool_name}'."

    # Check update tools have at least one field to update
    if tool_name == "update_task":
        updatable = {"title", "description", "status", "priority", "due_date"}
        if not any(k in arguments for k in updatable):
            return False, "At least one task field must be provided to update_task."
    elif tool_name == "update_event":
        updatable = {"title", "description", "start_time", "end_time", "location"}
        if not any(k in arguments for k in updatable):
            return False, "At least one event field must be provided to update_event."
    elif tool_name == "update_note":
        updatable = {"title", "content"}
        if not any(k in arguments for k in updatable):
            return False, "At least one note field must be provided to update_note."

    properties = input_schema.get("properties", {})
    if isinstance(properties, dict):
        # Disallow unknown arguments if properties are defined
        for arg_key in arguments:
            if properties and arg_key not in properties:
                return False, f"Unsupported argument '{arg_key}' provided for tool '{tool_name}'."

        type_mapping = {
            "string": (str,),
            "integer": (int,),
            "number": (int, float),
            "boolean": (bool,),
            "array": (list, tuple),
            "object": (dict,),
        }

        for arg_key, arg_val in arguments.items():
            if arg_val is None:
                continue

            prop_spec = properties.get(arg_key, {})
            if isinstance(prop_spec, dict):
                # Check for positive integer IDs
                if arg_key in ("task_id", "event_id", "note_id"):
                    if isinstance(arg_val, bool) or not isinstance(arg_val, int) or arg_val <= 0:
                        return False, f"Argument '{arg_key}' must be a positive integer."

                expected_type_str = prop_spec.get("type")
                if expected_type_str:
                    expected_types = type_mapping.get(expected_type_str)
                    if expected_types:
                        if expected_type_str in ("integer", "number") and isinstance(arg_val, bool):
                            return False, f"Argument '{arg_key}' expected {expected_type_str}, received boolean."
                        if not isinstance(arg_val, expected_types):
                            actual_type = type(arg_val).__name__
                            return False, f"Argument '{arg_key}' expected {expected_type_str}, received {actual_type}."
                elif "anyOf" in prop_spec and isinstance(prop_spec["anyOf"], list):
                    allowed_type_names = []
                    for opt in prop_spec["anyOf"]:
                        if isinstance(opt, dict) and "type" in opt:
                            allowed_type_names.append(opt["type"])
                    if allowed_type_names:
                        if isinstance(arg_val, bool) and "boolean" not in allowed_type_names and ("integer" in allowed_type_names or "number" in allowed_type_names):
                            return False, f"Argument '{arg_key}' expected integer, received boolean."
                        type_tuples = tuple(
                            t
                            for tname in allowed_type_names
                            if tname in type_mapping
                            for t in type_mapping[tname]
                        )
                        if type_tuples and not isinstance(arg_val, type_tuples):
                            actual_type = type(arg_val).__name__
                            return False, f"Argument '{arg_key}' expected {allowed_type_names[0]}, received {actual_type}."

            # Date/time format validation for date-time fields
            if arg_key in ("start_time", "end_time", "due_date", "start_date", "end_date"):
                if isinstance(arg_val, str):
                    try:
                        datetime.fromisoformat(arg_val.replace("Z", "+00:00"))
                    except (ValueError, TypeError):
                        return False, f"Argument '{arg_key}' must be a valid ISO 8601 date/time string."

            # Enum values validation
            if arg_key == "status" and isinstance(arg_val, str):
                if arg_val.strip() not in VALID_TASK_STATUSES:
                    return False, f"status must be one of: {', '.join(sorted(VALID_TASK_STATUSES))}"
            if arg_key == "priority" and isinstance(arg_val, str):
                if arg_val.strip() not in VALID_TASK_PRIORITIES:
                    return False, f"priority must be one of: {', '.join(sorted(VALID_TASK_PRIORITIES))}"

    # Cross-field chronological checks for calendar events
    if tool_name in ("create_event", "update_event"):
        st_val = arguments.get("start_time")
        et_val = arguments.get("end_time")
        if isinstance(st_val, str) and isinstance(et_val, str):
            try:
                st_dt = datetime.fromisoformat(st_val.replace("Z", "+00:00"))
                et_dt = datetime.fromisoformat(et_val.replace("Z", "+00:00"))
                if et_dt <= st_dt:
                    return False, "Argument 'end_time' must be after 'start_time'."
            except (ValueError, TypeError):
                pass

    if tool_name == "list_events":
        sd_val = arguments.get("start_date")
        ed_val = arguments.get("end_date")
        if isinstance(sd_val, str) and isinstance(ed_val, str):
            try:
                sd_dt = datetime.fromisoformat(sd_val.replace("Z", "+00:00"))
                ed_dt = datetime.fromisoformat(ed_val.replace("Z", "+00:00"))
                if ed_dt < sd_dt:
                    return False, "Argument 'end_date' must be at or after 'start_date'."
            except (ValueError, TypeError):
                pass

    return True, None
