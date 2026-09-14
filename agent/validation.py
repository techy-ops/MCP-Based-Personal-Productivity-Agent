from __future__ import annotations

from typing import Any


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
    4. Argument types match the JSON schema definitions.
    5. Disallows unknown arguments when the tool schema defines strict properties.
    6. Ensures non-empty strings for required text parameters.
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

    # Check required properties
    required_fields = input_schema.get("required", [])
    if isinstance(required_fields, list):
        for req in required_fields:
            if req not in arguments:
                return False, f"Missing required argument '{req}' for tool '{tool_name}'."
            val = arguments[req]
            if val is None:
                return False, f"Required argument '{req}' cannot be null for tool '{tool_name}'."
            if isinstance(val, str) and not val.strip():
                return False, f"Required argument '{req}' cannot be empty for tool '{tool_name}'."

    properties = input_schema.get("properties", {})
    if isinstance(properties, dict):
        # Disallow unknown arguments if properties are defined
        for arg_key in arguments:
            if arg_key not in properties:
                return False, f"Unsupported argument '{arg_key}' provided for tool '{tool_name}'."

        # Type validation
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
            if isinstance(prop_spec, dict) and "type" in prop_spec:
                expected_type_str = prop_spec["type"]
                expected_types = type_mapping.get(expected_type_str)
                if expected_types:
                    # In Python, bool is a subclass of int, so explicitly disallow bool for integer/number
                    if expected_type_str in ("integer", "number") and isinstance(arg_val, bool):
                        return False, f"Argument '{arg_key}' expected {expected_type_str}, received boolean."
                    if not isinstance(arg_val, expected_types):
                        actual_type = type(arg_val).__name__
                        return False, f"Argument '{arg_key}' expected {expected_type_str}, received {actual_type}."

    return True, None
