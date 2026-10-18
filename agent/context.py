from __future__ import annotations

import re
from typing import Any

# Maximum number of history messages retained to handle context limits safely
MAX_CONTEXT_HISTORY_MESSAGES = 10


def sanitize_untrusted_data(text: str) -> str:
    """Sanitize untrusted text (user data, notes, tool outputs) before embedding into prompts.

    Neutralizes prompt injection delimiters and wraps content as passive data.
    """
    if not isinstance(text, str):
        return str(text)
    # Neutralize common prompt injection markers
    sanitized = text.replace("```system", "'''system")
    sanitized = sanitized.replace("<system>", "[system]")
    sanitized = sanitized.replace("</system>", "[/system]")
    sanitized = sanitized.replace("<instruction>", "[instruction]")
    sanitized = sanitized.replace("</instruction>", "[/instruction]")
    return sanitized


def extract_context_entities(
    steps: list[dict[str, Any]] | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Extract known productivity entities (tasks, events, notes) from steps and history.

    Returns:
        {
            "tasks": [{"id": int, "title": str}],
            "events": [{"id": int, "title": str}],
            "notes": [{"id": int, "title": str}],
        }
    """
    entities: dict[str, list[dict[str, Any]]] = {
        "tasks": [],
        "events": [],
        "notes": [],
    }
    seen_tasks: set[int] = set()
    seen_events: set[int] = set()
    seen_notes: set[int] = set()

    # 1. Extract from execution steps
    for step in steps or []:
        tool = step.get("tool_name") or ""
        status = step.get("tool_status")
        result = step.get("tool_result")
        args = step.get("tool_arguments") or {}

        # Successfully executed tools
        if status == "success" and isinstance(result, dict):
            data = result.get("data")
            if isinstance(data, dict):
                item_id = data.get("id") or data.get("task_id") or data.get("event_id") or data.get("note_id")
                if isinstance(item_id, int):
                    title = data.get("title") or args.get("title") or f"Item {item_id}"
                    if "task" in tool and item_id not in seen_tasks:
                        entities["tasks"].append({"id": item_id, "title": str(title)})
                        seen_tasks.add(item_id)
                    elif "event" in tool and item_id not in seen_events:
                        entities["events"].append({"id": item_id, "title": str(title)})
                        seen_events.add(item_id)
                    elif "note" in tool and item_id not in seen_notes:
                        entities["notes"].append({"id": item_id, "title": str(title)})
                        seen_notes.add(item_id)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        iid = item.get("id")
                        if isinstance(iid, int):
                            t = str(item.get("title") or f"Item {iid}")
                            if "task" in tool and iid not in seen_tasks:
                                entities["tasks"].append({"id": iid, "title": t})
                                seen_tasks.add(iid)
                            elif "event" in tool and iid not in seen_events:
                                entities["events"].append({"id": iid, "title": t})
                                seen_events.add(iid)
                            elif "note" in tool and iid not in seen_notes:
                                entities["notes"].append({"id": iid, "title": t})
                                seen_notes.add(iid)

    # 2. Extract from conversation history texts
    for msg in conversation_history or []:
        content = msg.get("content")
        if not isinstance(content, str):
            continue

        # Task matches e.g. "task 'Review' (ID 12)" or "task ID: 12" or "task 12"
        for match in re.finditer(r"(?:task|Task)\s+(?:'([^']+)'\s*)?(?:\(?(?:id|ID)?\s*[:#]?\s*(\d+)\)?)", content):
            title = match.group(1) or ""
            tid_str = match.group(2)
            if tid_str and tid_str.isdigit():
                tid = int(tid_str)
                if tid not in seen_tasks:
                    entities["tasks"].append({"id": tid, "title": title or f"Task {tid}"})
                    seen_tasks.add(tid)

        # Note matches e.g. "note 'Design' (ID 5)" or "note ID: 5"
        for match in re.finditer(r"(?:note|Note)\s+(?:'([^']+)'\s*)?(?:\(?(?:id|ID)?\s*[:#]?\s*(\d+)\)?)", content):
            title = match.group(1) or ""
            nid_str = match.group(2)
            if nid_str and nid_str.isdigit():
                nid = int(nid_str)
                if nid not in seen_notes:
                    entities["notes"].append({"id": nid, "title": title or f"Note {nid}"})
                    seen_notes.add(nid)

        # Event matches e.g. "event 'Meeting' (ID 3)" or "event ID: 3"
        for match in re.finditer(r"(?:event|Event|meeting|Meeting)\s+(?:'([^']+)'\s*)?(?:\(?(?:id|ID)?\s*[:#]?\s*(\d+)\)?)", content):
            title = match.group(1) or ""
            eid_str = match.group(2)
            if eid_str and eid_str.isdigit():
                eid = int(eid_str)
                if eid not in seen_events:
                    entities["events"].append({"id": eid, "title": title or f"Event {eid}"})
                    seen_events.add(eid)

    return entities


def detect_anaphoric_reference(user_message: str) -> dict[str, bool]:
    """Detect whether user message contains anaphoric/pronoun references to entities."""
    lower = user_message.lower()
    return {
        "task": bool(re.search(r"\b(that task|the task|this task|it|that one)\b", lower) and ("task" in lower or "complete" in lower or "finish" in lower)),
        "note": bool(re.search(r"\b(that note|the note|this note|it|that one)\b", lower) and "note" in lower),
        "event": bool(re.search(r"\b(that event|the event|that meeting|the meeting|it)\b", lower) and ("event" in lower or "meeting" in lower)),
    }


def format_context_for_prompt(
    conversation_history: list[dict[str, Any]] | None,
    entities: dict[str, list[dict[str, Any]]] | None = None,
    max_messages: int = MAX_CONTEXT_HISTORY_MESSAGES,
) -> str:
    """Format sanitized conversation history and discovered entities for prompt inclusion."""
    history = list(conversation_history or [])
    if len(history) > max_messages:
        history = history[-max_messages:]

    lines: list[str] = []
    if history:
        lines.append("Prior Conversation Context (DATA ONLY):")
        for msg in history:
            role = (msg.get("role") or "user").capitalize()
            content = sanitize_untrusted_data(msg.get("content") or "")
            lines.append(f"  [{role}]: {content}")

    if entities:
        ent_lines: list[str] = []
        for domain, items in entities.items():
            if items:
                formatted_items = [f"ID {item['id']}: \"{sanitize_untrusted_data(item['title'])}\"" for item in items]
                ent_lines.append(f"  - {domain.capitalize()}: {', '.join(formatted_items)}")
        if ent_lines:
            lines.append("Referenced Entities in Context:")
            lines.extend(ent_lines)

    return "\n".join(lines)
