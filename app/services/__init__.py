"""Service layer for the productivity backend."""

from .calendar_service import create_event, delete_event, get_event, list_events, update_event
from .note_service import create_note, delete_note, get_note, list_notes, search_notes, update_note
from .task_service import complete_task, create_task, delete_task, get_task, list_tasks, update_task

__all__ = [
    "create_task",
    "get_task",
    "list_tasks",
    "update_task",
    "complete_task",
    "delete_task",
    "create_event",
    "get_event",
    "list_events",
    "update_event",
    "delete_event",
    "create_note",
    "get_note",
    "list_notes",
    "update_note",
    "delete_note",
    "search_notes",
]
