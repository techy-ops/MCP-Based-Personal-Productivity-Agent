from __future__ import annotations

from datetime import datetime, timedelta

from app.database.connection import create_database
from app.services.calendar_service import create_event, list_events
from app.services.note_service import create_note, search_notes
from app.services.task_service import create_task, list_tasks


def demo() -> None:
    """Run a simple backend verification demo using the local SQLite database."""
    create_database()

    task = create_task(
        title="Complete project report",
        description="Finish the first draft",
        priority="high",
        due_date=datetime.now() + timedelta(days=2),
    )
    tasks = list_tasks()

    start_time = datetime.now() + timedelta(days=12, hours=9)
    end_time = datetime.now() + timedelta(days=12, hours=10)
    event = create_event(
        title="Demo planning session",
        description="Sprint review",
        start_time=start_time,
        end_time=end_time,
    )
    events = list_events()

    note = create_note(title="Machine Learning Notes", content="Review gradient descent and model evaluation.")
    matches = search_notes("learning")

    print("Task created:", task.title)
    print("Task count:", len(tasks))
    print("Event created:", event.title)
    print("Event count:", len(events))
    print("Note created:", note.title)
    print("Search results:", [item.title for item in matches])


if __name__ == "__main__":
    demo()
