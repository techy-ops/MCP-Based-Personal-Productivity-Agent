from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database.connection import create_database
from app.services.calendar_service import create_event
from app.services.note_service import create_note
from app.services.task_service import create_task


def seed_database() -> None:
    """Populate the local SQLite database with demo data if it is not already seeded."""
    create_database()

    existing_tasks = list_task_titles()
    if not existing_tasks:
        create_task(title="Complete project report", description="Finish the first draft", priority="high", due_date=datetime.now() + timedelta(days=2))
        create_task(title="Study Machine Learning", description="Review supervised learning notes", status="in_progress", priority="high")
        create_task(title="Prepare presentation", description="Draft slides for the final demo", priority="medium")
        create_task(title="Submit assignment", description="Upload the assignment before Friday", priority="low")

    existing_events = list_event_titles()
    if not existing_events:
        create_event(title="Project meeting", description="Weekly sprint review", start_time=datetime.now() + timedelta(days=1, hours=9), end_time=datetime.now() + timedelta(days=1, hours=10))
        create_event(title="Team discussion", description="Sync on blockers", start_time=datetime.now() + timedelta(days=2, hours=13), end_time=datetime.now() + timedelta(days=2, hours=14))
        create_event(title="Viva preparation", description="Practice answering design questions", start_time=datetime.now() + timedelta(days=3, hours=11), end_time=datetime.now() + timedelta(days=3, hours=12))
        create_event(title="Hackathon review", description="Identify improvements", start_time=datetime.now() + timedelta(days=4, hours=15), end_time=datetime.now() + timedelta(days=4, hours=16))

    existing_notes = list_note_titles()
    if not existing_notes:
        create_note(title="Machine Learning notes", content="Review gradient descent, regularization, and evaluation metrics.")
        create_note(title="MCP architecture notes", content="The backend should expose service functions that can later be wrapped by MCP tools.")
        create_note(title="Python revision notes", content="Use type hints and small, explicit functions to keep code readable.")
        create_note(title="Project ideas", content="Create a productivity assistant that integrates tasks, calendar events, and notes.")

    print("Database seeded successfully.")


def list_task_titles() -> list[str]:
    from app.database.models import Task
    from app.database.connection import SessionLocal

    with SessionLocal() as session:
        return [task.title for task in session.query(Task).all()]


def list_event_titles() -> list[str]:
    from app.database.models import CalendarEvent
    from app.database.connection import SessionLocal

    with SessionLocal() as session:
        return [event.title for event in session.query(CalendarEvent).all()]


def list_note_titles() -> list[str]:
    from app.database.models import Note
    from app.database.connection import SessionLocal

    with SessionLocal() as session:
        return [note.title for note in session.query(Note).all()]


if __name__ == "__main__":
    seed_database()
