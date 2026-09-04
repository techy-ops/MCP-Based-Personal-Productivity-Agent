from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import connection as db_connection
import app.services.task_service as task_service
from app.services.task_service import create_task, delete_task, get_task, list_tasks, update_task, complete_task
from app.utils.validators import ValidationError, NotFoundError


@pytest.fixture
def task_db(monkeypatch):
    test_engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
    from app.database.models import Base

    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)

    monkeypatch.setattr(db_connection, "engine", test_engine)
    monkeypatch.setattr(db_connection, "SessionLocal", TestSessionLocal)
    monkeypatch.setattr(task_service, "SessionLocal", TestSessionLocal)
    yield TestSessionLocal


def test_create_task(task_db):
    task = create_task(title="Complete project report", description="Finish the first draft", priority="high")
    assert task.title == "Complete project report"
    assert task.status == "pending"
    assert task.priority == "high"
    assert task.id is not None


def test_get_task(task_db):
    created = create_task(title="Study ML", priority="medium")
    fetched = get_task(created.id)
    assert fetched.title == "Study ML"


def test_list_tasks(task_db):
    create_task(title="Task 1", priority="low")
    create_task(title="Task 2", priority="high")
    tasks = list_tasks()
    assert len(tasks) >= 2


def test_update_task(task_db):
    task = create_task(title="Old title", description="Old desc", priority="low")
    updated = update_task(task.id, title="New title", priority="high")
    assert updated.title == "New title"
    assert updated.priority == "high"


def test_complete_task(task_db):
    task = create_task(title="Prepare presentation", priority="medium")
    completed = complete_task(task.id)
    assert completed.status == "completed"


def test_delete_task(task_db):
    task = create_task(title="Delete me", priority="low")
    result = delete_task(task.id)
    assert result is True
    with pytest.raises(NotFoundError):
        get_task(task.id)


def test_filter_tasks_by_status(task_db):
    create_task(title="A", priority="low")
    create_task(title="B", priority="medium", status="in_progress")
    tasks = list_tasks(status="in_progress")
    assert all(task.status == "in_progress" for task in tasks)


def test_filter_tasks_by_priority(task_db):
    create_task(title="A", priority="high")
    create_task(title="B", priority="low")
    tasks = list_tasks(priority="high")
    assert all(task.priority == "high" for task in tasks)


def test_invalid_task_status(task_db):
    with pytest.raises(ValidationError):
        create_task(title="Bad status", status="blocked")


def test_invalid_task_priority(task_db):
    with pytest.raises(ValidationError):
        create_task(title="Bad priority", priority="urgent")


def test_missing_task(task_db):
    with pytest.raises(NotFoundError):
        get_task(999)


def test_task_due_date_validation(task_db):
    due_date = datetime.now() + timedelta(days=1)
    task = create_task(title="Due task", due_date=due_date, priority="medium")
    assert task.due_date == due_date
