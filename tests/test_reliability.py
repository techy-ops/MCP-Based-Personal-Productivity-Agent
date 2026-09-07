from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import connection as db_connection
from app.database.models import Base, Task
import app.services.task_service as task_service
from app.services.task_service import create_task, get_task
from app.utils.validators import DatabaseError


@pytest.fixture
def reliability_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(db_connection, "engine", engine)
    monkeypatch.setattr(db_connection, "SessionLocal", session_factory)
    monkeypatch.setattr(task_service, "SessionLocal", session_factory)
    yield session_factory


def test_failed_create_rolls_back_and_does_not_persist_partial_state(reliability_db, monkeypatch):
    session = reliability_db()
    rollback = Mock(wraps=session.rollback)

    def fail_commit():
        raise RuntimeError("controlled commit failure")

    session.commit = fail_commit
    session.rollback = rollback
    monkeypatch.setattr(task_service, "SessionLocal", lambda: session)

    with pytest.raises(DatabaseError, match="Unable to create task"):
        create_task(title="Should not persist")

    rollback.assert_called_once()
    session.close()

    verify = reliability_db()
    try:
        assert verify.execute(select(Task)).scalars().all() == []
    finally:
        verify.close()


def test_failed_update_preserves_previous_state(reliability_db, monkeypatch):
    task = create_task(title="Original title")
    session = reliability_db()

    def fail_commit():
        raise RuntimeError("controlled commit failure")

    session.commit = fail_commit
    monkeypatch.setattr(task_service, "SessionLocal", lambda: session)

    with pytest.raises(DatabaseError, match="Unable to update task"):
        task_service.update_task(task.id, title="Uncommitted title")

    session.close()
    monkeypatch.setattr(task_service, "SessionLocal", reliability_db)
    restored = get_task(task.id)
    assert restored.title == "Original title"