from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

# Import the ORM models so the metadata is populated before the schema is created.
# This ensures the runtime database is initialized for the task, calendar, and note tables.
from app.database import models  # noqa: F401,E402


def create_database() -> None:
    """Create all database tables required by the application."""
    Base.metadata.create_all(bind=engine)


create_database()


def get_db() -> Generator[Session, None, None]:
    """Provide a transactional database session for each request."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
