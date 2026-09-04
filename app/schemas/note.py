from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NoteCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if value is None or not value.strip():
            raise ValueError("title cannot be empty")
        return value.strip()

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if value is None or not value.strip():
            raise ValueError("content cannot be empty")
        return value.strip()


class NoteUpdate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    title: str | None = None
    content: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("title cannot be empty")
        return value.strip() if value is not None else None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("content cannot be empty")
        return value.strip() if value is not None else None


class NoteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
