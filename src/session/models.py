from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class Hint(BaseModel):
    title: str
    description: str


class Session(BaseModel):
    id: UUID
    title: str | None = None
    created_at: datetime | None = None
    type: str | None = Field(default=None, exclude=True)
    content_id: str | None = Field(default=None, exclude=True)
    hints: list[Hint] | None = None


class UserSessions(BaseModel):
    user_id: int
    sessions: list[Session]
