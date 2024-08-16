from typing_extensions import Annotated
from pydantic import (
    UUID4,
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
)
from enum import Enum


class PromptRequest(BaseModel):
    session_id: UUID4
    text: Annotated[str, StringConstraints(min_length=1)]
    message_id: int | None = None

    @field_validator("text")
    def check_text_not_whitespace(cls, value: str):
        if value.isspace():
            raise ValueError("The request field `text` cannot contain only whitespace.")
        return value


class SessionSummary(BaseModel):
    session_id: UUID4
    last_message_id: int = 0
    summary: str = ""


class Embedding(BaseModel):
    id: UUID4
    content: str
    metadata: dict = {}
    similarity_score: float | None = Field(default=None, exclude=True)

    @field_validator("similarity_score")
    @classmethod
    def make_positive(cls, v: float | None) -> float | None:
        return abs(v) if v != None else v

    def __setattr__(self, name, value):
        if name == "similarity_score":
            value = abs(value) if value != None else value
        super().__setattr__(name, value)


class QueryType(Enum):
    VALID = 0
    NO_CONTEXT = 1
    RANDOM = 2
    INVALID = 3
