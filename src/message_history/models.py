from pydantic import BaseModel, Field
from uuid import UUID


class MetadataRequest(BaseModel):
    session_id: UUID
    message_id: int


class ReviewRequest(BaseModel):
    session_id: UUID
    message_id: int
    text: str


class Image(BaseModel):
    url: str
    description: str


class OriginData(BaseModel):
    material_id: int
    title: str
    position: str
    source: str
    year: int


class EmbeddingScore(BaseModel):
    embedding_id: UUID
    similarity_score: float


class MessageMetadata(BaseModel):
    references: list[OriginData] = []
    images: list[Image] = []
    embeddings_with_score: list[EmbeddingScore] = Field(default=[], exclude=True)


class Message(BaseModel):
    id: int
    role: str
    text: str
    review: bool = False
    metadata: MessageMetadata = MessageMetadata()


class SessionHistory(BaseModel):
    session_id: str
    messages: list[Message]
