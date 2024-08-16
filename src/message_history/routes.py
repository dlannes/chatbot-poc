from fastapi import APIRouter, HTTPException, Response

from .models import SessionHistory, MetadataRequest, MessageMetadata, ReviewRequest
from . import service

router = APIRouter(prefix="message", tags=["Message History"])


@router.get("/session-last-message-id/{session_id}")
async def get_last_message(session_id: str) -> int:
    id = await service.session_last_message_id(session_id)
    if not id:
        raise HTTPException(
            status_code=404,
            detail=f"No message was found for the `session_id` {session_id}",
        )
    return id


@router.get("/get-session-history/{session_id}")
async def get_history(session_id: str) -> SessionHistory:
    messages = await service.fetch_session_history(session_id)
    return SessionHistory(session_id=session_id, messages=messages)


@router.post("/msg-references/")
async def message_references(data: MetadataRequest) -> MessageMetadata:
    metadata = await service.fetch_message_metadata(data.session_id, data.message_id)
    return metadata


@router.post("/message-review/")
async def store_message_review(data: ReviewRequest) -> Response:
    valid_message = await service.validate_message(data.message_id, data.session_id)
    if not valid_message:
        raise HTTPException(
            status_code=400,
            detail=f"`Message_id` {data.message_id} not found for session {data.session_id}.",
        )

    await service.create_review(data.message_id, data.text)
    return Response(status_code=200)
