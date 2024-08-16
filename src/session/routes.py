from fastapi import APIRouter, HTTPException, Response

from ..text_generation.service import validate_collection
from .models import Session, UserSessions
from . import service


router = APIRouter(prefix="session", tags=["Session"])


@router.get("/start/{user_id}/{session_type}")
async def start_session(
    user_id: int, session_type: str, content_id: str | None = None
) -> Session:
    if session_type != "chat" and content_id is None:
        raise HTTPException(
            400,
            detail=f"`content_id` is required for sessions of the type `{session_type}`.",
        )

    valid_collection = await validate_collection(session_type, str(content_id))
    if not valid_collection:
        raise HTTPException(
            404,
            detail=f"`content_id` {content_id} not found for the `session_type` {session_type}.",
        )

    session = await service.begin_session(user_id, session_type, content_id)
    return session


@router.get("/list-user-sessions/{user_id}")
async def get_user_sessions(user_id: int) -> UserSessions:
    sessions = await service.fetch_user_sessions(user_id)
    return UserSessions(user_id=user_id, sessions=sessions)


@router.delete("/delete/{session_id}")
async def delete_chat_session(session_id: str) -> Response:
    await service.delete_session(session_id)
    return Response(status_code=200)
