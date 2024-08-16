from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException
from models import PromptRequest, QueryType
import service
from ..session.service import get_session, validate_session

router = APIRouter(prefix="generate", tags=["Text Generation"])


@router.post("/stream-answer/")
async def stream_answer(data: PromptRequest) -> StreamingResponse:

    session = await get_session(data.session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid Session.")

    query_type = await service.classify_user_query(data.text)
    match query_type:
        case QueryType.INVALID | QueryType.RANDOM:
            result = service.generate_invalid_query_response(data.session_id, data.text)
        case QueryType.VALID | QueryType.NO_CONTEXT:
            result = service.create_stream_response(session, data.text, query_type)

    return StreamingResponse(result, media_type="text/event-stream")


@router.post("/update-answer/")
async def update_answer(data: PromptRequest) -> StreamingResponse:
    if not data.message_id:
        raise HTTPException(
            status_code=422,
            detail="Field `message_id` is not a valid integer.",
        )

    session = await get_session(data.session_id)
    if not session:
        raise HTTPException(status_code=400, detail="Invalid Session.")

    query_type = await service.classify_user_query(data.text)
    match query_type:
        case QueryType.INVALID | QueryType.RANDOM:
            result = service.generate_invalid_query_response(data.session_id, data.text)
        case QueryType.VALID | QueryType.NO_CONTEXT:
            result = service.update_stream_response(
                session,
                data.text,
                query_type,
                data.message_id,
            )

    return StreamingResponse(result, media_type="text/event-stream")


@router.post("/session-title/")
async def generate_chat_title(data: PromptRequest) -> str:
    if data.text.isspace() or not data.text:
        raise HTTPException(
            status_code=400, detail="The request field `text` is empty."
        )

    valid_session = await validate_session(data.session_id)
    if not valid_session:
        raise HTTPException(status_code=400, detail="Invalid Session.")

    title = await service.generate_title(data.session_id, data.text)
    return title
