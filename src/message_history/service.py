from langchain_community.chat_message_histories import postgres
from langchain_core.messages import HumanMessage, AIMessage
from psycopg.rows import class_row
from uuid import UUID
import psycopg
import json

from .models import MessageMetadata, Message
from .. import env_variables as env


async def session_last_message_id(session_id: str) -> int | None:

    query = """
        SELECT
            m.id
        FROM message_store AS m
        WHERE 
            m.session_id = %s
        ORDER BY id DESC
        LIMIT 1;
    """

    id: int | None = None

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(query, [session_id])
            row = await cursor.fetchone()
            if row:
                id = row[0]

    return id


async def validate_message(message_id: int, session_id: UUID) -> bool:
    result = False
    query = (
        "SELECT EXISTS(SELECT 1 FROM message_store WHERE id = %s AND session_id = %s)"
    )

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(query, (message_id, str(session_id)))

            row = await cursor.fetchone()
            if row:
                result = bool(row[0])

    return result


async def create_review(message_id: int, review: str):
    query = "INSERT INTO message_reviews (message_id, review) VALUES (%s, %s)"

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(query, (message_id, review))
            await connection.commit()


async def fetch_message_metadata(session_id: UUID, message_id: int) -> MessageMetadata:

    message_metadata = MessageMetadata()

    query = """
        SELECT message->'data'->>'additional_kwargs' AS metadata
        FROM message_store
        WHERE session_id = %s AND id = %s;
    """

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor(row_factory=class_row(MessageMetadata)) as cursor:
            await cursor.execute(query, (str(session_id), message_id))
            row = await cursor.fetchone()
            if row:
                message_metadata = row

    return message_metadata


async def fetch_session_history(session_id: str) -> list[Message]:
    messages: list[Message] = []
    query = """
        SELECT 
            ms.id,
            ms.message->>'type' AS role, 
            ms.message->'data'->>'content' AS text,
            CASE 
                WHEN mr.message_id IS NOT NULL 
                THEN 1 ELSE 0
            END AS review,
            ms.message->'data'->>'additional_kwargs' AS metadata
        FROM message_store AS ms
        LEFT JOIN message_reviews AS mr ON ms.id = mr.message_id
        WHERE ms.session_id = %s
        ORDER BY id ASC;
    """

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor(row_factory=class_row(Message)) as cursor:
            await cursor.execute(query, [session_id])
            rows = await cursor.fetchall()
            if rows:
                messages = rows

    return messages


async def add_messages(
    session_id: UUID, user_query: str, output: str, references: dict = {}
) -> None:
    message_store = postgres.PostgresChatMessageHistory(
        connection_string=env.DATABASE_CONN.get_secret_value(),
        session_id=str(session_id),
    )
    messages = [
        HumanMessage(content=user_query),
        AIMessage(content=output, additional_kwargs=references),
    ]
    await message_store.aadd_messages(messages)


async def update_messages(
    session_id: UUID,
    user_query: str,
    output: str,
    ai_message_id: int,
    references: dict = {},
) -> None:

    human_msg_json = json.dumps(
        {"type": "human", "data": HumanMessage(content=user_query).dict()}
    )
    ai_msg_json = json.dumps(
        {
            "type": "ai",
            "data": AIMessage(content=output, additional_kwargs=references).dict(),
        }
    )

    session_id_str = str(session_id)
    messages: list[tuple[str, str, int]] = [
        (human_msg_json, session_id_str, ai_message_id - 1),
        (ai_msg_json, session_id_str, ai_message_id),
    ]

    query = """
        UPDATE message_store
        SET message = %s
        WHERE session_id = %s AND id = %s;
    """

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.executemany(query, messages)
