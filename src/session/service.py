from psycopg.rows import class_row
from uuid import UUID
import psycopg
import uuid

from ..exceptions import DBEntityNotFoundException
from .. import env_variables as env
from .models import Session, Hint


async def begin_session(
    user_id: int, session_type: str, content_id: str | None
) -> Session:
    session_id = None
    hints: list[Hint] = []

    match session_type:
        case "chat":
            session_id = await _find_empty_chat_session(user_id)

        case _ if content_id:
            session_id = await _find_user_collection_session(
                user_id, session_type, content_id
            )

    if not session_id:
        session_id = await _create_session(user_id, session_type, content_id)

    return Session(id=session_id, hints=hints)


async def _create_session(
    user_id: int, session_type: str, content_id: str | None
) -> UUID:

    new_session_query = """
        INSERT INTO sessions (id, user_id, type, content_id) VALUES (%s, %s, %s, %s);
    """

    session_id = uuid.uuid4()

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                new_session_query, (session_id, user_id, session_type, content_id)
            )

    return session_id


async def get_session(session_id: UUID) -> Session | None:
    query = """
        SELECT id, title, created_at, type, content_id 
        FROM sessions WHERE id = %s;
    """

    session: Session | None = None

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor(row_factory=class_row(Session)) as cursor:
            await cursor.execute(query, [session_id])
            row = await cursor.fetchone()
            if row:
                session = row

    return session


async def validate_session(session_id: UUID) -> bool:
    query = "SELECT EXISTS(SELECT 1 FROM sessions WHERE id = %s);"

    is_valid = False

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(query, [session_id])
            row = await cursor.fetchone()
            if row:
                is_valid = row[0]

    return is_valid


async def _find_user_collection_session(
    user_id: int, session_type: str, content_id: str
) -> UUID | None:

    query = """
        SELECT id FROM sessions 
        WHERE user_id = %s AND type = %s AND content_id = %s;
    """

    session_id: UUID | None = None

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(query, (user_id, session_type, content_id))
            row = await cursor.fetchone()
            if row:
                session_id = row[0]

    return session_id


async def _find_empty_chat_session(user_id: int) -> UUID | None:

    query = """
        SELECT s.id FROM sessions AS s
        LEFT JOIN message_store AS m ON s.id::text = m.session_id
        WHERE s.user_id = %s
        AND m.message is NULL
        AND s.type = 'chat'
        ORDER BY s.created_at ASC
        LIMIT 1
    """

    session_id: UUID | None = None

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(query, [user_id])
            row = await cursor.fetchone()
            if row:
                session_id = row[0]

    return session_id


async def fetch_user_sessions(user_id: int) -> list[Session]:
    sessions: list[Session] = []
    query = """
        SELECT
            s.id,
            COALESCE(s.title, '') AS title,
            COALESCE(MAX(m.created_at), s.created_at) AS created_at
        FROM sessions AS s
        LEFT JOIN message_store AS m ON s.id::text = m.session_id
        WHERE s.deleted IS NOT TRUE AND s.user_id = %s
        GROUP BY s.id, s.title, s.created_at
        ORDER BY created_at DESC;
    """

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor(row_factory=class_row(Session)) as cursor:
            await cursor.execute(query, [user_id])
            sessions = await cursor.fetchall()

    return sessions


async def delete_session(session_id: str) -> None:
    query = "UPDATE sessions SET deleted = TRUE WHERE id = %s;"

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(query, [session_id])
            if cursor.rowcount == 0:
                raise DBEntityNotFoundException("Sessions", "id", session_id)
