from langchain_community.vectorstores.sklearn import SKLearnVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import AsyncIterable, Optional
from psycopg.rows import class_row
from openai import AsyncOpenAI
from uuid import UUID
import psycopg
import json

from ..message_history.models import Image, Message, MessageMetadata, OriginData
from ..message_history.service import add_messages, update_messages
from .models import QueryType, Embedding, SessionSummary
from ..session.models import Session
from .. import env_variables as env


openai_client = AsyncOpenAI(api_key=env.OPENAI_API_KEY.get_secret_value())


async def validate_collection(session_type: str, content_id: str) -> bool:
    if session_type == "chat":
        return True

    query = "SELECT EXISTS(SELECT 1 FROM langchain_pg_collection WHERE name = %s);"

    is_valid = False
    collection = _format_collection_name(session_type, content_id)

    async with await psycopg.AsyncConnection.connect(
        env.VECTOR_DB_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(query, [collection])
            exists = await cursor.fetchone()
            if exists:
                is_valid = exists[0]

    return is_valid


async def classify_user_query(user_query: str) -> QueryType:
    prompt_template = ChatPromptTemplate.from_template(
        env.prompt_templates["classify_query"]
    )
    model = ChatOpenAI(
        api_key=env.OPENAI_API_KEY, model="gpt-4o-2024-08-06", temperature=0.5
    )
    chain = (
        {"user_query": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

    query_type = QueryType.INVALID

    tries = 3
    while tries > 0:
        result = await chain.ainvoke({"user_query": user_query})
        try:
            query_type_value = int(result)
            query_type = QueryType(query_type_value)
            tries = 0
        except:
            tries -= 1

    return query_type


async def create_stream_response(
    session: Session, user_query: str, query_type: QueryType
) -> AsyncIterable[str]:
    async for chunk in stream_response(session, user_query, query_type):
        yield chunk


async def update_stream_response(
    session: Session, user_query: str, query_type: QueryType, ai_message_id: int
) -> AsyncIterable[str]:
    async for chunk in stream_response(
        session, user_query, query_type, update=True, ai_message_id=ai_message_id
    ):
        yield chunk


async def stream_response(
    session: Session,
    user_query: str,
    query_type: QueryType,
    update: bool = False,
    ai_message_id: Optional[int] = None,
) -> AsyncIterable[str]:

    embeddings: list[Embedding] = []
    session_summary, messages = await _fetch_session_summary_data(session.id)
    messages_prompt, embedding_ids = await _get_session_context(
        session_summary, messages
    )

    if embedding_ids:
        previous_embeddings = await _fetch_embeddings(embedding_ids)
        embeddings.extend(previous_embeddings)

    if query_type == QueryType.VALID:
        query_embeddings = await _search_embeddings(session, user_query)
        embeddings.extend(query_embeddings)

    response_chunks = []
    async for chunk in _generate_response(user_query, embeddings, messages_prompt):
        response_chunks.append(chunk)
        yield chunk

    output = "".join(response_chunks)
    references = _top_documents_for_output(embeddings, output)

    if not update:
        await add_messages(session.id, user_query, output, references)
    elif ai_message_id:
        await update_messages(session.id, user_query, output, ai_message_id, references)


async def _generate_response(
    user_query: str, embeddings: list[Embedding], previous_messages: str
) -> AsyncIterable[str]:

    prompt_template = ChatPromptTemplate.from_template(env.prompt_templates["answer"])
    model = ChatOpenAI(
        api_key=env.OPENAI_API_KEY,
        model="gpt-4o-2024-08-06",
        temperature=0.7,
        streaming=True,
        callbacks=[AsyncCallbackHandler()],
    )
    chain = (
        {
            "context": RunnablePassthrough(),
            "history": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | model
        | StrOutputParser()
    )

    context = "\n\n".join(emb.content for emb in embeddings)

    async for chunk in chain.astream(
        {
            "context": context,
            "history": previous_messages,
            "question": user_query,
        }
    ):
        yield chunk


async def generate_invalid_query_response(
    session_id: UUID, user_query: str
) -> AsyncIterable[str]:
    prompt_template = ChatPromptTemplate.from_template(
        env.prompt_templates["invalid_query"]
    )
    chunks: list[str] = []

    model = ChatOpenAI(
        api_key=env.OPENAI_API_KEY,
        model="gpt-4o-2024-08-06",
        temperature=0.5,
        streaming=True,
    )
    chain = (
        {"user_query": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

    try:
        async for chunk in chain.astream({"user_query": user_query}):
            chunks.append(chunk)
            yield chunk
    except:
        raise

    output = "".join(chunks)
    await add_messages(session_id, user_query, output)


async def generate_title(session_id: UUID, question: str) -> str:
    model = ChatOpenAI(
        api_key=env.OPENAI_API_KEY, model="gpt-4o-2024-08-06", temperature=0.7
    )
    prompt_template = ChatPromptTemplate.from_template(env.prompt_templates["title"])

    chain = (
        {
            "question": RunnablePassthrough(),
        }
        | prompt_template
        | model
        | StrOutputParser()
    )

    generated_title = await chain.ainvoke({"question": question})

    title = ""
    query = "UPDATE sessions SET title = %s WHERE ID = %s RETURNING title;"
    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(query, (generated_title, session_id))
            row = await cursor.fetchone()
            if row:
                title = row[0]

    return title


async def _get_session_context(
    session_summary: SessionSummary, messages: list[Message], n: int = 3
) -> tuple[str, list[UUID]]:

    n_msg = n * 2

    if len(messages) > n_msg:
        session_summary.summary = await _generate_session_summary(
            session_summary.summary, messages[:n_msg]
        )
        await _update_session_summary(session_summary)
        messages = messages[n_msg:]  # selecionar as `n` interações mais recentes

    context = ""
    if session_summary.summary.strip() != "":
        context += f"system: Those are the relevant themes for the recent interactions between the User and MedicAssist:\n"
        context += f"{session_summary.summary}\n"
        context += f"system: Those are the most recent messages between the User (human) and MedicAssist (ai).\n"
    context += _format_prompt_messages(messages)

    embedding_ids: list[UUID] = []
    embeddings = (emb for m in messages for emb in m.metadata.embeddings_with_score)
    for embedding in sorted(
        embeddings, key=lambda emb: emb.similarity_score, reverse=True
    )[:10]:
        embedding_ids.append(embedding.embedding_id)

    return context, embedding_ids


async def _fetch_session_summary_data(
    session_id: UUID,
) -> tuple[SessionSummary, list[Message]]:

    session_summary_query = """
        SELECT session_id, last_message_id, summary
        FROM session_summaries
        WHERE session_id = %s;
    """
    recent_messages_query = """
        SELECT 
            id,
            message->>'type' AS role, 
            message->'data'->>'content' AS text,
            message->'data'->>'additional_kwargs' AS metadata
        FROM message_store
        WHERE session_id = %s AND id > %s
        ORDER BY id ASC;
    """

    session_summary = SessionSummary(session_id=session_id)
    messages: list[Message] = []

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor(row_factory=class_row(SessionSummary)) as cursor:
            await cursor.execute(session_summary_query, [session_id])
            row = await cursor.fetchone()
            if row:
                session_summary = row

        async with connection.cursor() as cursor:
            await cursor.execute(
                recent_messages_query,
                (str(session_id), session_summary.last_message_id),
            )
            rows = await cursor.fetchall()
            for row in rows:
                metadata = MessageMetadata.model_validate(json.loads(row[3]))
                messages.append(
                    Message(id=row[0], role=row[1], text=row[2], metadata=metadata)
                )

    return session_summary, messages


async def _generate_session_summary(
    current_summary: str, messages: list[Message]
) -> str:

    prompt_template = ChatPromptTemplate.from_template(env.prompt_templates["summary"])
    model = ChatOpenAI(
        api_key=env.OPENAI_API_KEY, model="gpt-4o-2024-08-06", temperature=0.7
    )
    chain = (
        {
            "current_summary": RunnablePassthrough(),
            "messages": RunnablePassthrough(),
        }
        | prompt_template
        | model
        | StrOutputParser()
    )

    messages_str = _format_prompt_messages(messages)

    result = await chain.ainvoke(
        {"current_summary": current_summary, "messages": messages_str}
    )

    return result


async def _update_session_summary(session: SessionSummary) -> None:
    query = """
        INSERT INTO session_summaries (session_id, last_message_id, summary) 
        VALUES (%s, %s, %s)
        ON CONFLICT (session_id) 
        DO UPDATE SET 
            last_message_id = EXCLUDED.last_message_id,
            summary = EXCLUDED.summary;
    """

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                query, (session.session_id, session.last_message_id, session.summary)
            )


def _top_documents_for_output(embeddings: list[Embedding], output: str) -> dict:
    """Filter content retrieved by similarity search by it's relevance to the LLM's output.\n
    Returns `MessageMetadata` dictionary representation."""

    if not embeddings:
        return MessageMetadata().model_dump()

    distinct_references = set()
    distinct_embeddings: set[UUID] = set()
    distinct_images = set()
    embeddings_with_score: list[dict] = []
    references: list[OriginData] = []
    documents: list[Document] = []

    for x in embeddings:
        if x.id not in distinct_embeddings:
            document = Document(page_content=x.content, metadata=x.metadata)
            document.metadata["embedding_id"] = str(x.id)
            documents.append(document)
            distinct_embeddings.add(x.id)

    store = SKLearnVectorStore.from_documents(
        documents, OpenAIEmbeddings(api_key=env.OPENAI_API_KEY)
    )

    documents_with_score = store.similarity_search_with_relevance_scores(
        output, k=len(documents), score_threshold=0.9
    )

    for document, score in documents_with_score:
        embeddings_with_score.append(
            {
                "embedding_id": document.metadata["embedding_id"],
                "similarity_score": score,
            }
        )

        if "type" not in document.metadata:
            continue

        if document.metadata["material_id"] not in references:
            keys = ["material_id", "title", "position", "source", "year"]
            origin_data = {key: document.metadata[key] for key in keys}
            distinct_references.add(origin_data["material_id"])
            references.append(OriginData.model_validate(origin_data))

        for image in document.metadata["images"]:
            distinct_images.add(tuple(dict(image).items()))

    images = [Image.model_validate(dict(x)) for x in distinct_images]
    metadata = MessageMetadata(references=references, images=images).model_dump()
    metadata["embeddings_with_score"] = embeddings_with_score

    return metadata


async def _fetch_embeddings(embedding_ids: list[UUID]) -> list[Embedding]:

    query = """
        SELECT uuid AS id, document, cmetadata AS metadata
        FROM langchain_pg_embedding
        WHERE uuid = ANY(%s);
    """

    embeddings: list[Embedding] = []

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor(row_factory=class_row(Embedding)) as cursor:
            await cursor.execute(query, [embedding_ids])
            embeddings = await cursor.fetchall()

    return embeddings


async def _search_embeddings(session: Session, user_query: str) -> list[Embedding]:
    "Retrieve embeddings by similarity with the `user_query`."

    query = """
        WITH collection_ids AS (
            SELECT uuid FROM langchain_pg_collection
            WHERE name = %s
        )

        SELECT 
            uuid AS id, 
            document as content,
            cmetadata AS metadata,
            embedding <#> %s as similarity_score
        FROM langchain_pg_embedding
        WHERE
            collection_id IN (SELECT uuid FROM collection_ids)
        ORDER BY similarity_score
        LIMIT %s;
    """

    query_vector = await _generate_query_vector(user_query)
    score_threshold = 0.8
    max_embeddings = 5
    embeddings: list[Embedding] = []
    collection_name = _format_collection_name(session.type, session.content_id)

    async with await psycopg.AsyncConnection.connect(
        env.DATABASE_CONN.get_secret_value()
    ) as connection:
        async with connection.cursor(row_factory=class_row(Embedding)) as embed_cursor:
            await embed_cursor.execute(
                query,
                (
                    collection_name,
                    str(query_vector),
                    max_embeddings,
                ),
            )

            embeddings = await embed_cursor.fetchall()

    embeddings = list(
        filter(
            lambda x: x.similarity_score is not None
            and x.similarity_score >= score_threshold,
            embeddings,
        )
    )

    return embeddings


async def _generate_query_vector(search_input: str) -> list[float]:

    embbeding_response = await openai_client.embeddings.create(
        input=search_input, model="text-embedding-ada-002"
    )

    return embbeding_response.data[0].embedding


def _format_prompt_messages(messages: list[Message]) -> str:
    return "\n".join([f"{m.role}: {m.text}" for m in messages])


def _format_collection_name(session_type: str | None, content_id: str | None) -> str:
    if session_type == "chat":
        return "default"
    else:
        return f"{session_type}_{content_id}"
