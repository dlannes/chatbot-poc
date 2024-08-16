from typing import Sequence
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from psycopg import AsyncConnection
from fastapi import HTTPException
import traceback
import os

from .exceptions import DBEntityNotFoundException


DATABASE = os.getenv("DATABASE_CONNECTION", "")


class DBEntityNotFoundException(Exception):

    def __init__(
        self, table_name: str, columns: Sequence | str, values: Sequence | str
    ):
        entity_names = self.__format_str(columns)
        entity_values = self.__format_str(values)
        super().__init__(
            f"No results for `{entity_names}` = `{entity_values}` were found in the relation `{table_name}`."
        )

    def __format_str(self, data) -> str:
        if isinstance(data, Sequence) and not isinstance(data, str):
            return ", ".join(map(str, data))
        else:
            return data


class ExceptionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
        except DBEntityNotFoundException as e:
            await self.__create_log(request.url.path, e)
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            await self.__create_log(request.url.path, e)
            raise
        return response

    async def __create_log(self, endpoint: str, exception: Exception):
        ex_type = type(exception).__name__
        ex_message = str(exception)
        stack_trace = traceback.format_exc()
        query = """
            INSERT INTO exception_logs (endpoint, type, message, stack_trace)
            VALUES (%s, %s, %s, %s)
        """

        async with await AsyncConnection.connect(DATABASE) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, (endpoint, ex_type, ex_message, stack_trace))
