from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from os import getcwd
import warnings

from .message_history.routes import router as message_router
from .text_generation.routes import router as generation_router
from .session.routes import router as session_router
from .exceptions import ExceptionLoggingMiddleware


load_dotenv()
app = FastAPI()
app.add_middleware(CORSMiddleware)
app.add_middleware(ExceptionLoggingMiddleware)
app.include_router(session_router)
app.include_router(message_router)
app.include_router(generation_router)

warnings.filterwarnings(
    "ignore",
    message="No relevant docs were retrieved using the relevance score threshold*",
)


@app.get("/test", response_class=HTMLResponse)
async def get(request: Request):
    templates = Jinja2Templates(directory=getcwd())
    return templates.TemplateResponse("test.html", {"request": request})
