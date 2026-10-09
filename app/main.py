import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.bootstrap import init_db
from app.config import settings
from app.database import SessionLocal
from app.routers import assistant, evals, tasks, users
from app.services.assistant import TaskAssistant, stdio_server_params
from app.services.online_evals import OnlineEvaluator
from app.services.permissions import PermissionDenied

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    async with AsyncExitStack() as stack:
        app.state.assistant = await _start_assistant(stack)
        app.state.online_evaluator = await _start_online_evals(stack)
        yield


async def _start_assistant(stack: AsyncExitStack) -> Optional[TaskAssistant]:
    # The task CRUD API should keep working even if the assistant can't start.
    try:
        return await stack.enter_async_context(TaskAssistant(stdio_server_params()))
    except Exception:
        logger.exception("Could not start the assistant; POST /assistant will return 503")
        return None


async def _start_online_evals(stack: AsyncExitStack) -> Optional[OnlineEvaluator]:
    if not settings.online_eval_enabled:
        return None
    # Without an API key the judge can't run, but the code checks still can.
    judge_client = (
        AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, max_retries=4)
        if settings.llm_api_key else None
    )
    return await stack.enter_async_context(OnlineEvaluator(SessionLocal, judge_client))


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(tasks.router)
app.include_router(users.router)
app.include_router(assistant.router)
app.include_router(evals.router)


@app.exception_handler(PermissionDenied)
async def permission_denied(request: Request, exc: PermissionDenied) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(exc)})


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
