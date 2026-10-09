import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.bootstrap import init_db
from app.config import settings
from app.routers import assistant, tasks, users
from app.services.assistant import TaskAssistant, stdio_server_params
from app.services.permissions import PermissionDenied

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    async with AsyncExitStack() as stack:
        app.state.assistant = await _start_assistant(stack)
        yield


async def _start_assistant(stack: AsyncExitStack) -> Optional[TaskAssistant]:
    # The task CRUD API should keep working even if the assistant can't start.
    try:
        return await stack.enter_async_context(TaskAssistant(stdio_server_params()))
    except Exception:
        logger.exception("Could not start the assistant; POST /assistant will return 503")
        return None


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(tasks.router)
app.include_router(users.router)
app.include_router(assistant.router)


@app.exception_handler(PermissionDenied)
async def permission_denied(request: Request, exc: PermissionDenied) -> JSONResponse:
    return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": str(exc)})


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}
