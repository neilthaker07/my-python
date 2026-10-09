import time
import uuid
from datetime import date
from typing import Optional

import openai
from fastapi import APIRouter, HTTPException, Request, status

from app.auth import CurrentUser
from app.schemas import AssistantRequest, AssistantResponse, ToolCall, UserRead
from app.services.online_evals import OnlineEvaluator, TraceRecord

router = APIRouter(tags=["assistant"])


@router.post("/assistant", response_model=AssistantResponse)
async def ask_assistant(data: AssistantRequest, request: Request, user: CurrentUser) -> AssistantResponse:
    assistant = getattr(request.app.state, "assistant", None)
    if assistant is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Assistant is not available. Check the server logs for the startup error.",
        )

    asker = UserRead.model_validate(user)
    trace_id = uuid.uuid4().hex
    started = time.monotonic()

    def trace(answer: Optional[str] = None, results: list[tuple[ToolCall, str]] = (), error: Optional[str] = None):
        # Hand the request to the online evals. Non-blocking: they run in the background.
        evaluator: Optional[OnlineEvaluator] = getattr(request.app.state, "online_evaluator", None)
        if evaluator is not None:
            evaluator.submit(TraceRecord(
                trace_id=trace_id, user=asker, question=data.question, today=date.today().isoformat(),
                latency_ms=round((time.monotonic() - started) * 1000),
                answer=answer, results=list(results), error=error,
            ))

    try:
        response, results = await assistant.ask_with_results(data.question, asker)
    except Exception as exc:
        trace(error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, openai.OpenAIError):
            raise _llm_error(exc) from exc
        raise

    trace(answer=response.answer, results=results)
    return response.model_copy(update={"trace_id": trace_id})


def _llm_error(exc: openai.OpenAIError) -> HTTPException:
    if isinstance(exc, openai.AuthenticationError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "LLM API key is missing or invalid. Set LLM_API_KEY.")
    if isinstance(exc, openai.RateLimitError):
        # OpenAI uses 429 for an empty balance too, which retrying won't fix.
        if exc.code in ("credit_balance_exhausted", "insufficient_quota"):
            return HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The LLM account has no credits left. Add credits in the provider's billing settings.",
            )
        return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limited by the LLM API. Try again shortly.")
    return HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM API error: {exc}")
