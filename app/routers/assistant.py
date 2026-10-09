import openai
from fastapi import APIRouter, HTTPException, Request, status

from app.auth import CurrentUser
from app.schemas import AssistantRequest, AssistantResponse, UserRead

router = APIRouter(tags=["assistant"])


@router.post("/assistant", response_model=AssistantResponse)
async def ask_assistant(data: AssistantRequest, request: Request, user: CurrentUser) -> AssistantResponse:
    assistant = getattr(request.app.state, "assistant", None)
    if assistant is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Assistant is not available. Check the server logs for the startup error.",
        )

    try:
        return await assistant.ask(data.question, UserRead.model_validate(user))
    except openai.AuthenticationError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "LLM API key is missing or invalid. Set LLM_API_KEY.",
        )
    except openai.RateLimitError as exc:
        # OpenAI uses 429 for an empty balance too, which retrying won't fix.
        if exc.code in ("credit_balance_exhausted", "insufficient_quota"):
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The LLM account has no credits left. Add credits in the provider's billing settings.",
            )
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limited by the LLM API. Try again shortly.")
    except (openai.APIConnectionError, openai.APIStatusError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM API error: {exc}")
