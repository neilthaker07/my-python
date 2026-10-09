"""Online eval results: what the background evaluator found in live /assistant traffic."""

from collections import Counter
from datetime import timedelta
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy import select

from app.auth import CurrentUser
from app.database import DbSession
from app.models import AssistantTrace, utcnow
from app.schemas import OnlineEvalSummary, TraceRead
from app.services.permissions import check_can_view_traces

router = APIRouter(prefix="/evals/online", tags=["evals"])


@router.get("/summary", response_model=OnlineEvalSummary)
def summary(
    request: Request,
    db: DbSession,
    user: CurrentUser,
    hours: Annotated[int, Query(ge=1, le=24 * 90)] = 24,
) -> OnlineEvalSummary:
    check_can_view_traces(user)
    since = utcnow() - timedelta(hours=hours)
    traces = list(db.scalars(select(AssistantTrace).where(AssistantTrace.created_at >= since)))
    judged = [t for t in traces if t.grounded is not None]
    evaluator = getattr(request.app.state, "online_evaluator", None)

    def rate(values: list[bool]) -> Optional[float]:
        return round(sum(values) / len(values), 3) if values else None

    return OnlineEvalSummary(
        since=since,
        traces=len(traces),
        flagged=sum(1 for t in traces if t.flags),
        flags_by_check=dict(Counter(flag["check"] for t in traces for flag in t.flags)),
        judged=len(judged),
        grounded_rate=rate([t.grounded for t in judged]),
        complete_rate=rate([t.complete for t in judged]),
        judge_errors=sum(1 for t in traces if t.judge_error),
        avg_latency_ms=round(sum(t.latency_ms for t in traces) / len(traces)) if traces else None,
        pending=evaluator.pending if evaluator else 0,
        dropped=evaluator.dropped if evaluator else 0,
    )


@router.get("/traces", response_model=list[TraceRead])
def list_traces(
    db: DbSession,
    user: CurrentUser,
    problems_only: Annotated[bool, Query(description="Only flagged traces or failed judge grades")] = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[AssistantTrace]:
    """Newest first."""
    check_can_view_traces(user)
    traces = db.scalars(select(AssistantTrace).order_by(AssistantTrace.created_at.desc(), AssistantTrace.id.desc()))
    if problems_only:
        traces = (t for t in traces if t.flags or t.grounded is False or t.complete is False)
    return [t for _, t in zip(range(limit), traces)]


@router.get("/traces/{trace_id}", response_model=TraceRead)
def get_trace(trace_id: str, db: DbSession, user: CurrentUser) -> AssistantTrace:
    check_can_view_traces(user)
    trace = db.scalar(select(AssistantTrace).where(AssistantTrace.trace_id == trace_id))
    if trace is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Trace not found")
    return trace
