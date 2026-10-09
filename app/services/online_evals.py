"""Online evals: score live /assistant traffic in the background.

The request only hands a TraceRecord to `OnlineEvaluator.submit()`, a non-blocking put
on a bounded queue. Background workers then, in parallel with incoming requests:

1. Run the code checks on every trace (request errors, refusals, unverified answers,
   tool errors, weak knowledge-base hits, slow answers).
2. Save the trace and its flags to assistant_traces.
3. On a sample of answers (ONLINE_EVAL_JUDGE_SAMPLE_RATE), ask the LLM judge from
   evals/judge.py whether the answer is grounded in the tool results and complete,
   and save its verdict on the trace.

Nothing here can slow down or fail a request: a full queue drops the trace (counted in
`dropped`), and any error while evaluating is logged and swallowed.
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from openai import AsyncOpenAI
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import AssistantTrace
from app.schemas import ToolCall, UserRead
from app.services.assistant import PLACEHOLDER, REFUSAL_ANSWER, UNVERIFIED_ANSWER, PromptContext
from evals.judge import judge

logger = logging.getLogger(__name__)

WORKERS = 2  # judge calls at a time; more would mostly hit the provider's rate limits


@dataclass(frozen=True)
class TraceRecord:
    """Everything one /assistant request produced, as handed over by the request."""

    trace_id: str
    user: UserRead
    question: str
    today: str
    latency_ms: int
    answer: Optional[str] = None
    error: Optional[str] = None
    results: list[tuple[ToolCall, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Flag:
    check: str
    detail: str


def check_trace(record: TraceRecord) -> list[Flag]:
    """The code checks: cheap, deterministic, run on every trace."""
    flags: list[Flag] = []
    if record.error is not None:
        flags.append(Flag("request_error", record.error))
    if record.answer == REFUSAL_ANSWER:
        flags.append(Flag("refused", "the assistant refused or returned no answer"))
    if record.answer == UNVERIFIED_ANSWER:
        flags.append(Flag("unverified", "the answer's placeholders never matched the task data"))
    if record.answer and PLACEHOLDER.search(record.answer):
        flags.append(Flag("unfilled_placeholder", "the answer contains a {{task:...}} placeholder"))
    for call, output in record.results:
        if output.startswith("ERROR:"):
            flags.append(Flag("tool_error", f"{call.name}: {output.removeprefix('ERROR:').strip()[:200]}"))
        elif call.name == "search_knowledge_base":
            flags += _check_retrieval(call, output)
    if record.latency_ms > settings.online_eval_slow_ms:
        flags.append(Flag("slow", f"{record.latency_ms} ms"))
    return flags


def _check_retrieval(call: ToolCall, output: str) -> list[Flag]:
    query = call.input.get("query", "")
    try:
        hits = json.loads(output)
    except json.JSONDecodeError:
        return [Flag("low_retrieval", f"unreadable search output for {query!r}")]
    if not hits:
        return [Flag("low_retrieval", f"no knowledge-base hits for {query!r}")]
    best = max(hit["score"] for hit in hits)
    if best < settings.online_eval_min_retrieval_score:
        return [Flag("low_retrieval", f"best hit {best:.2f} for {query!r}")]
    return []


class OnlineEvaluator:
    """Background evaluation of /assistant traces. Use as `async with` to run the workers."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        judge_client: Optional[AsyncOpenAI],
        *,
        sample_rate: float = settings.online_eval_judge_sample_rate,
        queue_size: int = settings.online_eval_queue_size,
        sample: Callable[[], float] = random.random,
    ) -> None:
        self._sessions = session_factory
        self._judge_client = judge_client  # None: run the code checks only
        self._sample_rate = sample_rate
        self._sample = sample
        self._queue: asyncio.Queue[TraceRecord] = asyncio.Queue(maxsize=queue_size)
        self._workers: list[asyncio.Task] = []
        self.dropped = 0

    async def __aenter__(self) -> "OnlineEvaluator":
        self._workers = [asyncio.create_task(self._work(), name=f"online-eval-{i}") for i in range(WORKERS)]
        return self

    async def __aexit__(self, *exc_info) -> None:
        # Give queued traces a moment to finish on shutdown, then stop.
        try:
            await asyncio.wait_for(self._queue.join(), timeout=5)
        except asyncio.TimeoutError:
            logger.warning("Online evals: %d traces not evaluated at shutdown", self._queue.qsize())
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def submit(self, record: TraceRecord) -> bool:
        """Queue a trace for evaluation without waiting. Never raises; False if dropped."""
        try:
            self._queue.put_nowait(record)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            logger.warning("Online evals: queue full, dropped trace %s", record.trace_id)
            return False

    async def drain(self) -> None:
        """Wait until every queued trace has been evaluated (for tests and shutdown)."""
        await self._queue.join()

    async def _work(self) -> None:
        while True:
            record = await self._queue.get()
            try:
                await self.evaluate(record)
            except Exception:
                logger.exception("Online evals: failed to evaluate trace %s", record.trace_id)
            finally:
                self._queue.task_done()

    async def evaluate(self, record: TraceRecord) -> None:
        flags = check_trace(record)
        # Database calls are synchronous; a thread keeps them off the event loop.
        row_id = await asyncio.to_thread(self._save_trace, record, flags)
        if self._should_judge(record):
            await self._judge(record, row_id)

    def _should_judge(self, record: TraceRecord) -> bool:
        # Failed requests and the fixed refusal messages have nothing for the judge to grade;
        # the code checks already flag them.
        gradable = record.error is None and record.answer not in (None, REFUSAL_ANSWER, UNVERIFIED_ANSWER)
        return self._judge_client is not None and gradable and self._sample() < self._sample_rate

    async def _judge(self, record: TraceRecord, row_id: int) -> None:
        try:
            verdict = await judge(
                self._judge_client,
                question=record.question,
                answer=record.answer,
                results=record.results,
                context=PromptContext(today=record.today, user=record.user),
            )
        except Exception as exc:
            await asyncio.to_thread(self._save_judge_error, row_id, f"{type(exc).__name__}: {exc}")
            return
        await asyncio.to_thread(self._save_verdict, row_id, verdict)

    def _save_trace(self, record: TraceRecord, flags: list[Flag]) -> int:
        trace = AssistantTrace(
            trace_id=record.trace_id,
            user_id=record.user.id,
            question=record.question,
            answer=record.answer,
            error=record.error,
            tool_results=[{"tool": c.name, "input": c.input, "output": o} for c, o in record.results],
            latency_ms=record.latency_ms,
            router_model=settings.router_model,
            answer_model=settings.answer_model,
            flags=[{"check": f.check, "detail": f.detail} for f in flags],
        )
        with self._sessions() as db:
            db.add(trace)
            db.commit()
            return trace.id

    def _save_verdict(self, row_id: int, verdict) -> None:
        with self._sessions() as db:
            trace = db.get(AssistantTrace, row_id)
            trace.judge_model = settings.judge_model
            trace.grounded = verdict.grounded.passed
            trace.complete = verdict.complete.passed
            trace.judge_reasoning = {name: grade.reasoning for name, grade in verdict.grades().items()}
            db.commit()

    def _save_judge_error(self, row_id: int, error: str) -> None:
        with self._sessions() as db:
            trace = db.get(AssistantTrace, row_id)
            trace.judge_model = settings.judge_model
            trace.judge_error = error
            db.commit()
