"""Online evals: background scoring of live /assistant traffic.

The judge model is a fake client, so no LLM is needed. Traces are written to the
Postgres test database by the evaluator's own sessions, like in production.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Iterator

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.auth import USER_HEADER
from app.main import app
from app.models import AssistantTrace, UserRole
from app.schemas import AssistantResponse, ToolCall, UserRead
from app.services.assistant import REFUSAL_ANSWER
from app.services.online_evals import OnlineEvaluator, TraceRecord, check_trace
from tests.conftest import ALICE_ID

ALICE = UserRead(id=ALICE_ID, name="Alice", role=UserRole.EMPLOYEE, team="platform")

VERDICT = {
    "grounded": {"reasoning": "Matches the tool results.", "passed": True},
    "complete": {"reasoning": "Misses the second question.", "passed": False},
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def sessions(test_engine: Engine, db_session: Session) -> sessionmaker:
    """The evaluator's session factory, on the (freshly truncated) test database."""
    return sessionmaker(bind=test_engine, expire_on_commit=False)


class FakeJudge:
    """A judge client whose answer can be held back, to show requests don't wait for it."""

    def __init__(self, output: str = json.dumps(VERDICT), error: Exception = None) -> None:
        self.output, self.error = output, error
        self.release = asyncio.Event()
        self.release.set()
        self.calls = 0
        self.responses = self

    async def create(self, **kwargs):
        self.calls += 1
        await self.release.wait()
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=self.output)


def record(**overrides) -> TraceRecord:
    fields = dict(
        trace_id="t" * 32, user=ALICE, question="Any overdue tasks?", today="2026-10-09", latency_ms=1200,
        answer="You have no overdue tasks.",
        results=[(ToolCall(name="list_tasks", input={"overdue": True}), '{"tasks": [], "more_tasks_exist": false}')],
    )
    return TraceRecord(**{**fields, **overrides})


def traces(sessions: sessionmaker) -> list[AssistantTrace]:
    with sessions() as db:
        return list(db.scalars(select(AssistantTrace).order_by(AssistantTrace.id)))


# --- Code checks ---------------------------------------------------------------


def test_clean_trace_has_no_flags():
    assert check_trace(record()) == []


def test_code_checks_flag_problems():
    kb = ToolCall(name="search_knowledge_base", input={"query": "parking policy"})
    flags = check_trace(record(
        answer="Task 3 is {{task:3.status}}.",
        latency_ms=25_000,
        results=[
            (ToolCall(name="create_task", input={"title": "x"}), "ERROR: 403: Creating tasks through the assistant is limited"),
            (kb, json.dumps([{"id": 1, "score": 0.41}, {"id": 2, "score": 0.38}])),
        ],
    ))

    assert [(f.check, f.detail) for f in flags] == [
        ("unfilled_placeholder", "the answer contains a {{task:...}} placeholder"),
        ("tool_error", "create_task: 403: Creating tasks through the assistant is limited"),
        ("low_retrieval", "best hit 0.41 for 'parking policy'"),
        ("slow", "25000 ms"),
    ]


def test_request_errors_and_refusals_are_flagged():
    assert [f.check for f in check_trace(record(answer=None, results=[], error="APIConnectionError: boom"))] == ["request_error"]
    assert [f.check for f in check_trace(record(answer=REFUSAL_ANSWER))] == ["refused"]


# --- The evaluator -------------------------------------------------------------


@pytest.mark.anyio
async def test_submit_returns_immediately_while_the_judge_works(sessions: sessionmaker):
    fake = FakeJudge()
    fake.release.clear()  # the judge "takes forever" until released
    async with OnlineEvaluator(sessions, fake, sample_rate=1.0) as evaluator:
        started = time.monotonic()
        assert evaluator.submit(record()) is True
        assert time.monotonic() - started < 0.05  # the request isn't held up

        while fake.calls == 0:  # the worker picks the trace up in the background
            await asyncio.sleep(0.01)
        [saved] = traces(sessions)
        assert saved.grounded is None  # saved with its flags, judge still running

        fake.release.set()
        await evaluator.drain()

    [saved] = traces(sessions)
    assert (saved.grounded, saved.complete) == (True, False)
    assert saved.judge_reasoning == {"grounded": "Matches the tool results.", "complete": "Misses the second question."}
    assert saved.tool_results == [{"tool": "list_tasks", "input": {"overdue": True},
                                   "output": '{"tasks": [], "more_tasks_exist": false}'}]


@pytest.mark.anyio
async def test_judge_failure_is_recorded_not_raised(sessions: sessionmaker):
    request = httpx.Request("POST", "http://test")
    fake = FakeJudge(error=openai.APIConnectionError(request=request))
    async with OnlineEvaluator(sessions, fake, sample_rate=1.0) as evaluator:
        evaluator.submit(record())
        await evaluator.drain()

    [saved] = traces(sessions)
    assert saved.grounded is None and saved.judge_error.startswith("APIConnectionError")


@pytest.mark.anyio
async def test_judge_runs_only_on_the_sample(sessions: sessionmaker):
    fake = FakeJudge()
    samples = iter([0.9, 0.1])  # 0.9 is above the 0.2 rate: not judged; 0.1 is judged
    async with OnlineEvaluator(sessions, fake, sample_rate=0.2, sample=lambda: next(samples)) as evaluator:
        evaluator.submit(record(trace_id="a" * 32))
        evaluator.submit(record(trace_id="b" * 32))
        await evaluator.drain()

    assert fake.calls == 1
    assert [t.grounded for t in traces(sessions)].count(None) == 1  # both saved, one judged


@pytest.mark.anyio
async def test_refusals_and_failed_requests_are_saved_but_not_judged(sessions: sessionmaker):
    fake = FakeJudge()
    async with OnlineEvaluator(sessions, fake, sample_rate=1.0) as evaluator:
        evaluator.submit(record(trace_id="a" * 32, answer=REFUSAL_ANSWER))
        evaluator.submit(record(trace_id="b" * 32, answer=None, results=[], error="RuntimeError: boom"))
        await evaluator.drain()

    assert fake.calls == 0
    assert [t.flags[0]["check"] for t in traces(sessions)] == ["refused", "request_error"]


@pytest.mark.anyio
async def test_full_queue_drops_traces_instead_of_blocking(sessions: sessionmaker):
    evaluator = OnlineEvaluator(sessions, None, queue_size=1)  # workers not started
    assert evaluator.submit(record(trace_id="a" * 32)) is True
    assert evaluator.submit(record(trace_id="b" * 32)) is False
    assert evaluator.dropped == 1


# --- The /assistant route hands traces over --------------------------------------


class RecordingEvaluator:
    def __init__(self) -> None:
        self.records: list[TraceRecord] = []
        self.pending = self.dropped = 0

    def submit(self, record: TraceRecord) -> bool:
        self.records.append(record)
        return True


class FakeAssistant:
    def __init__(self, error: Exception = None) -> None:
        self.error = error

    async def ask_with_results(self, question, user):
        if self.error:
            raise self.error
        call = ToolCall(name="list_tasks", input={})
        return AssistantResponse(answer="No tasks.", tool_calls=[call]), [(call, '{"tasks": []}')]


@pytest.fixture
def evaluator() -> Iterator[RecordingEvaluator]:
    recording = RecordingEvaluator()
    app.state.online_evaluator = recording
    yield recording
    del app.state.online_evaluator
    if hasattr(app.state, "assistant"):
        del app.state.assistant


def test_route_submits_a_trace_with_the_response_trace_id(client: TestClient, evaluator: RecordingEvaluator):
    app.state.assistant = FakeAssistant()
    response = client.post("/assistant", json={"question": "my tasks?"}, headers={USER_HEADER: str(ALICE_ID)})

    [submitted] = evaluator.records
    assert response.json()["trace_id"] == submitted.trace_id
    assert (submitted.user.id, submitted.question, submitted.answer) == (ALICE_ID, "my tasks?", "No tasks.")
    assert submitted.results[0][0].name == "list_tasks"


def test_failed_requests_are_traced_too(client: TestClient, evaluator: RecordingEvaluator):
    app.state.assistant = FakeAssistant(error=openai.APIConnectionError(request=httpx.Request("POST", "http://x")))
    response = client.post("/assistant", json={"question": "my tasks?"})

    assert response.status_code == 502
    [submitted] = evaluator.records
    assert submitted.answer is None and submitted.error.startswith("APIConnectionError")


# --- Reading results -------------------------------------------------------------


@pytest.mark.anyio
async def test_admins_can_read_summary_and_traces(client: TestClient, sessions: sessionmaker):
    async with OnlineEvaluator(sessions, FakeJudge(), sample_rate=1.0) as evaluator:
        evaluator.submit(record(trace_id="a" * 32))
        evaluator.submit(record(trace_id="b" * 32, answer=REFUSAL_ANSWER))
        await evaluator.drain()

    summary = client.get("/evals/online/summary").json()
    assert {k: summary[k] for k in ("traces", "flagged", "flags_by_check", "judged", "grounded_rate", "complete_rate")} == {
        "traces": 2, "flagged": 1, "flags_by_check": {"refused": 1}, "judged": 1,
        "grounded_rate": 1.0, "complete_rate": 0.0,
    }
    # Both have a problem: one flagged as refused, the other failed the "complete" grade.
    problems = client.get("/evals/online/traces", params={"problems_only": True}).json()
    assert {t["trace_id"] for t in problems} == {"a" * 32, "b" * 32}
    assert client.get(f"/evals/online/traces/{'a' * 32}").json()["complete"] is False


def test_only_admins_can_read_traces(client: TestClient):
    as_alice = {USER_HEADER: str(ALICE_ID)}
    assert client.get("/evals/online/summary", headers=as_alice).status_code == 403
    assert client.get("/evals/online/traces", headers=as_alice).status_code == 403
