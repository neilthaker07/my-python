"""Tests for the eval harness itself (evals/). No LLM or Postgres: the judge gets a fake client."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.auth import USER_HEADER
from app.main import app
from app.schemas import ToolCall
from app.services.assistant import PromptContext
from evals.judge import JudgeError, judge
from evals.run import ALICE, CAROL, SEED_TASKS, USERS, Case, check_case, load_cases, task_database
from tests.conftest import ADMIN_ID

TOOL_NAMES = {"list_tasks", "get_task", "create_task", "list_users", "search_knowledge_base"}

VERDICT = {
    "grounded": {"reasoning": "All claims match.", "passed": True},
    "complete": {"reasoning": "Answers the question.", "passed": True},
    "expectations": {"reasoning": "Names task 1 only.", "passed": True},
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeJudgeClient:
    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict] = []
        self.responses = self

    async def create(self, **kwargs) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(output_text=self.outputs.pop(0))


def test_cases_file_is_valid():
    cases = load_cases()
    assert cases
    for case in cases:
        assert case.user_id in USERS, case.id
        named = {call.tool for call in case.expect_calls} | set(case.forbid_tools)
        assert named <= TOOL_NAMES, case.id


def test_check_case_passes_when_calls_and_answer_match():
    case = Case(
        id="c", user_id=ALICE, question="q", expectations="e",
        expect_calls=[{"tool": "list_tasks", "arguments": {"overdue": True}}],
        forbid_tools=["create_task"], max_tool_calls=2, answer_excludes=["secret"],
    )
    calls = [ToolCall(name="list_tasks", input={"overdue": True, "status": "todo"})]

    assert check_case(case, "Task 1 is overdue.", calls) == []


def test_check_case_reports_every_problem():
    case = Case(
        id="c", user_id=ALICE, question="q", expectations="e",
        expect_calls=[{"tool": "list_tasks", "arguments": {"overdue": True}}, {"tool": "search_knowledge_base"}],
        forbid_tools=["create_task"], max_tool_calls=1, answer_excludes=["Migrate CI"],
    )
    calls = [ToolCall(name="list_tasks", input={}), ToolCall(name="create_task", input={"title": "x"})]

    assert check_case(case, "Bob is working on migrate ci.", calls) == [
        "no list_tasks call with {'overdue': True}",
        "no search_knowledge_base call",
        "called create_task, which this case forbids",
        "made 2 tool calls, at most 1 expected",
        "answer mentions 'Migrate CI'",
    ]


def test_task_database_seeds_tasks_with_permissions_applied():
    with task_database():
        api = TestClient(app)
        all_tasks = api.get("/tasks", headers={USER_HEADER: str(ADMIN_ID)}).json()
        alice = api.get("/tasks", headers={USER_HEADER: str(ALICE)}).json()
        carol_overdue = api.get("/tasks", params={"overdue": True}, headers={USER_HEADER: str(CAROL)}).json()

    assert [t["title"] for t in all_tasks] == [t.title for t in sorted(SEED_TASKS, key=lambda t: t.due_in_days)]
    assert sorted(t["id"] for t in alice) == [1, 2, 3, 4, 6]
    assert [t["id"] for t in carol_overdue] == [1, 5]
    assert next(t for t in alice if t["id"] == 4)["completed_at"] is not None
    assert app.dependency_overrides == {}


@pytest.mark.anyio
async def test_judge_sends_evidence_and_parses_verdict():
    fake = FakeJudgeClient(json.dumps(VERDICT))
    results = [(ToolCall(name="list_tasks", input={"overdue": True}), '[{"id": 1, "title": "Fix login timeout bug"}]')]

    verdict = await judge(
        fake, question="Anything overdue?", answer="Task 1 is overdue.", results=results,
        context=PromptContext(today="2026-10-09", user=USERS[ALICE]), expectations="Names task 1 only.",
    )

    assert verdict.expectations.passed and verdict.grades()["grounded"].reasoning == "All claims match."
    request = fake.requests[0]
    for text in ("Anything overdue?", "Fix login timeout bug", "Task 1 is overdue.", "Names task 1 only."):
        assert text in request["input"]
    assert "Alice" in request["instructions"] and "2026-10-09" in request["instructions"]


@pytest.mark.anyio
async def test_judge_retries_once_then_gives_up_on_invalid_output():
    fake = FakeJudgeClient("not json", "still not json")

    with pytest.raises(JudgeError):
        await judge(
            fake, question="q", answer="a", results=[],
            context=PromptContext(today="2026-10-09", user=USERS[ALICE]), expectations="e",
        )
    assert len(fake.requests) == 2
    assert "(no tools were called)" in fake.requests[0]["input"]
