"""Runs the plan → execute → compose pipeline against the real MCP server in-process.

The LLM is replaced by a fake client and the knowledge-base search is stubbed out,
so no API key or Postgres is needed.
"""

import copy
import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app import mcp_server
from app.config import settings
from app.main import app
from app.rag.store import SearchHit
from app.models import UserRole
from app.schemas import ToolCall, UserRead
from app.services.assistant import REFUSAL_ANSWER, UNVERIFIED_ANSWER, TaskAssistant


ADMIN = UserRead(id=5, name="Erin", role=UserRole.ADMIN, team="it")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def reply(*output) -> SimpleNamespace:
    """A fake Responses API result; output_text mirrors the SDK's convenience property."""
    texts = [part.text for item in output if item.type == "message"
             for part in item.content if part.type == "output_text"]
    return SimpleNamespace(output=list(output), output_text="".join(texts))


def plan(*calls: tuple[str, dict], needs_results: bool = False) -> SimpleNamespace:
    """A router reply: a JSON plan of (tool, arguments) calls."""
    body = {"calls": [{"tool": name, "arguments": args} for name, args in calls], "needs_results": needs_results}
    return reply(text(json.dumps(body)))


def offered_tools(router_request: dict) -> set[str]:
    """The tools a router request lets the model plan, read from the plan's JSON schema."""
    branches = router_request["text"]["format"]["schema"]["properties"]["calls"]["items"]["anyOf"]
    return {branch["properties"]["tool"]["const"] for branch in branches}


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=value)])


def refusal() -> SimpleNamespace:
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="No.")])


class FakeResponses:
    """Answers the router and composer calls, told apart by model.

    Router replies are returned in order, one per planning round; once they run out
    the router replies with an empty plan, which ends the planning loop.
    """

    def __init__(self, router_replies: list[SimpleNamespace], composer_replies: list[SimpleNamespace]) -> None:
        self.replies = {settings.router_model: list(router_replies), settings.answer_model: list(composer_replies)}
        self.requests: dict[str, list[dict]] = {settings.router_model: [], settings.answer_model: []}

    async def create(self, **kwargs) -> SimpleNamespace:
        model = kwargs["model"]
        self.requests[model].append({**kwargs, "input": copy.deepcopy(kwargs["input"])})
        queue = self.replies[model]
        if queue:
            return queue.pop(0)
        return plan() if model == settings.router_model else reply()


class FakeOpenAI:
    def __init__(self, router_reply, composer_reply) -> None:
        as_list = lambda r: r if isinstance(r, list) else [r]
        self.responses = FakeResponses(as_list(router_reply), as_list(composer_reply))

    def router_requests(self) -> list[dict]:
        return self.responses.requests[settings.router_model]

    def composer_requests(self) -> list[dict]:
        return self.responses.requests[settings.answer_model]


@pytest.fixture
def task_api(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Route the MCP server's HTTP calls to the app in-process, on the test database.
    monkeypatch.setattr(
        mcp_server,
        "_task_api",
        lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test"),
    )
    return client


@pytest.fixture
def stub_search(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = SearchHit(id=1, category="status", question="What does blocked mean?",
                    answer="Blocked means waiting on someone else.", score=0.9)
    monkeypatch.setattr(mcp_server.store, "search", lambda query, top_k: [hit])


@pytest.mark.anyio
async def test_faq_question_routes_to_knowledge_base(stub_search):
    fake = FakeOpenAI(
        router_reply=plan(("search_knowledge_base", {"query": "blocked status"})),
        composer_reply=reply(text("Blocked means you're waiting on someone else.")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("What does blocked mean?", ADMIN)

    assert response.answer == "Blocked means you're waiting on someone else."
    assert response.tool_calls == [ToolCall(name="search_knowledge_base", input={"query": "blocked status"})]

    router_request = fake.router_requests()[0]
    assert offered_tools(router_request) == {"list_tasks", "get_task", "create_task", "list_users", "search_knowledge_base"}
    # The model can't see function tools in a plan, so their descriptions are in the prompt.
    assert "- search_knowledge_base: Semantic search" in router_request["instructions"]

    # The composer gets the question plus the tool output, and no tools of its own.
    composer_request = fake.composer_requests()[0]
    prompt = composer_request["input"]
    assert "What does blocked mean?" in prompt
    assert "Blocked means waiting on someone else." in prompt
    assert "tools" not in composer_request


@pytest.mark.anyio
async def test_tool_error_is_passed_to_composer(task_api: TestClient):
    fake = FakeOpenAI(
        router_reply=plan(("get_task", {"task_id": 999})),
        composer_reply=reply(text("Task 999 doesn't exist.")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        await assistant.ask("Show task 999", ADMIN)

    prompt = fake.composer_requests()[0]["input"]
    assert "ERROR:" in prompt and "Task 999 does not exist" in prompt


@pytest.mark.anyio
async def test_no_tool_needed_still_composes_answer():
    fake = FakeOpenAI(router_reply=plan(), composer_reply=reply(text("Hi! How can I help?")))

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("hello", ADMIN)

    assert response.answer == "Hi! How can I help?"
    assert response.tool_calls == []
    assert "(no tools were called)" in fake.composer_requests()[0]["input"]


@pytest.mark.anyio
async def test_router_refusal_skips_tools_and_composer():
    fake = FakeOpenAI(router_reply=reply(refusal()), composer_reply=reply())

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("something harmful", ADMIN)

    assert response.answer == REFUSAL_ANSWER
    assert fake.composer_requests() == []


@pytest.mark.anyio
@pytest.mark.parametrize("code", ["json_validate_failed", "tool_use_failed"])
async def test_router_retries_once_when_provider_rejects_the_plan(task_api: TestClient, code: str):
    fake = FakeOpenAI(
        router_reply=plan(("list_tasks", {"overdue": True})),
        composer_reply=reply(text("No overdue tasks.")),
    )
    original_create = fake.responses.create
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        if kwargs["model"] == settings.router_model:
            calls += 1
            if calls == 1:
                request = httpx.Request("POST", "http://test/responses")
                raise openai.BadRequestError(
                    "Plan validation failed",
                    response=httpx.Response(400, request=request),
                    body={"code": code},
                )
        return await original_create(**kwargs)

    fake.responses.create = create

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("Do I have overdue tasks?", ADMIN)

    assert calls == 2  # rejected, then retried; the plan is final, so no further round
    assert response.answer == "No overdue tasks."
    assert response.tool_calls == [ToolCall(name="list_tasks", input={"overdue": True})]


@pytest.mark.anyio
async def test_multi_part_question_is_planned_once_and_run_in_parallel(task_api: TestClient, stub_search):
    fake = FakeOpenAI(
        router_reply=plan(("search_knowledge_base", {"query": "leave"}), ("list_tasks", {"overdue": True})),
        composer_reply=reply(text("Hand over your tasks. You have no overdue tasks.")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("I'm going on leave. Any overdue tasks?", ADMIN)

    assert [call.name for call in response.tool_calls] == ["search_knowledge_base", "list_tasks"]
    assert len(fake.router_requests()) == 1  # needs_results is false: no second round
    # The router asks for a JSON plan, not function tool calls.
    router_request = fake.router_requests()[0]
    assert router_request["text"]["format"]["type"] == "json_schema" and "tools" not in router_request

    # The composer gets both results.
    prompt = fake.composer_requests()[0]["input"]
    assert "Blocked means waiting on someone else." in prompt and 'tool="list_tasks"' in prompt


@pytest.mark.anyio
async def test_router_plans_again_when_it_needs_results(task_api: TestClient):
    task = task_api.post("/tasks", json={"title": "Prepare demo"}).json()
    fake = FakeOpenAI(
        router_reply=[
            plan(("list_tasks", {}), needs_results=True),
            plan(("get_task", {"task_id": task["id"]})),
        ],
        composer_reply=reply(text("ok")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("Show my newest task", ADMIN)

    assert [call.name for call in response.tool_calls] == ["list_tasks", "get_task"]
    # Round 2 sees round 1's results, so the model knows what it already has.
    first_input, second_input = (request["input"] for request in fake.router_requests())
    assert first_input == "Show my newest task"
    assert 'tool="list_tasks"' in second_input and "Prepare demo" in second_input


@pytest.mark.anyio
async def test_write_waits_for_the_reads_it_depends_on(task_api: TestClient):
    from tests.conftest import DAVE_ID

    fake = FakeOpenAI(
        router_reply=[
            # The model plans create_task next to list_users, with a guessed assignee.
            plan(("list_users", {}), ("create_task", {"title": "Fix bug"}), needs_results=True),
            plan(("create_task", {"title": "Fix bug", "assignee_id": DAVE_ID})),
        ],
        composer_reply=reply(text("Created.")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("Create a task 'Fix bug' for Dave", ADMIN)

    assert response.tool_calls == [
        ToolCall(name="list_users", input={}),
        ToolCall(name="create_task", input={"title": "Fix bug", "assignee_id": DAVE_ID}),
    ]
    assert [t["assignee_id"] for t in task_api.get("/tasks").json()] == [DAVE_ID]


@pytest.mark.anyio
async def test_null_arguments_are_left_to_the_tool_defaults(task_api: TestClient):
    fake = FakeOpenAI(
        router_reply=plan(("list_tasks", {"status": None, "overdue": None, "due_on": None})),
        composer_reply=reply(text("ok")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("List my tasks", ADMIN)

    assert response.tool_calls == [ToolCall(name="list_tasks", input={})]
    assert "ERROR:" not in fake.composer_requests()[0]["input"]


@pytest.mark.anyio
async def test_invalid_plan_composes_from_no_data():
    fake = FakeOpenAI(router_reply=reply(text("not json")), composer_reply=reply(text("I couldn't find that.")))

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("Do I have overdue tasks?", ADMIN)

    assert response.tool_calls == []
    assert response.answer == "I couldn't find that."


@pytest.mark.anyio
async def test_planning_stops_after_max_rounds(task_api: TestClient):
    from app.services.assistant import MAX_PLANNING_ROUNDS

    fake = FakeOpenAI(
        router_reply=[plan(("list_tasks", {}), needs_results=True) for _ in range(MAX_PLANNING_ROUNDS + 2)],
        composer_reply=reply(text("ok")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("loop forever", ADMIN)

    assert len(fake.router_requests()) == MAX_PLANNING_ROUNDS
    assert len(response.tool_calls) == MAX_PLANNING_ROUNDS


@pytest.mark.anyio
async def test_task_values_in_answer_come_from_mcp_data(task_api: TestClient):
    task = task_api.post("/tasks", json={"title": "Prepare demo", "priority": "high"}).json()
    tid = task["id"]
    fake = FakeOpenAI(
        router_reply=plan(("get_task", {"task_id": tid})),
        composer_reply=reply(text(
            f'Task {tid} "{{{{task:{tid}.title}}}}" is {{{{task:{tid}.status}}}} '
            f"with {{{{task:{tid}.priority}}}} priority, due {{{{task:{tid}.due_date}}}}."
        )),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask(f"Show task {tid}", ADMIN)

    assert response.answer == f'Task {tid} "Prepare demo" is todo with high priority, due none.'
    # The model is told to use placeholders rather than writing values itself.
    assert "{{task:<id>.<field>}}" in fake.composer_requests()[0]["instructions"]


@pytest.mark.anyio
async def test_placeholder_for_unfetched_task_is_retried_with_feedback(task_api: TestClient):
    task = task_api.post("/tasks", json={"title": "Real task"}).json()
    tid = task["id"]
    fake = FakeOpenAI(
        router_reply=plan(("get_task", {"task_id": tid})),
        composer_reply=[
            reply(text("Task 999 is {{task:999.status}}.")),
            reply(text(f"Task {tid} is {{{{task:{tid}.status}}}}.")),
        ],
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask(f"Show task {tid}", ADMIN)

    assert response.answer == f"Task {tid} is todo."
    retry_prompt = fake.composer_requests()[1]["input"]
    assert "{{task:999.status}}: task 999 was not fetched" in retry_prompt


@pytest.mark.anyio
async def test_answer_that_never_matches_the_data_is_not_returned(task_api: TestClient):
    fake = FakeOpenAI(
        router_reply=plan(("list_tasks", {})),
        composer_reply=[reply(text("{{task:1.owner}}")), reply(text("{{task:1.owner}}"))],
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("who owns task 1?", ADMIN)

    assert response.answer == UNVERIFIED_ANSWER


@pytest.mark.anyio
async def test_task_values_are_filled_from_list_tasks_results(task_api: TestClient):
    tid = task_api.post("/tasks", json={"title": "Prepare demo"}).json()["id"]
    fake = FakeOpenAI(
        router_reply=plan(("list_tasks", {})),
        composer_reply=reply(text(f'Task {tid}: "{{{{task:{tid}.title}}}}".')),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("List my tasks", ADMIN)

    assert response.answer == f'Task {tid}: "Prepare demo".'
