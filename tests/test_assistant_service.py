"""Runs the route → fetch → compose pipeline against the real MCP server in-process.

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


def function_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", call_id=f"call_{name}", name=name, arguments=json.dumps(arguments))


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=value)])


def refusal() -> SimpleNamespace:
    return SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="No.")])


class FakeResponses:
    """Answers the router and composer calls, told apart by model.

    Router replies are returned in order, one per routing round; once they run out
    the router "replies" with no tool calls, which ends the routing loop.
    """

    def __init__(self, router_replies: list[SimpleNamespace], composer_replies: list[SimpleNamespace]) -> None:
        self.replies = {settings.router_model: list(router_replies), settings.answer_model: list(composer_replies)}
        self.requests: dict[str, list[dict]] = {settings.router_model: [], settings.answer_model: []}

    async def create(self, **kwargs) -> SimpleNamespace:
        model = kwargs["model"]
        # Snapshot the input: the assistant keeps appending to the same conversation list.
        self.requests[model].append({**kwargs, "input": copy.deepcopy(kwargs["input"])})
        queue = self.replies[model]
        return queue.pop(0) if queue else reply()


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
        router_reply=reply(function_call("search_knowledge_base", {"query": "blocked status"})),
        composer_reply=reply(text("Blocked means you're waiting on someone else.")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("What does blocked mean?", ADMIN)

    assert response.answer == "Blocked means you're waiting on someone else."
    assert response.tool_calls == [ToolCall(name="search_knowledge_base", input={"query": "blocked status"})]

    router_request = fake.router_requests()[0]
    assert {t["name"] for t in router_request["tools"]} == {"list_tasks", "get_task", "create_task", "list_users", "search_knowledge_base"}

    # The composer gets the question plus the tool output, and no tools of its own.
    composer_request = fake.composer_requests()[0]
    prompt = composer_request["input"]
    assert "What does blocked mean?" in prompt
    assert "Blocked means waiting on someone else." in prompt
    assert "tools" not in composer_request


@pytest.mark.anyio
async def test_tool_error_is_passed_to_composer(task_api: TestClient):
    fake = FakeOpenAI(
        router_reply=reply(function_call("get_task", {"task_id": 999})),
        composer_reply=reply(text("Task 999 doesn't exist.")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        await assistant.ask("Show task 999", ADMIN)

    prompt = fake.composer_requests()[0]["input"]
    assert "ERROR:" in prompt and "Task 999 does not exist" in prompt


@pytest.mark.anyio
async def test_no_tool_needed_still_composes_answer():
    fake = FakeOpenAI(router_reply=reply(), composer_reply=reply(text("Hi! How can I help?")))

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
async def test_router_retries_once_when_provider_rejects_tool_arguments(task_api: TestClient):
    fake = FakeOpenAI(
        router_reply=reply(function_call("list_tasks", {"overdue": True})),
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
                    "Tool call validation failed",
                    response=httpx.Response(400, request=request),
                    body={"code": "tool_use_failed"},
                )
        return await original_create(**kwargs)

    fake.responses.create = create

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("Do I have overdue tasks?", ADMIN)

    assert calls == 3  # rejected, retried, then the round that ends the loop
    assert response.answer == "No overdue tasks."
    assert response.tool_calls == [ToolCall(name="list_tasks", input={"overdue": True})]


@pytest.mark.anyio
async def test_router_loops_to_cover_multi_part_questions(task_api: TestClient, stub_search):
    fake = FakeOpenAI(
        router_reply=[
            reply(function_call("search_knowledge_base", {"query": "leave"})),
            reply(function_call("list_tasks", {"overdue": True})),
            reply(text("done")),
        ],
        composer_reply=reply(text("Hand over your tasks. You have no overdue tasks.")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("I'm going on leave. Any overdue tasks?", ADMIN)

    assert [call.name for call in response.tool_calls] == ["search_knowledge_base", "list_tasks"]

    # Round 2 sees round 1's tool call and output, so the model knows what it already has.
    second_round_input = fake.router_requests()[1]["input"]
    assert [item.get("type", "message") for item in second_round_input] == [
        "message", "function_call", "function_call_output",
    ]

    # The composer gets both results.
    prompt = fake.composer_requests()[0]["input"]
    assert "Blocked means waiting on someone else." in prompt and 'tool="list_tasks"' in prompt


@pytest.mark.anyio
async def test_routing_stops_after_max_rounds(task_api: TestClient):
    from app.services.assistant import MAX_ROUTING_ROUNDS

    fake = FakeOpenAI(
        router_reply=[reply(function_call("list_tasks", {})) for _ in range(MAX_ROUTING_ROUNDS + 2)],
        composer_reply=reply(text("ok")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("loop forever", ADMIN)

    assert len(fake.router_requests()) == MAX_ROUTING_ROUNDS
    assert len(response.tool_calls) == MAX_ROUTING_ROUNDS


@pytest.mark.anyio
async def test_task_values_in_answer_come_from_mcp_data(task_api: TestClient):
    task = task_api.post("/tasks", json={"title": "Prepare demo", "priority": "high"}).json()
    tid = task["id"]
    fake = FakeOpenAI(
        router_reply=reply(function_call("get_task", {"task_id": tid})),
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
        router_reply=reply(function_call("get_task", {"task_id": tid})),
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
        router_reply=reply(function_call("list_tasks", {})),
        composer_reply=[reply(text("{{task:1.owner}}")), reply(text("{{task:1.owner}}"))],
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("who owns task 1?", ADMIN)

    assert response.answer == UNVERIFIED_ANSWER
