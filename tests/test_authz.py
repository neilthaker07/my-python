"""Authorization rules, checked at every layer: REST API, MCP tools, and the assistant.

Sample users: Alice (1) and Bob (2) are employees on team platform, Carol (3) manages
platform, Dave (4) is an employee on team sales, Erin (5) is an admin.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client

from app import mcp_server
from app.auth import USER_HEADER, USER_META_KEY
from app.main import app
from app.models import UserRole
from app.schemas import UserRead
from app.services.assistant import TaskAssistant
from tests.conftest import ADMIN_ID, ALICE_ID, BOB_ID, CAROL_ID, DAVE_ID
from tests.test_assistant_service import FakeOpenAI, offered_tools, plan, reply, text


def as_user(user_id: int) -> dict[str, str]:
    return {USER_HEADER: str(user_id)}


def create(client: TestClient, user_id: int, **fields) -> httpx.Response:
    return client.post("/tasks", json={"title": "t", **fields}, headers=as_user(user_id))


# --- Authentication -----------------------------------------------------------


def test_requests_without_a_user_are_rejected(client: TestClient):
    anonymous = TestClient(app)  # no default X-User-Id header
    assert anonymous.get("/tasks").status_code == 401
    assert anonymous.post("/assistant", json={"question": "hi"}).status_code == 401


def test_unknown_user_is_rejected(client: TestClient):
    assert client.get("/tasks", headers=as_user(999)).status_code == 401


# --- Creating tasks ------------------------------------------------------------


def test_employee_creates_task_for_themselves_by_default(client: TestClient):
    task = create(client, ALICE_ID).json()
    assert task["created_by_id"] == ALICE_ID and task["assignee_id"] == ALICE_ID


def test_employee_cannot_create_task_for_someone_else(client: TestClient):
    response = create(client, ALICE_ID, assignee_id=BOB_ID)
    assert response.status_code == 403
    assert response.json()["detail"] == "Employees can only create tasks for themselves."


def test_manager_can_create_tasks_for_their_team_only(client: TestClient):
    assert create(client, CAROL_ID, assignee_id=BOB_ID).status_code == 201
    response = create(client, CAROL_ID, assignee_id=DAVE_ID)
    assert response.status_code == 403
    assert "own team" in response.json()["detail"]


def test_admin_can_create_tasks_for_anyone(client: TestClient):
    assert create(client, ADMIN_ID, assignee_id=DAVE_ID).status_code == 201


def test_unknown_assignee_is_a_validation_error(client: TestClient):
    assert create(client, ADMIN_ID, assignee_id=999).status_code == 422


# --- Seeing and changing tasks ------------------------------------------------


def test_employees_only_see_their_own_tasks(client: TestClient):
    alices = create(client, ALICE_ID).json()
    bobs = create(client, BOB_ID).json()

    assert [t["id"] for t in client.get("/tasks", headers=as_user(ALICE_ID)).json()] == [alices["id"]]
    # Someone else's task looks missing rather than forbidden, so ids don't leak.
    assert client.get(f"/tasks/{bobs['id']}", headers=as_user(ALICE_ID)).status_code == 404
    assert client.patch(f"/tasks/{bobs['id']}", json={"title": "x"}, headers=as_user(ALICE_ID)).status_code == 404


def test_manager_sees_their_teams_tasks_but_not_other_teams(client: TestClient):
    bobs = create(client, BOB_ID).json()
    daves = create(client, DAVE_ID).json()

    visible = {t["id"] for t in client.get("/tasks", headers=as_user(CAROL_ID)).json()}
    assert bobs["id"] in visible and daves["id"] not in visible


def test_delete_rules(client: TestClient):
    task = create(client, ALICE_ID).json()
    client.patch(f"/tasks/{task['id']}", json={"status": "in_progress"}, headers=as_user(ALICE_ID))

    # The creator can only delete while the task is 'todo'; the team manager always can.
    assert client.delete(f"/tasks/{task['id']}", headers=as_user(ALICE_ID)).status_code == 403
    assert client.delete(f"/tasks/{task['id']}", headers=as_user(CAROL_ID)).status_code == 204


def test_only_managers_and_admins_can_bulk_reschedule(client: TestClient):
    body = {"from_date": "2026-01-01", "to_date": "2026-01-02"}
    assert client.post("/tasks/reschedule", json=body, headers=as_user(ALICE_ID)).status_code == 403
    assert client.post("/tasks/reschedule", json=body, headers=as_user(CAROL_ID)).status_code == 200


# --- MCP tools act as the user in the request metadata -------------------------


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def task_api(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        mcp_server,
        "_task_api",
        lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test"),
    )
    return client


@pytest.mark.anyio
async def test_create_task_tool_uses_the_identity_from_metadata(task_api: TestClient):
    async with Client(mcp_server.mcp) as mcp:
        result = await mcp.call_tool("create_task", {"title": "Write report"}, meta={USER_META_KEY: CAROL_ID})

    assert json.loads(result.content[0].text)["created_by_id"] == CAROL_ID


@pytest.mark.anyio
async def test_create_task_tool_reports_api_permission_denial(task_api: TestClient):
    # Carol may use the tool, but the API's rule still applies: Dave isn't on her team.
    async with Client(mcp_server.mcp) as mcp:
        result = await mcp.call_tool(
            "create_task", {"title": "x", "assignee_id": DAVE_ID}, meta={USER_META_KEY: CAROL_ID}
        )

    assert result.is_error
    assert "403: Managers can only create tasks for members of their own team." in result.content[0].text


@pytest.mark.anyio
async def test_employees_cannot_create_tasks_via_mcp_but_can_via_the_api(task_api: TestClient):
    async with Client(mcp_server.mcp) as mcp:
        result = await mcp.call_tool("create_task", {"title": "Mine"}, meta={USER_META_KEY: ALICE_ID})

    assert result.is_error
    assert "403: Creating tasks through the assistant is limited to: admin, manager." in result.content[0].text
    assert task_api.get("/tasks").json() == []  # the check runs before anything is created

    # The MCP rule is stricter than the API: the same task via REST is fine.
    assert create(task_api, ALICE_ID, title="Mine").status_code == 201


@pytest.mark.anyio
async def test_mcp_create_roles_are_configurable(task_api: TestClient, monkeypatch: pytest.MonkeyPatch):
    from app.services import permissions

    monkeypatch.setattr(permissions, "MCP_CREATE_TASK_ROLES", frozenset(UserRole))
    async with Client(mcp_server.mcp) as mcp:
        result = await mcp.call_tool("create_task", {"title": "Mine"}, meta={USER_META_KEY: ALICE_ID})

    assert not result.is_error


@pytest.mark.anyio
async def test_tools_refuse_calls_without_a_user(task_api: TestClient):
    async with Client(mcp_server.mcp) as mcp:
        result = await mcp.call_tool("create_task", {"title": "x"})

    assert result.is_error and "No user on this request" in result.content[0].text


@pytest.mark.anyio
async def test_identity_is_not_a_tool_argument():
    async with Client(mcp_server.mcp) as mcp:
        tools = {tool.name: tool for tool in (await mcp.list_tools()).tools}

    assert set(tools["create_task"].input_schema["properties"]) == {
        "title", "description", "priority", "due_date", "assignee_id",
    }


# --- The assistant can't escalate the user's permissions -----------------------

ALICE = UserRead(id=ALICE_ID, name="Alice", role=UserRole.EMPLOYEE, team="platform")
CAROL = UserRead(id=CAROL_ID, name="Carol", role=UserRole.MANAGER, team="platform")


@pytest.mark.anyio
async def test_create_task_is_not_offered_to_users_who_cant_use_it(task_api: TestClient):
    fake = FakeOpenAI(router_reply=plan(), composer_reply=reply(text("You can't create tasks here.")))

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        await assistant.ask("Create a task 'Fix bug'", ALICE)
        await assistant.ask("Create a task 'Fix bug'", CAROL)

    alice_request, carol_request = fake.router_requests()
    assert "create_task" not in offered_tools(alice_request)
    assert "create_task" in offered_tools(carol_request)
    # Both models are told, so the answer can explain why nothing was created.
    assert "can't create tasks through the assistant" in fake.composer_requests()[0]["instructions"]


@pytest.mark.anyio
async def test_mcp_rule_holds_even_if_the_model_calls_a_hidden_tool(task_api: TestClient):
    # The model (wrongly) calls create_task although it wasn't offered to Alice.
    fake = FakeOpenAI(
        router_reply=plan(("create_task", {"title": "Fix bug"})),
        composer_reply=reply(text("You can't create tasks through the assistant.")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        await assistant.ask("Create a task 'Fix bug'", ALICE)

    assert "403: Creating tasks through the assistant is limited to" in fake.composer_requests()[0]["input"]
    assert task_api.get("/tasks").json() == []


@pytest.mark.anyio
async def test_assistant_cannot_create_tasks_the_user_isnt_allowed_to(task_api: TestClient):
    # The model tries to create a task for Dave (sales) on Carol's (platform) behalf.
    fake = FakeOpenAI(
        router_reply=plan(("create_task", {"title": "Fix bug", "assignee_id": DAVE_ID})),
        composer_reply=reply(text("You can only create tasks for your team.")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        await assistant.ask("Create a task 'Fix bug' for Dave", CAROL)

    prompt = fake.composer_requests()[0]["input"]
    assert "403: Managers can only create tasks for members of their own team." in prompt
    assert task_api.get("/tasks").json() == []  # nothing was created


@pytest.mark.anyio
async def test_assistant_creates_as_the_asking_user_and_never_twice(task_api: TestClient):
    create_call = ("create_task", {"title": "Prepare demo"})
    fake = FakeOpenAI(
        # The model plans the same create twice, then again in round 2; it must run once.
        router_reply=[plan(create_call, create_call, needs_results=True), plan(create_call)],
        composer_reply=reply(text("Created task.")),
    )

    async with TaskAssistant(mcp_server.mcp, openai_client=fake) as assistant:
        response = await assistant.ask("Create a task 'Prepare demo'", CAROL)

    tasks = task_api.get("/tasks").json()
    assert [(t["title"], t["created_by_id"], t["assignee_id"]) for t in tasks] == [("Prepare demo", CAROL_ID, CAROL_ID)]
    assert [call.name for call in response.tool_calls] == ["create_task"]
