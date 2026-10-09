"""Calls the MCP tools through a real MCP client connected to the server in-process.

The task tools hit the FastAPI app through httpx's ASGI transport, so no server
needs to be running. The knowledge-base search is stubbed out, so Postgres isn't needed.
"""

import json
from datetime import date, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp import Client

from app import mcp_server
from app.auth import USER_META_KEY
from app.main import app
from app.rag.store import SearchHit
from tests.conftest import ADMIN_ID

AS_ADMIN = {USER_META_KEY: ADMIN_ID}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def task_api(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # `client` installs the in-memory database override; route MCP HTTP calls to the same app.
    monkeypatch.setattr(
        mcp_server,
        "_task_api",
        lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test"),
    )
    return client


def result_json(result) -> object:
    return json.loads(result.content[0].text)


@pytest.mark.anyio
async def test_server_exposes_expected_tools():
    async with Client(mcp_server.mcp) as mcp:
        tools = (await mcp.list_tools()).tools

    assert {tool.name for tool in tools} == {"list_tasks", "get_task", "create_task", "list_users", "search_knowledge_base"}


@pytest.mark.anyio
async def test_list_tasks_wraps_get_endpoint(task_api: TestClient):
    yesterday = str(date.today() - timedelta(days=1))
    task_api.post("/tasks", json={"title": "overdue", "due_date": yesterday})
    task_api.post("/tasks", json={"title": "no date"})

    async with Client(mcp_server.mcp) as mcp:
        result = await mcp.call_tool("list_tasks", {"overdue": True}, meta=AS_ADMIN)

    assert result_json(result) == {"tasks": [result_json(result)["tasks"][0]], "more_tasks_exist": False}
    assert [t["title"] for t in result_json(result)["tasks"]] == ["overdue"]


@pytest.mark.anyio
async def test_list_tasks_returns_only_the_newest_and_says_more_exist(task_api: TestClient):
    for n in range(mcp_server.LIST_TASKS_LIMIT + 2):
        task_api.post("/tasks", json={"title": f"task {n}"})

    async with Client(mcp_server.mcp) as mcp:
        result = await mcp.call_tool("list_tasks", {}, meta=AS_ADMIN)

    newest = [f"task {n}" for n in reversed(range(2, mcp_server.LIST_TASKS_LIMIT + 2))]
    assert [t["title"] for t in result_json(result)["tasks"]] == newest
    assert result_json(result)["more_tasks_exist"] is True


@pytest.mark.anyio
async def test_get_task_returns_task(task_api: TestClient):
    task = task_api.post("/tasks", json={"title": "Prepare for interview"}).json()

    async with Client(mcp_server.mcp) as mcp:
        result = await mcp.call_tool("get_task", {"task_id": task["id"]}, meta=AS_ADMIN)

    assert result_json(result)["title"] == "Prepare for interview"


@pytest.mark.anyio
async def test_get_missing_task_is_a_tool_error(task_api: TestClient):
    async with Client(mcp_server.mcp) as mcp:
        result = await mcp.call_tool("get_task", {"task_id": 999}, meta=AS_ADMIN)

    assert result.is_error
    assert "999" in result.content[0].text


@pytest.mark.anyio
async def test_search_knowledge_base_returns_hits(monkeypatch: pytest.MonkeyPatch):
    hit = SearchHit(id=5, category="tasks", question="When is a task overdue?", answer="...", score=0.9)
    calls = []
    monkeypatch.setattr(mcp_server.store, "search", lambda q, k: calls.append((q, k)) or [hit])

    async with Client(mcp_server.mcp) as mcp:
        result = await mcp.call_tool("search_knowledge_base", {"query": "overdue rules", "top_k": 1})

    assert calls == [("overdue rules", 1)]
    assert result_json(result)[0]["id"] == 5
