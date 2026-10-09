"""MCP server exposing task tools to the assistant (or to any MCP client).

- list_tasks / get_task / create_task wrap the /tasks endpoints over HTTP.
- list_users wraps GET /users, so the model can find someone's id by name.
- search_knowledge_base runs a semantic search over the pgvector FAQ.

Task tools act as the user named in the request metadata (see app.auth.USER_META_KEY).
The REST API checks that user's permissions. create_task also has its own, stricter
rule: only the roles in MCP_CREATE_TASK_ROLES may create tasks through MCP.

Run standalone (stdio transport): python -m app.mcp_server
"""

import json
from dataclasses import asdict
from datetime import date
from typing import Annotated, Any, Optional

import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from app.auth import USER_HEADER, USER_META_KEY
from app.config import settings
from app.models import TaskPriority, TaskStatus, UserRole
from app.rag import store
from app.services.permissions import PermissionDenied, check_can_create_via_mcp

mcp = MCPServer(
    "task-assistant",
    instructions="Read and create the user's tasks and search the task-management knowledge base.",
)


def _task_api() -> httpx.AsyncClient:
    """HTTP client for the task REST API. Tests replace this to call the app in-process."""
    return httpx.AsyncClient(base_url=settings.task_api_url, timeout=10.0)


def _user_id(ctx: Context) -> int:
    meta = ctx.request_context.meta or {}
    user_id = meta.get(USER_META_KEY, settings.mcp_default_user_id)
    if user_id is None:
        raise ToolError(f"No user on this request: send {USER_META_KEY!r} in the request _meta.")
    return user_id


async def _request(ctx: Context, method: str, path: str, **kwargs: Any) -> Any:
    """Call the task API as the request's user. API refusals become tool errors the model can read."""
    headers = {USER_HEADER: str(_user_id(ctx))}
    async with _task_api() as client:
        response = await client.request(method, path, headers=headers, **kwargs)
    if response.status_code in (401, 403, 404, 422):
        # ToolError makes MCP return an error result with this message, so the model
        # knows why the call failed. Other exceptions only say "Error executing tool".
        raise ToolError(f"{response.status_code}: {response.json().get('detail')}")
    response.raise_for_status()
    return response.json()


@mcp.tool()
async def list_tasks(
    ctx: Context,
    status: Optional[TaskStatus] = None,
    overdue: bool = False,
    due_on: Optional[date] = None,
    completed_since: Optional[date] = None,
) -> str:
    """List the tasks the user can see. All filters are optional and can be combined.

    Args:
        status: Only tasks with this status.
        overdue: Only tasks whose due date has passed and that are not done.
        due_on: Only tasks due on this date (YYYY-MM-DD).
        completed_since: Only tasks completed on or after this date (YYYY-MM-DD).
    """
    params: dict[str, Any] = {"overdue": overdue}
    if status is not None:
        params["status"] = status.value
    if due_on is not None:
        params["due_on"] = due_on.isoformat()
    if completed_since is not None:
        params["completed_since"] = completed_since.isoformat()
    return json.dumps(await _request(ctx, "GET", "/tasks", params=params))


@mcp.tool()
async def get_task(ctx: Context, task_id: int) -> str:
    """Get one task by its id.

    Args:
        task_id: The task id.
    """
    try:
        return json.dumps(await _request(ctx, "GET", f"/tasks/{task_id}"))
    except ToolError as exc:
        if str(exc).startswith("404"):
            raise ToolError(f"Task {task_id} does not exist") from exc
        raise


@mcp.tool()
async def create_task(
    ctx: Context,
    title: Annotated[str, Field(min_length=1, max_length=200)],
    description: Optional[str] = None,
    priority: TaskPriority = TaskPriority.MEDIUM,
    due_date: Optional[date] = None,
    assignee_id: Optional[int] = None,
) -> str:
    """Create a task. Only call this when the user explicitly asks to create a task.

    Args:
        title: Short title of the task.
        description: Optional details.
        priority: low, medium or high.
        due_date: Optional due date (YYYY-MM-DD).
        assignee_id: Id of the user the task is for (find it with list_users).
            Leave empty to assign the task to the current user.
    """
    # MCP-only rule, checked against the user's role as the API reports it.
    me = await _request(ctx, "GET", "/users/me")
    try:
        check_can_create_via_mcp(UserRole(me["role"]))
    except PermissionDenied as exc:
        raise ToolError(f"403: {exc}") from exc

    body: dict[str, Any] = {"title": title, "description": description, "priority": priority.value}
    if due_date is not None:
        body["due_date"] = due_date.isoformat()
    if assignee_id is not None:
        body["assignee_id"] = assignee_id
    return json.dumps(await _request(ctx, "POST", "/tasks", json=body))


@mcp.tool()
async def list_users(ctx: Context) -> str:
    """List all users (id, name, role, team), e.g. to find the id of an assignee by name."""
    return json.dumps(await _request(ctx, "GET", "/users"))


@mcp.tool()
def search_knowledge_base(
    query: str,
    top_k: Annotated[int, Field(ge=1, le=10)] = 3,
) -> str:
    """Semantic search over the task-management knowledge base: policies, task
    statuses, priorities, permissions, overdue rules, and how-to guidance.

    Args:
        query: A natural-language question, e.g. "who can change the assignee?"
        top_k: How many entries to return.
    """
    return json.dumps([asdict(hit) for hit in store.search(query, top_k)])


if __name__ == "__main__":
    mcp.run()
