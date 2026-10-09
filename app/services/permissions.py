"""Authorization rules: who may see, create, change and delete which tasks.

- Employees see and change tasks they created or are assigned to, and can only
  create tasks for themselves.
- Managers can also see and change every task assigned to their team, and can
  create tasks for anyone on their team.
- Admins can do everything.
- Only managers and admins can bulk-reschedule, and only within what they can see.
- Deleting: admins, the team's manager, or the creator while the task is still 'todo'.
- Online eval traces hold every user's questions and task data, so only admins can read them.
- Creating through the MCP create_task tool (the assistant) is limited further, to
  the roles in MCP_CREATE_TASK_ROLES (managers and admins by default).

Every check lives here and runs in the API, never in the LLM. Kept free of HTTP so
routers, services and tests can all use it.
"""

from typing import Optional

from sqlalchemy import ColumnElement, or_, select

from app.config import settings
from app.models import Task, TaskStatus, User, UserRole

# Parsed at import, so a typo in MCP_CREATE_TASK_ROLES fails at startup, not mid-request.
MCP_CREATE_TASK_ROLES = frozenset(
    UserRole(role.strip()) for role in settings.mcp_create_task_roles.split(",") if role.strip()
)


class PermissionDenied(Exception):
    """The user is authenticated but not allowed to do this. The API returns 403."""


def visible_tasks(user: User) -> Optional[ColumnElement[bool]]:
    """SQL filter for the tasks `user` may see (None means all of them)."""
    if user.role == UserRole.ADMIN:
        return None
    own = or_(Task.created_by_id == user.id, Task.assignee_id == user.id)
    if user.role == UserRole.MANAGER:
        team_members = select(User.id).where(User.team == user.team)
        return or_(own, Task.assignee_id.in_(team_members))
    return own


def can_access(user: User, task: Task) -> bool:
    """May `user` see and change `task`? Same rule as `visible_tasks`, for one task."""
    if user.role == UserRole.ADMIN or user.id in (task.created_by_id, task.assignee_id):
        return True
    return user.role == UserRole.MANAGER and task.assignee is not None and task.assignee.team == user.team


def check_can_assign(user: User, assignee: User) -> None:
    if user.role == UserRole.ADMIN:
        return
    if user.role == UserRole.MANAGER:
        if assignee.team != user.team:
            raise PermissionDenied("Managers can only create tasks for members of their own team.")
        return
    if assignee.id != user.id:
        raise PermissionDenied("Employees can only create tasks for themselves.")


def check_can_delete(user: User, task: Task) -> None:
    if user.role == UserRole.ADMIN:
        return
    if user.role == UserRole.MANAGER and task.assignee is not None and task.assignee.team == user.team:
        return
    if task.created_by_id == user.id and task.status == TaskStatus.TODO:
        return
    raise PermissionDenied("Only the team manager, or the creator while the task is 'todo', can delete it.")


def check_can_reschedule(user: User) -> None:
    if user.role == UserRole.EMPLOYEE:
        raise PermissionDenied("Only managers and admins can bulk-reschedule tasks.")


def can_create_via_mcp(role: UserRole) -> bool:
    return role in MCP_CREATE_TASK_ROLES


def check_can_create_via_mcp(role: UserRole) -> None:
    if not can_create_via_mcp(role):
        allowed = ", ".join(sorted(r.value for r in MCP_CREATE_TASK_ROLES)) or "nobody"
        raise PermissionDenied(f"Creating tasks through the assistant is limited to: {allowed}.")


def check_can_view_traces(user: User) -> None:
    if user.role != UserRole.ADMIN:
        raise PermissionDenied("Only admins can view assistant traces.")
