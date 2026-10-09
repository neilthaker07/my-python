from datetime import date
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.auth import CurrentUser
from app.database import DbSession
from app.models import Task, TaskStatus, User
from app.schemas import RescheduleRequest, RescheduleResult, TaskCreate, TaskRead, TaskUpdate
from app.services import permissions
from app.services import tasks as service

# Every endpoint needs a user; PermissionDenied from the rules becomes a 403 (see app.main).
router = APIRouter(prefix="/tasks", tags=["tasks"])


def get_task_or_404(task_id: int, db: DbSession, user: CurrentUser) -> Task:
    task = service.get_task(db, task_id)
    # A task the user may not see is reported as missing, so its existence doesn't leak.
    if task is None or not permissions.can_access(user, task):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return task


ExistingTask = Annotated[Task, Depends(get_task_or_404)]


@router.post("", response_model=TaskRead, status_code=status.HTTP_201_CREATED)
def create_task(data: TaskCreate, db: DbSession, user: CurrentUser) -> Task:
    assignee = user if data.assignee_id is None else db.get(User, data.assignee_id)
    if assignee is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"User {data.assignee_id} does not exist")
    permissions.check_can_assign(user, assignee)
    return service.create_task(db, data, created_by=user, assignee=assignee)


@router.get("", response_model=list[TaskRead])
def list_tasks(
    db: DbSession,
    user: CurrentUser,
    status: Optional[TaskStatus] = None,
    overdue: bool = Query(False, description="Only tasks past their due date that aren't done"),
    due_on: Optional[date] = None,
    completed_since: Optional[date] = Query(None, description="Tasks completed on or after this date"),
) -> list[Task]:
    return service.list_tasks(
        db, status=status, overdue=overdue, due_on=due_on, completed_since=completed_since, visible_to=user
    )


@router.post("/reschedule", response_model=RescheduleResult)
def reschedule_unfinished(data: RescheduleRequest, db: DbSession, user: CurrentUser) -> RescheduleResult:
    permissions.check_can_reschedule(user)
    moved = service.reschedule_unfinished(db, data.from_date, data.to_date, visible_to=user)
    return RescheduleResult(moved=len(moved), tasks=moved)


@router.get("/{task_id}", response_model=TaskRead)
def get_task(task: ExistingTask) -> Task:
    return task


@router.patch("/{task_id}", response_model=TaskRead)
def update_task(task: ExistingTask, data: TaskUpdate, db: DbSession) -> Task:
    return service.update_task(db, task, data)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task: ExistingTask, db: DbSession, user: CurrentUser) -> None:
    permissions.check_can_delete(user, task)
    service.delete_task(db, task)
