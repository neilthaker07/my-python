"""Task business logic, kept separate from HTTP so it can be reused later
(e.g. as LLM tools in Stage 2 or LangGraph nodes in Stage 4)."""

from datetime import date, datetime, time
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Task, TaskStatus, User, utcnow
from app.schemas import TaskCreate, TaskUpdate
from app.services.permissions import visible_tasks


def create_task(db: Session, data: TaskCreate, created_by: User, assignee: User) -> Task:
    """Permission checks happen before this; see app.services.permissions."""
    task = Task(
        **data.model_dump(exclude={"assignee_id"}),
        created_by_id=created_by.id,
        assignee_id=assignee.id,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_task(db: Session, task_id: int) -> Optional[Task]:
    return db.get(Task, task_id)


def list_tasks(
    db: Session,
    *,
    status: Optional[TaskStatus] = None,
    overdue: bool = False,
    due_on: Optional[date] = None,
    completed_since: Optional[date] = None,
    today: Optional[date] = None,
    visible_to: Optional[User] = None,
    newest_first: bool = False,
    limit: Optional[int] = None,
) -> list[Task]:
    """Tasks ordered by due date (undated last), or by creation time with `newest_first`."""
    today = today or date.today()
    stmt = _visible(select(Task), visible_to)

    if status is not None:
        stmt = stmt.where(Task.status == status)
    if overdue:
        stmt = stmt.where(Task.due_date < today, Task.status != TaskStatus.DONE)
    if due_on is not None:
        stmt = stmt.where(Task.due_date == due_on)
    if completed_since is not None:
        stmt = stmt.where(Task.completed_at >= datetime.combine(completed_since, time.min))

    if newest_first:
        stmt = stmt.order_by(Task.created_at.desc(), Task.id.desc())
    else:
        stmt = stmt.order_by(Task.due_date.is_(None), Task.due_date, Task.id)
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt))


def update_task(db: Session, task: Task, data: TaskUpdate) -> Task:
    changes = data.model_dump(exclude_unset=True)

    if "status" in changes:
        _set_status(task, changes.pop("status"))
    for field, value in changes.items():
        setattr(task, field, value)

    db.commit()
    db.refresh(task)
    return task


def delete_task(db: Session, task: Task) -> None:
    db.delete(task)
    db.commit()


def reschedule_unfinished(
    db: Session, from_date: date, to_date: date, visible_to: Optional[User] = None
) -> list[Task]:
    """Move every task due on `from_date` that isn't done to `to_date`."""
    stmt = _visible(select(Task), visible_to).where(Task.due_date == from_date, Task.status != TaskStatus.DONE)
    tasks = list(db.scalars(stmt))
    for task in tasks:
        task.due_date = to_date
    db.commit()
    return tasks


def _visible(stmt, user: Optional[User]):
    """Limit a query to the tasks `user` may see (no limit when user is None)."""
    condition = visible_tasks(user) if user is not None else None
    return stmt if condition is None else stmt.where(condition)


def _set_status(task: Task, status: TaskStatus) -> None:
    if status == TaskStatus.DONE and task.status != TaskStatus.DONE:
        task.completed_at = utcnow()
    elif status != TaskStatus.DONE:
        task.completed_at = None
    task.status = status
