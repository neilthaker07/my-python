"""Pydantic models: the shape of data going in and out of the API."""

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models import TaskPriority, TaskStatus, UserRole


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[date] = None
    assignee_id: Optional[int] = Field(default=None, description="Defaults to the user creating the task")


class TaskUpdate(BaseModel):
    """Partial update: only fields present in the request body are changed."""

    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    due_date: Optional[date] = None

    @field_validator("title", "status", "priority")
    @classmethod
    def reject_explicit_null(cls, value: Any) -> Any:
        # Validators don't run on defaults, so this only fires when the client
        # sends e.g. {"title": null}. description/due_date may be cleared.
        if value is None:
            raise ValueError("may not be null")
        return value


class TaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: Optional[str]
    status: TaskStatus
    priority: TaskPriority
    due_date: Optional[date]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    created_by_id: Optional[int]
    assignee_id: Optional[int]


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    role: UserRole
    team: str


class RescheduleRequest(BaseModel):
    from_date: date
    to_date: date


class RescheduleResult(BaseModel):
    moved: int
    tasks: list[TaskRead]


class AssistantRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ToolCall(BaseModel):
    name: str
    input: dict[str, Any]


class AssistantResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCall]
