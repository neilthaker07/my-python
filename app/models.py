import enum
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    # Stored as naive UTC (TIMESTAMP WITHOUT TIME ZONE), so the API's output format stays the same.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TaskStatus(str, enum.Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class TaskPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UserRole(str, enum.Enum):
    EMPLOYEE = "employee"
    MANAGER = "manager"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, native_enum=False))
    team: Mapped[str] = mapped_column(String(50), index=True)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, native_enum=False), default=TaskStatus.TODO, index=True
    )
    priority: Mapped[TaskPriority] = mapped_column(
        Enum(TaskPriority, native_enum=False), default=TaskPriority.MEDIUM
    )
    due_date: Mapped[Optional[date]] = mapped_column(index=True)
    completed_at: Mapped[Optional[datetime]]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    # Nullable only because tasks created before users existed are backfilled at startup.
    created_by_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), index=True)
    assignee_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), index=True)
    assignee: Mapped[Optional[User]] = relationship(foreign_keys=[assignee_id])


class AssistantTrace(Base):
    """One /assistant request and its online eval results (see app/services/online_evals.py)."""

    __tablename__ = "assistant_traces"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Returned to the caller, so a response can be matched to its trace.
    trace_id: Mapped[str] = mapped_column(String(32), unique=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[Optional[str]] = mapped_column(Text)
    error: Mapped[Optional[str]] = mapped_column(Text)  # the request failed
    # [{"tool": ..., "input": {...}, "output": "..."}]: the data the answer was based on.
    tool_results: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[int]
    router_model: Mapped[str] = mapped_column(String(100))
    answer_model: Mapped[str] = mapped_column(String(100))

    # Code checks, run on every trace: [{"check": "tool_error", "detail": "..."}]
    flags: Mapped[list] = mapped_column(JSON, default=list)

    # LLM judge, run on a sample of traces. All None when the trace wasn't judged.
    judge_model: Mapped[Optional[str]] = mapped_column(String(100))
    grounded: Mapped[Optional[bool]]
    complete: Mapped[Optional[bool]]
    judge_reasoning: Mapped[Optional[dict]] = mapped_column(JSON)  # criterion -> reasoning
    judge_error: Mapped[Optional[str]] = mapped_column(Text)
