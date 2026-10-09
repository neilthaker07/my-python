"""Startup database setup: create tables, apply small schema upgrades, seed users.

Good enough for learning. Swap for Alembic migrations once the schema settles.
"""

from sqlalchemy import inspect, text, update

from app.database import Base, SessionLocal, engine
from app.models import Task
from app.services.users import SAMPLE_USERS, seed_sample_users

# Columns added to `tasks` after the first version. create_all() doesn't add
# columns to a table that already exists, so we add them here.
_NEW_TASK_COLUMNS = ("created_by_id", "assignee_id")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _add_missing_task_columns()
    with SessionLocal() as db:
        seed_sample_users(db)
        # Tasks from before users existed belong to the first sample user.
        owner = SAMPLE_USERS[0].id
        db.execute(update(Task).where(Task.created_by_id.is_(None)).values(created_by_id=owner))
        db.execute(update(Task).where(Task.assignee_id.is_(None)).values(assignee_id=owner))
        db.commit()


def _add_missing_task_columns() -> None:
    existing = {column["name"] for column in inspect(engine).get_columns("tasks")}
    with engine.begin() as conn:
        for column in _NEW_TASK_COLUMNS:
            if column not in existing:
                conn.execute(text(f"ALTER TABLE tasks ADD COLUMN {column} INTEGER REFERENCES users(id)"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_tasks_{column} ON tasks ({column})"))
