"""Users and the sample people the app starts with."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import User, UserRole

SAMPLE_USERS = [
    User(id=1, name="Alice", role=UserRole.EMPLOYEE, team="platform"),
    User(id=2, name="Bob", role=UserRole.EMPLOYEE, team="platform"),
    User(id=3, name="Carol", role=UserRole.MANAGER, team="platform"),
    User(id=4, name="Dave", role=UserRole.EMPLOYEE, team="sales"),
    User(id=5, name="Erin", role=UserRole.ADMIN, team="it"),
]


def seed_sample_users(db: Session) -> None:
    """Add the sample users if there are no users yet."""
    if db.scalar(select(User.id).limit(1)) is None:
        db.add_all(User(id=u.id, name=u.name, role=u.role, team=u.team) for u in SAMPLE_USERS)
        db.commit()


def list_users(db: Session) -> list[User]:
    return list(db.scalars(select(User).order_by(User.id)))
