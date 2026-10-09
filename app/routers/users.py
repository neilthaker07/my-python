from fastapi import APIRouter

from app.auth import CurrentUser
from app.database import DbSession
from app.models import User
from app.schemas import UserRead
from app.services import users as service

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserRead])
def list_users(db: DbSession, user: CurrentUser) -> list[User]:
    """Everyone can see the user directory (names, roles, teams), e.g. to pick an assignee."""
    return service.list_users(db)


@router.get("/me", response_model=UserRead)
def me(user: CurrentUser) -> User:
    return user
