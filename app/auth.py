"""Who is making the request.

Dev-grade authentication: the caller sends `X-User-Id: <id>` and we trust it.
Anyone can claim any id, so this is for learning only. To use real auth (JWT,
OAuth), replace `get_current_user`; everything else depends only on `CurrentUser`.
"""

from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, status

from app.database import DbSession
from app.models import User

USER_HEADER = "X-User-Id"

# Where the assistant puts the user's id in MCP request metadata. It travels outside
# the tool arguments, so the LLM can neither see nor change whose identity a call uses.
USER_META_KEY = "task-assistant/user_id"


def get_current_user(
    db: DbSession,
    x_user_id: Annotated[Optional[int], Header(description="Id of the user making the request")] = None,
) -> User:
    if x_user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Send the {USER_HEADER} header")
    user = db.get(User, x_user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Unknown user {x_user_id}")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
