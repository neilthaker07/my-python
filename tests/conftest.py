from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth import USER_HEADER
from app.database import Base, get_db
from app.main import app
from app.services.users import seed_sample_users


# Sample users from app.services.users
ALICE_ID, BOB_ID, CAROL_ID, DAVE_ID, ADMIN_ID = 1, 2, 3, 4, 5


@pytest.fixture
def db_session() -> Iterator[Session]:
    # Fresh in-memory database per test. StaticPool makes every connection
    # share the same in-memory DB (otherwise each one would get an empty DB).
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with TestingSession() as session:
        seed_sample_users(session)
        yield session
    engine.dispose()


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db_session
    # Not used as a context manager, so the app lifespan (which touches the
    # real database file) doesn't run. Acts as the admin (user 5) unless a test
    # sends its own X-User-Id header.
    yield TestClient(app, headers={USER_HEADER: str(ADMIN_ID)})
    app.dependency_overrides.clear()
