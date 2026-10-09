import os
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.auth import USER_HEADER
from app.config import settings
from app.database import Base, get_db
from app.main import app
from app.services.users import seed_sample_users


# Sample users from app.services.users
ALICE_ID, BOB_ID, CAROL_ID, DAVE_ID, ADMIN_ID = 1, 2, 3, 4, 5


def _test_database_url() -> URL:
    """TEST_DATABASE_URL, or the app's database server with "_test" added to the name,
    so tests never touch your real tasks."""
    if "TEST_DATABASE_URL" in os.environ:
        return make_url(os.environ["TEST_DATABASE_URL"])
    url = make_url(settings.database_url)
    return url.set(database=f"{url.database}_test")


def _create_database(url: URL) -> None:
    server = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with server.connect() as conn:
            exists = conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": url.database})
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    except OperationalError as exc:
        pytest.exit(f"The tests need Postgres at {url.render_as_string()}: {exc.orig}", returncode=1)
    finally:
        server.dispose()


@pytest.fixture(scope="session")
def test_engine() -> Iterator[Engine]:
    url = _test_database_url()
    _create_database(url)
    engine = create_engine(url)
    # Recreate the schema once per run, in case the models changed since the last one.
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(test_engine: Engine) -> Iterator[Session]:
    # Empty tables per test, with ids restarting at 1, like a fresh database.
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with test_engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    TestingSession = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)
    with TestingSession() as session:
        seed_sample_users(session)
        yield session


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db_session
    # Not used as a context manager, so the app lifespan (which touches the
    # real database) doesn't run. Acts as the admin (user 5) unless a test
    # sends its own X-User-Id header.
    yield TestClient(app, headers={USER_HEADER: str(ADMIN_ID)})
    app.dependency_overrides.clear()
