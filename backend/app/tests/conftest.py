from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from models import Base
from utils.database import engine


APP_DIRECTORY = Path(__file__).resolve().parents[1]


def get_alembic_config() -> Config:
    return Config(str(APP_DIRECTORY / "alembic.ini"))


@pytest.fixture(scope="session")
def test_database():
    """Build the test schema from Alembic and reject model/schema drift."""
    if engine.url.database != "gsk_test":
        raise RuntimeError(
            f"Tests must use gsk_test, not {engine.url.database!r}"
        )

    # Clean up schemas created by older test fixtures that used create_all().
    Base.metadata.drop_all(bind=engine)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))

    alembic_config = get_alembic_config()
    command.upgrade(alembic_config, "head")
    command.check(alembic_config)

    yield

    command.downgrade(alembic_config, "base")
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))


@pytest.fixture
def db_session(test_database) -> Session:
    """Run each database test inside a rolled-back transaction."""
    connection = engine.connect()
    transaction = connection.begin()

    testing_session = sessionmaker(
        bind=connection,
        autocommit=False,
        autoflush=False,
    )
    session = testing_session()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
