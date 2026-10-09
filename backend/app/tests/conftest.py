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
    """通过 Alembic 创建测试结构，并检查模型与数据库结构是否一致。"""
    if engine.url.database != "gsk_test":
        raise RuntimeError(
            f"Tests must use gsk_test, not {engine.url.database!r}"
        )

    # 清理旧测试夹具通过 create_all() 创建的数据库结构。
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
    """将每个数据库测试放在结束后回滚的事务中执行。"""
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
