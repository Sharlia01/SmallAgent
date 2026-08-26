import pytest
from fastapi.testclient import TestClient

from app_main import app
from models.user import User
from utils.database import SessionLocal


def delete_test_users():
    db = SessionLocal()
    try:
        db.query(User).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def clean_users(test_database):
    """Ensure every API test starts and ends with an empty users table."""
    delete_test_users()

    yield

    delete_test_users()


@pytest.fixture
def client(test_database):
    """Provide a FastAPI test client without starting a real HTTP server."""
    with TestClient(app) as test_client:
        yield test_client