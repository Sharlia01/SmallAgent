import pytest
from fastapi.testclient import TestClient

from app_main import app
from models.document_upload import DocumentUpload
from models.knowledgebase import KnowledgeBase
from models.message import Message
from models.session import Session as ChatSession
from models.user import User
from utils.database import SessionLocal


def delete_test_data():
    db = SessionLocal()
    try:
        db.query(DocumentUpload).delete(synchronize_session=False)
        db.query(Message).delete(synchronize_session=False)
        db.query(ChatSession).delete(synchronize_session=False)
        db.query(KnowledgeBase).delete(synchronize_session=False)
        db.query(User).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def clean_api_data(test_database):
    """Ensure every API test starts and ends with empty application tables."""
    delete_test_data()

    yield

    delete_test_data()


@pytest.fixture
def client(test_database):
    """Provide a FastAPI test client without starting a real HTTP server."""
    with TestClient(app) as test_client:
        yield test_client
