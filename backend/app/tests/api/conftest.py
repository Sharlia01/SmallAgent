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
    """确保每个接口测试开始和结束时，应用表均为空。"""
    delete_test_data()

    yield

    delete_test_data()


@pytest.fixture
def client(test_database):
    """提供 FastAPI 测试客户端，无需启动真实 HTTP 服务器。"""
    with TestClient(app) as test_client:
        yield test_client
