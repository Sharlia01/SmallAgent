import pytest
from sqlalchemy.exc import IntegrityError

from models.document_upload import DocumentUpload
from models.knowledgebase import KnowledgeBase
from models.message import Message
from models.session import Session as ChatSession
from models.user import User


@pytest.mark.integration
def test_deleting_user_cascades_to_owned_records(db_session):
    user = User(
        username="cascade_test_user",
        password_hash="fake-test-hash",
    )
    db_session.add(user)
    db_session.flush()

    chat_session = ChatSession(
        session_id="cascade_session1",
        session_name="Cascade test",
        user_id=user.id,
    )
    db_session.add(chat_session)
    db_session.flush()

    db_session.add_all(
        [
            Message(
                session_id=chat_session.session_id,
                user_question="Question",
                model_answer="Answer",
            ),
            DocumentUpload(
                session_id=chat_session.session_id,
                document_name="cascade.txt",
                document_type="txt",
            ),
            KnowledgeBase(
                user_id=user.id,
                file_name="cascade.txt",
            ),
        ]
    )
    db_session.flush()

    db_session.delete(user)
    db_session.flush()

    assert db_session.query(ChatSession).count() == 0
    assert db_session.query(Message).count() == 0
    assert db_session.query(DocumentUpload).count() == 0
    assert db_session.query(KnowledgeBase).count() == 0


@pytest.mark.integration
def test_message_requires_existing_session(db_session):
    with pytest.raises(IntegrityError):
        with db_session.begin_nested():
            db_session.add(
                Message(
                    session_id="missing_session1",
                    user_question="Question",
                    model_answer="Answer",
                )
            )
            db_session.flush()

    assert (
        db_session.query(Message)
        .filter(Message.session_id == "missing_session1")
        .count()
        == 0
    )
