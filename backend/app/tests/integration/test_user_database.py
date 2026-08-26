import pytest

from models.user import User


@pytest.mark.integration
def test_can_create_and_query_user(db_session):
    user = User(
        username="integration_test_user",
        password_hash="fake-test-hash",
    )

    db_session.add(user)
    db_session.flush()

    saved_user = (
        db_session.query(User)
        .filter(User.username == "integration_test_user")
        .one()
    )

    assert saved_user.id is not None
    assert saved_user.username == "integration_test_user"
    assert saved_user.password_hash == "fake-test-hash"