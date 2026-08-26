import pytest

from utils.password import hash_password, verify_password


@pytest.mark.unit
def test_hash_password_does_not_store_plain_text():
    password = "TestPassword123!"
    password_hash = hash_password(password)

    assert password_hash != password


@pytest.mark.unit
def test_verify_password_accepts_correct_password():
    password = "TestPassword123!"
    password_hash = hash_password(password)

    assert verify_password(password, password_hash) is True


@pytest.mark.unit
def test_verify_password_rejects_wrong_password():
    password_hash = hash_password("CorrectPassword123!")

    assert verify_password("WrongPassword123!", password_hash) is False