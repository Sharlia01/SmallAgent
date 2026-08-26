import pytest


USERNAME = "api_test_user"
PASSWORD = "TestPassword123!"


def register(client, username=USERNAME, password=PASSWORD):
    return client.post(
        "/register",
        json={
            "username": username,
            "password": password,
        },
    )


@pytest.mark.api
def test_register_user(client):
    response = register(client)

    assert response.status_code == 200
    assert response.json() == {
        "message": "User registered successfully"
    }


@pytest.mark.api
def test_register_rejects_duplicate_username(client):
    first_response = register(client)
    second_response = register(client)

    assert first_response.status_code == 200
    assert second_response.status_code == 400
    assert "用户名已存在" in second_response.json()["detail"]


@pytest.mark.api
def test_login_returns_access_token(client):
    register_response = register(client)

    response = client.post(
        "/login",
        json={
            "username": USERNAME,
            "password": PASSWORD,
        },
    )

    assert register_response.status_code == 200
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"
    assert response.json()["access_token"]


@pytest.mark.api
def test_login_rejects_wrong_password(client):
    register_response = register(client)

    response = client.post(
        "/login",
        json={
            "username": USERNAME,
            "password": "WrongPassword123!",
        },
    )

    assert register_response.status_code == 200
    assert response.status_code == 401
    assert response.json()["detail"] == "认证失败"