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


# 用例功能：验证使用合法用户名和密码可以成功注册用户。
# 执行步骤：
# 1. 向注册接口提交测试用户名和密码。
# 2. 验证响应状态码为 200。
# 3. 验证响应消息表明用户注册成功。
@pytest.mark.api
def test_register_user(client):
    response = register(client)

    assert response.status_code == 200
    assert response.json() == {
        "message": "User registered successfully"
    }


# 用例功能：验证注册接口会拒绝重复的用户名。
# 执行步骤：
# 1. 使用同一组用户名和密码完成首次注册。
# 2. 使用相同信息再次调用注册接口。
# 3. 验证首次请求成功，第二次请求返回 400。
# 4. 验证错误详情明确提示用户名已存在。
@pytest.mark.api
def test_register_rejects_duplicate_username(client):
    first_response = register(client)
    second_response = register(client)

    assert first_response.status_code == 200
    assert second_response.status_code == 400
    assert "用户名已存在" in second_response.json()["detail"]


# 用例功能：验证已注册用户使用正确密码登录后能获得 Bearer 访问令牌。
# 执行步骤：
# 1. 注册一个测试用户。
# 2. 使用该用户的正确凭据调用登录接口。
# 3. 验证注册和登录请求均成功。
# 4. 验证令牌类型为 bearer，且访问令牌非空。
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


# 用例功能：验证登录接口会拒绝密码错误的请求。
# 执行步骤：
# 1. 注册一个测试用户。
# 2. 使用正确用户名和错误密码调用登录接口。
# 3. 验证注册成功，登录请求返回 401。
# 4. 验证错误详情为统一的认证失败提示。
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
