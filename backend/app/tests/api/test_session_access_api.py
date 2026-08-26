import pytest

from service.auth import access_security


PASSWORD = "TestPassword123!"


def register_and_login(client, username: str) -> str:
    register_response = client.post(
        "/register",
        json={"username": username, "password": PASSWORD},
    )
    assert register_response.status_code == 200

    login_response = client.post(
        "/login",
        json={"username": username, "password": PASSWORD},
    )
    assert login_response.status_code == 200
    return login_response.json()["access_token"]


def authorization_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# 用例功能：验证创建会话后，后端会立即保存会话与当前用户的归属关系。
# 执行步骤：
# 1. 注册并登录一个用户。
# 2. 调用创建会话接口。
# 3. 查询该用户的会话列表。
# 4. 验证新会话已存在、默认名称正确，并且归属于当前用户。
@pytest.mark.api
def test_create_session_immediately_registers_owner(client):
    token = register_and_login(client, "session_owner")
    headers = authorization_header(token)

    create_response = client.post("/create_session", headers=headers)

    assert create_response.status_code == 200
    session_id = create_response.json()["session_id"]

    sessions_response = client.get("/get_sessions", headers=headers)
    assert sessions_response.status_code == 200
    assert isinstance(sessions_response.json()["user_id"], int)
    assert sessions_response.json()["sessions"] == [
        {
            "session_id": session_id,
            "session_name": "新对话",
            "user_id": sessions_response.json()["user_id"],
            "created_at": sessions_response.json()["sessions"][0]["created_at"],
            "updated_at": sessions_response.json()["sessions"][0]["updated_at"],
        }
    ]


# 用例功能：验证会话所有者可以正常读取一个尚无消息的新会话。
# 执行步骤：
# 1. 注册并登录会话所有者。
# 2. 创建一个新会话。
# 3. 使用所有者的令牌查询该会话的消息历史。
# 4. 验证请求成功，并返回空消息列表。
@pytest.mark.api
def test_session_owner_can_read_empty_history(client):
    token = register_and_login(client, "history_owner")
    headers = authorization_header(token)
    session_id = client.post("/create_session", headers=headers).json()["session_id"]

    response = client.get(
        "/get_messages",
        params={"session_id": session_id},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json() == []


# 用例功能：验证其他用户不能访问不属于自己的会话级资源。
# 执行步骤：
# 1. 分别注册并登录会话所有者和另一个用户。
# 2. 由所有者创建会话。
# 3. 使用另一个用户的令牌分别请求消息、解析内容、文档和聊天接口。
# 4. 验证所有请求均返回 404，且不会泄露会话是否真实存在。
@pytest.mark.api
@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("get", "/get_messages", {"params": {"session_id": "{session_id}"}}),
        ("get", "/get_parsed_content", {"params": {"session_id": "{session_id}"}}),
        ("get", "/sessions/{session_id}/documents", {}),
        ("get", "/sessions/{session_id}/documents/summary", {}),
        (
            "post",
            "/chat_on_docs",
            {"params": {"session_id": "{session_id}"}, "json": {"message": "hello"}},
        ),
    ],
)
def test_other_user_cannot_access_session(client, method, path, kwargs):
    owner_token = register_and_login(client, "resource_owner")
    other_token = register_and_login(client, "other_user")
    session_id = client.post(
        "/create_session",
        headers=authorization_header(owner_token),
    ).json()["session_id"]

    resolved_path = path.format(session_id=session_id)
    resolved_kwargs = {
        key: (
            {
                nested_key: nested_value.format(session_id=session_id)
                for nested_key, nested_value in value.items()
            }
            if isinstance(value, dict)
            else value
        )
        for key, value in kwargs.items()
    }
    response = getattr(client, method)(
        resolved_path,
        headers=authorization_header(other_token),
        **resolved_kwargs,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


# 用例功能：验证 JWT 即使签名有效，缺少 user_id 声明时也不能通过认证。
# 执行步骤：
# 1. 创建一个仅包含用户名、不包含 user_id 的访问令牌。
# 2. 使用该令牌请求用户会话列表。
# 3. 验证接口返回 401 和统一的无效认证提示。
@pytest.mark.api
def test_missing_user_id_claim_is_rejected(client):
    token = access_security.create_access_token(
        subject={"user_name": "claim_without_user_id"}
    )

    response = client.get(
        "/get_sessions",
        headers=authorization_header(token),
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authentication credentials"


# 用例功能：验证 JWT 的 user_id 不是整数时不能通过认证。
# 执行步骤：
# 1. 创建一个包含非整数 user_id 的已签名访问令牌。
# 2. 使用该令牌请求用户会话列表。
# 3. 验证接口返回 401，避免非法 ID 进入整数外键查询。
@pytest.mark.api
def test_non_integer_user_id_claim_is_rejected(client):
    token = access_security.create_access_token(
        subject={"user_id": "not-an-integer", "user_name": "invalid_claim"}
    )

    response = client.get(
        "/get_sessions",
        headers=authorization_header(token),
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid authentication credentials"
