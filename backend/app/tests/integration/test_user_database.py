import pytest

from models.user import User


# 用例功能：验证用户记录可以写入数据库并按用户名查询。
# 执行步骤：
# 1. 构造包含用户名和密码哈希的 User 对象。
# 2. 将用户加入会话并 flush，使其写入测试数据库。
# 3. 按用户名查询刚写入的记录。
# 4. 验证用户 ID 已生成，用户名和密码哈希保存正确。
@pytest.mark.integration
def test_can_create_and_query_user(db_session):
    user = User(
        username="integration_test_user",
        password_hash="fake-test-hash",
    )

    db_session.add(user)
    db_session.flush()

    #查询刚写入的记录
    saved_user = (
        db_session.query(User)
        .filter(User.username == "integration_test_user")
        .one()
    )

    assert saved_user.id is not None
    assert saved_user.username == "integration_test_user"
    assert saved_user.password_hash == "fake-test-hash"
