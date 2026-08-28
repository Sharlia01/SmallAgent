import pytest

from utils.password import hash_password, verify_password


# 用例功能：验证密码哈希结果不会以明文形式存储密码。
# 执行步骤：
# 1. 准备一个明文密码。
# 2. 调用密码哈希函数生成存储值。
# 3. 验证生成的哈希值与明文密码不同。
@pytest.mark.unit
def test_hash_password_does_not_store_plain_text():
    password = "TestPassword123!"
    password_hash = hash_password(password)

    assert password_hash != password


# 用例功能：验证正确密码能通过密码哈希校验。
# 执行步骤：
# 1. 对测试密码生成哈希值。
# 2. 使用原始密码和哈希值调用校验函数。
# 3. 验证校验结果为 True。
@pytest.mark.unit
def test_verify_password_accepts_correct_password():
    password = "TestPassword123!"
    password_hash = hash_password(password)

    assert verify_password(password, password_hash) is True


# 用例功能：验证错误密码无法通过密码哈希校验。
# 执行步骤：
# 1. 对正确密码生成哈希值。
# 2. 使用另一个错误密码和该哈希值调用校验函数。
# 3. 验证校验结果为 False。
@pytest.mark.unit
def test_verify_password_rejects_wrong_password():
    password_hash = hash_password("CorrectPassword123!")

    assert verify_password("WrongPassword123!", password_hash) is False
