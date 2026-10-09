"""补充数据库关系完整性约束。

版本编号：f475785c40c5
前序版本：43c6db52e840
创建时间：2026-08-26 08:50:25.770374
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# Alembic 使用的迁移版本标识。
revision: str = "f475785c40c5"
down_revision: Union[str, None] = "43c6db52e840"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """升级数据库结构。"""
    op.create_foreign_key(
        "fk_document_uploads_session_id_sessions",
        "document_uploads",
        "sessions",
        ["session_id"],
        ["session_id"],
        ondelete="CASCADE",
    )
    op.alter_column(
        "knowledgebases",
        "user_id",
        existing_type=sa.VARCHAR(length=255),
        type_=sa.Integer(),
        existing_nullable=False,
        postgresql_using="user_id::integer",
    )
    op.create_foreign_key(
        "fk_knowledgebases_user_id_users",
        "knowledgebases",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_messages_session_id_sessions",
        "messages",
        "sessions",
        ["session_id"],
        ["session_id"],
        ondelete="CASCADE",
    )
    op.alter_column(
        "sessions",
        "user_id",
        existing_type=sa.VARCHAR(length=255),
        type_=sa.Integer(),
        existing_nullable=False,
        postgresql_using="user_id::integer",
    )
    op.create_foreign_key(
        "fk_sessions_user_id_users",
        "sessions",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """回退数据库结构。"""
    op.drop_constraint(
        "fk_sessions_user_id_users", "sessions", type_="foreignkey"
    )
    op.alter_column(
        "sessions",
        "user_id",
        existing_type=sa.Integer(),
        type_=sa.VARCHAR(length=255),
        existing_nullable=False,
        postgresql_using="user_id::varchar(255)",
    )
    op.drop_constraint(
        "fk_messages_session_id_sessions", "messages", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_knowledgebases_user_id_users",
        "knowledgebases",
        type_="foreignkey",
    )
    op.alter_column(
        "knowledgebases",
        "user_id",
        existing_type=sa.Integer(),
        type_=sa.VARCHAR(length=255),
        existing_nullable=False,
        postgresql_using="user_id::varchar(255)",
    )
    op.drop_constraint(
        "fk_document_uploads_session_id_sessions",
        "document_uploads",
        type_="foreignkey",
    )
