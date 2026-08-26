"""add relational integrity

Revision ID: f475785c40c5
Revises: 43c6db52e840
Create Date: 2026-08-26 08:50:25.770374

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f475785c40c5"
down_revision: Union[str, None] = "43c6db52e840"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
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
    """Downgrade schema."""
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
