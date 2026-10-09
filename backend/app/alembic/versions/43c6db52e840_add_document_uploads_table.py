"""新增 document_uploads 表。

版本编号：43c6db52e840
前序版本：980b32f130df
创建时间：2025-06-19 22:39:20.210028
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# Alembic 使用的迁移版本标识。
revision: str = '43c6db52e840'
down_revision: Union[str, None] = '980b32f130df'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """升级数据库结构。"""
    # 创建 document_uploads 文档上传记录表。
    op.create_table(
        'document_uploads',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(length=16), nullable=False),
        sa.Column('document_name', sa.String(length=255), nullable=False),
        sa.Column('document_type', sa.String(length=50), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('upload_time', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 创建索引。
    op.create_index('idx_document_uploads_session_id', 'document_uploads', ['session_id'])
    op.create_index('idx_document_uploads_upload_time', 'document_uploads', ['upload_time'])


def downgrade() -> None:
    """回退数据库结构。"""
    # 删除索引。
    op.drop_index('idx_document_uploads_upload_time', table_name='document_uploads')
    op.drop_index('idx_document_uploads_session_id', table_name='document_uploads')
    
    # 删除表。
    op.drop_table('document_uploads')
