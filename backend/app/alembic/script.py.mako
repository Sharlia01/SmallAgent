"""${message}

版本编号：${up_revision}
前序版本：${down_revision | comma,n}
创建时间：${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# Alembic 使用的迁移版本标识。
revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    """升级数据库结构。"""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """回退数据库结构。"""
    ${downgrades if downgrades else "pass"}
