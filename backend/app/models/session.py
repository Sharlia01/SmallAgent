from sqlalchemy import Column, ForeignKey, Index, Integer, String, TIMESTAMP
from sqlalchemy.sql import func

from models.base import Base


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        Index("idx_sessions_user_id", "user_id"),
        Index("idx_sessions_created_at", "created_at"),
    )

    session_id = Column(String(16), primary_key=True)
    session_name = Column(String(255), nullable=False)
    user_id = Column(
        Integer,
        ForeignKey(
            "users.id",
            name="fk_sessions_user_id_users",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
