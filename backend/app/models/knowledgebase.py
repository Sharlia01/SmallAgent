from sqlalchemy import Column, Index, Integer, String, TIMESTAMP
from sqlalchemy.sql import func

from models.base import Base


class KnowledgeBase(Base):
    __tablename__ = "knowledgebases"
    __table_args__ = (
        Index("idx_knowledgebases_user_id", "user_id"),
        Index("idx_knowledgebases_created_at", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
