from sqlalchemy import Column, ForeignKey, Index, Integer, String, TIMESTAMP
from sqlalchemy.sql import func

from models.base import Base


class DocumentUpload(Base):
    __tablename__ = "document_uploads"
    __table_args__ = (
        Index("idx_document_uploads_session_id", "session_id"),
        Index("idx_document_uploads_upload_time", "upload_time"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(16),
        ForeignKey(
            "sessions.session_id",
            name="fk_document_uploads_session_id_sessions",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    document_name = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)
    file_size = Column(Integer)
    upload_time = Column(TIMESTAMP, nullable=False, server_default=func.now())
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
