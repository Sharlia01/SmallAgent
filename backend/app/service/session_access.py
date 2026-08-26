from fastapi import HTTPException, status
from fastapi_jwt import JwtAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.orm import Session as DatabaseSession

from models.session import Session as ChatSession


DEFAULT_SESSION_NAME = "新对话"


def get_authenticated_user_id(
    credentials: JwtAuthorizationCredentials,
) -> int:
    """Return the authenticated user id without accepting missing token claims."""
    subject = credentials.subject
    raw_user_id = subject.get("user_id") if isinstance(subject, dict) else None
    if isinstance(raw_user_id, bool) or not isinstance(raw_user_id, (int, str)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    if isinstance(raw_user_id, str) and not raw_user_id.isdecimal():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    user_id = int(raw_user_id)

    if user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

    return user_id


def require_owned_session(
    db: DatabaseSession,
    session_id: str,
    user_id: int,
) -> ChatSession:
    """Return a session only when it belongs to the authenticated user."""
    chat_session = db.execute(
        select(ChatSession).where(
            ChatSession.session_id == session_id,
            ChatSession.user_id == user_id,
        )
    ).scalar_one_or_none()

    if chat_session is None:
        # Use the same response for a missing session and somebody else's session.
        # This avoids revealing which session ids exist.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return chat_session
