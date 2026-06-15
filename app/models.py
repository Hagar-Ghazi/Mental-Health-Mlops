from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    Float,
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    started_at = Column(DateTime, default=utcnow)
    turn_count = Column(Integer, default=0)
    prior_crisis = Column(Boolean, default=False)

    messages = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        String(64), ForeignKey("chat_sessions.session_id"), nullable=False
    )
    role = Column(String(16), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    emotion = Column(String(30), nullable=True)
    emotion_conf = Column(Float, nullable=True)
    language = Column(String(10), nullable=True)
    intent = Column(String(50), nullable=True)
    crisis_flag = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)

    session = relationship("ChatSession", back_populates="messages")


class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    vote = Column(String(10), nullable=False)  # "up" or "down"
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)


class CrisisEvent(Base):
    __tablename__ = "crisis_events"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), nullable=False, index=True)
    trigger_text = Column(Text, nullable=False)
    detected_at = Column(DateTime, default=utcnow)
