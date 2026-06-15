from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user message to the chatbot")


class ChatResponse(BaseModel):
    response: str = Field(..., description="The generated empathetic response")
    answer: str = Field(
        ..., description="Duplicate of response for frontend compatibility"
    )
    session_id: str = Field(..., description="The session ID associated with this chat")
    emotion: Optional[str] = Field(None, description="Classified emotion label")
    emotion_conf: Optional[float] = Field(
        None, description="Confidence of emotion classification"
    )
    language: str = Field(..., description="Detected language code")
    intent: str = Field(..., description="Classified user intent")
    crisis_flag: bool = Field(..., description="Whether a crisis was flagged")


class FeedbackRequest(BaseModel):
    vote: str = Field(..., description="Feedback vote: 'up' or 'down'")
    user_message: str = Field(..., description="The original user query")
    bot_response: str = Field(
        ..., description="The generated bot response being evaluated"
    )


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Status of the API server")
    active_sessions: int = Field(
        ..., description="Number of currently active sessions in memory"
    )
