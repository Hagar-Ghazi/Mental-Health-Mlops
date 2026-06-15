import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SessionMemory:
    """
    Manages state and history for a single chat session.
    Keeps a sliding window of the last 6 turns (12 messages) of history.
    """

    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.history: List[Dict[str, str]] = []
        self.prior_crisis: bool = False
        self.emotion_history: List[str] = []
        self.topics_discussed: List[str] = []
        self.turn_count: int = 0
        self.started_at: datetime = utcnow()
        self.last_active: datetime = utcnow()

    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        emotion: Optional[str] = None,
        emotion_conf: Optional[float] = None,
        language: Optional[str] = None,
        intent: Optional[str] = None,
        crisis_flag: bool = False,
        topics: Optional[List[str]] = None,
    ) -> None:
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_response})

        # Slide window to retain only the last 6 turns (12 messages)
        if len(self.history) > 12:
            self.history = self.history[-12:]

        if crisis_flag:
            self.prior_crisis = True

        if emotion:
            self.emotion_history.append(emotion)

        if topics:
            for topic in topics:
                if topic and topic not in self.topics_discussed:
                    self.topics_discussed.append(topic)

        self.turn_count += 1
        self.last_active = utcnow()

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "prior_crisis": self.prior_crisis,
            "emotion_history": self.emotion_history,
            "topics_discussed": self.topics_discussed,
            "started_at": self.started_at.isoformat(),
            "last_active": self.last_active.isoformat(),
        }


class SessionStore:
    """In-memory dictionary mapping session_id (e.g. Client IP or UUID) to SessionMemory."""

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}

    def get_or_create(self, session_id: str) -> SessionMemory:
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory(session_id=session_id)
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def active_count(self) -> int:
        return len(self._sessions)


# Global store instance
session_store = SessionStore()
