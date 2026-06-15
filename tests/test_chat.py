from app.models import Message, ChatSession


def test_chat_happy_path(client, db_session):
    response = client.post("/chat", json={"message": "I feel anxiety"})
    assert response.status_code == 200

    data = response.json()
    assert "response" in data
    assert "answer" in data
    assert data["answer"] == data["response"]
    assert "session_id" in data
    assert data["emotion"] == "joy"  # matches conftest mock
    assert data["crisis_flag"] is False

    # Check database messages logged
    messages = db_session.query(Message).all()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "I feel anxiety"
    assert messages[1].role == "assistant"


def test_chat_empty_message(client):
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 400
    assert "Message cannot be empty" in response.json()["detail"]

    response_spaces = client.post("/chat", json={"message": "    "})
    assert response_spaces.status_code == 400


def test_chat_crisis_flagged(client, monkeypatch, db_session):
    # Mock pipeline output to simulate a crisis event
    from app.services.nlp_pipeline import nlp_pipeline

    def mock_run_crisis(query, session, country="United States"):
        return {
            "answer": "Please stay safe! We are here for you.",
            "sources": [],
            "emotion": "sadness",
            "emotion_conf": 0.95,
            "language": "en",
            "intent": "asking_mental_health_question",
            "crisis_flag": True,
            "latency_ms": 150.0,
            "rag_scores": [],
        }

    monkeypatch.setattr(nlp_pipeline, "run", mock_run_crisis)

    response = client.post("/chat", json={"message": "I want to harm myself"})
    assert response.status_code == 200
    assert response.json()["crisis_flag"] is True

    # Check DB session has prior_crisis=True and crisis log was created
    session_row = db_session.query(ChatSession).first()
    assert session_row is not None
    assert session_row.prior_crisis is True


def test_rate_limiter(client):
    from app.utils.rate_limiter import _request_history

    _request_history.clear()

    # Send 20 requests (within limit)
    for _ in range(20):
        response = client.post("/chat", json={"message": "hello"})
        assert response.status_code == 200

    # 21st request must trigger 429
    response_429 = client.post("/chat", json={"message": "hello too fast"})
    assert response_429.status_code == 429
    assert "You're sending messages too quickly" in response_429.json()["detail"]
