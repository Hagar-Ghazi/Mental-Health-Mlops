from app.models import Feedback


def test_submit_valid_feedback(client, db_session):
    feedback_data = {
        "vote": "up",
        "user_message": "Hello, I feel anxious.",
        "bot_response": "I hear you, let us work on it.",
    }
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 201
    assert response.json()["status"] == "success"

    # Verify database persistence
    db_feedback = db_session.query(Feedback).first()
    assert db_feedback is not None
    assert db_feedback.vote == "up"
    assert db_feedback.user_message == "Hello, I feel anxious."
    assert db_feedback.bot_response == "I hear you, let us work on it."


def test_submit_invalid_feedback(client):
    feedback_data = {
        "vote": "invalid-vote-type",
        "user_message": "Hello",
        "bot_response": "Hi",
    }
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 400
    assert "Vote must be either 'up' or 'down'" in response.json()["detail"]
