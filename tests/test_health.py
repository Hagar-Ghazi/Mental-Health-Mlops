def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["active_sessions"] == 0


def test_health_session_counts(client):
    # Simulate a chat message to create a session in memory
    chat_response = client.post("/chat", json={"message": "Hello"})
    assert chat_response.status_code == 200

    # Check that health endpoint reports 1 active session
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["active_sessions"] == 1
