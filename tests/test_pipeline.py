from app.services.nlp_pipeline import _detect_quick_response, nlp_pipeline
from app.services.session import SessionMemory
from app.services.rag import rag_service


def test_detect_quick_response():
    # Greetings
    assert _detect_quick_response("hello")[0] == "greeting"
    assert _detect_quick_response("صباح الخير")[0] == "greeting"

    # Gratitude
    assert _detect_quick_response("thank you so much")[0] == "gratitude"

    # Goodbye
    assert _detect_quick_response("goodbye")[0] == "goodbye"

    # Out of scope
    assert _detect_quick_response("whats the weather")[0] == "out_of_scope"

    # None
    assert _detect_quick_response("i need help with depression") is None


def test_pipeline_quick_response_routing():
    session = SessionMemory(session_id="test_ip")
    result = nlp_pipeline.run("hi", session)

    assert result["intent"] == "greeting"
    assert result["crisis_flag"] is False
    assert len(result["sources"]) == 0
    assert "Safeness" not in result["answer"]  # should use greeting text


def test_rag_reranking_logic(monkeypatch):
    # Test that rag_service correctly boost matches for priority topics
    # Mock embedding and Qdrant call
    monkeypatch.setattr(rag_service, "_embed", lambda text: [0.1] * 384)

    mock_results = [
        # Match topics sadness -> loneliness
        {
            "context": "anxiety and fear coping",
            "response": "anxiety response",
            "topics": ["anxiety"],
            "similarity": 0.50,
            "has_empathy": False,
        },
        {
            "context": "loneliness counselor talk",
            "response": "loneliness response",
            "topics": ["loneliness"],
            "similarity": 0.45,
            "has_empathy": True,
        },
    ]

    # Mock the Qdrant retrieval method
    def mock_query(*args, **kwargs):
        class MockPoint:
            def __init__(self, payload, score):
                self.payload = payload
                self.score = score

        return type(
            "MockPoints",
            (),
            {
                "points": [
                    MockPoint(mock_results[0], 0.50),
                    MockPoint(mock_results[1], 0.45),
                ]
            },
        )()

    class MockQdrantClient:
        def query_points(self, *args, **kwargs):
            return mock_query()

    # Assign MockQdrantClient directly to avoid AttributeError
    rag_service._qdrant_client = MockQdrantClient()
    monkeypatch.setattr(rag_service, "_load", lambda: None)

    # Restore the original retrieve_and_rerank method since it was mocked globally in conftest.py
    from app.services.rag import RAGService

    monkeypatch.setattr(
        rag_service,
        "retrieve_and_rerank",
        RAGService.retrieve_and_rerank.__get__(rag_service, RAGService),
    )

    # With sadness the second document should be boosted to first place due to "loneliness" topic match (boost = 0.08) and "has_empathy" boost = 0.05
    # Document 1 score: 0.50 + 0 = 0.50
    # Document 2 score: 0.45 + 0.08 + 0.05 = 0.58 (Reranked first)
    reranked = rag_service.retrieve_and_rerank("lonely", emotion="sadness")
    assert len(reranked) == 2
    assert "loneliness counselor" in reranked[0]["context"]
