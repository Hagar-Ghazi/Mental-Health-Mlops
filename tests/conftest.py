import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Mock environment variables BEFORE importing application modules
os.environ["GROQ_API_KEY"] = "mock-groq-key"
os.environ["HF_TOKEN"] = "mock-hf-token"
os.environ["QDRANT_URL"] = "https://mock-qdrant.io"
os.environ["QDRANT_API_KEY"] = "mock-qdrant-key"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from app.database import Base, get_db
from app.main import app
from app.services.session import session_store

# In-memory SQLite database for testing
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function", autouse=True)
def setup_test_db():
    """Creates tables before each test and drops them afterward."""
    Base.metadata.create_all(bind=engine)
    # Clear rate limiter request history to avoid test interference
    from app.utils.rate_limiter import _request_history
    _request_history.clear()
    yield
    Base.metadata.drop_all(bind=engine)
    # Clear in-memory session memory
    session_store._sessions.clear()

@pytest.fixture(scope="function")
def db_session():
    """Provides a transactional database session for unit testing."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="function")
def client(db_session):
    """Provides a TestClient with database session overrides."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
            
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# Mocks for NLP services to prevent network calls during testing
@pytest.fixture(autouse=True)
def mock_external_calls(monkeypatch):
    """Mocks network-dependent calls in Groq, Qdrant, and Transformers."""
    # Mock language detection
    from app.services.language import language_detector
    def mock_detect(text):
        return {
            "prediction": "en",
            "lang_name": "English",
            "confidence": 0.99,
            "trusted": True
        }
    monkeypatch.setattr(language_detector, "detect", mock_detect)


    # Mock emotion classification
    from app.services.emotion import emotion_classifier
    def mock_classify(text, threshold=0.40):
        return {
            "emotion": "joy",
            "confidence": 0.85,
            "risk_flag": False,
            "tone": "Encouraging, celebratory, engaging"
        }
    monkeypatch.setattr(emotion_classifier, "classify", mock_classify)


    # Mock Groq classifier completion
    from app.services.intent import intent_classifier
    def mock_intent_classify(text, detected_emotion=None, detected_language=None):
        return {
            "intent": "greeting",
            "routing": "direct",
            "crisis_flag": False,
            "response_style": "empathetic_support",
            "confidence": "high"
        }
    monkeypatch.setattr(intent_classifier, "classify", mock_intent_classify)


    # Mock Qdrant retrieval
    from app.services.rag import rag_service
    def mock_retrieve(query, emotion=None):
        return [
            {
                "context": "Context doc for anxiety management",
                "response": "Response guidance excerpt",
                "topics": ["anxiety", "stress"],
                "risk_level": "low",
                "quality_score": 5,
                "has_empathy": True,
                "similarity": 0.82
            }
        ]
    monkeypatch.setattr(rag_service, "retrieve_and_rerank", mock_retrieve)


    # Mock Groq Chat completions in nlp_pipeline
    from app.services.nlp_pipeline import nlp_pipeline
    def mock_call_llm(query, prompt, history):
        if "⚠ CRISIS CONTEXT ACTIVE" in prompt:
            return "I hear you, you are not alone. Please call the hotline: 988"
        return "This is a mocked empathetic therapist response."
    monkeypatch.setattr(nlp_pipeline, "_call_therapist_llm", mock_call_llm)
