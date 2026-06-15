import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the serenity-backend project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env in the project root
load_dotenv(BASE_DIR / ".env")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mental_health.db")

# NLP Model Credentials & Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_NAME = os.getenv(
    "HF_MODEL_NAME", "HagarGhazi/emotion-classifier-mental-health"
)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Groq Model name (kept for fallback compatibility)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Google Gemini Model Configuration (Primary LLM Provider)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Paths to Local Model Artifacts
ARTIFACTS_DIR = BASE_DIR / "app" / "artifacts"
LANGUAGE_DETECTOR_PATH = ARTIFACTS_DIR / "language_detector.joblib"
INTENT_ARTIFACTS_DIR = ARTIFACTS_DIR / "intent_classifier"

# OpenTelemetry & Observability Config
ENABLE_TELEMETRY = os.getenv("ENABLE_TELEMETRY", "False").lower() in ("true", "1", "t")
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv(
    "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
)
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "serenity-chatbot-backend")
