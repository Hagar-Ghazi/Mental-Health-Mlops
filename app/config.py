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
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "HagarGhazi/emotion-classifier-mental-health")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Groq Model name (using a standard llama fallback if the instruction model is unavailable)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Paths to Local Model Artifacts
ARTIFACTS_DIR = BASE_DIR / "app" / "artifacts"
LANGUAGE_DETECTOR_PATH = ARTIFACTS_DIR / "language_detector.joblib"
INTENT_ARTIFACTS_DIR = ARTIFACTS_DIR / "intent_classifier"

# OpenTelemetry & Observability Config
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "serenity-chatbot-backend")
