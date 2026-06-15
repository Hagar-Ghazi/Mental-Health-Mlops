from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base
from app.routes import chat, feedback, health
from app.utils.logging import app_logger
from app.utils.rate_limiter import rate_limiter_middleware
import app.utils.metrics as metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for application startup and shutdown events."""
    app_logger.info("Starting up Serenity Chatbot Backend API...")

    # 1. Initialize SQLite Database Tables
    try:
        Base.metadata.create_all(bind=engine)
        app_logger.info("Database tables verified/created successfully.")
    except Exception as e:
        app_logger.error(f"Failed to create database tables: {e}", exc_info=True)

    # 2. Initialize OpenTelemetry Metrics Export
    metrics.init_metrics()

    # 3. Preload ML Models asynchronously
    app_logger.info(
        "Preloading ML models in background (this may take a minute on first run)..."
    )
    import threading
    from app.services.emotion import emotion_classifier
    from app.services.language import language_detector
    from app.services.rag import rag_service
    from app.services.intent import intent_classifier

    def preload_models():
        try:
            emotion_classifier._load()
            app_logger.info("✅ Emotion model loaded.")
            language_detector._load()
            app_logger.info("✅ Language model loaded.")
            rag_service._load()
            app_logger.info("✅ RAG model loaded.")
            intent_classifier._load()
            app_logger.info("✅ Intent classifier loaded.")
            app_logger.info("🎉 All ML models preloaded successfully!")
        except Exception as e:
            app_logger.error(f"❌ Failed to preload ML models: {e}", exc_info=True)

    threading.Thread(target=preload_models, daemon=True).start()

    yield
    app_logger.info("Shutting down Serenity Chatbot Backend API...")


app = FastAPI(
    title="Serenity Chatbot API",
    description="Empathetic mental health support API powering the Serenity chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS to allow access from the GitHub Pages frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allows any origin in student testing, can restrict to github pages in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate Limiting Middleware (exceeded requests return 429)
@app.middleware("http")
async def rate_limit(request: Request, call_next):
    return await rate_limiter_middleware(request, call_next)


# OpenTelemetry Request/Error Monitoring Middleware
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    path = request.url.path
    method = request.method

    # Skip spam logging for health check
    is_health = path == "/health"

    if not is_health:
        metrics.http_requests_counter.add(1, {"route": path, "method": method})
        app_logger.info(f"Incoming Request: {method} {path}")

    try:
        response = await call_next(request)

        if not is_health and response.status_code >= 400:
            metrics.http_errors_counter.add(
                1, {"route": path, "status": str(response.status_code)}
            )
            app_logger.warning(
                f"Request failed: {method} {path} - Status {response.status_code}"
            )

        return response
    except Exception as e:
        if not is_health:
            metrics.http_errors_counter.add(1, {"route": path, "status": "500"})
            app_logger.error(
                f"Internal server error handling request {method} {path}: {e}",
                exc_info=True,
            )
        raise e


# Mount routers
app.include_router(health.router, tags=["Utility"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(feedback.router, tags=["Feedback"])


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Serenity Chatbot API. Serve GET /health or POST /chat to interact.",
        "docs": "/docs",
    }
