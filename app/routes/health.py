from fastapi import APIRouter
from app.schemas import HealthResponse
from app.services.session import session_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Check service status")
def get_health():
    """Returns the API service health and current active session counts."""
    return HealthResponse(status="ok", active_sessions=session_store.active_count())
