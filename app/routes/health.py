from fastapi import APIRouter
from app.schemas import HealthResponse
from app.services.session import session_store

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Check service status")
def get_health():
    """Returns the API service health and current active session counts."""
    return HealthResponse(status="ok", active_sessions=session_store.active_count())


@router.get("/health/timings", summary="Get pipeline latency log")
def get_timings():
    """Reads the last few lines of the pipeline conversations log file to show performance timings."""
    import json
    from app.services.nlp_pipeline import LOG_FILE
    if not LOG_FILE.exists():
        return {"message": "No logs recorded yet"}
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        records = []
        for line in lines[-5:]:
            if line.strip():
                records.append(json.loads(line.strip()))
        return {"last_runs": records}
    except Exception as e:
        return {"error": str(e)}
