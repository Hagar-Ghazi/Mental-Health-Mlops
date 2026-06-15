import time
from fastapi import Request, status
from fastapi.responses import JSONResponse
from collections import defaultdict
from typing import Dict, List

# In-memory store mapping client IP to list of request timestamps
_request_history: Dict[str, List[float]] = defaultdict(list)

# Rate limit thresholds: 20 requests per minute
RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_MAX_REQUESTS = 20


async def rate_limiter_middleware(request: Request, call_next):
    """Custom middleware to rate limit request traffic to /chat and /feedback.

    Returns HTTP 429 Too Many Requests if the limit is exceeded.
    """
    path = request.url.path
    # Only rate limit chat and feedback endpoints to keep health checks open
    if path not in ("/chat", "/feedback"):
        return await call_next(request)

    # Extract client IP resolving reverse proxies
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    now = time.time()

    # Prune timestamps outside of the sliding window
    timestamps = _request_history[client_ip]
    active_timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
    _request_history[client_ip] = active_timestamps

    if len(active_timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "You're sending messages too quickly. Please wait a moment and try again."
            },
        )

    # Log timestamp and proceed
    _request_history[client_ip].append(now)

    return await call_next(request)
