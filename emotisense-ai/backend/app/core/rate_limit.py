"""Phase 18 — Simple in-process rate limiter for development/demo.

Uses a sliding-window counter per client IP stored in a thread-safe
in-memory dict.  For production, replace with Redis-backed middleware.

Defaults: 30 requests / 60 seconds per IP per endpoint prefix.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Optional

from fastapi import HTTPException, Request, status

_LOCK = Lock()
# {client_ip: deque of timestamps}
_WINDOWS: dict[str, deque] = defaultdict(deque)

DEFAULT_MAX_REQUESTS = 30
DEFAULT_WINDOW_SECONDS = 60


def check_rate_limit(
    request: Request,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> None:
    """Raise HTTP 429 if client IP exceeds the rate limit.

    Intended to be called as a FastAPI dependency.
    """
    client_ip = _get_client_ip(request)
    now = time.monotonic()

    with _LOCK:
        window = _WINDOWS[client_ip]
        # Remove expired timestamps
        while window and window[0] < now - window_seconds:
            window.popleft()
        if len(window) >= max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {max_requests} requests "
                    f"per {window_seconds} seconds."
                ),
                headers={"Retry-After": str(window_seconds)},
            )
        window.append(now)


def _get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"
