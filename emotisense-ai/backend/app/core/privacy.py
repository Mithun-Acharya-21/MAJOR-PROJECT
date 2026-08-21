"""Phase 18 — Privacy controls and upload validation.

Rules enforced here
-------------------
- Maximum upload size checked before reading content.
- Allowed MIME types enforced per modality.
- Temporary files are never written; data is processed in memory.
- File content is read once, validated, and discarded.
- No raw audio, image, or video is persisted by default.
- Request IDs are pseudonymous (UUID4, no user identity).
"""

from __future__ import annotations

import io
import uuid
from typing import Optional

from fastapi import HTTPException, UploadFile, status

ALLOWED_AUDIO_TYPES = frozenset({"audio/wav", "audio/wave", "audio/x-wav", "audio/mpeg",
                                  "audio/mp4", "audio/ogg", "audio/webm"})
ALLOWED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_VIDEO_TYPES = frozenset({"video/mp4", "video/webm", "video/quicktime"})

# File-extension fallbacks when content-type is not set correctly
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg", ".m4a", ".webm"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov"})

_HTTP_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE",
                    getattr(status, "HTTP_413_REQUEST_ENTITY_TOO_LARGE", 413))
_HTTP_415 = getattr(status, "HTTP_415_UNSUPPORTED_MEDIA_TYPE", 415)
_HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT",
                    getattr(status, "HTTP_422_UNPROCESSABLE_ENTITY", 422))


def new_request_id() -> str:
    """Generate a pseudonymous request ID (UUID4)."""
    return str(uuid.uuid4())


async def read_upload_bytes(
    upload: UploadFile,
    max_size_mb: int = 25,
    allowed_types: Optional[frozenset[str]] = None,
    allowed_extensions: Optional[frozenset[str]] = None,
) -> bytes:
    """Read an uploaded file into memory with size and type validation.

    Parameters
    ----------
    upload            : FastAPI UploadFile
    max_size_mb       : Hard size limit
    allowed_types     : Allowed MIME types (from Content-Type header)
    allowed_extensions: Allowed file extensions (fallback if MIME absent)

    Returns
    -------
    Raw bytes — caller is responsible for in-memory processing and NOT
    writing to disk.

    Raises
    ------
    HTTPException 413 if file too large.
    HTTPException 415 if type not allowed.
    """
    # Type check
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    filename = upload.filename or ""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if allowed_types and content_type and content_type not in allowed_types:
        # Try extension fallback
        if allowed_extensions and ext not in allowed_extensions:
            raise HTTPException(
                status_code=_HTTP_415,
                detail=f"File type '{content_type}' is not supported for this endpoint.",
            )

    # Read with size guard
    max_bytes = max_size_mb * 1024 * 1024
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=_HTTP_413,
            detail=f"Upload exceeds maximum allowed size of {max_size_mb} MB.",
        )
    return data


def validate_text_input(text: str, max_length: int = 10000) -> str:
    """Validate and sanitise text input."""
    stripped = text.strip()
    if not stripped:
        raise HTTPException(
            status_code=_HTTP_422,
            detail="Text input must not be empty.",
        )
    if len(stripped) > max_length:
        raise HTTPException(
            status_code=_HTTP_422,
            detail=f"Text input exceeds maximum length of {max_length} characters.",
        )
    return stripped
