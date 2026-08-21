"""Speech-to-Text for live analysis — faster-whisper singleton.

The model loads ONCE on first call and is reused for every audio chunk.
Uses the ``tiny`` model by default (fast, ~75 MB download on first use).
Override with the ``NEUROSCREEN_WHISPER_MODEL`` env var.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger("neuroscreen.live_stt")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_WHISPER_MODEL_NAME = os.environ.get("NEUROSCREEN_WHISPER_MODEL", "tiny")
_WHISPER_CACHE_DIR = Path(
    os.environ.get("NEUROSCREEN_WHISPER_DIR", str(Path(__file__).resolve().parent.parent.parent / ".whisper_models"))
)

# ---------------------------------------------------------------------------
# Lazy singleton — loaded once, reused forever (thread-safe)
# ---------------------------------------------------------------------------
_model = None
_lock = threading.Lock()


def _get_model():
    """Load the faster-whisper model exactly once."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            logger.info("Loading faster-whisper model '%s' (first use)...", _WHISPER_MODEL_NAME)
            from faster_whisper import WhisperModel

            _WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _model = WhisperModel(
                _WHISPER_MODEL_NAME,
                device="cpu",
                compute_type="int8",
                download_root=str(_WHISPER_CACHE_DIR),
            )
            logger.info("faster-whisper model ready.")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_audio(audio_bytes: bytes, return_segments: bool = False):
    """Transcribe an audio file (WebM, WAV, MP3, OGG, etc.) to text.

    The audio is written to a temporary file (required by faster-whisper's
    PyAV backend) and deleted immediately after transcription.

    Args:
        audio_bytes: Raw audio file content.
        return_segments: If True, return list of segment dicts instead of string.

    Returns:
        If return_segments=False: Transcribed text string.
        If return_segments=True: List of dicts with keys:
            {"text": str, "start": float, "end": float}

    Raises:
        Exception: On transcription failure — caller decides how to handle.
    """
    model = _get_model()

    # Write to a temp file — faster-whisper decodes via PyAV which needs a file
    fd, tmp_path = tempfile.mkstemp(suffix=".webm", prefix="neuroscreen_audio_")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(audio_bytes)
        segments, _info = model.transcribe(
            tmp_path,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        if return_segments:
            result = []
            for seg in segments:
                text = seg.text.strip()
                if text:
                    result.append({
                        "text": text,
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                    })
            return result
        else:
            parts = [seg.text.strip() for seg in segments]
            return " ".join(p for p in parts if p).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
