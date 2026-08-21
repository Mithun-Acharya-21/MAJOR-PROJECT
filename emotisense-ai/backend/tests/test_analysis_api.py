"""Phase 18 — Integration tests for all analysis API routes."""

from __future__ import annotations

import io
import json
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes(duration_s: float = 2.0, sr: int = 16000, amplitude: float = 0.3) -> bytes:
    """Generate a minimal valid WAV file in memory."""
    import math, struct
    n_samples = int(sr * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        samples = [int(amplitude * 32767 * math.sin(2 * math.pi * 440 * i / sr))
                   for i in range(n_samples)]
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return buf.getvalue()


def _make_png_bytes() -> bytes:
    """Generate a minimal 1×1 white PNG."""
    import zlib
    def _pack32be(v): return struct.pack(">I", v)
    signature = b"\x89PNG\r\n\x1a\n"
    def _chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return _pack32be(len(data)) + tag + data + _pack32be(crc)
    ihdr = _chunk(b"IHDR", _pack32be(1) + _pack32be(1) + b"\x08\x02\x00\x00\x00")
    raw = b"\x00\xFF\xFF\xFF"  # filter=0, RGB white pixel
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Health + model info (pre-existing)
# ---------------------------------------------------------------------------

def test_health_returns_200():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_model_info_returns_disclaimer():
    r = client.get("/api/v1/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "disclaimer" in data
    assert "not a medical diagnosis" in data["disclaimer"].lower()


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------

def test_consent_accepted():
    r = client.post("/api/v1/analyze/consent", json={
        "consent_type": "audio_recording",
        "user_acknowledged": True,
    })
    assert r.status_code == 200
    assert r.json()["accepted"] is True


def test_consent_rejected_when_not_acknowledged():
    r = client.post("/api/v1/analyze/consent", json={
        "consent_type": "audio_recording",
        "user_acknowledged": False,
    })
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------

def test_text_analysis_returns_request_id():
    r = client.post("/api/v1/analyze/text", json={
        "text": "I have been feeling very low and tired for several weeks.",
        "consent_given": True,
    })
    assert r.status_code == 200
    data = r.json()
    assert "request_id" in data
    assert len(data["request_id"]) == 36  # UUID4

def test_text_analysis_has_disclaimer():
    r = client.post("/api/v1/analyze/text", json={
        "text": "I feel okay today.",
        "consent_given": True,
    })
    assert r.status_code == 200
    assert "disclaimer" in r.json()


def test_text_analysis_requires_consent():
    r = client.post("/api/v1/analyze/text", json={
        "text": "some text",
        "consent_given": False,
    })
    assert r.status_code == 422


def test_text_analysis_empty_text_rejected():
    r = client.post("/api/v1/analyze/text", json={
        "text": "",
        "consent_given": True,
    })
    assert r.status_code == 422


def test_text_analysis_model_not_yet_available():
    r = client.post("/api/v1/analyze/text", json={
        "text": "I feel tired and unmotivated.",
        "consent_given": True,
    })
    assert r.status_code == 200
    assert r.json()["model_available"] is False


def test_text_analysis_forbidden_diagnostic_phrase_not_in_response():
    """The system must never output forbidden diagnostic phrases."""
    r = client.post("/api/v1/analyze/text", json={
        "text": "I feel sad.",
        "consent_given": True,
    })
    body = r.text.lower()
    forbidden = ["you have depression", "you are clinically depressed",
                 "you need medication", "you definitely have anxiety"]
    for phrase in forbidden:
        assert phrase not in body, f"Forbidden phrase found: {phrase}"


# ---------------------------------------------------------------------------
# Voice analysis
# ---------------------------------------------------------------------------

def test_voice_analysis_valid_wav():
    wav_bytes = _make_wav_bytes(duration_s=3.0)
    r = client.post(
        "/api/v1/analyze/voice",
        files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data={"consent_given": "true"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "request_id" in data
    assert "disclaimer" in data
    assert data["model_available"] is False


def test_voice_analysis_requires_consent():
    wav_bytes = _make_wav_bytes()
    r = client.post(
        "/api/v1/analyze/voice",
        files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data={"consent_given": "false"},
    )
    assert r.status_code == 422


def test_voice_analysis_audio_quality_present():
    wav_bytes = _make_wav_bytes(duration_s=3.0)
    r = client.post(
        "/api/v1/analyze/voice",
        files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data={"consent_given": "true"},
    )
    assert r.status_code == 200
    data = r.json()
    if data["audio_quality"] is not None:
        assert "quality_score" in data["audio_quality"]


def test_voice_oversized_upload_rejected():
    # Create a fake "file" that reports size > limit
    # We create just over 25 MB of zeros
    oversized = b"\x00" * (26 * 1024 * 1024)
    r = client.post(
        "/api/v1/analyze/voice",
        files={"audio": ("big.wav", io.BytesIO(oversized), "audio/wav")},
        data={"consent_given": "true"},
    )
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------

def test_image_analysis_valid_png():
    png_bytes = _make_png_bytes()
    r = client.post(
        "/api/v1/analyze/image",
        files={"image": ("face.png", io.BytesIO(png_bytes), "image/png")},
        data={"consent_given": "true"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "request_id" in data
    assert "face_detected" in data
    assert "disclaimer" in data
    assert "not a depression indicator" in data["note"].lower()


def test_image_analysis_requires_consent():
    r = client.post(
        "/api/v1/analyze/image",
        files={"image": ("face.png", io.BytesIO(_make_png_bytes()), "image/png")},
        data={"consent_given": "false"},
    )
    assert r.status_code == 422


def test_image_analysis_model_not_available():
    png_bytes = _make_png_bytes()
    r = client.post(
        "/api/v1/analyze/image",
        files={"image": ("face.png", io.BytesIO(png_bytes), "image/png")},
        data={"consent_given": "true"},
    )
    assert r.status_code == 200
    assert r.json()["model_available"] is False


# ---------------------------------------------------------------------------
# Multimodal analysis
# ---------------------------------------------------------------------------

def test_multimodal_text_only():
    r = client.post(
        "/api/v1/analyze/multimodal",
        data={"consent_given": "true", "text": "I feel tired."},
    )
    assert r.status_code == 200
    data = r.json()
    assert "text" in data["used_modalities"]
    assert "disclaimer" in data


def test_multimodal_requires_consent():
    r = client.post(
        "/api/v1/analyze/multimodal",
        data={"consent_given": "false", "text": "hello"},
    )
    assert r.status_code == 422


def test_multimodal_reports_missing_modalities():
    r = client.post(
        "/api/v1/analyze/multimodal",
        data={"consent_given": "true", "text": "I feel anxious."},
    )
    data = r.json()
    # Audio and video not provided
    assert "audio" in data["missing_modalities"] or len(data["missing_modalities"]) > 0


def test_multimodal_model_not_available():
    r = client.post(
        "/api/v1/analyze/multimodal",
        data={"consent_given": "true", "text": "test"},
    )
    assert r.json()["model_available"] is False


# ---------------------------------------------------------------------------
# Privacy controls
# ---------------------------------------------------------------------------

def test_request_ids_are_unique():
    ids = set()
    for _ in range(5):
        r = client.post("/api/v1/analyze/text", json={
            "text": "test message", "consent_given": True
        })
        ids.add(r.json()["request_id"])
    assert len(ids) == 5  # All unique


def test_rate_limiter_module_importable():
    from app.core.rate_limit import check_rate_limit
    assert callable(check_rate_limit)


def test_privacy_module_importable():
    from app.core.privacy import new_request_id, validate_text_input
    rid = new_request_id()
    assert len(rid) == 36  # UUID4 format
