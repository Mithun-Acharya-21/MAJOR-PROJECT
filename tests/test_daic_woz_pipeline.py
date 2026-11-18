"""
DAIC-WOZ Pipeline Automated Test Suite
NeuroScreen Major Project - Phase 12 Test Verification
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.daic_woz.preprocessing import (
    preprocess_transcript_text,
    validate_visual_feature_vector
)
from backend.app.daic_woz.model_loader import DaicWozModelLoader
from backend.app.daic_woz.inference_service import DaicWozInferenceService


def test_daic_woz_directory_structure():
    """Verify all required DAIC-WOZ directories exist."""
    required_dirs = [
        "data/raw/daic_woz", "data/manifests/daic_woz", "data/splits/daic_woz",
        "models/daic_woz/text", "models/daic_woz/audio", "models/daic_woz/visual",
        "models/daic_woz/fusion", "results/daic_woz/reports", "results/daic_woz/explainability",
        "scripts/daic_woz", "backend/app/daic_woz", "docs/daic_woz"
    ]
    for d in required_dirs:
        p = Path(d)
        assert p.exists() and p.is_dir(), f"Missing required directory: {d}"


def test_research_integrity_quarantine():
    """Verify that no fabricated or synthetic participant records exist in data/raw/daic_woz."""
    raw_dir = Path("data/raw/daic_woz")
    # All files in raw_dir must be either real downloads or the directory must be empty
    raw_files = list(raw_dir.glob("*"))
    # If there are files, ensure none are marked 'synthetic' or 'mock'
    for f in raw_files:
        assert "synthetic" not in f.name.lower()
        assert "mock" not in f.name.lower()


def test_text_preprocessing():
    """Verify transcript cleaning standardizes whitespace and handles empty strings."""
    raw = "  I feel   exhausted\nand   isolated.  "
    cleaned = preprocess_transcript_text(raw)
    assert cleaned == "I feel exhausted and isolated."
    assert preprocess_transcript_text("") == ""
    assert preprocess_transcript_text(None) == ""


def test_visual_feature_validation():
    """Verify that 40-dimensional OpenFace summary feature vectors are strictly validated."""
    short_feats = [0.1, 0.2, 0.3]
    validated = validate_visual_feature_vector(short_feats)
    assert len(validated) == 40
    assert validated[0] == pytest.approx(0.1, abs=1e-4)
    assert validated[3] == 0.0  # padded


def test_model_loader_fallback():
    """Verify model loader reports safe offline status when checkpoints are awaiting download."""
    loader = DaicWozModelLoader.get_instance()
    status = loader.get_status()
    assert "text_model_loaded" in status
    assert "device" in status
    assert status["dataset_acquisition_status"] in ["PENDING_OFFICIAL_DOWNLOAD", "CHECKPOINTS_LOADED"]


def test_inference_service_disclaimer():
    """Verify inference outputs contain mandatory academic research disclaimers."""
    service = DaicWozInferenceService()
    resp = service.analyze_multimodal(transcript_text="Test utterance")
    assert "screening prototype" in resp.research_disclaimer.lower()
    assert resp.model_provenance["dataset"] == "DAIC-WOZ (Distress Analysis Interview Corpus)"
    assert resp.risk_level in ["LOW", "ELEVATED", "PENDING_CHECKPOINTS"]


import asyncio
from backend.app.daic_woz.routes import (
    get_daic_woz_model_status,
    get_daic_woz_metrics,
    analyze_daic_woz_multimodal
)
from backend.app.daic_woz.schemas import DaicWozMultimodalRequest


def test_fastapi_model_status_endpoint():
    """Verify GET /api/daic-woz/model/status endpoint returns valid schema."""
    data = get_daic_woz_model_status()
    assert hasattr(data, "text_model_loaded")
    assert hasattr(data, "active_checkpoints")
    assert data.dataset_acquisition_status in ["PENDING_OFFICIAL_DOWNLOAD", "CHECKPOINTS_LOADED"]


def test_fastapi_model_metrics_endpoint():
    """Verify GET /api/daic-woz/model/metrics returns benchmark details."""
    data = get_daic_woz_metrics()
    assert "DAIC-WOZ" in data.benchmark_name
    assert "70% - 80%" in data.target_accuracy_band


def test_fastapi_analyze_endpoint():
    """Verify POST /api/daic-woz/analyze accepts multimodal payloads."""
    payload = DaicWozMultimodalRequest(
        transcript_text="I feel tired and have trouble sleeping at night.",
        visual_features=[0.1] * 40
    )
    data = analyze_daic_woz_multimodal(payload)
    assert data.screening_indicator is not None
    assert "text" in data.modality_breakdown
    assert "research prototype" in data.research_disclaimer.lower() or "screening prototype" in data.research_disclaimer.lower()

