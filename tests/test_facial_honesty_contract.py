"""
Tests verifying the facial expression honesty contract and non-diagnostic semantics (NeuroScreen).

Verifications required:
1. Facial-expression scores are never returned as depression predictions.
2. Missing facial models / undetected faces do not generate fabricated scores or confidences.
3. Confidence values have a verified source (not arbitrary hardcoded numbers).
4. Unavailable models are reported honestly via model registry.
5. Experimental fusion is explicitly labeled as heuristic, not a trained model.
6. Live face analysis is labeled as auxiliary expression tracking, not depression diagnosis.
"""

import io
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from backend.app.services.photo_service import analyze_photo
from backend.app.services.face_service import analyze_live_face
from backend.app.services.fusion_service import fuse_multimodal
from backend.app.model_registry import audit_checkpoints


def _create_dummy_image(color=(128, 128, 128), size=(224, 224)) -> bytes:
    """Create a synthetic solid color image."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_facial_expression_never_returned_as_depression_prediction():
    """Verify photo analysis never claims to predict depression."""
    image_bytes = _create_dummy_image()
    result = analyze_photo(image_bytes)

    # Must explicitly state it is NOT a depression prediction or clinical model
    assert result.get("is_depression_prediction") is False
    assert result.get("is_depression_model") is False

    # The prediction label must never claim 'Depressed'
    prediction = result.get("prediction", "")
    assert prediction != "Depressed"

    # Signal type must indicate auxiliary or none, never trained depression model
    assert result.get("signal_type") in ("auxiliary_expression_score", "none", "proxy_heuristic")

    # Explanation note must declare it is not a validated depression probability
    note = result.get("explanation", {}).get("note", "") or result.get("note", "")
    assert "NOT a validated depression probability" in note or "No face detected" in note


def test_missing_face_does_not_fabricate_confidence_or_scores():
    """Verify that when no face is present, confidence is 0.0 and no scores are fabricated."""
    image_bytes = _create_dummy_image(color=(0, 0, 0))  # Blank black image has no face
    result = analyze_photo(image_bytes)

    if not result.get("face_detected"):
        assert result.get("confidence") == 0.0
        assert result.get("dominant_expression") is None
        assert result.get("confidence_source") == "unverified_no_face"
        assert "warning" in result
        assert "No face detected" in result["warning"]


def test_empty_image_handling():
    """Verify empty or corrupted images fail gracefully without fabricating scores."""
    result = analyze_photo(b"")
    assert result.get("error") is not None or result.get("face_detected") is False
    assert result.get("signal_type") in ("none", "invalid")


def test_confidence_values_have_verified_provenance():
    """Verify confidence values declare their exact source and are not arbitrary magic numbers."""
    image_bytes = _create_dummy_image()
    result = analyze_photo(image_bytes)

    assert "confidence_source" in result
    source = result["confidence_source"]
    assert source in ("dominant_expression_prominence", "unverified_no_face")

    # Hardcoded magic numbers (like legacy 65.0 or 25.0 without face) must not occur without a source
    if not result.get("face_detected"):
        assert result.get("confidence") == 0.0


def test_unavailable_models_reported_honestly_in_registry():
    """Verify model registry reports vision and fusion branches as unavailable (no trained weights)."""
    checkpoints = audit_checkpoints()

    assert "text" in checkpoints
    assert "vision" in checkpoints
    assert "fusion" in checkpoints

    # Vision branch must be reported unavailable (no trained facial depression weights exist)
    assert checkpoints["vision"]["available"] is False
    assert "no trained weights" in checkpoints["vision"]["reason"].lower()

    # Fusion branch must be reported unavailable (heuristic fusion, no trained neural weights)
    assert checkpoints["fusion"]["available"] is False
    assert "no trained weights" in checkpoints["fusion"]["reason"].lower()

    # Only the genuine EATD text model is available
    assert checkpoints["text"]["available"] is True


def test_experimental_fusion_not_presented_as_trained_model():
    """Verify fusion service explicitly declares itself as experimental heuristic."""
    mock_results = {
        "text": {"probability": 60.0, "confidence": 70.0, "signal_type": "trained_model_probability"},
        "photo": {"probability": 30.0, "confidence": 50.0, "signal_type": "auxiliary_expression_score"},
    }
    fused = fuse_multimodal(mock_results)

    assert fused.get("fusion_type") == "experimental_heuristic_fusion"
    assert fused.get("is_trained_model") is False
    assert fused.get("is_clinical_diagnosis") is False

    note = fused.get("explanation", {}).get("interpretation_note", "")
    assert "Experimental heuristic fusion" in note
    assert "NOT clinically validated" in note


def test_live_face_never_claims_depression_diagnosis():
    """Verify multi-frame live face analysis declares itself as auxiliary expression tracking."""
    frames = [_create_dummy_image() for _ in range(3)]
    result = analyze_live_face(frames, model_state=None)

    assert result.get("is_depression_prediction") is False
    assert result.get("is_depression_model") is False
    assert result.get("signal_type") == "auxiliary_expression_score"
    assert "MediaPipe" in result.get("model_name", "") or "FER" in result.get("model_name", "")
