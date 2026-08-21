"""Unit tests for face preprocessing (Phase 7).

OpenCV is an optional dependency during CI; tests skip gracefully if absent.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")

from training.vision.face_preprocess import FacePreprocessor, FaceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_rgb(h: int = 120, w: int = 120, value: int = 128) -> np.ndarray:
    """Create a uniform RGB image (no real face)."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _make_preprocessor() -> FacePreprocessor:
    return FacePreprocessor()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_face_result_metadata_serialisable() -> None:
    r = FaceResult(detected=True, n_faces=1, quality_score=0.75, reliability=True)
    meta = r.to_metadata()
    assert isinstance(meta, dict)
    assert meta["quality_score"] == 0.75
    assert meta["reliability"] is True


def test_blank_image_returns_no_face() -> None:
    p = _make_preprocessor()
    result = p.process_array(_blank_rgb(), is_bgr=False)
    assert not result.detected
    assert result.n_faces == 0
    assert "no_face_detected" in result.rejection_reasons
    assert result.quality_score == 0.0
    assert result.reliability is False


def test_blurry_image_flagged() -> None:
    """A flat uniform image (zero texture) has near-zero Laplacian variance."""
    p = _make_preprocessor()
    # Laplacian variance on uniform patch = 0 < blur_threshold=80 → blurry
    gray_flat = np.full((48, 48), 128, dtype=np.uint8)
    score = FacePreprocessor._blur_score(gray_flat)
    assert score < p.cfg["blur_threshold"]


def test_composite_quality_clamps_to_unit_interval() -> None:
    p = _make_preprocessor()
    r = FaceResult(blur_score=1000.0, brightness=128.0)
    q = p._composite_quality(r, multi_face=False)
    assert 0.0 <= q <= 1.0


def test_multi_face_reduces_quality() -> None:
    p = _make_preprocessor()
    r = FaceResult(blur_score=400.0, brightness=128.0)
    q_single = p._composite_quality(r, multi_face=False)
    q_multi = p._composite_quality(r, multi_face=True)
    assert q_multi < q_single


def test_process_bytes_bad_data_returns_decode_failed() -> None:
    p = _make_preprocessor()
    result = p.process_bytes(b"not_an_image")
    assert not result.detected
    assert "decode_failed" in result.rejection_reasons


def test_fer_result_has_disclaimer() -> None:
    from training.vision.fer_baseline import DISCLAIMER
    # The disclaimer must mention emotion and must not claim to predict depression
    assert "emotion" in DISCLAIMER.lower()
    # Acceptable to mention "depression label" to clarify what the output is NOT
    assert "not a depression" in DISCLAIMER.lower() or "not proof" in DISCLAIMER.lower()


def test_fer_model_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    from training.vision.fer_baseline import FERBaselineModel, FERBaselineConfig
    config = FERBaselineConfig(n_classes=7, image_size=48, embedding_dim=128)
    model = FERBaselineModel(config)
    x = torch.zeros((2, 1, 48, 48))
    outputs = model(x)
    assert outputs["logits"].shape == (2, 7)
    assert outputs["probs"].shape == (2, 7)
    assert outputs["embedding"].shape == (2, 128)
    assert "feature_maps" in outputs   # required for Grad-CAM (Phase 8)


def test_fer_emotion_labels_not_depression() -> None:
    from training.vision.fer_baseline import FER_EMOTIONS
    for label in FER_EMOTIONS:
        assert "depress" not in label.lower()
        assert "disorder" not in label.lower()
