"""Tests for Phase 8 — transfer learning and Grad-CAM."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision", reason="torchvision not installed")

from training.vision.vision_transfer import (
    TransferFaceEncoder,
    VisionTransferConfig,
    compute_gradcam,
    heatmap_to_overlay,
    DISCLAIMER,
)
from training.vision.fer_baseline import FER_EMOTIONS


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backbone", ["mobilenet_v2", "resnet18"])
def test_forward_produces_correct_shapes(backbone: str) -> None:
    config = VisionTransferConfig(backbone=backbone, n_classes=7, embedding_dim=64)
    model = TransferFaceEncoder(config)
    x = torch.zeros((2, 3, 224, 224))
    outputs = model(x)
    assert outputs["logits"].shape == (2, 7)
    assert outputs["probs"].shape == (2, 7)
    assert outputs["embedding"].shape == (2, 64)
    assert "feature_map" in outputs


def test_probs_sum_to_one() -> None:
    config = VisionTransferConfig(backbone="mobilenet_v2", n_classes=7, embedding_dim=64)
    model = TransferFaceEncoder(config)
    x = torch.rand((3, 3, 224, 224))
    probs = model(x)["probs"]
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)


def test_unknown_backbone_raises() -> None:
    config = VisionTransferConfig(backbone="vgg16", embedding_dim=64)
    with pytest.raises(ValueError, match="Unsupported backbone"):
        TransferFaceEncoder(config)


def test_freeze_unfreeze_backbone() -> None:
    config = VisionTransferConfig(backbone="mobilenet_v2", embedding_dim=64)
    model = TransferFaceEncoder(config)
    model.freeze_backbone()
    frozen = [p.requires_grad for p in model.backbone_features.parameters()]
    assert not any(frozen), "All backbone params should be frozen"
    model.unfreeze_backbone()
    unfrozen = [p.requires_grad for p in model.backbone_features.parameters()]
    assert all(unfrozen), "All backbone params should be trainable after unfreeze"


# ---------------------------------------------------------------------------
# Grad-CAM tests
# ---------------------------------------------------------------------------

def test_gradcam_returns_valid_heatmap() -> None:
    config = VisionTransferConfig(backbone="mobilenet_v2", n_classes=7, embedding_dim=64)
    model = TransferFaceEncoder(config)
    x = torch.rand((1, 3, 224, 224), requires_grad=False)
    result = compute_gradcam(model, x, image_size=(224, 224))
    heatmap = result["heatmap"]
    assert heatmap.shape == (224, 224)
    assert heatmap.dtype == np.float32
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert "disclaimer" in result
    assert result["emotion_label"] in FER_EMOTIONS


def test_gradcam_disclaimer_contains_required_text() -> None:
    assert "emotion" in DISCLAIMER.lower()
    assert "not clinical evidence" in DISCLAIMER.lower()
    assert "mental-health" in DISCLAIMER.lower()


def test_gradcam_target_class_respected() -> None:
    config = VisionTransferConfig(backbone="resnet18", n_classes=7, embedding_dim=64)
    model = TransferFaceEncoder(config)
    x = torch.rand((1, 3, 224, 224))
    for cls in [0, 3, 6]:
        result = compute_gradcam(model, x, target_class=cls, image_size=(112, 112))
        assert result["target_class"] == cls
        assert result["heatmap"].shape == (112, 112)


# ---------------------------------------------------------------------------
# Overlay test (requires OpenCV)
# ---------------------------------------------------------------------------

def test_heatmap_overlay_shape() -> None:
    cv2 = pytest.importorskip("cv2", reason="opencv not installed")
    heatmap = np.random.rand(224, 224).astype(np.float32)
    original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    overlay = heatmap_to_overlay(heatmap, original, alpha=0.4)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


# ---------------------------------------------------------------------------
# Safety label test
# ---------------------------------------------------------------------------

def test_emotion_labels_not_depression() -> None:
    for label in FER_EMOTIONS:
        assert "depress" not in label.lower()
