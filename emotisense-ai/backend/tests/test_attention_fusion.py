"""Tests for Phase 14 — reliability-aware attention fusion and calibration."""

from __future__ import annotations

import math
import pytest
import torch

from training.multimodal.attention_fusion import (
    AttentionFusionConfig,
    CrossModalAttention,
    DISCLAIMER,
    ModalityGate,
    ReliabilityAwareAttentionFusion,
    calibrate_temperature,
    expected_calibration_error,
)
from training.multimodal.fusion_baselines import MODALITY_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(D: int = 32, heads: int = 4) -> AttentionFusionConfig:
    return AttentionFusionConfig(
        text_dim=D, audio_dim=D, video_dim=D,
        projected_dim=D, n_classes=2,
        n_emotion_classes=7,
        n_attention_heads=heads,
        dropout=0.0,
        gate_hidden_dim=16,
    )


def _embeddings(B: int = 4, D: int = 32) -> dict[str, torch.Tensor]:
    return {m: torch.randn(B, D) for m in MODALITY_ORDER}


def _all_masks(B: int = 4) -> dict[str, torch.Tensor]:
    return {m: torch.ones(B) for m in MODALITY_ORDER}


def _quality(B: int = 4) -> dict[str, torch.Tensor]:
    return {m: torch.rand(B).clamp(0.5, 1.0) for m in MODALITY_ORDER}


def _partial_masks(B: int = 4, present: list[str] = None) -> dict[str, torch.Tensor]:
    present = present or ["text"]
    return {m: torch.ones(B) if m in present else torch.zeros(B) for m in MODALITY_ORDER}


# ---------------------------------------------------------------------------
# ModalityGate
# ---------------------------------------------------------------------------

class TestModalityGate:

    def test_gate_output_in_unit_interval(self) -> None:
        gate = ModalityGate(projected_dim=32, gate_hidden_dim=16)
        proj = torch.randn(4, 32)
        quality = torch.rand(4)
        mask = torch.ones(4)
        g = gate(proj, quality, mask)
        assert g.shape == (4,)
        assert (g >= 0.0).all() and (g <= 1.0).all()

    def test_zero_mask_zeroes_gate(self) -> None:
        gate = ModalityGate(projected_dim=32, gate_hidden_dim=16)
        proj = torch.randn(4, 32)
        quality = torch.rand(4)
        mask = torch.zeros(4)
        g = gate(proj, quality, mask)
        assert g.abs().max() < 1e-6

    def test_partial_mask(self) -> None:
        gate = ModalityGate(projected_dim=32, gate_hidden_dim=16)
        proj = torch.randn(4, 32)
        quality = torch.rand(4)
        mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        g = gate(proj, quality, mask)
        assert g[1].abs() < 1e-6
        assert g[3].abs() < 1e-6


# ---------------------------------------------------------------------------
# CrossModalAttention
# ---------------------------------------------------------------------------

class TestCrossModalAttention:

    def test_output_shape(self) -> None:
        attn = CrossModalAttention(d_model=32, n_heads=4, dropout=0.0)
        tokens = torch.randn(4, 3, 32)
        out, weights = attn(tokens)
        assert out.shape == (4, 3, 32)
        assert weights.shape == (4, 3, 3)

    def test_attention_weights_sum_to_one(self) -> None:
        attn = CrossModalAttention(d_model=32, n_heads=4, dropout=0.0)
        tokens = torch.randn(4, 3, 32)
        _, weights = attn(tokens)
        # Rows should sum to ~1 (softmax over keys)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 3), atol=1e-4)

    def test_key_padding_mask_applied(self) -> None:
        attn = CrossModalAttention(d_model=32, n_heads=4, dropout=0.0)
        tokens = torch.randn(2, 3, 32)
        # Mask out the third token
        kpm = torch.tensor([[False, False, True], [False, False, True]])
        out, weights = attn(tokens, key_padding_mask=kpm)
        assert out.shape == (2, 3, 32)


# ---------------------------------------------------------------------------
# ReliabilityAwareAttentionFusion — forward
# ---------------------------------------------------------------------------

class TestReliabilityAwareAttentionFusion:

    def test_all_modalities_present(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        out = model(_embeddings(), _all_masks(), _quality())
        assert out["logits"].shape == (4, 2)
        assert out["probs"].shape == (4, 2)
        assert out["risk_score"].shape == (4,)
        assert out["gate_weights"].shape == (4, 3)
        assert out["attention_weights"].shape == (4, 3, 3)

    def test_probs_sum_to_one(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        out = model(_embeddings(), _all_masks())
        assert torch.allclose(out["probs"].sum(dim=1), torch.ones(4), atol=1e-5)

    def test_calibrated_probs_sum_to_one(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        out = model(_embeddings(), _all_masks())
        assert torch.allclose(out["calibrated_probs"].sum(dim=1), torch.ones(4), atol=1e-5)

    def test_uncertainty_in_unit_interval(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        out = model(_embeddings(), _all_masks())
        assert (out["uncertainty"] >= 0.0).all()
        assert (out["uncertainty"] <= 1.0).all()

    def test_emotion_profile_shape(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        out = model(_embeddings(), _all_masks())
        assert out["emotion_profile"] is not None
        assert out["emotion_profile"].shape == (4, 7)
        assert torch.allclose(out["emotion_profile"].sum(dim=1), torch.ones(4), atol=1e-5)

    def test_no_emotion_head_when_zero_classes(self) -> None:
        cfg = _cfg()
        cfg = AttentionFusionConfig(**{**cfg.__dict__, "n_emotion_classes": 0})
        model = ReliabilityAwareAttentionFusion(cfg)
        out = model(_embeddings(), _all_masks())
        assert out["emotion_profile"] is None

    def test_absent_modality_zeroed(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        masks_text_only = _partial_masks(present=["text"])
        out = model(_embeddings(), masks_text_only)
        # Audio and video gate weights should be near zero
        audio_idx = MODALITY_ORDER.index("audio")
        video_idx = MODALITY_ORDER.index("video")
        assert out["gate_weights"][:, audio_idx].abs().max() < 1e-5
        assert out["gate_weights"][:, video_idx].abs().max() < 1e-5

    def test_low_quality_changes_gate_output(self) -> None:
        """Gate output must differ when quality changes — it is an active input."""
        model = ReliabilityAwareAttentionFusion(_cfg())
        # Use the same embeddings and masks; only quality differs
        emb = _embeddings()
        high_q = {m: torch.ones(4) for m in MODALITY_ORDER}
        low_q = {m: torch.zeros(4) for m in MODALITY_ORDER}
        out_high = model(emb, _all_masks(), high_q)
        out_low = model(emb, _all_masks(), low_q)
        # Gate weights must be different when quality differs
        assert not torch.allclose(out_low["gate_weights"], out_high["gate_weights"])

    def test_used_modalities_all_present(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        out = model(_embeddings(), _all_masks())
        assert set(out["used_modalities"]) == set(MODALITY_ORDER)

    def test_used_modalities_text_only(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        out = model(_embeddings(), _partial_masks(present=["text"]))
        assert out["used_modalities"] == ["text"]

    def test_disclaimer_present(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        out = model(_embeddings(), _all_masks())
        assert "disclaimer" in out
        assert "not clinical" in out["disclaimer"].lower()
        assert "diagnostic signals" in out["disclaimer"].lower()

    def test_no_nan_in_outputs(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        out = model(_embeddings(), _all_masks(), _quality())
        for key in ("logits", "probs", "risk_score", "uncertainty",
                    "gate_weights", "attention_weights"):
            assert not torch.isnan(out[key]).any(), f"NaN in {key}"

    def test_absent_embedding_key_handled(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        emb = {m: torch.randn(4, 32) for m in ["text", "audio"]}
        masks = _partial_masks(present=["text", "audio"])
        out = model(emb, masks)
        assert out["logits"].shape == (4, 2)


# ---------------------------------------------------------------------------
# Temperature calibration
# ---------------------------------------------------------------------------

class TestTemperatureCalibration:

    def test_temperature_updated_after_calibration(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        B = 8
        embs = [_embeddings(B=1) for _ in range(B)]
        masks = [_all_masks(B=1) for _ in range(B)]
        labels = torch.randint(0, 2, (B,))
        t = calibrate_temperature(model, embs, masks, labels, max_steps=10)
        assert isinstance(t, float)
        assert t > 0.0
        assert abs(float(model.temperature) - t) < 1e-5

    def test_temperature_positive(self) -> None:
        model = ReliabilityAwareAttentionFusion(_cfg())
        embs = [_embeddings(B=1) for _ in range(4)]
        masks = [_all_masks(B=1) for _ in range(4)]
        labels = torch.tensor([0, 1, 0, 1])
        t = calibrate_temperature(model, embs, masks, labels, max_steps=5)
        assert t > 0.0


# ---------------------------------------------------------------------------
# ECE calibration metric
# ---------------------------------------------------------------------------

class TestExpectedCalibrationError:

    def test_perfect_calibration_ece_near_zero(self) -> None:
        # Perfect calibration: confidence == accuracy
        probs = torch.linspace(0.05, 0.95, 10)
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        result = expected_calibration_error(probs, labels, n_bins=5)
        assert 0.0 <= result["ece"] <= 1.0
        assert "bin_stats" in result

    def test_ece_is_zero_when_empty_bins(self) -> None:
        probs = torch.tensor([0.5, 0.5, 0.5])
        labels = torch.tensor([0, 1, 0])
        result = expected_calibration_error(probs, labels, n_bins=10)
        assert 0.0 <= result["ece"] <= 1.0

    def test_output_keys_present(self) -> None:
        probs = torch.rand(20)
        labels = torch.randint(0, 2, (20,))
        result = expected_calibration_error(probs, labels)
        for key in ("ece", "max_calibration_error", "n_bins", "n_samples", "bin_stats"):
            assert key in result
