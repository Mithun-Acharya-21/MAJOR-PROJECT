"""Tests for Phase 13 — fusion baselines and missing-modality handling."""

from __future__ import annotations

import pytest
import torch

from training.multimodal.fusion_baselines import (
    DISCLAIMER,
    FUSION_VARIANTS,
    MODALITY_ORDER,
    ConcatFusion,
    FusionConfig,
    MeanPoolFusion,
    WeightedSumFusion,
    apply_modality_dropout,
    build_fusion_model,
    run_fusion_ablation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(fusion_type: str = "concat", dim: int = 32) -> FusionConfig:
    return FusionConfig(
        text_dim=dim, audio_dim=dim, video_dim=dim,
        projected_dim=dim, n_classes=2, dropout=0.0,
        fusion_type=fusion_type,
    )


def _embeddings(B: int = 4, dim: int = 32) -> dict[str, torch.Tensor]:
    return {m: torch.randn(B, dim) for m in MODALITY_ORDER}


def _all_present(B: int = 4) -> dict[str, torch.Tensor]:
    return {m: torch.ones(B) for m in MODALITY_ORDER}


def _partial_mask(B: int = 4, present: list[str] = None) -> dict[str, torch.Tensor]:
    present = present or ["text"]
    return {m: torch.ones(B) if m in present else torch.zeros(B) for m in MODALITY_ORDER}


# ---------------------------------------------------------------------------
# Shared output-shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fusion_type", ["concat", "mean_pool", "weighted_sum"])
class TestFusionOutputShapes:

    def test_logits_shape(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        out = model(_embeddings(), _all_present())
        assert out["logits"].shape == (4, 2)

    def test_probs_sum_to_one(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        out = model(_embeddings(), _all_present())
        assert torch.allclose(out["probs"].sum(dim=1), torch.ones(4), atol=1e-5)

    def test_risk_score_shape(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        out = model(_embeddings(), _all_present())
        assert out["risk_score"].shape == (4,)

    def test_risk_score_in_unit_interval(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        out = model(_embeddings(), _all_present())
        assert (out["risk_score"] >= 0.0).all()
        assert (out["risk_score"] <= 1.0).all()

    def test_fusion_weights_shape(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        out = model(_embeddings(), _all_present())
        assert out["fusion_weights"].shape == (4, 3)

    def test_disclaimer_present(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        out = model(_embeddings(), _all_present())
        assert "disclaimer" in out
        assert "not clinical" in out["disclaimer"].lower()

    def test_no_nan_in_outputs(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        out = model(_embeddings(), _all_present())
        for key in ("logits", "probs", "risk_score", "fusion_weights"):
            assert not torch.isnan(out[key]).any(), f"NaN in {key} for {fusion_type}"


# ---------------------------------------------------------------------------
# Missing-modality handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fusion_type", ["concat", "mean_pool", "weighted_sum"])
class TestMissingModalityHandling:

    def test_text_only_does_not_crash(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        out = model(_embeddings(), _partial_mask(present=["text"]))
        assert out["logits"].shape == (4, 2)

    def test_audio_only_does_not_crash(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        out = model(_embeddings(), _partial_mask(present=["audio"]))
        assert out["logits"].shape == (4, 2)

    def test_zero_mask_modality_has_zero_weight(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        masks = _partial_mask(present=["text", "audio"])  # video = 0
        out = model(_embeddings(), masks)
        video_idx = MODALITY_ORDER.index("video")
        # Video weight must be near zero for all samples
        assert out["fusion_weights"][:, video_idx].abs().max() < 1e-5

    def test_absent_embedding_key_handled(self, fusion_type: str) -> None:
        """Model must cope when a modality key is absent from embeddings dict."""
        model = build_fusion_model(_config(fusion_type))
        emb = {m: torch.randn(4, 32) for m in ["text", "audio"]}  # no video
        masks = _partial_mask(present=["text", "audio"])
        out = model(emb, masks)
        assert out["logits"].shape == (4, 2)

    def test_used_modalities_reflects_mask(self, fusion_type: str) -> None:
        model = build_fusion_model(_config(fusion_type))
        masks = _partial_mask(present=["text"])
        out = model(_embeddings(), masks)
        assert "text" in out["used_modalities"]
        assert "audio" not in out["used_modalities"]
        assert "video" not in out["used_modalities"]

    def test_all_zero_mask_produces_finite_output(self, fusion_type: str) -> None:
        """All masks zero — output should still be finite, not NaN."""
        model = build_fusion_model(_config(fusion_type))
        all_zero = {m: torch.zeros(4) for m in MODALITY_ORDER}
        out = model(_embeddings(), all_zero)
        assert not torch.isnan(out["logits"]).any()


# ---------------------------------------------------------------------------
# Modality dropout
# ---------------------------------------------------------------------------

class TestModalityDropout:

    def test_dropout_preserves_at_least_one(self) -> None:
        masks = _all_present(B=16)
        dropped = apply_modality_dropout(masks, drop_prob=0.99, training=True)
        for b in range(16):
            present = sum(dropped[m][b].item() for m in MODALITY_ORDER)
            assert present >= 1, f"Sample {b} has no modalities after dropout"

    def test_no_dropout_at_inference(self) -> None:
        masks = _all_present(B=8)
        result = apply_modality_dropout(masks, drop_prob=0.99, training=False)
        for m in MODALITY_ORDER:
            assert torch.equal(result[m], masks[m])

    def test_zero_prob_no_change(self) -> None:
        masks = _all_present(B=8)
        result = apply_modality_dropout(masks, drop_prob=0.0, training=True)
        for m in MODALITY_ORDER:
            assert torch.equal(result[m], masks[m])

    def test_single_modality_not_dropped(self) -> None:
        """A sample with only one modality must never be dropped to zero."""
        masks = _partial_mask(B=8, present=["text"])
        dropped = apply_modality_dropout(masks, drop_prob=0.99, training=True)
        for b in range(8):
            assert dropped["text"][b].item() > 0


# ---------------------------------------------------------------------------
# Factory and ablation
# ---------------------------------------------------------------------------

class TestFactory:

    def test_unknown_fusion_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown fusion_type"):
            build_fusion_model(FusionConfig(fusion_type="unknown"))

    def test_all_variants_registered(self) -> None:
        for name in ["concat", "mean_pool", "weighted_sum"]:
            assert name in FUSION_VARIANTS


class TestFusionAblation:

    def test_ablation_returns_all_subset_keys(self) -> None:
        model = build_fusion_model(_config("mean_pool"))
        emb = _embeddings()
        masks = _all_present()
        result = run_fusion_ablation(model, emb, masks)
        assert "all" in result
        for m in MODALITY_ORDER:
            assert f"only_{m}" in result

    def test_single_modality_subsets_have_correct_used(self) -> None:
        model = build_fusion_model(_config("concat"))
        result = run_fusion_ablation(model, _embeddings(), _all_present())
        for m in MODALITY_ORDER:
            subset = result[f"only_{m}"]
            assert subset["used_modalities"] == [m]

    def test_risk_scores_in_unit_interval(self) -> None:
        model = build_fusion_model(_config("weighted_sum"))
        result = run_fusion_ablation(model, _embeddings(), _all_present())
        for key, vals in result.items():
            assert 0.0 <= vals["risk_score_mean"] <= 1.0, f"Bad risk in subset {key}"
