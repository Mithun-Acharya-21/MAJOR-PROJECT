"""Tests for Phase 17 — LIME, SHAP, Grad-CAM, and fusion explanation views."""

from __future__ import annotations

import base64
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Text explainer tests
# ---------------------------------------------------------------------------

class TestTextExplainerHelpers:

    def test_attention_weights_normalised(self) -> None:
        from app.ml.explainability.text_explainer import format_attention_weights
        tokens = ["I", "feel", "tired", "today"]
        weights = np.array([0.1, 0.4, 0.3, 0.2])
        result = format_attention_weights(tokens, weights, top_k=3)
        assert len(result["all_tokens"]) == 4
        assert len(result["top_k_tokens"]) == 3
        assert "disclaimer" in result
        assert "supplementary" in result["disclaimer"].lower()

    def test_attention_top_k_respects_limit(self) -> None:
        from app.ml.explainability.text_explainer import format_attention_weights
        tokens = [f"tok_{i}" for i in range(10)]
        weights = np.random.rand(10)
        result = format_attention_weights(tokens, weights, top_k=4)
        assert len(result["top_k_tokens"]) == 4

    def test_attention_weights_sum_to_one(self) -> None:
        from app.ml.explainability.text_explainer import format_attention_weights
        tokens = ["a", "b", "c"]
        weights = np.array([2.0, 3.0, 5.0])
        result = format_attention_weights(tokens, weights)
        total = sum(t["weight"] for t in result["all_tokens"])
        assert abs(total - 1.0) < 1e-4

    def test_disclaimer_correct_for_attention(self) -> None:
        from app.ml.explainability.text_explainer import ATTENTION_DISCLAIMER
        assert "supplementary diagnostic signal" in ATTENTION_DISCLAIMER.lower()
        assert "not a causal" in ATTENTION_DISCLAIMER.lower()

    def test_lime_disclaimer_present(self) -> None:
        from app.ml.explainability.text_explainer import DISCLAIMER
        assert "model behaviour" in DISCLAIMER.lower()
        assert "not clinical causality" in DISCLAIMER.lower()

    def test_lime_text_explain(self) -> None:
        pytest.importorskip("lime", reason="lime not installed")
        from app.ml.explainability.text_explainer import explain_text_lime

        def fake_predict(texts):
            return np.array([[0.4, 0.6]] * len(texts))

        result = explain_text_lime(
            "I feel very tired and hopeless",
            fake_predict,
            class_names=["control", "depressed"],
            num_features=5,
            num_samples=50,
        )
        assert "attributions" in result
        assert "top_tokens" in result
        assert "disclaimer" in result
        assert result["method"] == "lime"

    def test_shap_text_explain(self) -> None:
        pytest.importorskip("shap", reason="shap not installed")
        from app.ml.explainability.text_explainer import explain_text_shap

        def fake_predict(texts):
            return np.array([[0.3, 0.7]] * len(texts))

        result = explain_text_shap(
            "feeling empty and tired",
            fake_predict,
            max_evals=50,
        )
        assert "attributions" in result
        assert "disclaimer" in result
        assert result["method"] == "shap"


# ---------------------------------------------------------------------------
# Acoustic explainer tests
# ---------------------------------------------------------------------------

class TestAcousticExplainer:

    def test_format_permutation_importance(self) -> None:
        from app.ml.explainability.acoustic_explainer import format_permutation_importance
        importance = {"mfcc_0": 0.05, "mfcc_1": 0.12, "pitch": 0.02, "zcr": 0.08}
        result = format_permutation_importance(importance, top_k=3)
        assert result["method"] == "permutation_importance"
        assert len(result["top_features"]) == 3
        assert "disclaimer" in result
        # Top feature should have highest absolute importance
        assert result["top_features"][0] == "mfcc_1"

    def test_acoustic_disclaimer_correct(self) -> None:
        from app.ml.explainability.acoustic_explainer import DISCLAIMER
        assert "model-observed patterns" in DISCLAIMER.lower()
        assert "not clinical evidence" in DISCLAIMER.lower()

    def test_format_asr_quality_view(self) -> None:
        from app.ml.explainability.acoustic_explainer import format_asr_quality_view
        meta = {
            "language": "en", "language_prob": 0.92,
            "no_speech_prob": 0.03, "word_count": 12,
            "quality_score": 0.88, "rejection_reasons": [],
        }
        result = format_asr_quality_view(meta)
        assert result["type"] == "asr_quality"
        assert result["quality_score"] == 0.88
        assert "disclaimer" in result

    def test_shap_acoustic_explain(self) -> None:
        pytest.importorskip("shap", reason="shap not installed")
        from app.ml.explainability.acoustic_explainer import explain_acoustic_shap

        features = np.array([0.5, 0.3, 0.8, 0.1, 0.6], dtype=np.float32)
        names = ["mfcc_mean", "pitch_mean", "rms_mean", "zcr_mean", "spectral_centroid"]

        def fake_predict(arr):
            return np.column_stack([1 - arr[:, 0], arr[:, 0]])

        result = explain_acoustic_shap(features, names, fake_predict, max_evals=50)
        assert "attributions" in result
        assert len(result["attributions"]) == 5
        assert "disclaimer" in result


# ---------------------------------------------------------------------------
# Fusion explainer tests
# ---------------------------------------------------------------------------

class TestFusionExplainer:

    def _mock_model_output(self, B: int = 1) -> dict:
        torch = pytest.importorskip("torch")
        from training.multimodal.fusion_baselines import MODALITY_ORDER
        return {
            "risk_score": torch.tensor([0.72]),
            "uncertainty": torch.tensor([0.18]),
            "gate_weights": torch.tensor([[0.8, 0.5, 0.3]]),
            "attention_weights": torch.rand(B, 3, 3),
            "emotion_profile": torch.softmax(torch.randn(B, 7), dim=1),
            "used_modalities": MODALITY_ORDER,
        }

    def test_build_fusion_explanation_keys(self) -> None:
        from app.ml.explainability.fusion_explainer import build_fusion_explanation
        out = self._mock_model_output()
        result = build_fusion_explanation(out)
        for key in ("risk_score", "uncertainty", "risk_tier", "used_modalities",
                    "gate_contribution", "cross_modal_attention", "disclaimer"):
            assert key in result

    def test_risk_tier_from_score(self) -> None:
        from app.ml.explainability.fusion_explainer import _score_to_tier_label
        assert _score_to_tier_label(0.70) == "high"
        assert _score_to_tier_label(0.50) == "moderate"
        assert _score_to_tier_label(0.20) == "low"

    def test_gate_contribution_percentages_sum_to_100(self) -> None:
        from app.ml.explainability.fusion_explainer import format_gate_contribution
        result = format_gate_contribution(
            gate_weights=[0.8, 0.5, 0.3],
            used_modalities=["text", "audio", "video"],
        )
        total_pct = sum(m["contribution_pct"] for m in result["modalities"])
        assert abs(total_pct - 100.0) < 0.5

    def test_gate_contribution_disclaimer(self) -> None:
        from app.ml.explainability.fusion_explainer import format_gate_contribution, GATE_DISCLAIMER
        result = format_gate_contribution([0.9, 0.0, 0.0], ["text"])
        assert result["disclaimer"] == GATE_DISCLAIMER
        assert "not causal proof" in GATE_DISCLAIMER.lower()

    def test_attention_matrix_row_count(self) -> None:
        from app.ml.explainability.fusion_explainer import format_attention_matrix
        attn = np.random.rand(3, 3)
        result = format_attention_matrix(attn)
        assert len(result["matrix"]) == 3
        assert "disclaimer" in result

    def test_attention_matrix_batch_averaged(self) -> None:
        from app.ml.explainability.fusion_explainer import format_attention_matrix
        attn = np.random.rand(4, 3, 3)  # batch of 4
        result = format_attention_matrix(attn)
        assert len(result["matrix"]) == 3  # still 3 rows after averaging

    def test_emotion_profile_in_output(self) -> None:
        from app.ml.explainability.fusion_explainer import build_fusion_explanation
        from training.vision.fer_baseline import FER_EMOTIONS
        out = self._mock_model_output()
        result = build_fusion_explanation(out, emotion_labels=FER_EMOTIONS)
        ep = result["emotion_profile"]
        assert ep is not None
        assert ep["predicted_emotion"] in FER_EMOTIONS

    def test_fusion_disclaimer_correct(self) -> None:
        from app.ml.explainability.fusion_explainer import DISCLAIMER
        assert "diagnostic signals" in DISCLAIMER.lower()
        assert "not causal proof" in DISCLAIMER.lower()
        assert "research screening tool" in DISCLAIMER.lower()


# ---------------------------------------------------------------------------
# Vision explainer tests
# ---------------------------------------------------------------------------

class TestVisionExplainer:

    def test_build_vision_explanation_keys(self) -> None:
        cv2 = pytest.importorskip("cv2", reason="opencv not installed")
        from app.ml.explainability.vision_explainer import build_vision_explanation

        heatmap = np.random.rand(224, 224).astype(np.float32)
        gradcam_result = {
            "heatmap": heatmap,
            "emotion_label": "sad",
            "target_class": 5,
            "disclaimer": "Emotion model visualization.",
        }
        result = build_vision_explanation(gradcam_result, include_heatmap_png=True)
        assert result["type"] == "gradcam"
        assert result["emotion_label"] == "sad"
        assert "disclaimer" in result

    def test_heatmap_base64_is_valid_png(self) -> None:
        cv2 = pytest.importorskip("cv2", reason="opencv not installed")
        from app.ml.explainability.vision_explainer import gradcam_to_base64_png

        heatmap = np.random.rand(48, 48).astype(np.float32)
        b64 = gradcam_to_base64_png(heatmap, height=64, width=64)
        raw = base64.b64decode(b64)
        # PNG magic bytes
        assert raw[:4] == b"\x89PNG"

    def test_vision_disclaimer_contains_emotion(self) -> None:
        from app.ml.explainability.vision_explainer import DISCLAIMER
        assert "emotion" in DISCLAIMER.lower()
        assert "not a depression-risk indicator" in DISCLAIMER.lower()

    def test_face_quality_included_in_output(self) -> None:
        cv2 = pytest.importorskip("cv2", reason="opencv not installed")
        from app.ml.explainability.vision_explainer import build_vision_explanation

        heatmap = np.zeros((48, 48), dtype=np.float32)
        gradcam_result = {"heatmap": heatmap, "emotion_label": "happy",
                          "target_class": 3, "disclaimer": "test"}
        quality_meta = {
            "quality_score": 0.82, "reliability": True, "rejection_reasons": []
        }
        result = build_vision_explanation(
            gradcam_result, face_quality_metadata=quality_meta,
            include_heatmap_png=False
        )
        assert "face_quality" in result
        assert result["face_quality"]["quality_score"] == 0.82
