"""Tests for Phase 10 — speech emotion model and feature attribution."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("librosa", reason="librosa not installed")

from training.voice.speech_emotion_model import (
    CNNBiLSTMEmotionModel,
    SpeechEmotionConfig,
    SPEECH_EMOTIONS,
    SPEECH_EMOTION_TO_ID,
    DISCLAIMER,
    acoustic_permutation_importance,
    infer_speech_emotion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(n_classes: int = 7, bilstm_hidden: int = 32, embedding_dim: int = 64) -> SpeechEmotionConfig:
    cfg = SpeechEmotionConfig(
        n_classes=n_classes,
        bilstm_hidden=bilstm_hidden,
        embedding_dim=embedding_dim,
        cnn_channels=[16, 32, 64],
        n_mfcc_features=120,
    )
    return cfg


def _fake_mfcc(batch: int = 2, n_features: int = 120, T: int = 160) -> "torch.Tensor":
    return torch.randn(batch, n_features, T)


# ---------------------------------------------------------------------------
# Architecture tests
# ---------------------------------------------------------------------------

class TestCNNBiLSTMEmotionModel:

    def test_forward_output_shapes(self) -> None:
        cfg = _config()
        model = CNNBiLSTMEmotionModel(cfg)
        mfcc = _fake_mfcc()
        outputs = model(mfcc)
        assert outputs["logits"].shape == (2, 7)
        assert outputs["probs"].shape == (2, 7)
        assert outputs["embedding"].shape == (2, 64)
        assert outputs["attention_weights"] is not None

    def test_probs_sum_to_one(self) -> None:
        model = CNNBiLSTMEmotionModel(_config())
        probs = model(_fake_mfcc())["probs"]
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_embedding_dimension_configurable(self) -> None:
        for dim in [64, 128, 256]:
            cfg = _config(embedding_dim=dim)
            model = CNNBiLSTMEmotionModel(cfg)
            out = model(_fake_mfcc())
            assert out["embedding"].shape[1] == dim

    def test_attention_weights_shape_matches_sequence(self) -> None:
        cfg = _config()
        model = CNNBiLSTMEmotionModel(cfg)
        mfcc = _fake_mfcc(batch=3, T=160)
        out = model(mfcc)
        # After 3 MaxPool2 → T/8 = 20 frames
        assert out["attention_weights"].ndim == 2
        assert out["attention_weights"].shape[0] == 3

    def test_different_sequence_lengths_handled(self) -> None:
        """Model must handle variable-length inputs without error."""
        model = CNNBiLSTMEmotionModel(_config())
        for T in [80, 160, 320]:
            out = model(_fake_mfcc(T=T))
            assert out["logits"].shape == (2, 7)

    def test_no_nan_in_outputs(self) -> None:
        model = CNNBiLSTMEmotionModel(_config())
        out = model(_fake_mfcc())
        for key in ("logits", "probs", "embedding"):
            assert not torch.isnan(out[key]).any(), f"NaN in {key}"


# ---------------------------------------------------------------------------
# Emotion label safety
# ---------------------------------------------------------------------------

class TestEmotionLabelSafety:

    def test_emotion_labels_not_depression(self) -> None:
        for label in SPEECH_EMOTIONS:
            assert "depress" not in label.lower()
            assert "disorder" not in label.lower()

    def test_disclaimer_present_and_correct(self) -> None:
        assert "emotion" in DISCLAIMER.lower()
        assert "not clinical evidence" in DISCLAIMER.lower()
        assert "mental-health" in DISCLAIMER.lower()

    def test_infer_result_has_disclaimer(self) -> None:
        model = CNNBiLSTMEmotionModel(_config())
        mfcc = np.random.randn(120, 200).astype(np.float32)
        result = infer_speech_emotion(model, mfcc, max_frames=160)
        assert "disclaimer" in result
        assert result["predicted_emotion"] in SPEECH_EMOTIONS
        assert 0.0 <= result["confidence"] <= 1.0

    def test_infer_embedding_length(self) -> None:
        cfg = _config(embedding_dim=64)
        model = CNNBiLSTMEmotionModel(cfg)
        mfcc = np.random.randn(120, 200).astype(np.float32)
        result = infer_speech_emotion(model, mfcc, max_frames=160)
        assert len(result["embedding"]) == 64

    def test_all_emotion_probs_present(self) -> None:
        model = CNNBiLSTMEmotionModel(_config())
        mfcc = np.random.randn(120, 200).astype(np.float32)
        result = infer_speech_emotion(model, mfcc, max_frames=160)
        assert set(result["emotion_probabilities"].keys()) == set(SPEECH_EMOTIONS)


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

class TestPermutationImportance:

    def test_importance_keys_match_n_features(self) -> None:
        model = CNNBiLSTMEmotionModel(_config())
        mfcc = _fake_mfcc(batch=8, T=160)
        labels = torch.randint(0, 7, (8,))
        result = acoustic_permutation_importance(model, mfcc, labels, n_repeats=2)
        assert len(result["importances"]) == 120   # n_mfcc_features

    def test_baseline_accuracy_in_unit_interval(self) -> None:
        model = CNNBiLSTMEmotionModel(_config())
        mfcc = _fake_mfcc(batch=8, T=160)
        labels = torch.randint(0, 7, (8,))
        result = acoustic_permutation_importance(model, mfcc, labels, n_repeats=2)
        assert 0.0 <= result["baseline_accuracy"] <= 1.0

    def test_importance_disclaimer_present(self) -> None:
        model = CNNBiLSTMEmotionModel(_config())
        mfcc = _fake_mfcc(batch=4, T=160)
        labels = torch.randint(0, 7, (4,))
        result = acoustic_permutation_importance(model, mfcc, labels, n_repeats=1)
        assert "disclaimer" in result
        assert "attribution" in result["disclaimer"].lower()
        assert "not causal" in result["disclaimer"].lower()

    def test_custom_feature_names_respected(self) -> None:
        model = CNNBiLSTMEmotionModel(_config())
        mfcc = _fake_mfcc(batch=4, T=160)
        labels = torch.randint(0, 7, (4,))
        names = [f"feat_{i}" for i in range(120)]
        result = acoustic_permutation_importance(model, mfcc, labels, n_repeats=1, feature_names=names)
        assert all(k.startswith("feat_") for k in result["importances"])
