"""Phase 2 tests — dataset/split integrity, preprocessing, baselines, artifacts.

These tests never train on the test split, never download a model, and never
touch models/final or the serving gate. Heavy encoders are not instantiated;
the small trained heads are exercised on CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.phase2 import dataset as ds  # noqa: E402
from ml.phase2.evaluation import (  # noqa: E402
    aggregate_by_participant,
    metric_bundle,
    participant_bootstrap_ci,
)


# ---------------------------------------------------------------------------
# Dataset / manifest
# ---------------------------------------------------------------------------

def test_manifest_has_verified_counts_and_labels():
    rows = ds.load_manifest()
    participants = ds.load_participants()
    assert len(rows) == 486
    assert len(participants) == 162
    assert len({r["participant_key"] for r in rows}) == 162
    assert sum(1 for r in rows if r["audio_usable"]) == 485
    assert sum(p["label"] for p in participants) == 53
    # SDS derivation documented in Phase 1 must hold exactly.
    for p in participants:
        assert p["sds_index"] == pytest.approx(p["sds_raw"] * 1.25)
        assert p["label"] == (1 if p["sds_index"] >= 50 else 0)


def test_participant_key_keeps_source_partition():
    participants = ds.load_participants()
    by_id = {}
    for p in participants:
        by_id.setdefault(p["source_subject_id"], set()).add(p["source_split"])
    colliding_ids = {k: v for k, v in by_id.items() if len(v) > 1}
    # 58 numeric ids exist in both legacy folders; a bare subject_id key would
    # silently merge 58 different people.
    assert len(colliding_ids) == 58
    for p in participants:
        assert p["participant_key"] == f"eatd_{p['source_split']}_{p['source_subject_id']}"


def test_splits_are_participant_level_and_leak_free():
    splits = ds.load_splits()
    counts = ds.assert_split_integrity(splits)
    assert counts == {"train": 114, "validation": 24, "test": 24}

    assigned = set(splits)
    for split in ("train", "validation", "test"):
        rows = ds.load_sessions(split, modality="text")
        keys = {r["participant_key"] for r in rows}
        assert keys, f"{split} unexpectedly empty"
        assert keys <= assigned
        # Three prompted topics per participant, all in the same split.
        for key in keys:
            assert splits[key] == split
    train = {r["participant_key"] for r in ds.load_sessions("train", modality="text")}
    test = {r["participant_key"] for r in ds.load_sessions("test", modality="text")}
    assert not (train & test)


def test_text_and_audio_sessions_align():
    text_rows = {(r["participant_key"], r["topic"]): r for r in ds.load_sessions("test", modality="text")}
    audio_rows = {(r["participant_key"], r["topic"]): r for r in ds.load_sessions("test", modality="audio")}
    assert set(text_rows) == set(audio_rows)
    for key, row in audio_rows.items():
        assert ds.resolve_path(row["audio_path"]).exists()
        assert row["audio_duration_sec"] > 0
        assert text_rows[key]["participant_label"] == row["participant_label"]


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def test_cjk_tokenization_is_character_level_and_normalized():
    from ml.phase2.text_tfidf import cjk_tokenize

    tokens = cjk_tokenize("我 今天 很累，hello world 123")
    assert "我" in tokens and "累" in tokens
    assert "hello" in tokens and "world" in tokens and "123" in tokens
    assert all(len(t) > 0 for t in tokens)
    # URLs are masked by the shared preprocessor, not tokenized as words.
    assert "<URL>" not in "".join(cjk_tokenize("see https://example.com now"))


def test_phase2_specific_tokens_are_cjk_single_chars():
    from ml.phase2.text_tfidf import cjk_tokenize

    tokens = cjk_tokenize("焦虑和失眠")
    assert tokens == ["焦", "虑", "和", "失", "眠"]


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

def test_chunk_audio_respects_chunk_size_and_cap():
    from ml.phase2.voice_wav2vec2 import chunk_audio

    sr = 16000
    y = np.zeros(sr * 21, dtype=np.float32)  # 21 s
    chunks = chunk_audio(y, sr, chunk_sec=8.0, max_chunks=12)
    assert [len(c) for c in chunks] == [sr * 8, sr * 8, sr * 5]
    capped = chunk_audio(np.zeros(sr * 200, dtype=np.float32), sr, chunk_sec=8.0, max_chunks=3)
    assert len(capped) == 3
    # Sub-second tails are dropped, never padded with silence.
    tail = chunk_audio(np.zeros(int(sr * 8.5), dtype=np.float32), sr)
    assert len(tail) == 1


def test_egemaps_extraction_produces_real_features():
    import opensmile  # noqa: F401  (environment probe)

    from ml.phase2.voice_egemaps import FeatureExtractionConfig, extract_session_features

    row = next(
        r
        for r in ds.load_manifest()
        if r["audio_usable"] and r["audio_duration_sec"] > 5
    )
    values, names, info = extract_session_features(row)
    assert values is not None and not info["excluded"]
    assert len(names) == len(values) == 88
    assert np.isfinite(values).all()
    assert any("F0semitone" in n for n in names)
    assert FeatureExtractionConfig(opensmile_version="test").feature_set == "eGeMAPSv02"


def test_egemaps_excludes_unusable_session_with_reason():
    from ml.phase2.voice_egemaps import extract_session_features

    unusable = next(r for r in ds.load_manifest() if not r["audio_usable"])
    values, names, info = extract_session_features(unusable)
    assert values is None and names == []
    assert info["excluded"] is True
    assert "unusable" in info["reason"]


# ---------------------------------------------------------------------------
# Checkpoint loading / reproducibility
# ---------------------------------------------------------------------------

def _tiny_head(in_dim: int = 8):
    import torch

    from ml.phase2.text_transformer import FrozenTextHead

    torch.manual_seed(0)
    return FrozenTextHead(in_dim, hidden=4, dropout=0.0)


def test_phase2_head_roundtrip_is_reproducible():
    import tempfile

    import torch

    from ml.phase2.text_transformer import head_probabilities, load_text_head

    model = _tiny_head()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        torch.save(model.state_dict(), tmp_path / "head.pt")
        (tmp_path / "metadata.json").write_text(
            json.dumps(
                {"config": {"embedding_dim": 8, "head": {"hidden": 4, "dropout": 0.0}}}
            ),
            encoding="utf-8",
        )
        loaded, meta = load_text_head(tmp_path)
    assert meta["config"]["embedding_dim"] == 8

    X = np.random.default_rng(0).normal(size=(6, 8)).astype(np.float32)
    p1 = head_probabilities(model, X)
    p2 = head_probabilities(loaded, X)
    p3 = head_probabilities(loaded, X)
    assert np.allclose(p1, p2, atol=1e-6)
    assert np.allclose(p2, p3, atol=1e-9)  # eval mode → no dropout randomness
    assert ((p2 >= 0) & (p2 <= 1)).all()


def test_phase2_head_runs_on_cpu():
    import torch

    from ml.phase2.text_transformer import head_probabilities

    model = _tiny_head()
    assert next(model.parameters()).device.type == "cpu"
    probs = head_probabilities(model, np.zeros((2, 8), dtype=np.float32))
    assert probs.shape == (2,)


def test_tfidf_pipeline_is_deterministic():
    from ml.phase2.text_tfidf import fit_model, predict_rows

    rows = ds.load_sessions("train", modality="text")[:40]
    features, clf = fit_model(rows, kind="word", C=1.0, class_weight="balanced")
    p1 = predict_rows(features, clf, rows[:8])
    p2 = predict_rows(features, clf, rows[:8])
    assert np.array_equal(p1, p2)
    assert ((p1 >= 0) & (p1 <= 1)).all()


def test_phase2_artifacts_are_not_on_the_serving_path():
    from backend.app import model_registry

    assert model_registry.TEXT_MODEL_DIR.parts[-3:] == ("models", "final", "text_model")
    assert "experiments" not in model_registry.TEXT_MODEL_DIR.parts
    # Phase 2 checkpoints live outside the registry's text path.
    phase2_text = PROJECT_ROOT / "models" / "experiments" / "eatd_text_tfidf_v1"
    if phase2_text.exists():
        assert model_registry.TEXT_MODEL_DIR != phase2_text


# ---------------------------------------------------------------------------
# Metrics / aggregation
# ---------------------------------------------------------------------------

def test_metric_bundle_reports_undefined_auc_as_none():
    m = metric_bundle([0, 0, 0], [0.1, 0.2, 0.3])
    assert m["roc_auc"] is None and m["pr_auc"] is None
    m2 = metric_bundle([0, 1], [0.2, 0.8])
    assert m2["roc_auc"] == 1.0 and m2["confusion_matrix"]["tp"] == 1


def test_participant_aggregation_keeps_sessions_together():
    keys = ["p1", "p1", "p1", "p2", "p2", "p2"]
    labels = [1, 1, 1, 0, 0, 0]
    probs = [0.2, 0.4, 0.6, 0.1, 0.3, 0.5]
    agg = aggregate_by_participant(keys, labels, probs)
    assert list(agg["participants"]) == ["p1", "p2"]
    assert np.allclose(agg["probs"], [0.4, 0.3])

    with pytest.raises(AssertionError):
        aggregate_by_participant(["p1", "p1"], [0, 1], [0.1, 0.9])


def test_participant_bootstrap_is_cluster_based():
    participant_labels = [0 if i < 6 else 1 for i in range(12)]
    keys = [f"p{i}" for i in range(12)] * 3
    labels = participant_labels * 3
    probs = [0.2 if y == 0 else 0.8 for y in labels]
    ci = participant_bootstrap_ci(keys, labels, probs, n_boot=200, seed=1)
    assert ci["n_participants"] == 12
    assert ci["n_positive_participants"] == 6
    assert 0.0 <= ci["ci"]["roc_auc"]["lo"] <= ci["ci"]["roc_auc"]["hi"] <= 1.0
