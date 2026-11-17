"""Tests for Phase 5F visual feature extraction, cache validation, and integrity."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Synthetic fixtures and helper functions
# --------------------------------------------------------------------------- #
def make_synthetic_visual_csv(n_frames: int = 100) -> bytes:
    """Create synthetic OpenFace-like CSV data with 538 columns."""
    rng = np.random.default_rng(42)
    cols = ["frame", "face_id", "timestamp", "confidence", "success"]
    # Add gaze (8)
    cols += [f"gaze_{i}_{axis}" for i in (0, 1) for axis in ("x", "y", "z")] + ["gaze_angle_x", "gaze_angle_y"]
    # Add pose (6)
    cols += ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
    # Add AUs (35)
    cols += [f"AU{i:02d}_r" for i in range(1, 18)] + [f"AU{i:02d}_c" for i in range(1, 19)]
    # Add remaining landmarks to reach 538
    while len(cols) < 538:
        cols.append(f"eye_lmk_x_{len(cols)}")

    data = {}
    data["frame"] = np.arange(n_frames)
    data["face_id"] = np.zeros(n_frames, dtype=int)  # constant
    data["timestamp"] = np.linspace(0, 3.3, n_frames)
    data["confidence"] = rng.uniform(0.85, 0.99, size=n_frames)
    data["success"] = np.ones(n_frames, dtype=int)  # constant

    for c in cols[5:]:
        data[c] = rng.normal(0.0, 1.0, size=n_frames)

    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def slope_vs_normalized_index(values: np.ndarray) -> np.ndarray:
    """Least-squares slope of each column against normalized frame index 0..1."""
    n = values.shape[0]
    if n < 2:
        return np.zeros(values.shape[1], dtype=np.float64)
    x = np.arange(n, dtype=np.float64)
    x = (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else x
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom == 0:
        return np.zeros(values.shape[1], dtype=np.float64)
    y = values.astype(np.float64)
    yc = y - y.mean(axis=0, keepdims=True)
    return (xc @ yc) / denom


def aggregate_frame_matrix(frame_values: np.ndarray) -> np.ndarray:
    """Return one vector of 7 STATS for each column (flattened column-major)."""
    n_frames, n_cols = frame_values.shape
    out: list[float] = []
    arr = frame_values.astype(np.float64)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0, ddof=0)
    medians = np.median(arr, axis=0)
    p25 = np.percentile(arr, 25, axis=0)
    p75 = np.percentile(arr, 75, axis=0)
    p05 = np.percentile(arr, 5, axis=0)
    p95 = np.percentile(arr, 95, axis=0)
    slopes = slope_vs_normalized_index(arr)
    for j in range(n_cols):
        out.extend(
            [
                float(means[j]),
                float(stds[j]),
                float(medians[j]),
                float(p75[j] - p25[j]),
                float(p05[j]),
                float(p95[j]),
                float(slopes[j]),
            ]
        )
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_visual_zip_streaming_and_sample_id_extraction():
    """Verify in-memory extraction of sample ID and data from zip without writing to disk."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("AVDD-TCMI-dataset/HDS/101/101_visual.csv", make_synthetic_visual_csv(50))
        zf.writestr("AVDD-TCMI-dataset/HDS_labels.csv", "101,1,M\n")

    bio.seek(0)
    with zipfile.ZipFile(bio, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith("_visual.csv")]
        assert len(members) == 1
        sid = int(members[0].split("/")[-2])
        assert sid == 101

        raw = zf.read(members[0])
        df = pd.read_csv(io.BytesIO(raw))
        assert df.shape[1] == 538
        assert df.shape[0] == 50


def test_column_policy_filtering():
    """Verify Section 6.4: 2 constant dropped, 3 metadata excluded, 533 retained."""
    csv_bytes = make_synthetic_visual_csv(30)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    all_cols = list(df.columns)
    assert len(all_cols) == 538

    drop_cols = {"face_id", "success"}
    exclude_cols = {"frame", "timestamp", "confidence"}

    retained = [c for c in all_cols if c not in drop_cols and c not in exclude_cols]
    assert len(retained) == 533
    assert "face_id" not in retained
    assert "success" not in retained
    assert "confidence" not in retained
    assert "frame" not in retained
    assert "timestamp" not in retained


def test_aggregation_functionals_dimension_and_finite():
    """Verify 533 columns * 7 functionals = 3,731 features per sample, all finite."""
    rng = np.random.default_rng(123)
    frames = rng.normal(0, 1, size=(80, 533)).astype(np.float32)
    agg = aggregate_frame_matrix(frames)

    assert agg.ndim == 1
    assert len(agg) == 533 * 7
    assert len(agg) == 3731
    assert np.all(np.isfinite(agg))
    assert agg.dtype == np.float32


def test_slope_calculation_invariance():
    """Verify slope functional detects positive/negative/flat trends."""
    x_pos = np.linspace(0, 10, 50).reshape(-1, 1)
    s_pos = slope_vs_normalized_index(x_pos)
    assert s_pos[0] > 0

    x_neg = np.linspace(10, 0, 50).reshape(-1, 1)
    s_neg = slope_vs_normalized_index(x_neg)
    assert s_neg[0] < 0

    x_flat = np.full((50, 1), 5.0)
    s_flat = slope_vs_normalized_index(x_flat)
    assert s_flat[0] == 0.0


def test_split_integrity_fold_local_scaling():
    """Verify standardization is strictly fold-local (no leakage across train/test)."""
    rng = np.random.default_rng(42)
    X = rng.normal(10.0, 5.0, size=(100, 20))
    y = np.array([0] * 50 + [1] * 50)

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    train_idx, test_idx = next(rkf.split(X, y))

    scaler = StandardScaler()
    scaler.fit(X[train_idx])

    # Scaler mean must match train mean, not full dataset mean
    assert np.allclose(scaler.mean_, X[train_idx].mean(axis=0))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))

    X_te_scaled = scaler.transform(X[test_idx])
    assert np.all(np.isfinite(X_te_scaled))


def test_cached_visual_matrix_integrity():
    """If real avdd_visual_cache.npz exists, verify shape, dtype, and finiteness."""
    cache_path = PROJECT_ROOT / "data" / "processed" / "avdd_tcmi_phase5" / "avdd_visual_cache.npz"
    if not cache_path.exists():
        pytest.skip("avdd_visual_cache.npz not found")

    data = np.load(cache_path, allow_pickle=False)
    assert "X" in data
    assert "sample_ids" in data
    assert "subset" in data
    assert "labels" in data
    assert "confidence_mean" in data

    X = data["X"]
    assert X.shape == (1230, 3731)
    assert X.dtype == np.float32
    assert np.all(np.isfinite(X))

    subsets = data["subset"]
    assert (subsets == "HDS").sum() == 282
    assert (subsets == "SDS").sum() == 948

    # Sample IDs are 0..281 for HDS and 0..947 for SDS; within each subset they are unique
    hds_ids = data["sample_ids"][subsets == "HDS"]
    assert len(np.unique(hds_ids)) == 282, "HDS sample IDs must be unique (0..281)"

    sds_ids = data["sample_ids"][subsets == "SDS"]
    assert len(np.unique(sds_ids)) == 948, "SDS sample IDs must be unique (0..947)"

    # Composite key (subset, sample_id) must be globally unique
    composite = [f"{s}_{sid}" for s, sid in zip(subsets, data["sample_ids"])]
    assert len(np.unique(composite)) == 1230, "Composite (subset, sample_id) must be 1,230 unique keys"


def test_visual_column_audit_json_validity():
    """Verify data/reports/avdd_visual_feature_column_audit.json exists and adheres to schema."""
    audit_path = PROJECT_ROOT / "data" / "reports" / "avdd_visual_feature_column_audit.json"
    if not audit_path.exists():
        pytest.skip("avdd_visual_feature_column_audit.json not found")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["total_columns"] == 538
    assert audit["retained_columns_count"] == 533
    assert audit["excluded_columns_count"] == 5
    assert len(audit["columns"]) == 538
