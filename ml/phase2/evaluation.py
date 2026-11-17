"""
Phase 2 evaluation utilities — session metrics, participant aggregation,
participant-level bootstrap confidence intervals, calibration, artifacts.

Design notes (small-sample honesty):
  * The locked test split has 24 participants / 8 positive. Every headline
    number in the Phase 2 report is accompanied by a participant-level
    bootstrap CI, because session-level CIs are far too narrow: the three
    sessions of one participant are not independent observations.
  * Participant aggregation = mean of that participant's session probabilities
    (documented alternative: max). Labels are participant-level, so a
    participant score is compared against the participant label.
  * No metric in this module has a default value that could hide a failure:
    undefined metrics (e.g. ROC-AUC with a single class) return None.

Exports:
    metric_bundle, confusion, aggregate_by_participant,
    bootstrap_ci, participant_bootstrap_ci, calibration_bundle,
    write_json, experiment_metadata
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ml.evaluation.metrics import (
    brier_score,
    compute_full_metrics,
    expected_calibration_error,
    reliability_curve,
)
from ml.phase2.dataset import (
    DATASET_VERSION,
    LABEL_DEFINITION,
    PARTICIPANT_KEY_DEFINITION,
    PROJECT_ROOT,
    SPLIT_VERSION,
)

_RESULT_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
)


def confusion(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> Dict[str, int]:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
    }


def metric_bundle(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, object]:
    """All requested metrics from actual predictions. Undefined AUC → None."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if len(y) == 0:
        return {k: None for k in _RESULT_KEYS} | {"n": 0, "n_positive": 0}
    if len(np.unique(y)) < 2:
        base = {k: None for k in _RESULT_KEYS}
        base.update({"n": int(len(y)), "n_positive": int(y.sum())})
        return base

    m = compute_full_metrics(y, p, threshold)
    out: Dict[str, object] = {
        "accuracy": round(float(m["accuracy"]), 4),
        "precision": round(float(m["precision"]), 4),
        "recall": round(float(m["recall"]), 4),
        "f1": round(float(m["f1"]), 4),
        "roc_auc": round(float(m["auc"]), 4),
        "pr_auc": round(float(m["pr_auc"]), 4),
        "sensitivity": round(float(m["sensitivity"]), 4),
        "specificity": round(float(m["specificity"]), 4),
        "balanced_accuracy": round(float(m["balanced_accuracy"]), 4),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "threshold": threshold,
        "confusion_matrix": confusion(y, p, threshold),
    }
    return out


def aggregate_by_participant(
    participant_keys: Sequence[str],
    labels: Sequence[int],
    probs: Sequence[float],
    how: str = "mean",
) -> Dict[str, np.ndarray]:
    """Collapse session predictions to one score per participant.

    Returns {'participants', 'labels', 'probs'}. ``how`` is 'mean' (primary)
    or 'max' (reported as a sensitivity check).
    """
    if len(participant_keys) != len(probs) or len(probs) != len(labels):
        raise ValueError("participant_keys, labels and probs must be aligned")
    order: List[str] = []
    bucket: Dict[str, List[float]] = {}
    plabel: Dict[str, int] = {}
    for pk, y, p in zip(participant_keys, labels, probs):
        if pk not in bucket:
            bucket[pk] = []
            order.append(pk)
            plabel[pk] = int(y)
        bucket[pk].append(float(p))
        if plabel[pk] != int(y):
            raise AssertionError(f"participant {pk} has inconsistent labels across sessions")
    if how == "mean":
        agg = [float(np.mean(bucket[pk])) for pk in order]
    elif how == "max":
        agg = [float(np.max(bucket[pk])) for pk in order]
    else:
        raise ValueError("how must be 'mean' or 'max'")
    return {
        "participants": np.asarray(order, dtype=object),
        "labels": np.asarray([plabel[pk] for pk in order], dtype=int),
        "probs": np.asarray(agg, dtype=float),
    }


def bootstrap_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Percentile bootstrap over sessions (row-level resampling)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {k: [] for k in _RESULT_KEYS}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metric_bundle(y[idx], p[idx])
        for k in _RESULT_KEYS:
            if m.get(k) is not None:
                draws[k].append(float(m[k]))
    return {
        k: (
            {
                "lo": round(float(np.percentile(v, 100 * alpha / 2)), 4),
                "hi": round(float(np.percentile(v, 100 * (1 - alpha / 2))), 4),
                "n_boot_valid": len(v),
            }
            if len(v) >= max(20, n_boot // 10)
            else {"lo": None, "hi": None, "n_boot_valid": len(v)}
        )
        for k, v in draws.items()
    }


def participant_bootstrap_ci(
    participant_keys: Sequence[str],
    labels: Sequence[int],
    probs: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    how: str = "mean",
) -> Dict[str, object]:
    """Cluster bootstrap: resample PARTICIPANTS, keep all their sessions.

    This is the CI quoted in the Phase 2 report. With 24 test participants
    (8 positive) the intervals are wide by construction — that is the point.
    """
    agg = aggregate_by_participant(participant_keys, labels, probs, how=how)
    y, p = agg["labels"], agg["probs"]
    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {k: [] for k in _RESULT_KEYS}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = metric_bundle(y[idx], p[idx])
        for k in _RESULT_KEYS:
            if m.get(k) is not None:
                draws[k].append(float(m[k]))
    ci = {
        k: {
            "lo": round(float(np.percentile(v, 100 * alpha / 2)), 4),
            "hi": round(float(np.percentile(v, 100 * (1 - alpha / 2))), 4),
            "n_boot_valid": len(v),
        }
        for k, v in draws.items()
        if len(v) >= max(20, n_boot // 10)
    }
    return {
        "method": "participant_cluster_bootstrap",
        "n_participants": int(n),
        "n_positive_participants": int(y.sum()),
        "n_boot": n_boot,
        "seed": seed,
        "ci": ci,
        "note": (
            "Percentile CI from resampling participants (all sessions of a "
            "resampled participant are kept together). Wide intervals reflect "
            "the 24-participant test split."
        ),
    }


def calibration_bundle(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10) -> Dict[str, object]:
    return {
        "brier_score": round(float(brier_score(y_true, y_prob)), 4),
        "expected_calibration_error": round(
            float(expected_calibration_error(y_true, y_prob, n_bins=n_bins)), 4
        ),
        "n_bins": n_bins,
        "reliability_curve": reliability_curve(y_true, y_prob, n_bins=n_bins),
    }


def evaluate_predictions(
    rows: Sequence[dict],
    probs: Sequence[float],
    threshold: float = 0.5,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, object]:
    """Session + participant metrics, CIs, calibration for one prediction set."""
    y_true = [int(r["participant_label"]) for r in rows]
    keys = [r["participant_key"] for r in rows]
    session = metric_bundle(y_true, probs, threshold)
    agg = aggregate_by_participant(keys, y_true, probs, how="mean")
    participant = metric_bundle(agg["labels"], agg["probs"], threshold)
    agg_max = aggregate_by_participant(keys, y_true, probs, how="max")
    return {
        "session_level": session,
        "session_bootstrap_ci_rowlevel": bootstrap_ci(y_true, probs, n_boot=n_boot, seed=seed),
        "participant_level": participant,
        "participant_level_max_pooling": metric_bundle(agg_max["labels"], agg_max["probs"], threshold),
        "participant_bootstrap_ci": participant_bootstrap_ci(
            keys, y_true, probs, n_boot=n_boot, seed=seed
        ),
        "calibration": calibration_bundle(y_true, probs),
        "aggregation": "participant probability = mean of that participant's session probabilities",
    }


def _sha256(path: Path) -> Optional[str]:
    if not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def experiment_metadata(
    experiment_id: str,
    model_version: str,
    modality: str,
    config: Dict[str, object],
    training: Dict[str, object],
    validation_metrics: Dict[str, object],
    test_metrics: Optional[Dict[str, object]] = None,
    checkpoint_paths: Sequence[Path] = (),
    notes: Sequence[str] = (),
) -> Dict[str, object]:
    """Provenance block required for every Phase 2 artifact."""
    import torch

    return {
        "experiment_id": experiment_id,
        "model_version": model_version,
        "modality": modality,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "name": "EATD",
            "dataset_version": DATASET_VERSION,
            "split_version": SPLIT_VERSION,
            "splits_csv": "data/processed/eatd/splits.csv",
            "manifest_csv": "data/processed/eatd/manifest.csv",
            "participant_key": PARTICIPANT_KEY_DEFINITION,
            "label_definition": LABEL_DEFINITION,
        },
        "config": config,
        "training": training,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "checkpoint_provenance": {
            str(Path(p)): _sha256(Path(p)) for p in checkpoint_paths
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "git_commit": _git_commit(),
        },
        "serving": {
            "enabled": False,
            "reason": (
                "Phase 2 research artifact. Serving is gated on "
                "backend/app/model_registry.py and only reads "
                "models/final/text_model; these checkpoints are not served."
            ),
        },
        "clinical_validation": False,
        "notes": list(notes),
    }


def write_json(path: Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
