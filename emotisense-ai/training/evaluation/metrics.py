"""Phase 15 — Comprehensive evaluation metrics.

Covers every metric class required by the project spec:
  - Classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC
  - Clinical proxies: sensitivity, specificity, confusion matrix
  - Calibration: Brier score, ECE (via attention_fusion module)
  - Per-split latency placeholders
  - Confidence intervals via bootstrap

All results carry a mandatory disclaimer.
The final test set must NOT be evaluated until the model is locked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DISCLAIMER = (
    "Metrics are computed on a research dataset. "
    "Performance on clinical populations has not been established "
    "and may differ substantially. "
    "These scores are not evidence of clinical validity."
)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    actual: list[int],
    predicted: list[int],
    probs: list[float],
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Compute full classification metric suite.

    Parameters
    ----------
    actual       : True integer labels (0 or 1)
    predicted    : Predicted integer labels
    probs        : Predicted positive-class probabilities
    bootstrap_ci : Whether to compute 95 % bootstrap CIs for F1 and ROC-AUC
    n_bootstrap  : Number of bootstrap samples
    ci_alpha     : Two-tailed alpha level

    Returns
    -------
    dict with all metrics + optional CIs + disclaimer
    """
    from sklearn.metrics import (
        accuracy_score, average_precision_score, brier_score_loss,
        confusion_matrix, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    actual_arr = np.array(actual)
    pred_arr = np.array(predicted)
    probs_arr = np.array(probs)

    tn, fp, fn, tp = confusion_matrix(actual_arr, pred_arr, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    has_both_classes = len(set(actual)) > 1
    metrics = {
        "accuracy":    round(float(accuracy_score(actual_arr, pred_arr)), 4),
        "precision":   round(float(precision_score(actual_arr, pred_arr, zero_division=0)), 4),
        "recall":      round(float(recall_score(actual_arr, pred_arr, zero_division=0)), 4),
        "f1":          round(float(f1_score(actual_arr, pred_arr, zero_division=0)), 4),
        "roc_auc":     round(float(roc_auc_score(actual_arr, probs_arr)), 4) if has_both_classes else None,
        "pr_auc":      round(float(average_precision_score(actual_arr, probs_arr)), 4) if has_both_classes else None,
        "brier_score": round(float(brier_score_loss(actual_arr, probs_arr)), 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": len(actual),
        "disclaimer": DISCLAIMER,
    }

    if bootstrap_ci and has_both_classes:
        metrics["bootstrap_ci"] = _bootstrap_ci(
            actual_arr, pred_arr, probs_arr, n_bootstrap, ci_alpha, seed
        )

    return metrics


def _bootstrap_ci(
    actual: np.ndarray,
    predicted: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score

    rng = np.random.default_rng(seed)
    f1_scores, aucs = [], []
    n = len(actual)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        a, p, pr = actual[idx], predicted[idx], probs[idx]
        if len(set(a)) < 2:
            continue
        f1_scores.append(f1_score(a, p, zero_division=0))
        aucs.append(roc_auc_score(a, pr))

    lo = alpha / 2
    hi = 1.0 - lo
    return {
        "f1_ci": [round(float(np.quantile(f1_scores, lo)), 4),
                  round(float(np.quantile(f1_scores, hi)), 4)],
        "roc_auc_ci": [round(float(np.quantile(aucs, lo)), 4),
                       round(float(np.quantile(aucs, hi)), 4)],
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Fairness / subgroup checks
# ---------------------------------------------------------------------------

def compute_subgroup_metrics(
    actual: list[int],
    predicted: list[int],
    probs: list[float],
    subgroup_labels: list[str],
) -> dict:
    """Compute metrics per subgroup (e.g. demographic / domain groups).

    Parameters
    ----------
    subgroup_labels : One label per sample identifying the subgroup.

    Returns
    -------
    dict with per-subgroup metrics and a disparity summary.
    """
    groups = sorted(set(subgroup_labels))
    subgroup_results: dict[str, dict] = {}

    for g in groups:
        idx = [i for i, s in enumerate(subgroup_labels) if s == g]
        if len(idx) < 2:
            subgroup_results[g] = {"skipped": "too_few_samples", "n": len(idx)}
            continue
        a = [actual[i] for i in idx]
        p = [predicted[i] for i in idx]
        pr = [probs[i] for i in idx]
        subgroup_results[g] = compute_classification_metrics(a, p, pr)

    # Disparity: max gap in F1 and sensitivity across groups
    f1_vals = [v["f1"] for v in subgroup_results.values() if "f1" in v]
    sens_vals = [v["sensitivity"] for v in subgroup_results.values() if "sensitivity" in v]

    disparity = {
        "f1_max_gap": round(float(max(f1_vals) - min(f1_vals)), 4) if len(f1_vals) > 1 else None,
        "sensitivity_max_gap": round(float(max(sens_vals) - min(sens_vals)), 4) if len(sens_vals) > 1 else None,
        "disclaimer": (
            "Large gaps between subgroups indicate potential bias. "
            "Investigate dataset balance, demographic representation, and "
            "domain shift before deployment."
        ),
    }

    return {
        "subgroups": subgroup_results,
        "disparity": disparity,
        "n_groups": len(groups),
    }


# ---------------------------------------------------------------------------
# Domain-shift detector
# ---------------------------------------------------------------------------

def detect_domain_shift(
    train_features: np.ndarray,
    eval_features: np.ndarray,
    method: str = "mmd",
) -> dict:
    """Estimate distributional shift between train and eval feature sets.

    Methods
    -------
    "mmd"    : Maximum Mean Discrepancy with RBF kernel (normalised)
    "kl_div" : Approximate KL divergence via Gaussian fit

    Parameters
    ----------
    train_features : (N_train, D) float array
    eval_features  : (N_eval, D) float array

    Returns
    -------
    dict with shift_score, method, interpretation, warning flag
    """
    if method == "mmd":
        score = _mmd_rbf(train_features, eval_features)
    elif method == "kl_div":
        score = _gaussian_kl(train_features, eval_features)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mmd' or 'kl_div'.")

    HIGH_SHIFT_THRESHOLD = 0.1
    warning = score > HIGH_SHIFT_THRESHOLD
    return {
        "shift_score": round(float(score), 4),
        "method": method,
        "high_shift_warning": warning,
        "interpretation": (
            "Score near 0 → similar distributions. "
            f"Score > {HIGH_SHIFT_THRESHOLD} → potential domain shift; "
            "performance may degrade on evaluation set."
        ),
    }


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Normalised squared MMD with RBF kernel on a random subsample."""
    n_sub = min(200, len(X), len(Y))
    rng = np.random.default_rng(0)
    X = X[rng.choice(len(X), n_sub, replace=False)]
    Y = Y[rng.choice(len(Y), n_sub, replace=False)]

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def _rbf(A: np.ndarray, B: np.ndarray) -> float:
        diff = A[:, None, :] - B[None, :, :]
        return float(np.exp(-gamma * (diff ** 2).sum(axis=2)).mean())

    return max(0.0, _rbf(X, X) + _rbf(Y, Y) - 2 * _rbf(X, Y))


def _gaussian_kl(X: np.ndarray, Y: np.ndarray) -> float:
    """Approximate KL(P‖Q) assuming Gaussian distributions."""
    mu_x, sigma_x = X.mean(axis=0), X.std(axis=0) + 1e-8
    mu_y, sigma_y = Y.mean(axis=0), Y.std(axis=0) + 1e-8
    kl = float(np.mean(
        np.log(sigma_y / sigma_x)
        + (sigma_x ** 2 + (mu_x - mu_y) ** 2) / (2 * sigma_y ** 2)
        - 0.5
    ))
    return max(0.0, kl)


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Container for a complete model evaluation result."""

    split: str
    metrics: dict = field(default_factory=dict)
    subgroup_metrics: Optional[dict] = None
    domain_shift: Optional[dict] = None
    modality_ablation: Optional[dict] = None
    calibration: Optional[dict] = None
    latency_ms: Optional[float] = None
    hardware: str = "unknown"
    model_checkpoint: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "metrics": self.metrics,
            "subgroup_metrics": self.subgroup_metrics,
            "domain_shift": self.domain_shift,
            "modality_ablation": self.modality_ablation,
            "calibration": self.calibration,
            "latency_ms": self.latency_ms,
            "hardware": self.hardware,
            "model_checkpoint": str(self.model_checkpoint),
            "notes": self.notes,
            "disclaimer": DISCLAIMER,
        }

    def save(self, output_path: Path) -> None:
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test-set guard
# ---------------------------------------------------------------------------

def assert_not_final_test(split_name: str, locked: bool) -> None:
    """Raise if attempting final test evaluation before model is locked."""
    if split_name.lower() == "test" and not locked:
        raise RuntimeError(
            "Final test evaluation is only permitted after the model is locked. "
            "Set locked=True only when no further model changes will be made."
        )


# ---------------------------------------------------------------------------
# Latency measurement helper
# ---------------------------------------------------------------------------

def measure_latency_ms(fn, n_warmup: int = 3, n_measure: int = 10) -> float:
    """Measure mean inference latency in milliseconds."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(times)), 2)
