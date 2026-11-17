"""
Metrics — Evaluation metrics for depression detection.

Computes standard classification metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, PR-AUC
  - Sensitivity, Specificity, Balanced Accuracy
  - Confusion matrix
  - Calibration: Brier score, Expected Calibration Error (ECE),
    reliability curve

Exports:
  compute_metrics — compute all metrics from labels and probabilities
  compute_confusion_matrix — get TP, FP, FN, TN counts
  compute_full_metrics — adds sensitivity/specificity/balanced accuracy/PR-AUC
  brier_score, expected_calibration_error, reliability_curve
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def compute_confusion_matrix(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float]:
    """Compute confusion matrix components.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Dict with 'tp', 'fp', 'fn', 'tn' counts.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR AUC via the precision–recall staircase (handles ties)."""
    n_pos = int(np.sum(y_true))
    if n_pos == 0 or len(y_true) - n_pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    ys = y_true[order]
    scores = y_score[order]
    tp = 0.0
    fp = 0.0
    auc = 0.0
    prev_recall = 0.0
    i = 0
    n = len(ys)
    while i < n:
        j = i
        while j < n and scores[j] == scores[i]:
            j += 1
        tp += float(np.sum(ys[i:j]))
        fp += float((j - i) - np.sum(ys[i:j]))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / n_pos
        auc += precision * (recall - prev_recall)
        prev_recall = recall
        i = j
    return float(auc)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC via staircase estimator (handles ties, no sklearn dependency)."""
    if np.isnan(y_score).any():
        mask = ~np.isnan(y_score)
        y_true = y_true[mask]
        y_score = y_score[mask]

    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(-y_score, kind="mergesort")
    ys = y_true[order]
    scores = y_score[order]

    area = 0.0
    tpr = 0.0
    i = 0
    n = len(ys)
    while i < n:
        j = i
        while j < n and scores[j] == scores[i]:
            j += 1
        p = int(np.sum(ys[i:j]))
        neg = (j - i) - p
        area += (tpr + (p / n_pos) / 2.0) * (neg / n_neg)
        tpr += p / n_pos
        i = j

    return float(area)


def compute_metrics(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all classification metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for class 1 (depressed).
        threshold: Classification threshold.

    Returns:
        Dict with 'accuracy', 'precision', 'recall', 'f1', 'auc'.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = compute_confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    auc = _roc_auc(y_true, y_prob)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def compute_full_metrics(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Extended evaluation metrics (adds to compute_metrics, never replaces).

    Returns accuracy, precision, recall(sensitivity), specificity, f1,
    balanced accuracy, ROC-AUC and PR-AUC. All computed from actual held-out
    predictions supplied by the caller — never fabricated.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    y_pred = (p >= threshold).astype(np.int64)
    cm = compute_confusion_matrix(y, y_pred)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    base = compute_metrics(y, p, threshold)
    y_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.asarray(y_prob, dtype=np.float64)
    base["sensitivity"] = float(sensitivity)
    base["specificity"] = float(specificity)
    base["balanced_accuracy"] = float((sensitivity + specificity) / 2.0)
    base["pr_auc"] = float(_pr_auc(y_arr, p_arr))
    return base


# ---------------------------------------------------------------------------
# Calibration metrics (evaluation only — fitting lives in
# backend/app/services/calibration_service.py, on validation data)
# ---------------------------------------------------------------------------

def brier_score(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    """Brier score = mean((p - y)^2). Lower is better."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> float:
    """ECE = sum_b (n_b/N) * |acc_b - conf_b| over equal-width bins."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def reliability_curve(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> List[Dict[str, Optional[float]]]:
    """Reliability diagram data: per-bin mean predicted prob vs observed frequency."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    curve: List[Dict[str, Optional[float]]] = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        curve.append(
            {
                "bin_lower": lo,
                "bin_upper": hi,
                "count": count,
                "mean_predicted_prob": round(float(p[mask].mean()), 4) if count else None,
                "observed_frequency": round(float(y[mask].mean()), 4) if count else None,
            }
        )
    return curve


# ===========================================================================
# Phase 5B — guarded metric layer
# ===========================================================================
#
# The functions above are the Phase 2 evaluators: they are frozen and they
# return 0.0 (ROC-AUC: 0.5) when a metric is mathematically undefined, which is
# acceptable for a single fixed protocol but not for a research harness. The
# layer below returns an explicit ``MetricResult`` that says whether a value is
# defined, why not when it is not, which data unit it was computed on, and how
# many observations it saw. Nothing here fabricates a number.
#
# Positive-class orientation: class 1 is the positive class in every metric.
# For EATD, class 1 = SDS index >= 50 (probable depression); recall is therefore
# sensitivity to depression, and specificity is the control pass rate.

POSITIVE_CLASS = 1
POSITIVE_CLASS_LABEL = (
    "class 1 is the positive class for every metric "
    "(EATD: SDS index >= 50, i.e. probable depression)"
)
EVALUATION_UNITS: Tuple[str, ...] = ("participant", "session", "sample")

#: Below this many positives in a fold, aggregate metrics are flagged.
MIN_POSITIVES_PER_FOLD_FOR_AGGREGATE = 5
#: Below this many positives, a test-set estimate is not reported as definitive.
MIN_POSITIVES_FOR_TEST_ESTIMATE = 10
#: An equal-frequency calibration bin with fewer members is suppressed.
DEFAULT_ECE_MIN_BIN_N = 2


class MetricInputError(ValueError):
    """Invalid metric input (misaligned, non-binary, or out-of-range)."""


class MetricUndefinedError(RuntimeError):
    """A metric was requested as a number but is undefined for this input."""


@dataclass(frozen=True)
class MetricResult:
    """One metric value plus everything needed to read it honestly."""

    name: str
    value: Optional[float]
    unit: str
    n: int
    n_positive: int
    defined: bool
    reason: Optional[str] = None
    threshold: Optional[float] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def as_float(self) -> float:
        """Return the value, or raise if the metric is undefined."""
        if not self.defined or self.value is None:
            raise MetricUndefinedError(
                f"metric {self.name!r} is undefined ({self.reason})"
            )
        return float(self.value)

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "name": self.name,
            "value": self.value,
            "defined": self.defined,
            "unit": self.unit,
            "n": self.n,
            "n_positive": self.n_positive,
        }
        if self.reason is not None:
            out["reason"] = self.reason
        if self.threshold is not None:
            out["threshold"] = float(self.threshold)
        if self.extra:
            out["extra"] = dict(self.extra)
        return out

    def __float__(self) -> float:
        return self.as_float()


def _validate_inputs(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    unit: str,
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and return float64 arrays, or raise with a specific reason."""
    if unit not in EVALUATION_UNITS:
        raise MetricInputError(
            f"{name}: unknown evaluation unit {unit!r}; expected one of "
            f"{list(EVALUATION_UNITS)}"
        )
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y.shape != p.shape:
        raise MetricInputError(
            f"{name}: y_true has {y.size} values but y_prob has {p.size}"
        )
    if y.size == 0:
        raise MetricInputError(f"{name}: empty input")
    if not np.isfinite(y).all():
        raise MetricInputError(f"{name}: y_true contains non-finite values")
    if not np.isfinite(p).all():
        raise MetricInputError(f"{name}: y_prob contains non-finite values")
    if not np.isin(y, (0.0, 1.0)).all():
        raise MetricInputError(
            f"{name}: y_true must be binary 0/1; got {sorted(set(y.tolist()))[:5]}"
        )
    if (p < 0.0).any() or (p > 1.0).any():
        raise MetricInputError(
            f"{name}: y_prob must be probabilities in [0, 1]; got "
            f"min={float(p.min()):.4f}, max={float(p.max()):.4f}"
        )
    return y, p


def _counts(y: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum((y_pred == 1) & (y == 1)))
    fp = float(np.sum((y_pred == 1) & (y == 0)))
    fn = float(np.sum((y_pred == 0) & (y == 1)))
    tn = float(np.sum((y_pred == 0) & (y == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _result(
    name: str,
    value: float,
    y: np.ndarray,
    unit: str,
    *,
    threshold: Optional[float] = None,
    extra: Optional[Dict[str, object]] = None,
) -> MetricResult:
    return MetricResult(
        name=name,
        value=float(value),
        unit=unit,
        n=int(y.size),
        n_positive=int(np.sum(y)),
        defined=True,
        threshold=None if threshold is None else float(threshold),
        extra=dict(extra or {}),
    )


def _undefined(
    name: str,
    reason: str,
    y: np.ndarray,
    unit: str,
    *,
    threshold: Optional[float] = None,
    extra: Optional[Dict[str, object]] = None,
) -> MetricResult:
    return MetricResult(
        name=name,
        value=None,
        unit=unit,
        n=int(y.size),
        n_positive=int(np.sum(y)),
        defined=False,
        reason=reason,
        threshold=None if threshold is None else float(threshold),
        extra=dict(extra or {}),
    )


# -- threshold-free ranking metrics -----------------------------------------

def roc_auc_metric(
    y_true: Sequence[float], y_prob: Sequence[float], *, unit: str = "sample"
) -> MetricResult:
    """ROC-AUC, undefined (never 0.5) when a class is absent."""
    y, p = _validate_inputs(y_true, y_prob, unit, "roc_auc")
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return _undefined(
            "roc_auc",
            f"single_class_in_y_true(pos={n_pos},neg={n_neg})",
            y,
            unit,
        )
    return _result("roc_auc", _roc_auc(y, p), y, unit)


def pr_auc_metric(
    y_true: Sequence[float], y_prob: Sequence[float], *, unit: str = "sample"
) -> MetricResult:
    """PR-AUC (average precision, staircase), undefined if a class is absent."""
    y, p = _validate_inputs(y_true, y_prob, unit, "pr_auc")
    n_pos = int(np.sum(y))
    if n_pos == 0 or n_pos == y.size:
        return _undefined(
            "pr_auc",
            f"single_class_in_y_true(pos={n_pos},n={y.size})",
            y,
            unit,
        )
    return _result(
        "pr_auc",
        _pr_auc(y, p),
        y,
        unit,
        extra={"prevalence": round(n_pos / y.size, 4)},
    )


# -- thresholded metrics -----------------------------------------------------

def confusion_matrix_counts(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    *,
    unit: str = "sample",
) -> Dict[str, object]:
    """Confusion matrix at ``threshold`` with orientation spelled out."""
    y, p = _validate_inputs(y_true, y_prob, unit, "confusion_matrix")
    cm = _counts(y, (p >= threshold).astype(np.int64))
    return {
        "name": "confusion_matrix",
        "unit": unit,
        "n": int(y.size),
        "n_positive": int(np.sum(y)),
        "threshold": float(threshold),
        "tp": int(cm["tp"]),
        "fp": int(cm["fp"]),
        "fn": int(cm["fn"]),
        "tn": int(cm["tn"]),
        "positive_class": POSITIVE_CLASS,
        "orientation": POSITIVE_CLASS_LABEL,
    }


def accuracy_at(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    *,
    unit: str = "sample",
) -> MetricResult:
    y, p = _validate_inputs(y_true, y_prob, unit, "accuracy")
    cm = _counts(y, (p >= threshold).astype(np.int64))
    value = (cm["tp"] + cm["tn"]) / y.size
    prevalence = float(np.sum(y)) / y.size
    extra = {"prevalence": round(prevalence, 4)}
    if prevalence < 0.2 or prevalence > 0.8:
        extra["warning"] = (
            "accuracy is class-imbalance sensitive at this prevalence; report "
            "balanced accuracy and per-class recall instead"
        )
    return _result("accuracy", value, y, unit, threshold=threshold, extra=extra)


def precision_at(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    *,
    unit: str = "sample",
) -> MetricResult:
    y, p = _validate_inputs(y_true, y_prob, unit, "precision")
    cm = _counts(y, (p >= threshold).astype(np.int64))
    if cm["tp"] + cm["fp"] == 0:
        return _undefined(
            "precision",
            "no_positive_predictions",
            y,
            unit,
            threshold=threshold,
        )
    return _result(
        "precision", cm["tp"] / (cm["tp"] + cm["fp"]), y, unit, threshold=threshold
    )


def recall_at(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    *,
    unit: str = "sample",
) -> MetricResult:
    """Recall = sensitivity = TP / (TP + FN) for the positive class."""
    y, p = _validate_inputs(y_true, y_prob, unit, "recall")
    cm = _counts(y, (p >= threshold).astype(np.int64))
    if cm["tp"] + cm["fn"] == 0:
        return _undefined(
            "recall", "no_positive_cases", y, unit, threshold=threshold
        )
    return _result("recall", cm["tp"] / (cm["tp"] + cm["fn"]), y, unit, threshold=threshold)


def specificity_at(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    *,
    unit: str = "sample",
) -> MetricResult:
    y, p = _validate_inputs(y_true, y_prob, unit, "specificity")
    cm = _counts(y, (p >= threshold).astype(np.int64))
    if cm["tn"] + cm["fp"] == 0:
        return _undefined(
            "specificity", "no_negative_cases", y, unit, threshold=threshold
        )
    return _result(
        "specificity", cm["tn"] / (cm["tn"] + cm["fp"]), y, unit, threshold=threshold
    )


def balanced_accuracy_at(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    *,
    unit: str = "sample",
) -> MetricResult:
    """(sensitivity + specificity) / 2; undefined if a class is absent."""
    y, p = _validate_inputs(y_true, y_prob, unit, "balanced_accuracy")
    return balanced_accuracy_from_counts(
        _counts(y, (p >= threshold).astype(np.int64)), y, unit, threshold
    )


def balanced_accuracy_from_counts(
    cm: Dict[str, float],
    y: np.ndarray,
    unit: str,
    threshold: Optional[float],
) -> MetricResult:
    n_pos = cm["tp"] + cm["fn"]
    n_neg = cm["tn"] + cm["fp"]
    if n_pos == 0 or n_neg == 0:
        return _undefined(
            "balanced_accuracy",
            f"single_class_in_y_true(pos={int(n_pos)},neg={int(n_neg)})",
            y,
            unit,
            threshold=threshold,
        )
    sens = cm["tp"] / n_pos
    spec = cm["tn"] / n_neg
    return _result(
        "balanced_accuracy",
        (sens + spec) / 2.0,
        y,
        unit,
        threshold=threshold,
        extra={"sensitivity": round(sens, 6), "specificity": round(spec, 6)},
    )


def macro_f1_at(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    *,
    unit: str = "sample",
) -> MetricResult:
    """Unweighted mean of per-class F1 over classes {0, 1}."""
    y, p = _validate_inputs(y_true, y_prob, unit, "macro_f1")
    cm = _counts(y, (p >= threshold).astype(np.int64))
    if cm["tp"] + cm["fn"] == 0 or cm["tn"] + cm["fp"] == 0:
        return _undefined(
            "macro_f1",
            "single_class_in_y_true",
            y,
            unit,
            threshold=threshold,
        )
    f1_pos = 2 * cm["tp"] / (2 * cm["tp"] + cm["fp"] + cm["fn"])
    f1_neg = 2 * cm["tn"] / (2 * cm["tn"] + cm["fn"] + cm["fp"])
    return _result(
        "macro_f1",
        (f1_pos + f1_neg) / 2.0,
        y,
        unit,
        threshold=threshold,
        extra={"f1_positive": round(f1_pos, 6), "f1_negative": round(f1_neg, 6)},
    )


def mcc_at(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    *,
    unit: str = "sample",
) -> MetricResult:
    """Matthews correlation coefficient; undefined if its denominator is 0."""
    y, p = _validate_inputs(y_true, y_prob, unit, "mcc")
    cm = _counts(y, (p >= threshold).astype(np.int64))
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    denom = float(
        np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    if denom == 0.0:
        return _undefined(
            "mcc",
            "zero_denominator_single_class_or_constant_prediction",
            y,
            unit,
            threshold=threshold,
        )
    return _result(
        "mcc", (tp * tn - fp * fn) / denom, y, unit, threshold=threshold
    )


def sensitivity_at_specificity(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    *,
    min_specificity: float = 0.8,
    unit: str = "sample",
) -> MetricResult:
    """Highest sensitivity over operating points whose specificity >= target.

    Candidate thresholds are the unique predicted probabilities (``p >= t``
    semantics). A reject-all threshold above the maximum score is not offered:
    it would make every target specific always attainable with sensitivity 0.
    When no operating point reaches the target the metric is undefined and the
    best achievable specificity is reported instead.
    """
    y, p = _validate_inputs(y_true, y_prob, unit, "sensitivity_at_specificity")
    if not 0.0 <= min_specificity <= 1.0:
        raise MetricInputError("min_specificity must be in [0, 1]")
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return _undefined(
            "sensitivity_at_specificity",
            f"single_class_in_y_true(pos={n_pos},neg={n_neg})",
            y,
            unit,
        )

    candidates = np.unique(p)[::-1]
    best: Optional[Tuple[float, float, float]] = None  # (sens, spec, threshold)
    best_spec = 0.0
    for t in candidates:
        cm = _counts(y, (p >= t).astype(np.int64))
        sens = cm["tp"] / n_pos
        spec = cm["tn"] / n_neg
        best_spec = max(best_spec, spec)
        if spec >= min_specificity - 1e-12:
            if best is None or sens > best[0] + 1e-12:
                best = (sens, spec, float(t))
    if best is None:
        return _undefined(
            "sensitivity_at_specificity",
            "specificity_target_not_attainable",
            y,
            unit,
            extra={
                "target_specificity": min_specificity,
                "max_achievable_specificity": round(best_spec, 6),
            },
        )
    sens, spec, t = best
    return _result(
        "sensitivity_at_specificity",
        sens,
        y,
        unit,
        threshold=t,
        extra={"target_specificity": min_specificity, "achieved_specificity": round(spec, 6)},
    )


# -- calibration metrics -----------------------------------------------------

def brier_metric(
    y_true: Sequence[float], y_prob: Sequence[float], *, unit: str = "sample"
) -> MetricResult:
    """Brier score = mean((p - y)^2); lower is better."""
    y, p = _validate_inputs(y_true, y_prob, unit, "brier")
    return _result("brier", brier_score(y, p), y, unit)


def ece_equal_frequency(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    *,
    n_bins: int = 10,
    min_bin_n: int = DEFAULT_ECE_MIN_BIN_N,
    unit: str = "sample",
) -> MetricResult:
    """Equal-frequency ECE, suppressed when an occupied bin is too small.

    Equal-width bins (the Phase 2 ``expected_calibration_error``) put almost
    every prediction of a weakly separated model in one or two bins. Here the
    bin edges are quantiles of the predicted probabilities, so each bin carries
    roughly the same number of observations. If any bin holds fewer than
    ``min_bin_n`` observations the ECE is *suppressed* (undefined) rather than
    reported from a bin too small to interpret.
    """
    y, p = _validate_inputs(y_true, y_prob, unit, "ece")
    if n_bins < 2:
        raise MetricInputError("n_bins must be >= 2")
    if min_bin_n < 1:
        raise MetricInputError("min_bin_n must be >= 1")
    bins = int(min(n_bins, y.size))
    if bins < 2:
        return _undefined("ece", "fewer_than_two_observations", y, unit)

    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, bins)
    counts = [int(g.size) for g in groups]
    if min(counts) < min_bin_n:
        return _undefined(
            "ece",
            f"bin_below_min_observations(min_bin_n={min_bin_n})",
            y,
            unit,
            extra={"n_bins": bins, "bin_counts": counts},
        )
    total = float(y.size)
    ece = 0.0
    for g in groups:
        ece += (g.size / total) * abs(float(y[g].mean()) - float(p[g].mean()))
    return _result(
        "ece",
        ece,
        y,
        unit,
        extra={"n_bins": bins, "binning": "equal_frequency", "bin_counts": counts},
    )


def calibration_slope_intercept(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    *,
    unit: str = "sample",
) -> Dict[str, MetricResult]:
    """Logistic recalibration fit: logit(y) ~ intercept + slope * logit(p).

    A perfect calibration has slope 1 and intercept 0. This is a *measurement*
    of calibration, fitted on the evaluated data; it is never used to adjust the
    predictions being measured.
    """
    y, p = _validate_inputs(y_true, y_prob, unit, "calibration_slope")
    n_pos = int(np.sum(y))
    reference = brier_metric(y, p, unit=unit)

    def _both(reason: str) -> Dict[str, MetricResult]:
        return {
            "slope": _undefined("calibration_slope", reason, y, unit),
            "intercept": _undefined("calibration_intercept", reason, y, unit),
            "brier": reference,
        }

    if n_pos == 0 or n_pos == y.size:
        return _both(f"single_class_in_y_true(pos={n_pos})")
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped))
    if np.unique(np.round(logit, 12)).size < 2:
        return _both("constant_predicted_probability")
    if not np.isfinite(logit).all():
        return _both("non_finite_logit")

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=1e6, solver="lbfgs", fit_intercept=True, max_iter=1000, tol=1e-8
    )
    model.fit(logit.reshape(-1, 1), y.astype(np.int64))
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return {
        "slope": _result(
            "calibration_slope",
            slope,
            y,
            unit,
            extra={"perfect_value": 1.0, "fitted_on": "evaluated_data"},
        ),
        "intercept": _result(
            "calibration_intercept",
            intercept,
            y,
            unit,
            extra={"perfect_value": 0.0, "fitted_on": "evaluated_data"},
        ),
        "brier": reference,
    }


# -- aggregate entry point ---------------------------------------------------

DEFAULT_METRIC_NAMES: Tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "sensitivity_at_specificity",
    "brier",
    "ece",
)


def compute_evaluation_metrics(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    *,
    unit: str = "sample",
    threshold: float = 0.5,
    target_specificity: float = 0.8,
    n_bins: int = 10,
    ece_min_bin_n: int = DEFAULT_ECE_MIN_BIN_N,
    include_calibration_fit: bool = True,
) -> Dict[str, object]:
    """Every Phase 5 metric at one threshold, each carrying its own metadata.

    Args:
        y_true: binary labels, 1 = positive (EATD probable depression).
        y_prob: predicted probability of the positive class.
        unit: the evaluation unit these numbers describe (participant, session,
            sample). It is recorded, not inferred.
        threshold: decision threshold for the thresholded metrics.
        target_specificity: operating-point target for
            ``sensitivity_at_specificity``.
        n_bins, ece_min_bin_n: equal-frequency calibration binning controls.
        include_calibration_fit: also report calibration slope/intercept.

    Returns:
        Dict with ``unit``, ``n``, ``n_positive``, ``prevalence``, ``threshold``,
        ``positive_class``, ``metrics`` (name -> MetricResult dict) and
        ``confusion_matrix``. Undefined metrics appear with ``defined: false``
        and a ``reason``; no value is invented for them.
    """
    y, p = _validate_inputs(y_true, y_prob, unit, "compute_evaluation_metrics")
    results: Dict[str, MetricResult] = {
        "roc_auc": roc_auc_metric(y, p, unit=unit),
        "pr_auc": pr_auc_metric(y, p, unit=unit),
        "accuracy": accuracy_at(y, p, threshold, unit=unit),
        "precision": precision_at(y, p, threshold, unit=unit),
        "recall": recall_at(y, p, threshold, unit=unit),
        "specificity": specificity_at(y, p, threshold, unit=unit),
        "balanced_accuracy": balanced_accuracy_at(y, p, threshold, unit=unit),
        "macro_f1": macro_f1_at(y, p, threshold, unit=unit),
        "mcc": mcc_at(y, p, threshold, unit=unit),
        "sensitivity_at_specificity": sensitivity_at_specificity(
            y, p, min_specificity=target_specificity, unit=unit
        ),
        "brier": brier_metric(y, p, unit=unit),
        "ece": ece_equal_frequency(
            y, p, n_bins=n_bins, min_bin_n=ece_min_bin_n, unit=unit
        ),
    }
    if include_calibration_fit:
        results.update(calibration_slope_intercept(y, p, unit=unit))

    n_pos = int(np.sum(y))
    prevalence = n_pos / y.size
    notes: List[str] = [POSITIVE_CLASS_LABEL]
    if prevalence < 0.2 or prevalence > 0.8:
        notes.append(
            f"class imbalance: prevalence={prevalence:.3f}; unweighted accuracy is "
            "misleading, use balanced accuracy and per-class recall"
        )
    if n_pos < MIN_POSITIVES_PER_FOLD_FOR_AGGREGATE:
        notes.append(
            f"few positives (n={n_pos} < {MIN_POSITIVES_PER_FOLD_FOR_AGGREGATE}): "
            "treat this estimate as unstable, do not aggregate without flagging"
        )
    if n_pos < MIN_POSITIVES_FOR_TEST_ESTIMATE:
        notes.append(
            f"n_positive={n_pos} < {MIN_POSITIVES_FOR_TEST_ESTIMATE}: too few "
            "positives for a definitive single-split estimate"
        )
    undefined = sorted(
        name for name, r in results.items() if not r.defined
    )
    if undefined:
        notes.append("undefined metrics (no value reported): " + ", ".join(undefined))

    return {
        "unit": unit,
        "n": int(y.size),
        "n_positive": n_pos,
        "n_negative": int(y.size - n_pos),
        "prevalence": round(prevalence, 6),
        "threshold": float(threshold),
        "positive_class": POSITIVE_CLASS,
        "positive_class_definition": POSITIVE_CLASS_LABEL,
        "metrics": {name: r.to_dict() for name, r in results.items()},
        "confusion_matrix": confusion_matrix_counts(
            y, p, threshold, unit=unit
        ),
        "notes": notes,
    }
