"""
Calibration Service — Probability calibration and calibration evaluation.

Implements:
  - Evaluation: reliability curve (ECE bins), Brier score, Expected
    Calibration Error (ECE) — computed by ``evaluate_calibration``.
  - Methods: Platt scaling (logistic on logit(p)) and isotonic regression
    (PAVA). A method is ONLY applied at inference when it was fitted on
    VALIDATION data and improved Brier/ECE over the uncalibrated baseline —
    the identity transform is used otherwise.
  - Calibration NEVER uses the final test set: fitting functions take only
    validation predictions/labels supplied by the caller.

No sklearn dependency required: Platt scaling uses Newton iterations on
logistic regression (1-D), isotonic regression uses PAVA — both in numpy.

Exports:
    brier_score, expected_calibration_error, reliability_curve
    fit_platt_scaling, fit_isotonic_regression   (validation data only)
    CalibrationModel — fitted calibrator applied at inference
    load_calibration_model — load fitted parameters from disk (if any)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("emotiondep.calibration")

_CALIBRATION_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "calibration.json"


# ---------------------------------------------------------------------------
# Calibration evaluation (never fit on the test set — evaluation only here)
# ---------------------------------------------------------------------------

def brier_score(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    """Brier score = mean((p - y)^2). Lower is better; 0.25 = no skill."""
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
    """ECE = sum over bins of (n_b / N) * |accuracy_b - confidence_b|."""
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
        bin_conf = float(p[mask].mean())
        bin_acc = float(y[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_curve(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Reliability diagram data: per-bin mean predicted prob vs observed frequency."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    curve: List[Dict[str, float]] = []
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


def evaluate_calibration(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Full calibration evaluation on a validation/evaluation split."""
    return {
        "brier_score": round(brier_score(y_true, y_prob), 4),
        "ece": round(expected_calibration_error(y_true, y_prob, n_bins), 4),
        "reliability_curve": reliability_curve(y_true, y_prob, n_bins),
        "n_samples": len(y_true),
    }


# ---------------------------------------------------------------------------
# Platt scaling (1-D logistic regression on the logit of p)
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_platt_scaling(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Tuple[float, float]:
    """Fit Platt scaling a=p^-, b: sigmoid(a * logit(p) + b) via Newton's method.

    MUST be called with validation data only.
    Returns (a, b).
    """
    y = np.asarray(y_true, dtype=np.float64)
    x = _logit(np.asarray(y_prob, dtype=np.float64))
    n = len(y)
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        z = a * x + b
        s = 1.0 / (1.0 + np.exp(-z))
        # Weighted logistic loss gradient / Hessian (Platt's target smoothing)
        t = np.where(y == 1, 1.0, 0.0)
        t = (n * t + 1.0) / (n + 2.0)  # Platt smoothing
        grad_a = np.sum((s - t) * x)
        grad_b = np.sum(s - t)
        h_aa = np.sum(s * (1 - s) * x * x)
        h_ab = np.sum(s * (1 - s) * x)
        h_bb = np.sum(s * (1 - s))
        hess = np.array([[h_aa, h_ab], [h_ab, h_bb]])
        if np.linalg.det(hess) < 1e-12:
            break
        step = np.linalg.solve(hess, np.array([grad_a, grad_b]))
        a_new, b_new = a - step[0], b - step[1]
        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return float(a), float(b)


# ---------------------------------------------------------------------------
# Isotonic regression (PAVA)
# ---------------------------------------------------------------------------

def fit_isotonic_regression(
    y_true: Sequence[float],
    y_prob: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit isotonic regression (monotone step function) via PAVA.

    MUST be called with validation data only.
    Returns (thresholds, values): the fitted step function.
    Apply with np.interp(p, thresholds, values).
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]

    # PAVA blocks: (start_weighted_value, weight, threshold)
    values: List[float] = []
    weights: List[float] = []
    thresholds: List[float] = []
    for pv, yv in zip(p_sorted, y_sorted):
        values.append(float(yv))
        weights.append(1.0)
        thresholds.append(float(pv))
        # Merge while the monotonicity constraint is violated
        while len(values) > 1 and values[-2] > values[-1]:
            total_w = weights[-2] + weights[-1]
            merged_v = (values[-2] * weights[-2] + values[-1] * weights[-1]) / total_w
            values[-2:] = [merged_v]
            weights[-2:] = [total_w]
            thresholds[-2:] = [thresholds[-2]]

    return np.asarray(thresholds), np.asarray(values)


# ---------------------------------------------------------------------------
# CalibrationModel — applied at inference
# ---------------------------------------------------------------------------

class CalibrationModel:
    """A fitted calibrator. Applies Platt scaling or isotonic regression.

    ``method == "identity"`` means no validated improvement was found on the
    validation split, so raw probabilities pass through unchanged.
    """

    def __init__(
        self,
        method: str = "identity",
        platt_a: Optional[float] = None,
        platt_b: Optional[float] = None,
        iso_thresholds: Optional[List[float]] = None,
        iso_values: Optional[List[float]] = None,
        validation_metrics: Optional[Dict[str, Any]] = None,
    ):
        self.method = method
        self.platt_a = platt_a
        self.platt_b = platt_b
        self.iso_thresholds = iso_thresholds or []
        self.iso_values = iso_values or []
        self.validation_metrics = validation_metrics or {}

    def transform(self, p: float) -> float:
        """Calibrate a single probability."""
        p = float(min(max(p, 0.01), 0.99))
        if self.method == "platt" and self.platt_a is not None and self.platt_b is not None:
            z = self.platt_a * float(_logit(np.array([p]))[0]) + self.platt_b
            return float(min(max(1.0 / (1.0 + np.exp(-z)), 0.01), 0.99))
        if self.method == "isotonic" and self.iso_thresholds:
            return float(min(max(np.interp(p, self.iso_thresholds, self.iso_values), 0.01), 0.99))
        return p  # identity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "iso_thresholds": [round(float(t), 6) for t in self.iso_thresholds],
            "iso_values": [round(float(v), 6) for v in self.iso_values],
            "validation_metrics": self.validation_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationModel":
        return cls(
            method=data.get("method", "identity"),
            platt_a=data.get("platt_a"),
            platt_b=data.get("platt_b"),
            iso_thresholds=data.get("iso_thresholds"),
            iso_values=data.get("iso_values"),
            validation_metrics=data.get("validation_metrics"),
        )


def select_best_calibration(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> Tuple[CalibrationModel, Dict[str, Any]]:
    """Fit both candidate methods on VALIDATION data and select the best.

    A method is only adopted if it improves BOTH Brier score and ECE over the
    uncalibrated baseline; otherwise identity is kept. Selection never touches
    the test set.
    """
    baseline = evaluate_calibration(y_true, y_prob, n_bins)
    candidates: Dict[str, CalibrationModel] = {"identity": CalibrationModel("identity")}

    try:
        a, b = fit_platt_scaling(y_true, y_prob)
        candidates["platt"] = CalibrationModel("platt", platt_a=a, platt_b=b)
    except Exception as e:
        logger.warning("Platt scaling failed: %s", e)

    try:
        th, vals = fit_isotonic_regression(y_true, y_prob)
        candidates["isotonic"] = CalibrationModel(
            "isotonic",
            iso_thresholds=[float(t) for t in th],
            iso_values=[float(v) for v in vals],
        )
    except Exception as e:
        logger.warning("Isotonic regression failed: %s", e)

    results: Dict[str, Dict[str, Any]] = {"identity": baseline}
    best_method = "identity"
    best_brier = baseline["brier_score"]
    for name, model in candidates.items():
        if name == "identity":
            continue
        p_cal = [model.transform(float(p)) for p in y_prob]
        metrics = evaluate_calibration(y_true, p_cal, n_bins)
        results[name] = metrics
        # Adopt only if BOTH Brier and ECE improve over identity
        if (
            metrics["brier_score"] < results["identity"]["brier_score"] - 1e-4
            and metrics["ece"] < results["identity"]["ece"] - 1e-4
            and metrics["brier_score"] < best_brier
        ):
            best_method = name
            best_brier = metrics["brier_score"]

    chosen = candidates[best_method]
    chosen.validation_metrics = {
        "validation": results,
        "selected": best_method,
        "note": "Fitted on validation split only; identity kept when no validated improvement.",
    }
    return chosen, chosen.validation_metrics


def save_calibration_model(model: CalibrationModel, path: Optional[Path] = None) -> Path:
    target = Path(path) if path else _CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    logger.info("Calibration model saved: %s (method=%s)", target, model.method)
    return target


def load_calibration_model(path: Optional[Path] = None) -> Optional[CalibrationModel]:
    """Load a previously fitted calibrator, or None if none exists/is valid."""
    target = Path(path) if path else _CALIBRATION_PATH
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return CalibrationModel.from_dict(data)
    except Exception as e:
        logger.warning("Could not load calibration model from %s: %s", target, e)
        return None
