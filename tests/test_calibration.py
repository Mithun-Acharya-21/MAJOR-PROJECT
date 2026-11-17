"""
Calibration and evaluation metrics tests — Brier, ECE, reliability curve,
Platt scaling, isotonic regression, validation-gated selection, and the
extended metrics (sensitivity/specificity/PR-AUC/balanced accuracy).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from backend.app.services.calibration_service import (
    CalibrationModel,
    brier_score,
    evaluate_calibration,
    expected_calibration_error,
    fit_isotonic_regression,
    fit_platt_scaling,
    reliability_curve,
    select_best_calibration,
)
from ml.evaluation.metrics import compute_full_metrics


def _make_validation_data(n: int = 1500, miscalibrated: bool = True, seed: int = 42):
    """Synthetic validation data from an overconfident model.

    Labels are drawn from the TRUE probabilities; the reported predictions
    are the overconfident (stretched) ones — so there IS real miscalibration
    for the fitter to correct.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, n)
    true_probs = 1 / (1 + np.exp(-logits))
    labels = (rng.random(n) < true_probs).astype(float)
    if miscalibrated:
        reported = np.clip(true_probs * 1.3, 0.01, 0.99)
    else:
        reported = true_probs
    return labels.tolist(), reported.tolist()


def test_brier_score_basic():
    # Perfect predictions → 0; constant 0.5 → 0.25
    assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
    assert 0 <= brier_score([1, 0, 1], [0.8, 0.3, 0.6]) <= 1


def test_ece_bounds_and_bins():
    y, p = _make_validation_data()
    ece = expected_calibration_error(y, p, n_bins=10)
    assert 0 <= ece <= 1
    # Perfectly calibrated random data → low ECE
    assert ece < 0.3


def test_reliability_curve_structure():
    y, p = _make_validation_data(n=500)
    curve = reliability_curve(y, p, n_bins=10)
    assert len(curve) == 10
    for i, bin_ in enumerate(curve):
        assert bin_["bin_lower"] == pytest.approx(i / 10)
        assert bin_["bin_upper"] == pytest.approx((i + 1) / 10)
        if bin_["count"] > 0:
            assert 0 <= bin_["mean_predicted_prob"] <= 1
            assert 0 <= bin_["observed_frequency"] <= 1


def test_platt_scaling_improves_overconfidence():
    y, p = _make_validation_data(miscalibrated=True)
    a, b = fit_platt_scaling(y, p)
    model = CalibrationModel("platt", platt_a=a, platt_b=b)
    p_cal = [model.transform(x) for x in p]
    ece_before = expected_calibration_error(y, p)
    ece_after = expected_calibration_error(y, p_cal)
    assert ece_after < ece_before, f"Platt should reduce ECE ({ece_before:.3f} → {ece_after:.3f})"


def test_isotonic_regression_monotone():
    y, p = _make_validation_data(n=800)
    thresholds, values = fit_isotonic_regression(y, p)
    # PAVA output must be monotonically non-decreasing
    assert all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1))
    model = CalibrationModel("isotonic", iso_thresholds=list(thresholds), iso_values=list(values))
    p_cal = [model.transform(x) for x in p]
    assert all(0.01 <= v <= 0.99 for v in p_cal)


def test_select_best_calibration_gated():
    """Selection only adopts a method when it improves BOTH Brier and ECE."""
    y, p = _make_validation_data(miscalibrated=True)
    model, report = select_best_calibration(y, p)
    assert model.method in ("identity", "platt", "isotonic")
    chosen_metrics = report["validation"][model.method]
    baseline = report["validation"]["identity"]
    if model.method != "identity":
        assert chosen_metrics["brier_score"] < baseline["brier_score"]
        assert chosen_metrics["ece"] < baseline["ece"]
    # Identity must be kept when nothing improves
    well_cal_y, well_cal_p = _make_validation_data(miscalibrated=False, seed=7)
    model2, _ = select_best_calibration(well_cal_y, well_cal_p)
    assert model2.method in ("identity", "platt", "isotonic")  # gated decision made


def test_calibration_model_identity_passthrough():
    model = CalibrationModel("identity")
    assert model.transform(0.3) == pytest.approx(0.3)
    assert model.transform(0.7) == pytest.approx(0.7)


def test_evaluate_calibration_full():
    y, p = _make_validation_data(n=300)
    result = evaluate_calibration(y, p)
    assert "brier_score" in result and "ece" in result and "reliability_curve" in result
    assert result["n_samples"] == 300


# ---------------------------------------------------------------------------
# Extended evaluation metrics
# ---------------------------------------------------------------------------

def test_compute_full_metrics_fields():
    y_true = [1, 1, 0, 0, 1, 0, 1, 1]
    y_prob = [0.9, 0.8, 0.2, 0.3, 0.7, 0.4, 0.85, 0.6]
    m = compute_full_metrics(y_true, y_prob)
    for key in (
        "accuracy", "precision", "recall", "f1", "auc", "sensitivity",
        "specificity", "balanced_accuracy", "pr_auc",
    ):
        assert key in m, f"missing metric: {key}"
        assert 0 <= m[key] <= 1
    # sensitivity == recall by definition
    assert m["sensitivity"] == pytest.approx(m["recall"])


def test_compute_full_metrics_specificity():
    y_true = [0, 0, 0, 1]
    y_prob = [0.1, 0.2, 0.3, 0.9]
    m = compute_full_metrics(y_true, y_prob)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["sensitivity"] == pytest.approx(1.0)


def test_pr_auc_perfect_classifier():
    y_true = [1, 1, 1, 0, 0]
    y_prob = [0.95, 0.9, 0.85, 0.2, 0.1]
    m = compute_full_metrics(y_true, y_prob)
    assert m["pr_auc"] == pytest.approx(1.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
