"""Phase 5B — guarded metrics (Task E) and bootstrap intervals (Task F).

Every fixture here is synthetic. No model performance is measured or claimed; the
tests check that the harness computes, labels and refuses correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation import bootstrap as B  # noqa: E402
from ml.evaluation import metrics as M  # noqa: E402


PERFECT_Y = [1, 1, 0, 0]
PERFECT_P = [0.9, 0.7, 0.2, 0.1]


# ---------------------------------------------------------------------------
# Task E — metrics
# ---------------------------------------------------------------------------

def test_perfect_separation_gives_expected_values():
    out = M.compute_evaluation_metrics(PERFECT_Y, PERFECT_P, unit="participant")
    for name in ("roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "mcc"):
        assert out["metrics"][name]["defined"] is True, name
        assert out["metrics"][name]["value"] == pytest.approx(1.0), name
    assert out["confusion_matrix"]["tp"] == 2
    assert out["confusion_matrix"]["tn"] == 2
    assert out["confusion_matrix"]["fp"] == 0
    assert out["confusion_matrix"]["fn"] == 0


def test_thresholded_metrics_use_the_supplied_threshold():
    y = [1, 1, 0, 0]
    p = [0.9, 0.7, 0.55, 0.1]
    out_low = M.balanced_accuracy_at(y, p, 0.3)
    out_high = M.balanced_accuracy_at(y, p, 0.65)
    assert out_low.threshold == 0.3
    assert out_high.threshold == 0.65
    # 0.65 splits the classes perfectly here; 0.3 admits one control.
    assert out_low.as_float() == pytest.approx(0.75)
    assert out_high.as_float() == pytest.approx(1.0)


def test_single_class_input_reports_undefined_with_a_reason():
    result = M.roc_auc_metric([0, 0, 0], [0.1, 0.2, 0.3])
    assert result.defined is False
    assert result.value is None
    assert "single_class" in (result.reason or "")
    with pytest.raises(M.MetricUndefinedError):
        result.as_float()

    ba = M.balanced_accuracy_at([1, 1], [0.8, 0.9])
    assert ba.defined is False and "single_class" in (ba.reason or "")


def test_precision_is_undefined_when_nothing_is_predicted_positive():
    result = M.precision_at([1, 0, 1, 0], [0.1, 0.2, 0.3, 0.4], 0.5)
    assert result.defined is False
    assert result.reason == "no_positive_predictions"


def test_mcc_with_zero_denominator_is_undefined_not_zero():
    result = M.mcc_at([1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6], 0.5)
    assert result.defined is False
    assert "denominator" in (result.reason or "")


def test_every_metric_records_unit_sample_count_and_positives():
    y = [1, 0, 1, 0, 0, 0]
    p = [0.8, 0.2, 0.7, 0.4, 0.3, 0.1]
    out = M.compute_evaluation_metrics(y, p, unit="session")
    for name, entry in out["metrics"].items():
        assert entry["unit"] == "session", name
        assert entry["n"] == 6, name
        assert entry["n_positive"] == 2, name
    assert out["unit"] == "session"
    assert out["n"] == 6
    assert out["n_positive"] == 2
    assert out["positive_class"] == 1


def test_positive_class_orientation_is_class_one():
    y = [1, 1, 0, 0]
    p = [0.9, 0.8, 0.2, 0.1]
    recall = M.recall_at(y, p, 0.5)
    sensitivity = M.recall_at(y, p, 0.5)
    specificity = M.specificity_at(y, p, 0.5)
    assert recall.as_float() == pytest.approx(1.0)
    assert sensitivity.as_float() == pytest.approx(1.0)
    assert specificity.as_float() == pytest.approx(1.0)

    flipped = M.recall_at([0, 0, 1, 1], p, 0.5)
    assert flipped.as_float() == pytest.approx(0.0), "class 1 is the positive class"


def test_sensitivity_at_specificity_reports_operating_point():
    y = [1, 0, 1, 0]
    p = [0.8, 0.6, 0.4, 0.2]
    result = M.sensitivity_at_specificity(y, p, min_specificity=0.5)
    assert result.defined is True
    assert result.value == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.4)
    assert result.extra["achieved_specificity"] >= 0.5 - 1e-12


def test_sensitivity_at_unattainable_specificity_is_undefined():
    result = M.sensitivity_at_specificity([1, 0], [0.9, 0.9], min_specificity=0.5)
    assert result.defined is False
    assert result.reason == "specificity_target_not_attainable"
    assert result.extra["max_achievable_specificity"] == pytest.approx(0.0)


def test_ece_is_suppressed_when_a_bin_is_too_small():
    result = M.ece_equal_frequency([1, 0] * 4, [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert result.defined is False
    assert "bin_below_min_observations" in (result.reason or "")


def test_ece_equal_frequency_is_defined_with_enough_observations():
    rng = np.random.default_rng(3)
    p = rng.random(400)
    y = (rng.random(400) < p).astype(int).tolist()
    result = M.ece_equal_frequency(y, p.tolist())
    assert result.defined is True
    assert 0.0 <= result.as_float() <= 1.0
    assert result.extra["binning"] == "equal_frequency"


def test_calibration_slope_and_intercept_recover_a_calibrated_model():
    rng = np.random.default_rng(11)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(int).tolist()
    out = M.calibration_slope_intercept(y, p.tolist())
    assert out["slope"].as_float() == pytest.approx(1.0, abs=0.15)
    assert out["intercept"].as_float() == pytest.approx(0.0, abs=0.2)
    assert out["brier"].defined is True


def test_calibration_fit_is_undefined_for_single_class():
    out = M.calibration_slope_intercept([0, 0, 0], [0.1, 0.2, 0.3])
    assert out["slope"].defined is False
    assert "single_class" in (out["slope"].reason or "")


@pytest.mark.parametrize(
    "y, p, message",
    [
        ([0, 1], [0.5], "y_true has 2 values"),
        ([0, 2], [0.5, 0.5], "binary 0/1"),
        ([0, 1], [0.5, 1.4], "probabilities in"),
        ([0, float("nan")], [0.5, 0.5], "non-finite"),
        ([], [], "empty input"),
    ],
)
def test_invalid_metric_inputs_raise(y, p, message):
    with pytest.raises(M.MetricInputError, match=message):
        M.compute_evaluation_metrics(y, p)


def test_unknown_evaluation_unit_is_rejected():
    with pytest.raises(M.MetricInputError, match="unknown evaluation unit"):
        M.roc_auc_metric([0, 1], [0.2, 0.8], unit="fold")


def test_aggregate_metrics_carry_guards_and_notes():
    y = [0] * 20 + [1] * 2
    p = [0.1] * 20 + [0.9, 0.8]
    out = M.compute_evaluation_metrics(y, p)
    notes = " ".join(out["notes"])
    assert "class 1 is the positive class" in notes
    assert "class imbalance" in notes
    assert "few positives" in notes
    assert "too few positives" in notes


def test_phase2_metric_helpers_are_unchanged():
    base = M.compute_metrics(PERFECT_Y, PERFECT_P)
    full = M.compute_full_metrics(PERFECT_Y, PERFECT_P)
    assert set(base) == {"accuracy", "precision", "recall", "f1", "auc"}
    assert full["auc"] == pytest.approx(1.0)
    assert full["pr_auc"] == pytest.approx(1.0)
    assert M.brier_score([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    assert 0.0 <= M.expected_calibration_error([0, 1], [0.2, 0.8]) <= 1.0
    assert len(M.reliability_curve([0, 1], [0.2, 0.8])) == 10


# ---------------------------------------------------------------------------
# Task F — bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _participant_fixture(n_participants: int = 60, sessions: int = 3, seed: int = 5):
    """Deliberately imperfect predictions: overlapping, not separable."""
    rng = np.random.default_rng(seed)
    labels, probs, ids = [], [], []
    for i in range(n_participants):
        label = 1 if i % 3 == 0 else 0
        base = 0.62 if label else 0.38
        for s in range(sessions):
            labels.append(label)
            probs.append(float(np.clip(base + rng.normal(0, 0.25), 0.01, 0.99)))
            ids.append(f"p{i:03d}")
    return labels, probs, ids


def test_default_bootstrap_settings_match_the_statistical_plan():
    config = B.BootstrapConfig()
    assert config.n_resamples == 2000
    assert config.seed == 0
    assert config.alpha == 0.05
    assert "roc_auc" in config.metrics
    assert config.min_valid_fraction == 0.95


def test_participant_cluster_bootstrap_is_deterministic_and_records_its_seed():
    labels, probs, ids = _participant_fixture()
    first = B.participant_cluster_bootstrap(labels, probs, ids, n_resamples=200)
    second = B.participant_cluster_bootstrap(labels, probs, ids, n_resamples=200)
    assert first["config"]["seed"] == 0
    assert first["metrics"] == second["metrics"]

    other = B.participant_cluster_bootstrap(
        labels, probs, ids, n_resamples=200, seed=1
    )
    assert other["metrics"]["roc_auc"]["ci_lower"] != first["metrics"]["roc_auc"]["ci_lower"]


def test_participant_bootstrap_resamples_people_not_sessions():
    labels, probs, ids = _participant_fixture()
    result = B.participant_cluster_bootstrap(labels, probs, ids, n_resamples=200)
    assert result["unit"] == "participant"
    assert result["n_participants"] == 60
    assert result["n_observations"] == 180
    assert result["clustering_respected"] is True
    assert result["resampling"].startswith("participants")
    assert result["sessions_per_participant"] == {"min": 3, "median": 3.0, "max": 3}


def test_bootstrap_point_estimate_matches_direct_computation():
    labels, probs, ids = _participant_fixture()
    result = B.participant_cluster_bootstrap(labels, probs, ids, n_resamples=100)
    aggregated = B.aggregate_by_participant(labels, probs, ids)
    direct = M.roc_auc_metric(aggregated["labels"], aggregated["probs"], unit="participant")
    assert result["metrics"]["roc_auc"]["point"] == pytest.approx(direct.as_float())
    assert result["metrics"]["roc_auc"]["status"] == "ok"
    assert result["metrics"]["roc_auc"]["ci_lower"] <= result["metrics"]["roc_auc"]["ci_upper"]


def test_bootstrap_interval_contains_the_point_estimate():
    labels, probs, ids = _participant_fixture()
    result = B.participant_cluster_bootstrap(labels, probs, ids, n_resamples=500)
    for name, entry in result["metrics"].items():
        assert entry["status"] == "ok", name
        assert entry["point_inside_interval"] is True, name


def test_bootstrap_rejects_single_class_input():
    with pytest.raises(B.BootstrapError, match="both classes"):
        B.participant_cluster_bootstrap([0, 0, 0], [0.1, 0.2, 0.3], ["a", "b", "c"])


def test_bootstrap_withholds_an_interval_when_the_metric_is_never_defined():
    labels, probs, ids = _participant_fixture()
    result = B.participant_cluster_bootstrap(
        labels,
        [0.1] * len(probs),
        ids,
        n_resamples=50,
        metrics=("precision",),
        threshold=0.5,
    )
    entry = result["metrics"]["precision"]
    assert entry["ci_lower"] is None and entry["ci_upper"] is None
    assert entry["status"] == "no_valid_resamples"
    assert result["n_valid_resamples"] == 0
    assert result["n_invalid_resamples"] == 50
    assert result["invalid_reasons"], "invalid resamples must be counted with a reason"


def test_bootstrap_requires_the_resampling_unit_the_dataset_supports():
    labels, probs, ids = _participant_fixture(n_participants=15)
    with pytest.raises(B.BootstrapError, match="requires participant_ids"):
        B.bootstrap_for_dataset("eatd", labels, probs, n_resamples=20)
    with pytest.raises(B.BootstrapError, match="no released participant identity"):
        B.bootstrap_for_dataset(
            "avdd", labels, probs, participant_ids=ids, n_resamples=20
        )
    ok = B.bootstrap_for_dataset(
        "avdd", labels, probs, n_resamples=20, metrics=("roc_auc",)
    )
    assert ok["method"] == "sample_level_bootstrap"


def test_sample_bootstrap_declares_that_clustering_is_not_respected():
    labels, probs, _ = _participant_fixture(n_participants=20)
    result = B.sample_bootstrap(labels, probs, n_resamples=100)
    assert result["unit"] == "sample"
    assert result["clustering_respected"] is False
    assert result["dataset"] == "avdd"
    assert any("participant clustering" in note for note in result["notes"])


def test_aggregate_by_participant_averages_sessions_and_rejects_mixed_labels():
    aggregated = B.aggregate_by_participant([1, 1, 1], [0.1, 0.5, 0.9], ["p1"] * 3)
    assert aggregated["n_units"] == 1
    assert aggregated["probs"][0] == pytest.approx(0.5)
    assert aggregated["labels"][0] == 1
    assert aggregated["session_counts"] == (3,)

    with pytest.raises(B.BootstrapError, match="mixed labels"):
        B.aggregate_by_participant([1, 0], [0.5, 0.5], ["p1", "p1"])


def test_bootstrap_rejects_unknown_metrics_and_bad_resample_counts():
    with pytest.raises(B.BootstrapError, match="unsupported bootstrap metric"):
        B.BootstrapConfig(metrics=("auc",))
    with pytest.raises(B.BootstrapError, match="n_resamples"):
        B.BootstrapConfig(n_resamples=0)
