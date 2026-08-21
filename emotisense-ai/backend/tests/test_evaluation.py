"""Tests for Phase 15 — evaluation metrics, ablation, fairness, domain-shift."""

from __future__ import annotations

import json
import numpy as np
import pytest
import torch

from training.evaluation.metrics import (
    DISCLAIMER,
    EvaluationReport,
    assert_not_final_test,
    compute_classification_metrics,
    compute_subgroup_metrics,
    detect_domain_shift,
    measure_latency_ms,
)
from training.evaluation.ablation_runner import (
    analyse_gating_impact,
    compare_fusion_variants,
    run_modality_ablation,
)
from training.multimodal.fusion_baselines import MODALITY_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_data(n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    actual = rng.integers(0, 2, n).tolist()
    # Predicted is mostly correct to get non-trivial metrics
    predicted = [a if rng.random() > 0.25 else 1 - a for a in actual]
    probs = [float(p) + rng.uniform(-0.1, 0.1) for p in predicted]
    probs = [max(0.0, min(1.0, p)) for p in probs]
    return actual, predicted, probs


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

class TestComputeClassificationMetrics:

    def test_all_keys_present(self) -> None:
        a, p, pr = _binary_data()
        result = compute_classification_metrics(a, p, pr)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc",
                    "pr_auc", "brier_score", "sensitivity", "specificity",
                    "confusion_matrix", "n_samples", "disclaimer"):
            assert key in result, f"Missing key: {key}"

    def test_confusion_matrix_sums_to_n(self) -> None:
        a, p, pr = _binary_data()
        result = compute_classification_metrics(a, p, pr)
        cm = result["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(a)

    def test_metrics_in_unit_interval(self) -> None:
        a, p, pr = _binary_data()
        result = compute_classification_metrics(a, p, pr)
        for key in ("accuracy", "f1", "precision", "recall", "sensitivity",
                    "specificity", "roc_auc", "pr_auc", "brier_score"):
            if result[key] is not None:
                assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of range"

    def test_disclaimer_present(self) -> None:
        a, p, pr = _binary_data()
        result = compute_classification_metrics(a, p, pr)
        assert "disclaimer" in result
        assert "research dataset" in result["disclaimer"].lower()

    def test_bootstrap_ci_keys(self) -> None:
        a, p, pr = _binary_data(n=30)
        result = compute_classification_metrics(a, p, pr, bootstrap_ci=True, n_bootstrap=50)
        assert "bootstrap_ci" in result
        ci = result["bootstrap_ci"]
        assert "f1_ci" in ci and "roc_auc_ci" in ci
        # CI lower ≤ CI upper
        assert ci["f1_ci"][0] <= ci["f1_ci"][1]
        assert ci["roc_auc_ci"][0] <= ci["roc_auc_ci"][1]

    def test_result_json_serialisable(self) -> None:
        a, p, pr = _binary_data()
        json.dumps(compute_classification_metrics(a, p, pr))


# ---------------------------------------------------------------------------
# Subgroup / fairness metrics
# ---------------------------------------------------------------------------

class TestComputeSubgroupMetrics:

    def test_per_group_metrics_present(self) -> None:
        a, p, pr = _binary_data(n=40)
        groups = ["A"] * 20 + ["B"] * 20
        result = compute_subgroup_metrics(a, p, pr, groups)
        assert "A" in result["subgroups"]
        assert "B" in result["subgroups"]

    def test_disparity_keys_present(self) -> None:
        a, p, pr = _binary_data(n=40)
        groups = ["A"] * 20 + ["B"] * 20
        result = compute_subgroup_metrics(a, p, pr, groups)
        assert "f1_max_gap" in result["disparity"]
        assert "sensitivity_max_gap" in result["disparity"]

    def test_f1_gap_non_negative(self) -> None:
        a, p, pr = _binary_data(n=40)
        groups = ["X"] * 20 + ["Y"] * 20
        result = compute_subgroup_metrics(a, p, pr, groups)
        gap = result["disparity"]["f1_max_gap"]
        if gap is not None:
            assert gap >= 0.0

    def test_too_small_group_skipped(self) -> None:
        a, p, pr = _binary_data(n=20)
        groups = ["A"] * 19 + ["B"]  # B has only 1 sample
        result = compute_subgroup_metrics(a, p, pr, groups)
        assert result["subgroups"]["B"].get("skipped") == "too_few_samples"


# ---------------------------------------------------------------------------
# Domain-shift detection
# ---------------------------------------------------------------------------

class TestDetectDomainShift:

    def test_identical_distributions_low_score(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 16))
        result = detect_domain_shift(X, X.copy(), method="mmd")
        assert result["shift_score"] < 0.05

    def test_different_distributions_higher_score(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((100, 16))
        Y = rng.standard_normal((100, 16)) + 5.0  # Large shift
        result = detect_domain_shift(X, Y, method="mmd")
        assert result["shift_score"] > 0.0

    def test_kl_div_method(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 8))
        Y = rng.standard_normal((50, 8))
        result = detect_domain_shift(X, Y, method="kl_div")
        assert "shift_score" in result
        assert result["shift_score"] >= 0.0

    def test_unknown_method_raises(self) -> None:
        X = np.ones((10, 4))
        with pytest.raises(ValueError, match="Unknown method"):
            detect_domain_shift(X, X, method="bad")

    def test_result_json_serialisable(self) -> None:
        X = np.random.randn(20, 4)
        json.dumps(detect_domain_shift(X, X))


# ---------------------------------------------------------------------------
# Test-set guard
# ---------------------------------------------------------------------------

class TestAssertNotFinalTest:

    def test_allows_validation_unlocked(self) -> None:
        assert_not_final_test("validation", locked=False)

    def test_allows_test_when_locked(self) -> None:
        assert_not_final_test("test", locked=True)

    def test_blocks_test_when_unlocked(self) -> None:
        with pytest.raises(RuntimeError, match="locked"):
            assert_not_final_test("test", locked=False)


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------

class TestEvaluationReport:

    def test_to_dict_has_disclaimer(self) -> None:
        a, p, pr = _binary_data()
        report = EvaluationReport(
            split="validation",
            metrics=compute_classification_metrics(a, p, pr),
        )
        d = report.to_dict()
        assert "disclaimer" in d

    def test_save_creates_file(self, tmp_path) -> None:
        a, p, pr = _binary_data()
        report = EvaluationReport(
            split="validation",
            metrics=compute_classification_metrics(a, p, pr),
        )
        out = tmp_path / "reports" / "eval.json"
        report.save(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["split"] == "validation"


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

class TestRunModalityAblation:

    def _make_samples(self, n: int = 20, D: int = 32):
        rng = np.random.default_rng(42)
        samples = []
        for i in range(n):
            label = "depressed" if i % 2 == 0 else "control"
            samples.append({
                "label": label,
                "embeddings": {m: torch.randn(1, D) for m in MODALITY_ORDER},
                "masks": {m: torch.ones(1) for m in MODALITY_ORDER},
                "quality": {m: torch.rand(1) for m in MODALITY_ORDER},
            })
        return samples

    def test_all_subset_keys_present(self) -> None:
        from training.multimodal.attention_fusion import (
            AttentionFusionConfig, ReliabilityAwareAttentionFusion,
        )
        cfg = AttentionFusionConfig(
            text_dim=32, audio_dim=32, video_dim=32,
            projected_dim=32, n_classes=2, n_attention_heads=4,
            dropout=0.0, gate_hidden_dim=16,
        )
        model = ReliabilityAwareAttentionFusion(cfg)
        samples = self._make_samples()
        result = run_modality_ablation(model, samples)
        assert "all" in result["subset_results"]
        for m in MODALITY_ORDER:
            assert f"only_{m}" in result["subset_results"]

    def test_metrics_in_unit_interval(self) -> None:
        from training.multimodal.attention_fusion import (
            AttentionFusionConfig, ReliabilityAwareAttentionFusion,
        )
        cfg = AttentionFusionConfig(
            text_dim=32, audio_dim=32, video_dim=32,
            projected_dim=32, n_classes=2, n_attention_heads=4,
            dropout=0.0, gate_hidden_dim=16,
        )
        model = ReliabilityAwareAttentionFusion(cfg)
        samples = self._make_samples()
        result = run_modality_ablation(model, samples)
        for subset, metrics in result["subset_results"].items():
            if "f1" in metrics:
                assert 0.0 <= metrics["f1"] <= 1.0


# ---------------------------------------------------------------------------
# Fusion variant comparison
# ---------------------------------------------------------------------------

class TestCompareFusionVariants:

    def test_ranking_correct(self) -> None:
        variants = {
            "concat":       {"f1": 0.72},
            "mean_pool":    {"f1": 0.68},
            "attention":    {"f1": 0.79},
        }
        result = compare_fusion_variants(variants)
        assert result["best_variant"] == "attention"
        assert result["ranking_by_f1"][0][0] == "attention"

    def test_delta_vs_best_non_positive(self) -> None:
        variants = {"A": {"f1": 0.80}, "B": {"f1": 0.70}}
        result = compare_fusion_variants(variants)
        for name, delta in result["f1_delta_vs_best"].items():
            assert delta <= 0.0


# ---------------------------------------------------------------------------
# Gating impact
# ---------------------------------------------------------------------------

class TestAnalyseGatingImpact:

    def test_strata_keys_present(self) -> None:
        rng = np.random.default_rng(5)
        actual = rng.integers(0, 2, 40).tolist()
        predicted = actual[:]
        quality = rng.uniform(0, 1, 40).tolist()
        result = analyse_gating_impact(quality, actual, predicted)
        assert "quality_quartiles" in result
        assert "Q1 (low)" in result["quality_quartiles"]
        assert "Q4 (high)" in result["quality_quartiles"]

    def test_perfect_predictor_high_correlation(self) -> None:
        actual = [0] * 20 + [1] * 20
        predicted = actual[:]
        quality = [0.1] * 20 + [0.9] * 20  # high quality → correct label
        result = analyse_gating_impact(quality, actual, predicted)
        corr = result["quality_accuracy_correlation"]
        if corr is not None:
            assert corr >= 0.0  # positive when quality predicts correctness
