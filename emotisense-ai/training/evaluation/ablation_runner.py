"""Phase 15 — Modality and architecture ablation runner.

Runs a trained fusion model under every modality subset and records
per-subset metrics.  Also supports fusion-variant comparison
(concat / mean_pool / weighted_sum / attention) on the same split.

Results are written to a JSON report with:
  - per-subset classification metrics
  - reliability-gating impact (quality scores vs. accuracy)
  - fusion-variant ranking by validation F1
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from training.evaluation.metrics import DISCLAIMER, compute_classification_metrics
from training.multimodal.fusion_baselines import MODALITY_ORDER


# ---------------------------------------------------------------------------
# Modality-subset ablation
# ---------------------------------------------------------------------------

def run_modality_ablation(
    predict_fn,
    samples: list[dict],
    label_key: str = "label",
    label_map: Optional[dict[str, int]] = None,
) -> dict:
    """Evaluate predict_fn under every modality subset.

    Parameters
    ----------
    predict_fn : callable(embeddings, masks, quality) → dict with 'risk_score'
    samples    : list of dicts; each must contain:
                   embeddings: {modality: Tensor (1, D)}
                   masks:      {modality: Tensor (1,)}
                   quality:    {modality: Tensor (1,)}
                   label_key:  str label
    label_map  : {label_str: int}; auto-detected if None

    Returns
    -------
    dict mapping subset_name → metric dict
    """
    if not samples:
        return {}

    # Build label map
    all_labels = [s[label_key] for s in samples]
    if label_map is None:
        unique = sorted(set(all_labels))
        if len(unique) != 2:
            raise ValueError(f"Expected binary labels, got: {unique}")
        label_map = {l: i for i, l in enumerate(unique)}

    # Determine which modalities are present in the data
    present_modalities = [
        m for m in MODALITY_ORDER
        if any(s["masks"].get(m, torch.zeros(1)).item() > 0 for s in samples)
    ]

    # Build subset mask configurations
    subsets: dict[str, dict[str, bool]] = {"all": {m: True for m in present_modalities}}
    for m in present_modalities:
        subsets[f"only_{m}"] = {mod: (mod == m) for mod in present_modalities}
    for i, m1 in enumerate(present_modalities):
        for m2 in present_modalities[i + 1:]:
            subsets[f"{m1}+{m2}"] = {mod: (mod in (m1, m2)) for mod in present_modalities}

    results: dict[str, dict] = {}
    for subset_name, subset_config in subsets.items():
        actual, predicted, probs = [], [], []
        for sample in samples:
            # Override masks based on subset config
            subset_masks = {
                m: sample["masks"][m] if subset_config.get(m, False)
                else torch.zeros_like(sample["masks"].get(m, torch.zeros(1)))
                for m in MODALITY_ORDER
            }
            out = predict_fn(sample["embeddings"], subset_masks, sample.get("quality"))
            risk = float(out["risk_score"].item())
            pred_class = int(risk >= 0.5)
            true_class = label_map[sample[label_key]]
            actual.append(true_class)
            predicted.append(pred_class)
            probs.append(risk)

        metrics = compute_classification_metrics(actual, predicted, probs)
        metrics["subset"] = subset_name
        metrics["n_modalities"] = sum(1 for v in subset_config.values() if v)
        results[subset_name] = metrics

    # Rank subsets by F1
    ranked = sorted(
        [(k, v["f1"]) for k, v in results.items() if "f1" in v],
        key=lambda x: x[1], reverse=True,
    )
    return {
        "subset_results": results,
        "ranking_by_f1": ranked,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Fusion variant comparison
# ---------------------------------------------------------------------------

def compare_fusion_variants(
    variant_results: dict[str, dict],
) -> dict:
    """Rank fusion variants by validation F1 and produce a comparison table.

    Parameters
    ----------
    variant_results : {variant_name: metrics_dict from compute_classification_metrics}

    Returns
    -------
    dict with ranking_by_f1 list and delta table vs best
    """
    ranked = sorted(
        [(k, v.get("f1", 0.0)) for k, v in variant_results.items()],
        key=lambda x: x[1], reverse=True,
    )
    best_name, best_f1 = ranked[0] if ranked else ("none", 0.0)

    deltas = {
        name: round(f1 - best_f1, 4)
        for name, f1 in ranked
    }

    return {
        "ranking_by_f1": ranked,
        "best_variant": best_name,
        "best_f1": best_f1,
        "f1_delta_vs_best": deltas,
        "variant_metrics": variant_results,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Reliability-gating impact analysis
# ---------------------------------------------------------------------------

def analyse_gating_impact(
    quality_scores: list[float],
    actual: list[int],
    predicted: list[int],
) -> dict:
    """Stratify accuracy by quality score quartile.

    Shows whether high-quality inputs lead to better predictions
    — which validates that the reliability gate is discriminative.
    """
    q_arr = np.array(quality_scores)
    quartile_boundaries = np.quantile(q_arr, [0.0, 0.25, 0.50, 0.75, 1.0])
    strata: dict[str, dict] = {}

    labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    for i, label in enumerate(labels):
        lo, hi = quartile_boundaries[i], quartile_boundaries[i + 1]
        idx = np.where((q_arr >= lo) & (q_arr <= hi))[0]
        if len(idx) == 0:
            strata[label] = {"n": 0}
            continue
        a = [actual[j] for j in idx]
        p = [predicted[j] for j in idx]
        pr = [float(predicted[j]) for j in idx]  # proxy; real probs needed for AUC
        from sklearn.metrics import accuracy_score, f1_score
        strata[label] = {
            "n": len(idx),
            "mean_quality": round(float(q_arr[idx].mean()), 4),
            "accuracy": round(float(accuracy_score(a, p)), 4),
            "f1": round(float(f1_score(a, p, zero_division=0)), 4),
            "quality_range": [round(float(lo), 4), round(float(hi), 4)],
        }

    correctness = np.array([int(a == p) for a, p in zip(actual, predicted)], dtype=float)
    if q_arr.std() < 1e-9 or correctness.std() < 1e-9:
        quality_accuracy_correlation = None
    else:
        quality_accuracy_correlation = float(np.corrcoef(q_arr, correctness)[0, 1])

    return {
        "quality_quartiles": strata,
        "quality_accuracy_correlation": (
            round(quality_accuracy_correlation, 4)
            if quality_accuracy_correlation is not None else None
        ),
        "interpretation": (
            "Positive correlation confirms the reliability gate helps. "
            "Near-zero correlation suggests quality scores are not discriminative."
        ),
    }


# ---------------------------------------------------------------------------
# CLI-friendly report writer
# ---------------------------------------------------------------------------

def write_ablation_report(
    ablation_result: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ablation_result, indent=2), encoding="utf-8")
