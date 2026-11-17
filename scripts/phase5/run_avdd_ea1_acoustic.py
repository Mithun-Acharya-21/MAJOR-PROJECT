"""Phase 5F — E-A1 acoustic baseline on AVDD-TCMI (HDS and SDS separately)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (  # noqa: F401
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_NPZ = PROJECT_ROOT / "data" / "processed" / "avdd_tcmi_phase5" / "avdd_acoustic_cache.npz"
OUT_BASE = PROJECT_ROOT / "results" / "phase5" / "avdd_features"

SEED = 42
N_SPLITS = 5
N_REPEATS = 5
THRESHOLD = 0.5
N_BOOT = 1000


def load_subset(subset: str):
    data = np.load(CACHE_NPZ, allow_pickle=False)
    mask = data["subset"] == subset
    X = data["X"][mask].astype(np.float64)
    y = data["labels"][mask].astype(np.int64)
    ids = data["sample_ids"][mask]
    return X, y, ids, list(data["feature_names"])


def mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def full_metrics(y, p, threshold=THRESHOLD):
    from sklearn.metrics import (  # noqa
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        average_precision_score,
        brier_score_loss,
    )

    pred = (np.asarray(p) >= threshold).astype(int)
    y = np.asarray(y).astype(int)
    try:
        roc = float(roc_auc_score(y, p))
    except ValueError:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y, p))
    except ValueError:
        pr = float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(recall_score(1 - y, 1 - pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": roc,
        "pr_auc": pr,
        "mcc": mcc(y, pred),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_positive": int(y.sum()),
    }


def make_model(name):
    if name == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=50,
                        learning_rate=0.1,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    raise ValueError(name)


MODEL_NAMES = ["logistic_regression", "random_forest", "hist_gradient_boosting"]


def bootstrap_ci(y, p, metric, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = np.asarray(y)[idx]
        pp = np.asarray(p)[idx]
        if len(np.unique(yy)) < 2:
            continue
        if metric == "roc_auc":
            from sklearn.metrics import roc_auc_score

            try:
                vals.append(float(roc_auc_score(yy, pp)))
            except ValueError:
                pass
        elif metric == "pr_auc":
            from sklearn.metrics import average_precision_score

            try:
                vals.append(float(average_precision_score(yy, pp)))
            except ValueError:
                pass
        elif metric == "balanced_accuracy":
            from sklearn.metrics import balanced_accuracy_score

            pred = (pp >= THRESHOLD).astype(int)
            vals.append(float(balanced_accuracy_score(yy, pred)))
    if not vals:
        return {"ci_lower": None, "ci_upper": None, "n_boot": 0}
    return {
        "ci_lower": float(np.percentile(vals, 2.5)),
        "ci_upper": float(np.percentile(vals, 97.5)),
        "n_boot": int(len(vals)),
    }


def run_subset(subset: str):
    out_dir = OUT_BASE / subset / "E-A1"
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, ids, feature_names = load_subset(subset)
    n = len(y)
    print(f"START {subset}: n={n} features={X.shape[1]}", flush=True)

    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    results = {m: {"repeat_metrics": [], "fold_train_time": [], "fold_infer_time": [],
                    "fold_infer_n": [], "oof": None} for m in MODEL_NAMES}
    for m in MODEL_NAMES:
        results[m]["oof"] = [np.full(n, np.nan) for _ in range(N_REPEATS)]

    proc = psutil.Process()
    peak_rss = 0
    fold_counter = 0
    for train_idx, test_idx in rkf.split(X, y):
        repeat_idx = fold_counter // N_SPLITS
        print(f"  fold {fold_counter+1}/{N_SPLITS*N_REPEATS} repeat {repeat_idx}", flush=True)
        fold_counter += 1
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        for model_name in MODEL_NAMES:
            model = make_model(model_name)
            t0 = time.time()
            model.fit(Xtr, ytr)
            train_time = time.time() - t0
            t1 = time.time()
            proba = model.predict_proba(Xte)[:, 1]
            infer_time = time.time() - t1
            results[model_name]["fold_train_time"].append(float(train_time))
            results[model_name]["fold_infer_time"].append(float(infer_time))
            results[model_name]["fold_infer_n"].append(int(len(test_idx)))
            oof = results[model_name]["oof"][repeat_idx]
            oof[test_idx] = proba
            peak_rss = max(peak_rss, proc.memory_info().rss)

    # Per-repeat metrics
    for model_name in MODEL_NAMES:
        rm = []
        for r in range(N_REPEATS):
            oof = results[model_name]["oof"][r]
            if np.isnan(oof).any():
                raise RuntimeError(f"{model_name} repeat {r} has missing OOF predictions")
            m = full_metrics(y, oof, THRESHOLD)
            rm.append(m)
        results[model_name]["repeat_metrics"] = rm

    # Bootstrap CI on repeat 0
    for model_name in MODEL_NAMES:
        oof0 = results[model_name]["oof"][0]
        results[model_name]["bootstrap_ci"] = {
            "roc_auc": bootstrap_ci(y, oof0, "roc_auc"),
            "pr_auc": bootstrap_ci(y, oof0, "pr_auc"),
            "balanced_accuracy": bootstrap_ci(y, oof0, "balanced_accuracy"),
        }

    # Confusion matrix at threshold on repeat 0
    for model_name in MODEL_NAMES:
        oof0 = results[model_name]["oof"][0]
        pred0 = (oof0 >= THRESHOLD).astype(int)
        from sklearn.metrics import confusion_matrix as cmf

        cm = cmf(y, pred0, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        results[model_name]["confusion_matrix_repeat0"] = {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }

    # Aggregate repeat metrics
    summary = {}
    for model_name in MODEL_NAMES:
        rm = results[model_name]["repeat_metrics"]
        agg = {}
        for key in rm[0].keys():
            if key in ("threshold", "n", "n_positive"):
                agg[key] = rm[0][key]
                continue
            vals = [m[key] for m in rm]
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "per_repeat": [float(v) for v in vals],
            }
        # timing
        train_times = results[model_name]["fold_train_time"]
        infer_times = results[model_name]["fold_infer_time"]
        infer_ns = results[model_name]["fold_infer_n"]
        agg["training_time_sec"] = {
            "mean_per_fold": float(np.mean(train_times)),
            "std_per_fold": float(np.std(train_times, ddof=1)),
            "total": float(np.sum(train_times)),
        }
        total_infer_time = float(np.sum(infer_times))
        total_infer_n = int(np.sum(infer_ns))
        agg["inference_latency_ms_per_sample"] = float(total_infer_time / max(total_infer_n, 1) * 1000.0)
        agg["bootstrap_ci"] = results[model_name]["bootstrap_ci"]
        agg["confusion_matrix_repeat0"] = results[model_name]["confusion_matrix_repeat0"]
        summary[model_name] = agg

    # Save OOF predictions
    np.savez_compressed(
        out_dir / "oof_predictions.npz",
        y_true=y,
        sample_ids=ids,
        **{m: results[m]["oof"][0] for m in MODEL_NAMES},
    )

    # Save metrics JSON
    metrics_payload = {
        "phase": "5F",
        "experiment": "E-A1",
        "subset": subset,
        "modality": "acoustic",
        "n_samples": int(n),
        "n_positive": int(y.sum()),
        "n_features": int(X.shape[1]),
        "threshold": THRESHOLD,
        "cv": {
            "type": "RepeatedStratifiedKFold",
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": SEED,
            "stratification": "label",
            "grouping": "none (no participant id released); sample-level only",
        },
        "models": summary,
        "peak_rss_mb": float(peak_rss / (1024 ** 2)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # Config
    config = {
        "phase": "5F",
        "experiment": "E-A1",
        "subset": subset,
        "modality": "acoustic",
        "cache": str(CACHE_NPZ.relative_to(PROJECT_ROOT)),
        "n_samples": int(n),
        "n_features": int(X.shape[1]),
        "models": {
            "logistic_regression": {"C": 1.0, "class_weight": "balanced", "solver": "liblinear", "max_iter": 5000},
            "random_forest": {"n_estimators": 300, "class_weight": "balanced_subsample"},
            "hist_gradient_boosting": {"max_iter": 50, "learning_rate": 0.1, "l2_regularization": 1.0, "class_weight": "balanced"},
        },
        "preprocessing": "SimpleImputer(median) in-fold; StandardScaler in-fold for LR only",
        "random_state": SEED,
        "threshold": THRESHOLD,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Confusion matrices separate file
    cm_payload = {m: summary[m]["confusion_matrix_repeat0"] for m in MODEL_NAMES}
    (out_dir / "confusion_matrices.json").write_text(json.dumps(cm_payload, indent=2), encoding="utf-8")

    # ROC/PR plots using repeat 0 OOF
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name in MODEL_NAMES:
        oof0 = results[model_name]["oof"][0]
        fpr, tpr, _ = roc_curve(y, oof0)
        prec, rec, _ = precision_recall_curve(y, oof0)
        auc = summary[model_name]["roc_auc"]["mean"]
        ap = summary[model_name]["pr_auc"]["mean"]
        axes[0].plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
        axes[1].plot(rec, prec, label=f"{model_name} (AP={ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC - E-A1 acoustic {subset}")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR - E-A1 acoustic {subset}")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    print(f"Completed {subset}: n={n}, features={X.shape[1]}")
    print(json.dumps({m: {k: summary[m][k] for k in ('roc_auc', 'pr_auc', 'balanced_accuracy', 'f1')} for m in MODEL_NAMES}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["HDS", "SDS", "both"], default="both")
    args = parser.parse_args()
    subsets = ["HDS", "SDS"] if args.subset == "both" else [args.subset]
    all_summary = {}
    for subset in subsets:
        all_summary[subset] = run_subset(subset)
    if len(subsets) == 2:
        comp = {
            "experiment": "E-A1",
            "modality": "acoustic",
            "subsets": {},
        }
        for subset in subsets:
            comp["subsets"][subset] = {
                m: {
                    "roc_auc_mean": all_summary[subset][m]["roc_auc"]["mean"],
                    "roc_auc_std": all_summary[subset][m]["roc_auc"]["std"],
                    "pr_auc_mean": all_summary[subset][m]["pr_auc"]["mean"],
                    "balanced_accuracy_mean": all_summary[subset][m]["balanced_accuracy"]["mean"],
                    "f1_mean": all_summary[subset][m]["f1"]["mean"],
                }
                for m in MODEL_NAMES
            }
        out = OUT_BASE / "avdd_ea1_comparison.json"
        out.write_text(json.dumps(comp, indent=2), encoding="utf-8")
        lines = ["# AVDD E-A1 acoustic comparison", "", "| Subset | Model | ROC-AUC (mean+-sd) | PR-AUC | Balanced Acc | F1 |", "|---|---|---:|---:|---:|---:|"]
        for subset in subsets:
            for m in MODEL_NAMES:
                s = all_summary[subset][m]
                lines.append(
                    f"| {subset} | {m} | {s['roc_auc']['mean']:.3f}+-{s['roc_auc']['std']:.3f} | "
                    f"{s['pr_auc']['mean']:.3f} | {s['balanced_accuracy']['mean']:.3f} | {s['f1']['mean']:.3f} |"
                )
        (OUT_BASE / "avdd_ea1_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("Wrote comparison reports")


if __name__ == "__main__":
    main()
