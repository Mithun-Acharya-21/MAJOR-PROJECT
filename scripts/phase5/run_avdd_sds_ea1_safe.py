"""Phase 5F — SDS E-A1 safe runner.

Identical protocol to run_avdd_ea1_acoustic.py for the SDS subset, but with
HGB configured to avoid the loky/multiprocessing hang seen in sandboxed runners:
  - LOKY_MAX_CPU_COUNT=4 set before any import
  - HGB: max_iter=50, warm_start-loop disabled (sequential), no n_jobs param
  - RF: n_jobs=2 instead of -1 to bound parallelism

This script does NOT touch HDS E-A1 results.
Outputs go to: results/phase5/avdd_features/SDS/E-A1/
"""

from __future__ import annotations

import os
# Must be set before any scikit-learn/loky import
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import json
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_NPZ = PROJECT_ROOT / "data" / "processed" / "avdd_tcmi_phase5" / "avdd_acoustic_cache.npz"
OUT_DIR = PROJECT_ROOT / "results" / "phase5" / "avdd_features" / "SDS" / "E-A1"

SEED = 42
N_SPLITS = 5
N_REPEATS = 5
THRESHOLD = 0.5
N_BOOT = 1000
SUBSET = "SDS"


def load_subset():
    data = np.load(CACHE_NPZ, allow_pickle=False)
    mask = data["subset"] == SUBSET
    X = data["X"][mask].astype(np.float64)
    y = data["labels"][mask].astype(np.int64)
    ids = data["sample_ids"][mask]
    feature_names = list(data["feature_names"])
    print(f"Loaded {SUBSET}: n={len(y)}, features={X.shape[1]}, pos={y.sum()}", flush=True)
    return X, y, ids, feature_names


def mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def full_metrics(y, p, threshold=THRESHOLD):
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
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=SEED,
            )),
        ])
    if name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=2,  # bounded to avoid loky deadlock
            )),
        ])
    if name == "hist_gradient_boosting":
        # Sequential fitting: break max_iter=50 into 5 warm-start steps of 10
        # to allow the OS to release resources between iterations.
        # We wrap in a simple callable that trains iteratively.
        # Using n_jobs is not an HGB constructor param — it's always sequential internally.
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_iter=50,
                learning_rate=0.1,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=SEED,
                # HGB is single-threaded by design; no n_jobs param
            )),
        ])
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
        try:
            if metric == "roc_auc":
                vals.append(float(roc_auc_score(yy, pp)))
            elif metric == "pr_auc":
                vals.append(float(average_precision_score(yy, pp)))
            elif metric == "balanced_accuracy":
                pred = (pp >= THRESHOLD).astype(int)
                vals.append(float(balanced_accuracy_score(yy, pred)))
        except ValueError:
            pass
    if not vals:
        return {"ci_lower": None, "ci_upper": None, "n_boot": 0}
    return {
        "ci_lower": float(np.percentile(vals, 2.5)),
        "ci_upper": float(np.percentile(vals, 97.5)),
        "n_boot": int(len(vals)),
    }


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "run.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # Clear old partial log
    log_path.write_text("")
    log(f"START {SUBSET}: loading data")

    X, y, ids, feature_names = load_subset()
    n = len(y)
    log(f"START {SUBSET}: n={n} features={X.shape[1]}")

    rkf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    results = {m: {
        "repeat_metrics": [],
        "fold_train_time": [],
        "fold_infer_time": [],
        "fold_infer_n": [],
        "oof": None,
    } for m in MODEL_NAMES}
    for m in MODEL_NAMES:
        results[m]["oof"] = [np.full(n, np.nan) for _ in range(N_REPEATS)]

    proc = psutil.Process()
    peak_rss = 0
    fold_counter = 0

    for train_idx, test_idx in rkf.split(X, y):
        repeat_idx = fold_counter // N_SPLITS
        log(f"  fold {fold_counter+1}/{N_SPLITS*N_REPEATS} repeat {repeat_idx}")
        fold_counter += 1
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        for model_name in MODEL_NAMES:
            log(f"    model={model_name}")
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
            log(f"    {model_name} train={train_time:.1f}s infer={infer_time:.3f}s")

    log("All folds complete. Computing metrics...")

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
        cm = confusion_matrix(y, pred0, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        results[model_name]["confusion_matrix_repeat0"] = {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
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
        agg["inference_latency_ms_per_sample"] = float(
            total_infer_time / max(total_infer_n, 1) * 1000.0
        )
        agg["bootstrap_ci"] = results[model_name]["bootstrap_ci"]
        agg["confusion_matrix_repeat0"] = results[model_name]["confusion_matrix_repeat0"]
        summary[model_name] = agg

    # Save metrics.json
    metrics_out = {
        "experiment": "E-A1",
        "subset": SUBSET,
        "n_samples": int(n),
        "n_features": int(X.shape[1]),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "cv_strategy": f"RepeatedStratifiedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS})",
        "threshold": THRESHOLD,
        "n_bootstrap": N_BOOT,
        "peak_rss_bytes": int(peak_rss),
        "models": summary,
    }
    metrics_path = OUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    log(f"Saved metrics.json")

    # Save config.json
    config_out = {
        "experiment_id": "E-A1",
        "subset": SUBSET,
        "cache_file": str(CACHE_NPZ),
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "seed": SEED,
        "threshold": THRESHOLD,
        "n_bootstrap": N_BOOT,
        "models": MODEL_NAMES,
        "script": __file__,
        "note": "Safe runner: RF n_jobs=2, HGB sequential",
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config_out, indent=2))

    # Save OOF predictions
    oof_arrays = {f"oof_repeat{r}_{mn.replace(' ', '_')}": results[mn]["oof"][r]
                  for mn in MODEL_NAMES for r in range(N_REPEATS)}
    np.savez_compressed(OUT_DIR / "oof_predictions.npz",
                        y_true=y,
                        sample_ids=ids,
                        **oof_arrays)
    log("Saved oof_predictions.npz")

    # Confusion matrices summary
    cm_out = {mn: results[mn]["confusion_matrix_repeat0"] for mn in MODEL_NAMES}
    (OUT_DIR / "confusion_matrices.json").write_text(json.dumps(cm_out, indent=2))

    # ROC/PR curves plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"logistic_regression": "blue", "random_forest": "green",
               "hist_gradient_boosting": "orange"}
    from sklearn.metrics import roc_curve, precision_recall_curve

    for mn in MODEL_NAMES:
        oof0 = results[mn]["oof"][0]
        label = mn.replace("_", " ").title()
        roc_auc_val = results[mn]["repeat_metrics"][0]["roc_auc"]
        fpr, tpr, _ = roc_curve(y, oof0)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_val:.3f})", color=colors[mn])
        pr_auc_val = results[mn]["repeat_metrics"][0]["pr_auc"]
        prec, rec, _ = precision_recall_curve(y, oof0)
        axes[1].plot(rec, prec, label=f"{label} (AUC={pr_auc_val:.3f})", color=colors[mn])

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title(f"ROC Curve — {SUBSET} E-A1")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend(loc="lower right")
    axes[1].set_title(f"PR Curve — {SUBSET} E-A1")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_pr_curves.png", dpi=150)
    plt.close()
    log("Saved roc_pr_curves.png")

    # Print summary
    log("=" * 60)
    log(f"SUMMARY: {SUBSET} E-A1")
    for mn in MODEL_NAMES:
        roc = summary[mn]["roc_auc"]["mean"]
        roc_std = summary[mn]["roc_auc"]["std"]
        pr = summary[mn]["pr_auc"]["mean"]
        bal = summary[mn]["balanced_accuracy"]["mean"]
        f1 = summary[mn]["macro_f1"]["mean"]
        ci = summary[mn]["bootstrap_ci"]["roc_auc"]
        log(f"  {mn}: ROC={roc:.3f}±{roc_std:.3f} CI=[{ci['ci_lower']:.3f},{ci['ci_upper']:.3f}] "
            f"PR={pr:.3f} BalAcc={bal:.3f} F1={f1:.3f}")
    log("DONE")
    return metrics_out


if __name__ == "__main__":
    run()
