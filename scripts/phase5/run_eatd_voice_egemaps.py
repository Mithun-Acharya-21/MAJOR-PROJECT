"""Phase 5D — E-V1/E-V2 eGeMAPS voice (EATD, dev pool only).

Usage:
    python scripts/phase5/run_eatd_voice_egemaps.py --model lr   # E-V1
    python scripts/phase5/run_eatd_voice_egemaps.py --model svm  # E-V2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.phase2.dataset import load_participants, load_splits
from ml.phase2.evaluation import metric_bundle
from ml.evaluation.splits import (
    eatd_development_units,
    eatd_locked_test_group_ids,
    inner_group_folds,
    repeated_stratified_group_folds,
    unit_index,
)
from ml.evaluation.thresholds import PRIMARY_POLICY, select_threshold
from ml.evaluation.bootstrap import participant_cluster_bootstrap
from ml.evaluation.metrics import compute_evaluation_metrics, confusion_matrix_counts, reliability_curve
from ml.evaluation.artifacts import build_experiment_artifact, sha256_of_files, validate_experiment_artifact, write_experiment_artifact
from ml.evaluation.provenance import build_provenance, OPERATION_OOF_PREDICTION, PARTITION_INNER_TRAIN, PARTITION_INNER_VALIDATION

OUTER_SEEDS = (42, 43, 44, 45, 46)
OUTER_FOLDS = 5
INNER_FOLDS = 4

CACHE_NPZ = PROJECT_ROOT / "models" / "experiments" / "eatd_voice_egemaps_v1" / "egemaps_features.npz"
CACHE_JSON = PROJECT_ROOT / "models" / "experiments" / "eatd_voice_egemaps_v1" / "egemaps_features.npz.json"


def _load_features():
    splits = load_splits()
    dev_keys = {k for k, s in splits.items() if s in ("train", "validation")}
    data = np.load(CACHE_NPZ, allow_pickle=False)
    meta = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
    X_all = data["X"]
    rows_all = meta["rows"]
    feature_names = meta["feature_names"]
    X_dev = []
    rows_dev = []
    for i, r in enumerate(rows_all):
        if r["participant_key"] in dev_keys:
            X_dev.append(X_all[i])
            rows_dev.append(r)
    X_dev = np.asarray(X_dev, dtype=float)
    units_all = eatd_development_units()
    excluded_uids = set()
    for u in units_all:
        if u.unit_id not in {f"{r['participant_key']}/{r['topic']}" for r in rows_dev}:
            excluded_uids.add(u.unit_id)
    units = [u for u in units_all if u.unit_id not in excluded_uids]
    rows_by_unit = {}
    features_by_unit = {}
    for r, x in zip(rows_dev, X_dev):
        uid = f"{r['participant_key']}/{r['topic']}"
        rows_by_unit[uid] = {
            "unit_id": uid,
            "participant_key": r["participant_key"],
            "topic": r["topic"],
            "participant_label": int(r["participant_label"]),
            "audio_duration_sec": float(r["audio_duration_sec"]),
        }
        features_by_unit[uid] = np.asarray(x, dtype=float)
    return units, rows_by_unit, features_by_unit, feature_names


def _fit_preprocess(X_train):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    return pipe.fit(X_train)


def _grid(model_type):
    if model_type == "lr":
        return [{"C": C, "class_weight": cw} for C in (0.01, 0.1, 1.0, 10.0) for cw in (None, "balanced")]
    return [{"C": C, "gamma": g, "class_weight": cw}
            for C in (0.1, 1.0, 10.0)
            for g in ("scale", "auto")
            for cw in (None, "balanced")]


def _fit_model(model_type, X, y, params):
    if model_type == "lr":
        clf = LogisticRegression(C=params["C"], class_weight=params["class_weight"], solver="liblinear", max_iter=5000, random_state=0)
    else:
        clf = SVC(C=params["C"], gamma=params["gamma"], class_weight=params["class_weight"], probability=True, random_state=0)
    clf.fit(X, list(y))
    return clf


def _participant_auc(rows, probs):
    agg = __import__("ml.evaluation.bootstrap", fromlist=["aggregate_by_participant"]).aggregate_by_participant(
        [r["participant_label"] for r in rows],
        list(probs),
        [r["participant_key"] for r in rows],
        aggregation="mean",
    )
    m = metric_bundle(agg["labels"], agg["probs"])
    return float(m["roc_auc"]) if m["roc_auc"] is not None else 0.0


def _select_best_params(model_type, train_uids, inner_folds, rows_by_unit, features_by_unit):
    best_params = None
    best_score = -1.0
    for params in _grid(model_type):
        scores = []
        for inner_fold in inner_folds:
            itr_uids = list(inner_fold.train_unit_ids)
            iev_uids = list(inner_fold.eval_unit_ids)
            Xtr = np.asarray([features_by_unit[u] for u in itr_uids], dtype=float)
            Xev = np.asarray([features_by_unit[u] for u in iev_uids], dtype=float)
            itr_rows = [rows_by_unit[u] for u in itr_uids]
            iev_rows = [rows_by_unit[u] for u in iev_uids]
            pre = _fit_preprocess(Xtr)
            clf = _fit_model(model_type, pre.transform(Xtr), [r["participant_label"] for r in itr_rows], params)
            probs = clf.predict_proba(pre.transform(Xev))[:, 1]
            scores.append(_participant_auc(iev_rows, probs))
        mean_auc = float(np.mean(scores)) if scores else 0.0
        if mean_auc > best_score + 1e-12:
            best_score = mean_auc
            best_params = dict(params)
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lr", "svm"], required=True)
    args = parser.parse_args()
    model_type = args.model
    exp_id = "E-V1" if model_type == "lr" else "E-V2"
    model_name = "logistic_regression" if model_type == "lr" else "support_vector_machine_rbf"
    out_dir = PROJECT_ROOT / "results" / "phase5" / "eatd_voice" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    units, rows_by_unit, features_by_unit, feature_names = _load_features()
    units_idx = unit_index(units)
    locked = eatd_locked_test_group_ids()
    reps = repeated_stratified_group_folds(units, n_splits=OUTER_FOLDS, seeds=OUTER_SEEDS, locked_group_ids=locked)

    fold_rows = []
    t_start = time.time()
    inference_times = []
    plot_labels = None
    plot_probs = None
    plot_threshold = 0.5

    for rep in reps:
        for outer_fold in rep.folds:
            train_uids = list(outer_fold.train_unit_ids)
            eval_uids = list(outer_fold.eval_unit_ids)
            Xtr = np.asarray([features_by_unit[u] for u in train_uids], dtype=float)
            Xev = np.asarray([features_by_unit[u] for u in eval_uids], dtype=float)
            train_rows = [rows_by_unit[u] for u in train_uids]
            eval_rows = [rows_by_unit[u] for u in eval_uids]

            inner_rep = inner_group_folds(units_idx, outer_fold, n_splits=INNER_FOLDS, locked_group_ids=locked)
            best_params, _ = _select_best_params(model_type, train_uids, inner_rep.folds, rows_by_unit, features_by_unit)

            pre = _fit_preprocess(Xtr)
            clf = _fit_model(model_type, pre.transform(Xtr), [r["participant_label"] for r in train_rows], best_params)
            t_infer = time.time()
            eval_probs = clf.predict_proba(pre.transform(Xev))[:, 1]
            inference_times.append(time.time() - t_infer)

            for u, p in zip(eval_uids, eval_probs):
                pass

            # inner OOF threshold
            inner_session = {}
            for inner_fold in inner_rep.folds:
                itr_uids = list(inner_fold.train_unit_ids)
                iev_uids = list(inner_fold.eval_unit_ids)
                Xtr_i = np.asarray([features_by_unit[u] for u in itr_uids], dtype=float)
                Xev_i = np.asarray([features_by_unit[u] for u in iev_uids], dtype=float)
                itr_rows = [rows_by_unit[u] for u in itr_uids]
                iev_rows = [rows_by_unit[u] for u in iev_uids]
                pre_i = _fit_preprocess(Xtr_i)
                clf_i = _fit_model(model_type, pre_i.transform(Xtr_i), [r["participant_label"] for r in itr_rows], best_params)
                ps = clf_i.predict_proba(pre_i.transform(Xev_i))[:, 1]
                for u, p in zip(iev_uids, ps):
                    inner_session[u] = float(p)

            uids = list(inner_session.keys())
            labels_inner = [rows_by_unit[u]["participant_label"] for u in uids]
            probs_inner = [inner_session[u] for u in uids]
            groups_inner = [rows_by_unit[u]["participant_key"] for u in uids]
            agg_inner = __import__("ml.evaluation.bootstrap", fromlist=["aggregate_by_participant"]).aggregate_by_participant(
                labels_inner, probs_inner, groups_inner, aggregation="mean"
            )
            prov_inner = build_provenance(
                partition=PARTITION_INNER_VALIDATION,
                operation=OPERATION_OOF_PREDICTION,
                dataset="eatd",
                unit="participant",
                labels=agg_inner["labels"],
                n_groups=agg_inner["n_units"],
                seeds=(rep.seed,),
                source=exp_id,
                contributing_partitions=(PARTITION_INNER_TRAIN,),
            )
            tsel = select_threshold(agg_inner["labels"], agg_inner["probs"], provenance=prov_inner, policy=PRIMARY_POLICY)

            agg_eval = __import__("ml.evaluation.bootstrap", fromlist=["aggregate_by_participant"]).aggregate_by_participant(
                [r["participant_label"] for r in eval_rows],
                list(eval_probs),
                [r["participant_key"] for r in eval_rows],
                aggregation="mean",
            )
            metrics_sel = compute_evaluation_metrics(agg_eval["labels"], agg_eval["probs"], unit="participant", threshold=tsel.threshold)
            metrics_half = compute_evaluation_metrics(agg_eval["labels"], agg_eval["probs"], unit="participant", threshold=0.5)

            bs = participant_cluster_bootstrap(
                [r["participant_label"] for r in eval_rows],
                list(eval_probs),
                [r["participant_key"] for r in eval_rows],
                n_resamples=2000,
                seed=0,
                threshold=tsel.threshold,
            )

            fold_rows.append({
                "repeat_seed": rep.seed,
                "fold": outer_fold.fold,
                "n_eval_participants": int(agg_eval["n_units"]),
                "n_eval_sessions": len(eval_uids),
                "best_params": best_params,
                "threshold": tsel.to_dict(),
                "metrics_at_selected_threshold": metrics_sel,
                "metrics_at_0_5": metrics_half,
                "bootstrap": bs,
            })

            if rep.seed == 42 and plot_labels is None:
                plot_labels = np.asarray(agg_eval["labels"], dtype=float)
                plot_probs = np.asarray(agg_eval["probs"], dtype=float)
                plot_threshold = float(tsel.threshold)

    def get_metric(fr, name):
        m = fr["metrics_at_selected_threshold"]["metrics"][name]
        return float(m["value"]) if m.get("defined") and m.get("value") is not None else None

    per_fold_roc = [get_metric(fr, "roc_auc") for fr in fold_rows]
    per_fold_pr = [get_metric(fr, "pr_auc") for fr in fold_rows]
    aggregate = {
        "n_folds": len(fold_rows),
        "mean_roc_auc_selected": round(float(np.nanmean([x for x in per_fold_roc if x is not None])), 4) if any(x is not None for x in per_fold_roc) else None,
        "sd_roc_auc_selected": round(float(np.nanstd([x for x in per_fold_roc if x is not None], ddof=1)), 4) if sum(x is not None for x in per_fold_roc) > 1 else None,
        "mean_pr_auc_selected": round(float(np.nanmean([x for x in per_fold_pr if x is not None])), 4) if any(x is not None for x in per_fold_pr) else None,
        "sd_pr_auc_selected": round(float(np.nanstd([x for x in per_fold_pr if x is not None], ddof=1)), 4) if sum(x is not None for x in per_fold_pr) > 1 else None,
    }

    # plots
    if plot_labels is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve

        fpr, tpr, _ = roc_curve(plot_labels, plot_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC (AUC={aggregate['mean_roc_auc_selected']})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{exp_id} ROC (seed 42, participant-level)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=150)
        plt.close()

        prec, rec, _ = precision_recall_curve(plot_labels, plot_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{exp_id} Precision-Recall (seed 42)")
        plt.tight_layout()
        plt.savefig(out_dir / "pr_curve.png", dpi=150)
        plt.close()

        curve = reliability_curve(plot_labels, plot_probs, n_bins=10)
        xs = [ (c["bin_lower"]+c["bin_upper"])/2 for c in curve if c["count"] ]
        ys = [ c["observed_frequency"] for c in curve if c["count"] ]
        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.scatter(xs, ys)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed frequency")
        plt.title(f"{exp_id} Calibration (seed 42)")
        plt.tight_layout()
        plt.savefig(out_dir / "calibration_plot.png", dpi=150)
        plt.close()

        cm = confusion_matrix_counts(plot_labels, plot_probs, threshold=plot_threshold, unit="participant")
        (out_dir / "confusion_matrix.json").write_text(json.dumps(cm, indent=2), encoding="utf-8")

    report = {
        "phase": "5D",
        "experiment_id": exp_id,
        "dataset": "eatd",
        "modality": "voice",
        "status": "measured",
        "elapsed_seconds": round(time.time() - t_start, 2),
        "training_seconds_total": round(time.time() - t_start, 2),
        "inference_seconds_mean_per_fold": round(float(np.mean(inference_times)), 4) if inference_times else None,
        "inference_ms_per_session_mean": round(float(np.mean(inference_times)) * 1000 / max(1, len(eval_uids)), 4) if inference_times else None,
        "gpu_memory_gb": None,
        "locked_test_used": False,
        "folds": fold_rows,
        "aggregate": aggregate,
        "known_limitations": [
            "Development pool only; locked test not used.",
            "One unusable audio session excluded (0.48 s).",
            "No clinical validation.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    audit_path = PROJECT_ROOT / "data" / "reports" / "phase_5b_split_audit.json"
    fold_hash = sha256_of_files([audit_path]) if audit_path.exists() else "0" * 64
    artifact = build_experiment_artifact(
        experiment_id=exp_id,
        dataset={"name": "eatd", "version": "eatd_manifest_v1", "provenance_reference": "data/reports/phase_5b_split_audit.json", "sha256": fold_hash},
        feature_configuration={"feature_set": "eGeMAPSv02", "n_features": len(feature_names), "imputation": "train_median", "scaling": "standard", "model": model_name},
        label_definition={"rule": "sds_index_ge_50", "positive_class": 1, "definition": "label = 1 if sds_index >= 50"},
        evaluation={"unit": "participant", "n_units": 138, "n_positive": 45, "split_strategy": "repeated_nested_stratified_group_cv", "seeds": list(OUTER_SEEDS), "fold_assignments_reference": "data/reports/phase_5b_split_audit.json", "fold_assignments_sha256": fold_hash},
        preprocessing={"configuration": {"imputer": "median", "scaler": "standard"}, "fit_partition": "outer_train"},
        model={"configuration": {"family": model_name}, "hyperparameters": {"grid": "inner_cv"}},
        selection={"threshold_method": "balanced_accuracy", "threshold_partition": "inner_validation", "calibration_method": "none", "calibration_partition": "inner_validation"},
        modality="voice",
        phase="5D",
        results={
            "status": "measured",
            "metrics": {
                "roc_auc": {"value": aggregate["mean_roc_auc_selected"], "unit": "participant"},
                "pr_auc": {"value": aggregate["mean_pr_auc_selected"], "unit": "participant"},
            },
            "confidence_intervals": {},
            "source": f"results/phase5/eatd_voice/{exp_id}/report.json",
        },
        known_limitations=["Development pool only; locked test not used.", "One unusable audio session excluded.", "No clinical validation."],
    )
    validate_experiment_artifact(artifact)
    write_experiment_artifact(artifact, out_dir / f"{exp_id}.json")
    print(f"{exp_id} complete: mean ROC-AUC={aggregate['mean_roc_auc_selected']}, mean PR-AUC={aggregate['mean_pr_auc_selected']}")


if __name__ == "__main__":
    main()
