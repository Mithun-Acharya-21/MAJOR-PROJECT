"""Phase 5C — E-L2 metadata confound baseline (EATD, dev pool only).

Logistic regression on audio duration, transcript length and topic proportions,
under the same repeated nested group CV as E-T1. Measures how much of any
text signal is explainable by recording length / topic prompt.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.phase2.dataset import load_manifest, load_participants, load_splits, read_transcript
from ml.evaluation.splits import (
    eatd_development_units,
    eatd_locked_test_group_ids,
    inner_group_folds,
    repeated_stratified_group_folds,
    unit_index,
)
from ml.evaluation.thresholds import PRIMARY_POLICY, apply_threshold, select_threshold
from ml.evaluation.bootstrap import aggregate_by_participant as bs_aggregate
from ml.evaluation.metrics import compute_evaluation_metrics
from ml.evaluation.artifacts import build_experiment_artifact, sha256_of_files, validate_experiment_artifact, write_experiment_artifact
from ml.evaluation.provenance import build_provenance, OPERATION_OOF_PREDICTION, PARTITION_INNER_TRAIN, PARTITION_INNER_VALIDATION
from ml.phase2.evaluation import metric_bundle

OUTER_SEEDS = (42, 43, 44, 45, 46)
OUTER_FOLDS = 5
INNER_FOLDS = 4


def _load_dev_rows():
    splits = load_splits()
    participants = load_participants()
    manifest = load_manifest()
    dev_keys = {k for k, s in splits.items() if s in ("train", "validation")}
    label_by_pk = {p["participant_key"]: int(p["label"]) for p in participants}
    rows_by_unit = {}
    for r in manifest:
        pk = r["participant_key"]
        if pk not in dev_keys:
            continue
        text = read_transcript(r["text_path"])
        uid = f"{pk}/{r['topic']}"
        rows_by_unit[uid] = {
            "unit_id": uid,
            "participant_key": pk,
            "topic": r["topic"],
            "text": text,
            "label": int(r["label"]),
            "participant_label": int(label_by_pk[pk]),
            "audio_duration_sec": float(r["audio_duration_sec"]),
        }
    units = eatd_development_units(manifest=manifest, splits=splits, participants=participants)
    topics = sorted({r["topic"] for r in rows_by_unit.values()})
    return units, rows_by_unit, topics


def _features(rows, topics, scaler=None, fit=False):
    X = np.array(
        [[r["audio_duration_sec"], len(r["text"])] + [1.0 if r["topic"] == t else 0.0 for t in topics] for r in rows],
        dtype=float,
    )
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler


def _fit_predict(train_rows, eval_rows, topics):
    Xtr, scaler = _features(train_rows, topics, fit=True)
    Xev, _ = _features(eval_rows, topics, scaler=scaler)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=0)
    clf.fit(Xtr, [r["label"] for r in train_rows])
    probs = clf.predict_proba(Xev)[:, 1]
    return probs


def main():
    units, rows_by_unit, topics = _load_dev_rows()
    units_idx = unit_index(units)
    locked = eatd_locked_test_group_ids()
    reps = repeated_stratified_group_folds(units, n_splits=OUTER_FOLDS, seeds=OUTER_SEEDS, locked_group_ids=locked)

    fold_rows = []
    t_start = time.time()
    for rep in reps:
        oof_session = {}
        for outer_fold in rep.folds:
            train_rows = [rows_by_unit[u] for u in outer_fold.train_unit_ids]
            eval_rows = [rows_by_unit[u] for u in outer_fold.eval_unit_ids]

            eval_probs = _fit_predict(train_rows, eval_rows, topics)
            for u, p in zip(outer_fold.eval_unit_ids, eval_probs):
                oof_session[u] = float(p)

            # inner OOF for threshold
            inner_rep = inner_group_folds(units_idx, outer_fold, n_splits=INNER_FOLDS, locked_group_ids=locked)
            inner_session = {}
            for inner_fold in inner_rep.folds:
                itr = [rows_by_unit[u] for u in inner_fold.train_unit_ids]
                iev = [rows_by_unit[u] for u in inner_fold.eval_unit_ids]
                ps = _fit_predict(itr, iev, topics)
                for u, p in zip(inner_fold.eval_unit_ids, ps):
                    inner_session[u] = float(p)

            uids = list(inner_session.keys())
            labels = [rows_by_unit[u]["participant_label"] for u in uids]
            probs = [inner_session[u] for u in uids]
            groups = [rows_by_unit[u]["participant_key"] for u in uids]
            agg_inner = bs_aggregate(labels, probs, groups, aggregation="mean")
            prov_inner = build_provenance(
                partition=PARTITION_INNER_VALIDATION,
                operation=OPERATION_OOF_PREDICTION,
                dataset="eatd",
                unit="participant",
                labels=agg_inner["labels"],
                n_groups=agg_inner["n_units"],
                seeds=(rep.seed,),
                source="E-L2",
                contributing_partitions=(PARTITION_INNER_TRAIN,),
            )
            tsel = select_threshold(agg_inner["labels"], agg_inner["probs"], provenance=prov_inner, policy=PRIMARY_POLICY)

            agg_eval = bs_aggregate(
                [r["participant_label"] for r in eval_rows],
                list(eval_probs),
                [r["participant_key"] for r in eval_rows],
                aggregation="mean",
            )
            metrics_sel = compute_evaluation_metrics(agg_eval["labels"], agg_eval["probs"], unit="participant", threshold=tsel.threshold)
            metrics_half = compute_evaluation_metrics(agg_eval["labels"], agg_eval["probs"], unit="participant", threshold=0.5)

            fold_rows.append({
                "repeat_seed": rep.seed,
                "fold": outer_fold.fold,
                "n_eval_participants": int(agg_eval["n_units"]),
                "threshold": tsel.to_dict(),
                "metrics_at_selected_threshold": metrics_sel,
                "metrics_at_0_5": metrics_half,
            })

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

    out_dir = PROJECT_ROOT / "results" / "phase5" / "eatd" / "E-L2"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "5C",
        "experiment_id": "E-L2",
        "dataset": "eatd",
        "modality": "metadata",
        "status": "measured",
        "elapsed_seconds": round(time.time() - t_start, 2),
        "locked_test_used": False,
        "folds": fold_rows,
        "aggregate": aggregate,
        "known_limitations": [
            "Development pool only; locked test not used.",
            "Metadata-only confound baseline (duration + transcript length + topic).",
            "No clinical validation.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    audit_path = PROJECT_ROOT / "data" / "reports" / "phase_5b_split_audit.json"
    fold_hash = sha256_of_files([audit_path]) if audit_path.exists() else "0" * 64
    artifact = build_experiment_artifact(
        experiment_id="E-L2",
        dataset={"name": "eatd", "version": "eatd_manifest_v1", "provenance_reference": "data/reports/phase_5b_split_audit.json", "sha256": fold_hash},
        feature_configuration={"features": "duration+transcript_length+topic_onehot", "standardized": True},
        label_definition={"rule": "sds_index_ge_50", "positive_class": 1, "definition": "label = 1 if sds_index >= 50"},
        evaluation={"unit": "participant", "n_units": 138, "n_positive": 45, "split_strategy": "repeated_nested_stratified_group_cv", "seeds": list(OUTER_SEEDS), "fold_assignments_reference": "data/reports/phase_5b_split_audit.json", "fold_assignments_sha256": fold_hash},
        preprocessing={"configuration": {"standardize": True}, "fit_partition": "outer_train"},
        model={"configuration": {"family": "logistic_regression"}, "hyperparameters": {"C": 1.0}},
        selection={"threshold_method": "balanced_accuracy", "threshold_partition": "inner_validation", "calibration_method": "none", "calibration_partition": "inner_validation"},
        modality="text",
        phase="5C",
        results={
            "status": "measured",
            "metrics": {
                "roc_auc": {"value": aggregate["mean_roc_auc_selected"], "unit": "participant"},
                "pr_auc": {"value": aggregate["mean_pr_auc_selected"], "unit": "participant"},
            },
            "confidence_intervals": {},
            "source": "results/phase5/eatd/E-L2/report.json",
        },
        known_limitations=["Development pool only; locked test not used.", "No clinical validation."],
    )
    validate_experiment_artifact(artifact)
    write_experiment_artifact(artifact, out_dir / "E-L2.json")
    print(f"E-L2 complete: mean ROC-AUC={aggregate['mean_roc_auc_selected']}, mean PR-AUC={aggregate['mean_pr_auc_selected']}")


if __name__ == "__main__":
    main()
