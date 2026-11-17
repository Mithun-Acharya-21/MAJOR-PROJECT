"""Phase 5D — E-V3 frozen Wav2Vec2 voice (EATD, dev pool only)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.phase2.dataset import load_participants, load_splits
from ml.phase2.text_transformer import FrozenTextHead
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
EPOCHS = 40
LR = 1e-4
HIDDEN = 128
DROPOUT = 0.3
BATCH_SIZE = 32

CACHE_NPZ = PROJECT_ROOT / "models" / "experiments" / "eatd_voice_wav2vec2_v1" / "wav2vec2_TencentGameMate__chinese_wav2vec2_base_chunk8s_max12.npz"
CACHE_JSON = PROJECT_ROOT / "models" / "experiments" / "eatd_voice_wav2vec2_v1" / "wav2vec2_TencentGameMate__chinese_wav2vec2_base_chunk8s_max12.npz.json"


def _load_features():
    splits = load_splits()
    dev_keys = {k for k, s in splits.items() if s in ("train", "validation")}
    data = np.load(CACHE_NPZ, allow_pickle=False)
    meta = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
    X_all = data["X"]
    rows_all = meta["rows"]
    X_dev = []
    rows_dev = []
    for i, r in enumerate(rows_all):
        if r["participant_key"] in dev_keys:
            X_dev.append(X_all[i])
            rows_dev.append(r)
    X_dev = np.asarray(X_dev, dtype=np.float32)
    units_all = eatd_development_units()
    uid_set = {f"{r['participant_key']}/{r['topic']}" for r in rows_dev}
    units = [u for u in units_all if u.unit_id in uid_set]
    rows_by_unit = {}
    features_by_unit = {}
    for r, x in zip(rows_dev, X_dev):
        uid = f"{r['participant_key']}/{r['topic']}"
        rows_by_unit[uid] = {
            "unit_id": uid,
            "participant_key": r["participant_key"],
            "topic": r["topic"],
            "participant_label": int(r["participant_label"]),
        }
        features_by_unit[uid] = np.asarray(x, dtype=np.float32)
    return units, rows_by_unit, features_by_unit


def _train_head(Xtr, ytr, device):
    set_seed(42)
    in_dim = int(Xtr.shape[1])
    model = FrozenTextHead(in_dim, hidden=HIDDEN, dropout=DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    yt = np.asarray(ytr, dtype=np.float32)
    pos = float(yt.sum())
    neg = float(len(yt) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(yt, dtype=torch.float32, device=device)
    n = len(yt)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n).to(device)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            logits = model(Xtr_t[idx])
            loss = criterion(logits, ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _probs(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy().astype(float)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    units, rows_by_unit, features_by_unit = _load_features()
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
            Xtr = np.asarray([features_by_unit[u] for u in train_uids], dtype=np.float32)
            Xev = np.asarray([features_by_unit[u] for u in eval_uids], dtype=np.float32)
            train_rows = [rows_by_unit[u] for u in train_uids]
            eval_rows = [rows_by_unit[u] for u in eval_uids]

            model = _train_head(Xtr, [r["participant_label"] for r in train_rows], device)
            t_infer = time.time()
            eval_probs = _probs(model, Xev, device)
            inference_times.append(time.time() - t_infer)

            inner_rep = inner_group_folds(units_idx, outer_fold, n_splits=INNER_FOLDS, locked_group_ids=locked)
            inner_session = {}
            for inner_fold in inner_rep.folds:
                itr_uids = list(inner_fold.train_unit_ids)
                iev_uids = list(inner_fold.eval_unit_ids)
                Xtr_i = np.asarray([features_by_unit[u] for u in itr_uids], dtype=np.float32)
                Xev_i = np.asarray([features_by_unit[u] for u in iev_uids], dtype=np.float32)
                itr_rows = [rows_by_unit[u] for u in itr_uids]
                m = _train_head(Xtr_i, [r["participant_label"] for r in itr_rows], device)
                ps = _probs(m, Xev_i, device)
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
                source="E-V3",
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

    peak_mem = round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None

    out_dir = PROJECT_ROOT / "results" / "phase5" / "eatd_voice" / "E-V3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if plot_labels is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve

        fpr, tpr, _ = roc_curve(plot_labels, plot_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("E-V3 ROC (seed 42, participant-level)")
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=150)
        plt.close()

        prec, rec, _ = precision_recall_curve(plot_labels, plot_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("E-V3 Precision-Recall (seed 42)")
        plt.tight_layout()
        plt.savefig(out_dir / "pr_curve.png", dpi=150)
        plt.close()

        curve = reliability_curve(plot_labels, plot_probs, n_bins=10)
        xs = [(c["bin_lower"] + c["bin_upper"]) / 2 for c in curve if c["count"]]
        ys = [c["observed_frequency"] for c in curve if c["count"]]
        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.scatter(xs, ys)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed frequency")
        plt.title("E-V3 Calibration (seed 42)")
        plt.tight_layout()
        plt.savefig(out_dir / "calibration_plot.png", dpi=150)
        plt.close()

        cm = confusion_matrix_counts(plot_labels, plot_probs, threshold=plot_threshold, unit="participant")
        (out_dir / "confusion_matrix.json").write_text(json.dumps(cm, indent=2), encoding="utf-8")

    report = {
        "phase": "5D",
        "experiment_id": "E-V3",
        "dataset": "eatd",
        "modality": "voice",
        "status": "measured",
        "elapsed_seconds": round(time.time() - t_start, 2),
        "training_seconds_total": round(time.time() - t_start, 2),
        "inference_seconds_mean_per_fold": round(float(np.mean(inference_times)), 4) if inference_times else None,
        "gpu_memory_gb": peak_mem,
        "locked_test_used": False,
        "folds": fold_rows,
        "aggregate": aggregate,
        "known_limitations": [
            "Development pool only; locked test not used.",
            "One unusable audio session excluded.",
            "Frozen encoder; head trained with fixed epochs (no early stopping).",
            "No clinical validation.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    audit_path = PROJECT_ROOT / "data" / "reports" / "phase_5b_split_audit.json"
    fold_hash = sha256_of_files([audit_path]) if audit_path.exists() else "0" * 64
    artifact = build_experiment_artifact(
        experiment_id="E-V3",
        dataset={"name": "eatd", "version": "eatd_manifest_v1", "provenance_reference": "data/reports/phase_5b_split_audit.json", "sha256": fold_hash},
        feature_configuration={"encoder": "TencentGameMate/chinese-wav2vec2-base", "pooling": "mean+std over frames per chunk, mean over chunks", "chunk_sec": 8, "max_chunks": 12, "frozen": True},
        label_definition={"rule": "sds_index_ge_50", "positive_class": 1, "definition": "label = 1 if sds_index >= 50"},
        evaluation={"unit": "participant", "n_units": 138, "n_positive": 45, "split_strategy": "repeated_nested_stratified_group_cv", "seeds": list(OUTER_SEEDS), "fold_assignments_reference": "data/reports/phase_5b_split_audit.json", "fold_assignments_sha256": fold_hash},
        preprocessing={"configuration": {"standardize": False}, "fit_partition": "outer_train"},
        model={"configuration": {"family": "frozen_wav2vec2_head"}, "hyperparameters": {"epochs": EPOCHS, "lr": LR, "hidden": HIDDEN}},
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
            "source": "results/phase5/eatd_voice/E-V3/report.json",
        },
        known_limitations=["Development pool only; locked test not used.", "One unusable audio session excluded.", "No clinical validation."],
    )
    validate_experiment_artifact(artifact)
    write_experiment_artifact(artifact, out_dir / "E-V3.json")
    print(f"E-V3 complete: mean ROC-AUC={aggregate['mean_roc_auc_selected']}, mean PR-AUC={aggregate['mean_pr_auc_selected']}")


if __name__ == "__main__":
    main()
