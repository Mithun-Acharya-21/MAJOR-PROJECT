"""Phase 5C — E-T2 frozen multilingual transformer (EATD, dev pool only)."""

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

from ml.phase2.dataset import load_manifest, load_participants, load_splits, read_transcript
from ml.phase2.text_transformer import embed_texts, FrozenTextHead
from ml.evaluation.splits import (
    eatd_development_units,
    eatd_locked_test_group_ids,
    inner_group_folds,
    repeated_stratified_group_folds,
    unit_index,
)
from ml.evaluation.thresholds import PRIMARY_POLICY, select_threshold
from ml.evaluation.bootstrap import aggregate_by_participant as bs_aggregate
from ml.evaluation.metrics import compute_evaluation_metrics
from ml.evaluation.artifacts import build_experiment_artifact, sha256_of_files, validate_experiment_artifact, write_experiment_artifact
from ml.evaluation.provenance import build_provenance, OPERATION_OOF_PREDICTION, PARTITION_INNER_TRAIN, PARTITION_INNER_VALIDATION

OUTER_SEEDS = (42, 43, 44, 45, 46)
OUTER_FOLDS = 5
INNER_FOLDS = 4
EPOCHS = 40
LR = 1e-4
HIDDEN = 128
DROPOUT = 0.3
MAX_SEQ_LEN = 128
BATCH_SIZE = 32


def _load_dev():
    splits = load_splits()
    participants = load_participants()
    manifest = load_manifest()
    dev_keys = {k for k, s in splits.items() if s in ("train", "validation")}
    label_by_pk = {p["participant_key"]: int(p["label"]) for p in participants}
    ordered_uids = []
    rows_by_unit = {}
    texts = []
    for r in manifest:
        pk = r["participant_key"]
        if pk not in dev_keys:
            continue
        text = read_transcript(r["text_path"])
        uid = f"{pk}/{r['topic']}"
        ordered_uids.append(uid)
        rows_by_unit[uid] = {
            "unit_id": uid,
            "participant_key": pk,
            "topic": r["topic"],
            "text": text,
            "label": int(r["label"]),
            "participant_label": int(label_by_pk[pk]),
        }
        texts.append(text)
    units = eatd_development_units(manifest=manifest, splits=splits, participants=participants)
    return units, rows_by_unit, ordered_uids, texts


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
    units, rows_by_unit, ordered_uids, texts = _load_dev()
    units_idx = unit_index(units)
    locked = eatd_locked_test_group_ids()
    reps = repeated_stratified_group_folds(units, n_splits=OUTER_FOLDS, seeds=OUTER_SEEDS, locked_group_ids=locked)

    # embeddings
    cache_dir = PROJECT_ROOT / "results" / "phase5" / "eatd" / "E-T2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "embeddings_dev.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        embeddings = data["embeddings"]
    else:
        embeddings, stats = embed_texts(texts, encoder_name="distilbert/distilbert-base-multilingual-cased", max_seq_len=MAX_SEQ_LEN, batch_size=8)
        np.savez_compressed(cache_path, embeddings=embeddings)
        (cache_dir / "embeddings_dev.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    uid_to_idx = {uid: i for i, uid in enumerate(ordered_uids)}

    fold_rows = []
    t_start = time.time()
    for rep in reps:
        oof_session = {}
        for outer_fold in rep.folds:
            train_uids = list(outer_fold.train_unit_ids)
            eval_uids = list(outer_fold.eval_unit_ids)
            Xtr = embeddings[[uid_to_idx[u] for u in train_uids]]
            Xev = embeddings[[uid_to_idx[u] for u in eval_uids]]
            ytr = [rows_by_unit[u]["label"] for u in train_uids]
            model = _train_head(Xtr, ytr, device)
            eval_probs = _probs(model, Xev, device)
            for u, p in zip(eval_uids, eval_probs):
                oof_session[u] = float(p)

            # inner OOF for threshold
            inner_rep = inner_group_folds(units_idx, outer_fold, n_splits=INNER_FOLDS, locked_group_ids=locked)
            inner_session = {}
            for inner_fold in inner_rep.folds:
                itr_uids = list(inner_fold.train_unit_ids)
                iev_uids = list(inner_fold.eval_unit_ids)
                Xtr_i = embeddings[[uid_to_idx[u] for u in itr_uids]]
                Xev_i = embeddings[[uid_to_idx[u] for u in iev_uids]]
                ytr_i = [rows_by_unit[u]["label"] for u in itr_uids]
                m = _train_head(Xtr_i, ytr_i, device)
                ps = _probs(m, Xev_i, device)
                for u, p in zip(iev_uids, ps):
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
                source="E-T2",
                contributing_partitions=(PARTITION_INNER_TRAIN,),
            )
            tsel = select_threshold(agg_inner["labels"], agg_inner["probs"], provenance=prov_inner, policy=PRIMARY_POLICY)

            agg_eval = bs_aggregate(
                [rows_by_unit[u]["participant_label"] for u in eval_uids],
                list(eval_probs),
                [rows_by_unit[u]["participant_key"] for u in eval_uids],
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

    out_dir = cache_dir
    report = {
        "phase": "5C",
        "experiment_id": "E-T2",
        "dataset": "eatd",
        "modality": "text",
        "status": "measured",
        "elapsed_seconds": round(time.time() - t_start, 2),
        "locked_test_used": False,
        "folds": fold_rows,
        "aggregate": aggregate,
        "known_limitations": [
            "Development pool only; locked test not used.",
            "Frozen encoder; head trained with fixed epochs (no early stopping).",
            "No clinical validation.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    audit_path = PROJECT_ROOT / "data" / "reports" / "phase_5b_split_audit.json"
    fold_hash = sha256_of_files([audit_path]) if audit_path.exists() else "0" * 64
    artifact = build_experiment_artifact(
        experiment_id="E-T2",
        dataset={"name": "eatd", "version": "eatd_manifest_v1", "provenance_reference": "data/reports/phase_5b_split_audit.json", "sha256": fold_hash},
        feature_configuration={"encoder": "distilbert-base-multilingual-cased", "pooling": "masked_mean", "max_seq_len": MAX_SEQ_LEN, "frozen": True},
        label_definition={"rule": "sds_index_ge_50", "positive_class": 1, "definition": "label = 1 if sds_index >= 50"},
        evaluation={"unit": "participant", "n_units": 138, "n_positive": 45, "split_strategy": "repeated_nested_stratified_group_cv", "seeds": list(OUTER_SEEDS), "fold_assignments_reference": "data/reports/phase_5b_split_audit.json", "fold_assignments_sha256": fold_hash},
        preprocessing={"configuration": {"standardize": False}, "fit_partition": "outer_train"},
        model={"configuration": {"family": "frozen_transformer_head"}, "hyperparameters": {"epochs": EPOCHS, "lr": LR, "hidden": HIDDEN}},
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
            "source": "results/phase5/eatd/E-T2/report.json",
        },
        known_limitations=["Development pool only; locked test not used.", "No clinical validation."],
    )
    validate_experiment_artifact(artifact)
    write_experiment_artifact(artifact, out_dir / "E-T2.json")
    print(f"E-T2 complete: mean ROC-AUC={aggregate['mean_roc_auc_selected']}, mean PR-AUC={aggregate['mean_pr_auc_selected']}")


if __name__ == "__main__":
    main()
