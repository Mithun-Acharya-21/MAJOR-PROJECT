"""Phase 5E - EATD paired text/voice error analysis and late fusion."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.phase2.dataset import load_manifest, load_participants, load_splits, read_transcript
from ml.phase2.text_tfidf import build_features, candidate_grid, fit_model as tfidf_fit_model, predict_rows as tfidf_predict_rows
from ml.phase2.text_transformer import FrozenTextHead, embed_texts
from ml.evaluation.splits import (
    Unit,
    eatd_development_units,
    eatd_locked_test_group_ids,
    inner_group_folds,
    repeated_stratified_group_folds,
    unit_index,
)
from ml.evaluation.thresholds import PRIMARY_POLICY, select_threshold
from ml.evaluation.bootstrap import aggregate_by_participant as bs_aggregate
from ml.evaluation.metrics import compute_evaluation_metrics
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
VOICE_CACHE_NPZ = PROJECT_ROOT / "models" / "experiments" / "eatd_voice_wav2vec2_v1" / "wav2vec2_TencentGameMate__chinese_wav2vec2_base_chunk8s_max12.npz"
VOICE_CACHE_JSON = PROJECT_ROOT / "models" / "experiments" / "eatd_voice_wav2vec2_v1" / "wav2vec2_TencentGameMate__chinese_wav2vec2_base_chunk8s_max12.npz.json"
TEXT_CACHE_PATH = PROJECT_ROOT / "results" / "phase5" / "eatd" / "E-T2" / "embeddings_dev.npz"

OUT_DIR = PROJECT_ROOT / "results" / "phase5" / "eatd_fusion"
OOF_DIR = OUT_DIR / "oof_predictions"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _load_text_rows():
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
    return units, rows_by_unit

def _load_text_transformer_rows():
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

def _train_head_text(Xtr, ytr, device):
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

def _probs_text(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy().astype(float)

def _agg_inner_session(session_probs, rows_by_unit):
    uids = list(session_probs.keys())
    labels = [rows_by_unit[u]["participant_label"] for u in uids]
    probs = [session_probs[u] for u in uids]
    groups = [rows_by_unit[u]["participant_key"] for u in uids]
    return bs_aggregate(labels, probs, groups, aggregation="mean")

def _metric_val(metrics_dict, name):
    m = metrics_dict["metrics"][name]
    if m.get("defined") and m.get("value") is not None:
        return float(m["value"])
    return None

def _get_metric(fr, name):
    m = fr["metrics_at_selected_threshold"]["metrics"][name]
    return float(m["value"]) if m.get("defined") and m.get("value") is not None else None

# ---------------- E-T1 OOF reconstruction ----------------
print("Reconstructing E-T1 OOF predictions...", flush=True)
units_t1, rows_by_unit_t1 = _load_text_rows()
units_idx_t1 = unit_index(units_t1)
locked_t1 = eatd_locked_test_group_ids()
reps_t1 = repeated_stratified_group_folds(units_t1, n_splits=OUTER_FOLDS, seeds=OUTER_SEEDS, locked_group_ids=locked_t1)

outer_session_t1 = {}  # uid -> prob
outer_fold_of_t1 = {}
inner_session_t1 = {}  # (seed,fold) -> {uid:prob}
outer_part_t1 = {}     # (seed,fold) -> (labels, probs, ids)
inner_part_t1 = {}
thresholds_t1 = {}
fold_records_t1 = []
oof_records_t1 = []

t_start = time.time()
for rep in reps_t1:
    for outer_fold in rep.folds:
        train_rows = [rows_by_unit_t1[u] for u in outer_fold.train_unit_ids]
        eval_rows = [rows_by_unit_t1[u] for u in outer_fold.eval_unit_ids]

        # hyperparameter selection
        best_cfg = None
        best_key = (-1.0, -1.0)
        trials = []
        for cfg in candidate_grid():
            aucs = []
            f1s = []
            for inner_fold in inner_group_folds(units_idx_t1, outer_fold, n_splits=INNER_FOLDS, locked_group_ids=locked_t1).folds:
                itr = [rows_by_unit_t1[u] for u in inner_fold.train_unit_ids]
                iev = [rows_by_unit_t1[u] for u in inner_fold.eval_unit_ids]
                features, clf = tfidf_fit_model(itr, kind=cfg["kind"], C=cfg["C"], class_weight=cfg["class_weight"])
                probs = tfidf_predict_rows(features, clf, iev)
                agg = bs_aggregate([r["participant_label"] for r in iev], list(probs), [r["participant_key"] for r in iev], aggregation="mean")
                from ml.phase2.evaluation import metric_bundle
                m = metric_bundle(agg["labels"], agg["probs"])
                aucs.append(float(m["roc_auc"]) if m["roc_auc"] is not None else 0.0)
                f1s.append(float(m["f1"]) if m["f1"] is not None else 0.0)
            mean_auc = float(np.mean(aucs)) if aucs else 0.0
            mean_f1 = float(np.mean(f1s)) if f1s else 0.0
            trials.append({**cfg, "inner_mean_participant_roc_auc": round(mean_auc, 4)})
            if (mean_auc, mean_f1) > best_key:
                best_key = (mean_auc, mean_f1)
                best_cfg = cfg
        if best_cfg is None:
            best_cfg = candidate_grid()[0]

        features, clf = tfidf_fit_model(train_rows, kind=best_cfg["kind"], C=best_cfg["C"], class_weight=best_cfg["class_weight"])
        eval_probs = tfidf_predict_rows(features, clf, eval_rows)
        for u, p in zip(outer_fold.eval_unit_ids, eval_probs):
            outer_session_t1[u] = float(p)
            outer_fold_of_t1[u] = (rep.seed, outer_fold.fold)

        # inner OOF
        inner_rep = inner_group_folds(units_idx_t1, outer_fold, n_splits=INNER_FOLDS, locked_group_ids=locked_t1)
        inner_session = {}
        for inner_fold in inner_rep.folds:
            itr = [rows_by_unit_t1[u] for u in inner_fold.train_unit_ids]
            iev = [rows_by_unit_t1[u] for u in inner_fold.eval_unit_ids]
            f, c = tfidf_fit_model(itr, kind=best_cfg["kind"], C=best_cfg["C"], class_weight=best_cfg["class_weight"])
            ps = tfidf_predict_rows(f, c, iev)
            for u, p in zip(inner_fold.eval_unit_ids, ps):
                inner_session[u] = float(p)
        inner_session_t1[(rep.seed, outer_fold.fold)] = inner_session

        agg_inner = _agg_inner_session(inner_session, rows_by_unit_t1)
        prov_inner = build_provenance(partition=PARTITION_INNER_VALIDATION, operation=OPERATION_OOF_PREDICTION, dataset="eatd", unit="participant", labels=agg_inner["labels"], n_groups=agg_inner["n_units"], seeds=(rep.seed,), source="E-T1", contributing_partitions=(PARTITION_INNER_TRAIN,))
        tsel = select_threshold(agg_inner["labels"], agg_inner["probs"], provenance=prov_inner, policy=PRIMARY_POLICY)
        thresholds_t1[(rep.seed, outer_fold.fold)] = float(tsel.threshold)

        agg_eval = bs_aggregate([r["participant_label"] for r in eval_rows], list(eval_probs), [r["participant_key"] for r in eval_rows], aggregation="mean")
        outer_part_t1[(rep.seed, outer_fold.fold)] = (np.asarray(agg_eval["labels"]), np.asarray(agg_eval["probs"]), list(agg_eval["participant_ids"]))
        inner_part_t1[(rep.seed, outer_fold.fold)] = (np.asarray(agg_inner["labels"]), np.asarray(agg_inner["probs"]), list(agg_inner["participant_ids"]))

        fold_records_t1.append({"repeat_seed": rep.seed, "fold": outer_fold.fold, "threshold": float(tsel.threshold), "n_eval_participants": int(agg_eval["n_units"])})
        for u in outer_fold.eval_unit_ids:
            oof_records_t1.append({
                "model": "E-T1",
                "repeat_seed": rep.seed,
                "fold": outer_fold.fold,
                "unit_id": u,
                "participant_key": rows_by_unit_t1[u]["participant_key"],
                "topic": rows_by_unit_t1[u]["topic"],
                "session_label": int(rows_by_unit_t1[u]["label"]),
                "participant_label": int(rows_by_unit_t1[u]["participant_label"]),
                "probability": float(outer_session_t1[u]),
                "threshold": float(tsel.threshold),
            })

print(f"E-T1 done in {time.time()-t_start:.1f}s", flush=True)

# ---------------- E-T2 OOF reconstruction ----------------
print("Reconstructing E-T2 OOF predictions...", flush=True)
units_t2, rows_by_unit_t2, ordered_uids_t2, texts_t2 = _load_text_transformer_rows()
units_idx_t2 = unit_index(units_t2)
locked_t2 = eatd_locked_test_group_ids()
reps_t2 = repeated_stratified_group_folds(units_t2, n_splits=OUTER_FOLDS, seeds=OUTER_SEEDS, locked_group_ids=locked_t2)

# embeddings
if TEXT_CACHE_PATH.exists():
    data = np.load(TEXT_CACHE_PATH)
    embeddings_t2 = data["embeddings"]
else:
    embeddings_t2, _ = embed_texts(texts_t2, encoder_name="distilbert/distilbert-base-multilingual-cased", max_seq_len=MAX_SEQ_LEN, batch_size=8)
    np.savez_compressed(TEXT_CACHE_PATH, embeddings=embeddings_t2)
uid_to_idx_t2 = {uid: i for i, uid in enumerate(ordered_uids_t2)}

outer_session_t2 = {}
inner_session_t2 = {}
outer_part_t2 = {}
inner_part_t2 = {}
thresholds_t2 = {}
oof_records_t2 = []

t_start = time.time()
for rep in reps_t2:
    for outer_fold in rep.folds:
        train_uids = list(outer_fold.train_unit_ids)
        eval_uids = list(outer_fold.eval_unit_ids)
        Xtr = embeddings_t2[[uid_to_idx_t2[u] for u in train_uids]]
        Xev = embeddings_t2[[uid_to_idx_t2[u] for u in eval_uids]]
        ytr = [rows_by_unit_t2[u]["label"] for u in train_uids]
        model = _train_head_text(Xtr, ytr, device)
        eval_probs = _probs_text(model, Xev, device)
        for u, p in zip(eval_uids, eval_probs):
            outer_session_t2[u] = float(p)

        inner_rep = inner_group_folds(units_idx_t2, outer_fold, n_splits=INNER_FOLDS, locked_group_ids=locked_t2)
        inner_session = {}
        for inner_fold in inner_rep.folds:
            itr_uids = list(inner_fold.train_unit_ids)
            iev_uids = list(inner_fold.eval_unit_ids)
            Xtr_i = embeddings_t2[[uid_to_idx_t2[u] for u in itr_uids]]
            Xev_i = embeddings_t2[[uid_to_idx_t2[u] for u in iev_uids]]
            ytr_i = [rows_by_unit_t2[u]["label"] for u in itr_uids]
            m = _train_head_text(Xtr_i, ytr_i, device)
            ps = _probs_text(m, Xev_i, device)
            for u, p in zip(iev_uids, ps):
                inner_session[u] = float(p)
        inner_session_t2[(rep.seed, outer_fold.fold)] = inner_session

        agg_inner = _agg_inner_session(inner_session, rows_by_unit_t2)
        prov_inner = build_provenance(partition=PARTITION_INNER_VALIDATION, operation=OPERATION_OOF_PREDICTION, dataset="eatd", unit="participant", labels=agg_inner["labels"], n_groups=agg_inner["n_units"], seeds=(rep.seed,), source="E-T2", contributing_partitions=(PARTITION_INNER_TRAIN,))
        tsel = select_threshold(agg_inner["labels"], agg_inner["probs"], provenance=prov_inner, policy=PRIMARY_POLICY)
        thresholds_t2[(rep.seed, outer_fold.fold)] = float(tsel.threshold)

        eval_rows = [rows_by_unit_t2[u] for u in eval_uids]
        agg_eval = bs_aggregate([rows_by_unit_t2[u]["participant_label"] for u in eval_uids], list(eval_probs), [rows_by_unit_t2[u]["participant_key"] for u in eval_uids], aggregation="mean")
        outer_part_t2[(rep.seed, outer_fold.fold)] = (np.asarray(agg_eval["labels"]), np.asarray(agg_eval["probs"]), list(agg_eval["participant_ids"]))
        inner_part_t2[(rep.seed, outer_fold.fold)] = (np.asarray(agg_inner["labels"]), np.asarray(agg_inner["probs"]), list(agg_inner["participant_ids"]))

        for u in eval_uids:
            oof_records_t2.append({
                "model": "E-T2",
                "repeat_seed": rep.seed,
                "fold": outer_fold.fold,
                "unit_id": u,
                "participant_key": rows_by_unit_t2[u]["participant_key"],
                "topic": rows_by_unit_t2[u]["topic"],
                "session_label": int(rows_by_unit_t2[u]["label"]),
                "participant_label": int(rows_by_unit_t2[u]["participant_label"]),
                "probability": float(outer_session_t2[u]),
                "threshold": float(tsel.threshold),
            })

print(f"E-T2 done in {time.time()-t_start:.1f}s", flush=True)

# ---------------- E-V3 OOF reconstruction ----------------
print("Reconstructing E-V3 OOF predictions...", flush=True)
import json as _json
vdata = np.load(VOICE_CACHE_NPZ, allow_pickle=False)
vmeta = _json.loads(VOICE_CACHE_JSON.read_text(encoding="utf-8"))
X_all_v = vdata["X"]
rows_all_v = vmeta["rows"]
splits = load_splits()
dev_keys = {k for k, s in splits.items() if s in ("train", "validation")}

X_dev_v = []
rows_dev_v = []
for i, r in enumerate(rows_all_v):
    if r["participant_key"] in dev_keys:
        X_dev_v.append(X_all_v[i])
        rows_dev_v.append(r)
X_dev_v = np.asarray(X_dev_v, dtype=np.float32)

from ml.evaluation.splits import eatd_development_units, eatd_locked_test_group_ids, inner_group_folds, repeated_stratified_group_folds, unit_index
units_v = eatd_development_units()
uid_set_v = {f"{r['participant_key']}/{r['topic']}" for r in rows_dev_v}
units_v = [u for u in units_v if u.unit_id in uid_set_v]
rows_by_unit_v = {}
features_by_unit_v = {}
for r, x in zip(rows_dev_v, X_dev_v):
    uid = f"{r['participant_key']}/{r['topic']}"
    rows_by_unit_v[uid] = {
        "unit_id": uid,
        "participant_key": r["participant_key"],
        "topic": r["topic"],
        "participant_label": int(r["participant_label"]),
    }
    features_by_unit_v[uid] = np.asarray(x, dtype=np.float32)

units_idx_v = unit_index(units_v)
locked_v = eatd_locked_test_group_ids()
reps_v = repeated_stratified_group_folds(units_v, n_splits=OUTER_FOLDS, seeds=OUTER_SEEDS, locked_group_ids=locked_v)

outer_session_v = {}
inner_session_v = {}
outer_part_v = {}
inner_part_v = {}
thresholds_v = {}
oof_records_v = []

def _train_head_v(Xtr, ytr, device):
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

def _probs_v(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy().astype(float)

t_start = time.time()
for rep in reps_v:
    for outer_fold in rep.folds:
        train_uids = list(outer_fold.train_unit_ids)
        eval_uids = list(outer_fold.eval_unit_ids)
        Xtr = np.asarray([features_by_unit_v[u] for u in train_uids], dtype=np.float32)
        Xev = np.asarray([features_by_unit_v[u] for u in eval_uids], dtype=np.float32)
        train_rows = [rows_by_unit_v[u] for u in train_uids]
        eval_rows = [rows_by_unit_v[u] for u in eval_uids]

        model = _train_head_v(Xtr, [r["participant_label"] for r in train_rows], device)
        eval_probs = _probs_v(model, Xev, device)
        for u, p in zip(eval_uids, eval_probs):
            outer_session_v[u] = float(p)

        inner_rep = inner_group_folds(units_idx_v, outer_fold, n_splits=INNER_FOLDS, locked_group_ids=locked_v)
        inner_session = {}
        for inner_fold in inner_rep.folds:
            itr_uids = list(inner_fold.train_unit_ids)
            iev_uids = list(inner_fold.eval_unit_ids)
            Xtr_i = np.asarray([features_by_unit_v[u] for u in itr_uids], dtype=np.float32)
            Xev_i = np.asarray([features_by_unit_v[u] for u in iev_uids], dtype=np.float32)
            itr_rows = [rows_by_unit_v[u] for u in itr_uids]
            m = _train_head_v(Xtr_i, [r["participant_label"] for r in itr_rows], device)
            ps = _probs_v(m, Xev_i, device)
            for u, p in zip(iev_uids, ps):
                inner_session[u] = float(p)
        inner_session_v[(rep.seed, outer_fold.fold)] = inner_session

        agg_inner = _agg_inner_session(inner_session, rows_by_unit_v)
        prov_inner = build_provenance(partition=PARTITION_INNER_VALIDATION, operation=OPERATION_OOF_PREDICTION, dataset="eatd", unit="participant", labels=agg_inner["labels"], n_groups=agg_inner["n_units"], seeds=(rep.seed,), source="E-V3", contributing_partitions=(PARTITION_INNER_TRAIN,))
        tsel = select_threshold(agg_inner["labels"], agg_inner["probs"], provenance=prov_inner, policy=PRIMARY_POLICY)
        thresholds_v[(rep.seed, outer_fold.fold)] = float(tsel.threshold)

        agg_eval = bs_aggregate([r["participant_label"] for r in eval_rows], list(eval_probs), [r["participant_key"] for r in eval_rows], aggregation="mean")
        outer_part_v[(rep.seed, outer_fold.fold)] = (np.asarray(agg_eval["labels"]), np.asarray(agg_eval["probs"]), list(agg_eval["participant_ids"]))
        inner_part_v[(rep.seed, outer_fold.fold)] = (np.asarray(agg_inner["labels"]), np.asarray(agg_inner["probs"]), list(agg_inner["participant_ids"]))

        for u in eval_uids:
            oof_records_v.append({
                "model": "E-V3",
                "repeat_seed": rep.seed,
                "fold": outer_fold.fold,
                "unit_id": u,
                "participant_key": rows_by_unit_v[u]["participant_key"],
                "topic": rows_by_unit_v[u]["topic"],
                "session_label": None,
                "participant_label": int(rows_by_unit_v[u]["participant_label"]),
                "probability": float(outer_session_v[u]),
                "threshold": float(tsel.threshold),
            })

print(f"E-V3 done in {time.time()-t_start:.1f}s", flush=True)

# ---------------- Save OOF predictions ----------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
OOF_DIR.mkdir(parents=True, exist_ok=True)

def _save_oof(records, name):
    import csv
    path = OOF_DIR / f"{name}_oof.csv"
    if not records:
        return
    keys = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)
    (OOF_DIR / f"{name}_oof.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

_save_oof(oof_records_t1, "E-T1")
_save_oof(oof_records_t2, "E-T2")
_save_oof(oof_records_v, "E-V3")

# ---------------- Participant-level first-repeat OOF ----------------
# Build participant-level probabilities for repeat seed 42 for each model.
def _participant_oof_for_repeat(records, seed=42):
    # records: list of session dicts
    from collections import defaultdict
    by_p = defaultdict(list)
    for r in records:
        if r["repeat_seed"] != seed:
            continue
        by_p[r["participant_key"]].append(r)
    out = {}
    for pk, rs in by_p.items():
        probs = [r["probability"] for r in rs]
        label = rs[0]["participant_label"]
        fold = rs[0]["fold"]
        thr = rs[0]["threshold"]
        out[pk] = {"label": int(label), "prob": float(np.mean(probs)), "fold": int(fold), "threshold": float(thr), "n_sessions": len(rs)}
    return out

part_t1_r42 = _participant_oof_for_repeat(oof_records_t1, 42)
part_t2_r42 = _participant_oof_for_repeat(oof_records_t2, 42)
part_v_r42 = _participant_oof_for_repeat(oof_records_v, 42)

common_participants = sorted(set(part_t1_r42) & set(part_t2_r42) & set(part_v_r42))
print("Common participants (first repeat):", len(common_participants), flush=True)

# ---------------- Error analysis E-T1 vs E-V3 ----------------
def _pred(part, thr=None):
    thr = thr if thr is not None else part["threshold"]
    return int(part["prob"] >= thr)

# use per-participant fold thresholds
both_correct = tc_only = vc_only = both_wrong = 0
fp_overlap = fn_overlap = 0
fp_t = fp_v = fn_t = fn_v = 0
for pk in common_participants:
    a = part_t1_r42[pk]
    b = part_v_r42[pk]
    pa = _pred(a)
    pb = _pred(b)
    y = a["label"]
    if pa == y and pb == y:
        both_correct += 1
    elif pa == y and pb != y:
        tc_only += 1
    elif pa != y and pb == y:
        vc_only += 1
    else:
        both_wrong += 1
    if pa == 1 and y == 0:
        fp_t += 1
    if pb == 1 and y == 0:
        fp_v += 1
    if pa == 1 and y == 0 and pb == 1 and y == 0:
        fp_overlap += 1
    if pa == 0 and y == 1:
        fn_t += 1
    if pb == 0 and y == 1:
        fn_v += 1
    if pa == 0 and y == 1 and pb == 0 and y == 1:
        fn_overlap += 1

# probability correlation
probs_t = np.array([part_t1_r42[pk]["prob"] for pk in common_participants])
probs_v = np.array([part_v_r42[pk]["prob"] for pk in common_participants])
labels_common = np.array([part_t1_r42[pk]["label"] for pk in common_participants])
from scipy.stats import pearsonr, spearmanr
pear = pearsonr(probs_t, probs_v)[0]
spear = spearmanr(probs_t, probs_v)[0]

# stability across repeats: for each model, fraction of participants with same predicted label in all repeats
def _stability(records):
    from collections import defaultdict
    by_p = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_p[r["participant_key"]][r["repeat_seed"]].append(r)
    stable = 0
    total = 0
    for pk, byseed in by_p.items():
        preds = []
        for seed in OUTER_SEEDS:
            rs = byseed.get(seed)
            if not rs:
                continue
            thr = rs[0]["threshold"]
            prob = float(np.mean([r["probability"] for r in rs]))
            preds.append(int(prob >= thr))
        if len(preds) == len(OUTER_SEEDS):
            total += 1
            if len(set(preds)) == 1:
                stable += 1
    return (stable / total if total else None, total)

stab_t1 = _stability(oof_records_t1)
stab_v = _stability(oof_records_v)

error_analysis = {
    "evaluation_repeat": 42,
    "n_common_participants": len(common_participants),
    "n_positive": int(labels_common.sum()),
    "E_T1_vs_E_V3": {
        "both_correct": both_correct,
        "text_correct_voice_incorrect": tc_only,
        "voice_correct_text_incorrect": vc_only,
        "both_incorrect": both_wrong,
        "text_false_positives": fp_t,
        "voice_false_positives": fp_v,
        "false_positive_overlap": fp_overlap,
        "text_false_negatives": fn_t,
        "voice_false_negatives": fn_v,
        "false_negative_overlap": fn_overlap,
        "probability_pearson_r": round(float(pear), 4),
        "probability_spearman_r": round(float(spear), 4),
    },
    "prediction_stability_across_repeats": {
        "E_T1": {"stable_participants": stab_t1[0], "n_participants": stab_t1[1]},
        "E_V3": {"stable_participants": stab_v[0], "n_participants": stab_v[1]},
    },
    "excluded_voice_session": "eatd_validation_79/positive (0.48 s) excluded from E-V3; participant still has 2 usable sessions.",
}
(OUT_DIR / "error_analysis.json").write_text(json.dumps(error_analysis, indent=2), encoding="utf-8")
print("Error analysis written.", flush=True)

# ---------------- Late fusion ----------------
# Fusion variants evaluated with the same outer folds. We fuse participant-level
# probabilities. Weights/thresholds come from inner-OOF only.

def _inner_fused_probs(inner_part, weight_text=0.5, weight_voice=0.5, kind="two"):
    # inner_part: dict model -> (labels, probs, ids) for a fold
    # returns aligned labels, fused probs, ids
    # align by participant id intersection
    ids_t = set(inner_part["text"][2])
    ids_v = set(inner_part["voice"][2])
    common = sorted(ids_t & ids_v)
    # map
    map_t = {pid: p for pid, p in zip(inner_part["text"][2], inner_part["text"][1])}
    map_v = {pid: p for pid, p in zip(inner_part["voice"][2], inner_part["voice"][1])}
    map_y = {pid: y for pid, y in zip(inner_part["text"][2], inner_part["text"][0])}
    # if labels differ, use text labels (should match)
    ys = np.array([map_y[pid] for pid in common], dtype=float)
    ps_t = np.array([map_t[pid] for pid in common], dtype=float)
    ps_v = np.array([map_v[pid] for pid in common], dtype=float)
    if kind == "two":
        fused = weight_text * ps_t + weight_voice * ps_v
    elif kind == "three":
        map_t2 = {pid: p for pid, p in zip(inner_part["text2"][2], inner_part["text2"][1])}
        ps_t2 = np.array([map_t2[pid] for pid in common], dtype=float)
        fused = (ps_t + ps_t2 + ps_v) / 3.0
    return ys, fused, common

def _optimize_weight(inner_t, inner_v):
    # grid search weight for text maximizing balanced accuracy on inner fused (threshold 0.5)
    from sklearn.metrics import balanced_accuracy_score
    map_t = {pid: p for pid, p in zip(inner_t[2], inner_t[1])}
    map_v = {pid: p for pid, p in zip(inner_v[2], inner_v[1])}
    common = sorted(set(inner_t[2]) & set(inner_v[2]))
    ys = np.array([inner_t[0][inner_t[2].index(pid)] for pid in common], dtype=float)
    ps_t = np.array([map_t[pid] for pid in common])
    ps_v = np.array([map_v[pid] for pid in common])
    best_w = 0.5
    best_score = -1.0
    for w in np.linspace(0, 1, 101):
        fused = w * ps_t + (1 - w) * ps_v
        preds = (fused >= 0.5).astype(int)
        score = balanced_accuracy_score(ys, preds)
        if score > best_score:
            best_score = score
            best_w = float(w)
    return best_w

fusion_variants = [
    "E-T1 only",
    "E-V3 only",
    "E-T1+E-V3 equal",
    "E-T1+E-V3 weighted",
    "E-T2+E-V3 equal",
    "E-T1+E-T2+E-V3 equal",
]

fusion_fold_rows = {v: [] for v in fusion_variants}

t_start = time.time()
for rep in reps_t1:  # iterate folds by E-T1 reps (same seeds/folds)
    for outer_fold in rep.folds:
        seed = rep.seed
        fold = outer_fold.fold
        key = (seed, fold)

        # single model outer participant probs
        labels_t1, probs_t1, ids_t1 = outer_part_t1[key]
        labels_v, probs_v, ids_v = outer_part_v[key]
        labels_t2, probs_t2, ids_t2 = outer_part_t2[key]

        # align outer participants for fusion
        def _align(labels_a, probs_a, ids_a, labels_b, probs_b, ids_b):
            common = sorted(set(ids_a) & set(ids_b))
            ma = {pid: (lab, prob) for pid, lab, prob in zip(ids_a, labels_a, probs_a)}
            mb = {pid: (lab, prob) for pid, lab, prob in zip(ids_b, labels_b, probs_b)}
            ys = np.array([ma[pid][0] for pid in common], dtype=float)
            pa = np.array([ma[pid][1] for pid in common], dtype=float)
            pb = np.array([mb[pid][1] for pid in common], dtype=float)
            return ys, pa, pb, common

        ys_tv, pa_t, pb_v, common_tv = _align(labels_t1, probs_t1, ids_t1, labels_v, probs_v, ids_v)
        ys_t2v, pa_t2, pb_v2, common_t2v = _align(labels_t2, probs_t2, ids_t2, labels_v, probs_v, ids_v)

        # inner parts for threshold/weight
        inner_t1 = inner_part_t1[key]
        inner_v = inner_part_v[key]
        inner_t2 = inner_part_t2[key]

        # --- E-T1 only ---
        thr = thresholds_t1[key]
        metrics = compute_evaluation_metrics(ys_tv, pa_t, unit="participant", threshold=thr)
        metrics_half = compute_evaluation_metrics(ys_tv, pa_t, unit="participant", threshold=0.5)
        fusion_fold_rows["E-T1 only"].append({"repeat_seed": seed, "fold": fold, "threshold": thr, "metrics_at_selected_threshold": metrics, "metrics_at_0_5": metrics_half, "n_eval_participants": len(ys_tv)})

        # --- E-V3 only ---
        thr_v = thresholds_v[key]
        metrics = compute_evaluation_metrics(ys_tv, pb_v, unit="participant", threshold=thr_v)
        metrics_half = compute_evaluation_metrics(ys_tv, pb_v, unit="participant", threshold=0.5)
        fusion_fold_rows["E-V3 only"].append({"repeat_seed": seed, "fold": fold, "threshold": thr_v, "metrics_at_selected_threshold": metrics, "metrics_at_0_5": metrics_half, "n_eval_participants": len(ys_tv)})

        # --- E-T1+E-V3 equal ---
        fused_equal = 0.5 * pa_t + 0.5 * pb_v
        # inner fused for threshold
        ys_inner_tv, fused_inner_equal, _ = _inner_fused_probs({"text": inner_t1, "voice": inner_v}, 0.5, 0.5, "two")
        # threshold selection on inner fused
        prov = build_provenance(partition=PARTITION_INNER_VALIDATION, operation=OPERATION_OOF_PREDICTION, dataset="eatd", unit="participant", labels=ys_inner_tv, n_groups=len(ys_inner_tv), seeds=(seed,), source="fusion_equal", contributing_partitions=(PARTITION_INNER_TRAIN,))
        tsel = select_threshold(ys_inner_tv, fused_inner_equal, provenance=prov, policy=PRIMARY_POLICY)
        metrics = compute_evaluation_metrics(ys_tv, fused_equal, unit="participant", threshold=tsel.threshold)
        metrics_half = compute_evaluation_metrics(ys_tv, fused_equal, unit="participant", threshold=0.5)
        fusion_fold_rows["E-T1+E-V3 equal"].append({"repeat_seed": seed, "fold": fold, "threshold": float(tsel.threshold), "weight_text": 0.5, "metrics_at_selected_threshold": metrics, "metrics_at_0_5": metrics_half, "n_eval_participants": len(ys_tv)})

        # --- E-T1+E-V3 weighted ---
        w = _optimize_weight(inner_t1, inner_v)
        fused_w = w * pa_t + (1 - w) * pb_v
        ys_inner_w, fused_inner_w, _ = _inner_fused_probs({"text": inner_t1, "voice": inner_v}, w, 1 - w, "two")
        prov = build_provenance(partition=PARTITION_INNER_VALIDATION, operation=OPERATION_OOF_PREDICTION, dataset="eatd", unit="participant", labels=ys_inner_w, n_groups=len(ys_inner_w), seeds=(seed,), source="fusion_weighted", contributing_partitions=(PARTITION_INNER_TRAIN,))
        tsel = select_threshold(ys_inner_w, fused_inner_w, provenance=prov, policy=PRIMARY_POLICY)
        metrics = compute_evaluation_metrics(ys_tv, fused_w, unit="participant", threshold=tsel.threshold)
        metrics_half = compute_evaluation_metrics(ys_tv, fused_w, unit="participant", threshold=0.5)
        fusion_fold_rows["E-T1+E-V3 weighted"].append({"repeat_seed": seed, "fold": fold, "threshold": float(tsel.threshold), "weight_text": float(w), "metrics_at_selected_threshold": metrics, "metrics_at_0_5": metrics_half, "n_eval_participants": len(ys_tv)})

        # --- E-T2+E-V3 equal ---
        fused_t2v = 0.5 * pa_t2 + 0.5 * pb_v2
        ys_inner_t2v, fused_inner_t2v, _ = _inner_fused_probs({"text": inner_t2, "voice": inner_v}, 0.5, 0.5, "two")
        prov = build_provenance(partition=PARTITION_INNER_VALIDATION, operation=OPERATION_OOF_PREDICTION, dataset="eatd", unit="participant", labels=ys_inner_t2v, n_groups=len(ys_inner_t2v), seeds=(seed,), source="fusion_t2v", contributing_partitions=(PARTITION_INNER_TRAIN,))
        tsel = select_threshold(ys_inner_t2v, fused_inner_t2v, provenance=prov, policy=PRIMARY_POLICY)
        metrics = compute_evaluation_metrics(ys_t2v, fused_t2v, unit="participant", threshold=tsel.threshold)
        metrics_half = compute_evaluation_metrics(ys_t2v, fused_t2v, unit="participant", threshold=0.5)
        fusion_fold_rows["E-T2+E-V3 equal"].append({"repeat_seed": seed, "fold": fold, "threshold": float(tsel.threshold), "metrics_at_selected_threshold": metrics, "metrics_at_0_5": metrics_half, "n_eval_participants": len(ys_t2v)})

        # --- E-T1+E-T2+E-V3 equal ---
        # align three
        common3 = sorted(set(ids_t1) & set(ids_t2) & set(ids_v))
        m1 = {pid: p for pid, p in zip(ids_t1, probs_t1)}
        m2 = {pid: p for pid, p in zip(ids_t2, probs_t2)}
        mv = {pid: p for pid, p in zip(ids_v, probs_v)}
        my = {pid: lab for pid, lab in zip(ids_t1, labels_t1)}
        ys3 = np.array([my[pid] for pid in common3], dtype=float)
        fused3 = np.array([(m1[pid] + m2[pid] + mv[pid]) / 3.0 for pid in common3], dtype=float)
        # inner three
        ids_inner_common = sorted(set(inner_t1[2]) & set(inner_t2[2]) & set(inner_v[2]))
        map_i1 = {pid: p for pid, p in zip(inner_t1[2], inner_t1[1])}
        map_i2 = {pid: p for pid, p in zip(inner_t2[2], inner_t2[1])}
        map_iv = {pid: p for pid, p in zip(inner_v[2], inner_v[1])}
        map_iy = {pid: y for pid, y in zip(inner_t1[2], inner_t1[0])}
        ys_inner3 = np.array([map_iy[pid] for pid in ids_inner_common], dtype=float)
        fused_inner3 = np.array([(map_i1[pid] + map_i2[pid] + map_iv[pid]) / 3.0 for pid in ids_inner_common], dtype=float)
        prov = build_provenance(partition=PARTITION_INNER_VALIDATION, operation=OPERATION_OOF_PREDICTION, dataset="eatd", unit="participant", labels=ys_inner3, n_groups=len(ys_inner3), seeds=(seed,), source="fusion_three", contributing_partitions=(PARTITION_INNER_TRAIN,))
        tsel = select_threshold(ys_inner3, fused_inner3, provenance=prov, policy=PRIMARY_POLICY)
        metrics = compute_evaluation_metrics(ys3, fused3, unit="participant", threshold=tsel.threshold)
        metrics_half = compute_evaluation_metrics(ys3, fused3, unit="participant", threshold=0.5)
        fusion_fold_rows["E-T1+E-T2+E-V3 equal"].append({"repeat_seed": seed, "fold": fold, "threshold": float(tsel.threshold), "metrics_at_selected_threshold": metrics, "metrics_at_0_5": metrics_half, "n_eval_participants": len(ys3)})

print(f"Fusion outer loop done", flush=True)

# ---------------- Aggregate fusion metrics ----------------
def _agg_fusion(rows):
    def gm(name):
        vals = []
        for fr in rows:
            m = fr["metrics_at_selected_threshold"]["metrics"][name]
            if m.get("defined") and m.get("value") is not None:
                vals.append(float(m["value"]))
        return (round(float(np.mean(vals)), 4) if vals else None,
                round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None)
    roc, roc_sd = gm("roc_auc")
    pr, pr_sd = gm("pr_auc")
    bacc, _ = gm("balanced_accuracy")
    f1, _ = gm("macro_f1")
    mcc, _ = gm("mcc")
    brier, _ = gm("brier")
    return {
        "n_folds": len(rows),
        "mean_roc_auc_selected": roc,
        "sd_roc_auc_selected": roc_sd,
        "mean_pr_auc_selected": pr,
        "sd_pr_auc_selected": pr_sd,
        "mean_balanced_accuracy_selected": bacc,
        "mean_macro_f1_selected": f1,
        "mean_mcc_selected": mcc,
        "mean_brier_selected": brier,
        "mean_threshold": round(float(np.mean([fr["threshold"] for fr in rows])), 4),
    }

fusion_summary = {}
for v in fusion_variants:
    fusion_summary[v] = _agg_fusion(fusion_fold_rows[v])

for v in fusion_variants:
    safe = v.replace(" ", "_").replace("+", "plus").replace("/", "_")
    (OUT_DIR / f"fusion_{safe}_folds.json").write_text(json.dumps(fusion_fold_rows[v], indent=2, default=str), encoding="utf-8")

# complementarity already classified in error_analysis? finalize here
ea = error_analysis["E_T1_vs_E_V3"]
total = ea["both_correct"] + ea["text_correct_voice_incorrect"] + ea["voice_correct_text_incorrect"] + ea["both_incorrect"]
non_overlap_correct = ea["text_correct_voice_incorrect"] + ea["voice_correct_text_incorrect"]
if non_overlap_correct >= 0.15 * total and ea["probability_pearson_r"] < 0.7:
    complementarity = "PARTIALLY COMPLEMENTARY"
elif non_overlap_correct >= 0.25 * total:
    complementarity = "COMPLEMENTARY"
else:
    complementarity = "NO CLEAR COMPLEMENTARITY"

fusion_comparison = {
    "phase": "5E",
    "dataset": "eatd",
    "evaluation_unit": "participant",
    "n_common_participants_first_repeat": len(common_participants),
    "single_models_reference": {
        "E-T1": {"mean_roc_auc_selected": 0.6756, "mean_pr_auc_selected": 0.5282},
        "E-T2": {"mean_roc_auc_selected": 0.5857, "mean_pr_auc_selected": 0.4609},
        "E-V3": {"mean_roc_auc_selected": 0.6315, "mean_pr_auc_selected": 0.4868},
    },
    "fusion_variants": fusion_summary,
    "complementarity_classification": complementarity,
    "error_analysis_reference": "results/phase5/eatd_fusion/error_analysis.json",
    "known_limitations": [
        "Development pool only; locked test not used.",
        "Fusion is participant-level late fusion; weights selected on inner OOF.",
        "No fusion model trained on in-sample base predictions.",
        "Complementarity based on first-repeat OOF and fold-level proxies.",
    ],
}
(OUT_DIR / "fusion_comparison.json").write_text(json.dumps(fusion_comparison, indent=2), encoding="utf-8")

report = {
    "phase": "5E",
    "experiment_group": "EATD text/voice paired analysis and late fusion",
    "status": "measured",
    "evaluation_unit": "participant",
    "locked_test_used": False,
    "n_development_participants": 138,
    "common_participants_first_repeat": len(common_participants),
    "error_analysis": error_analysis,
    "fusion_summary": fusion_summary,
    "complementarity_classification": complementarity,
    "artifacts": {
        "oof_predictions": str(OOF_DIR),
        "error_analysis": str(OUT_DIR / "error_analysis.json"),
        "fusion_comparison": str(OUT_DIR / "fusion_comparison.json"),
    },
}
(OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print("Phase 5E complete. Complementarity:", complementarity, flush=True)
print("Fusion summary:", json.dumps(fusion_summary, indent=2))
