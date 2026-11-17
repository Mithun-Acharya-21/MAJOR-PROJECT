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


# ---------------- Pickle state for stage 2 ----------------
import pickle
STATE_PATH = Path("results/phase5/eatd_fusion/state.pkl")
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
state = {
    "units_t1": units_t1, "rows_by_unit_t1": rows_by_unit_t1,
    "units_t2": units_t2, "rows_by_unit_t2": rows_by_unit_t2, "ordered_uids_t2": ordered_uids_t2, "texts_t2": texts_t2,
    "units_v": units_v, "rows_by_unit_v": rows_by_unit_v,
    "outer_session_t1": outer_session_t1, "inner_session_t1": inner_session_t1, "outer_part_t1": outer_part_t1, "inner_part_t1": inner_part_t1, "thresholds_t1": thresholds_t1, "oof_records_t1": oof_records_t1,
    "outer_session_t2": outer_session_t2, "inner_session_t2": inner_session_t2, "outer_part_t2": outer_part_t2, "inner_part_t2": inner_part_t2, "thresholds_t2": thresholds_t2, "oof_records_t2": oof_records_t2,
    "outer_session_v": outer_session_v, "inner_session_v": inner_session_v, "outer_part_v": outer_part_v, "inner_part_v": inner_part_v, "thresholds_v": thresholds_v, "oof_records_v": oof_records_v,
    "embeddings_t2": embeddings_t2, "uid_to_idx_t2": uid_to_idx_t2,
}
with STATE_PATH.open("wb") as f:
    pickle.dump(state, f)
print("State pickled to", STATE_PATH, flush=True)
