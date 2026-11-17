"""
Experiment T2 — Transformer text model (frozen multilingual encoder + head).

Encoder decision (verified, not assumed):
    The EATD transcripts are Chinese (see data/eatd/*/*.txt). The existing
    ``TextBranch`` already uses ``distilbert/distilbert-base-multilingual-cased``
    (104 languages, cased) for exactly this reason. That is the right family and
    it is kept — no English-only DistilBERT, and no encoder swap without a
    language check. At ~135M parameters it fits a 6 GB GPU comfortably.

Two variants:
    T2a (primary)  frozen encoder → frozen [CLS]/mean-pooled sentence embedding
                   computed once on GPU, cached to disk, then a small
                   torch head (Dropout → Linear → ReLU → Linear) trained on
                   those embeddings. Since the encoder is frozen, this is
                   mathematically identical to re-encoding every epoch, and
                   it keeps the 3050 cool.
    T2b (optional) partial fine-tuning of the encoder's last 2 layers with a
                   low encoder LR (2e-5) and head LR (1e-4), AMP + gradient
                   accumulation. Run only when the frozen baseline has been
                   evaluated; reported separately so the two are never mixed.

Pooling: masked mean over token states (documented choice). The existing
BiLSTM+attention branch is still available through ``backend.app.models`` but
is not re-used here, so T1/T2 differ in representation only by the encoder.

Exports:
    FrozenTextHead, embed_texts, train_head, load_text_head,
    run_text_transformer, run_text_finetune
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from backend.app.models.text_branch import get_tokenizer, preprocess_text
from ml.phase2.dataset import load_sessions
from ml.phase2.evaluation import (
    aggregate_by_participant,
    evaluate_predictions,
    experiment_metadata,
    metric_bundle,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_ID = "eatd_text_transformer_v1"
FINETUNE_ID = "eatd_text_transformer_ft_v1"
MODEL_VERSION = "text-eatd-transformer-2.0.0"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "experiments" / EXPERIMENT_ID
FINETUNE_DIR = PROJECT_ROOT / "models" / "experiments" / FINETUNE_ID
RESULT_PATH = PROJECT_ROOT / "results" / "phase2" / "text" / f"{EXPERIMENT_ID}.json"
FINETUNE_RESULT_PATH = PROJECT_ROOT / "results" / "phase2" / "text" / f"{FINETUNE_ID}.json"

DEFAULT_ENCODER = "distilbert/distilbert-base-multilingual-cased"
POOLING = "masked_mean"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FrozenTextHead(nn.Module):
    """Dropout → Linear(h, hidden) → ReLU → Dropout → Linear(hidden, 1)."""

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Frozen encoder embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_texts(
    texts: Sequence[str],
    encoder_name: str = DEFAULT_ENCODER,
    max_seq_len: int = 128,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Masked-mean pooled sentence embeddings from a FROZEN encoder."""
    from transformers import AutoModel

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    out: List[np.ndarray] = []
    t0 = time.time()
    amp_enabled = use_amp and device.type == "cuda"
    for start in range(0, len(texts), batch_size):
        batch = [preprocess_text(t) or " " for t in texts[start : start + batch_size]]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            hidden = encoder(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        out.append(pooled.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(out, axis=0)
    stats = {
        "encoder": encoder_name,
        "encoder_frozen": True,
        "pooling": POOLING,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "embedding_dim": int(embeddings.shape[1]),
        "device": str(device),
        "amp": bool(amp_enabled),
        "encode_seconds": round(time.time() - t0, 2),
        "encode_seconds_per_session": round((time.time() - t0) / max(len(texts), 1), 4),
        "peak_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated() / 1e9, 3)
            if device.type == "cuda"
            else None
        ),
    }
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings, stats


# ---------------------------------------------------------------------------
# Head training
# ---------------------------------------------------------------------------

def _tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def train_head(
    X_train: np.ndarray,
    y_train: Sequence[int],
    X_val: np.ndarray,
    y_val: Sequence[int],
    groups_train: Sequence[str],
    groups_val: Sequence[str],
    seed: int = 42,
    epochs: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    hidden: int = 128,
    dropout: float = 0.3,
    patience: int = 30,
    device: Optional[torch.device] = None,
) -> Tuple[FrozenTextHead, List[dict], Dict[str, object]]:
    """Train the head on frozen embeddings; select on validation participant AUC."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    in_dim = int(X_train.shape[1])
    model = FrozenTextHead(in_dim, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    yt = np.asarray(y_train, dtype=np.float32)
    pos = float(yt.sum())
    neg = float(len(yt) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr = _tensor(X_train, device)
    ytr = _tensor(yt, device)
    Xva = _tensor(X_val, device)

    best_key = (-1.0, -1.0)
    best_state = None
    best_epoch = 0
    history: List[dict] = []
    patience_left = patience
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(yt)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            logits = model(Xtr[idx])
            loss = criterion(logits, ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        train_loss = total / n

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_loss = float(
            nn.functional.binary_cross_entropy_with_logits(
                val_logits, _tensor(np.asarray(y_val, dtype=np.float32), device)
            )
        )
        agg = aggregate_by_participant(groups_val, y_val, val_probs)
        pm = metric_bundle(agg["labels"], agg["probs"])
        auc = float(pm["roc_auc"]) if pm["roc_auc"] is not None else 0.0
        f1 = float(pm["f1"]) if pm["f1"] is not None else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_participant_roc_auc": round(auc, 4),
                "val_participant_f1": round(f1, 4),
            }
        )
        key = (auc, f1)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("head training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    info = {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_participant_roc_auc": round(best_key[0], 4),
        "optimizer": "AdamW",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "loss": "BCEWithLogitsLoss(pos_weight=neg/pos)",
        "batch_size": batch_size,
        "hidden": hidden,
        "dropout": dropout,
        "early_stopping_patience": patience,
        "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
        "selection_metric": "validation participant-level ROC-AUC (tie-break: participant F1)",
        "device": str(device),
    }
    return model, history, info


def head_probabilities(model: FrozenTextHead, X: np.ndarray, device=None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(_tensor(np.asarray(X, dtype=np.float32), device))
        return torch.sigmoid(logits).cpu().numpy().astype(float)


def load_text_head(artifact_dir: Path = ARTIFACT_DIR):
    """Load a saved Phase 2 text head (not used by the API server)."""
    artifact_dir = Path(artifact_dir)
    meta = _read_json(artifact_dir / "metadata.json")
    ckpt = torch.load(artifact_dir / "head.pt", map_location="cpu", weights_only=True)
    model = FrozenTextHead(
        in_dim=int(meta["config"]["embedding_dim"]),
        hidden=int(meta["config"]["head"]["hidden"]),
        dropout=float(meta["config"]["head"]["dropout"]),
    )
    model.load_state_dict(ckpt)
    model.eval()
    return model, meta


def _read_json(path: Path) -> Dict[str, object]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cache_path(encoder_name: str, max_seq_len: int) -> Path:
    tag = encoder_name.replace("/", "__").replace("-", "_")
    return ARTIFACT_DIR / f"embeddings_{tag}_len{max_seq_len}.npz"


def run_text_transformer(
    seed: int = 42,
    encoder_name: str = DEFAULT_ENCODER,
    max_seq_len: int = 128,
    embed_batch_size: int = 8,
    head_epochs: int = 200,
    head_lr: float = 1e-4,
    n_boot: int = 2000,
    write: bool = True,
    reuse_cache: bool = True,
) -> Dict[str, object]:
    """T2a: frozen encoder embeddings + small trained head."""
    t_start = time.time()
    train_rows = load_sessions("train", modality="text")
    val_rows = load_sessions("validation", modality="text")
    test_rows = load_sessions("test", modality="text")
    rows = list(train_rows) + list(val_rows) + list(test_rows)
    n_tr, n_va = len(train_rows), len(val_rows)

    cache = _cache_path(encoder_name, max_seq_len)
    if reuse_cache and cache.exists():
        data = np.load(cache, allow_pickle=False)
        if data["embeddings"].shape[0] != len(rows):
            raise RuntimeError("cached embeddings do not match the manifest row count")
        embeddings = data["embeddings"]
        encode_stats = _read_json(cache.with_suffix(".json"))
        encode_stats["cached"] = True
    else:
        embeddings, encode_stats = embed_texts(
            [r["text"] for r in rows],
            encoder_name=encoder_name,
            max_seq_len=max_seq_len,
            batch_size=embed_batch_size,
        )
        if write:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, embeddings=embeddings)
            write_json(cache.with_suffix(".json"), encode_stats)

    X_tr, X_va = embeddings[:n_tr], embeddings[n_tr : n_tr + n_va]
    X_te = embeddings[n_tr + n_va :]

    model, history, head_info = train_head(
        X_tr,
        [r["label"] for r in train_rows],
        X_va,
        [r["participant_label"] for r in val_rows],
        [r["participant_key"] for r in train_rows],
        [r["participant_key"] for r in val_rows],
        seed=seed,
        epochs=head_epochs,
        lr=head_lr,
    )
    val_probs = head_probabilities(model, X_va)
    t_infer = time.time()
    test_probs = head_probabilities(model, X_te)
    inference_seconds = round(time.time() - t_infer, 4)

    val_eval = evaluate_predictions(val_rows, val_probs, n_boot=n_boot, seed=seed)
    test_eval = evaluate_predictions(test_rows, test_probs, n_boot=n_boot, seed=seed)

    checkpoint_paths: List[Path] = []
    if write:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ARTIFACT_DIR / "head.pt")
        checkpoint_paths = [ARTIFACT_DIR / "head.pt"]
        write_json(
            ARTIFACT_DIR / "test_predictions.json",
            {
                "participant_key": [r["participant_key"] for r in test_rows],
                "topic": [r["topic"] for r in test_rows],
                "label": [r["participant_label"] for r in test_rows],
                "probability": [round(float(p), 6) for p in test_probs],
                "model_version": MODEL_VERSION,
            },
        )

    config = {
        "encoder_name": encoder_name,
        "encoder_frozen": True,
        "pooling": POOLING,
        "max_seq_len": max_seq_len,
        "embedding_dim": int(embeddings.shape[1]),
        "head": {"hidden": head_info["hidden"], "dropout": head_info["dropout"]},
        "classifier": "Dropout→Linear→ReLU→Dropout→Linear(1) trained on frozen embeddings",
        "head_training": head_info,
    }
    metadata = experiment_metadata(
        experiment_id=EXPERIMENT_ID,
        model_version=MODEL_VERSION,
        modality="text",
        config=config,
        training={
            "seed": seed,
            "embedding": encode_stats,
            "history": history,
            "fit_seconds": round(time.time() - t_start, 2),
            "inference_seconds_test": inference_seconds,
            "test_sessions": len(test_rows),
            "refit_on_train_plus_validation": False,
            "note": (
                "Head trained on TRAIN embeddings only; validation used for early "
                "stopping/selection; test scored once at the end."
            ),
        },
        validation_metrics=val_eval,
        test_metrics=test_eval,
        checkpoint_paths=checkpoint_paths,
        notes=[
            "Frozen multilingual encoder: no encoder weights were updated.",
            "Encoder embeddings are cached to disk; re-running with reuse_cache=True reproduces predictions exactly.",
            "Participant-level metrics aggregate session probabilities by mean.",
        ],
    )
    if write:
        write_json(ARTIFACT_DIR / "metadata.json", metadata)
        write_json(RESULT_PATH, metadata)
    return metadata


# ---------------------------------------------------------------------------
# T2b — optional encoder fine-tuning (last N layers)
# ---------------------------------------------------------------------------

def _encoder_layers(encoder: nn.Module):
    """Locate the transformer layer list (BERT/DistilBERT/DeBERTa layouts)."""
    for attr_path in (("encoder", "layer"), ("transformer", "layer"), ("layers",)):
        obj = encoder
        for part in attr_path:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(
        f"cannot locate transformer layers on {type(encoder).__name__}; "
        "fine-tuning needs an explicit layer path"
    )


class FineTuneTextModel(nn.Module):
    """Frozen lower encoder layers, trainable top layers, masked-mean pooling, head."""

    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        trainable_layers: int = 2,
        hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        layers = _encoder_layers(self.encoder)
        for layer in list(layers)[-trainable_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        import inspect

        self.accepts_token_type_ids = (
            "token_type_ids" in inspect.signature(self.encoder.forward).parameters
        )
        dim = int(self.encoder.config.hidden_size)
        self.head = FrozenTextHead(dim, hidden=hidden, dropout=dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # BERT tokenizers emit token_type_ids; DistilBERT does not accept them.
        if token_type_ids is not None and self.accepts_token_type_ids:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.head(pooled)


def _make_batches(n: int, batch_size: int, shuffle: bool, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    return [idx[i : i + batch_size].tolist() for i in range(0, n, batch_size)]


def run_text_finetune(
    seed: int = 42,
    encoder_name: str = DEFAULT_ENCODER,
    max_seq_len: int = 128,
    trainable_layers: int = 2,
    encoder_lr: float = 2e-5,
    head_lr: float = 1e-4,
    batch_size: int = 4,
    grad_accum: int = 2,
    epochs: int = 6,
    patience: int = 2,
    n_boot: int = 2000,
    write: bool = True,
) -> Dict[str, object]:
    """T2b: fine-tune the top encoder layers. Explicitly optional/exploratory."""
    from transformers import AutoTokenizer

    t_start = time.time()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_rows = load_sessions("train", modality="text")
    val_rows = load_sessions("validation", modality="text")
    test_rows = load_sessions("test", modality="text")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = FineTuneTextModel(
        encoder_name=encoder_name, trainable_layers=trainable_layers
    ).to(device)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=0.01,
    )
    n_pos = sum(r["label"] for r in train_rows)
    pos_weight = torch.tensor(
        [(len(train_rows) - n_pos) / max(n_pos, 1)], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    def encode(rows_batch):
        enc = tokenizer(
            [preprocess_text(r["text"]) or " " for r in rows_batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        return {k: v.to(device) for k, v in enc.items()}

    @torch.no_grad()
    def probs_for(rows_batch, bs: int = 8) -> np.ndarray:
        model.eval()
        out = []
        for start in range(0, len(rows_batch), bs):
            chunk = rows_batch[start : start + bs]
            enc = encode(chunk)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(**enc)
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out).astype(float)

    best_key = (-1.0, -1.0)
    best_state = None
    best_epoch = 0
    history: List[dict] = []
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        batches = _make_batches(len(train_rows), batch_size, True, seed + epoch)
        total = 0.0
        opt.zero_grad(set_to_none=True)
        for step, idx in enumerate(batches):
            batch = [train_rows[i] for i in idx]
            enc = encode(batch)
            y = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(**enc)
                loss = criterion(logits.float(), y) / grad_accum
            scaler.scale(loss).backward()
            total += float(loss.item()) * grad_accum * len(idx)
            if (step + 1) % grad_accum == 0 or step == len(batches) - 1:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        train_loss = total / max(len(train_rows), 1)

        val_probs = probs_for(val_rows)
        agg = aggregate_by_participant(
            [r["participant_key"] for r in val_rows],
            [r["participant_label"] for r in val_rows],
            val_probs,
        )
        pm = metric_bundle(agg["labels"], agg["probs"])
        auc = float(pm["roc_auc"]) if pm["roc_auc"] is not None else 0.0
        f1 = float(pm["f1"]) if pm["f1"] is not None else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_participant_roc_auc": round(auc, 4),
                "val_participant_f1": round(f1, 4),
            }
        )
        print(f"  [T2b] epoch {epoch}: loss={train_loss:.4f} val_part_auc={auc:.3f} f1={f1:.3f}",
              flush=True)
        if (auc, f1) > best_key:
            best_key = (auc, f1)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  [T2b] early stop at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("fine-tuning produced no checkpoint")
    model.load_state_dict(best_state)
    val_probs = probs_for(val_rows)
    t_infer = time.time()
    test_probs = probs_for(test_rows)
    inference_seconds = round(time.time() - t_infer, 4)

    val_eval = evaluate_predictions(val_rows, val_probs, n_boot=n_boot, seed=seed)
    test_eval = evaluate_predictions(test_rows, test_probs, n_boot=n_boot, seed=seed)

    peak_mem = (
        round(torch.cuda.max_memory_allocated() / 1e9, 3)
        if device.type == "cuda"
        else None
    )
    checkpoint_paths: List[Path] = []
    if write:
        FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
        # Save only trainable tensors (top encoder layers + head); the frozen
        # base is reloaded from the pinned Hugging Face encoder.
        n_layers = len(_encoder_layers(model.encoder))
        first_trainable = max(n_layers - trainable_layers, 0)
        layer_markers = tuple(f".layer.{i}." for i in range(first_trainable, n_layers))
        trainable = {
            k: v for k, v in model.state_dict().items()
            if k.startswith("head.") or any(m in k for m in layer_markers)
        }
        torch.save(trainable, FINETUNE_DIR / "trainable_state.pt")
        checkpoint_paths = [FINETUNE_DIR / "trainable_state.pt"]
        write_json(
            FINETUNE_DIR / "test_predictions.json",
            {
                "participant_key": [r["participant_key"] for r in test_rows],
                "topic": [r["topic"] for r in test_rows],
                "label": [r["participant_label"] for r in test_rows],
                "probability": [round(float(p), 6) for p in test_probs],
                "model_version": MODEL_VERSION + "-ft",
            },
        )

    metadata = experiment_metadata(
        experiment_id=FINETUNE_ID,
        model_version=MODEL_VERSION + "-ft",
        modality="text",
        config={
            "encoder_name": encoder_name,
            "trainable_encoder_layers": trainable_layers,
            "frozen_encoder_layers": "all but the last trainable_layers",
            "pooling": POOLING,
            "max_seq_len": max_seq_len,
            "head": "Dropout→Linear(hidden)→ReLU→Dropout→Linear(1)",
            "encoder_lr": encoder_lr,
            "head_lr": head_lr,
            "batch_size": batch_size,
            "gradient_accumulation": grad_accum,
            "amp": device.type == "cuda",
            "trainable_parameters": int(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            ),
        },
        training={
            "seed": seed,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "history": history,
            "fit_seconds": round(time.time() - t_start, 2),
            "inference_seconds_test": inference_seconds,
            "peak_gpu_memory_gb": peak_mem,
            "selection_metric": "validation participant-level ROC-AUC",
        },
        validation_metrics=val_eval,
        test_metrics=test_eval,
        checkpoint_paths=checkpoint_paths,
        notes=[
            "Exploratory fine-tuning of the top encoder layers; the frozen baseline (T2a) is the primary result.",
            "Only trainable tensors are saved; the frozen base is reloaded from the pinned Hugging Face encoder.",
        ],
    )
    if write:
        write_json(FINETUNE_DIR / "metadata.json", metadata)
        write_json(FINETUNE_RESULT_PATH, metadata)
    return metadata
