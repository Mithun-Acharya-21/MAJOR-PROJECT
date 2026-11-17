"""
Real Training Pipeline — Multimodal Early Depression Detection

Trains the EmotionDep models (TextBranch / VisionBranch /
MultimodalFusionModel) on user-provided local datasets:

  * Text datasets        — CSV with ``text`` and ``label`` columns
  * Image datasets       — class folder (one subfolder per class) or CSV
                           with ``image`` and ``label`` columns
  * Multimodal datasets  — CSV with ``text``, ``image`` and ``label``
                           columns

Labels may be numeric (0/1) or strings (depressed / non_depressed,
positive / negative, yes / no, ...). Image paths in CSVs are resolved
relative to the CSV file when not absolute.

The training loop is a standard PyTorch fine-tune: AdamW + CrossEntropy
over train/validation splits, with per-epoch metrics (loss, accuracy,
precision, recall, F1, AUC), best-checkpoint saving and baseline-metric
updates so the dashboard reflects real results.

Exports:
  TrainConfig, resolve_device, parse_label, prepare_data,
  build_models, set_trainable, train, evaluate, compute_metrics,
  save_checkpoint, apply_checkpoint, load_models_from_checkpoint,
  build_updated_baselines
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from backend.app.models.fusion_layer import MultimodalFusionModel
from backend.app.models.text_branch import TextBranch, get_tokenizer, preprocess_text
from backend.app.models.vision_branch import VisionBranch, preprocess_image

DEFAULT_CHECKPOINT_DIR = "checkpoints/latest"

# ---------------------------------------------------------------------------
# Label parsing — accept common numeric / string encodings
# ---------------------------------------------------------------------------

_LABEL_POS = {
    "1", "1.0", "positive", "pos", "depressed", "depression",
    "yes", "y", "true", "t", "case", "risk",
}
_LABEL_NEG = {
    "0", "0.0", "negative", "neg", "non_depressed", "nondepressed",
    "no", "n", "false", "f", "control", "controls", "healthy",
    "normal", "notdepressed",
}


def parse_label(value: Any) -> int:
    """Normalize a label value to ``1`` (depressed) or ``0`` (non-depressed)."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    s = str(value).strip().lower()
    s = s.replace("_", "").replace("-", "").replace(" ", "")
    if s in _LABEL_POS:
        return 1
    if s in _LABEL_NEG:
        return 0
    raise ValueError(
        f"Cannot interpret label {value!r} — expected 0/1, "
        "depressed/non_depressed, positive/negative, yes/no, ..."
    )


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    text_csv: Optional[str] = None
    image_dir: Optional[str] = None
    image_csv: Optional[str] = None
    multimodal_csv: Optional[str] = None
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 2e-4
    val_split: float = 0.2
    seed: int = 42
    unfreeze_encoder: bool = False
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR  # noqa: F841
    device: str = "auto"
    max_seq_len: int = 128


def resolve_device(device: str) -> torch.device:
    """Map ``auto`` to CUDA when available, else CPU."""
    if device in ("auto", None, ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def validate_config(cfg: TrainConfig) -> None:
    """Raise ValueError on invalid training configuration."""
    sources = [cfg.text_csv, cfg.image_dir, cfg.image_csv, cfg.multimodal_csv]
    provided = [s for s in sources if s]
    if not provided:
        raise ValueError(
            "Provide one dataset source: text_csv, image_dir/image_csv, or multimodal_csv"
        )
    if len(provided) > 1:
        raise ValueError("Provide exactly one dataset source per training run")
    if cfg.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not (0.0 < cfg.val_split < 1.0):
        raise ValueError("val_split must be between 0 and 1")


# ---------------------------------------------------------------------------
# Dataset loaders — return (items, modality, source_desc, label_map)
# ---------------------------------------------------------------------------

def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")
    with open(p, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV file is empty: {p}")
    return rows


def _find_col(row: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        for key in row:
            if key.strip().lower() == c:
                return key
    return None


def load_text_csv(path: str) -> Tuple[List[dict], str, str, Optional[Dict[str, int]]]:
    """Load a CSV with ``text`` and ``label`` columns."""
    rows = _read_csv_rows(path)
    text_col = _find_col(rows[0], ("text", "content", "sentence", "post", "tweet"))
    label_col = _find_col(rows[0], ("label", "class", "target", "depression"))
    if text_col is None or label_col is None:
        raise ValueError(
            f"Text CSV {path!r} must have 'text' and 'label' columns "
            f"(found: {list(rows[0].keys())})"
        )
    items = []
    for i, row in enumerate(rows, start=2):
        raw = (row.get(text_col) or "").strip()
        if not raw:
            continue
        items.append({"text": raw, "label": parse_label(row.get(label_col))})
    if not items:
        raise ValueError(f"No non-empty text rows found in {path!r}")
    source = f"Text CSV: {path} ({len(items)} samples)"
    return items, "text", source, None


def _resolve_image_path(base_dir: Path, raw: str) -> str:
    p = Path(raw)
    if not p.is_absolute():
        p = base_dir / p
    return str(p)


def load_image_csv(path: str) -> Tuple[List[dict], str, str, Optional[Dict[str, int]]]:
    """Load a CSV with ``image`` and ``label`` columns (paths relative to the CSV)."""
    rows = _read_csv_rows(path)
    img_col = _find_col(rows[0], ("image", "image_path", "path", "file", "filename"))
    label_col = _find_col(rows[0], ("label", "class", "target"))
    if img_col is None or label_col is None:
        raise ValueError(
            f"Image CSV {path!r} must have 'image' and 'label' columns "
            f"(found: {list(rows[0].keys())})"
        )
    base = Path(path).resolve().parent
    items = []
    for i, row in enumerate(rows, start=2):
        raw = (row.get(img_col) or "").strip()
        if not raw:
            continue
        items.append({"image": _resolve_image_path(base, raw), "label": parse_label(row.get(label_col))})
    if not items:
        raise ValueError(f"No image rows found in {path!r}")
    source = f"Image CSV: {path} ({len(items)} samples)"
    return items, "vision", source, None


def load_image_folder(path: str) -> Tuple[List[dict], str, str, Dict[str, int]]:
    """Load images from a directory with one subfolder per class."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Image folder not found: {root}")
    classes = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if len(classes) != 2:
        raise ValueError(
            f"Image folder {path!r} must contain exactly 2 class subfolders "
            f"(found {len(classes)}: {classes or 'none'})"
        )

    # Recognizable names map to the depression label; otherwise alphabetical order.
    label_map: Dict[str, int] = {}
    for name in classes:
        try:
            label_map[name] = parse_label(name)
        except ValueError:
            label_map[name] = None  # resolved below
    remaining = [name for name in classes if label_map[name] is None]
    for i, name in enumerate(sorted(remaining)):
        label_map[name] = i

    items = []
    for name in classes:
        cls_dir = root / name
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                continue
            items.append({"image": str(img), "label": label_map[name]})
    if not items:
        raise ValueError(f"No images found under {path!r}")
    source = f"Image folder: {path} ({len(items)} samples, classes={label_map})"
    return items, "vision", source, label_map


def load_multimodal_csv(path: str) -> Tuple[List[dict], str, str, Optional[Dict[str, int]]]:
    """Load a CSV with ``text``, ``image`` and ``label`` columns."""
    rows = _read_csv_rows(path)
    text_col = _find_col(rows[0], ("text", "content", "sentence", "post", "tweet"))
    img_col = _find_col(rows[0], ("image", "image_path", "path", "file", "filename"))
    label_col = _find_col(rows[0], ("label", "class", "target"))
    if not (text_col and img_col and label_col):
        raise ValueError(
            f"Multimodal CSV {path!r} must have 'text', 'image' and 'label' columns "
            f"(found: {list(rows[0].keys())})"
        )
    base = Path(path).resolve().parent
    items = []
    skipped = 0
    for i, row in enumerate(rows, start=2):
        raw_text = (row.get(text_col) or "").strip()
        raw_img = (row.get(img_col) or "").strip()
        if not raw_text or not raw_img:
            skipped += 1
            continue
        items.append({
            "text": raw_text,
            "image": _resolve_image_path(base, raw_img),
            "label": parse_label(row.get(label_col)),
        })
    if not items:
        raise ValueError(f"No complete text+image rows found in {path!r}")
    source = f"Multimodal CSV: {path} ({len(items)} samples"
    if skipped:
        source += f", {skipped} rows skipped — missing text/image"
    source += ")"
    return items, "multimodal", source, None


def prepare_data(cfg: TrainConfig) -> Tuple[List[dict], str, str, Optional[Dict[str, int]]]:
    """Load items for the configured dataset source.

    Returns:
        (items, modality, source_desc, label_map)
        modality is one of ``"text"``, ``"vision"``, ``"multimodal"``.
    """
    validate_config(cfg)
    if cfg.multimodal_csv:
        return load_multimodal_csv(cfg.multimodal_csv)
    if cfg.text_csv:
        return load_text_csv(cfg.text_csv)
    if cfg.image_dir:
        return load_image_folder(cfg.image_dir)
    return load_image_csv(cfg.image_csv)


# ---------------------------------------------------------------------------
# Torch datasets
# ---------------------------------------------------------------------------

def build_tokenize_fn(tokenizer, max_seq_len: int = 128) -> Callable[[str], Tuple[torch.Tensor, torch.Tensor]]:
    """Return ``fn(text) -> (input_ids (S,), attention_mask (S,))``."""

    def _fn(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = tokenizer(
            text,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    return _fn


def _load_image_tensor(image_path: str) -> torch.Tensor:
    """Load + preprocess an image file exactly like inference (face crop, 224×224)."""
    with open(image_path, "rb") as f:
        return preprocess_image(f.read()).squeeze(0)  # (3, 224, 224)


class TextDataset(Dataset):
    """Text-only samples: (input_ids, attention_mask, label)."""

    def __init__(self, items: List[dict], tokenize_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]]):
        self.items = items
        self.tokenize_fn = tokenize_fn

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        input_ids, attention_mask = self.tokenize_fn(preprocess_text(item["text"]))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


class ImageDataset(Dataset):
    """Vision-only samples: (image, label)."""

    def __init__(self, items: List[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return {
            "image": _load_image_tensor(item["image"]),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


class MultimodalDataset(Dataset):
    """Paired text+image samples: (input_ids, attention_mask, image, label)."""

    def __init__(self, items: List[dict], tokenize_fn: Callable[[str], Tuple[torch.Tensor, torch.Tensor]]):
        self.items = items
        self.tokenize_fn = tokenize_fn

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        input_ids, attention_mask = self.tokenize_fn(preprocess_text(item["text"]))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": _load_image_tensor(item["image"]),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

def build_models(modality: str, device: torch.device) -> Dict[str, Any]:
    """Build fresh pretrained models for the given modality.

    Returns a dict with ``text_branch`` / ``vision_branch`` / ``fusion_model``
    (unused branches are ``None``) plus ``tokenizer`` for text modalities.
    """
    models: Dict[str, Any] = {
        "text_branch": None,
        "vision_branch": None,
        "fusion_model": None,
        "tokenizer": None,
    }
    if modality in ("text", "multimodal"):
        text_branch = TextBranch().to(device)
        text_branch.eval()
        models["text_branch"] = text_branch
        models["tokenizer"] = get_tokenizer()
    if modality in ("vision", "multimodal"):
        vision_branch = VisionBranch().to(device)
        vision_branch.eval()
        models["vision_branch"] = vision_branch
    models["fusion_model"] = MultimodalFusionModel().to(device)
    models["fusion_model"].eval()
    return models


def set_trainable(models: Dict[str, Any], modality: str, unfreeze_encoder: bool) -> None:
    """Control which parameters receive gradients.

    Frozen encoders (DistilBERT / ResNet backbones) stay frozen unless
    ``unfreeze_encoder`` is set; projection / attention / fusion /
    classification heads are always trainable.
    """
    if unfreeze_encoder:
        tb = models.get("text_branch")
        if tb is not None:
            for p in tb.encoder.parameters():
                p.requires_grad = True
        vb = models.get("vision_branch")
        if vb is not None:
            for p in vb.backbone.parameters():
                p.requires_grad = True


def _model_map(models: Dict[str, Any]) -> Dict[str, nn.Module]:
    return {
        key: models[key]
        for key in ("text_branch", "vision_branch", "fusion_model")
        if models.get(key) is not None
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney / staircase estimator (handles ties)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    # NaN scores break the tie-scan below (NaN != NaN) — drop them.
    if np.isnan(y_score).any():
        mask = ~np.isnan(y_score)
        y_true = y_true[mask]
        y_score = y_score[mask]
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(-y_score, kind="mergesort")
    ys = y_true[order]
    scores = y_score[order]

    area = 0.0
    tpr = 0.0
    i = 0
    n = len(ys)
    while i < n:
        j = i
        while j < n and scores[j] == scores[i]:
            j += 1
        p = int(np.sum(ys[i:j]))
        neg = (j - i) - p
        area += (tpr + (p / n_pos) / 2.0) * (neg / n_neg)
        tpr += p / n_pos
        i = j
    return float(area)


def compute_metrics(y_true: Sequence[float], y_prob: Sequence[float]) -> Dict[str, float]:
    """Binary classification metrics from true labels and depressed probabilities."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": _auc(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Forward + evaluation helpers
# ---------------------------------------------------------------------------

def _forward(
    models: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    modality: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run branches + fusion for a batch; returns (logits, probabilities)."""
    if modality == "multimodal":
        text_features, _ = models["text_branch"](
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        vision_features = models["vision_branch"](batch["image"].to(device))
        return models["fusion_model"](text_features=text_features, vision_features=vision_features)
    if modality == "text":
        text_features, _ = models["text_branch"](
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        return models["fusion_model"](text_features=text_features)
    vision_features = models["vision_branch"](batch["image"].to(device))
    return models["fusion_model"](vision_features=vision_features)


@torch.no_grad()
def evaluate(
    models: Dict[str, Any],
    loader: DataLoader,
    modality: str,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[Dict[str, float], float]:
    """Compute validation metrics + average loss for a dataloader."""
    for m in _model_map(models).values():
        m.eval()

    total_loss = 0.0
    total = 0
    y_true: List[float] = []
    y_prob: List[float] = []
    for batch in loader:
        logits, probs = _forward(models, batch, modality, device)
        loss = criterion(logits, batch["label"].to(device))
        batch_n = len(batch["label"])
        total_loss += loss.item() * batch_n
        total += batch_n
        y_true.extend(batch["label"].tolist())
        y_prob.extend(probs[:, 1].tolist())

    metrics = compute_metrics(y_true, y_prob)
    avg_loss = total_loss / max(total, 1)
    return metrics, avg_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    cfg: TrainConfig,
    items: List[dict],
    modality: str,
    models: Dict[str, Any],
    device: torch.device,
    on_epoch: Optional[Callable[[dict], None]] = None,
    log: Optional[Callable[..., None]] = None,
    source_desc: str = "",
) -> Dict[str, Any]:
    """Run the fine-tuning loop and return history + best validation metrics.

    ``models`` must contain the branch(es) matching ``modality`` plus
    ``fusion_model``. Text modalities use ``models["tokenizer"]`` unless a
    ``tokenize_fn`` callable is provided (used by tests / custom pipelines).
    """
    log = log or (lambda *a, **k: None)
    if len(items) < 3:
        raise ValueError(f"Need at least 3 samples to train (got {len(items)})")

    # --- Train / validation split ---
    rng = random.Random(cfg.seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * cfg.val_split))
    n_val = min(n_val, len(shuffled) - 1)
    val_items, train_items = shuffled[:n_val], shuffled[n_val:]

    # --- Datasets / loaders ---
    tokenize_fn = models.get("tokenize_fn") or (
        build_tokenize_fn(models["tokenizer"], cfg.max_seq_len)
        if modality != "vision" else None
    )
    if modality == "text":
        train_ds = TextDataset(train_items, tokenize_fn)
        val_ds = TextDataset(val_items, tokenize_fn)
    elif modality == "vision":
        train_ds = ImageDataset(train_items)
        val_ds = ImageDataset(val_items)
    else:
        train_ds = MultimodalDataset(train_items, tokenize_fn)
        val_ds = MultimodalDataset(val_items, tokenize_fn)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # --- Optimizer ---
    set_trainable(models, modality, cfg.unfreeze_encoder)
    model_map = _model_map(models)
    params = [p for m in model_map.values() for p in m.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters — nothing to optimize")
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # --- Loop ---
    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    best_metrics: Optional[Dict[str, float]] = None

    for epoch in range(1, cfg.epochs + 1):
        for m in model_map.values():
            m.train()

        epoch_loss = 0.0
        correct = 0
        total = 0
        y_true: List[float] = []
        y_prob: List[float] = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits, probs = _forward(models, batch, modality, device)
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()

            batch_n = len(batch["label"])
            epoch_loss += loss.item() * batch_n
            total += batch_n
            correct += int((logits.argmax(dim=1) == batch["label"].to(device)).sum().item())
            y_true.extend(batch["label"].tolist())
            y_prob.extend(probs[:, 1].tolist())

        train_loss = epoch_loss / max(total, 1)
        train_metrics = compute_metrics(y_true, y_prob)
        val_metrics, val_loss = evaluate(models, val_loader, modality, device, criterion)

        entry = {
            "epoch": epoch,
            "loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
            "accuracy": round(float(val_metrics["accuracy"]), 4),
            "precision": round(float(val_metrics["precision"]), 4),
            "recall": round(float(val_metrics["recall"]), 4),
            "f1": round(float(val_metrics["f1"]), 4),
            "auc": round(float(val_metrics["auc"]), 4),
        }
        history.append(entry)
        log(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} (acc {train_metrics['accuracy']:.4f}) | "
            f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f}"
        )
        if on_epoch:
            on_epoch(entry)

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_metrics = dict(val_metrics)
            best_state = {
                key: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                for key, m in model_map.items()
            }

    if best_metrics is None or best_state is None:
        best_metrics = val_metrics
        best_state = {
            key: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            for key, m in model_map.items()
        }

    # --- Checkpoint ---
    checkpoint_dir = save_checkpoint(
        cfg, best_state, modality, best_metrics, history, source_desc
    )

    return {
        "history": history,
        "metrics": best_metrics,
        "modality": modality,
        "source": source_desc,
        "checkpoint_dir": checkpoint_dir,
        "epochs_run": cfg.epochs,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    cfg: TrainConfig,
    state: Dict[str, Dict[str, torch.Tensor]],
    modality: str,
    metrics: Dict[str, float],
    history: List[Dict[str, Any]],
    source_desc: str = "",
) -> str:
    """Persist best model weights + run metadata; returns the checkpoint dir."""
    out = Path(cfg.checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, sd in state.items():
        torch.save(sd, out / f"{key}.pt")

    metadata = {
        "modality": modality,
        "source": source_desc,
        "config": asdict(cfg),
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "history": history,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return str(out)


def apply_checkpoint(
    models: Dict[str, Any], checkpoint_dir: str, device: Optional[torch.device] = None
) -> List[str]:
    """Load state dicts from a checkpoint dir into an existing models dict."""
    d = Path(checkpoint_dir)
    loaded: List[str] = []
    for key in ("text_branch", "vision_branch", "fusion_model"):
        p = d / f"{key}.pt"
        if p.exists() and models.get(key) is not None:
            sd = torch.load(p, map_location="cpu")
            models[key].load_state_dict(sd)
            loaded.append(key)
    return loaded


def load_models_from_checkpoint(
    checkpoint_dir: str, device: torch.device
) -> Dict[str, Any]:
    """Rebuild models from a saved checkpoint (fresh instances + weights)."""
    d = Path(checkpoint_dir)
    if not (d / "fusion_model.pt").exists():
        raise FileNotFoundError(f"No trained checkpoint found in {d}")

    modality = "multimodal"
    meta_path = d / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        modality = meta.get("modality", "multimodal")

    models = build_models(modality, device)
    apply_checkpoint(models, str(d), device)
    for m in models.values():
        if isinstance(m, nn.Module):
            m.eval()
    return models


# ---------------------------------------------------------------------------
# Baseline updates — make trained results visible in the dashboard
# ---------------------------------------------------------------------------

def build_updated_baselines(
    eval_metrics: Dict[str, float], modality: str, source_desc: str
) -> Dict[str, Any]:
    """Return baseline benchmark data with the trained model's row replaced
    by real evaluation metrics."""
    from backend.app.baselines import get_baseline_metrics

    base = get_baseline_metrics()
    models = [dict(m) for m in base["models"]]

    if modality == "multimodal":
        idx = next((i for i, m in enumerate(models) if m["type"] == "multimodal"), None)
        name = "Multimodal Fusion (Trained)"
    elif modality == "text":
        idx = next((i for i, m in enumerate(models) if m["name"].startswith("DeBERTa")), None)
        name = "DeBERTa + BiLSTM (Text) — Trained"
    else:
        idx = next((i for i, m in enumerate(models) if m["type"] == "vision"), None)
        name = "ResNet (Vision Only) — Trained"

    if idx is not None:
        models[idx].update({
            "name": name,
            "accuracy": round(eval_metrics["accuracy"], 4),
            "precision": round(eval_metrics["precision"], 4),
            "recall": round(eval_metrics["recall"], 4),
            "f1": round(eval_metrics["f1"], 4),
            "auc": round(eval_metrics["auc"], 4),
        })

    return {
        **base,
        "models": models,
        "description": (
            "Benchmark comparison. The trained row reflects real validation "
            "metrics from the most recent fine-tuning run. Other rows are "
            "representative literature baselines."
        ),
        "notes": (
            f"Latest run: {source_desc or 'local dataset'}. "
            "Metrics are from the held-out validation split of the trained model."
        ),
    }
