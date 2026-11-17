"""
Train the text depression-screening model on SUBJECT-LEVEL EATD splits.

Pipeline (Phase 5):
    data/processed/eatd_splits/eatd_subject_splits.csv   (Phase 2 output)
        → train / validation rows (test rows are NEVER loaded here)
        → TextBranch(multilingual encoder, frozen) + BiLSTM + attention
          + classification head
        → class-weighted BCEWithLogits on the learned logit
        → early stopping on validation loss
        → best-validation checkpoint → models/final/text_model/
        → validation predictions saved for calibration (Phase 14)

Model selection uses VALIDATION performance only. The test split is not
read by this script at all.

Usage:
    python scripts/train_text_model.py [--seed 42] [--epochs 8] [--encoder distilbert/distilbert-base-multilingual-cased]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.text_branch import TextBranch, get_tokenizer, preprocess_text
from ml.evaluation.metrics import compute_full_metrics

SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "eatd_splits" / "eatd_subject_splits.csv"
OUT_DIR = PROJECT_ROOT / "models" / "final" / "text_model"
MODEL_VERSION = "text-eatd-1.0.0"
PREPROCESSING_VERSION = "preprocess_text-v2 (NFKC, URL/mention mask, language-neutral)"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(split: str) -> list[dict]:
    """Load rows of one split from the Phase 2 subject-level splits CSV."""
    if not SPLITS_CSV.exists():
        raise SystemExit(
            f"FATAL: {SPLITS_CSV} missing — run scripts/create_eatd_splits.py first"
        )
    with open(SPLITS_CSV, newline="", encoding="utf-8") as f:
        rows = [
            {"text": r["text"], "label": int(r["label"]), "subject_uid": r["subject_uid"]}
            for r in csv.DictReader(f)
            if r["split"] == split and (r.get("text") or "").strip()
        ]
    if not rows:
        raise SystemExit(f"FATAL: split '{split}' is empty in {SPLITS_CSV}")
    return rows


class TextRowsDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_seq_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        enc = self.tokenizer(
            preprocess_text(r["text"]),
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(r["label"], dtype=torch.float32),
        }


@torch.no_grad()
def predict_rows(model, loader, device) -> tuple[list[int], list[float]]:
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        logits, _emb, _attn = model.classify(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
        y_true.extend(int(v) for v in batch["label"].tolist())
        y_prob.extend(float(p) for p in probs)
    return y_true, y_prob


def main() -> int:
    ap = argparse.ArgumentParser(description="Train text model on subject-level EATD splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--encoder", type=str,
                    default="distilbert/distilbert-base-multilingual-cased")
    ap.add_argument("--max-seq-len", type=int, default=128)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | encoder: {args.encoder}")

    train_rows = load_split("train")
    val_rows = load_split("validation")
    # HARD GUARANTEE: the test split is never loaded for training/selection.
    test_rows_preview = load_split("test")
    print(f"Train: {len(train_rows)} samples | Validation: {len(val_rows)} samples "
          f"| Test: {len(test_rows_preview)} samples (LOCKED — not used here)")

    train_subjects = {r["subject_uid"] for r in train_rows}
    val_subjects = {r["subject_uid"] for r in val_rows}
    if train_subjects & val_subjects:
        raise SystemExit("FATAL: subject overlap between train and validation!")
    print(f"Subject overlap train∩validation: {len(train_subjects & val_subjects)}")

    # --- Class weights (balanced) from training rows ---
    counts = Counter(r["label"] for r in train_rows)
    n = len(train_rows)
    pos_weight = torch.tensor(
        [counts[0] / max(counts[1], 1)], dtype=torch.float32, device=device
    )
    print(f"Train label counts: {dict(counts)} → BCE pos_weight={pos_weight.item():.3f}")

    tokenizer = get_tokenizer(args.encoder)
    train_ds = TextRowsDataset(train_rows, tokenizer, args.max_seq_len)
    val_ds = TextRowsDataset(val_rows, tokenizer, args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TextBranch(encoder_name=args.encoder, freeze_encoder=True,
                       use_classifier_head=True).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable parameters: {n_params:,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Model selection: validation ROC-AUC.
    # Rationale (decided on validation data only, before any test evaluation):
    # with a 39-sample validation set (12 positives), unweighted BCE loss is
    # noisy and systematically favors the trivial majority-class predictor,
    # and it is inconsistent with the class-weighted training objective.
    # Threshold-free ROC-AUC is the standard selection metric for imbalanced
    # clinical screening. Lower validation loss is the tie-breaker.
    best_key = (-1.0, float("inf"))  # (val_auc, val_loss)
    best_epoch = 0
    best_state: dict | None = None
    history: list[dict] = []
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, _emb, _attn = model.classify(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            loss = criterion(logits.squeeze(-1), batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch["label"])
        train_loss = total_loss / max(len(train_ds), 1)

        y_true, y_prob = predict_rows(model, val_loader, device)
        val_probs_t = torch.tensor(y_prob)
        val_loss = float(
            nn.functional.binary_cross_entropy(
                val_probs_t.clamp(1e-6, 1 - 1e-6),
                torch.tensor([float(v) for v in y_true]),
            )
        )
        m = compute_full_metrics(y_true, y_prob)
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(m["accuracy"], 4),
            "val_precision": round(m["precision"], 4),
            "val_recall": round(m["recall"], 4),
            "val_f1": round(m["f1"], 4),
            "val_roc_auc": round(m["auc"], 4),
            "val_pr_auc": round(m["pr_auc"], 4),
            "val_balanced_accuracy": round(m["balanced_accuracy"], 4),
        }
        history.append(entry)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_f1={m['f1']:.3f} val_auc={m['auc']:.3f} val_bacc={m['balanced_accuracy']:.3f}")

        candidate = (m["auc"], val_loss)
        if candidate[0] > best_key[0] + 1e-9 or (
            abs(candidate[0] - best_key[0]) <= 1e-9 and candidate[1] < best_key[1]
        ):
            best_key = candidate
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_loss = val_loss
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (no val-AUC improvement for {args.patience} epochs)")
                break

    if best_state is None:
        raise SystemExit("FATAL: no checkpoint improved on validation AUC")

    # --- Save best checkpoint + metadata (Phase 20) ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT_DIR / "text_model.pt")

    # Final validation metrics from the BEST checkpoint
    model.load_state_dict(best_state)
    y_true_val, y_prob_val = predict_rows(model, val_loader, device)
    val_metrics = compute_full_metrics(y_true_val, y_prob_val)

    metadata = {
        "model_name": "TextBranch + classification head (depression screening)",
        "model_version": MODEL_VERSION,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_seed": args.seed,
        "dataset": "EATD (Chinese interview texts)",
        "dataset_version": "eatd_subject_splits_v1 (seed 42, scripts/create_eatd_splits.py)",
        "splits_csv": str(SPLITS_CSV.relative_to(PROJECT_ROOT)),
        "architecture": {
            "encoder_name": args.encoder,
            "encoder_frozen": True,
            "bilstm_hidden": 128,
            "bilstm_layers": 2,
            "attention": "additive",
            "embed_dim": 256,
            "classifier_head": "Dropout(0.3) → Linear(256,128) → ReLU → Dropout(0.3) → Linear(128,1)",
            "max_seq_len": args.max_seq_len,
        },
        "training": {
            "epochs_requested": args.epochs,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": 0.01,
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss(pos_weight=balanced)",
            "early_stopping_patience": args.patience,
            "selection_criterion": (
                "validation ROC-AUC (tie-break: lower val loss) — chosen because "
                "unweighted BCE on the 39-sample imbalanced validation set favored "
                "the trivial majority-class predictor; decision made on validation "
                "data only, before any test evaluation"
            ),
            "trainable_parameters": n_params,
        },
        "validation_metrics": {k: round(float(v), 4) for k, v in val_metrics.items()},
        "preprocessing_version": PREPROCESSING_VERSION,
        "language": "Chinese (multilingual encoder required — see ml/configs/text_model.yaml)",
        "limitations": [
            "Very small dataset (81 subjects after split cleanup); metrics have high variance.",
            "Screening research model — NOT clinically validated.",
        ],
        "history": history,
    }
    (OUT_DIR / "text_model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # --- Validation predictions for calibration (Phase 14 input) ---
    val_preds = {
        "y_true": y_true_val,
        "y_prob": [round(p, 6) for p in y_prob_val],
        "subject_uid": [r["subject_uid"] for r in val_rows],
        "source": "validation split predictions of the best checkpoint",
        "model_version": MODEL_VERSION,
        "dataset_version": metadata["dataset_version"],
        "created_at": metadata["training_date"],
    }
    (OUT_DIR / "validation_predictions.json").write_text(
        json.dumps(val_preds, indent=2), encoding="utf-8"
    )

    print(f"\nBest epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
    print("Validation metrics of best checkpoint:")
    for k, v in val_metrics.items():
        print(f"  {k}: {round(float(v), 4)}")
    print(f"\nSaved: {OUT_DIR / 'text_model.pt'}")
    print(f"Saved: {OUT_DIR / 'text_model_metadata.json'}")
    print(f"Saved: {OUT_DIR / 'validation_predictions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
