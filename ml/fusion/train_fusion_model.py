"""
Train Fusion Model — End-to-end multimodal training.

Trains text branch + vision branch + fusion layer together on multimodal data.

Usage:
    python -m src.fusion.train_fusion_model --csv data/processed/multimodal.csv --epochs 5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.text_pipeline.preprocessing import preprocess_text
from ml.text_pipeline.tokenizer import get_tokenizer
from ml.text_pipeline.deberta_encoder import DeBERTaEncoder
from ml.text_pipeline.bilstm_attention import BiLSTMAttention
from ml.face_pipeline.cnn_model import FaceCNN
from ml.fusion.feature_fusion import CrossAttentionFusion
from ml.fusion.classifier_head import ClassifierHead
from ml.evaluation.metrics import compute_metrics
from ml.utils.config_loader import load_config
from ml.utils.logger import setup_logger
from ml.utils.seed import set_seed

logger = setup_logger("train_fusion")


class MultimodalDataset(Dataset):
    """Dataset for paired text + image + label."""

    def __init__(self, items: list[dict], tokenizer, max_length: int = 128):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # Tokenize text
        text = preprocess_text(item["text"])
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Load image
        from ml.face_pipeline.preprocessing import get_inference_transform
        transform = get_inference_transform()
        img = Image.open(item["image"]).convert("RGB")
        img_tensor = transform(img)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": img_tensor,
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


class MultimodalModel(nn.Module):
    """Full multimodal model: TextBranch + VisionBranch + Fusion + Classifier."""

    def __init__(self):
        super().__init__()
        self.text_encoder = DeBERTaEncoder(freeze=True)
        self.text_feature = BiLSTMAttention(
            input_size=self.text_encoder.hidden_size,
            hidden_size=128, num_layers=2, projection_dim=256,
        )
        self.vision_branch = FaceCNN(num_classes=2, freeze_backbone=True)
        self.fusion = CrossAttentionFusion(embed_dim=256)
        self.classifier = ClassifierHead(input_dim=512, hidden_dim=256)

    def forward(self, input_ids, attention_mask, image):
        text_hidden = self.text_encoder(input_ids, attention_mask)
        text_emb, _ = self.text_feature(text_hidden, attention_mask)
        _, vision_emb = self.vision_branch(image)
        fused = self.fusion(text_emb, vision_emb)
        logits = self.classifier(fused)
        return logits


def train_fusion_model(csv_path: str, config: dict | None = None) -> dict:
    """Train the full multimodal model."""
    cfg = config or {}
    epochs = cfg.get("epochs", 5)
    batch_size = cfg.get("batch_size", 8)
    lr = cfg.get("learning_rate", 2e-4)
    val_split = cfg.get("val_split", 0.2)
    seed = cfg.get("seed", 42)
    checkpoint_dir = cfg.get("checkpoint_dir", "models/checkpoints/fusion_model")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load CSV
    p = Path(csv_path)
    with open(p, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    items = []
    for row in rows:
        text = (row.get("text") or "").strip()
        img_path = (row.get("image") or "").strip()
        label = int(float(row.get("label", 0)))
        if text and img_path:
            items.append({"text": text, "image": str(p.parent / img_path), "label": label})

    logger.info("Loaded %d multimodal samples", len(items))

    # Split
    random.seed(seed)
    random.shuffle(items)
    n_val = max(1, int(len(items) * val_split))
    val_items, train_items = items[:n_val], items[n_val:]

    tokenizer = get_tokenizer("deberta-v3")
    train_ds = MultimodalDataset(train_items, tokenizer)
    val_ds = MultimodalDataset(val_items, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = MultimodalModel().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attn_mask, image)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        train_acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        # Validation
        model.eval()
        val_true, val_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                image = batch["image"].to(device)
                logits = model(input_ids, attn_mask, image)
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_true.extend(batch["label"].tolist())
                val_probs.extend(probs.cpu().tolist())

        val_metrics = compute_metrics(val_true, val_probs)
        entry = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 4),
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
        }
        history.append(entry)
        logger.info(
            "Epoch %d/%d — loss=%.4f acc=%.4f val_f1=%.4f val_auc=%.4f",
            epoch, epochs, avg_loss, train_acc,
            val_metrics["f1"], val_metrics["auc"],
        )

        if val_metrics["f1"] >= best_f1:
            best_f1 = val_metrics["f1"]
            out_dir = Path(checkpoint_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "fusion_model.pt")
            (out_dir / "metrics.json").write_text(
                json.dumps(val_metrics, indent=2), encoding="utf-8"
            )

    return {"history": history, "metrics": entry}


def main():
    parser = argparse.ArgumentParser(description="Train fusion model")
    parser.add_argument("--csv", required=True, help="Path to multimodal CSV")
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    })

    result = train_fusion_model(args.csv, config)
    logger.info("Training complete. Best metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
