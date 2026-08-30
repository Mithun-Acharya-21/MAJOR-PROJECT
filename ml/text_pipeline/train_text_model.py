"""
Train Text Model — Fine-tune DeBERTa + BiLSTM + Attention on text data.

Usage:
    python -m src.text_pipeline.train_text_model --csv data/processed/text/train.csv --epochs 5

This script trains only the text branch of the multimodal pipeline.
For the full multimodal training, see src/fusion/train_fusion_model.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.text_pipeline.preprocessing import preprocess_text
from ml.text_pipeline.tokenizer import get_tokenizer
from ml.text_pipeline.deberta_encoder import DeBERTaEncoder
from ml.text_pipeline.bilstm_attention import BiLSTMAttention
from ml.evaluation.metrics import compute_metrics
from ml.utils.config_loader import load_config
from ml.utils.logger import setup_logger
from ml.utils.seed import set_seed


logger = setup_logger("train_text")


class TextClassifier(nn.Module):
    """DeBERTa + BiLSTM + Attention + Classification Head."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = DeBERTaEncoder(freeze=True)
        self.feature_extractor = BiLSTMAttention(
            input_size=self.encoder.hidden_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            projection_dim=256,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)
        embedding, attn_weights = self.feature_extractor(hidden, attention_mask)
        logits = self.classifier(embedding)
        return logits, attn_weights


def train_text_model(
    csv_path: str,
    config: dict | None = None,
) -> dict:
    """Train the text-only model on a CSV dataset.

    Args:
        csv_path: Path to CSV with 'text' and 'label' columns.
        config: Optional config override dict.

    Returns:
        Dict with training history and final metrics.
    """
    import csv as csv_module

    cfg = config or {}
    epochs = cfg.get("epochs", 5)
    batch_size = cfg.get("batch_size", 8)
    lr = cfg.get("learning_rate", 2e-4)
    max_length = cfg.get("max_seq_length", 128)
    val_split = cfg.get("val_split", 0.2)
    seed = cfg.get("seed", 42)
    checkpoint_dir = cfg.get("checkpoint_dir", "models/checkpoints/text_model")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load data
    p = Path(csv_path)
    with open(p, newline="", encoding="utf-8-sig") as f:
        rows = list(csv_module.DictReader(f))

    texts = []
    labels = []
    for row in rows:
        text = preprocess_text(row.get("text", ""))
        label = int(float(row.get("label", 0)))
        if text.strip():
            texts.append(text)
            labels.append(label)

    logger.info("Loaded %d samples from %s", len(texts), csv_path)

    # Split
    n_val = max(1, int(len(texts) * val_split))
    train_texts, val_texts = texts[n_val:], texts[:n_val]
    train_labels, val_labels = labels[n_val:], labels[:n_val]

    # Tokenize
    tokenizer = get_tokenizer("deberta-v3")
    train_enc = tokenizer(
        train_texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    val_enc = tokenizer(
        val_texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt"
    )

    train_dataset = torch.utils.data.TensorDataset(
        train_enc["input_ids"], train_enc["attention_mask"],
        torch.tensor(train_labels, dtype=torch.long),
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_enc["input_ids"], val_enc["attention_mask"],
        torch.tensor(val_labels, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = TextClassifier().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = []
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for input_ids, attn_mask, labels_batch in train_loader:
            input_ids, attn_mask, labels_batch = (
                input_ids.to(device), attn_mask.to(device), labels_batch.to(device)
            )
            optimizer.zero_grad()
            logits, _ = model(input_ids, attn_mask)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels_batch)
            correct += (logits.argmax(1) == labels_batch).sum().item()
            total += len(labels_batch)

        train_acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        # Validation
        model.eval()
        val_preds = []
        val_true = []
        val_probs = []
        with torch.no_grad():
            for input_ids, attn_mask, labels_batch in val_loader:
                input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
                logits, _ = model(input_ids, attn_mask)
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_true.extend(labels_batch.tolist())
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
            torch.save(model.state_dict(), out_dir / "text_model.pt")
            (out_dir / "metrics.json").write_text(
                json.dumps(val_metrics, indent=2), encoding="utf-8"
            )

    return {"history": history, "metrics": entry}


def main():
    parser = argparse.ArgumentParser(description="Train text model")
    parser.add_argument("--csv", required=True, help="Path to text CSV")
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

    result = train_text_model(args.csv, config)
    logger.info("Training complete. Best metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
