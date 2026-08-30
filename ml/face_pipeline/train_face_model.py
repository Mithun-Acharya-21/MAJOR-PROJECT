"""
Train Face Model — Fine-tune ResNet-18 on facial expression data.

Usage:
    python -m src.face_pipeline.train_face_model --data-dir data/processed/face/train --epochs 5

This script trains only the vision branch of the multimodal pipeline.
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
from torch.utils.data import DataLoader, Dataset
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ml.face_pipeline.cnn_model import FaceCNN
from ml.face_pipeline.preprocessing import get_inference_transform
from ml.evaluation.metrics import compute_metrics
from ml.utils.config_loader import load_config
from ml.utils.logger import setup_logger
from ml.utils.seed import set_seed

logger = setup_logger("train_face")


class FaceDataset(Dataset):
    """Dataset for class-organized face images.

    Expects directory structure:
        root/
          depressed/  (label=1)
          non_depressed/ (label=0)
    """

    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []

        class_map = {"depressed": 1, "non_depressed": 0, "positive": 1, "negative": 0}
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_map.get(class_dir.name.lower())
            if label is None:
                try:
                    label = int(float(class_dir.name))
                except ValueError:
                    continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    self.samples.append((str(img_path), label))

        logger.info("Loaded %d face images from %s", len(self.samples), root_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def train_face_model(data_dir: str, config: dict | None = None) -> dict:
    """Train the face-only model.

    Args:
        data_dir: Directory with class-organized face images.
        config: Optional config dict.

    Returns:
        Dict with training history and metrics.
    """
    cfg = config or {}
    epochs = cfg.get("epochs", 5)
    batch_size = cfg.get("batch_size", 16)
    lr = cfg.get("learning_rate", 1e-3)
    val_split = cfg.get("val_split", 0.2)
    seed = cfg.get("seed", 42)
    checkpoint_dir = cfg.get("checkpoint_dir", "models/checkpoints/face_model")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load data
    transform = get_inference_transform()
    full_dataset = FaceDataset(data_dir, transform=transform)

    # Split
    n_val = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = FaceCNN(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(images)
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
        val_preds = []
        val_true = []
        val_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits, _ = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_true.extend(labels.tolist())
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
            torch.save(model.state_dict(), out_dir / "face_model.pt")
            (out_dir / "metrics.json").write_text(
                json.dumps(val_metrics, indent=2), encoding="utf-8"
            )

    return {"history": history, "metrics": entry}


def main():
    parser = argparse.ArgumentParser(description="Train face model")
    parser.add_argument("--data-dir", required=True, help="Path to face image directory")
    parser.add_argument("--config", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)
    config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    })

    result = train_face_model(args.data_dir, config)
    logger.info("Training complete. Best metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
