"""Lightweight CNN baseline for facial expression recognition (FER).

Architecture:  4-block Conv→BN→ReLU→MaxPool  →  FC → softmax.
Trained on emotion datasets (FER-2013 / RAF-DB / AffectNet) only.
Output labels are *emotion* classes, never depression labels.

This baseline is used to:
1. Establish a performance floor before transfer learning (Phase 8).
2. Provide face-embedding features for the multimodal fusion layer.

The model is intentionally small so it can run CPU-only for development.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Emotion labels (FER-2013 standard)
# ---------------------------------------------------------------------------

FER_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FER_EMOTION_TO_ID = {label: idx for idx, label in enumerate(FER_EMOTIONS)}

DISCLAIMER = (
    "These are model-derived facial-expression signals. "
    "Facial expression is an emotion indicator, not a depression label."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FERBaselineConfig:
    n_classes: int = 7
    image_size: int = 48            # FER-2013 default; 224 for larger datasets
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 20
    dropout: float = 0.3
    early_stopping_patience: int = 5
    embedding_dim: int = 256        # Output embedding size


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FERBaselineModel(nn.Module):
    """4-block CNN for facial expression classification.

    Produces both class logits and a fixed-size embedding for downstream fusion.
    """

    def __init__(self, config: FERBaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            ConvBlock(1, 32),   # → size/2
            ConvBlock(32, 64),  # → size/4
            ConvBlock(64, 128), # → size/8
            ConvBlock(128, 256, pool=False),  # keep spatial for Grad-CAM (Phase 8)
        )
        # Adaptive pool so the model accepts variable input sizes
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flat_size = 256 * 4 * 4
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(config.embedding_dim, config.n_classes)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Return logits, softmax probabilities, and embedding."""
        feature_maps = self.features(x)
        pooled = self.pool(feature_maps)
        embedding = self.embedding_head(pooled)
        logits = self.classifier(embedding)
        probs = torch.softmax(logits, dim=1)
        return {
            "logits": logits,
            "probs": probs,
            "embedding": embedding,
            "feature_maps": feature_maps,   # retained for Grad-CAM (Phase 8)
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FERDataset(Dataset):
    """CSV-backed dataset.  Expects columns: path, label (emotion string).

    Images are loaded as grayscale, resized to ``image_size × image_size``,
    and normalised to [0, 1].
    """

    def __init__(self, csv_path: Path, config: FERBaselineConfig) -> None:
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow is required for FERDataset. pip install Pillow")
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        if not rows:
            raise ValueError(f"{csv_path} is empty")
        if not {"path", "label"} <= set(rows[0]):
            raise ValueError(f"{csv_path} must contain 'path' and 'label' columns")
        unknown = {r["label"] for r in rows if r["label"] not in FER_EMOTION_TO_ID}
        if unknown:
            raise ValueError(f"Unknown emotion labels: {unknown}. Use: {FER_EMOTIONS}")
        self.rows = rows
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("L").resize(
            (self.config.image_size, self.config.image_size), Image.BILINEAR
        )
        tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255.0
        label = FER_EMOTION_TO_ID[row["label"]]
        return tensor, label


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train_fer_baseline(
    train_csv: Path,
    validation_csv: Path,
    output_dir: Path,
    config: Optional[FERBaselineConfig] = None,
    seed: int = 42,
) -> dict:
    """Train the FER baseline; return a results dict with validation metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    if config is None:
        config = FERBaselineConfig()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data = FERDataset(train_csv, config)
    val_data = FERDataset(validation_csv, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FERBaselineModel(config).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    best_acc, stale, best_epoch = -1.0, 0, 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images)["logits"], labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        actual, predicted = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device))
                actual.extend(labels.tolist())
                predicted.extend(outputs["logits"].argmax(1).cpu().tolist())

        acc = accuracy_score(actual, predicted)
        if acc > best_acc:
            best_acc, stale, best_epoch = acc, 0, epoch
            torch.save(
                {"state_dict": model.state_dict(), "config": asdict(config)},
                output_dir / "best_fer.pt",
            )
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break

    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    result = {
        "disclaimer": DISCLAIMER,
        "best_validation_accuracy": round(best_acc, 4),
        "best_epoch": best_epoch,
        "macro_f1": round(macro_f1, 4),
        "device": str(device),
        "note": "Emotion labels only. Not a depression predictor.",
    }
    (output_dir / "fer-baseline-summary.json").write_text(
        json.dumps({"config": asdict(config), **result}, indent=2), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# Inference helper (for downstream use)
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_expression(
    model: FERBaselineModel,
    face_tensor: np.ndarray,
    device: torch.device | None = None,
) -> dict:
    """Run inference on a (C, H, W) float32 numpy face tensor.

    Returns emotion probabilities, predicted label, embedding, and disclaimer.
    """
    if device is None:
        device = torch.device("cpu")
    # Grayscale: if C == 3 convert to 1 channel
    if face_tensor.shape[0] == 3:
        gray = 0.299 * face_tensor[0] + 0.587 * face_tensor[1] + 0.114 * face_tensor[2]
        face_tensor = gray[np.newaxis, ...]

    t = torch.tensor(face_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    outputs = model(t)
    probs = outputs["probs"][0].cpu().numpy()
    predicted_idx = int(probs.argmax())
    return {
        "disclaimer": DISCLAIMER,
        "emotion_probabilities": {
            emotion: round(float(probs[i]), 4) for i, emotion in enumerate(FER_EMOTIONS)
        },
        "predicted_emotion": FER_EMOTIONS[predicted_idx],
        "confidence": round(float(probs[predicted_idx]), 4),
        "embedding": outputs["embedding"][0].cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train FER CNN baseline (emotion only).")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=48)
    args = parser.parse_args()
    config = FERBaselineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    result = train_fer_baseline(args.train, args.validation, args.output_dir, config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
