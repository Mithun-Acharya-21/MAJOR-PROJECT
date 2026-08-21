"""Phase 8 — Transfer-learning vision encoder and Grad-CAM explainability.

Architecture
------------
Pretrained backbone (MobileNetV2 or ResNet-18)
        ↓
Fine-tune final blocks on emotion dataset
        ↓
Replace classifier with emotion head (n_classes = 7)
        ↓
Extract penultimate feature map for Grad-CAM
        ↓
Project to fixed-size face embedding for multimodal fusion

Design constraints
------------------
- Labels are *emotion* classes only (never depression).
- The backbone's convolutional feature map is retained for Grad-CAM.
- Grad-CAM is labelled as an *emotion-model visualization*, not clinical evidence.
- The face embedding produced here feeds the fusion layer (Phase 14).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

from training.vision.fer_baseline import FER_EMOTIONS, FERDataset

DISCLAIMER = (
    "Grad-CAM highlights the image regions that influenced this emotion prediction. "
    "This is an emotion-model visualization. "
    "It is not clinical evidence and does not indicate a mental-health condition."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VisionTransferConfig:
    backbone: str = "mobilenet_v2"   # "mobilenet_v2" | "resnet18"
    n_classes: int = 7
    embedding_dim: int = 256
    image_size: int = 224
    learning_rate: float = 1e-4
    fine_tune_lr_multiplier: float = 0.1   # Lower LR for backbone layers
    batch_size: int = 32
    epochs: int = 15
    dropout: float = 0.4
    early_stopping_patience: int = 4
    freeze_backbone_epochs: int = 3        # Warm-up: freeze backbone, train head only
    weight_decay: float = 1e-4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TransferFaceEncoder(nn.Module):
    """Pretrained CNN backbone + emotion head + Grad-CAM hook.

    The model is RGB-input, so FER-48×48 grayscale images must be expanded
    to 3 channels before being passed in (done in the Dataset below).
    """

    def __init__(self, config: VisionTransferConfig) -> None:
        super().__init__()
        if not _TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required. pip install torchvision")

        self.config = config
        self._build_backbone(config.backbone)

        self.embedding_head = nn.Sequential(
            nn.Linear(self._backbone_out_features, config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(config.embedding_dim, config.n_classes)

        # Grad-CAM state
        self._gradcam_activations: Optional[Tensor] = None
        self._gradcam_gradients: Optional[Tensor] = None
        self._hook_handles: list = []

    # ------------------------------------------------------------------
    # Backbone construction
    # ------------------------------------------------------------------

    def _build_backbone(self, name: str) -> None:
        if name == "mobilenet_v2":
            base = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
            # Target layer for Grad-CAM: last conv block before classifier
            self._target_layer = base.features[-1]
            self.backbone_features = base.features
            self._backbone_out_features = base.last_channel  # 1280
        elif name == "resnet18":
            base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            self._target_layer = base.layer4
            # Rebuild as sequential for uniform feature extraction
            self.backbone_features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
            self._backbone_out_features = 512
        else:
            raise ValueError(f"Unsupported backbone: {name!r}. Use 'mobilenet_v2' or 'resnet18'.")

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        feature_map = self.backbone_features(x)     # (B, C, H, W)
        pooled = self.global_pool(feature_map).flatten(1)  # (B, C)
        embedding = self.embedding_head(pooled)
        logits = self.classifier(embedding)
        probs = torch.softmax(logits, dim=1)
        return {
            "logits": logits,
            "probs": probs,
            "embedding": embedding,
            "feature_map": feature_map,
        }

    # ------------------------------------------------------------------
    # Backbone freezing / unfreezing
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        for param in self.backbone_features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone_features.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Grad-CAM hooks
    # ------------------------------------------------------------------

    def register_gradcam_hooks(self) -> None:
        """Attach forward/backward hooks to the target layer."""
        self._remove_gradcam_hooks()

        def _save_activation(module: nn.Module, input: tuple, output: Tensor) -> None:
            self._gradcam_activations = output.detach()

        def _save_gradient(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            self._gradcam_gradients = grad_output[0].detach()

        h1 = self._target_layer.register_forward_hook(_save_activation)
        h2 = self._target_layer.register_full_backward_hook(_save_gradient)
        self._hook_handles = [h1, h2]

    def _remove_gradcam_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []


# ---------------------------------------------------------------------------
# Grad-CAM computation
# ---------------------------------------------------------------------------

def compute_gradcam(
    model: TransferFaceEncoder,
    face_tensor: Tensor,
    target_class: Optional[int] = None,
    image_size: tuple[int, int] = (224, 224),
) -> dict:
    """Compute a Grad-CAM heatmap for one face image.

    Parameters
    ----------
    model       : TransferFaceEncoder with hooks registered
    face_tensor : (1, C, H, W) float32 in [0, 1]
    target_class: class index to compute CAM for; defaults to argmax
    image_size  : output heatmap spatial size (H, W)

    Returns
    -------
    dict with keys:
      heatmap      – (H, W) float32 in [0, 1], normalized
      target_class – index used
      emotion_label – string label
      disclaimer   – mandatory safety note
    """
    model.register_gradcam_hooks()
    model.eval()

    face_tensor = face_tensor.requires_grad_(False)
    outputs = model(face_tensor)
    logits = outputs["logits"]

    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    # Backprop the score for the target class
    model.zero_grad()
    score = logits[0, target_class]
    score.backward()

    model._remove_gradcam_hooks()

    activations = model._gradcam_activations  # (1, C, h, w)
    gradients = model._gradcam_gradients      # (1, C, h, w)

    if activations is None or gradients is None:
        return {
            "heatmap": np.zeros(image_size, dtype=np.float32),
            "target_class": target_class,
            "emotion_label": FER_EMOTIONS[target_class],
            "disclaimer": DISCLAIMER,
            "error": "Grad-CAM hooks did not fire",
        }

    # Global-average-pool the gradients over spatial dims → weights
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
    cam = F.relu(cam)

    # Upsample to image size
    cam_up = F.interpolate(cam, size=image_size, mode="bilinear", align_corners=False)
    cam_np = cam_up[0, 0].cpu().numpy()

    # Normalize to [0, 1]
    cam_min, cam_max = cam_np.min(), cam_np.max()
    if cam_max > cam_min:
        cam_np = (cam_np - cam_min) / (cam_max - cam_min)
    else:
        cam_np = np.zeros_like(cam_np)

    return {
        "heatmap": cam_np.astype(np.float32),
        "target_class": target_class,
        "emotion_label": FER_EMOTIONS[target_class],
        "disclaimer": DISCLAIMER,
    }


def heatmap_to_overlay(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a [0,1] heatmap with an RGB image (H×W×3 uint8).

    Returns an RGB uint8 overlay for visualization.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for overlay. pip install opencv-python-headless") from exc

    h, w = original_image.shape[:2]
    heatmap_resized = (heatmap * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_resized, (w, h))
    color_map = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
    overlay = (alpha * color_map_rgb + (1 - alpha) * original_image).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_rgb_transform(image_size: int):
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required")
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),  # FER is grayscale; expand to RGB
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _make_val_transform(image_size: int):
    if not _TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required")
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class _RGBFERDataset(FERDataset):
    """FER dataset that applies a torchvision transform (RGB output)."""

    def __init__(self, csv_path: Path, config: VisionTransferConfig, transform) -> None:
        from training.vision.fer_baseline import FERBaselineConfig
        super().__init__(csv_path, FERBaselineConfig(
            image_size=config.image_size,
            n_classes=config.n_classes,
        ))
        self.transform = transform
        self._config_transfer = config

    def __getitem__(self, idx: int):
        try:
            from PIL import Image as PilImage
        except ImportError as exc:
            raise ImportError("Pillow required") from exc

        row = self.rows[idx]
        from training.vision.fer_baseline import FER_EMOTION_TO_ID
        img = PilImage.open(row["path"])
        tensor = self.transform(img)
        label = FER_EMOTION_TO_ID[row["label"]]
        return tensor, label


def train_transfer(
    train_csv: Path,
    validation_csv: Path,
    output_dir: Path,
    config: Optional[VisionTransferConfig] = None,
    seed: int = 42,
) -> dict:
    """Fine-tune a pretrained backbone on the emotion dataset.

    Returns a results dict; writes best checkpoint to output_dir/best_transfer.pt.
    """
    import random
    from sklearn.metrics import accuracy_score, f1_score

    if config is None:
        config = VisionTransferConfig()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_tf = _make_rgb_transform(config.image_size)
    val_tf = _make_val_transform(config.image_size)
    train_data = _RGBFERDataset(train_csv, config, train_tf)
    val_data = _RGBFERDataset(validation_csv, config, val_tf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransferFaceEncoder(config).to(device)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Param groups: lower LR for backbone
    backbone_params = list(model.backbone_features.parameters())
    head_params = list(model.embedding_head.parameters()) + list(model.classifier.parameters())
    optimizer = AdamW([
        {"params": backbone_params, "lr": config.learning_rate * config.fine_tune_lr_multiplier},
        {"params": head_params, "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.CrossEntropyLoss()

    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc, stale, best_epoch = -1.0, 0, 0

    # Warm-up: freeze backbone
    model.freeze_backbone()

    for epoch in range(1, config.epochs + 1):
        if epoch == config.freeze_backbone_epochs + 1:
            model.unfreeze_backbone()

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images)["logits"], labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

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
                output_dir / "best_transfer.pt",
            )
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break

    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    result = {
        "disclaimer": DISCLAIMER,
        "backbone": config.backbone,
        "best_validation_accuracy": round(best_acc, 4),
        "best_epoch": best_epoch,
        "macro_f1": round(macro_f1, 4),
        "device": str(device),
        "note": "Emotion labels only. Not a depression predictor.",
    }
    (output_dir / "transfer-summary.json").write_text(
        json.dumps({"config": asdict(config), **result}, indent=2), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Transfer-learn face emotion encoder (emotion only).")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=["mobilenet_v2", "resnet18"], default="mobilenet_v2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    config = VisionTransferConfig(
        backbone=args.backbone, epochs=args.epochs, image_size=args.image_size
    )
    result = train_transfer(args.train, args.validation, args.output_dir, config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
