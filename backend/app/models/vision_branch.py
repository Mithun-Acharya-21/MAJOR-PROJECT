"""
Vision Branch — ResNet-18 fine-tuned for facial expression features

Architecture:
  ResNet-18 (backbone, pretrained on ImageNet)
    → AdaptiveAvgPool2d
    → 256-d vision embedding

Grad-CAM uses the final convolutional layer (layer4).

Exports:
  VisionBranch     — nn.Module
  preprocess_image — bytes → (1, 3, 224, 224) tensor
  generate_gradcam — model + tensor → (224, 224) heatmap
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Decode image bytes and return a (1, 3, 224, 224) float tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _TRANSFORM(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Vision Branch
# ---------------------------------------------------------------------------

class VisionBranch(nn.Module):
    """ResNet-18 backbone → 256-d vision embedding."""

    EMBED_DIM = 256

    def __init__(self) -> None:
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Keep everything up to (and including) layer4 for Grad-CAM
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → embedding (B, 256)"""
        feat = self.backbone(x)          # (B, 512, 7, 7)
        pooled = self.pool(feat)         # (B, 512, 1, 1)
        return self.projection(pooled)   # (B, 256)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: VisionBranch,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for the given image tensor.

    Returns a (224, 224) float32 array in [0, 1].
    """
    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    def _save_activation(module, inp, out):
        activations.append(out.detach())

    def _save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Hook onto the last ResNet block inside backbone
    target_layer = model.backbone[-1]  # layer4
    h_fwd = target_layer.register_forward_hook(_save_activation)
    h_bwd = target_layer.register_full_backward_hook(_save_gradient)

    model.eval()
    image_tensor = image_tensor.requires_grad_(False)

    # Use a simple classifier head (fusion model's classification) or just
    # back-prop the sum of the embedding for demo purposes
    embedding = model(image_tensor)          # (1, 256)
    score = embedding.sum()
    model.zero_grad()
    score.backward()

    h_fwd.remove()
    h_bwd.remove()

    if not activations or not gradients:
        return np.zeros((224, 224), dtype=np.float32)

    act = activations[0].squeeze(0)       # (C, h, w)
    grad = gradients[0].squeeze(0)        # (C, h, w)
    weights = grad.mean(dim=(1, 2))       # (C,)
    cam = (weights[:, None, None] * act).sum(dim=0)
    cam = F.relu(cam)

    cam_np = cam.cpu().numpy()
    cam_up = np.array(
        Image.fromarray(cam_np).resize((224, 224), Image.BILINEAR)
    )
    mn, mx = cam_up.min(), cam_up.max()
    if mx > mn:
        cam_up = (cam_up - mn) / (mx - mn)
    else:
        cam_up = np.zeros_like(cam_up)

    return cam_up.astype(np.float32)
