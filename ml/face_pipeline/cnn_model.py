"""
CNN Model — Facial expression classifier for the vision pipeline.

Uses a ResNet-18 backbone pretrained on ImageNet, fine-tuned for
facial expression recognition (emotion classification).

Exports:
  FaceCNN — nn.Module for facial emotion classification
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FaceCNN(nn.Module):
    """ResNet-18 based facial expression classifier.

    Architecture:
      ResNet-18 backbone (ImageNet pretrained)
        → AdaptiveAvgPool2d
        → Dropout
        → Classification head

    Args:
        num_classes: Number of output classes (default: 2 — depressed/non-depressed).
        embed_dim: Dimension of the intermediate embedding (for fusion).
        dropout: Dropout rate.
        freeze_backbone: Whether to freeze ResNet backbone initially.
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Keep full backbone for Grad-CAM compatibility
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) — preprocessed face images.

        Returns:
            logits: (B, num_classes) — raw classification scores.
            embedding: (B, embed_dim) — feature embedding for fusion.
        """
        feat = self.backbone(x)          # (B, 512, 7, 7)
        pooled = self.pool(feat)         # (B, 512, 1, 1)
        embed = self.embedding(pooled)   # (B, embed_dim)
        logits = self.classifier(embed)  # (B, num_classes)
        return logits, embed

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding[1].out_features
