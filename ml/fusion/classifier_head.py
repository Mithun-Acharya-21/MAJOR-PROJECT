"""
Classifier Head — Final classification layer on top of fused features.

Takes the fused embedding and produces binary depression / non-depression
classification logits.

Exports:
  ClassifierHead — nn.Module
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Binary classifier on fused multimodal features.

    Args:
        input_dim: Dimension of the fused feature vector.
        hidden_dim: Hidden layer size.
        num_classes: Number of output classes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_features: (B, input_dim) — fused embedding.

        Returns:
            logits: (B, num_classes) — raw classification scores.
        """
        return self.classifier(fused_features)
