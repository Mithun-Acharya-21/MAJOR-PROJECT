"""
Multimodal Fusion Layer — Cross-Attention Fusion

Takes text embeddings (256-d) and vision embeddings (256-d), applies
cross-attention, and produces a binary classification output.

Works gracefully with text-only or vision-only inputs (missing branch
gets a zero vector so the model still runs).

Exports:
  MultimodalFusionModel — nn.Module
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusionModel(nn.Module):
    """Cross-attention fusion of text and vision embeddings → 2-class logits.

    Input dimensions:
      text_features   : (B, 256)  or None
      vision_features : (B, 256)  or None

    At least one must be provided at inference time.
    """

    EMBED_DIM = 256

    def __init__(self, n_heads: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        D = self.EMBED_DIM

        # Cross-modal attention (both directions)
        self.cross_attn_tv = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_vt = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
        self.dropout = nn.Dropout(dropout)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(D, 2),
        )

    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits       (B, 2)
            probabilities (B, 2)
        """
        # Determine batch size
        if text_features is not None:
            B = text_features.shape[0]
            device = text_features.device
        elif vision_features is not None:
            B = vision_features.shape[0]
            device = vision_features.device
        else:
            raise ValueError("At least one of text_features or vision_features must be provided")

        D = self.EMBED_DIM

        # Zero-pad missing modalities
        if text_features is None:
            text_features = torch.zeros(B, D, device=device)
        if vision_features is None:
            vision_features = torch.zeros(B, D, device=device)

        # Reshape to sequence (B, 1, D) for attention
        t = text_features.unsqueeze(1)    # (B, 1, D)
        v = vision_features.unsqueeze(1)  # (B, 1, D)

        # Text attends to vision
        tv_out, _ = self.cross_attn_tv(query=t, key=v, value=v)
        t_fused = self.norm1(t + self.dropout(tv_out)).squeeze(1)  # (B, D)

        # Vision attends to text
        vt_out, _ = self.cross_attn_vt(query=v, key=t, value=t)
        v_fused = self.norm2(v + self.dropout(vt_out)).squeeze(1)  # (B, D)

        # Concatenate and classify
        combined = torch.cat([t_fused, v_fused], dim=1)  # (B, D*2)
        logits = self.classifier(combined)                # (B, 2)
        probs = F.softmax(logits, dim=1)

        return logits, probs
