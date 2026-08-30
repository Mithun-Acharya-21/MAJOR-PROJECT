"""
Feature Fusion — Combine text and vision embeddings.

Supports:
  - Concatenation fusion
  - Cross-attention fusion (primary)
  - Gated fusion

Exports:
  ConcatFusion — simple concatenation
  CrossAttentionFusion — bidirectional cross-attention
  GatedFusion — learnable gating mechanism
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Simple concatenation of text and vision embeddings.

    Concatenates the two embeddings and passes through a projection layer.
    """

    def __init__(self, embed_dim: int = 256, output_dim: int = 256) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([text_features, vision_features], dim=1)
        return self.projection(combined)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention fusion.

    Text attends to vision, vision attends to text, then concatenate.
    Handles missing modalities gracefully (zero vector).
    """

    def __init__(self, embed_dim: int = 256, n_heads: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.cross_attn_tv = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn_vt = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_features is not None:
            B = text_features.shape[0]
            device = text_features.device
        elif vision_features is not None:
            B = vision_features.shape[0]
            device = vision_features.device
        else:
            raise ValueError("At least one modality must be provided")

        D = self.embed_dim
        if text_features is None:
            text_features = torch.zeros(B, D, device=device)
        if vision_features is None:
            vision_features = torch.zeros(B, D, device=device)

        t = text_features.unsqueeze(1)
        v = vision_features.unsqueeze(1)

        tv_out, _ = self.cross_attn_tv(query=t, key=v, value=v)
        t_fused = self.norm1(t + self.dropout(tv_out)).squeeze(1)

        vt_out, _ = self.cross_attn_vt(query=v, key=t, value=t)
        v_fused = self.norm2(v + self.dropout(vt_out)).squeeze(1)

        return torch.cat([t_fused, v_fused], dim=1)


class GatedFusion(nn.Module):
    """Learnable gating mechanism for feature fusion.

    A gate network decides how much to weight each modality
    based on the input features.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([text_features, vision_features], dim=1)
        gate_values = self.gate(combined)  # (B, embed_dim)

        gated_text = gate_values * text_features
        gated_vision = (1 - gate_values) * vision_features

        return self.projection(gated_text + gated_vision)
