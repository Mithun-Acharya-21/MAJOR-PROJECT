"""
BiLSTM + Additive Attention — Sequential modelling on top of transformer embeddings.

Architecture:
  Transformer hidden states (B, T, H)
    → BiLSTM (2-layer, bidirectional)
    → Additive Attention pooling
    → 256-d text embedding

Exports:
  AdditiveAttentionPooling — attention mechanism
  BiLSTMAttention — full module (BiLSTM + attention + projection)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdditiveAttentionPooling(nn.Module):
    """Bahdanau-style additive attention over a sequence.

    Computes a learned weighted sum of hidden states, where the weight
    for each position is determined by a feed-forward network.

    Args:
        hidden_size: Dimension of input hidden states.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (B, T, H) — sequence of hidden states.
            mask:   (B, T) — 1 for valid tokens, 0 for padding.

        Returns:
            context:  (B, H) — weighted sum of hidden states.
            weights:  (B, T) — attention weights (before masking).
        """
        energy = self.score(torch.tanh(self.projection(hidden))).squeeze(-1)  # (B, T)
        energy = energy.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(energy, dim=1)  # (B, T)
        context = (weights.unsqueeze(-1) * hidden).sum(dim=1)  # (B, H)
        return context, weights


class BiLSTMAttention(nn.Module):
    """BiLSTM + Additive Attention for text feature extraction.

    Takes transformer hidden states and produces a fixed-size embedding
    by applying bidirectional LSTM layers followed by attention pooling.

    Args:
        input_size: Input feature dimension (must match transformer hidden size).
        hidden_size: LSTM hidden size per direction.
        num_layers: Number of LSTM layers.
        dropout: Dropout rate.
        projection_dim: Output embedding dimension.
    """

    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        projection_dim: int = 256,
    ) -> None:
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # BiLSTM output size = hidden_size * 2 (bidirectional)
        lstm_output_size = hidden_size * 2
        self.attention = AdditiveAttentionPooling(hidden_size=lstm_output_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(lstm_output_size, projection_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, H_in) — transformer output.
            attention_mask: (B, T) — 1=valid, 0=pad.

        Returns:
            embedding:    (B, projection_dim) — final text embedding.
            attn_weights: (B, T) — attention weights.
        """
        lstm_out, _ = self.bilstm(hidden_states)  # (B, T, hidden_size * 2)
        context, attn_weights = self.attention(lstm_out, attention_mask)
        embedding = self.projection(self.dropout(context))
        return embedding, attn_weights
