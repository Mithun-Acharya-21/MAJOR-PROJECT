"""
DeBERTa Encoder — Pretrained transformer encoder for text features.

Extracts contextual embeddings from DeBERTa-v3-base for downstream
BiLSTM + Attention processing.

Exports:
  DeBERTaEncoder — nn.Module wrapper
  load_pretrained_encoder — factory function
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


_MODEL_NAME = "microsoft/deberta-v3-base"


class DeBERTaEncoder(nn.Module):
    """Thin wrapper around DeBERTa-v3-base for embedding extraction.

    Args:
        model_name: HuggingFace model identifier.
        freeze: Whether to freeze encoder parameters.
        pooling: Pooling strategy — 'cls', 'mean', or 'last'.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        freeze: bool = True,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (B, T) token IDs.
            attention_mask: (B, T) attention mask (1=valid, 0=pad).

        Returns:
            Hidden states (B, T, H) where H is the hidden dimension.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    @property
    def hidden_size(self) -> int:
        """Return the encoder's hidden dimension."""
        return self.encoder.config.hidden_size


def load_pretrained_encoder(
    model_name: str = _MODEL_NAME,
    freeze: bool = True,
) -> DeBERTaEncoder:
    """Factory function to create a DeBERTa encoder.

    Args:
        model_name: HuggingFace model identifier.
        freeze: Whether to freeze encoder weights.

    Returns:
        Initialized DeBERTaEncoder instance.
    """
    return DeBERTaEncoder(model_name=model_name, freeze=freeze)
