"""
Text Branch — DistilBERT + BiLSTM + Additive Attention

Architecture:
  DistilBERT (encoder, frozen by default)
    → BiLSTM (2-layer, bidirectional, hidden=256)
    → Additive attention pooling
    → 256-d text embedding

Exports:
  TextBranch   — nn.Module
  get_tokenizer — returns a DistilBERT fast tokenizer
  preprocess_text — basic text normalizer
"""

from __future__ import annotations

import re
import unicodedata

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

_MODEL_NAME = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Normalize text while preserving emotionally meaningful content."""
    if not text or not text.strip():
        return ""
    text = unicodedata.normalize("NFKC", text).strip()
    # Replace URLs with placeholder
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    # Replace @mentions
    text = re.sub(r"@\w+", "<USER>", text)
    # Normalize excess whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Additive attention pooling
# ---------------------------------------------------------------------------

class AdditiveAttentionPooling(nn.Module):
    """Compute a weighted context vector over a sequence using learned scores."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden : (B, T, H)
        mask   : (B, T)  — 1=valid, 0=pad
        returns: context (B, H), weights (B, T)
        """
        energy = self.score(torch.tanh(self.projection(hidden))).squeeze(-1)  # (B, T)
        energy = energy.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(energy, dim=1)          # (B, T)
        context = (weights.unsqueeze(-1) * hidden).sum(dim=1)  # (B, H)
        return context, weights


# ---------------------------------------------------------------------------
# Text Branch
# ---------------------------------------------------------------------------

class TextBranch(nn.Module):
    """DistilBERT → BiLSTM → Additive Attention → 256-d embedding."""

    EMBED_DIM = 256

    def __init__(self, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(_MODEL_NAME)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        enc_hidden = self.encoder.config.hidden_size  # 768
        self.bilstm = nn.LSTM(
            input_size=enc_hidden,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.attention = AdditiveAttentionPooling(hidden_size=256)  # BiLSTM out = 256
        self.dropout = nn.Dropout(0.3)
        self.projection = nn.Linear(256, self.EMBED_DIM)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding    (B, 256)
            attn_weights (B, T)
        """
        with torch.set_grad_enabled(not all(
            not p.requires_grad for p in self.encoder.parameters()
        )):
            enc_out = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state  # (B, T, 768)

        lstm_out, _ = self.bilstm(enc_out)            # (B, T, 256)
        context, attn_weights = self.attention(lstm_out, attention_mask)
        embedding = self.projection(self.dropout(context))  # (B, 256)
        return embedding, attn_weights


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    return _tokenizer
