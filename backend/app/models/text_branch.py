"""
Text Branch — Transformer → BiLSTM → Additive Attention → Classification Head

Architecture (Phase 4):
  Transformer encoder (frozen by default)
    → BiLSTM (2-layer, bidirectional, hidden=128)
    → Additive attention pooling
    → 256-d embedding
    → Classification head: Dropout → Linear(256,128) → ReLU → Dropout → Linear(128,1)
    → logit  →  sigmoid → P(depressed)

Two supported encoders:
  * ``distilbert-base-uncased``                              (English; legacy)
  * ``distilbert/distilbert-base-multilingual-cased``        (104 languages incl. Chinese — used for EATD, see ml/configs/text_model.yaml)

The classification head produces an ACTUAL learned logit — there is no
hard-coded probability anywhere in this module. A randomly initialized head
is NOT a valid predictor: ``ModelState`` only exposes model probabilities
when a TRAINED checkpoint containing ``classifier.*`` weights has been
loaded (see backend/app/models/text_classifier.py).

Exports:
  TextBranch        — nn.Module (embedding + attention + classification head)
  get_tokenizer     — cached tokenizer per encoder
  preprocess_text   — basic text normalizer (language-neutral)
  ENCODER_PRESETS   — documented encoder metadata
"""

from __future__ import annotations

import re
import unicodedata

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Documented encoder presets (Phase 3 language decision — see ml/configs/text_model.yaml)
ENCODER_PRESETS = {
    "distilbert-base-uncased": {
        "languages": "English",
        "note": "Legacy English encoder kept for English-input compatibility.",
    },
    "distilbert/distilbert-base-multilingual-cased": {
        "languages": "104 languages (incl. Chinese)",
        "note": "Used for the Chinese EATD dataset — multilingual cased distillation.",
    },
}

DEFAULT_ENCODER = "distilbert-base-uncased"
CLASSIFIER_HIDDEN = 128


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Normalize text while preserving emotionally meaningful content.

    Language-neutral: NFKC normalization, URL/mention masking, whitespace
    cleanup. No lowercasing (cased encoders) and no language-specific
    stemming or stopword removal.
    """
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
    """Transformer → BiLSTM → Additive Attention → embedding → classification head."""

    EMBED_DIM = 256

    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        freeze_encoder: bool = True,
        use_classifier_head: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        enc_hidden = self.encoder.config.hidden_size  # 768 for both presets
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

        # --- Phase 4: real classification head (learned logit, no constants) ---
        self.use_classifier_head = use_classifier_head
        if use_classifier_head:
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.EMBED_DIM, CLASSIFIER_HIDDEN),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(CLASSIFIER_HIDDEN, 1),  # single logit → sigmoid at inference
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Embedding-only forward (used by the cross-attention fusion model).

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

    def classify(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full classification forward.

        Returns:
            logits       (B, 1)  — raw learned logit (apply sigmoid for probability)
            embedding    (B, 256)
            attn_weights (B, T)
        """
        if not self.use_classifier_head:
            raise RuntimeError(
                "This TextBranch was built without a classification head "
                "(use_classifier_head=False)."
            )
        embedding, attn_weights = self.forward(input_ids, attention_mask)
        logits = self.classifier(embedding)           # (B, 1)
        return logits, embedding, attn_weights


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_tokenizers: dict[str, "AutoTokenizer"] = {}


def get_tokenizer(encoder_name: str = DEFAULT_ENCODER):
    """Return a cached tokenizer for the given encoder."""
    if encoder_name not in _tokenizers:
        _tokenizers[encoder_name] = AutoTokenizer.from_pretrained(encoder_name)
    return _tokenizers[encoder_name]
