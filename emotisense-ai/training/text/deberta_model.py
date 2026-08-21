"""DeBERTa-v3 encoder followed by BiLSTM and trainable attention pooling."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from torch import Tensor, nn
from transformers import AutoModel


@dataclass(frozen=True)
class DebertaTextConfig:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 256
    learning_rate: float = 2e-5
    batch_size: int = 8
    epochs: int = 5
    dropout: float = 0.2
    weight_decay: float = 0.01
    lstm_hidden_size: int = 256
    early_stopping_patience: int = 2
    gradient_clip_norm: float = 1.0
    warmup_ratio: float = 0.1
    use_bilstm: bool = True
    use_attention: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AdditiveAttentionPooling(nn.Module):
    """Mask-aware trainable attention. We expose weights as a diagnostic only."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, sequence: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.score(torch.tanh(self.projection(sequence))).squeeze(-1)
        logits = logits.masked_fill(attention_mask == 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        context = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)
        return context, weights


class ContextualSequenceHead(nn.Module):
    """Ablatable sequence head operating on contextual token embeddings."""

    def __init__(self, input_size: int, config: DebertaTextConfig) -> None:
        super().__init__()
        self.use_bilstm = config.use_bilstm
        self.use_attention = config.use_attention
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=input_size,
                hidden_size=config.lstm_hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.output_size = config.lstm_hidden_size * 2
        else:
            self.bilstm = None
            self.output_size = input_size
        self.attention = AdditiveAttentionPooling(self.output_size) if self.use_attention else None

    def forward(self, sequence: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.bilstm is not None:
            sequence, _ = self.bilstm(sequence)
        if self.attention is not None:
            return self.attention(sequence, attention_mask)
        weights = attention_mask.to(sequence.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.bmm(weights.unsqueeze(1), sequence).squeeze(1), None


class DebertaBiLSTMAttentionClassifier(nn.Module):
    """Configurable DeBERTa text classifier; labels denote dataset targets, never diagnoses."""

    def __init__(self, config: DebertaTextConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.model_name)
        encoder_hidden = self.encoder.config.hidden_size
        self.sequence_head = ContextualSequenceHead(encoder_hidden, config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.sequence_head.output_size, 2)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embedding, attention_weights = self.sequence_head(encoded, attention_mask)
        logits = self.classifier(self.dropout(embedding))
        return {"logits": logits, "embedding": embedding, "attention_weights": attention_weights}
