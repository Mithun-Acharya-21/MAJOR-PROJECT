"""Phase 10 — Speech-emotion model and acoustic feature attribution.

Two complementary architectures are provided so they can be benchmarked
on the same frozen split:

1. CNN-BiLSTM (lightweight, no pretrained weights required)
   Input: (B, n_mfcc*3, T) MFCC feature sequence
   Conv blocks → BiLSTM → attention pooling → embedding + emotion head

2. WavLM/wav2vec fine-tune wrapper (pretrained, needs transformers)
   Input: (B, T_samples) raw waveform at 16 kHz
   Pretrained encoder → mean-pool → projection → embedding + emotion head

Both produce:
  - 7-class emotion logits + softmax probabilities
  - Fixed-size acoustic embedding for multimodal fusion (Phase 14)

Feature attribution uses permutation importance over acoustic summary
statistics (MFCC means/stds, pitch stats, RMS, ZCR, spectral centroid).
This is a model-observed attribution, not causal evidence.

IMPORTANT: all outputs carry emotion labels only.
           They must never be re-labelled as depression scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from training.voice.acoustic_features import AcousticFeatureExtractor

SPEECH_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SPEECH_EMOTION_TO_ID = {e: i for i, e in enumerate(SPEECH_EMOTIONS)}

DISCLAIMER = (
    "These are model-derived speech-emotion signals. "
    "Acoustic patterns are model-observed features, not clinical evidence, "
    "and must not be interpreted as indicators of a mental-health condition."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpeechEmotionConfig:
    model_type: str = "cnn_bilstm"   # "cnn_bilstm" | "wavlm"
    n_classes: int = 7
    n_mfcc_features: int = 120       # 40 MFCCs × 3 (+ Δ + ΔΔ)
    embedding_dim: int = 256
    cnn_channels: list = None        # filled in __post_init__
    bilstm_hidden: int = 128
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 30
    early_stopping_patience: int = 5
    weight_decay: float = 1e-4
    pretrained_model_name: str = "microsoft/wavlm-base"  # used if model_type == "wavlm"
    max_audio_length_s: float = 10.0  # clip/pad waveforms to this length
    sample_rate: int = 16000

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [64, 128, 256]


# ---------------------------------------------------------------------------
# CNN-BiLSTM model
# ---------------------------------------------------------------------------

class _ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CNNBiLSTMEmotionModel(nn.Module):
    """Lightweight CNN-BiLSTM for MFCC-based speech emotion recognition."""

    def __init__(self, config: SpeechEmotionConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.cnn_channels

        cnn_layers: list[nn.Module] = []
        in_ch = config.n_mfcc_features
        for out_ch in channels:
            cnn_layers.append(_ConvBlock1d(in_ch, out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        self.bilstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=config.bilstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.dropout > 0 else 0,
        )
        lstm_out = config.bilstm_hidden * 2
        self.attention = nn.Linear(lstm_out, 1)
        self.embedding_head = nn.Sequential(
            nn.Linear(lstm_out, config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(config.embedding_dim, config.n_classes)

    def forward(self, mfcc: Tensor) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        mfcc : (B, n_features, T) — MFCC + delta + delta-delta sequence

        Returns
        -------
        dict with logits, probs, embedding, attention_weights
        """
        x = self.cnn(mfcc)                           # (B, C, T')
        x = x.permute(0, 2, 1)                       # (B, T', C)
        lstm_out, _ = self.bilstm(x)                 # (B, T', H*2)
        attn_scores = self.attention(lstm_out)        # (B, T', 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T', 1)
        context = (attn_weights * lstm_out).sum(dim=1)    # (B, H*2)
        embedding = self.embedding_head(context)
        logits = self.classifier(embedding)
        probs = torch.softmax(logits, dim=1)
        return {
            "logits": logits,
            "probs": probs,
            "embedding": embedding,
            "attention_weights": attn_weights.squeeze(-1),
        }


# ---------------------------------------------------------------------------
# WavLM fine-tune wrapper
# ---------------------------------------------------------------------------

class WavLMEmotionModel(nn.Module):
    """Fine-tune a WavLM/wav2vec2 encoder for speech emotion classification."""

    def __init__(self, config: SpeechEmotionConfig) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("transformers is required for WavLM model") from exc

        self.config = config
        self.encoder = AutoModel.from_pretrained(config.pretrained_model_name)
        encoder_dim = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(config.embedding_dim, config.n_classes)

    def forward(self, waveform: Tensor, attention_mask: Optional[Tensor] = None) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        waveform       : (B, T_samples) float32 at 16 kHz
        attention_mask : (B, T_samples) optional
        """
        outputs = self.encoder(waveform, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state               # (B, T', H)
        embedding = self.projection(hidden.mean(dim=1))  # (B, D)
        logits = self.classifier(embedding)
        probs = torch.softmax(logits, dim=1)
        return {
            "logits": logits,
            "probs": probs,
            "embedding": embedding,
            "attention_weights": None,
        }

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Dataset — MFCC path
# ---------------------------------------------------------------------------

class MFCCDataset(Dataset):
    """CSV-backed dataset.  Columns: path (audio file), label (emotion string).

    Each item is a fixed-length MFCC tensor padded/truncated to max_frames.
    """

    def __init__(
        self,
        csv_path: Path,
        config: SpeechEmotionConfig,
        extractor: Optional[AcousticFeatureExtractor] = None,
    ) -> None:
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        if not rows:
            raise ValueError(f"{csv_path} is empty")
        if not {"path", "label"} <= set(rows[0]):
            raise ValueError(f"{csv_path} must have 'path' and 'label' columns")
        unknown = {r["label"] for r in rows if r["label"] not in SPEECH_EMOTION_TO_ID}
        if unknown:
            raise ValueError(f"Unknown emotion labels: {unknown}")
        self.rows = rows
        self.config = config
        self.extractor = extractor or AcousticFeatureExtractor(
            {"sample_rate": config.sample_rate}
        )
        # Max frames after CNN downsampling (3 MaxPool2 layers → /8)
        max_s = config.max_audio_length_s
        hop = 160  # 10 ms at 16 kHz
        self.max_frames = int(max_s * config.sample_rate / hop)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        row = self.rows[idx]
        feat = self.extractor.extract_from_path(Path(row["path"]))
        mfcc = feat.mfcc  # (n_mfcc*3, T) or None
        if mfcc is None:
            mfcc = np.zeros((self.config.n_mfcc_features, self.max_frames), dtype=np.float32)

        # Pad or truncate to fixed length
        T = mfcc.shape[1]
        if T >= self.max_frames:
            mfcc = mfcc[:, :self.max_frames]
        else:
            pad = self.max_frames - T
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")

        label = SPEECH_EMOTION_TO_ID[row["label"]]
        return torch.tensor(mfcc, dtype=torch.float32), label


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_speech_emotion(
    train_csv: Path,
    validation_csv: Path,
    output_dir: Path,
    config: Optional[SpeechEmotionConfig] = None,
    seed: int = 42,
) -> dict:
    """Train CNN-BiLSTM speech emotion model on emotion-labelled data."""
    from sklearn.metrics import accuracy_score, f1_score

    if config is None:
        config = SpeechEmotionConfig()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data = MFCCDataset(train_csv, config)
    val_data = MFCCDataset(validation_csv, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTMEmotionModel(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc, stale, best_epoch = -1.0, 0, 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        for mfcc_batch, labels in train_loader:
            mfcc_batch, labels = mfcc_batch.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(mfcc_batch)["logits"], labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        actual, predicted = [], []
        with torch.no_grad():
            for mfcc_batch, labels in val_loader:
                outputs = model(mfcc_batch.to(device))
                actual.extend(labels.tolist())
                predicted.extend(outputs["logits"].argmax(1).cpu().tolist())

        acc = accuracy_score(actual, predicted)
        if acc > best_acc:
            best_acc, stale, best_epoch = acc, 0, epoch
            torch.save(
                {"state_dict": model.state_dict(), "config": asdict(config)},
                output_dir / "best_speech_emotion.pt",
            )
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break

    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    result = {
        "disclaimer": DISCLAIMER,
        "best_validation_accuracy": round(best_acc, 4),
        "best_epoch": best_epoch,
        "macro_f1": round(macro_f1, 4),
        "device": str(device),
        "note": "Emotion labels only. Not a depression predictor.",
    }
    (output_dir / "speech-emotion-summary.json").write_text(
        json.dumps({"config": asdict(config), **result}, indent=2), encoding="utf-8"
    )
    return result


# ---------------------------------------------------------------------------
# Feature attribution — permutation importance
# ---------------------------------------------------------------------------

def acoustic_permutation_importance(
    model: CNNBiLSTMEmotionModel,
    mfcc_batch: Tensor,
    labels: Tensor,
    feature_names: Optional[list[str]] = None,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict:
    """Estimate feature importance by permuting MFCC coefficient rows.

    Each row of the MFCC matrix (one coefficient across all frames) is
    shuffled independently.  The drop in accuracy approximates that
    feature's contribution.

    This is a model-observed attribution signal, not causal proof.

    Returns
    -------
    dict: {
        "importances": {feature_name: mean_accuracy_drop},
        "baseline_accuracy": float,
        "disclaimer": str,
        "note": str,
    }
    """
    rng = np.random.default_rng(seed)
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        baseline_preds = model(mfcc_batch.to(device))["logits"].argmax(1).cpu()
    baseline_acc = float((baseline_preds == labels).float().mean())

    n_features = mfcc_batch.shape[1]
    if feature_names is None:
        feature_names = [f"mfcc_row_{i}" for i in range(n_features)]

    importances: dict[str, float] = {}
    mfcc_np = mfcc_batch.numpy().copy()

    for feat_idx in range(n_features):
        drops = []
        for _ in range(n_repeats):
            permuted = mfcc_np.copy()
            idx_perm = rng.permutation(permuted.shape[0])
            permuted[:, feat_idx, :] = permuted[idx_perm, feat_idx, :]
            t = torch.tensor(permuted, dtype=torch.float32)
            with torch.no_grad():
                preds = model(t.to(device))["logits"].argmax(1).cpu()
            perm_acc = float((preds == labels).float().mean())
            drops.append(baseline_acc - perm_acc)
        importances[feature_names[feat_idx]] = round(float(np.mean(drops)), 4)

    # Sort descending by importance
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    return {
        "disclaimer": (
            "Permutation importance shows which MFCC rows most affect model accuracy. "
            "This is a model-observed attribution signal, not causal evidence."
        ),
        "baseline_accuracy": round(baseline_acc, 4),
        "importances": importances,
        "note": "Emotion-model attribution only. Not depression evidence.",
    }


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_speech_emotion(
    model: CNNBiLSTMEmotionModel,
    mfcc_tensor: np.ndarray,
    device: Optional[torch.device] = None,
    max_frames: int = 1600,
) -> dict:
    """Run inference on a (n_features, T) MFCC numpy array.

    Returns emotion probabilities, predicted label, embedding, and disclaimer.
    """
    if device is None:
        device = torch.device("cpu")

    # Pad / truncate
    n_feat, T = mfcc_tensor.shape
    if T >= max_frames:
        mfcc_tensor = mfcc_tensor[:, :max_frames]
    else:
        mfcc_tensor = np.pad(mfcc_tensor, ((0, 0), (0, max_frames - T)), mode="constant")

    t = torch.tensor(mfcc_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    outputs = model(t)
    probs = outputs["probs"][0].cpu().numpy()
    idx = int(probs.argmax())
    return {
        "disclaimer": DISCLAIMER,
        "emotion_probabilities": {
            e: round(float(probs[i]), 4) for i, e in enumerate(SPEECH_EMOTIONS)
        },
        "predicted_emotion": SPEECH_EMOTIONS[idx],
        "confidence": round(float(probs[idx]), 4),
        "embedding": outputs["embedding"][0].cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Train CNN-BiLSTM speech emotion model.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    config = SpeechEmotionConfig(epochs=args.epochs, batch_size=args.batch_size)
    result = train_speech_emotion(args.train, args.validation, args.output_dir, config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
