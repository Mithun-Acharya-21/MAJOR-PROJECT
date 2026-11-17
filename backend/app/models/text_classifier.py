"""
Trained Text Classifier — loads the EATD-trained TextBranch checkpoint.

Honesty contract (research-integrity rules):
  * ``available`` is True ONLY when a trained checkpoint with classification
    weights exists on disk (models/final/text_model.pt).
  * If no checkpoint exists, ``available`` is False and callers MUST fall
    back to a clearly-labeled lexicon score — never to a fabricated
    probability.
  * Probabilities are sigmoid(logit) from the trained classification head.

Exports:
    TrainedTextClassifier
    load_text_classifier   — returns instance or None (no trained model)
    DEFAULT_MODEL_DIR
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from backend.app.models.text_branch import TextBranch, get_tokenizer, preprocess_text

logger = logging.getLogger("emotiondep.text_classifier")

# backend/app/models/text_classifier.py → backend/app/models → backend/app → backend → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "final" / "text_model"


class TrainedTextClassifier:
    """Wrapper around a trained TextBranch classification head."""

    def __init__(self, model_dir: Path, device: Optional[torch.device] = None) -> None:
        self.model_dir = Path(model_dir)
        self.device = device or torch.device("cpu")
        self.available = False

        weights_path = self.model_dir / "text_model.pt"
        meta_path = self.model_dir / "text_model_metadata.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"No trained text model weights at {weights_path}")

        self.metadata: Dict[str, Any] = {}
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        self.encoder_name = self.metadata.get("architecture", {}).get(
            "encoder_name", "distilbert-base-uncased"
        )
        self.max_seq_len = int(self.metadata.get("architecture", {}).get("max_seq_len", 128))
        self.model_version = self.metadata.get("model_version", "unknown")
        self.dataset_version = self.metadata.get("dataset_version", "unknown")

        self.model = TextBranch(
            encoder_name=self.encoder_name, freeze_encoder=True, use_classifier_head=True
        )
        state = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = get_tokenizer(self.encoder_name)
        self.available = True
        logger.info(
            "Trained text classifier loaded: version=%s dataset=%s encoder=%s",
            self.model_version, self.dataset_version, self.encoder_name,
        )

    # ------------------------------------------------------------------
    def _encode(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, float]:
        """text → {"logit", "probability"} from the trained head."""
        cleaned = preprocess_text(text)
        if not cleaned:
            return {"logit": 0.0, "probability": 0.5}
        input_ids, attention_mask = self._encode([cleaned])
        logits, _embedding, _attn = self.model.classify(input_ids, attention_mask)
        logit = float(logits.squeeze().item())
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        return {"logit": logit, "probability": prob}

    @torch.no_grad()
    def predict_proba_batch(self, texts: List[str]) -> np.ndarray:
        """Batch predict for LIME: (N, 2) array [[P(non-depressed), P(depressed)]]."""
        cleaned = [preprocess_text(t) if t and t.strip() else " " for t in texts]
        input_ids, attention_mask = self._encode(cleaned)
        logits, _emb, _attn = self.model.classify(input_ids, attention_mask)
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        probs = np.asarray(probs).reshape(-1)
        return np.stack([1.0 - probs, probs], axis=1)

    @torch.no_grad()
    def attention_explanation(self, text: str) -> List[Dict[str, Any]]:
        """Token attention weights from the trained model's own encoder."""
        cleaned = preprocess_text(text)
        if not cleaned:
            return []
        input_ids, attention_mask = self._encode([cleaned])
        _logits, _emb, attn = self.model.classify(input_ids, attention_mask)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        weights = attn.squeeze().cpu().tolist()
        if tokens and isinstance(weights[0], list):  # defensive: batch of 1
            weights = weights[0]
        return [
            {"token": t, "weight": round(float(w), 4)}
            for t, w in zip(tokens, weights)
            if t not in {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
        ]


def load_text_classifier(
    model_dir: Optional[Path] = None, device: Optional[torch.device] = None
) -> Optional[TrainedTextClassifier]:
    """Load the trained text classifier, or None when no checkpoint exists.

    Returning None is the HONEST signal: callers fall back to clearly-labeled
    heuristic scoring and must not pretend a trained model produced the result.
    """
    target = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    try:
        return TrainedTextClassifier(target, device=device)
    except FileNotFoundError:
        logger.info("No trained text classifier at %s — lexicon fallback will be labeled as such.", target)
        return None
    except Exception as e:  # corrupted checkpoint etc.
        logger.warning("Failed to load trained text classifier: %s", e)
        return None


_cached_classifier: Optional[TrainedTextClassifier] = None
_cached_checked = False


def get_text_classifier() -> Optional[TrainedTextClassifier]:
    """Process-wide cached accessor (loads the ~500 MB encoder at most once).

    Returns None when no trained checkpoint exists — callers must then use a
    clearly-labeled heuristic fallback.
    """
    global _cached_classifier, _cached_checked
    if not _cached_checked:
        _cached_checked = True
        _cached_classifier = load_text_classifier()
    return _cached_classifier
