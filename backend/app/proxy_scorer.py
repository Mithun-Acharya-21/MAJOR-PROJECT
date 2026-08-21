"""
Proxy Scoring — Meaningful demo predictions before full model fine-tuning.

Combines lexicon-based text analysis with lightweight image statistics
to produce calibrated depression-risk probabilities when neural weights
are untrained. Blends with model logits for smoother outputs.
"""

from __future__ import annotations

import io
import re
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageStat

# ---------------------------------------------------------------------------
# Text lexicons (research-inspired, not clinical)
# ---------------------------------------------------------------------------

_RISK_TERMS = {
    "hopeless", "worthless", "empty", "exhausted", "numb", "lonely",
    "anxious", "overwhelmed", "burden", "crying", "tired", "insomnia",
    "worthless", "suicidal", "depressed", "miserable", "helpless",
    "pointless", "failure", "hate", "pain", "sad", "worry", "stress",
    "cannot", "cant", "never", "nothing", "nobody", "alone", "trapped",
    "drowning", "breaking", "falling", "lost", "dark", "hurt",
}

_PROTECTIVE_TERMS = {
    "happy", "grateful", "excited", "joy", "love", "hope", "better",
    "progress", "thankful", "blessed", "amazing", "wonderful", "great",
    "good", "relief", "peace", "calm", "confident", "optimistic",
    "refreshed", "family", "friends", "support", "therapy", "healing",
}

_NEGATION = {"not", "no", "never", "neither", "nobody", "nothing", "dont", "doesnt", "didnt", "wont", "cant", "cannot"}


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def score_text(text: str) -> Tuple[float, dict[str, float]]:
    """Return depressed probability and per-word saliency hints."""
    words = _tokenize_words(text)
    if not words:
        return 0.5, {}

    risk_hits = 0.0
    protective_hits = 0.0
    word_scores: dict[str, float] = {}

    for i, word in enumerate(words):
        prev = words[i - 1] if i > 0 else ""
        negated = prev in _NEGATION

        score = 0.0
        if word in _RISK_TERMS:
            score = -0.85 if negated else 0.85
            risk_hits += abs(score)
        elif word in _PROTECTIVE_TERMS:
            score = 0.85 if negated else -0.55
            protective_hits += abs(score)

        if score != 0.0:
            word_scores[word] = max(word_scores.get(word, 0.0), abs(score))

    # Length-normalized signal
    raw = (risk_hits - protective_hits * 0.6) / max(len(words), 1)
    # Sigmoid-like mapping to [0.08, 0.92]
    depressed_prob = float(1.0 / (1.0 + np.exp(-raw * 4.0)))
    depressed_prob = 0.08 + depressed_prob * 0.84

    return depressed_prob, word_scores


def score_image(image_bytes: bytes) -> float:
    """Estimate affect proxy from brightness / contrast (demo heuristic)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img = img.resize((64, 64))
        stat = ImageStat.Stat(img)
        mean_brightness = stat.mean[0] / 255.0
        std_dev = stat.stddev[0] / 128.0

        # Lower brightness + lower contrast → slightly elevated risk signal
        risk = 0.35 + (1.0 - mean_brightness) * 0.25 + max(0.0, 0.3 - std_dev) * 0.2
        return float(max(0.1, min(0.75, risk)))
    except Exception:
        return 0.45


def blend_probabilities(
    model_probs: list[float],
    text: Optional[str],
    image_bytes: Optional[bytes],
    modalities: list[str],
    blend_weight: float = 0.65,
) -> Tuple[list[float], float]:
    """Blend neural softmax with proxy heuristics for demo-quality outputs."""
    text_prob = 0.5
    vision_prob = 0.5

    if "text" in modalities and text:
        text_prob, _ = score_text(text)

    if "image" in modalities and image_bytes:
        vision_prob = score_image(image_bytes)

    if len(modalities) == 2:
        proxy_depressed = 0.55 * text_prob + 0.45 * vision_prob
    elif "text" in modalities:
        proxy_depressed = text_prob
    else:
        proxy_depressed = vision_prob

    proxy_probs = [1.0 - proxy_depressed, proxy_depressed]

    blended = [
        blend_weight * proxy_probs[i] + (1.0 - blend_weight) * model_probs[i]
        for i in range(2)
    ]
    total = sum(blended)
    if total > 0:
        blended = [b / total for b in blended]

    return blended, proxy_depressed
