"""
Uncertainty Service — Real uncertainty estimation for screening predictions.

Method (documented, not fabricated):
  1. Predictive-entropy term: normalized Shannon entropy of the fused
     Bernoulli probability — high when the fused probability is near 0.5.
        H(p) = -p*log2(p) - (1-p)*log2(1-p)   (1 bit max)
  2. Epistemic (model-disagreement) term: cross-modality standard deviation
     of per-modality probabilities, scaled by the number of modalities used —
     when modalities disagree or few are available, the estimate is less
     certain.
  3. Evidence-strength term: mean confidence of the structured questionnaire
     features (when available) — low-confidence extracted features increase
     uncertainty.

    uncertainty = clip(0.5 * entropy + 0.3 * disagreement + 0.2 * (1 - evidence_confidence))

Interpretation bands: low (< 0.33), moderate (0.33–0.66), high (> 0.66).

PHASE 16 NOTES — what this IS and IS NOT:
  * This is MODEL/SCREENING UNCERTAINTY — a property of the screening
    pipeline's outputs. It is NEVER "clinical confidence" and must not be
    presented as a probability that the screening category is correct.
  * Component meanings:
      - predictive_entropy: normalized Shannon entropy of the fused
        Bernoulli probability — high when the fused probability is near 0.5
        (the decision itself is least informative there).
      - modality_disagreement: population std of per-modality probabilities
        (scaled) — proxies epistemic uncertainty: when independent signal
        sources disagree, the estimate is less certain. With a single
        modality a fixed 0.35 floor applies (irreducible at this sample).
      - evidence_confidence: mean confidence of the structured questionnaire
        features — weak lexical evidence increases uncertainty.
  * Distinct from, and complementary to, the interview-time
    ``interview_uncertainty`` (1 - mean feature confidence) used by the
    adaptive stopping logic (Phase 13/16): the stopping measure reflects
    how much evidence the interview has gathered so far; this measure
    reflects uncertainty of the final multimodal estimate. Both are
    documented; neither is clinical confidence.

Exports:
    compute_uncertainty      — main entry point
    uncertainty_label        — numeric uncertainty → label
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger("emotiondep.uncertainty")


def normalized_entropy(p: float) -> float:
    """Shannon entropy of a Bernoulli(p) in bits, normalized to 0..1."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    h = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
    return h / 1.0  # max entropy of Bernoulli is exactly 1 bit


def compute_uncertainty(
    fused_probability: float,
    modality_probabilities: Optional[Dict[str, float]] = None,
    feature_confidences: Optional[List[float]] = None,
) -> float:
    """Compute a real, reproducible uncertainty estimate in 0..1.

    Args:
        fused_probability: final fused P(depressed) in 0..1.
        modality_probabilities: per-modality P(depressed) in 0..1 (available ones only).
        feature_confidences: confidences of extracted structured features (0..1).
    """
    fused_probability = min(max(float(fused_probability), 0.0), 1.0)

    # 1. Predictive entropy (aleatoric proxy)
    entropy_term = normalized_entropy(fused_probability)

    # 2. Modality disagreement (epistemic proxy)
    disagreement_term = 0.0
    if modality_probabilities:
        probs = list(modality_probabilities.values())
        if len(probs) >= 2:
            mean = sum(probs) / len(probs)
            variance = sum((x - mean) ** 2 for x in probs) / len(probs)
            std = math.sqrt(variance)
            disagreement_term = min(1.0, std * 2.5)
        else:
            # A single modality carries irreducible epistemic uncertainty
            disagreement_term = 0.35

    # 3. Evidence strength from structured features
    evidence_term = 0.5
    if feature_confidences:
        evidence_term = sum(feature_confidences) / len(feature_confidences)

    uncertainty = 0.5 * entropy_term + 0.3 * disagreement_term + 0.2 * (1.0 - evidence_term)
    return round(min(max(uncertainty, 0.0), 1.0), 4)


def uncertainty_label(uncertainty: float) -> str:
    """Map numeric uncertainty to a human-readable band.

    Label semantics: model/screening uncertainty of the fused estimate —
    NOT clinical confidence.
    """
    if uncertainty < 0.33:
        return "Low"
    if uncertainty < 0.66:
        return "Moderate"
    return "High"


def uncertainty_breakdown(
    fused_probability: float,
    modality_probabilities: Optional[Dict[str, float]] = None,
    feature_confidences: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Return the uncertainty estimate with its components (for transparency)."""
    components: Dict[str, Any] = {
        "predictive_entropy": round(normalized_entropy(fused_probability), 4),
        "modality_disagreement": 0.0,
        "evidence_confidence": None,
    }
    if modality_probabilities:
        probs = list(modality_probabilities.values())
        if len(probs) >= 2:
            mean = sum(probs) / len(probs)
            variance = sum((x - mean) ** 2 for x in probs) / len(probs)
            components["modality_disagreement"] = round(min(1.0, math.sqrt(variance) * 2.5), 4)
        else:
            components["modality_disagreement"] = 0.35
    if feature_confidences:
        components["evidence_confidence"] = round(
            sum(feature_confidences) / len(feature_confidences), 4
        )
    return components


def track_history(
    session: Any,
    fused_probability: float,
    uncertainty: float,
) -> None:
    """Append to the session's prediction/uncertainty history (spec state)."""
    try:
        session.prediction_history.append(round(float(fused_probability), 4))
        session.uncertainty_history.append(round(float(uncertainty), 4))
    except Exception as e:  # non-fatal
        logger.debug("Could not track history: %s", e)
