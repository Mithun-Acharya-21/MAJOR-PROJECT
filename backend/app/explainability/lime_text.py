"""
LIME Text Explainer — explains the ACTUAL prediction function (Phase 17).

``explain_text_with_model`` runs LIME against ``predict_fn`` — the real
prediction callable (trained model head, or the labeled lexicon proxy when
no trained model is available). LIME output is MODEL FEATURE ATTRIBUTION:
it shows which words moved the prediction function's output. It is NOT a
causal explanation and NOT a claim that words caused anything.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

import numpy as np

logger = logging.getLogger("emotiondep.lime_text")


def explain_text_with_model(
    text: str,
    predict_fn: Callable[[List[str]], np.ndarray],
    num_features: int = 10,
    num_samples: int = 50,
) -> List[Dict[str, Any]]:
    """Explain ``predict_fn`` (N,2) → probabilities with LIME.

    Args:
        text: the text instance to explain.
        predict_fn: the ACTUAL batch prediction function taking a list of
            texts and returning an (N, 2) array of
            [P(non-depressed), P(depressed)].
        num_features / num_samples: LIME parameters.

    Returns:
        List of {feature, weight, direction, contribution} dicts.
    """
    try:
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=["Non-Depressed", "Depressed"])
        explanation = explainer.explain_instance(
            text,
            predict_fn,
            labels=(1,),
            num_features=min(num_features, max(1, len(text.split()))),
            num_samples=num_samples,
        )

        result = []
        for word, weight in explanation.as_list(label=1):
            result.append({
                "feature": word,
                "weight": round(float(abs(weight)), 4),
                "direction": "risk_increasing" if weight > 0 else "risk_decreasing",
                "contribution": round(float(weight), 4),
            })
        return result

    except ImportError:
        logger.debug("LIME not installed — no explanation generated")
        return []
    except Exception as e:
        logger.debug("LIME explanation failed: %s", e)
        return []
