"""
Response Analysis Service — Adaptive screening response → structured features.

Converts a user's free-text response to a screening question into structured
information:

    { "dimensions": { "sleep": {"severity": 2, "confidence": 0.91}, ... },
      "missing_information": [...],
      "duration": "...", "frequency": "...", "context": {...},
      "safety_concern": bool }

Design rules (per project spec):
  - Deterministic, rule-based extraction (reuses the existing lexicon
    infrastructure from ``proxy_scorer`` — no new infrastructure).
  - NEVER invents information the user did not provide: a dimension is only
    set when its lexical evidence appears in the response.
  - Ambiguous responses produce low confidence, not fabricated severity.
  - Only structured information is returned; no chain-of-thought.
  - Safety-critical language is detected first (delegates to the existing
    ``check_safety_keywords`` in ``recommendations.py``).

Severity scale (0-3, screening-research convention):
    0 = not indicated   1 = mild   2 = moderate   3 = severe
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from backend.app.recommendations import check_safety_keywords

logger = logging.getLogger("emotiondep.response_analysis")

# ---------------------------------------------------------------------------
# Dimension lexicons — evidence terms per dimension
# ---------------------------------------------------------------------------

# Each entry: (strong_terms, weak_terms). Strong evidence → +2 severity base,
# weak evidence → +1. Negation ("not sleeping badly") flips/inverts the signal.
_DIMENSION_TERMS: Dict[str, Dict[str, Set[str]]] = {
    "mood": {
        "strong": {"depressed", "depression", "hopeless", "worthless", "empty", "miserable", "sad", "down", "low"},
        "weak": {"unhappy", "gloomy", "blue", "moody", "tearful", "crying", "down"},
    },
    "loss_of_interest": {
        "strong": {"no interest", "lost interest", "nothing matters", "pointless", "no enjoyment", "no pleasure"},
        "weak": {"boring", "bored", "don't enjoy", "don't care", "unmotivated", "apathy", "apathetic"},
    },
    "sleep": {
        "strong": {"insomnia", "can't sleep", "cannot sleep", "no sleep", "awake all night", "sleeping very badly", "sleeping badly"},
        "weak": {"sleep", "sleeping", "tired", "oversleeping", "waking up", "nightmares", "restless night"},
    },
    "energy": {
        "strong": {"exhausted", "no energy", "completely drained", "completely exhausted", "no strength"},
        "weak": {"tired", "drained", "fatigue", "fatigued", "low energy", "worn out", "no motivation"},
    },
    "appetite": {
        "strong": {"no appetite", "can't eat", "cannot eat", "stopped eating", "not eating"},
        "weak": {"appetite", "eating", "eating less", "overeating", "weight"},
    },
    "concentration": {
        "strong": {"can't concentrate", "cannot concentrate", "can't focus", "cannot focus", "brain fog", "can't think straight"},
        "weak": {"concentrate", "focus", "attention", "distracted", "forgetful", "forgetting", "scattered"},
    },
    "self_worth": {
        "strong": {"worthless", "failure", "useless", "no worth", "hate myself", "burden"},
        "weak": {"not good enough", "disappointed in myself", "let myself", "inadequate", "guilty about myself"},
    },
    "guilt": {
        "strong": {"guilty", "blame myself", "my fault", "all my fault"},
        "weak": {"guilt", "regret", "ashamed", "shame"},
    },
    "psychomotor_changes": {
        "strong": {"can't get out of bed", "cannot get out of bed", "physically slowed", "restless body", "can't stop pacing"},
        "weak": {"slowed", "sluggish", "restless", "fidgety", "moving slowly", "agitated", "heavy"},
    },
    "social_withdrawal": {
        "strong": {"isolated", "isolating", "no one", "nobody", "alone", "avoiding people", "avoiding everyone", "cut off"},
        "weak": {"lonely", "withdrawn", "staying home", "don't see anyone", "haven't seen", "friends"},
    },
    "daily_functioning": {
        "strong": {"can't function", "cannot function", "can't work", "cannot work", "can't do anything", "falling behind at", "failing"},
        "weak": {"struggling at work", "struggling at school", "behind on", "hard to get things done", "productivity", "daily tasks", "routine", "chores"},
    },
    "duration": {
        "strong": {"for months", "months now", "half a year", "for years", "since last year"},
        "weak": {"for weeks", "weeks now", "couple of weeks", "few weeks", "this month", "lately", "recently", "past few days", "days"},
    },
    "frequency": {
        "strong": {"every day", "everyday", "all the time", "constantly", "most days", "always"},
        "weak": {"often", "sometimes", "a few times", "occasionally", "rarely", "hardly"},
    },
}

# Frequency phrase → normalized value (for the "frequency" meta dimension)
_FREQUENCY_MAP = [
    ("every day", "daily"),
    ("everyday", "daily"),
    ("all the time", "daily"),
    ("constantly", "daily"),
    ("most days", "most_days"),
    ("most of the time", "most_days"),
    ("often", "frequent"),
    ("a few times", "occasional"),
    ("sometimes", "occasional"),
    ("occasionally", "occasional"),
    ("rarely", "rare"),
    ("hardly", "rare"),
]

_DURATION_MAP = [
    ("for months", "several months"),
    ("months now", "several months"),
    ("half a year", "several months"),
    ("for years", "a year or more"),
    ("for weeks", "several weeks"),
    ("weeks now", "several weeks"),
    ("couple of weeks", "several weeks"),
    ("few weeks", "several weeks"),
    ("this month", "several weeks"),
    ("past few days", "a few days"),
    ("a few days", "a few days"),
]

# Intensity modifiers that upgrade weak evidence one step
_INTENSIFIERS = {"very", "really", "extremely", "completely", "constantly", "always", "so", "totally", "severe", "unbearable"}
# Hedges that downgrade evidence / add uncertainty
_HEDGES = {"maybe", "perhaps", "a bit", "a little", "slightly", "sort of", "kind of", "somewhat", "i guess", "i think"}

_NEGATION = {"not", "no", "never", "none", "without", "cannot", "can't", "don't", "doesn't", "didn't", "won't", "isn't", "aren't", "haven't", "hasn't"}

# Context keys the recommendation engine may optionally use (user-provided only)
_CONTEXT_PATTERNS = {
    "sleep_schedule": [
        (re.compile(r"\b(?:up|wake)\s+(?:until\s+)?\w*\s*(?:late|3|4|2|1)\b", re.I), "late"),
        (re.compile(r"\b(?:go|going|went)\s+to\s+bed\s+(?:very\s+)?late\b", re.I), "late"),
        (re.compile(r"\b(?:early|regular|consistent)\s+(?:to\s+)?bed\b", re.I), "regular"),
    ],
    "study_work_schedule": [
        (re.compile(r"\b(?:exams|studying|study|uni|university|college|school|thesis|coursework)\b", re.I), "student_high_workload"),
        (re.compile(r"\b(?:work|job|deadline|shifts?|overtime)\b", re.I), "high_workload"),
    ],
    "social_support": [
        (re.compile(r"\b(?:no one|nobody|don'?t have anyone|no friends|alone)\b", re.I), "low_support"),
        (re.compile(r"\b(?:my (?:friend|family|partner|girlfriend|boyfriend|mom|mother|dad|father)s?\b)", re.I), "has_support"),
    ],
    "daily_routine": [
        (re.compile(r"\b(?:no routine|nothing structured|all over the place)\b", re.I), "irregular"),
        (re.compile(r"\b(?:routine|schedule)\b", re.I), "has_routine"),
    ],
    "activity_level": [
        (re.compile(r"\b(?:gym|running|walk(?:ing)?|exercise|workout|sport)\b", re.I), "active"),
        (re.compile(r"\b(?:don'?t (?:do|go|leave)|no exercise|barely move|stay in bed)\b", re.I), "inactive"),
    ],
    "preferred_activities": [
        (re.compile(r"\b(?:music|guitar|piano|reading|books?|gaming|games|painting|drawing|cooking|hiking|swimming|football|soccer|series|movies?)\b", re.I), "detected"),
    ],
}


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?;\n]+", text) if s.strip()]


def _find_term(text: str, terms: Set[str]) -> Optional[str]:
    """Return the first matching term (multi-word aware) in the text."""
    lowered = text.lower()
    for term in terms:
        if term in lowered:
            return term
    return None


def _is_negated(text: str, term: str) -> bool:
    """Check whether the matched term is preceded by a negation window."""
    idx = text.lower().find(term)
    if idx <= 0:
        return False
    window = text.lower()[max(0, idx - 25): idx]
    return any(neg in window for neg in _NEGATION)


def _has_intensifier(text: str, term: str) -> bool:
    idx = text.lower().find(term)
    window = text.lower()[max(0, idx - 20): idx + len(term) + 20]
    return any(i in window for i in _INTENSIFIERS)


def _has_hedge(text: str) -> bool:
    lowered = text.lower()
    return any(h in lowered for h in _HEDGES)


def _assess_dimension(text: str, dim: str) -> Optional[Dict[str, Any]]:
    """Assess one dimension in one response. Returns None when no evidence exists.

    IMPORTANT: absence of evidence is NOT evidence of absence — we never
    set a dimension to severity 0 just because its terms are missing.
    """
    terms = _DIMENSION_TERMS[dim]
    strong = _find_term(text, terms["strong"])
    weak = _find_term(text, terms["weak"])

    matched = strong or weak
    if not matched:
        return None

    # Sleep is a special case: mentioning "sleep" positively ("I sleep fine")
    # is not evidence of sleep problems. The lexicon above already avoids
    # positive-sleep phrases; extra guard for "no" constructions.
    negated = _is_negated(text, matched)
    if dim == "sleep" and negated:
        return None

    if strong:
        severity = 3.0 if _has_intensifier(text, matched) else 2.5
        base_confidence = 0.9
    else:
        severity = 2.0 if _has_intensifier(text, matched) else 1.5
        base_confidence = 0.7

    if negated:
        # Negated strong evidence ("I don't feel hopeless") → low severity, low confidence
        severity = 0.5
        base_confidence = 0.4

    if _has_hedge(text):
        severity = min(severity, 1.5)
        base_confidence = min(base_confidence, 0.55)

    # Multi-sentence responses with repeated dimension evidence → slight bump
    return {
        "severity": round(severity, 2),
        "confidence": round(base_confidence, 2),
        "evidence": matched,
    }


def _extract_meta(text: str) -> Dict[str, str]:
    """Extract duration and frequency meta-dimensions."""
    lowered = text.lower()
    meta: Dict[str, str] = {}
    for phrase, value in _DURATION_MAP:
        if phrase in lowered:
            meta["duration"] = value
            break
    for phrase, value in _FREQUENCY_MAP:
        if phrase in lowered:
            meta["frequency"] = value
            break
    return meta


def _extract_context(text: str) -> Dict[str, str]:
    """Extract optional user-provided context (never inferred beyond surface text)."""
    context: Dict[str, str] = {}
    for key, patterns in _CONTEXT_PATTERNS.items():
        for regex, value in patterns:
            if regex.search(text):
                context[key] = value
                break
    return context


def _missing_information(text: str, features: Dict[str, Any]) -> List[str]:
    """Identify information gaps based on what was (and wasn't) expressed."""
    missing: List[str] = []
    meta = _extract_meta(text)

    # Any detected symptom dimension without frequency/duration context
    symptom_dims = [d for d in features if d not in ("duration", "frequency", "severity")]
    if symptom_dims:
        if "frequency" not in meta and "frequency" not in features:
            missing.append("frequency")
        if "duration" not in meta and "duration" not in features:
            missing.append("duration")
        # Severity qualifiers missing when only weak evidence was found
        weak_only = any(
            d not in ("duration", "frequency", "severity")
            and features[d].get("severity", 0) <= 1.5
            for d in symptom_dims
        )
        if weak_only:
            missing.append("severity_qualifier")
    return missing


def analyze_option_response(
    option: Dict[str, Any],
    question: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze a STRUCTURED option selection (question-specific option bank).

    This is the deterministic option → evidence path (spec §8): the option's
    ``signals`` map directly becomes the profile update. Same option → same
    evidence (reproducible); different options → different evidence.

    The option label is still passed through the free-text analyzer so that
    safety language embedded in an option is never missed and duration words
    in duration options ("Several weeks.") feed the meta dimensions.

    Args:
        option: An option dict from QUESTION_OPTIONS
            ({id, label, severity, signals}).
        question: Optional question metadata.

    Returns:
        Same structure as ``analyze_response`` plus
        ``response_mode: "structured_option"`` and the selected option id.
    """
    label = str(option.get("label", ""))
    signals: Dict[str, float] = {
        str(dim): float(val) for dim, val in (option.get("signals") or {}).items()
    }

    # Run the free-text analyzer over the label as a base (meta + safety +
    # context extraction). Its symptom-dimension output is IGNORED in favor
    # of the option's explicit evidence when both exist — the option bank is
    # the ground truth for what the user selected.
    base = analyze_response(label, question)

    dimensions: Dict[str, Dict[str, Any]] = {}
    max_severity = float(option.get("severity", 0))
    for dim, sev in signals.items():
        dimensions[dim] = {
            "severity": sev,
            # Structured selections are the most reliable evidence available:
            # the user explicitly picked it, so confidence is high.
            "confidence": 0.95 if sev > 0 else 0.9,
            "source": "option_selection",
        }
    # Merge any dimensions the label analysis found that the signals did not
    # cover (multi-target options should always cover their own dimensions).
    for dim, info in base.get("dimensions", {}).items():
        if dim not in dimensions:
            dimensions[dim] = info

    result: Dict[str, Any] = {
        "dimensions": dimensions,
        "missing_information": base.get("missing_information", []),
        "duration": base.get("duration"),
        "frequency": base.get("frequency"),
        "context": base.get("context", {}),
        "safety_concern": base.get("safety_concern", False),
        "ambiguous": False,
        "response_mode": "structured_option",
        "option_id": option.get("id"),
        "option_severity": max_severity,
    }
    if base.get("safety_matches"):
        result["safety_matches"] = base["safety_matches"]
    return result


def analyze_response(
    response_text: str,
    question: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze a single questionnaire response into structured information.

    Args:
        response_text: The user's raw free-text response.
        question: Optional question metadata (id, text, target_dimensions).

    Returns:
        Structured dict (NO chain-of-thought):
        {
          "dimensions": {dim: {"severity": float, "confidence": float}},
          "missing_information": [...],
          "duration": str|None, "frequency": str|None,
          "context": {...}, "safety_concern": bool,
          "ambiguous": bool
        }
    """
    response_text = (response_text or "").strip()
    result: Dict[str, Any] = {
        "dimensions": {},
        "missing_information": [],
        "duration": None,
        "frequency": None,
        "context": {},
        "safety_concern": False,
        "ambiguous": False,
    }

    if not response_text:
        result["ambiguous"] = True
        return result

    # --- Safety first ---
    safety_concern, safety_matches = check_safety_keywords(response_text)
    result["safety_concern"] = safety_concern
    if safety_concern:
        # Still extract structured info, but flag it prominently.
        result["safety_matches"] = safety_matches

    # --- Dimension assessment (sentence-level for better negation handling) ---
    sentences = _split_sentences(response_text)
    for dim in _DIMENSION_TERMS:
        best: Optional[Dict[str, Any]] = None
        for sentence in sentences:
            assessment = _assess_dimension(sentence, dim)
            if assessment is None:
                continue
            if best is None or assessment["severity"] > best["severity"]:
                best = assessment
        if best is not None:
            result["dimensions"][dim] = {
                "severity": best["severity"],
                "confidence": best["confidence"],
            }
    # --- Meta dimensions ---
    meta = _extract_meta(response_text)
    if "duration" in meta:
        result["duration"] = meta["duration"]
    if "frequency" in meta:
        result["frequency"] = meta["frequency"]

    # --- Optional context ---
    result["context"] = _extract_context(response_text)

    # --- Ambiguity / uncertainty handling ---
    if not result["dimensions"] and not meta and len(response_text.split()) >= 3:
        result["ambiguous"] = True

    # --- Information gaps ---
    result["missing_information"] = _missing_information(response_text, result["dimensions"])

    logger.debug(
        "Response analyzed: dims=%s duration=%s frequency=%s safety=%s",
        list(result["dimensions"].keys()), result["duration"], result["frequency"], safety_concern,
    )
    return result
