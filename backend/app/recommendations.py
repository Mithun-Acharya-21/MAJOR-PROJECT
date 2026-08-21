"""
Recommendation Engine — Generates explainable wellness recommendations
based on analysis results.

This is NOT a medical diagnosis system. Recommendations are general
wellness guidance based on detected signals, never clinical advice.

Exports:
    generate_recommendations — analysis result → recommendation dict
    check_safety_keywords — transcript → safety concern flag
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ── Safety keyword detection ──────────────────────────────────────────────────

# Explicit crisis indicators — NOT inferred from sadness/tiredness/low mood
_SAFETY_PATTERNS = [
    # Direct suicidal ideation
    r"\b(?:want|going|plan|think about?|wish|need)\s+(?:to\s+)?(?:kill|end|die|suicide|my\s+life)\b",
    r"\b(?:suicid(?:e|al|al)\b)",
    r"\b(?:end\s+(?:it\s+all|my\s+life|myself)\b)",
    r"\b(?:kill\s+(?:myself|me)\b)",
    r"\b(?:want\s+to\s+die)\b",
    r"\b(?:don'?t\s+(?:want|need)\s+to\s+be\s+(?:here|alive|around|anymore))\b",
    r"\b(?:no\s+(?:reason|point)\s+to\s+(?:live|go\s+on))\b",
    # Note: "can't take it anymore" alone is NOT a safety indicator — only match with explicit crisis context
    # r"\b(?:can'?t\s+(?:take|stand|bear)\s+it\s+(?:anymore|no\s+more))\b",  # intentionally excluded
    r"\b(?:harm\s+(?:myself|me)\b)",
    r"\b(?:hurt\s+(?:myself|me)\b)(?!\s+(?:when|because|if|by|during))",
    r"\b(?:cut(?:ting)?\s+(?:myself|me)\b)",
    r"\b(?:overdose(?:d)?\b)",
    r"\b(?:jump(?:ing)?\s+(?:off|from)\b)",
    r"\b(?:hang(?:ing)?\s+(?:myself|me)\b)",
    r"\b(?:drown(?:ing)?\s+(?:myself|me)\b)",
    r"\b(?:slit(?:ting)?\s+(?:my|me)\s+(?:wrists?|arms?|throat))\b",
    r"\b(?:bridge|tall\s+building|cliff|rooftop)\s+(?:jump|fall|end)\b",
    r"\b(?:goodbye\s+(?:cruel|forever|world|letter))\b",
    r"\b(?:final\s+(?:words?|message|note|letter))\b",
]

_safety_re = [re.compile(p, re.IGNORECASE) for p in _SAFETY_PATTERNS]


def check_safety_keywords(transcript: str) -> Tuple[bool, List[str]]:
    """Check transcript for explicit crisis/suicide indicators.

    Returns:
        (is_concern, matched_phrases)

    IMPORTANT: This checks ONLY for explicit self-harm/suicide language.
    It does NOT flag sadness, tiredness, low motivation, or neutral expressions.
    """
    if not transcript:
        return False, []

    matches = []
    for pattern in _safety_re:
        found = pattern.findall(transcript)
        if found:
            matches.extend(found)

    return len(matches) > 0, list(set(matches))


# ── Signal mapping ────────────────────────────────────────────────────────────

_SIGNAL_MAP = {
    "low_energy": {
        "signal": "Low energy",
        "description": "Reduced energy levels were detected in your session.",
    },
    "social_withdrawal": {
        "signal": "Social withdrawal",
        "description": "Patterns suggesting reduced social engagement were detected.",
    },
    "reduced_interest": {
        "signal": "Reduced interest",
        "description": "Signs of decreased interest in usual activities were detected.",
    },
    "fatigue": {
        "signal": "Fatigue",
        "description": "Fatigue-related indicators were detected.",
    },
    "sleep_disruption": {
        "signal": "Sleep disruption",
        "description": "Indicators of sleep difficulties were detected.",
    },
    "low_speaking_rate": {
        "signal": "Slowed speech",
        "description": "Reduced speaking rate was detected in your voice.",
    },
    "high_silence": {
        "signal": "Speech pauses",
        "description": "More pauses or silence than typical were detected in speech.",
    },
    "monotone_speech": {
        "signal": "Monotone speech",
        "description": "Reduced vocal expressiveness was detected.",
    },
    "negative_text": {
        "signal": "Negative language patterns",
        "description": "Text content contained patterns associated with low mood.",
    },
    "neutral_expression": {
        "signal": "Neutral facial expression",
        "description": "Facial analysis detected limited emotional expressiveness.",
    },
}

# Recommendation templates per level
_REC_TEMPLATES = {
    "low": {
        "level": "low_concern",
        "status": "Low Concern",
        "headline": "No strong concerning signals were detected in this session.",
        "what_you_can_do": {
            "today": [
                "Continue with your normal daily routines.",
                "Stay connected with people you care about.",
                "Make time for activities you enjoy.",
            ],
            "this_week": [
                "Maintain good sleep habits and regular meals.",
                "Take breaks and make time for hobbies.",
                "Continue checking in with your well-being.",
            ],
            "when_to_seek_help": [
                "If you notice persistent changes in mood, sleep, or energy, consider reaching out to a healthcare provider.",
            ],
        },
        "explainable_signals": [],
    },
    "moderate": {
        "level": "moderate_indicators",
        "status": "Some Indicators",
        "headline": "Some signals were detected that may be worth paying attention to.",
        "what_you_can_do": {
            "today": [
                "Take a moment to check in with how you're feeling.",
                "Try to maintain your normal routine where possible.",
                "Reach out to someone you trust if you feel comfortable.",
            ],
            "this_week": [
                "Monitor your mood and note any patterns.",
                "Prioritize sleep and physical activity.",
                "Consider journaling about how you feel.",
                "Stay connected with friends or family.",
            ],
            "when_to_seek_help": [
                "If these feelings continue or begin affecting your daily life, consider talking with a qualified mental-health professional.",
            ],
        },
    },
    "high": {
        "level": "high_concern",
        "status": "Elevated Concern",
        "headline": "Several signals were detected that may be worth discussing with a professional.",
        "what_you_can_do": {
            "today": [
                "Acknowledge what you're feeling — it's valid.",
                "Try to do one small thing that normally brings you comfort.",
                "Reach out to a trusted person if possible.",
            ],
            "this_week": [
                "Monitor your symptoms and note changes over time.",
                "Maintain basic self-care: sleep, nutrition, hydration.",
                "Reduce alcohol or substance use if applicable.",
            ],
            "when_to_seek_help": [
                "Speaking with a qualified mental-health professional (therapist, psychologist, counselor, or doctor) may be helpful.",
                "If you have a trusted person in your life, consider sharing how you've been feeling.",
                "Monitor your symptoms — if they persist or worsen, professional support is recommended.",
            ],
        },
    },
    "safety": {
        "level": "safety_concern",
        "status": "Immediate Support Recommended",
        "headline": None,  # Handled separately
        "what_you_can_do": {
            "today": [],
            "this_week": [],
            "when_to_seek_help": [],
        },
    },
}


def _detect_signals(result: Dict[str, Any]) -> List[str]:
    """Detect specific signals from analysis results for explainable recommendations."""
    signals = []
    probs = result.get("probabilities", {})
    depressed_prob = probs.get("depressed", 50) / 100.0

    # Text-based signals
    text_explanation = result.get("text_explanation", [])
    if text_explanation:
        risk_tokens = [t for t in text_explanation if t.get("weight", 0) > 0.6]
        if risk_tokens:
            signals.append("negative_text")

    # Voice-based signals
    voice_features = result.get("voice_features")
    if voice_features:
        if voice_features.get("silence_ratio", 0) > 0.25:
            signals.append("high_silence")
        if voice_features.get("speaking_rate", 5) < 2.5:
            signals.append("low_speaking_rate")
        if voice_features.get("pitch_variation", 50) < 8.0:
            signals.append("monotone_speech")
        if voice_features.get("rms_energy", 0.5) < 0.03:
            signals.append("low_energy")

    # Transcript-based text signals
    transcript = result.get("transcript", "")
    if transcript:
        t_lower = transcript.lower()
        if any(w in t_lower for w in ["tired", "exhausted", "no energy", "fatigue"]):
            signals.append("fatigue")
        if any(w in t_lower for w in ["alone", "lonely", "isolated", "no one", "nobody"]):
            signals.append("social_withdrawal")
        if any(w in t_lower for w in ["no interest", "don't care", "nothing matters", "boring", "pointless"]):
            signals.append("reduced_interest")
        if any(w in t_lower for w in ["can't sleep", "insomnia", "awake all night", "sleep"]):
            signals.append("sleep_disruption")

    # Facial-based signals — only note presence, never infer depression from neutral expression
    # (neutral face is NOT a depression indicator; removed to avoid misinterpretation)

    return signals


def _make_explainable_recommendations(signals: List[str]) -> List[Dict[str, str]]:
    """Map detected signals to explainable recommendation pairs."""
    recs = []
    _SIGNAL_RECS = {
        "low_energy": "Pay attention to sleep, energy levels, and daily routine.",
        "social_withdrawal": "Consider reconnecting with someone you trust.",
        "reduced_interest": "Monitor changes in interest and consider professional support if this persists.",
        "fatigue": "Prioritize rest and consider evaluating your sleep patterns.",
        "sleep_disruption": "Consider establishing a consistent sleep schedule.",
        "low_speaking_rate": "Note your communication patterns and overall energy.",
        "high_silence": "Your speech patterns may reflect current emotional state.",
        "monotone_speech": "Reduced vocal expressiveness can reflect emotional state.",
        "negative_text": "The language patterns in your text may indicate low mood.",
        "neutral_expression": "Facial expression analysis is one signal among many.",
    }
    for sig_id in signals:
        sig_info = _SIGNAL_MAP.get(sig_id, {})
        rec_text = _SIGNAL_RECS.get(sig_id, "Monitor this signal over time.")
        recs.append({
            "signal": sig_info.get("signal", sig_id),
            "description": sig_info.get("description", ""),
            "recommendation": rec_text,
        })
    return recs


def generate_recommendations(result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate wellness recommendations based on analysis results.

    Args:
        result: The analysis result dict from _run_analysis + session data.

    Returns:
        Dict with keys:
            - status: "Low Concern" | "Some Indicators" | "Elevated Concern" | "Immediate Support Recommended" | "Inconclusive"
            - headline: Main summary message
            - what_you_can_do: Dict with "today", "this_week", "when_to_seek_help"
            - explainable_signals: List of {signal, recommendation} dicts
            - safety_concern: bool
            - confidence_level: "high" | "moderate" | "low"
    """
    probs = result.get("probabilities", {})
    confidence = result.get("confidence", 50)
    depressed_prob = probs.get("depressed", 50) / 100.0
    modalities = result.get("modalities_used", [])

    # Check safety first
    transcript = result.get("transcript", "")
    safety_concern, safety_matches = check_safety_keywords(transcript)

    # Detect signals for explainable recommendations
    signals = _detect_signals(result)

    # Determine confidence level
    if len(modalities) >= 2:
        conf_level = "high" if confidence > 65 else "moderate"
    elif len(modalities) == 1:
        conf_level = "moderate" if confidence > 60 else "low"
    else:
        conf_level = "low"

    # Safety concern ALWAYS overrides everything — even inconclusive
    if safety_concern:
        return {
            "status": "Immediate Support Recommended",
            "headline": (
                "Your session contained language that suggests you may be "
                "going through a difficult time. This application cannot provide "
                "crisis support, but please reach out for help."
            ),
            "what_you_can_do": {
                "today": [
                    "If you are in immediate danger, please contact your local emergency services.",
                    "Contact a trusted person nearby — you don't have to go through this alone.",
                ],
                "this_week": [
                    "Consider reaching out to a crisis support service.",
                    "Speak with a mental-health professional as soon as possible.",
                ],
                "when_to_seek_help": [
                    "Crisis resources:",
                    "  • International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/",
                    "  • Crisis Text Line: Text HOME to 741741",
                    "  • National Suicide Prevention Lifeline (US): 988",
                    "  • Your local emergency number",
                ],
            },
            "explainable_signals": [
                {"signal": "Crisis language detected", "recommendation": "Please reach out for immediate support."}
            ],
            "safety_concern": True,
            "confidence_level": conf_level,
        }

    # Handle inconclusive cases (after safety check)
    if confidence < 55 or len(modalities) == 0:
        return {
            "status": "Inconclusive",
            "headline": (
                "The available signals are not strong enough to provide "
                "a reliable screening indication."
            ),
            "what_you_can_do": {
                "today": [
                    "Consider providing more information (text, image, or speech) for a more complete assessment.",
                    "If you're concerned about your well-being, speaking with a professional is always an option.",
                ],
                "this_week": [],
                "when_to_seek_help": [
                    "If you have ongoing concerns, consider talking with a qualified mental-health professional.",
                ],
            },
            "explainable_signals": _make_explainable_recommendations(signals),
            "safety_concern": False,
            "confidence_level": "low",
        }

    # Determine concern level based on depressed probability
    if depressed_prob >= 0.65:
        level = "high"
    elif depressed_prob >= 0.45:
        level = "moderate"
    else:
        level = "low"

    template = _REC_TEMPLATES[level].copy()
    template["explainable_signals"] = _make_explainable_recommendations(signals)
    template["safety_concern"] = False
    template["confidence_level"] = conf_level

    return template
