"""
Recommendation Engine — Generates explainable wellness recommendations
based on analysis results.

Two engines live in this module:

1. LEGACY (below): ``generate_recommendations`` — template-based generic
   lists keyed on risk level. Kept for the pre-existing analysis endpoints.

2. PERSONALIZED (appended at the bottom): ``UserAssessmentProfile`` +
   ``generate_personalized_recommendations`` — a controlled-library engine
   driven by the individual user's assessment profile. Recommendations are
   selected per-user by scoring dimension match, severity, confidence,
   functional impact, context, diversity and history — never from the final
   category alone.

This is NOT a medical diagnosis system. Recommendations are general
wellness guidance based on detected signals, never clinical advice.

Exports:
    generate_recommendations — legacy: analysis result → recommendation dict
    check_safety_keywords — transcript → safety concern flag
    UserAssessmentProfile — structured profile built from assessment results
    RECOMMENDATION_LIBRARY — controlled recommendation library
    generate_personalized_recommendations — profile → ranked, diverse recs
    get_crisis_response — region-aware safety override response
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    r"\b(?:think(?:ing)?\s+about\s+(?:ending|taking|ending)\s+(?:my\s+)?(?:life|it|myself))\b",
    r"\b(?:not\s+worth\s+living)\b",
    r"\b(?:better\s+off\s+dead)\b",
    r"\b(?:everyone\s+(?:would\s+be\s+)?better\s+(?:off\s+)?without\s+me)\b",
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


# =============================================================================
# PERSONALIZED RECOMMENDATION ENGINE (adaptive screening)
# =============================================================================

import re as _re  # local alias — module already imports re above

# ---------------------------------------------------------------------------
# UserAssessmentProfile — structured input to the recommendation engine
# ---------------------------------------------------------------------------

SEVERITY_LABELS = {
    "none": (0.0, 0.5),
    "mild": (0.5, 1.5),
    "moderate": (1.5, 2.5),
    "severe": (2.5, 3.01),
}

DIMENSION_LABELS = {
    "mood": "mood",
    "loss_of_interest": "loss of interest",
    "sleep": "sleep difficulties",
    "energy": "low energy",
    "appetite": "appetite changes",
    "concentration": "concentration difficulties",
    "self_worth": "reduced self-worth",
    "guilt": "feelings of guilt",
    "psychomotor_changes": "psychomotor changes",
    "social_withdrawal": "social withdrawal",
    "daily_functioning": "daily functioning difficulties",
}


def _severity_label(value: float) -> str:
    for label, (lo, hi) in SEVERITY_LABELS.items():
        if lo <= value < hi:
            return label
    return "mild"


@dataclass
class UserAssessmentProfile:
    """Structured user profile built from completed assessment results.

    This profile is the INPUT to the personalized recommendation engine.
    """

    overall_screening_level: str = "minimal"  # minimal | lower | some | elevated | high
    dimensions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    functional_impact: str = "low"            # low | moderate | high
    duration: str = "unknown"
    uncertainty: float = 0.5
    modalities_used: List[str] = field(default_factory=list)
    user_context: Dict[str, Any] = field(default_factory=dict)
    safety_concern: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_screening_level": self.overall_screening_level,
            "dimensions": self.dimensions,
            "functional_impact": self.functional_impact,
            "duration": self.duration,
            "uncertainty": self.uncertainty,
            "modalities_used": self.modalities_used,
            "user_context": self.user_context,
            "safety_concern": self.safety_concern,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserAssessmentProfile":
        return cls(
            overall_screening_level=data.get("overall_screening_level", "minimal"),
            dimensions=data.get("dimensions", {}),
            functional_impact=data.get("functional_impact", "low"),
            duration=data.get("duration", "unknown"),
            uncertainty=data.get("uncertainty", 0.5),
            modalities_used=data.get("modalities_used", []),
            user_context=data.get("user_context", {}),
            safety_concern=data.get("safety_concern", False),
        )


# ---------------------------------------------------------------------------
# Controlled recommendation library (rule-based safety first)
# ---------------------------------------------------------------------------

# Each entry is a CONTROLLED recommendation — a generative model never invents
# advice. Selection/ranking is dynamic; content is fixed and reviewed.
# reason_template receives (profile, matched_dims) and must tie the reason to
# the user's actual detected pattern.
RECOMMENDATION_LIBRARY: List[Dict[str, Any]] = [
    # ---- sleep_routine ----
    {
        "id": "sleep_consistent_schedule",
        "category": "sleep_routine",
        "target_dimensions": ["sleep", "energy"],
        "minimum_severity": 1,
        "evidence_level": "research_informed",
        "title": "Create a consistent sleep schedule",
        "reason_template": "Your responses indicate {sleep_ref} together with reduced daytime energy.",
        "action": "Try keeping your wake-up time consistent each day and keep your bedtime routine simple and predictable.",
        "priority": 0.9,
        "context_keys": [],
    },
    {
        "id": "sleep_wind_down",
        "category": "sleep_routine",
        "target_dimensions": ["sleep"],
        "minimum_severity": 2,
        "evidence_level": "research_informed",
        "title": "Reduce activities that interfere with sleep",
        "reason_template": "Your responses describe frequent sleep difficulties.",
        "action": "Consider reducing screen use and caffeine in the evening and reserving your bed for sleep.",
        "priority": 0.8,
        "context_keys": ["sleep_schedule"],
    },
    # ---- daily_structure ----
    {
        "id": "structured_daily_routine",
        "category": "daily_structure",
        "target_dimensions": ["energy", "daily_functioning"],
        "minimum_severity": 1,
        "evidence_level": "research_informed",
        "title": "Create a manageable daily routine",
        "reason_template": "Your responses suggest {functioning_ref} while energy has been reduced.",
        "action": "Try anchoring your day with two or three fixed points (meals, a short walk, bedtime) rather than a full schedule.",
        "priority": 0.85,
        "context_keys": ["daily_routine"],
    },
    {
        "id": "small_achievable_tasks",
        "category": "daily_structure",
        "target_dimensions": ["daily_functioning", "loss_of_interest"],
        "minimum_severity": 2,
        "evidence_level": "research_informed",
        "title": "Start with small achievable activities",
        "reason_template": "Your responses indicate {functioning_ref} and reduced interest in usual activities.",
        "action": "Pick one small task each day that feels achievable and build from there — completion matters more than size.",
        "priority": 0.8,
        "context_keys": [],
    },
    # ---- physical_activity ----
    {
        "id": "gentle_movement",
        "category": "physical_activity",
        "target_dimensions": ["energy", "mood"],
        "minimum_severity": 1,
        "evidence_level": "strong_evidence",
        "title": "Add gentle movement to your day",
        "reason_template": "Your responses show {mood_ref} alongside reduced energy.",
        "action": "A short walk or light stretching — even five minutes — can lift energy slightly; intensity is not the goal.",
        "priority": 0.7,
        "context_keys": ["activity_level"],
    },
    # ---- social_connection ----
    {
        "id": "maintain_contact",
        "category": "social_connection",
        "target_dimensions": ["social_withdrawal"],
        "minimum_severity": 1,
        "evidence_level": "research_informed",
        "title": "Maintain contact with a trusted person",
        "reason_template": "Your responses suggest {social_ref}.",
        "action": "Reach out to one person you trust — a short message or call counts; you don't have to explain everything.",
        "priority": 0.85,
        "context_keys": ["social_support"],
    },
    {
        "id": "avoid_isolation",
        "category": "social_connection",
        "target_dimensions": ["social_withdrawal", "loss_of_interest"],
        "minimum_severity": 2,
        "evidence_level": "research_informed",
        "title": "Avoid complete social isolation",
        "reason_template": "Your responses describe {social_ref} together with {interest_ref}.",
        "action": "Low-effort company helps — being around others in a shared space (a café, a study room) still provides contact.",
        "priority": 0.8,
        "context_keys": [],
    },
    # ---- enjoyable_activities ----
    {
        "id": "reintroduce_enjoyable_activity",
        "category": "enjoyable_activities",
        "target_dimensions": ["loss_of_interest"],
        "minimum_severity": 1,
        "evidence_level": "research_informed",
        "title": "Reintroduce one manageable enjoyable activity",
        "reason_template": "Your responses indicate {interest_ref}.",
        "action": "Choose one activity you used to enjoy and try it briefly, without expecting to enjoy it immediately — interest often returns gradually.",
        "priority": 0.85,
        "context_keys": ["preferred_activities"],
    },
    # ---- concentration_support ----
    {
        "id": "break_tasks_into_blocks",
        "category": "concentration_support",
        "target_dimensions": ["concentration"],
        "minimum_severity": 1,
        "evidence_level": "research_informed",
        "title": "Break demanding tasks into smaller blocks",
        "reason_template": "Your responses indicate {concentration_ref}.",
        "action": "Use short focused work periods (10–25 minutes) with planned breaks, and finish one block before starting another.",
        "priority": 0.85,
        "context_keys": [],
    },
    {
        "id": "reduce_multitasking",
        "category": "concentration_support",
        "target_dimensions": ["concentration", "energy"],
        "minimum_severity": 2,
        "evidence_level": "research_informed",
        "title": "Reduce multitasking",
        "reason_template": "Your responses suggest {concentration_ref} while energy is limited.",
        "action": "Single-task where possible: one window, one task, and note distractions on paper instead of following them.",
        "priority": 0.8,
        "context_keys": ["study_work_schedule"],
    },
    {
        "id": "schedule_high_energy_tasks",
        "category": "concentration_support",
        "target_dimensions": ["concentration", "energy"],
        "minimum_severity": 1,
        "evidence_level": "research_informed",
        "title": "Schedule demanding tasks during higher-energy periods",
        "reason_template": "Your responses show {concentration_ref} with fluctuating energy.",
        "action": "Notice when your energy tends to be highest and place your most demanding task in that window.",
        "priority": 0.75,
        "context_keys": [],
    },
    # ---- stress_management ----
    {
        "id": "workload_decomposition",
        "category": "stress_management",
        "target_dimensions": ["concentration", "daily_functioning"],
        "minimum_severity": 1,
        "evidence_level": "research_informed",
        "title": "Decompose your workload",
        "reason_template": "Your responses mention {functioning_ref} alongside concentration difficulties.",
        "action": "List what you must do this week, then cut each item into the smallest next step and do only that step today.",
        "priority": 0.75,
        "context_keys": ["study_work_schedule"],
    },
    # ---- relaxation ----
    {
        "id": "relaxation_practice",
        "category": "relaxation",
        "target_dimensions": ["mood", "psychomotor_changes"],
        "minimum_severity": 1,
        "evidence_level": "research_informed",
        "title": "Try a brief relaxation practice",
        "reason_template": "Your responses describe {mood_ref} and physical tension or restlessness.",
        "action": "Slow breathing (in for 4, out for 6) for a few minutes, twice a day, is a simple starting point.",
        "priority": 0.7,
        "context_keys": [],
    },
    # ---- journaling ----
    {
        "id": "mood_journaling",
        "category": "journaling",
        "target_dimensions": ["mood", "self_worth", "guilt"],
        "minimum_severity": 1,
        "evidence_level": "research_informed",
        "title": "Keep a short mood journal",
        "reason_template": "Your responses indicate {mood_ref} and self-critical thoughts.",
        "action": "Each evening, note your mood (1–10) and one thing that influenced it — patterns become visible within a week or two.",
        "priority": 0.65,
        "context_keys": [],
    },
    # ---- professional_support ----
    {
        "id": "professional_consultation",
        "category": "professional_support",
        "target_dimensions": ["daily_functioning", "mood"],
        "minimum_severity": 1,
        "evidence_level": "strong_evidence",
        "title": "Consider talking with a qualified professional",
        "reason_template": "Your responses suggest symptoms that have been persisting{duration_clause} and affecting daily functioning.",
        "action": "A general practitioner, counselor, or therapist can assess this properly — this tool cannot.",
        "priority": 0.9,
        "context_keys": [],
    },
]

# Maximum recommendations shown per category (diversity constraint)
MAX_PER_CATEGORY = 2


def _dim_ref(profile: UserAssessmentProfile, dim: str) -> str:
    """Build a human phrase for a dimension mention, honoring severity."""
    label = DIMENSION_LABELS.get(dim, dim.replace("_", " "))
    info = profile.dimensions.get(dim)
    if not info:
        return label
    sev = _severity_label(float(info.get("severity", 1.0)))
    if sev == "severe":
        return f"significant {label}"
    if sev == "moderate":
        return f"{label}"
    return f"some {label}"


def _duration_clause(profile: UserAssessmentProfile) -> str:
    d = profile.duration or "unknown"
    if d in ("several months", "a year or more"):
        return " over several months"
    if d == "several weeks":
        return " for several weeks"
    if d == "a few days":
        return " in recent days"
    return ""


def _render_reason(template: str, profile: UserAssessmentProfile, matched: List[str]) -> str:
    """Fill a reason template with the user's actual detected pattern."""
    subs = {
        "sleep_ref": _dim_ref(profile, "sleep"),
        "energy_ref": _dim_ref(profile, "energy"),
        "mood_ref": _dim_ref(profile, "mood"),
        "interest_ref": _dim_ref(profile, "loss_of_interest"),
        "social_ref": _dim_ref(profile, "social_withdrawal"),
        "concentration_ref": _dim_ref(profile, "concentration"),
        "functioning_ref": _dim_ref(profile, "daily_functioning"),
        "duration_clause": _duration_clause(profile),
    }
    reason = template.format(**subs)
    # Mention duration when persistent and relevant
    if "persist" in reason and profile.duration in ("several weeks", "several months", "a year or more"):
        reason = reason  # duration_clause already embedded
    return reason


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _profile_severity(profile: UserAssessmentProfile, dim: str) -> float:
    info = profile.dimensions.get(dim)
    return float(info.get("severity", 0.0)) if info else 0.0


def _profile_confidence(profile: UserAssessmentProfile, dim: str) -> float:
    info = profile.dimensions.get(dim)
    return float(info.get("confidence", 0.0)) if info else 0.0


def _score_recommendation(
    rec: Dict[str, Any],
    profile: UserAssessmentProfile,
    category_counts: Dict[str, int],
    recommendation_history: List[str],
) -> Tuple[float, List[str]]:
    """Compute personalized score (internal only — never exposed).

    recommendation_score = (dimension_match + severity_match + confidence
        + functional_impact_match + context_match + diversity_bonus)
        / redundancy_penalty
    """
    targets = rec["target_dimensions"]

    # --- dimension_match: overlap with prominent (detected) dimensions ---
    matched = [d for d in targets if d in profile.dimensions and _profile_severity(profile, d) >= 0.5]
    if not matched:
        return 0.0, []
    dimension_match = sum(min(3.0, _profile_severity(profile, d)) / 3.0 for d in matched) / len(targets)

    # --- severity_match: profile severity must meet the minimum ---
    max_sev = max(_profile_severity(profile, d) for d in matched)
    if max_sev < float(rec["minimum_severity"]):
        return 0.0, []
    severity_match = min(1.0, max_sev / 3.0)

    # --- confidence: how sure are we about the matched dimensions ---
    confidence = sum(_profile_confidence(profile, d) for d in matched) / len(matched)
    if confidence < 0.4:
        # Low-confidence features shouldn't drive recommendations alone
        dimension_match *= 0.5

    # --- functional_impact_match ---
    functional_impact_match = 0.0
    impact = profile.functional_impact
    if "daily_functioning" in matched:
        functional_impact_match = {"low": 0.2, "moderate": 0.6, "high": 1.0}.get(impact, 0.4)
    elif rec["category"] == "professional_support":
        functional_impact_match = {"low": 0.3, "moderate": 0.7, "high": 1.0}.get(impact, 0.4)

    # --- context_match: user-provided context only ---
    context_match = 0.0
    for key in rec.get("context_keys", []):
        if profile.user_context.get(key):
            context_match += 0.5
    context_match = min(1.0, context_match)

    # --- duration relevance for professional support ---
    if rec["category"] == "professional_support":
        if profile.duration in ("several weeks", "several months", "a year or more"):
            context_match = min(1.0, context_match + 0.5)

    # --- diversity_bonus: prefer under-represented categories ---
    category = rec["category"]
    diversity_bonus = 0.3 if category_counts.get(category, 0) == 0 else 0.0

    # --- redundancy_penalty: previously shown items are penalized ---
    redundancy = 2.5 if rec["id"] in recommendation_history else 1.0

    score = (
        dimension_match
        + severity_match
        + confidence
        + functional_impact_match
        + context_match
        + diversity_bonus
    ) / redundancy

    return score, matched


def generate_personalized_recommendations(
    profile: UserAssessmentProfile,
    recommendation_history: Optional[List[str]] = None,
    max_recommendations: int = 4,
) -> List[Dict[str, Any]]:
    """Generate personalized recommendations from a completed assessment.

    Selection is driven by the profile (dimensions, severity, confidence,
    duration, functional impact, context) — never by the screening category
    alone. Returns 3-5 diverse recommendations with reasons tied to the
    user's detected pattern.

    Returns a list of:
        { "id", "title", "reason", "action", "category", "priority" }
    """
    history = list(recommendation_history or [])

    # Safety takes absolute priority over normal recommendations
    if profile.safety_concern:
        return get_crisis_response().get("recommendations", [])

    category_counts: Dict[str, int] = {}
    scored: List[Tuple[float, Dict[str, Any], List[str]]] = []

    for rec in RECOMMENDATION_LIBRARY:
        score, matched = _score_recommendation(rec, profile, category_counts, history)
        if score > 0:
            scored.append((score, rec, matched))

    # Rank candidates
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: List[Dict[str, Any]] = []
    for score, rec, matched in scored:
        if len(selected) >= max_recommendations:
            break
        # Diversity constraint: max 2 per category
        if category_counts.get(rec["category"], 0) >= MAX_PER_CATEGORY:
            continue
        reason = _render_reason(rec["reason_template"], profile, matched)
        # Spec §27: name the screening area that drove this suggestion.
        # The first matched target is the strongest evidence dimension.
        matched_area = matched[0] if matched else rec["target_dimensions"][0]
        selected.append(
            {
                "id": rec["id"],
                "title": rec["title"],
                "reason": reason,
                "action": rec["action"],
                "category": rec["category"],
                "matched_area": matched_area,
                "priority": round(rec["priority"], 2),
                "evidence_level": rec["evidence_level"],
            }
        )
        category_counts[rec["category"]] = category_counts.get(rec["category"], 0) + 1

    # Professional support should always be present when symptoms are
    # persistent (weeks+) or functioning is significantly affected.
    has_professional = any(r["category"] == "professional_support" for r in selected)
    if not has_professional and (
        profile.duration in ("several weeks", "several months", "a year or more")
        or profile.functional_impact in ("moderate", "high")
        or profile.overall_screening_level in ("elevated", "high")
    ):
        prof = next(r for r in RECOMMENDATION_LIBRARY if r["category"] == "professional_support")
        selected.append(
            {
                "id": prof["id"],
                "title": prof["title"],
                "reason": _render_reason(prof["reason_template"], profile, ["daily_functioning"]),
                "action": prof["action"],
                "category": prof["category"],
                "priority": round(prof["priority"], 2),
                "evidence_level": prof["evidence_level"],
            }
        )

    return selected


# ---------------------------------------------------------------------------
# Safety override — crisis response (region-aware)
# ---------------------------------------------------------------------------

# Region-aware crisis resources. ``region`` is user-supplied metadata (never
# inferred); unknown regions fall back to international resources.
CRISIS_RESOURCES: Dict[str, List[Dict[str, str]]] = {
    "us": [
        {"label": "988 Suicide & Crisis Lifeline (US)", "value": "Call or text 988"},
        {"label": "Crisis Text Line", "value": "Text HOME to 741741"},
        {"label": "Emergency services", "value": "Call 911"},
    ],
    "uk": [
        {"label": "Samaritans (UK & Ireland)", "value": "Call 116 123"},
        {"label": "Emergency services", "value": "Call 999"},
    ],
    "eu": [
        {"label": "Emergency services (EU)", "value": "Call 112"},
        {"label": "Find a local helpline", "value": "https://findahelpline.com"},
    ],
    "international": [
        {
            "label": "International Association for Suicide Prevention",
            "value": "https://www.iasp.info/resources/Crisis_Centres/",
        },
        {"label": "Find a helpline (worldwide)", "value": "https://findahelpline.com"},
        {"label": "Local emergency services", "value": "Your local emergency number"},
    ],
}


def get_crisis_response(region: Optional[str] = None) -> Dict[str, Any]:
    """Safety override response — replaces normal recommendations entirely.

    Never returns wellness advice when a high-risk signal is detected.
    """
    key = (region or "international").lower()
    resources = CRISIS_RESOURCES.get(key, CRISIS_RESOURCES["international"])
    return {
        "status": "Immediate Support Recommended",
        "safety_override": True,
        "message": (
            "Some of your responses suggest you may be in serious distress. "
            "This tool cannot provide crisis support — please reach out now. "
            "If you are in immediate danger, contact your local emergency "
            "services or a crisis helpline right away. Talking to someone "
            "you trust can also help in this moment."
        ),
        "recommendations": [
            {
                "id": "crisis_support",
                "title": "Reach out for immediate support",
                "reason": "Your responses included signals that deserve immediate, human support — not self-help steps.",
                "action": "Please contact a crisis helpline or emergency service now, and stay with or near someone you trust if you can.",
                "category": "crisis_support",
                "priority": 1.0,
            }
        ],
        "crisis_resources": resources,
    }
