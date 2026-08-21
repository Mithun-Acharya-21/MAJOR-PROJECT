"""Phase 16 — Controlled personalised-support policy engine.

Design constraints
------------------
- The system will NOT diagnose, prescribe, or recommend medication changes.
- Every suggestion is non-prescriptive, supportive, and auditable.
- High-priority self-harm / immediate-danger signals bypass all ordinary
  suggestions and return a mandatory crisis-resource response only.
- Suggestions are drawn from a fixed, approved set of categories.
- The policy is rule-based and fully auditable — no free-text generation.
- All outputs carry a mandatory disclaimer.
- Crisis resources are jurisdiction-configurable; this module never
  fabricates location-specific emergency contacts.

Signal taxonomy
---------------
The policy maps the following *observed indicators* to suggestions:

  sleep_disruption        → sleep hygiene suggestions
  social_isolation        → connection / community suggestions
  stress_fatigue          → rest / grounding / breathing suggestions
  low_energy              → gentle activity / routine suggestions
  hopelessness_language   → professional support reminder
  self_harm_language      → CRISIS RESPONSE (highest priority)
  immediate_danger        → CRISIS RESPONSE (highest priority)

Suggestion categories (approved)
---------------------------------
  SLEEP_HYGIENE
  SOCIAL_CONNECTION
  STRESS_MANAGEMENT
  PHYSICAL_ACTIVITY
  PROFESSIONAL_SUPPORT
  MINDFULNESS_GROUNDING
  CRISIS_RESOURCES          ← mandatory for self-harm/danger signals

Risk tiers
----------
  LOW       → general wellness suggestions
  MODERATE  → targeted suggestions for observed indicators
  HIGH      → professional support + targeted suggestions
  CRISIS    → crisis resources ONLY; no wellness suggestions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RiskTier(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRISIS = "crisis"


class SuggestionCategory(str, Enum):
    SLEEP_HYGIENE = "sleep_hygiene"
    SOCIAL_CONNECTION = "social_connection"
    STRESS_MANAGEMENT = "stress_management"
    PHYSICAL_ACTIVITY = "physical_activity"
    PROFESSIONAL_SUPPORT = "professional_support"
    MINDFULNESS_GROUNDING = "mindfulness_grounding"
    CRISIS_RESOURCES = "crisis_resources"


# ---------------------------------------------------------------------------
# Approved suggestion library
# (short, non-prescriptive, supportive — never medical advice)
# ---------------------------------------------------------------------------

SUGGESTION_LIBRARY: dict[SuggestionCategory, list[str]] = {
    SuggestionCategory.SLEEP_HYGIENE: [
        "Consider keeping a consistent sleep schedule, even on weekends.",
        "A brief wind-down routine before bed, such as dimming lights, can help signal rest.",
        "Limiting screens an hour before sleep may improve sleep quality for some people.",
    ],
    SuggestionCategory.SOCIAL_CONNECTION: [
        "Reaching out to a trusted friend or family member, even briefly, can help.",
        "Community groups, clubs, or volunteering can be a low-pressure way to connect.",
        "Online communities around shared interests are another option if in-person feels difficult.",
    ],
    SuggestionCategory.STRESS_MANAGEMENT: [
        "Short breaks during demanding tasks — even 5 minutes — can help reset focus.",
        "Writing down what is on your mind, without editing, can help externalise stress.",
        "Identifying one small, manageable task to complete can provide a sense of progress.",
    ],
    SuggestionCategory.PHYSICAL_ACTIVITY: [
        "A short walk, even indoors, can gently lift mood for some people.",
        "Gentle stretching or light movement at your own pace may help with energy.",
        "Any movement that feels comfortable to you counts — there is no required intensity.",
    ],
    SuggestionCategory.MINDFULNESS_GROUNDING: [
        "Slow, deliberate breathing — such as breathing in for 4 counts and out for 6 — is a simple grounding tool.",
        "Noticing five things you can see around you right now can help with anxious feelings.",
        "Grounding techniques work differently for different people; explore what suits you.",
    ],
    SuggestionCategory.PROFESSIONAL_SUPPORT: [
        "Speaking with a mental health professional can provide personalised support.",
        "Your GP or primary care provider is a good starting point for mental health referrals.",
        "Many workplaces and universities offer free or subsidised counselling — it may be worth checking.",
    ],
    SuggestionCategory.CRISIS_RESOURCES: [
        # Generic, non-fabricated — jurisdiction-specific contacts injected at runtime
        "If you are in immediate danger or at risk of harming yourself or others, please contact emergency services (e.g. 999, 911, or your local emergency number) now.",
        "Crisis support lines are available in most countries. A local search for 'mental health crisis line' or visiting findahelpline.com can provide relevant contacts.",
        "You do not need to be in immediate crisis to use a support line — they are there for distress of any level.",
    ],
}


# ---------------------------------------------------------------------------
# Signal and indicator types
# ---------------------------------------------------------------------------

class IndicatorSignal(str, Enum):
    SLEEP_DISRUPTION = "sleep_disruption"
    SOCIAL_ISOLATION = "social_isolation"
    STRESS_FATIGUE = "stress_fatigue"
    LOW_ENERGY = "low_energy"
    HOPELESSNESS_LANGUAGE = "hopelessness_language"
    SELF_HARM_LANGUAGE = "self_harm_language"
    IMMEDIATE_DANGER = "immediate_danger"


# Signals that always trigger crisis response regardless of risk score
CRISIS_SIGNALS = frozenset({
    IndicatorSignal.SELF_HARM_LANGUAGE,
    IndicatorSignal.IMMEDIATE_DANGER,
})

# Signal → suggestion category mapping
SIGNAL_TO_CATEGORIES: dict[IndicatorSignal, list[SuggestionCategory]] = {
    IndicatorSignal.SLEEP_DISRUPTION: [SuggestionCategory.SLEEP_HYGIENE],
    IndicatorSignal.SOCIAL_ISOLATION: [SuggestionCategory.SOCIAL_CONNECTION],
    IndicatorSignal.STRESS_FATIGUE: [
        SuggestionCategory.STRESS_MANAGEMENT,
        SuggestionCategory.MINDFULNESS_GROUNDING,
    ],
    IndicatorSignal.LOW_ENERGY: [SuggestionCategory.PHYSICAL_ACTIVITY],
    IndicatorSignal.HOPELESSNESS_LANGUAGE: [SuggestionCategory.PROFESSIONAL_SUPPORT],
    IndicatorSignal.SELF_HARM_LANGUAGE: [SuggestionCategory.CRISIS_RESOURCES],
    IndicatorSignal.IMMEDIATE_DANGER: [SuggestionCategory.CRISIS_RESOURCES],
}

POLICY_DISCLAIMER = (
    "These are general supportive suggestions based on observed signals in this session. "
    "They are not medical advice, diagnosis, or treatment. "
    "Please consult a qualified professional for personal support."
)


# ---------------------------------------------------------------------------
# Policy state and output
# ---------------------------------------------------------------------------

@dataclass
class PolicyInput:
    """Structured input to the support policy engine."""
    risk_tier: RiskTier
    observed_signals: list[IndicatorSignal] = field(default_factory=list)
    risk_score: Optional[float] = None       # ∈ [0, 1] from fusion model
    uncertainty: Optional[float] = None     # ∈ [0, 1]
    crisis_resources: Optional[list[str]] = None  # Jurisdiction-specific overrides


@dataclass
class PolicyOutput:
    """Structured output of the support policy engine."""
    risk_tier: RiskTier
    is_crisis: bool
    suggestions: list[dict]             # [{category, text}]
    triggered_signals: list[str]
    suppressed_by_crisis: bool          # True when non-crisis suggestions suppressed
    disclaimer: str = POLICY_DISCLAIMER
    crisis_resources_shown: bool = False

    def to_dict(self) -> dict:
        return {
            "risk_tier": self.risk_tier.value,
            "is_crisis": self.is_crisis,
            "suggestions": self.suggestions,
            "triggered_signals": self.triggered_signals,
            "suppressed_by_crisis": self.suppressed_by_crisis,
            "crisis_resources_shown": self.crisis_resources_shown,
            "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

class SupportPolicyEngine:
    """Rule-based, auditable suggestion policy.

    Rules are deterministic.  No free-text generation is performed.
    All outputs are drawn from SUGGESTION_LIBRARY.
    """

    # Risk-score thresholds → tier
    RISK_THRESHOLDS = {
        RiskTier.CRISIS:   0.0,   # Crisis is signal-driven, not score-driven
        RiskTier.HIGH:     0.65,
        RiskTier.MODERATE: 0.40,
        RiskTier.LOW:      0.0,
    }

    # Categories always shown at each tier (regardless of signals)
    TIER_BASE_CATEGORIES: dict[RiskTier, list[SuggestionCategory]] = {
        RiskTier.LOW: [SuggestionCategory.MINDFULNESS_GROUNDING],
        RiskTier.MODERATE: [
            SuggestionCategory.STRESS_MANAGEMENT,
            SuggestionCategory.MINDFULNESS_GROUNDING,
        ],
        RiskTier.HIGH: [
            SuggestionCategory.PROFESSIONAL_SUPPORT,
            SuggestionCategory.STRESS_MANAGEMENT,
        ],
        RiskTier.CRISIS: [SuggestionCategory.CRISIS_RESOURCES],
    }

    def __init__(self, max_suggestions_per_session: int = 4) -> None:
        self.max_suggestions = max_suggestions_per_session

    # ------------------------------------------------------------------
    # Classify risk tier from score (when not overridden)
    # ------------------------------------------------------------------

    @staticmethod
    def score_to_tier(risk_score: float) -> RiskTier:
        if risk_score >= 0.65:
            return RiskTier.HIGH
        if risk_score >= 0.40:
            return RiskTier.MODERATE
        return RiskTier.LOW

    # ------------------------------------------------------------------
    # Main policy evaluation
    # ------------------------------------------------------------------

    def evaluate(self, policy_input: PolicyInput) -> PolicyOutput:
        signals = policy_input.observed_signals or []

        # 1. Crisis check — bypasses all ordinary suggestions
        crisis_signals_present = [s for s in signals if s in CRISIS_SIGNALS]
        is_crisis = bool(crisis_signals_present) or policy_input.risk_tier == RiskTier.CRISIS

        if is_crisis:
            return self._crisis_response(policy_input, crisis_signals_present)

        # 2. Collect categories from base tier + observed signals
        tier = policy_input.risk_tier
        categories: list[SuggestionCategory] = list(self.TIER_BASE_CATEGORIES.get(tier, []))

        for signal in signals:
            for cat in SIGNAL_TO_CATEGORIES.get(signal, []):
                if cat not in categories and cat != SuggestionCategory.CRISIS_RESOURCES:
                    categories.append(cat)

        # 3. Deduplicate while preserving order
        seen: set[SuggestionCategory] = set()
        unique_categories: list[SuggestionCategory] = []
        for cat in categories:
            if cat not in seen:
                seen.add(cat)
                unique_categories.append(cat)

        # 4. Build suggestions (one per category, up to max)
        suggestions = self._build_suggestions(unique_categories)

        return PolicyOutput(
            risk_tier=tier,
            is_crisis=False,
            suggestions=suggestions,
            triggered_signals=[s.value for s in signals],
            suppressed_by_crisis=False,
        )

    # ------------------------------------------------------------------
    # Crisis response
    # ------------------------------------------------------------------

    def _crisis_response(
        self,
        policy_input: PolicyInput,
        crisis_signals: list[IndicatorSignal],
    ) -> PolicyOutput:
        # Use jurisdiction-specific resources if supplied, otherwise generic
        crisis_texts = (
            policy_input.crisis_resources
            or SUGGESTION_LIBRARY[SuggestionCategory.CRISIS_RESOURCES]
        )
        suggestions = [
            {"category": SuggestionCategory.CRISIS_RESOURCES.value, "text": text}
            for text in crisis_texts
        ]
        return PolicyOutput(
            risk_tier=RiskTier.CRISIS,
            is_crisis=True,
            suggestions=suggestions,
            triggered_signals=[s.value for s in (policy_input.observed_signals or [])],
            suppressed_by_crisis=True,    # Non-crisis suggestions suppressed
            crisis_resources_shown=True,
        )

    # ------------------------------------------------------------------
    # Build suggestion list
    # ------------------------------------------------------------------

    def _build_suggestions(
        self,
        categories: list[SuggestionCategory],
    ) -> list[dict]:
        suggestions: list[dict] = []
        for cat in categories:
            if len(suggestions) >= self.max_suggestions:
                break
            texts = SUGGESTION_LIBRARY.get(cat, [])
            if texts:
                suggestions.append({
                    "category": cat.value,
                    "text": texts[0],  # First suggestion per category
                })
        return suggestions


# ---------------------------------------------------------------------------
# Convenience factory: policy input from model output
# ---------------------------------------------------------------------------

def build_policy_input(
    risk_score: float,
    uncertainty: float,
    observed_signals: Optional[list[str]] = None,
    crisis_resources: Optional[list[str]] = None,
) -> PolicyInput:
    """Build a PolicyInput from fusion model outputs.

    Parameters
    ----------
    risk_score       : calibrated positive-class probability ∈ [0, 1]
    uncertainty      : normalised entropy ∈ [0, 1]
    observed_signals : list of IndicatorSignal string values (may be empty)
    crisis_resources : jurisdiction-specific crisis lines (optional override)
    """
    # Parse signal strings; silently drop unknown strings
    signals: list[IndicatorSignal] = []
    for s in (observed_signals or []):
        try:
            signals.append(IndicatorSignal(s))
        except ValueError:
            pass

    # Determine tier — crisis signals override score-based tier
    has_crisis = any(sig in CRISIS_SIGNALS for sig in signals)
    if has_crisis:
        tier = RiskTier.CRISIS
    else:
        tier = SupportPolicyEngine.score_to_tier(risk_score)

    return PolicyInput(
        risk_tier=tier,
        observed_signals=signals,
        risk_score=risk_score,
        uncertainty=uncertainty,
        crisis_resources=crisis_resources,
    )
