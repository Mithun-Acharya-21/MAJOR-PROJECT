"""Tests for Phase 16 — controlled personalised-support policy engine."""

from __future__ import annotations

import pytest

from app.ml.support_policy import (
    CRISIS_SIGNALS,
    POLICY_DISCLAIMER,
    SUGGESTION_LIBRARY,
    IndicatorSignal,
    PolicyInput,
    PolicyOutput,
    RiskTier,
    SuggestionCategory,
    SupportPolicyEngine,
    build_policy_input,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine() -> SupportPolicyEngine:
    return SupportPolicyEngine(max_suggestions_per_session=5)


def _input(
    tier: RiskTier = RiskTier.LOW,
    signals: list[IndicatorSignal] | None = None,
) -> PolicyInput:
    return PolicyInput(risk_tier=tier, observed_signals=signals or [])


# ---------------------------------------------------------------------------
# Crisis-first safety tests
# ---------------------------------------------------------------------------

class TestCrisisResponse:

    def test_self_harm_signal_triggers_crisis(self) -> None:
        engine = _engine()
        inp = _input(RiskTier.HIGH, [IndicatorSignal.SELF_HARM_LANGUAGE])
        out = engine.evaluate(inp)
        assert out.is_crisis is True
        assert out.risk_tier == RiskTier.CRISIS
        assert out.crisis_resources_shown is True

    def test_immediate_danger_triggers_crisis(self) -> None:
        engine = _engine()
        inp = _input(RiskTier.LOW, [IndicatorSignal.IMMEDIATE_DANGER])
        out = engine.evaluate(inp)
        assert out.is_crisis is True

    def test_crisis_suppresses_non_crisis_suggestions(self) -> None:
        engine = _engine()
        inp = _input(RiskTier.HIGH, [IndicatorSignal.SELF_HARM_LANGUAGE])
        out = engine.evaluate(inp)
        assert out.suppressed_by_crisis is True
        # Only crisis-resource category permitted
        for s in out.suggestions:
            assert s["category"] == SuggestionCategory.CRISIS_RESOURCES.value

    def test_crisis_tier_input_triggers_crisis_even_without_signal(self) -> None:
        engine = _engine()
        inp = _input(RiskTier.CRISIS)
        out = engine.evaluate(inp)
        assert out.is_crisis is True

    def test_crisis_suggestions_non_empty(self) -> None:
        engine = _engine()
        inp = _input(RiskTier.CRISIS)
        out = engine.evaluate(inp)
        assert len(out.suggestions) > 0

    def test_custom_crisis_resources_used(self) -> None:
        engine = _engine()
        custom = ["Call 999 for immediate help."]
        inp = PolicyInput(
            risk_tier=RiskTier.CRISIS,
            observed_signals=[IndicatorSignal.SELF_HARM_LANGUAGE],
            crisis_resources=custom,
        )
        out = engine.evaluate(inp)
        assert any(s["text"] == custom[0] for s in out.suggestions)

    def test_no_wellness_suggestions_in_crisis(self) -> None:
        engine = _engine()
        inp = _input(RiskTier.CRISIS, [IndicatorSignal.SELF_HARM_LANGUAGE])
        out = engine.evaluate(inp)
        wellness_categories = {
            SuggestionCategory.SLEEP_HYGIENE.value,
            SuggestionCategory.PHYSICAL_ACTIVITY.value,
            SuggestionCategory.MINDFULNESS_GROUNDING.value,
        }
        for s in out.suggestions:
            assert s["category"] not in wellness_categories


# ---------------------------------------------------------------------------
# Ordinary policy tests
# ---------------------------------------------------------------------------

class TestOrdinaryPolicy:

    def test_low_tier_returns_suggestions(self) -> None:
        engine = _engine()
        out = engine.evaluate(_input(RiskTier.LOW))
        assert not out.is_crisis
        assert len(out.suggestions) > 0

    def test_high_tier_includes_professional_support(self) -> None:
        engine = _engine()
        out = engine.evaluate(_input(RiskTier.HIGH))
        categories = [s["category"] for s in out.suggestions]
        assert SuggestionCategory.PROFESSIONAL_SUPPORT.value in categories

    def test_sleep_signal_adds_sleep_suggestion(self) -> None:
        engine = _engine()
        out = engine.evaluate(_input(RiskTier.MODERATE, [IndicatorSignal.SLEEP_DISRUPTION]))
        categories = [s["category"] for s in out.suggestions]
        assert SuggestionCategory.SLEEP_HYGIENE.value in categories

    def test_isolation_signal_adds_connection_suggestion(self) -> None:
        engine = _engine()
        out = engine.evaluate(_input(RiskTier.LOW, [IndicatorSignal.SOCIAL_ISOLATION]))
        categories = [s["category"] for s in out.suggestions]
        assert SuggestionCategory.SOCIAL_CONNECTION.value in categories

    def test_hopelessness_adds_professional_support(self) -> None:
        engine = _engine()
        out = engine.evaluate(_input(RiskTier.MODERATE, [IndicatorSignal.HOPELESSNESS_LANGUAGE]))
        categories = [s["category"] for s in out.suggestions]
        assert SuggestionCategory.PROFESSIONAL_SUPPORT.value in categories

    def test_max_suggestions_respected(self) -> None:
        engine = SupportPolicyEngine(max_suggestions_per_session=2)
        signals = [
            IndicatorSignal.SLEEP_DISRUPTION,
            IndicatorSignal.SOCIAL_ISOLATION,
            IndicatorSignal.STRESS_FATIGUE,
            IndicatorSignal.LOW_ENERGY,
        ]
        out = engine.evaluate(_input(RiskTier.MODERATE, signals))
        assert len(out.suggestions) <= 2

    def test_no_duplicate_categories(self) -> None:
        engine = _engine()
        signals = [IndicatorSignal.STRESS_FATIGUE, IndicatorSignal.STRESS_FATIGUE]
        out = engine.evaluate(_input(RiskTier.LOW, signals))
        categories = [s["category"] for s in out.suggestions]
        assert len(categories) == len(set(categories))

    def test_triggered_signals_recorded(self) -> None:
        engine = _engine()
        signals = [IndicatorSignal.SLEEP_DISRUPTION, IndicatorSignal.LOW_ENERGY]
        out = engine.evaluate(_input(RiskTier.LOW, signals))
        assert IndicatorSignal.SLEEP_DISRUPTION.value in out.triggered_signals
        assert IndicatorSignal.LOW_ENERGY.value in out.triggered_signals


# ---------------------------------------------------------------------------
# Disclaimer and safety text tests
# ---------------------------------------------------------------------------

class TestDisclaimerAndSafety:

    def test_disclaimer_in_every_output(self) -> None:
        engine = _engine()
        for tier in [RiskTier.LOW, RiskTier.MODERATE, RiskTier.HIGH]:
            out = engine.evaluate(_input(tier))
            assert out.disclaimer == POLICY_DISCLAIMER

    def test_disclaimer_not_medical_advice(self) -> None:
        assert "not medical advice" in POLICY_DISCLAIMER.lower()

    def test_suggestion_texts_not_prescriptive(self) -> None:
        """No suggestion should contain diagnostic or prescriptive language."""
        forbidden = ["diagnos", "prescri", "medication", "drug", "dose", "treat your"]
        for cat, texts in SUGGESTION_LIBRARY.items():
            for text in texts:
                for word in forbidden:
                    assert word not in text.lower(), (
                        f"Prescriptive word '{word}' found in {cat}: {text}"
                    )

    def test_to_dict_json_serialisable(self) -> None:
        import json
        engine = _engine()
        out = engine.evaluate(_input(RiskTier.MODERATE, [IndicatorSignal.STRESS_FATIGUE]))
        json.dumps(out.to_dict())

    def test_crisis_output_to_dict_has_all_keys(self) -> None:
        engine = _engine()
        out = engine.evaluate(_input(RiskTier.CRISIS, [IndicatorSignal.SELF_HARM_LANGUAGE]))
        d = out.to_dict()
        for key in ("risk_tier", "is_crisis", "suggestions", "triggered_signals",
                    "suppressed_by_crisis", "disclaimer", "crisis_resources_shown"):
            assert key in d


# ---------------------------------------------------------------------------
# build_policy_input helper
# ---------------------------------------------------------------------------

class TestBuildPolicyInput:

    def test_high_score_maps_to_high_tier(self) -> None:
        inp = build_policy_input(risk_score=0.80, uncertainty=0.1)
        assert inp.risk_tier == RiskTier.HIGH

    def test_low_score_maps_to_low_tier(self) -> None:
        inp = build_policy_input(risk_score=0.20, uncertainty=0.1)
        assert inp.risk_tier == RiskTier.LOW

    def test_self_harm_signal_overrides_to_crisis(self) -> None:
        inp = build_policy_input(
            risk_score=0.10,
            uncertainty=0.0,
            observed_signals=["self_harm_language"],
        )
        assert inp.risk_tier == RiskTier.CRISIS

    def test_unknown_signal_strings_silently_dropped(self) -> None:
        inp = build_policy_input(
            risk_score=0.50,
            uncertainty=0.2,
            observed_signals=["unknown_signal_xyz"],
        )
        assert inp.observed_signals == []

    def test_custom_crisis_resources_passed_through(self) -> None:
        custom = ["Local crisis line: 0800-HELP"]
        inp = build_policy_input(0.9, 0.3, crisis_resources=custom)
        assert inp.crisis_resources == custom

    def test_score_to_tier_boundaries(self) -> None:
        assert SupportPolicyEngine.score_to_tier(0.39) == RiskTier.LOW
        assert SupportPolicyEngine.score_to_tier(0.40) == RiskTier.MODERATE
        assert SupportPolicyEngine.score_to_tier(0.64) == RiskTier.MODERATE
        assert SupportPolicyEngine.score_to_tier(0.65) == RiskTier.HIGH
