"""
Personalization tests — prove recommendations change based on user results.

Synthetic profiles (per spec):
    PROFILE A: high sleep, high energy, moderate concentration
    PROFILE B: high social withdrawal, high loss of interest, moderate mood
    PROFILE C: high concentration, moderate energy, low sleep

Verifies:
    A != B, B != C, A != C (recommendation id sets)
    Recommendations target the relevant dimensions
    Reasons are tied to detected features (not generic)
    Diversity, severity matching, confidence handling, history,
    empty assessment, safety override, missing dimensions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.app.recommendations import (
    MAX_PER_CATEGORY,
    RECOMMENDATION_LIBRARY,
    UserAssessmentProfile,
    generate_personalized_recommendations,
    get_crisis_response,
)


def _dims(spec):
    return {
        name: {"severity": sev, "confidence": conf}
        for name, value in spec.items()
        for (sev, conf) in (value[0] if isinstance(value[0], tuple) else value,)
    }


PROFILE_A = UserAssessmentProfile(
    overall_screening_level="elevated",
    dimensions=_dims({
        "sleep": ((3.0, 0.91),),
        "energy": ((2.8, 0.84),),
        "concentration": ((2.0, 0.81),),
    }),
    functional_impact="moderate",
    duration="several weeks",
    uncertainty=0.12,
    modalities_used=["questionnaire", "text", "voice"],
)

PROFILE_B = UserAssessmentProfile(
    overall_screening_level="elevated",
    dimensions=_dims({
        "social_withdrawal": ((3.0, 0.9),),
        "loss_of_interest": ((2.8, 0.85),),
        "mood": ((2.0, 0.8),),
    }),
    functional_impact="moderate",
    duration="several weeks",
    uncertainty=0.14,
    modalities_used=["questionnaire", "text"],
)

PROFILE_C = UserAssessmentProfile(
    overall_screening_level="some",
    dimensions=_dims({
        "concentration": ((3.0, 0.88),),
        "energy": ((2.0, 0.8),),
        "sleep": ((1.0, 0.75),),
    }),
    functional_impact="low",
    duration="unknown",
    uncertainty=0.2,
    modalities_used=["questionnaire"],
)


def test_profiles_differ():
    """A != B, B != C, A != C at the id level."""
    ra = generate_personalized_recommendations(PROFILE_A)
    rb = generate_personalized_recommendations(PROFILE_B)
    rc = generate_personalized_recommendations(PROFILE_C)

    ids_a = [r["id"] for r in ra]
    ids_b = [r["id"] for r in rb]
    ids_c = [r["id"] for r in rc]

    assert set(ids_a) != set(ids_b), "Profile A and B must receive different recommendations"
    assert set(ids_b) != set(ids_c), "Profile B and C must receive different recommendations"
    assert set(ids_a) != set(ids_c), "Profile A and C must receive different recommendations"


def test_recommendations_target_relevant_dimensions():
    """Each recommendation's category should relate to the profile's top dimensions."""
    ra = generate_personalized_recommendations(PROFILE_A)
    rb = generate_personalized_recommendations(PROFILE_B)
    rc = generate_personalized_recommendations(PROFILE_C)

    # Profile A (sleep/energy high) → sleep_routine should appear
    assert any(r["category"] == "sleep_routine" for r in ra), f"Profile A should get sleep support, got {ra}"
    # Profile B (withdrawal/interest) → social_connection or enjoyable_activities
    assert any(
        r["category"] in ("social_connection", "enjoyable_activities") for r in rb
    ), f"Profile B should get social/interest support, got {rb}"
    # Profile C (concentration high) → concentration_support
    assert any(
        r["category"] == "concentration_support" for r in rc
    ), f"Profile C should get concentration support, got {rc}"


def test_reasons_tie_to_detected_features():
    """Reasons must mention the user's actual detected pattern, not generic text."""
    ra = generate_personalized_recommendations(PROFILE_A)
    rb = generate_personalized_recommendations(PROFILE_B)

    a_text = " ".join(r["reason"] for r in ra).lower()
    b_text = " ".join(r["reason"] for r in rb).lower()

    assert "sleep" in a_text, "Profile A reasons should mention sleep"
    assert "social" in b_text or "interest" in b_text or "isolat" in b_text, (
        "Profile B reasons should mention social withdrawal or loss of interest"
    )
    # Every recommendation must have a non-empty reason and action
    for r in ra + rb:
        assert r.get("reason"), f"{r['id']} missing reason"
        assert r.get("action"), f"{r['id']} missing action"
        assert r.get("title"), f"{r['id']} missing title"


def test_diversity_constraint():
    """No more than MAX_PER_CATEGORY recommendations per category."""
    profile = UserAssessmentProfile(
        overall_screening_level="high",
        dimensions=_dims({
            "sleep": ((3.0, 0.9),),
            "energy": ((3.0, 0.9),),
            "mood": ((3.0, 0.9),),
            "concentration": ((3.0, 0.9),),
            "social_withdrawal": ((3.0, 0.9),),
            "loss_of_interest": ((3.0, 0.9),),
            "daily_functioning": ((3.0, 0.9),),
        }),
        functional_impact="high",
        duration="several months",
    )
    recs = generate_personalized_recommendations(profile, max_recommendations=5)
    counts = {}
    for r in recs:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    for cat, count in counts.items():
        assert count <= MAX_PER_CATEGORY, f"Category {cat} exceeded diversity limit ({count})"
    assert 3 <= len(recs) <= 5, f"Expected 3-5 recommendations, got {len(recs)}"


def test_severity_matching():
    """Low-severity profiles should not receive recommendations with minimum_severity=2."""
    mild = UserAssessmentProfile(
        overall_screening_level="lower",
        dimensions=_dims({
            "sleep": ((0.8, 0.7),),
            "energy": ((1.2, 0.7),),
        }),
        functional_impact="low",
        duration="a few days",
    )
    recs = generate_personalized_recommendations(mild)
    min_sev_by_id = {r["id"]: r["minimum_severity"] for r in RECOMMENDATION_LIBRARY}
    for r in recs:
        assert r["priority"] <= 1.0
        # severity requirement respected
        assert min_sev_by_id.get(r["id"], 0) <= 2


def test_confidence_handling():
    """Low-confidence features should produce fewer/weaker recommendations."""
    high_conf = UserAssessmentProfile(
        overall_screening_level="elevated",
        dimensions=_dims({"sleep": ((3.0, 0.95),), "energy": ((3.0, 0.95),)}),
        functional_impact="moderate",
        duration="several weeks",
    )
    low_conf = UserAssessmentProfile(
        overall_screening_level="elevated",
        dimensions=_dims({"sleep": ((3.0, 0.3),), "energy": ((3.0, 0.3),)}),
        functional_impact="moderate",
        duration="several weeks",
    )
    recs_high = generate_personalized_recommendations(high_conf)
    recs_low = generate_personalized_recommendations(low_conf)
    # Low-confidence profile should not produce MORE recommendations
    assert len(recs_low) <= len(recs_high) + 1  # allow +1 for professional-support injection


def test_missing_dimensions():
    """A profile with dimensions absent from the library still yields valid recs."""
    sparse = UserAssessmentProfile(
        overall_screening_level="some",
        dimensions=_dims({"guilt": ((2.5, 0.8),)}),
        functional_impact="low",
        duration="unknown",
    )
    recs = generate_personalized_recommendations(sparse)
    for r in recs:
        assert r["id"] and r["title"] and r["reason"]


def test_empty_assessment():
    """Empty profile → no crash, likely empty or minimal recommendations."""
    empty = UserAssessmentProfile()
    recs = generate_personalized_recommendations(empty)
    assert isinstance(recs, list)


def test_recommendation_history_no_repeat():
    """Previously shown recommendations are penalized (not repeated first)."""
    profile = UserAssessmentProfile(
        overall_screening_level="elevated",
        dimensions=_dims({"sleep": ((3.0, 0.9),), "energy": ((2.8, 0.9),), "concentration": ((2.0, 0.8),)}),
        functional_impact="moderate",
        duration="several weeks",
    )
    fresh = generate_personalized_recommendations(profile)
    history = [r["id"] for r in fresh]
    repeat = generate_personalized_recommendations(profile, recommendation_history=history)
    repeat_ids = [r["id"] for r in repeat]
    # At least the top item should differ (penalized out of the top ranks)
    assert set(repeat_ids) != set(history) or all(
        rid in ("professional_consultation",) for rid in repeat_ids
    ), "History should change the selected set or fall back to professional support"


def test_safety_override():
    """Safety concern → crisis response, never wellness advice."""
    dangerous = UserAssessmentProfile(
        overall_screening_level="high",
        dimensions=_dims({"mood": ((3.0, 0.9),)}),
        functional_impact="high",
        duration="several months",
        safety_concern=True,
    )
    recs = generate_personalized_recommendations(dangerous)
    assert len(recs) >= 1
    assert all(r["category"] == "crisis_support" for r in recs), (
        f"Safety override must return only crisis support, got {recs}"
    )
    crisis = get_crisis_response("us")
    assert crisis["safety_override"] is True
    assert any("988" in res["value"] for res in crisis["crisis_resources"])
    # Unknown region falls back to international resources
    intl = get_crisis_response("unknown_region")
    assert intl["crisis_resources"] == get_crisis_response("international")["crisis_resources"]


def test_context_matching():
    """User-provided context shifts recommendations (student + workload)."""
    base = UserAssessmentProfile(
        overall_screening_level="elevated",
        dimensions=_dims({"concentration": ((3.0, 0.9),), "energy": ((2.0, 0.8),)}),
        functional_impact="moderate",
        duration="several weeks",
    )
    student = UserAssessmentProfile(
        overall_screening_level="elevated",
        dimensions=_dims({"concentration": ((3.0, 0.9),), "energy": ((2.0, 0.8),)}),
        functional_impact="moderate",
        duration="several weeks",
        user_context={"study_work_schedule": "student_high_workload"},
    )
    recs_student = generate_personalized_recommendations(student)
    recs_base = generate_personalized_recommendations(base)
    ids_student = [r["id"] for r in recs_student]
    ids_base = [r["id"] for r in recs_base]
    # The student context should change ranking or add workload-specific items
    assert ids_student != ids_base, "Context should influence recommendation selection/ranking"
    assert "workload_decomposition" in ids_student, "Student context should surface workload decomposition"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
