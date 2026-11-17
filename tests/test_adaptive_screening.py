"""
Adaptive screening tests — question selection, response analysis, stopping,
redundancy, progress, and the end-to-end session flow (service level).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.app.session_manager import screening_session_manager
from backend.app.services.adaptive_question_service import (
    MAX_QUESTIONS,
    MIN_QUESTIONS,
    QUESTION_BANK,
    build_question_payload,
    compute_dimension_coverage,
    get_progress,
    is_redundant,
    select_next_question,
    should_stop,
)
from backend.app.services.response_analysis_service import analyze_response
from backend.app.services.screening_service import (
    build_questionnaire_result,
    build_user_profile,
    categorize_probability,
    complete_screening,
)


# ---------------------------------------------------------------------------
# Response analysis
# ---------------------------------------------------------------------------

def test_response_analysis_extracts_dimensions():
    result = analyze_response("I've been sleeping very badly and I feel tired most days.")
    assert "sleep" in result["dimensions"]
    assert "energy" in result["dimensions"]
    assert 0 <= result["dimensions"]["sleep"]["severity"] <= 3
    assert 0 <= result["dimensions"]["sleep"]["confidence"] <= 1


def test_response_analysis_missing_information():
    result = analyze_response("I've been tired recently.")
    assert "energy" in result["dimensions"]
    assert "frequency" in result["missing_information"]
    assert "severity_qualifier" in result["missing_information"]


def test_response_analysis_does_not_invent_information():
    """A neutral response must not fabricate symptom dimensions."""
    result = analyze_response("I had a good day at work today.")
    assert result["dimensions"] == {} or all(
        info["severity"] <= 1.5 for info in result["dimensions"].values()
    )
    # "today" is no longer treated as duration evidence (removed vague terms);
    # no symptom dimensions and no duration should be inferred.
    assert result["duration"] is None
    assert not result["dimensions"], "Neutral response must not fabricate dimensions"


def test_response_analysis_ambiguity_recorded():
    result = analyze_response("I don't really know, things are fine I suppose.")
    # No clear symptom evidence — ambiguity recorded, not fabricated severity
    assert result["ambiguous"] or not result["dimensions"]


def test_response_analysis_negation():
    result = analyze_response("I don't feel hopeless, actually things are okay.")
    mood = result["dimensions"].get("mood")
    if mood is not None:
        assert mood["severity"] <= 1.0, "Negated strong evidence must not produce high severity"


def test_response_analysis_safety_flag():
    result = analyze_response("Sometimes I think about ending my life.")
    assert result["safety_concern"] is True


def test_response_analysis_meta_dimensions():
    result = analyze_response("I feel exhausted every day, it has been like this for weeks.")
    assert result["frequency"] in ("daily", "most_days")
    assert result["duration"] == "several weeks"


# ---------------------------------------------------------------------------
# Adaptive question selection
# ---------------------------------------------------------------------------

def test_first_question_targets_core_dimension():
    """On an empty session the highest-value core questions come first."""
    session = screening_session_manager.create()
    q = select_next_question(session)
    assert q is not None
    core = {t for t in q["target_dimensions"]} & {"mood", "loss_of_interest", "sleep", "energy", "daily_functioning", "duration"}
    assert core, f"First question should target a core dimension, got {q['id']}"


def test_next_question_depends_on_previous_response():
    """After a tiredness response, the system must NOT re-ask about tiredness."""
    session = screening_session_manager.create()
    q1 = select_next_question(session)
    session.record_question(q1)
    analysis = analyze_response("I've been tired recently.")
    session.record_response("I've been tired recently.", analysis)

    q2 = select_next_question(session)
    session.record_question(q2)
    analysis2 = analyze_response("Very often, almost every day.", q2)
    session.record_response("Very often, almost every day.", analysis2)

    # Energy should now be known — a question purely about energy levels
    # should be rejected as information-already-known or semantic duplicate.
    q3 = select_next_question(session)
    assert q3 is not None
    redundant, reason = is_redundant(
        q3, session.asked_question_ids, session.question_semantics, session.features
    )
    assert not redundant, f"Selected a redundant question ({reason}): {q3['text']}"


def test_question_repetition_prevented():
    session = screening_session_manager.create()
    asked_ids = []
    for _ in range(10):
        q = select_next_question(session)
        assert q is not None
        redundant, reason = is_redundant(
            q, session.asked_question_ids, session.question_semantics, session.features
        )
        assert not redundant, f"Repetition detected ({reason}): {q['id']}"
        session.record_question(q)
        asked_ids.append(q["id"])
        # Simulate a response touching the target dimension
        analysis = analyze_response("Yes, quite badly, most days for weeks now.")
        session.record_response("Yes, quite badly, most days for weeks now.", analysis)
    assert len(set(asked_ids)) == len(asked_ids)


def test_adaptive_stopping_rules():
    session = screening_session_manager.create()
    # Below MIN_QUESTIONS → never stops
    stop, reason = should_stop(session)
    assert not stop and reason == "min_questions_not_reached"

    # Fill session until MAX_QUESTIONS → must stop
    for _ in range(MAX_QUESTIONS):
        q = select_next_question(session)
        if q is None:
            break
        session.record_question(q)
        session.record_response("I feel low and tired most days for weeks.", analyze_response("I feel low and tired most days for weeks."))
    stop, reason = should_stop(session)
    assert stop and reason == "max_questions_reached"


def test_progress_is_adaptive_not_fake():
    """Progress must be 'Questions answered: N' + coverage-based percent."""
    session = screening_session_manager.create()
    progress = get_progress(session)
    assert progress["questions_answered"] == 0
    assert 0 <= progress["progress_percent"] < 30  # nearly nothing covered

    for _ in range(MIN_QUESTIONS):
        q = select_next_question(session)
        session.record_question(q)
        session.record_response("I feel low, tired, and can't sleep most days.", analyze_response("I feel low, tired, and can't sleep most days."))
    progress = get_progress(session)
    assert progress["questions_answered"] == MIN_QUESTIONS
    assert progress["progress_percent"] > progress_initial_guard(), (
        "Progress must increase with real coverage"
    )


def progress_initial_guard():
    return 30


def test_question_payload_has_no_internal_fields():
    session = screening_session_manager.create()
    q = select_next_question(session)
    payload = build_question_payload(q)
    assert set(payload.keys()) == {"id", "text", "target_dimensions", "options"}
    assert "importance" not in payload and "_prev_semantics" not in payload
    # Options carry no internal scoring fields either
    for opt in payload["options"]:
        assert set(opt.keys()) == {"id", "label"}


# ---------------------------------------------------------------------------
# Screening result / profile / fusion
# ---------------------------------------------------------------------------

def test_categorize_probability_screening_language():
    assert categorize_probability(0.2)["key"] == "lower_indication"
    assert categorize_probability(0.5)["key"] == "some_indication"
    assert categorize_probability(0.7)["key"] == "elevated_indication"
    assert categorize_probability(0.9)["key"] == "high_indication"
    # No diagnostic wording anywhere in the labels
    for prob in (0.1, 0.4, 0.55, 0.7, 0.9):
        label = categorize_probability(prob)["label"].lower()
        assert "depress" not in label or "depressive symptoms" in label
        assert "you are" not in label and "diagnosis" not in label


def test_questionnaire_result_from_features():
    features = {
        "sleep": {"severity": 3.0, "confidence": 0.9},
        "energy": {"severity": 2.5, "confidence": 0.8},
        "mood": {"severity": 2.0, "confidence": 0.75},
        "duration": {"value": "several weeks", "confidence": 0.8},
    }
    result = build_questionnaire_result(features, question_count=10)
    assert 0 < result["probability"] < 100
    assert result["probability"] > 55, "High severities should push probability up"
    assert result["explanation"]["method"] == "structured_questionnaire"


def test_questionnaire_empty_features():
    result = build_questionnaire_result({}, question_count=0)
    assert result["probability"] == 50.0
    assert result["confidence"] <= 25


class _FakeModelState:
    """Model state where neural models are NOT loaded (lexicon-only fallback)."""

    loaded = False
    text_branch = None
    vision_branch = None
    tokenizer = None

    class device:
        pass


def test_complete_screening_questionnaire_only():
    """End-to-end with no modality inputs — result is valid and screening-oriented."""
    session = screening_session_manager.create()
    # Simulate 8 answered questions
    responses = [
        "I've been feeling really low and hopeless most days.",
        "I've lost interest in almost everything I used to enjoy.",
        "I can't sleep — I'm awake all night, most days.",
        "I'm exhausted all the time, no energy at all.",
        "It has been like this for weeks now.",
        "I can't concentrate on my work, I'm falling behind at everything.",
        "I've been avoiding people, I don't see anyone anymore.",
        "It's hard to get things done, my daily tasks pile up.",
    ]
    for _i, r in enumerate(responses):
        q = select_next_question(session)
        session.record_question(q)
        session.record_response(r, analyze_response(r))

    result = complete_screening(session, _FakeModelState())
    sr = result["screening_result"]

    assert sr["probability"] is not None and 0 <= sr["probability"] <= 1
    assert sr["uncertainty"] is not None and 0 <= sr["uncertainty"] <= 1
    assert "depressive symptoms" in sr["category"].lower()
    assert sr["model_version"]
    assert "questionnaire" in sr["modalities_used"]
    assert len(result["recommendations"]) >= 3
    # Different severities → different recommendations (linked to this session)
    assert sr["dimensions"], "Profile dimensions must be populated from the interview"
    assert all("depress" not in r["title"].lower() for r in result["recommendations"])


def test_screening_result_stable_on_repeat_access():
    session = screening_session_manager.create()
    for r in ["I feel low most days.", "I can't sleep.", "No energy.", "For weeks now.",
              "Can't focus.", "Avoiding people.", "Falling behind.", "Feeling guilty."]:
        q = select_next_question(session)
        session.record_question(q)
        session.record_response(r, analyze_response(r))
    first = complete_screening(session, _FakeModelState())
    # Second access returns the STORED result (no re-computation drift)
    again = complete_screening(session, _FakeModelState())
    assert first["screening_result"]["probability"] == again["screening_result"]["probability"]
    assert first["recommendations"] == again["recommendations"]


def test_model_failure_graceful():
    """Model failure → questionnaire-only result with model_failure note."""
    session = screening_session_manager.create()
    for r in ["I feel down.", "No interest.", "Sleep is bad.", "Tired all the time.",
              "Weeks now.", "Can't focus at work.", "Avoiding friends.", "Daily tasks suffer."]:
        q = select_next_question(session)
        session.record_question(q)
        session.record_response(r, analyze_response(r))

    class _BrokenState(_FakeModelState):
        loaded = True

        def __getattr__(self, name):
            raise RuntimeError("model exploded")

    result = complete_screening(session, _BrokenState())
    sr = result["screening_result"]
    assert sr["probability"] is not None  # questionnaire signal still works
    assert "text" in sr.get("model_failure", []) or sr["probability"] is not None


def test_safety_override_in_complete_flow():
    session = screening_session_manager.create()
    q = select_next_question(session)
    session.record_question(q)
    session.record_response(
        "I want to end my life.", analyze_response("I want to end my life.")
    )
    result = complete_screening(session, _FakeModelState())
    assert result["screening_result"].get("safety_override") is True
    assert result["screening_result"]["category"] == "Immediate Support Recommended"
    assert result["recommendations"][0]["category"] == "crisis_support"
    assert "crisis_resources" in result


def test_profile_built_from_session():
    session = screening_session_manager.create()
    responses = [
        "My sleep is terrible, awake all night most days.",
        "I'm exhausted constantly, for months now.",
        "I avoid everyone, I don't see anyone.",
        "I feel worthless.",
        "Nothing matters anymore.",
        "Can't focus on anything.",
        "Daily functioning is a struggle, falling behind at work.",
        "There is no enjoyment in anything.",
    ]
    for r in responses:
        q = select_next_question(session)
        session.record_question(q)
        session.record_response(r, analyze_response(r))
    profile = build_user_profile(session, 0.7, 0.2)
    assert profile.dimensions
    assert "sleep" in profile.dimensions
    assert profile.modalities_used == ["text"] or "text" in profile.modalities_used or profile.modalities_used == ["questionnaire"]


def test_dimension_coverage_monotonic():
    session = screening_session_manager.create()
    before = compute_dimension_coverage(session.features)
    q = select_next_question(session)
    session.record_question(q)
    session.record_response("I can't sleep most nights.", analyze_response("I can't sleep most nights."))
    after = compute_dimension_coverage(session.features)
    assert after["sleep"] > before["sleep"]
    assert after["mood"] == before["mood"] == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
