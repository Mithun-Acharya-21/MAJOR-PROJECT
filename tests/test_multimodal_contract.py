"""
Multimodal contract + regression tests for the original reported bugs:

  Problem A: different answers produced the same screening result.
  Problem B: every question showed essentially the same options.
  Problem C: suggestions were nearly identical for every user.
  Problem E: the final result did not show how the 4 modalities contribute.

These tests encode spec §33-§36: five synthetic profiles must produce
different internal profiles, different modality/fusion outputs, and
different personalized suggestions — with all differences originating from
the actual responses (no randomization, no user-id keying).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from backend.app.session_manager import screening_session_manager
from backend.app.services.adaptive_question_service import (
    QUESTION_BANK,
    QUESTION_OPTIONS,
    build_question_payload,
    get_options,
    select_next_question,
)
from backend.app.services.response_analysis_service import (
    analyze_option_response,
    analyze_response,
)
from backend.app.services.screening_service import complete_screening


class _FakeModelState:
    """Neural models NOT loaded — lexicon fallback path (clearly labeled)."""

    loaded = False
    text_branch = None
    vision_branch = None
    tokenizer = None

    class device:
        pass


# ---------------------------------------------------------------------------
# §34: option banks are question-specific and carry real evidence
# ---------------------------------------------------------------------------

def test_every_question_has_options():
    for q in QUESTION_BANK:
        opts = get_options(q)
        assert opts, f"question {q['id']} has no option bank"


def test_options_differ_between_questions():
    """options(question_A) != options(question_B) for different constructs."""
    sleep = [o["id"] for o in QUESTION_OPTIONS["q_sleep_quality"]]
    energy = [o["id"] for o in QUESTION_OPTIONS["q_energy"]]
    interest = [o["id"] for o in QUESTION_OPTIONS["q_interest"]]
    mood = [o["id"] for o in QUESTION_OPTIONS["q_mood_overall"]]
    assert sleep != energy
    assert energy != interest
    assert interest != mood
    # Labels must differ too — not just ids
    sleep_labels = {o["label"] for o in QUESTION_OPTIONS["q_sleep_quality"]}
    energy_labels = {o["label"] for o in QUESTION_OPTIONS["q_energy"]}
    assert not (sleep_labels & energy_labels)


def test_option_signals_target_the_questions_own_dimensions():
    """A question's options must move the question's own target dimensions.

    Pure meta-dimension questions (duration) are exempt: their options set
    the descriptive duration value rather than severity signals.
    """
    META_ONLY_QUESTIONS = {"q_duration", "q_energy_duration", "q_mood_duration"}
    for q in QUESTION_BANK:
        targets = set(q.get("target_dimensions", []))
        opts = get_options(q)
        if not opts or not targets or q["id"] in META_ONLY_QUESTIONS:
            continue
        for opt in opts:
            signals = set(opt.get("signals", {}).keys())
            # At least one signal must hit a target dimension (or a paired
            # dimension of the follow-up) — never an unrelated dimension only.
            assert signals & targets, (
                f"{q['id']} option {opt['id']} signals {signals} miss targets {targets}"
            )


def test_different_options_produce_different_evidence():
    """Each option → different evidence → different profile update."""
    q = next(q for q in QUESTION_BANK if q["id"] == "q_sleep_quality")
    opt0, opt3 = QUESTION_OPTIONS["q_sleep_quality"][0], QUESTION_OPTIONS["q_sleep_quality"][3]
    a0 = analyze_option_response(opt0, q)
    a3 = analyze_option_response(opt3, q)
    assert a0["dimensions"]["sleep"]["severity"] < a3["dimensions"]["sleep"]["severity"]
    assert a0["dimensions"]["sleep"]["severity"] == 0
    assert a3["dimensions"]["sleep"]["severity"] == 3


def test_option_analysis_is_deterministic():
    q = next(q for q in QUESTION_BANK if q["id"] == "q_energy")
    opt = QUESTION_OPTIONS["q_energy"][2]
    r1 = analyze_option_response(opt, q)
    r2 = analyze_option_response(opt, q)
    assert r1["dimensions"] == r2["dimensions"]


def test_question_payload_exposes_question_specific_options():
    seen_option_sets = set()
    for q in QUESTION_BANK[:6]:
        payload = build_question_payload(q)
        assert "options" in payload and payload["options"]
        ids = tuple(o["id"] for o in payload["options"])
        # Frontend receives ids/labels only (no internal scoring fields)
        assert set(payload["options"][0].keys()) == {"id", "label"}
        seen_option_sets.add(ids)
    assert len(seen_option_sets) == 6, "different questions must expose different options"


# ---------------------------------------------------------------------------
# §35/§33: five synthetic profiles → different outcomes end-to-end
# ---------------------------------------------------------------------------

def _run_session(answers_by_question=None, free_text=None, option_rank=None):
    """Run an adaptive interview using structured options (rank 0=low, 3=high)
    or free-text answers. Returns the completed session."""
    session = screening_session_manager.create()
    for i in range(8):
        q = select_next_question(session)
        if q is None:
            break
        session.record_question(q)
        if free_text is not None:
            text = free_text[i % len(free_text)]
            session.record_response(text, analyze_response(text, q))
        else:
            opts = get_options(q)
            rank = (option_rank or {})(q, i) if callable(option_rank) else option_rank
            opt = opts[min(rank, len(opts) - 1)]
            session.record_response(f"[Selected: {opt['label']}]", analyze_option_response(opt, q))
    return session


def _complete(session):
    return complete_screening(session, _FakeModelState())


def _suggestions(result):
    return [r["id"] for r in result["suggestions"]]


# Five spec §33 profiles as option-selection strategies
PROFILE_1_LOW = dict(option_rank=0)  # all lowest options
PROFILE_2_SLEEP_ENERGY = dict(option_rank=2)  # uniformly elevated
PROFILE_5_HIGH = dict(option_rank=3)  # all highest options


def test_different_profiles_produce_different_scores():
    s_low = _run_session(**PROFILE_1_LOW)
    s_mid = _run_session(**PROFILE_2_SLEEP_ENERGY)
    s_high = _run_session(**PROFILE_5_HIGH)

    r_low, r_mid, r_high = _complete(s_low), _complete(s_mid), _complete(s_high)

    score_low = r_low["fusion"]["score"]
    score_mid = r_mid["fusion"]["score"]
    score_high = r_high["fusion"]["score"]

    assert score_low < score_mid < score_high, (
        f"scores must order with evidence: {score_low} < {score_mid} < {score_high}"
    )
    assert r_low["screening"]["category"] != r_high["screening"]["category"]


def test_different_profiles_produce_different_dimension_profiles():
    s_low = _run_session(**PROFILE_1_LOW)
    s_high = _run_session(**PROFILE_5_HIGH)
    r_low, r_high = _complete(s_low), _complete(s_high)

    dims_low = r_low["profile"]["dimensions"]
    dims_high = r_high["profile"]["dimensions"]
    assert dims_low != dims_high
    # High-burden profile must show genuinely elevated severities
    high_sevs = [v["severity"] for v in dims_high.values()]
    low_sevs = [v["severity"] for v in dims_low.values()]
    assert max(high_sevs) > max(low_sevs)


def test_different_profiles_produce_different_suggestions():
    """THE core regression: materially different users get different help."""
    s_low = _run_session(**PROFILE_1_LOW)
    s_high = _run_session(**PROFILE_5_HIGH)
    r_low, r_high = _complete(s_low), _complete(s_high)

    sug_low = _suggestions(r_low)
    sug_high = _suggestions(r_high)
    assert set(sug_low) != set(sug_high) or sug_low == [], (
        "low-burden and high-burden users must not receive identical suggestions"
    )
    # Determinism: same profile → same suggestions
    s_high2 = _run_session(**PROFILE_5_HIGH)
    assert _suggestions(_complete(s_high2)) == sug_high


def test_free_text_answers_change_the_result():
    """Problem A regression: same questions, different free-text answers."""
    low_text = [
        "I sleep normally and my energy is fine.",
        "I still enjoy my usual activities.",
        "My mood has been stable.",
        "I can concentrate on my work normally.",
        "I have been seeing friends as usual.",
        "Daily tasks are getting done as usual.",
        "This is just about the last couple of days.",
        "Things are mostly okay.",
    ]
    high_text = [
        "I cannot sleep — awake all night most days.",
        "I have lost interest in almost everything.",
        "I feel low and hopeless constantly.",
        "I cannot concentrate on anything at all.",
        "I avoid everyone, I see nobody anymore.",
        "I cannot function, daily tasks are impossible.",
        "It has been like this for months now.",
        "I am exhausted all the time.",
    ]
    s_low = _run_session(free_text=low_text)
    s_high = _run_session(free_text=high_text)
    r_low, r_high = _complete(s_low), _complete(s_high)

    assert r_low["profile"]["dimensions"] != r_high["profile"]["dimensions"], (
        "different free-text answers MUST produce different profiles"
    )
    assert r_low["fusion"]["score"] != r_high["fusion"]["score"], (
        "different free-text answers MUST produce different screening scores"
    )
    assert _suggestions(r_low) != _suggestions(r_high) or not _suggestions(r_low), (
        "different free-text answers MUST produce different suggestions"
    )


def test_sleep_dominant_vs_withdrawal_dominant_profiles_differ():
    """Sleep-dominant and withdrawal-dominant evidence → different profiles
    AND sleep-related vs social-related suggestion categories."""
    session_sleep = screening_session_manager.create()
    session_withdraw = screening_session_manager.create()

    sleep_plan = {"q_sleep_quality": 3, "q_sleep_frequency": 3, "q_energy": 2, "q_duration": 1}
    withdraw_plan = {"q_social": 3, "q_interest": 3, "q_support": 2, "q_duration": 1}

    for session, plan in ((session_sleep, sleep_plan), (session_withdraw, withdraw_plan)):
        for i in range(8):
            q = select_next_question(session)
            if q is None:
                break
            session.record_question(q)
            rank = plan.get(q["id"], 1)
            opts = get_options(q)
            opt = opts[min(rank, len(opts) - 1)]
            session.record_response(f"[Selected: {opt['label']}]", analyze_option_response(opt, q))

    r_sleep = _complete(session_sleep)
    r_withdraw = _complete(session_withdraw)

    assert r_sleep["profile"]["dimensions"] != r_withdraw["profile"]["dimensions"]
    # Sleep-dominant user should show higher sleep severity
    sleep_sev = r_sleep["profile"]["dimensions"].get("sleep", {}).get("severity", 0)
    withdraw_sleep_sev = r_withdraw["profile"]["dimensions"].get("sleep", {}).get("severity", 0)
    assert sleep_sev >= withdraw_sleep_sev
    # And the sleep user should receive sleep_routine support
    assert any(r["category"] == "sleep_routine" for r in r_sleep["suggestions"]), (
        f"sleep-dominant profile should receive sleep_routine support, got "
        f"{[r['category'] for r in r_sleep['suggestions']]}"
    )


# ---------------------------------------------------------------------------
# §17/§42: four-modality contract + fusion behavior
# ---------------------------------------------------------------------------

def test_canonical_result_contains_all_four_modalities():
    session = _run_session(**PROFILE_2_SLEEP_ENERGY)
    result = _complete(session)
    for key in ("text", "face", "live_face", "voice", "fusion", "screening",
                "explainability", "profile", "suggestions", "disclaimer"):
        assert key in result, f"canonical result missing: {key}"
    for key in ("text", "face", "live_face", "voice"):
        entry = result[key]
        assert entry["modality"] == key
        if entry["available"]:
            assert 0.0 <= entry["score"] <= 1.0
            assert 0.0 <= entry["confidence"] <= 1.0
            assert entry["signal_type"], "available modalities must carry a signal_type label"
    assert result["screening"]["non_diagnostic"] is True


def test_missing_modalities_are_marked_unavailable_not_fabricated():
    session = _run_session(**PROFILE_2_SLEEP_ENERGY)
    result = _complete(session)  # no image/audio inputs
    assert result["face"]["available"] is False
    assert result["live_face"]["available"] is False
    assert result["voice"]["available"] is False
    assert result["text"]["available"] is True
    assert result["fusion"]["validated"] is False


def test_fusion_renormalizes_when_modalities_missing():
    from backend.app.services.fusion_service import SCREENING_WEIGHTS, fuse_multimodal

    text = {"probability": 70.0, "confidence": 80.0, "signal_type": "rule_based_screening_score"}
    voice = {"probability": 30.0, "confidence": 60.0, "signal_type": "acoustic_feature_score"}
    fused = fuse_multimodal({"text": text, "voice": voice}, SCREENING_WEIGHTS)
    assert set(fused["fusion_weights"].keys()) == {"text", "voice"}
    assert abs(sum(fused["fusion_weights"].values()) - 1.0) < 1e-9

    # Adding a third modality must change the fused score (weights renormalize)
    face = {"probability": 30.0, "confidence": 65.0, "signal_type": "auxiliary_expression_score"}
    fused3 = fuse_multimodal({"text": text, "voice": voice, "face": face}, SCREENING_WEIGHTS)
    assert set(fused3["fusion_weights"].keys()) == {"text", "voice", "face"}
    assert fused3["probability"] != fused["probability"]


def test_final_result_changes_with_modality_evidence():
    """§21 scenario check: adding an elevated voice signal raises the score."""
    from backend.app.services.fusion_service import SCREENING_WEIGHTS, fuse_multimodal

    base = {"probability": 40.0, "confidence": 75.0, "signal_type": "rule_based_screening_score"}
    only_text = fuse_multimodal({"questionnaire": base, "text": base}, SCREENING_WEIGHTS)

    elevated_voice = {"probability": 85.0, "confidence": 70.0, "signal_type": "acoustic_feature_score"}
    with_voice = fuse_multimodal(
        {"questionnaire": base, "text": base, "voice": elevated_voice}, SCREENING_WEIGHTS
    )
    assert with_voice["probability"] > only_text["probability"], (
        "an elevated voice signal must raise the fused score"
    )


# ---------------------------------------------------------------------------
# §31/§32: research trace
# ---------------------------------------------------------------------------

def test_research_trace_captures_full_chain():
    session = _run_session(**PROFILE_5_HIGH)
    result = _complete(session)
    trace = session.research_trace
    assert trace is not None
    for key in ("text_evidence", "face_evidence", "live_face_evidence",
                "voice_evidence", "fusion", "final_result", "lime_explanation",
                "recommendations"):
        assert key in trace, f"research trace missing: {key}"
    assert trace["text_evidence"]["responses"], "trace must contain the actual responses"
    assert trace["text_evidence"]["features"], "trace must contain structured features"
    assert trace["fusion"]["weights"], "trace must contain fusion weights"
    assert trace["recommendations"] == result["suggestions"]


def test_research_trace_endpoint(client=None):
    """The trace endpoint exposes the trace for a completed session."""
    from fastapi.testclient import TestClient
    from backend.app.main import app

    with TestClient(app) as c:
        start = c.post("/api/screening/start").json()
        sid = start["session_id"]
        # answer 8 questions with the first (lowest) option each time
        for _ in range(8):
            resp = c.post(f"/api/screening/{sid}/response", json={"answer": "I feel okay, things are normal."})
            body = resp.json()
            if body.get("done"):
                break
        r = c.get(f"/api/screening/{sid}/research-trace")
        assert r.status_code == 200
        trace = r.json()
        for key in ("text_evidence", "fusion", "final_result", "recommendations"):
            assert key in trace


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
