"""
API endpoint tests for the adaptive screening routes (POST /start,
POST /{id}/response, GET /state, GET /result, GET /recommendations,
POST /complete) plus invalid-session, modality normalisation and
existing-endpoint regression checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Import the real app; heavy models load lazily and lexicon fallbacks
    # cover the pipeline when torch weights are absent.
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


def _run_interview(client, responses):
    start = client.post("/api/screening/start")
    assert start.status_code == 200
    data = start.json()
    session_id = data["session_id"]
    for answer in responses:
        resp = client.post(f"/api/screening/{session_id}/response", json={"answer": answer})
        assert resp.status_code == 200
        body = resp.json()
        if body.get("done"):
            break
    return session_id


RESPONSES_HIGH = [
    "I've been feeling really low and hopeless most days.",
    "I've lost interest in almost everything I used to enjoy.",
    "I can't sleep — I'm awake all night most days.",
    "I'm exhausted all the time, no energy at all.",
    "It has been like this for weeks now.",
    "I can't concentrate on my work, I'm falling behind at everything.",
    "I've been avoiding people, I don't see anyone anymore.",
    "It's hard to get things done, my daily tasks pile up.",
]

RESPONSES_LOW = [
    "My mood has been okay, a few normal ups and downs.",
    "I still enjoy my hobbies most evenings.",
    "I sleep fine, maybe a late night on weekends.",
    "My energy is decent, I walk most mornings.",
    "This is just about the last couple of days.",
    "Concentration at work has been normal.",
    "I've seen friends and family this week.",
    "Daily tasks are getting done as usual.",
]


def test_start_returns_first_question(client):
    r = client.post("/api/screening/start")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    assert body["question"]["id"] and body["question"]["text"]
    assert body["progress"]["questions_answered"] == 0
    assert "progress_percent" in body["progress"]
    assert body["status"] == "in_progress"


def test_full_flow_to_completion(client):
    session_id = _run_interview(client, RESPONSES_HIGH)
    state = client.get(f"/api/screening/{session_id}/state").json()
    assert state["question_count"] >= 8
    assert state["features"], "structured features must be populated"
    assert state["status"] in ("in_progress", "completed")

    result = client.get(f"/api/screening/{session_id}/result").json()
    sr = result["screening_result"]
    assert sr["category"]
    assert sr["probability"] is None or 0 <= sr["probability"] <= 1
    assert sr["uncertainty"] is None or 0 <= sr["uncertainty"] <= 1
    assert "model_version" in sr

    recs = client.get(f"/api/screening/{session_id}/recommendations").json()
    assert len(recs["recommendations"]) >= 3
    for rec in recs["recommendations"]:
        assert rec["title"] and rec["reason"] and rec["action"] and rec["category"]


def test_low_profile_gets_different_recommendations(client):
    """End-to-end: two different users must receive different recommendations."""
    sid_high = _run_interview(client, RESPONSES_HIGH)
    sid_low = _run_interview(client, RESPONSES_LOW)

    recs_high = client.get(f"/api/screening/{sid_high}/recommendations").json()["recommendations"]
    recs_low = client.get(f"/api/screening/{sid_low}/recommendations").json()["recommendations"]

    ids_high = {r["id"] for r in recs_high}
    ids_low = {r["id"] for r in recs_low}
    assert ids_high != ids_low, "Different users must not receive identical recommendations"

    # High profile should skew towards symptom-directed categories
    cats_high = {r["category"] for r in recs_high}
    assert cats_high & {
        "sleep_routine", "daily_structure", "concentration_support",
        "social_connection", "professional_support", "enjoyable_activities",
    }, f"High profile categories unexpected: {cats_high}"


def test_invalid_session_404(client):
    assert client.get("/api/screening/does-not-exist/state").status_code == 404
    assert client.get("/api/screening/does-not-exist/result").status_code == 404
    assert client.get("/api/screening/does-not-exist/recommendations").status_code == 404
    assert client.post("/api/screening/does-not-exist/response", json={"answer": "hi"}).status_code == 404


def test_empty_response_rejected(client):
    start = client.post("/api/screening/start").json()
    r = client.post(f"/api/screening/{start['session_id']}/response", json={"answer": "   "})
    assert r.status_code == 400


def test_state_contains_required_schema(client):
    start = client.post("/api/screening/start").json()
    sid = start["session_id"]
    client.post(f"/api/screening/{sid}/response", json={"answer": "I feel low most days."})
    state = client.get(f"/api/screening/{sid}/state").json()
    for key in (
        "session_id", "questions", "responses", "features", "dimension_coverage",
        "asked_question_ids", "prediction_history", "uncertainty_history",
        "modalities_used", "question_count", "status", "recommendation_history",
    ):
        assert key in state, f"state missing required key: {key}"


def test_safety_override_via_api(client):
    start = client.post("/api/screening/start").json()
    sid = start["session_id"]
    r = client.post(
        f"/api/screening/{sid}/response",
        json={"answer": "I want to end my life."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["safety_concern"] is True
    assert body["question"] is None and body["done"] is True
    assert body["crisis_resources"], "crisis resources must be returned"
    recs = client.get(f"/api/screening/{sid}/recommendations").json()
    assert all(rec["category"] == "crisis_support" for rec in recs["recommendations"])


def test_complete_with_only_questionnaire(client):
    """POST /complete with no modality files — questionnaire-only result."""
    session_id = _run_interview(client, RESPONSES_HIGH)
    r = client.post(f"/api/screening/{session_id}/complete", files={})
    assert r.status_code == 200
    sr = r.json()["screening_result"]
    assert "questionnaire" in sr["modalities_used"]
    assert sr["category"]
    assert len(r.json()["recommendations"]) >= 3


def test_existing_endpoints_still_work(client):
    """Regression: pre-existing endpoints are untouched."""
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/baselines").status_code == 200
    assert client.get("/api/recommendations").status_code == 200
    # text analysis still functional
    r = client.post("/api/analyze/text", data={"text": "I feel very low and hopeless lately."})
    assert r.status_code == 200
    assert "probability" in r.json()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
