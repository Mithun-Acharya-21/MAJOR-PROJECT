"""
Phase 0 — model integrity tests.

These encode the serving contract:

  * ModelState never instantiates a randomly initialized network.
  * Unverified checkpoints (vision/fusion) are reported as not servable.
  * The legacy analysis endpoints label heuristic signals honestly and only
    report a model_version when a verified text checkpoint produced the score.
  * A training run cannot silently swap its checkpoint into serving.
  * GET / works even though frontend/index.html does not exist.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_META = PROJECT_ROOT / "models" / "final" / "text_model" / "text_model_metadata.json"


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app

    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Registry / ModelState
# ---------------------------------------------------------------------------

def test_model_state_never_builds_random_branches():
    from backend.app.main import _state

    _state.load()

    assert _state.text_branch is None
    assert _state.vision_branch is None
    assert _state.fusion_model is None
    assert _state.tokenizer is None


def test_vision_and_fusion_are_not_servable_without_participant_metadata():
    from backend.app.model_registry import audit_checkpoints

    status = audit_checkpoints()

    assert status["vision"]["available"] is False
    assert status["fusion"]["available"] is False
    assert status["vision"]["provenance"] == "unverified"
    assert status["fusion"]["provenance"] == "unverified"
    # The reason must be explicit, not a silent default.
    assert status["vision"]["reason"]
    assert status["fusion"]["reason"]


def test_text_availability_requires_metadata_and_weights():
    from backend.app.model_registry import audit_checkpoints

    status = audit_checkpoints()["text"]
    expected = TEXT_META.exists() and (
        PROJECT_ROOT / "models" / "final" / "text_model" / "text_model.pt"
    ).exists()

    assert status["available"] is expected
    if expected:
        assert status["provenance"] == "trained_metadata"
        assert status["model_version"]
        assert status["metrics"]
    else:
        assert status["reason"]


# ---------------------------------------------------------------------------
# Serving endpoints
# ---------------------------------------------------------------------------

def test_root_route_is_available_without_index_html(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"]
    assert body["status"] == "ok"


def test_english_text_is_labeled_lexicon_proxy(client):
    response = client.post(
        "/api/analyze",
        data={"text": "I feel very low and hopeless lately and cannot concentrate."},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["method"] == "lexicon_proxy"
    assert body["model_version"] is None
    assert body["signal_types"]["text"] == "lexicon_proxy"
    assert body["fusion_type"] == "experimental_heuristic_fusion"
    assert 0 <= body["probabilities"]["depressed"] <= 100


def test_trained_chinese_text_is_labeled_as_model_probability(client):
    if not TEXT_META.exists():
        pytest.skip("no trained text checkpoint")

    response = client.post(
        "/api/analyze",
        data={"text": "我最近情绪很低落，觉得很绝望，每天都很累，无法集中精力"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["method"] == "trained_model"
    assert body["model_version"]
    assert body["signal_types"]["text"] == "trained_model_probability"
    assert body["validated_modalities"] == ["text"]


def test_image_only_analysis_never_claims_a_model_probability(client):
    import io

    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(buffer, format="PNG")

    response = client.post(
        "/api/analyze",
        files={"file": ("photo.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["method"] is None
    assert body["model_version"] is None
    assert body["signal_types"]["photo"] in (
        "auxiliary_expression_score",
        "proxy_heuristic",
    )
    assert body["image_explanation"]["gradcam"] is None


def test_training_completion_cannot_swap_models_into_serving():
    from backend.app.main import _reload_from_training, _state

    _state.load()
    before = (_state.text_branch, _state.vision_branch, _state.fusion_model)

    _reload_from_training({"checkpoint_dir": "models/checkpoints/fusion_model"})

    assert (_state.text_branch, _state.vision_branch, _state.fusion_model) == before


def test_health_reports_verification_not_just_a_boolean(client):
    body = client.get("/api/health").json()
    assert body["status"] == "healthy"
    assert "verified_checkpoints" in body
    for branch in ("text", "vision", "fusion"):
        entry = body["verified_checkpoints"][branch]
        assert {"available", "reason", "provenance"} <= set(entry)


def test_no_endpoint_returns_a_probability_without_provenance(client):
    """Every analysis response must identify what produced its number."""
    response = client.post("/api/analyze", data={"text": "I am fine, just tired."})
    body = response.json()

    assert body["method"] in ("trained_model", "lexicon_proxy", "none")
    assert "signal_types" in body
    assert body["fusion_type"] == "experimental_heuristic_fusion"
    # A trained-model version may only appear on the trained-model path.
    if body["method"] != "trained_model":
        assert body["model_version"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
