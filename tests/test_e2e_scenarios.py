"""
End-to-end pipeline scenarios (spec §43), driven through the FastAPI app:

  Text only / + photo / + live frames / + voice / all four
  Missing photo, missing voice, missing live face
  Different text responses, different questionnaire paths
  Low-burden / high-burden / sleep-dominant / withdrawal-dominant /
  concentration-dominant profiles
  Crisis response
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


def _interview(client, mode, seed=0):
    """Run an adaptive interview; mode in {'low','high','sleep','withdraw','conc'}."""
    from PIL import Image as _I  # noqa

    start = client.post("/api/screening/start").json()
    sid = start["session_id"]

    # Option ranks per profile mode; default mid
    # We just answer with free text tuned per mode (options are also exercised
    # elsewhere); the analyzer extracts dimensions from the text.
    TEXTS = {
        "low": [
            "I sleep normally and my energy is fine.",
            "I still enjoy my usual activities.",
            "My mood has been stable.",
            "I can concentrate on my work normally.",
            "I have been seeing friends as usual.",
            "Daily tasks are getting done as usual.",
            "Just the last couple of days, if anything.",
            "Things are mostly okay.",
        ],
        "high": [
            "I cannot sleep, awake all night most days.",
            "I have lost interest in almost everything.",
            "I feel low and hopeless constantly.",
            "I cannot concentrate on anything at all.",
            "I avoid everyone, I see nobody anymore.",
            "I cannot function, daily tasks are impossible.",
            "It has been like this for months now.",
            "I am exhausted all the time.",
        ],
        "sleep": [
            "My sleep is terrible, awake all night most nights.",
            "I am exhausted during the day, no energy.",
            "Mood is mostly okay otherwise.",
            "Still enjoy my hobbies a bit.",
            "Concentration is somewhat affected by tiredness.",
            "I keep seeing friends.",
            "For several weeks now.",
            "It happens every day.",
        ],
        "withdraw": [
            "I have been avoiding everyone, I see nobody.",
            "Nothing interests me anymore.",
            "I have no one to talk to.",
            "My sleep is fine.",
            "Energy is okay.",
            "Mood is a bit flat.",
            "For weeks now.",
            "Most days.",
        ],
        "conc": [
            "I cannot concentrate on my studies at all.",
            "I am extremely stressed about deadlines.",
            "My sleep is fine.",
            "Energy is somewhat low.",
            "Mood is okay.",
            "I still see friends occasionally.",
            "A few weeks.",
            "Often.",
        ],
    }
    texts = TEXTS[mode]
    for i in range(10):
        resp = client.post(f"/api/screening/{sid}/response", json={"answer": texts[i % len(texts)]})
        body = resp.json()
        if body.get("done") or body.get("safety_concern"):
            break
    return sid


def _png_bytes(color=128):
    buf = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), color, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _result(client, sid, files=None):
    r = client.post(f"/api/screening/{sid}/complete", files=files or {})
    assert r.status_code == 200
    return r.json()


def test_text_only(client):
    sid = _interview(client, "high")
    res = _result(client, sid)
    assert res["text"]["available"] is True
    assert res["face"]["available"] is False
    assert res["live_face"]["available"] is False
    assert res["voice"]["available"] is False
    assert res["fusion"]["score"] > 0
    assert res["screening"]["non_diagnostic"] is True


def test_text_plus_photo(client):
    sid = _interview(client, "high")
    res = _result(client, sid, files={"photo": ("p.png", _png_bytes(90), "image/png")})
    assert res["face"]["available"] is True
    assert res["face"]["signal_type"] in ("auxiliary_expression_score", "proxy_heuristic")
    assert res["face"]["signal_type_label"], "must carry an honest signal label"


def test_text_plus_live_frames(client):
    sid = _interview(client, "high")
    frames = [("frames", (f"f{i}.png", _png_bytes(100 + i), "image/png")) for i in range(3)]
    res = _result(client, sid, files=frames)
    assert res["live_face"]["available"] is True
    # Multi-frame aggregation metadata must be present
    ev = res["live_face"].get("evidence", {})
    assert "frames_analyzed" in ev


def test_text_plus_voice(client):
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        t = np.linspace(0, 1.5, int(16000 * 1.5), endpoint=False)
        audio = (np.sin(2 * np.pi * 180 * t) * 8000).astype(np.int16)
        wf.writeframes(audio.tobytes())
    audio_bytes = buf.getvalue()

    sid = _interview(client, "high")
    res = _result(client, sid, files={"audio": ("a.wav", audio_bytes, "audio/wav")})
    assert res["voice"]["available"] is True
    assert res["voice"]["signal_type"] in ("acoustic_feature_score", "lexicon_proxy", "lexicon_proxy_plus_acoustic_feature_score")


def test_all_four_modalities(client):
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        t = np.linspace(0, 1.5, int(16000 * 1.5), endpoint=False)
        audio = (np.sin(2 * np.pi * 200 * t) * 9000).astype(np.int16)
        wf.writeframes(audio.tobytes())

    sid = _interview(client, "high")
    files = {
        "photo": ("p.png", _png_bytes(80), "image/png"),
        **{("frames"): (f"f{i}.png", _png_bytes(90 + i), "image/png") for i in range(2)},
        "audio": ("a.wav", buf.getvalue(), "audio/wav"),
    }
    res = _result(client, sid, files=files)
    for key in ("text", "face", "live_face", "voice"):
        assert res[key]["available"] is True, f"{key} should be available"
    # Contributions from all modalities must be present
    contribs = res["explainability"]["modality_contributions"]
    assert set(contribs.keys()) >= {"text", "face", "voice"}


def test_missing_modalities_renormalize(client):
    sid_low = _interview(client, "low")
    res_text_only = _result(client, sid_low)
    sid2 = _interview(client, "low")
    res_with_photo = _result(client, sid2, files={"photo": ("p.png", _png_bytes(60), "image/png")})

    # Both must produce valid scores; adding a modality changes the weights
    assert 0 <= res_text_only["fusion"]["score"] <= 1
    assert 0 <= res_with_photo["fusion"]["score"] <= 1
    assert set(res_text_only["fusion"]["weights"].keys()) != set(res_with_photo["fusion"]["weights"].keys())


def test_profiles_produce_different_final_results(client):
    scores = {}
    suggestions = {}
    for mode in ("low", "high", "sleep", "withdraw", "conc"):
        sid = _interview(client, mode)
        res = _result(client, sid)
        scores[mode] = res["fusion"]["score"]
        suggestions[mode] = sorted(r["id"] for r in res["suggestions"])
        assert res["profile"]["dimensions"], f"{mode} profile must have dimensions"

    assert scores["low"] < scores["high"], "low vs high burden must differ"
    assert len({tuple(s) for s in suggestions.values()}) >= 4, (
        f"different profiles must receive different suggestions: {suggestions}"
    )
    # Sleep-dominant user gets sleep_routine; withdrawal-dominant gets social support
    assert any("sleep" in r for r in suggestions["sleep"]), suggestions["sleep"]
    assert any(
        ("social" in r) or ("isolat" in r) or ("enjoyable" in r) for r in suggestions["withdraw"]
    ), suggestions["withdraw"]


def test_crisis_response_via_complete(client):
    start = client.post("/api/screening/start").json()
    sid = start["session_id"]
    client.post(f"/api/screening/{sid}/response", json={"answer": "I want to end my life."})
    res = _result(client, sid)
    assert res["screening"]["safety_override"] is True
    assert res["crisis_resources"]
    assert all(r["category"] == "crisis_support" for r in res["suggestions"])


def test_questionnaire_paths_diverge(client):
    """Different option selections → different next questions (adaptive)."""
    start = client.post("/api/screening/start").json()
    sid_a = start["session_id"]
    # A answers 'worst' options via free text; B answers mild ones —
    # the asked-question sequences must differ (adaptive branching).
    asked_a = []
    for i in range(3):
        r = client.post(
            f"/api/screening/{sid_a}/response",
            json={"answer": "I sleep terribly, awake all night, exhausted all day most days for weeks."},
        ).json()
        asked_a.append(r["question"]["id"] if r.get("question") else None)

    start_b = client.post("/api/screening/start").json()
    sid_b = start_b["session_id"]
    asked_b = []
    for i in range(3):
        r = client.post(
            f"/api/screening/{sid_b}/response",
            json={"answer": "Everything is completely fine, no problems at all."},
        ).json()
        asked_b.append(r["question"]["id"] if r.get("question") else None)

    assert asked_a[:2] != asked_b[:2] or asked_a[1:3] != asked_b[1:3], (
        "different response patterns must branch into different question paths"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
