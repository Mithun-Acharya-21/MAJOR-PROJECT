"""Extended backend endpoint probe — tests screening routes and multimodal."""
import urllib.request
import json

BASE = "http://localhost:8000"


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=15) as r:
            return {"status": r.status, "data": json.loads(r.read())}
    except Exception as e:
        return {"error": str(e)}


def post_json(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return {"status": r.status, "data": json.loads(r.read())}
    except Exception as e:
        return {"error": str(e)}


ok, fail = [], []

def check(name, r):
    has_err = "error" in r
    if has_err:
        fail.append(f"{name}: {r['error']}")
    else:
        ok.append(name)
    print(f"{'✅' if not has_err else '❌'} {name}")
    if not has_err:
        txt = json.dumps(r["data"], indent=2)
        print(txt[:600])
    else:
        print(f"   ERROR: {r['error']}")
    print()


# --- GET endpoints ---
check("GET /api/health", get("/api/health"))
check("GET /api/training-status", get("/api/training-status"))
check("GET /api/recommendations", get("/api/recommendations"))

# --- POST /api/analyze/text ---
r = post_json("/api/analyze/text", {"text": "I feel so hopeless and sad."})
check("POST /api/analyze/text (json)", r)

# --- POST /session/start ---
sess = post_json("/session/start", {})
check("POST /session/start", sess)

# Try session stop if we got an ID
if "data" in sess:
    sid = sess["data"].get("session_id", "")
    r2 = post_json(f"/session/{sid}/stop", {})
    check(f"POST /session/{sid}/stop", r2)

# --- POST /api/analyze/multimodal ---
r = post_json("/api/analyze/multimodal", {
    "text": "I feel very down and hopeless.",
    "voice_transcript": "",
    "face_scores": {}
})
check("POST /api/analyze/multimodal", r)

# --- POST /live-analyze ---
r = post_json("/live-analyze", {"frame_data": None, "session_id": "test"})
check("POST /live-analyze", r)

# --- Summary ---
print(f"\n{'='*50}")
print(f"PASSED: {len(ok)}/{len(ok)+len(fail)}")
if fail:
    print("FAILED:")
    for f in fail:
        print(f"  ❌ {f}")
else:
    print("All endpoints OK")
