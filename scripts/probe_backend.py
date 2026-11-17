"""Quick backend endpoint probe — run while backend is live on port 8000."""
import urllib.request
import json
import sys

BASE = "http://localhost:8000"


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=15) as r:
            return json.loads(r.read())
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
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def post_form(path, fields):
    boundary = "boundary_ns"
    body = b""
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


results = {}

print("=== GET /api/health ===")
r = get("/api/health")
results["health"] = r
print(json.dumps(r, indent=2)[:600])

print("\n=== GET /api/model/metrics ===")
r = get("/api/model/metrics")
results["model_metrics"] = r
print(json.dumps(r, indent=2)[:600])

print("\n=== GET /api/training-status ===")
r = get("/api/training-status")
results["training_status"] = r
print(json.dumps(r, indent=2)[:400])

print("\n=== POST /api/analyze/text ===")
r = post_form("/api/analyze/text", {
    "text": "I feel very sad, hopeless, and cannot sleep. Nothing brings me joy anymore."
})
results["analyze_text"] = r
print(json.dumps(r, indent=2)[:1000])

print("\n=== POST /session/start ===")
r = post_json("/session/start", {})
results["session_start"] = r
print(json.dumps(r, indent=2)[:600])

print("\n=== GET /api/recommendations ===")
r = get("/api/recommendations")
results["recommendations"] = r
print(json.dumps(r, indent=2)[:600])

# Save full results
with open("scripts/backend_probe_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n\nFull results saved to scripts/backend_probe_results.json")
print("\nSUMMARY:")
for k, v in results.items():
    has_error = "error" in v if isinstance(v, dict) else False
    print(f"  {'❌' if has_error else '✅'} {k}: {'ERROR: ' + v['error'] if has_error else 'OK'}")
