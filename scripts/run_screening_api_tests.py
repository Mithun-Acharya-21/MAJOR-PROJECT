"""
Run test_screening_api.py tests against a live server.
This is needed because TestClient hangs in the sandbox.
Requires the backend to be running on localhost:8000.
"""
import sys
import subprocess
import urllib.request
import json

BASE = "http://localhost:8000"

# Verify server is running first
try:
    with urllib.request.urlopen(BASE + "/api/health", timeout=5) as r:
        health = json.loads(r.read())
        assert health.get("status") == "healthy", f"Server not healthy: {health}"
    print(f"✅ Server healthy on {BASE}")
except Exception as e:
    print(f"❌ Server not reachable: {e}")
    sys.exit(1)

# Run the screening API tests
print("\nRunning test_screening_api.py...")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_screening_api.py", "-v", "--tb=short"],
    capture_output=True, text=True
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])
sys.exit(result.returncode)
