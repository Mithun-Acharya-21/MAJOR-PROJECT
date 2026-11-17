#!/usr/bin/env python3
"""Smoke-test EmotionDep backend endpoints."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://localhost:8000"

# Minimal valid JPEG (1x1)
JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432"
    "ffdb004301090909090c0b0c180d0d1832211c213232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232"
    "ffc00011080001000103011100021100031101ffc4001500010100000000000000000000000000080706ffc40014100100000000000000000000000000000000ffda000801010000003f00d2cfd0ffd9"
)


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path) as resp:
        return json.loads(resp.read())


def post_multipart(path: str, fields: dict | None = None, files: list | None = None) -> dict:
    boundary = "----EmotionDepTest"
    body = io.BytesIO()
    for key, value in (fields or {}).items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        body.write(f"{value}\r\n".encode())
    for name, filename, content, ctype in files or []:
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        body.write(content)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(BASE + path, data=body.getvalue(), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def post_urlencoded(path: str, fields: dict) -> dict:
    import urllib.parse

    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main() -> int:
    results: dict[str, str] = {}

    tests = [
        ("health", lambda: get("/api/health")["status"]),
        ("metrics", lambda: get("/api/model/metrics")["accuracy"]),
        ("baselines", lambda: get("/api/baselines")["models"][0]["name"]),
        ("recommendations", lambda: "ok" if get("/api/recommendations") else "fail"),
        (
            "text",
            lambda: post_urlencoded(
                "/api/analyze/text",
                {"text": "I feel tired and hopeless lately and cannot concentrate."},
            )["probability"],
        ),
        (
            "photo",
            lambda: post_multipart(
                "/api/analyze/photo",
                files=[("file", "test.jpg", JPEG, "image/jpeg")],
            )["probability"],
        ),
        (
            "face",
            lambda: post_multipart(
                "/api/analyze/face",
                files=[
                    ("frames", "f0.jpg", JPEG, "image/jpeg"),
                    ("frames", "f1.jpg", JPEG, "image/jpeg"),
                ],
            )["probability"],
        ),
        (
            "multimodal",
            lambda: post_multipart(
                "/api/analyze/multimodal",
                fields={"text": "I feel tired and hopeless lately."},
                files=[("photo", "test.jpg", JPEG, "image/jpeg")],
            )["probability"],
        ),
        ("text_invalid", lambda: _expect_400_text()),
        ("multimodal_empty", lambda: _expect_400_multimodal()),
    ]

    for name, fn in tests:
        try:
            results[name] = f"PASS ({fn()})"
        except urllib.error.HTTPError as e:
            results[name] = f"HTTP {e.code}"
        except Exception as e:
            results[name] = f"FAIL: {e}"

    print(json.dumps(results, indent=2))
    failed = [k for k, v in results.items() if not v.startswith("PASS")]
    return 1 if failed else 0


def _expect_400_text() -> str:
    try:
        post_urlencoded("/api/analyze/text", {"text": "   "})
        return "expected 400"
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return "400 as expected"
        raise


def _expect_400_multimodal() -> str:
    try:
        post_multipart("/api/analyze/multimodal", fields={})
        return "expected 400"
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return "400 as expected"
        raise


if __name__ == "__main__":
    sys.exit(main())
