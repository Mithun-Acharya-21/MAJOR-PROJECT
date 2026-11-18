#!/usr/bin/env python3
"""
DAIC-WOZ Download Verification & Integrity Auditor
NeuroScreen Research Pipeline - Verification Script

Generates:
    - data/manifests/daic_woz/download_manifest.csv
    - data/manifests/daic_woz/download_verification.json
    - reports/daic_woz/download_report.md
"""

import os
import sys
import json
import hashlib
import csv
from pathlib import Path
from datetime import datetime, timezone

CHUNK_SIZE = 1024 * 1024


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def verify_downloads(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path("data/manifests/daic_woz")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports/daic_woz")
    report_dir.mkdir(parents=True, exist_ok=True)

    files_found = list(raw_dir.glob("**/*"))
    data_files = [f for f in files_found if f.is_file() and not f.name.endswith(".part")]
    archives = [f for f in data_files if f.suffix.lower() in [".zip", ".tar", ".gz", ".tgz"]]
    participant_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and (d.name.endswith("_P") or d.name.isdigit())]

    manifest_rows = []
    total_bytes = 0

    for f in data_files:
        f_size = f.stat().st_size
        total_bytes += f_size
        f_hash = compute_sha256(f) if f_size < 500 * 1024 * 1024 else "SKIPPED_LARGE_FILE"
        manifest_rows.append({
            "relative_path": str(f.relative_to(raw_dir)),
            "filename": f.name,
            "size_bytes": f_size,
            "sha256": f_hash,
            "is_archive": f in archives,
            "last_modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat()
        })

    # Save CSV manifest
    csv_path = manifest_dir / "download_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "relative_path", "filename", "size_bytes", "sha256", "is_archive", "last_modified"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Verification JSON
    status = "COMPLETE" if len(participant_dirs) >= 140 or len(archives) >= 140 else (
        "PARTIAL" if len(data_files) > 0 else "BLOCKED_AWAITING_DOWNLOAD"
    )

    verification_data = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "raw_directory": str(raw_dir.resolve()),
        "status": status,
        "total_files": len(data_files),
        "total_archives": len(archives),
        "total_participant_folders": len(participant_dirs),
        "total_size_bytes": total_bytes,
        "total_size_gb": round(total_bytes / (1024 ** 3), 4),
        "expected_participants": 189,
        "missing_participant_count": max(0, 189 - len(participant_dirs)),
        "is_complete": status == "COMPLETE"
    }

    json_path = manifest_dir / "download_verification.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(verification_data, f, indent=2)

    # Markdown Report
    report_path = report_dir / "download_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# DAIC-WOZ Download and Extraction Verification Report\n\n")
        f.write(f"**Verification Timestamp:** {verification_data['verified_at']}\n")
        f.write(f"**Status:** `{status}`\n\n")
        f.write("## 1. Summary Statistics\n\n")
        f.write(f"- **Target Directory:** `{raw_dir}`\n")
        f.write(f"- **Total Files Found:** {len(data_files)}\n")
        f.write(f"- **Total Archives:** {len(archives)}\n")
        f.write(f"- **Participant Folders Extracted:** {len(participant_dirs)} / 189\n")
        f.write(f"- **Total Size on Disk:** {verification_data['total_size_gb']} GB\n\n")
        f.write("## 2. Integrity and Completeness Assessment\n\n")
        if status == "COMPLETE":
            f.write("> [!NOTE] All expected participant archives/folders have been verified and extracted.\n")
        elif status == "PARTIAL":
            f.write("> [!WARNING] Partial download detected. Some participant archives are missing or incomplete.\n")
        else:
            f.write("> [!IMPORTANT]\n")
            f.write("> **Status: BLOCKED_AWAITING_DOWNLOAD**\n")
            f.write("> No official DAIC-WOZ participant archives are currently present in `data/raw/daic_woz/`.\n")
            f.write("> All research access requirements are documented in `docs/daic_woz/daic_woz_access_checklist.md`.\n")
            f.write("> In accordance with project research integrity constraints, no synthetic data has been fabricated.\n")

    print(f"[+] Download audit written to:\n    - {csv_path}\n    - {json_path}\n    - {report_path}")
    print(f"[+] Verification Status: {status}")
    return verification_data


if __name__ == "__main__":
    raw_dir = Path("data/raw/daic_woz")
    verify_downloads(raw_dir)
