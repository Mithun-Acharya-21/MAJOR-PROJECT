"""Read-only audit of the datasets already present in the repository.

Reports, without modifying anything:
  - top-level data directories and their sizes
  - EATD manifest / participant / split integrity (Phase 1 artifacts)
  - existing Phase 2 result artifacts (existence only)
  - free disk space

Usage:
  .venv/Scripts/python scripts/datasets/audit_existing_datasets.py [--json PATH] [--quiet]
Exit codes: 0 ok, 1 an existing dataset failed an integrity check.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA, EXIT_FAIL, EXIT_OK, MANIFESTS, PROJECT_ROOT, REPORTS,
    count_files, dir_size, free_bytes, get_logger, human_bytes, utcnow, write_json,
)

EATD_MANIFEST = DATA / "processed" / "eatd" / "manifest.csv"
EATD_PARTICIPANTS = DATA / "processed" / "eatd" / "participants.csv"
EATD_SPLITS = DATA / "processed" / "eatd" / "splits.csv"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def audit_eatd(log) -> dict:
    manifest = _read_csv(EATD_MANIFEST)
    participants = _read_csv(EATD_PARTICIPANTS)
    splits = {r["participant_key"]: r["split"] for r in _read_csv(EATD_SPLITS)}

    issues: list[str] = []
    if not manifest:
        issues.append("manifest.csv missing or empty")
    if not participants:
        issues.append("participants.csv missing or empty")
    if not splits:
        issues.append("splits.csv missing or empty")

    keys = {r["participant_key"] for r in manifest}
    if splits and keys != set(splits):
        issues.append(f"participant set mismatch: manifest {len(keys)} vs splits {len(splits)}")
    if len(keys) != len({(r["source_split"], r["source_subject_id"]) for r in manifest}):
        issues.append("participant_key is not unique per (source_split, source_subject_id)")

    missing_text = [r["text_path"] for r in manifest if not (PROJECT_ROOT / r["text_path"]).exists()]
    missing_audio = [r["audio_path"] for r in manifest if not (PROJECT_ROOT / r["audio_path"]).exists()]
    if missing_text:
        issues.append(f"{len(missing_text)} manifest text paths missing on disk")
    if missing_audio:
        issues.append(f"{len(missing_audio)} manifest audio paths missing on disk")

    label_by_participant: dict[str, set[int]] = {}
    for r in manifest:
        label_by_participant.setdefault(r["participant_key"], set()).add(int(r["label"]))
    inconsistent = [k for k, v in label_by_participant.items() if len(v) > 1]
    if inconsistent:
        issues.append(f"{len(inconsistent)} participants have inconsistent session labels")

    split_by_key = Counter(splits.values())
    label_counts = Counter(int(r["label"]) for r in manifest)
    part_label = Counter()
    for key, labels in label_by_participant.items():
        part_label[int(next(iter(labels)))] += 1

    info = {
        "manifest_rows": len(manifest),
        "participants": len(keys),
        "sessions": len(manifest),
        "labels": {"0": label_counts.get(0, 0), "1": label_counts.get(1, 0)},
        "participants_by_label": {"0": part_label.get(0, 0), "1": part_label.get(1, 0)},
        "split_participants": dict(split_by_key),
        "sessions_by_split": dict(Counter(r["source_split"] or splits.get(r["participant_key"], "?") for r in manifest)),
        "audio_usable_sessions": sum(1 for r in manifest if str(r.get("audio_usable", "")).strip() in ("1", "1.0", "True", "true")),
        "text_files_on_disk": count_files(DATA / "eatd", ".txt"),
        "audio_files_on_disk": count_files(DATA / "eatd", ".wav"),
        "missing_text_paths": len(missing_text),
        "missing_audio_paths": len(missing_audio),
        "inconsistent_label_participants": len(inconsistent),
        "total_size": dir_size(DATA / "eatd"),
        "total_size_human": human_bytes(dir_size(DATA / "eatd")),
        "issues": issues,
        "ok": not issues,
    }
    for i in issues:
        log.warning("EATD: %s", i)
    return info


def audit_phase2(log) -> dict:
    results = sorted((PROJECT_ROOT / "results" / "phase2").rglob("*.json"))
    models = sorted(p.name for p in (PROJECT_ROOT / "models" / "experiments").glob("*") if p.is_dir())
    comparison = PROJECT_ROOT / "results" / "phase2" / "comparisons" / "comparison.json"
    return {
        "result_files": [str(p.relative_to(PROJECT_ROOT)) for p in results],
        "model_artifact_dirs": models,
        "comparison_present": comparison.exists(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=MANIFESTS / "existing_datasets_audit.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    log = get_logger("audit_existing_datasets")

    log.info("Phase A — read-only audit of %s", PROJECT_ROOT)
    eatd = audit_eatd(log)
    phase2 = audit_phase2(log)

    dirs = {}
    for p in sorted(DATA.iterdir()):
        if p.is_dir():
            dirs[p.name] = {"files": count_files(p), "bytes": dir_size(p)}

    report = {
        "generated_at": utcnow(),
        "project_root": str(PROJECT_ROOT),
        "disk": {"free_bytes": free_bytes(), "free_human": human_bytes(free_bytes())},
        "data_dirs": dirs,
        "eatd": eatd,
        "phase2": phase2,
    }
    write_json(args.json, report)
    log.info("disk free: %s", human_bytes(free_bytes()))
    for name, d in dirs.items():
        log.info("  data/%-12s %5d files  %8s", name, d["files"], human_bytes(d["bytes"]))
    log.info("EATD: %d participants, %d sessions, splits=%s, audio-usable=%d",
             eatd["participants"], eatd["sessions"], eatd["split_participants"], eatd["audio_usable_sessions"])
    log.info("Phase 2 artifacts: %d result files, %d model dirs", len(phase2["result_files"]), len(phase2["model_artifact_dirs"]))
    log.info("audit written to %s", args.json)
    if not eatd["ok"]:
        log.error("EATD integrity check failed")
        return EXIT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
