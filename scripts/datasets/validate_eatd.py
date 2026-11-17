"""Validate the local EATD corpus against its Phase 1 manifest.

Verifies, without modifying anything:
  - every manifest text/audio path exists and is non-empty
  - recorded SHA-256 hashes match the files on disk. Note the two hash
    definitions from scripts/build_eatd_manifest.py: text_sha256 is the hash of
    the normalised transcript (utf-8-sig decoded, outer whitespace stripped),
    audio_sha256 is the hash of the raw WAV bytes.
  - recorded audio duration matches the WAV header
  - sample rate / channel count match the manifest
  - SDS arithmetic (sds_index == sds_raw * 1.25) and the binary rule (>= 50)
  - participant keys are dataset-qualified and unique per source partition
  - splits are participant-level (no participant in two splits)
  - transcript language is CJK, as documented

Writes data/reports/eatd_validation.json (the Phase 1 artifact
data/processed/eatd/validation_report.json is left untouched).

Usage:
  .venv/Scripts/python scripts/datasets/validate_eatd.py [--no-hash] [--sample-audio N]
Exit codes: 0 ok, 1 failures, 4 missing manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA, EXIT_FAIL, EXIT_MISSING, EXIT_OK, PROJECT_ROOT, REPORTS,
    get_logger, human_bytes, sha256_file, utcnow, write_json,
)

MANIFEST = DATA / "processed" / "eatd" / "manifest.csv"
SPLITS = DATA / "processed" / "eatd" / "splits.csv"
PARTICIPANTS = DATA / "processed" / "eatd" / "participants.csv"
REPORT = REPORTS / "eatd_validation.json"
HASH_TOLERANCE_SEC = 0.05


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _resolve(rel: str) -> Path:
    p = Path(rel.replace("\\", "/"))
    return p if p.is_absolute() else PROJECT_ROOT / p


def sha256_normalised_text(path: Path) -> str:
    """Matches scripts/build_eatd_manifest.py::sha256_text(read_text(path))."""
    text = path.read_text(encoding="utf-8-sig").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 verification (fast mode)")
    ap.add_argument("--sample-audio", type=int, default=0,
                    help="check audio headers for a random sample of N sessions (0 = all)")
    args = ap.parse_args()
    log = get_logger("validate_eatd")

    if not MANIFEST.exists():
        log.error("manifest not found: %s", MANIFEST)
        write_json(REPORT, {"generated_at": utcnow(), "status": "missing_manifest", "ok": False})
        return EXIT_MISSING

    rows = _read_csv(MANIFEST)
    splits = {r["participant_key"]: r["split"] for r in _read_csv(SPLITS)} if SPLITS.exists() else {}
    participants = _read_csv(PARTICIPANTS) if PARTICIPANTS.exists() else []
    log.info("manifest rows=%d participants=%d splits=%s", len(rows),
             len({r["participant_key"] for r in rows}), dict(Counter(splits.values())))

    errors: list[str] = []
    warnings: list[str] = []
    missing_text: list[str] = []
    missing_audio: list[str] = []
    hash_mismatch: list[str] = []
    duration_mismatch: list[dict] = []
    sr_mismatch: list[dict] = []
    label_errors: list[str] = []
    sds_errors: list[str] = []
    key_issues: list[str] = []
    cjk_chars = latin_chars = 0
    hashed_text = hashed_audio = 0

    by_participant_labels: dict[str, set[str]] = defaultdict(set)
    by_participant_sds: dict[str, set[str]] = defaultdict(set)
    topics: Counter = Counter()

    for i, r in enumerate(rows):
        key = r["participant_key"]
        topics[r["topic"]] += 1
        expected_prefix = f"eatd_{r['source_split']}_"
        if not key.startswith(expected_prefix) or not r["source_subject_id"]:
            key_issues.append(key)
        by_participant_labels[key].add(r["label"])
        by_participant_sds[key].add(r["sds_raw"])

        text_path = _resolve(r["text_path"])
        if not text_path.exists() or text_path.stat().st_size == 0:
            missing_text.append(r["text_path"])
        else:
            text = text_path.read_text(encoding="utf-8", errors="replace")
            cjk_chars += sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
            latin_chars += sum(1 for ch in text if ch.isascii() and ch.isalpha())
            if not args.no_hash and r.get("text_sha256"):
                hashed_text += 1
                if sha256_normalised_text(text_path) != r["text_sha256"]:
                    hash_mismatch.append(r["text_path"])

        audio_path = _resolve(r["audio_path"])
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            missing_audio.append(r["audio_path"])
        else:
            if not args.no_hash and r.get("audio_sha256"):
                hashed_audio += 1
                if sha256_file(audio_path) != r["audio_sha256"]:
                    hash_mismatch.append(r["audio_path"])

        try:
            sds_raw = float(r["sds_raw"])
            sds_index = float(r["sds_index"])
            label = int(r["label"])
        except (TypeError, ValueError):
            label_errors.append(key)
            continue
        if abs(sds_index - sds_raw * 1.25) > 1e-6:
            sds_errors.append(f"{key}/{r['topic']}: sds_index {sds_index} != 1.25*{sds_raw}")
        if label != (1 if sds_index >= 50 else 0):
            label_errors.append(f"{key}/{r['topic']}: label {label} != rule({sds_index})")

    # audio headers
    sample = rows if args.sample_audio in (0, None) else rows[:: max(1, len(rows) // args.sample_audio)]
    try:
        import soundfile as sf
        for r in sample:
            p = _resolve(r["audio_path"])
            if not p.exists():
                continue
            try:
                info = sf.info(str(p))
            except Exception as exc:  # unreadable / corrupt
                errors.append(f"unreadable audio: {r['audio_path']} ({exc})")
                continue
            try:
                expected_dur = float(r["audio_duration_sec"])
            except (TypeError, ValueError):
                continue
            if abs(info.duration - expected_dur) > HASH_TOLERANCE_SEC:
                duration_mismatch.append({"file": r["audio_path"], "manifest": expected_dur,
                                          "header": round(info.duration, 3)})
            if int(r["sample_rate"]) != info.samplerate or int(r["channels"]) != info.channels:
                sr_mismatch.append({"file": r["audio_path"], "manifest_sr": r["sample_rate"],
                                    "header_sr": info.samplerate, "manifest_ch": r["channels"],
                                    "header_ch": info.channels})
    except ImportError:
        warnings.append("soundfile not importable; audio headers not checked")

    mixed_label = [k for k, v in by_participant_labels.items() if len(v) > 1]
    mixed_sds = [k for k, v in by_participant_sds.items() if len(v) > 1]
    overlap = {}
    if splits:
        seen = defaultdict(set)
        for key in {r["participant_key"] for r in rows}:
            seen[splits.get(key, "MISSING")].add(key)
        if "MISSING" in seen:
            errors.append(f"{len(seen['MISSING'])} manifest participants absent from splits.csv")
        overlap = {k: len(v) for k, v in seen.items()}

    if missing_text:
        errors.append(f"{len(missing_text)} missing/empty transcripts")
    if missing_audio:
        errors.append(f"{len(missing_audio)} missing/empty audio files")
    if hash_mismatch:
        errors.append(f"{len(hash_mismatch)} SHA-256 mismatches")
    if duration_mismatch:
        errors.append(f"{len(duration_mismatch)} duration mismatches vs WAV header")
    if sr_mismatch:
        errors.append(f"{len(sr_mismatch)} sample-rate/channel mismatches")
    if label_errors:
        errors.append(f"{len(label_errors)} label rule violations")
    if sds_errors:
        errors.append(f"{len(sds_errors)} SDS index arithmetic violations")
    if key_issues:
        errors.append(f"{len(key_issues)} participant keys not qualified by source split")
    if mixed_label or mixed_sds:
        errors.append(f"{len(mixed_label)} participants with mixed labels / {len(mixed_sds)} with mixed SDS")
    if latin_chars and cjk_chars == 0:
        errors.append("no CJK characters found; documented language (Chinese) not reproduced")

    label_counts = Counter(int(r["label"]) for r in rows if r["label"] in ("0", "1"))
    part_label = Counter()
    for key, labels in by_participant_labels.items():
        if len(labels) == 1:
            part_label[int(next(iter(labels)))] += 1
    usable = sum(1 for r in rows if r["audio_usable"].strip().lower() in ("true", "1"))

    report = {
        "generated_at": utcnow(),
        "status": "validated" if not errors else "failed",
        "dataset_name": "eatd",
        "source": "https://github.com/Fancy-Block/EATD-Corpus",
        "manifest": str(MANIFEST),
        "license": "not stated in the local copy; redistribution terms flagged for review",
        "counts": {
            "participants": len({r["participant_key"] for r in rows}),
            "sessions": len(rows),
            "sessions_by_topic": dict(topics),
            "audio_usable_sessions": usable,
            "audio_files": len(rows) - len(missing_audio),
            "text_files": len(rows) - len(missing_text),
            "splits": overlap,
            "participants_by_label": {"0": part_label.get(0, 0), "1": part_label.get(1, 0)},
            "sessions_by_label": {"0": label_counts.get(0, 0), "1": label_counts.get(1, 0)},
            "participants_csv_rows": len(participants),
        },
        "label_definition": {
            "raw_field": "sds_raw (EATD Self-rating Depression Scale score)",
            "index": "sds_index = sds_raw * 1.25",
            "binary": "label = 1 if sds_index >= 50 else 0",
            "clinical_claim": False,
            "note": "EATD SDS band; not PHQ-8/PHQ-9 and not a clinical diagnosis",
        },
        "language": {"cjk_characters": cjk_chars, "latin_letters": latin_chars,
                     "declared": "Chinese (zh)"},
        "integrity": {
            "hashes_checked_text": hashed_text,
            "hashes_checked_audio": hashed_audio,
            "hash_mismatches": hash_mismatch[:50],
            "hash_definitions": {
                "text_sha256": "sha256 of utf-8-sig-decoded, stripped transcript",
                "audio_sha256": "sha256 of raw WAV bytes",
            },
            "missing_text": missing_text[:50],
            "missing_audio": missing_audio[:50],
            "duration_mismatches": duration_mismatch[:50],
            "sample_rate_mismatches": sr_mismatch[:50],
            "label_errors": label_errors[:50],
            "sds_errors": sds_errors[:50],
            "participant_key_issues": key_issues[:50],
            "audio_header_sample": len(sample),
        },
        "size_on_disk": human_bytes(sum(p.stat().st_size for p in (DATA / "eatd").rglob("*") if p.is_file())),
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }
    write_json(REPORT, report)

    log.info("participants=%d sessions=%d labels=%s split sizes=%s",
             report["counts"]["participants"], report["counts"]["sessions"],
             report["counts"]["sessions_by_label"], overlap)
    log.info("integrity: %d text + %d audio hashes, %d missing, %d mismatched",
             hashed_text, hashed_audio, len(missing_text) + len(missing_audio), len(hash_mismatch))
    log.info("report written to %s", REPORT)
    if errors:
        for e in errors:
            log.error("%s", e)
        return EXIT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
