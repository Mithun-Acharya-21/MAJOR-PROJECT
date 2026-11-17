#!/usr/bin/env python3
"""Build the verified EATD participant manifest and participant-level splits.

Why this script exists
----------------------
``data/eatd`` was released as two source partitions (``train`` and
``validation``) whose numeric folder ids are only unique *within* a partition:
58 numeric ids appear in both partitions, but those folders contain different
transcripts, different audio, different durations, and (for 56 of 58) different
SDS scores. They are distinct participants, not duplicates.

This script therefore treats ``(source_partition, folder_id)`` as the
participant key and verifies that decision from content hashes rather than
assuming it.

Labels (verified, not assumed)
------------------------------
* ``label.txt``      — Zung Self-rating Depression Scale (SDS) raw total (20-80).
* ``new_label.txt``  — SDS index = raw total x 1.25 (25-100), verified for all
                       162 participants.
* binary label       — ``1`` when SDS index >= 50, else ``0``. This reproduces
                       ``data/eatd_text.csv`` for all 162 participants with
                       zero mismatches.

Audio/text alignment
--------------------
Each participant folder has three prompted sessions (positive / neutral /
negative), each with a transcript ``.txt`` and a 16 kHz WAV ``.wav``.
``_out.wav`` is a processed variant of the same session (shorter for many
files) and is NOT a separate sample.

Outputs (all derived; source data is never modified)
----------------------------------------------------
    data/processed/eatd/manifest.csv           one row per participant×topic
    data/processed/eatd/participants.csv       one row per participant
    data/processed/eatd/splits.csv             participant-level split assignment
    data/processed/eatd/validation_report.json integrity + statistics
    data/processed/eatd/split_report.json      split provenance + leakage checks

Usage:
    python scripts/build_eatd_manifest.py
    python scripts/build_eatd_manifest.py --seed 7 --ratios 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EATD_ROOT = PROJECT_ROOT / "data" / "eatd"
LEGACY_CSV = PROJECT_ROOT / "data" / "eatd_text.csv"
LEGACY_SPLITS = PROJECT_ROOT / "data" / "processed" / "eatd_splits" / "eatd_subject_splits.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "eatd"

TOPICS = ("positive", "neutral", "negative")
SOURCE_SPLITS = ("train", "validation")
SDS_DEPRESSED_THRESHOLD = 50.0
SDS_RAW_RANGE = (20.0, 80.0)
SDS_INDEX_RANGE = (25.0, 100.0)

# Audio quality thresholds. A session below MIN_USABLE is retained in the
# manifest (text is still usable) but flagged for exclusion from audio models.
MIN_USABLE_AUDIO_SECONDS = 1.0
SHORT_AUDIO_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def read_float(path: Path) -> float:
    return float(read_text(path))


def audio_info(path: Path) -> Dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return {
            "sample_rate": rate,
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "frames": frames,
            "duration_sec": round(frames / float(rate), 3) if rate else 0.0,
        }


def participant_key(source_split: str, subject_id: str) -> str:
    return f"eatd_{source_split}_{subject_id}"


def load_legacy_csv() -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return ``(split, subject_id) -> {label, texts{topic: text}}``."""
    if not LEGACY_CSV.exists():
        raise SystemExit(f"FATAL: legacy text CSV not found: {LEGACY_CSV}")

    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    with open(LEGACY_CSV, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups[(row["split"].strip(), row["subject_id"].strip())].append(row)

    legacy: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, rows in groups.items():
        labels = {str(row["label"]).strip() for row in rows}
        if len(labels) != 1:
            raise SystemExit(f"FATAL: legacy rows for {key} have inconsistent labels: {labels}")
        # The legacy CSV stores CRLF inside quoted fields; raw .txt files use
        # LF. Normalize line endings before comparing content.
        texts = {
            str(row["topic"]).strip(): str(row["text"])
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
            for row in rows
        }
        legacy[key] = {"label": int(float(next(iter(labels)))), "texts": texts}
    return legacy


def discover_participants() -> List[Tuple[str, str, Path]]:
    """Return sorted ``(source_split, subject_id, directory)`` triples."""
    found: List[Tuple[str, str, Path]] = []
    for source_split in SOURCE_SPLITS:
        directory = EATD_ROOT / source_split
        if not directory.is_dir():
            raise SystemExit(f"FATAL: missing EATD partition: {directory}")
        for child in directory.iterdir():
            if child.is_dir() and child.name.isdigit():
                found.append((source_split, child.name, child))
    found.sort(key=lambda item: (item[0], int(item[1])))
    if not found:
        raise SystemExit(f"FATAL: no participant directories under {EATD_ROOT}")
    return found


def legacy_participant_keys() -> set:
    """Participants kept by the legacy 81-subject split (for comparison only)."""
    if not LEGACY_SPLITS.exists():
        return set()
    with open(LEGACY_SPLITS, newline="", encoding="utf-8") as handle:
        return {
            f"eatd_{row['source_split'].strip()}_{row['orig_subject_id'].strip()}"
            for row in csv.DictReader(handle)
        }


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_participants(
    participants: Iterable[Tuple[str, str, Path]],
    legacy: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    """Audit every participant; return (manifest_rows, participant_rows, errors, warnings)."""
    manifest_rows: List[Dict[str, Any]] = []
    participant_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    audio_hashes: Dict[str, str] = {}
    text_hashes: Dict[str, str] = {}
    transcript_groups: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    audio_groups: Dict[Tuple[str, ...], List[str]] = defaultdict(list)

    for source_split, subject_id, directory in participants:
        key = participant_key(source_split, subject_id)
        participant_errors: List[str] = []

        # --- SDS labels ---------------------------------------------------
        sds_raw = sds_index = None
        try:
            sds_raw = read_float(directory / "label.txt")
            sds_index = read_float(directory / "new_label.txt")
        except Exception as exc:
            participant_errors.append(f"{key}: cannot read SDS files: {exc}")

        if sds_raw is not None and not (SDS_RAW_RANGE[0] <= sds_raw <= SDS_RAW_RANGE[1]):
            participant_errors.append(f"{key}: SDS raw {sds_raw} outside {SDS_RAW_RANGE}")
        if sds_index is not None and not (SDS_INDEX_RANGE[0] <= sds_index <= SDS_INDEX_RANGE[1]):
            participant_errors.append(f"{key}: SDS index {sds_index} outside {SDS_INDEX_RANGE}")
        if sds_raw is not None and sds_index is not None:
            expected_index = round(sds_raw * 1.25, 4)
            if abs(sds_index - expected_index) > 1e-3:
                participant_errors.append(
                    f"{key}: SDS index {sds_index} != 1.25 x raw {sds_raw} (expected {expected_index})"
                )

        label = None
        if sds_index is not None:
            label = 1 if sds_index >= SDS_DEPRESSED_THRESHOLD else 0

        # --- Cross-check against the legacy CSV ---------------------------
        legacy_entry = legacy.get((source_split, subject_id))
        if legacy_entry is None:
            participant_errors.append(f"{key}: no corresponding rows in data/eatd_text.csv")
        elif label is not None and int(legacy_entry["label"]) != label:
            participant_errors.append(
                f"{key}: legacy label {legacy_entry['label']} != SDS-derived label {label} "
                f"(index {sds_index})"
            )

        # --- Sessions -----------------------------------------------------
        text_list: List[str] = []
        audio_list: List[str] = []
        durations: List[float] = []
        usable_audio_count = 0

        for topic in TOPICS:
            text_path = directory / f"{topic}.txt"
            audio_path = directory / f"{topic}.wav"
            processed_path = directory / f"{topic}_out.wav"

            if not text_path.exists():
                participant_errors.append(f"{key}/{topic}: missing transcript")
                continue
            if not audio_path.exists():
                participant_errors.append(f"{key}/{topic}: missing audio")
                continue

            text = read_text(text_path)
            if not text:
                participant_errors.append(f"{key}/{topic}: empty transcript")
            text_list.append(text)

            if legacy_entry is not None:
                legacy_text = (legacy_entry["texts"].get(topic) or "").strip()
                if legacy_text and legacy_text != text:
                    participant_errors.append(f"{key}/{topic}: transcript != legacy CSV text")

            try:
                info = audio_info(audio_path)
            except Exception as exc:
                participant_errors.append(f"{key}/{topic}: unreadable audio: {exc}")
                continue

            quality_flags: List[str] = []
            audio_usable = True

            if info["sample_rate"] != 16000:
                participant_errors.append(
                    f"{key}/{topic}: sample rate {info['sample_rate']} != 16000"
                )
                audio_usable = False
            if info["channels"] not in (1, 2):
                participant_errors.append(f"{key}/{topic}: unexpected channels {info['channels']}")
                audio_usable = False
            if info["duration_sec"] < MIN_USABLE_AUDIO_SECONDS:
                quality_flags.append(f"short_audio_lt_{int(MIN_USABLE_AUDIO_SECONDS * 1000)}ms")
                audio_usable = False
            elif info["duration_sec"] < SHORT_AUDIO_SECONDS:
                quality_flags.append(f"short_audio_lt_{int(SHORT_AUDIO_SECONDS)}s")
            usable_audio_count += int(audio_usable)

            audio_hash = sha256_file(audio_path)
            text_hash = sha256_text(text)
            audio_list.append(audio_hash)
            durations.append(info["duration_sec"])

            if audio_hash in audio_hashes:
                participant_errors.append(
                    f"{key}/{topic}: duplicate audio content with {audio_hashes[audio_hash]}"
                )
            else:
                audio_hashes[audio_hash] = f"{key}/{topic}"
            if text_hash in text_hashes:
                participant_errors.append(
                    f"{key}/{topic}: duplicate transcript with {text_hashes[text_hash]}"
                )
            else:
                text_hashes[text_hash] = f"{key}/{topic}"

            processed_duration = None
            processed_delta = None
            if processed_path.exists():
                try:
                    processed_duration = audio_info(processed_path)["duration_sec"]
                    processed_delta = round(processed_duration - info["duration_sec"], 3)
                except Exception as exc:
                    warnings.append(f"{key}/{topic}: unreadable _out.wav: {exc}")

            manifest_rows.append(
                {
                    "participant_key": key,
                    "source_split": source_split,
                    "source_subject_id": subject_id,
                    "participant_index": int(subject_id),
                    "topic": topic,
                    "text_path": str(text_path.relative_to(PROJECT_ROOT)),
                    "audio_path": str(audio_path.relative_to(PROJECT_ROOT)),
                    "audio_processed_path": (
                        str(processed_path.relative_to(PROJECT_ROOT))
                        if processed_path.exists()
                        else ""
                    ),
                    "text_sha256": text_hash,
                    "audio_sha256": audio_hash,
                    "transcript_chars": len(text),
                    "audio_duration_sec": info["duration_sec"],
                    "audio_processed_duration_sec": processed_duration,
                    "audio_processed_delta_sec": processed_delta,
                    "sample_rate": info["sample_rate"],
                    "channels": info["channels"],
                    "audio_usable": audio_usable,
                    "quality_flags": ";".join(quality_flags),
                    "sds_raw": sds_raw,
                    "sds_index": sds_index,
                    "label": label,
                    "label_rule": f"1 if sds_index>={SDS_DEPRESSED_THRESHOLD:g} else 0",
                }
            )

        if len(text_list) == len(TOPICS):
            transcript_groups[tuple(sorted(text_list))].append(key)
        if len(audio_list) == len(TOPICS):
            audio_groups[tuple(sorted(audio_list))].append(key)

        participant_rows.append(
            {
                "participant_key": key,
                "source_split": source_split,
                "source_subject_id": subject_id,
                "participant_index": int(subject_id),
                "sds_raw": sds_raw,
                "sds_index": sds_index,
                "label": label,
                "n_sessions": len(text_list),
                "has_all_transcripts": len(text_list) == len(TOPICS),
                "has_all_audio": len(audio_list) == len(TOPICS),
                "usable_audio_sessions": usable_audio_count,
                "mean_audio_duration_sec": (
                    round(statistics.mean(durations), 3) if durations else None
                ),
                "mean_transcript_chars": (
                    round(statistics.mean(len(text) for text in text_list), 1)
                    if text_list
                    else None
                ),
            }
        )

        errors.extend(participant_errors)

    # --- Cross-participant duplicate detection ----------------------------
    for group_key, keys in transcript_groups.items():
        if len(keys) > 1:
            errors.append(f"duplicate transcript set across participants: {keys}")
    for group_key, keys in audio_groups.items():
        if len(keys) > 1:
            errors.append(f"duplicate audio set across participants: {keys}")

    return manifest_rows, participant_rows, errors, warnings


# ---------------------------------------------------------------------------
# Participant-level split
# ---------------------------------------------------------------------------

def make_participant_splits(
    participant_labels: Dict[str, int],
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Dict[str, str]:
    """Stratified participant-level split, deterministic for a fixed seed."""
    if abs(sum(ratios) - 1.0) > 1e-9 or any(not (0.0 < r < 1.0) for r in ratios):
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")

    rng = random.Random(seed)
    by_label: Dict[int, List[str]] = defaultdict(list)
    for key, label in participant_labels.items():
        by_label[int(label)].append(key)

    assignment: Dict[str, str] = {}
    for label in sorted(by_label):
        keys = sorted(by_label[label])
        rng.shuffle(keys)
        n = len(keys)
        if n >= 3:
            n_val = max(1, round(n * ratios[1]))
            n_test = max(1, round(n * ratios[2]))
        else:
            n_val = 0
            n_test = 0
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            while n_val + n_test > n - 1:
                if n_test > 0:
                    n_test -= 1
                elif n_val > 0:
                    n_val -= 1

        for index, key in enumerate(keys):
            if index < n_train:
                assignment[key] = "train"
            elif index < n_train + n_val:
                assignment[key] = "validation"
            else:
                assignment[key] = "test"
    return assignment


def verify_splits(assignment: Dict[str, str], labels: Dict[str, int]) -> Dict[str, Any]:
    per_split: Dict[str, set] = defaultdict(set)
    for key, split in assignment.items():
        per_split[split].add(key)

    overlaps = {
        f"{a}&{b}": sorted(per_split[a] & per_split[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    uncovered = sorted(set(labels) - set(assignment))
    unknown = sorted(set(assignment) - set(labels))
    class_presence = {
        split: sorted({labels[key] for key in keys})
        for split, keys in per_split.items()
        if keys
    }
    return {
        "overlaps": overlaps,
        "participants_in_multiple_splits": sum(len(value) for value in overlaps.values()),
        "uncovered_participants": uncovered,
        "unknown_participants": unknown,
        "class_presence": class_presence,
        "all_zero": all(not value for value in overlaps.values()) and not uncovered and not unknown,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_reports(
    manifest_rows: List[Dict[str, Any]],
    participant_rows: List[Dict[str, Any]],
    errors: List[str],
    warnings: List[str],
    assignment: Dict[str, str],
    seed: int,
    ratios: Tuple[float, float, float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    labels = {row["participant_key"]: int(row["label"]) for row in participant_rows}
    class_counts = Counter(labels.values())
    durations = [row["audio_duration_sec"] for row in manifest_rows]
    chars = [row["transcript_chars"] for row in manifest_rows]

    sds_by_class: Dict[str, Dict[str, float]] = {}
    for label in sorted(class_counts):
        values = [
            row["sds_index"] for row in participant_rows if int(row["label"]) == label
        ]
        sds_by_class[str(label)] = {
            "n": len(values),
            "min": round(min(values), 2),
            "median": round(statistics.median(values), 2),
            "max": round(max(values), 2),
        }

    sessions_per_participant = Counter(row["participant_key"] for row in manifest_rows)
    duplicate_audio = (
        len(manifest_rows) - len({row["audio_sha256"] for row in manifest_rows})
    )
    duplicate_text = (
        len(manifest_rows) - len({row["text_sha256"] for row in manifest_rows})
    )

    split_summary: Dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        keys = [key for key, value in assignment.items() if value == split]
        split_rows = [row for row in manifest_rows if row["participant_key"] in set(keys)]
        split_summary[split] = {
            "participants": len(keys),
            "sessions": len(split_rows),
            "label_counts": dict(
                Counter(int(labels[key]) for key in keys)
            ),
            "sds_index_median": (
                round(
                    statistics.median(
                        row["sds_index"] for row in participant_rows if row["participant_key"] in set(keys)
                    ),
                    2,
                )
                if keys
                else None
            ),
        }

    quality_flag_counts = Counter()
    for row in manifest_rows:
        for flag in (row.get("quality_flags") or "").split(";"):
            if flag:
                quality_flag_counts[flag] += 1
    usable_sessions = sum(1 for row in manifest_rows if row.get("audio_usable"))

    validation = {
        "dataset": "EATD (Chinese interview corpus, local copy)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": (
            "fail" if errors else ("pass_with_quality_flags" if quality_flag_counts else "pass")
        ),
        "source_root": str(EATD_ROOT.relative_to(PROJECT_ROOT)),
        "participant_key": "(source_split, source_subject_id) — numeric ids are partition-local",
        "counts": {
            "participants": len(participant_rows),
            "sessions": len(manifest_rows),
            "source_partitions": dict(Counter(row["source_split"] for row in participant_rows)),
            "sessions_per_participant": dict(Counter(sessions_per_participant.values())),
            "label_counts": {"0": class_counts.get(0, 0), "1": class_counts.get(1, 0)},
            "sessions_audio_usable": usable_sessions,
            "sessions_audio_flagged": len(manifest_rows) - usable_sessions,
        },
        "quality_flags": dict(quality_flag_counts),
        "quality_policy": {
            "min_usable_audio_sec": MIN_USABLE_AUDIO_SECONDS,
            "short_audio_sec": SHORT_AUDIO_SECONDS,
            "note": (
                "Flagged sessions are retained in the manifest. Text training may "
                "use them; audio training must filter on audio_usable == True. "
                "Quality flags never change labels or participant assignment."
            ),
        },
        "label_definition": {
            "instrument": "Zung Self-rating Depression Scale (SDS)",
            "raw_total_file": "label.txt",
            "raw_range_observed": [
                min(row["sds_raw"] for row in participant_rows),
                max(row["sds_raw"] for row in participant_rows),
            ],
            "index_file": "new_label.txt",
            "index_formula": "index = raw_total x 1.25",
            "index_range_observed": [
                min(row["sds_index"] for row in participant_rows),
                max(row["sds_index"] for row in participant_rows),
            ],
            "binary_rule": f"label = 1 if index >= {SDS_DEPRESSED_THRESHOLD:g} else 0",
            "binary_rule_verified_against": "data/eatd_text.csv",
            "mismatches_vs_legacy_csv": sum(
                1 for error in errors if "legacy label" in error
            ),
            "note": (
                "SDS is a self-report symptom scale, not a clinical diagnosis. "
                "Do not mix these labels with PHQ-8/PHQ-9 or clinical diagnoses."
            ),
        },
        "sds_by_class": sds_by_class,
        "integrity_checks": {
            "participant_dirs_match_legacy_groups": not any(
                "no corresponding rows" in error for error in errors
            ),
            "all_transcripts_present": all(
                row["has_all_transcripts"] for row in participant_rows
            ),
            "all_audio_present": all(row["has_all_audio"] for row in participant_rows),
            "all_audio_16khz": all(row["sample_rate"] == 16000 for row in manifest_rows),
            "all_text_matches_legacy_csv": not any(
                "transcript != legacy CSV" in error for error in errors
            ),
            "sds_index_equals_raw_x_1_25": not any(
                "SDS index" in error for error in errors
            ),
            "binary_label_reproduces_legacy_csv": not any(
                "legacy label" in error for error in errors
            ),
            "duplicate_audio_content": duplicate_audio,
            "duplicate_transcript_content": duplicate_text,
            "audio_out_treated_as_separate_sample": False,
            "audio_quality_flags_are_not_integrity_errors": True,
        },
        "audio": {
            "canonical_file": "<topic>.wav",
            "processed_variant": "<topic>_out.wav (derived; not a separate sample)",
            "sample_rate_hz": 16000,
            "channels_observed": sorted({row["channels"] for row in manifest_rows}),
            "duration_sec": {
                "min": min(durations),
                "median": round(statistics.median(durations), 2),
                "max": max(durations),
                "total_hours": round(sum(durations) / 3600.0, 3),
            },
            "transcript_chars": {
                "min": min(chars),
                "median": round(statistics.median(chars), 1),
                "max": max(chars),
            },
            "processed_variant_duration_diffs": sum(
                1
                for row in manifest_rows
                if row.get("audio_processed_delta_sec") is not None
                and abs(row["audio_processed_delta_sec"]) > 0.05
            ),
            "usable_sessions": usable_sessions,
            "flagged_sessions": len(manifest_rows) - usable_sessions,
        },
        "alignment": {
            "level": "participant x prompted topic",
            "topics": list(TOPICS),
            "statement": (
                "Each participant has one transcript and one canonical WAV per "
                "prompted topic. There are no word-level timestamps; alignment "
                "is guaranteed at the participant-session level and verified by "
                "exact transcript equality with the legacy CSV."
            ),
        },
        "modalities_present": ["text", "audio"],
        "modalities_absent": ["face/video"],
        "errors": errors,
        "warnings": warnings,
        "limitations": [
            "162 participants / 486 sessions is small; confidence intervals will be wide.",
            "EATD labels are Zung SDS self-report scores, not clinical diagnoses.",
            "No face or video data: this corpus supports text+voice fusion only.",
            "Each participant contributes three prompted sessions; sessions are not independent.",
            "Local copy usage/re-distribution terms must be confirmed before publication.",
        ],
    }

    legacy_kept = legacy_participant_keys()
    all_keys = set(labels)
    split_verification = verify_splits(assignment, labels)
    split_report = {
        "dataset": "EATD participant-level splits",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "ratios": list(ratios),
        "participant_level_split": True,
        "method": (
            "Stratified by participant label; participants shuffled per class "
            "with random.Random(seed); 70/15/15 approximately."
        ),
        "counts": split_summary,
        "overlap_verification": split_verification["overlaps"],
        "verification": split_verification,
        "leakage_statement": (
            "No participant appears in more than one split. Every participant "
            "is assigned exactly once. Sessions from a participant never cross "
            "splits."
        ),
        "legacy_comparison": {
            "legacy_split_file": str(LEGACY_SPLITS.relative_to(PROJECT_ROOT)),
            "legacy_participants_kept": len(legacy_kept),
            "new_participants": len(all_keys),
            "participants_added_by_corrected_key": len(all_keys - legacy_kept),
            "participants_only_in_legacy": sorted(legacy_kept - all_keys),
            "explanation": (
                "The legacy split treated numeric subject_id as globally unique. "
                "Content verification shows the 58 ids that occur in both source "
                "partitions are different participants (no shared audio hashes, "
                "no shared transcripts, different durations and mostly different "
                "SDS scores). The legacy script therefore dropped 23 ids (46 "
                "participants) as contradictory. Keying by (source_partition, "
                "subject_id) recovers all 162 verified participants without "
                "leakage."
            ),
            "legacy_split_was_leak_free": True,
            "legacy_split_was_over_conservative": True,
        },
        "label_definition": validation["label_definition"],
    }
    return validation, split_report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Build EATD manifest, reports and participant splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    participants = discover_participants()
    legacy = load_legacy_csv()

    manifest_rows, participant_rows, errors, warnings = audit_participants(participants, legacy)
    labels = {
        row["participant_key"]: int(row["label"])
        for row in participant_rows
        if row["label"] is not None
    }
    assignment = make_participant_splits(labels, seed=args.seed, ratios=tuple(args.ratios))
    validation, split_report = build_reports(
        manifest_rows, participant_rows, errors, warnings, assignment,
        args.seed, tuple(args.ratios),
    )

    out_dir = args.out_dir
    manifest_fields = [
        "participant_key", "source_split", "source_subject_id", "participant_index",
        "topic", "text_path", "audio_path", "audio_processed_path",
        "text_sha256", "audio_sha256", "transcript_chars",
        "audio_duration_sec", "audio_processed_duration_sec",
        "audio_processed_delta_sec", "sample_rate", "channels",
        "audio_usable", "quality_flags",
        "sds_raw", "sds_index", "label", "label_rule",
    ]
    participant_fields = [
        "participant_key", "source_split", "source_subject_id", "participant_index",
        "sds_raw", "sds_index", "label", "n_sessions", "has_all_transcripts",
        "has_all_audio", "usable_audio_sessions", "mean_audio_duration_sec",
        "mean_transcript_chars",
    ]

    write_csv(out_dir / "manifest.csv", manifest_rows, manifest_fields)
    write_csv(out_dir / "participants.csv", participant_rows, participant_fields)
    write_csv(
        out_dir / "splits.csv",
        [
            {
                "participant_key": row["participant_key"],
                "source_split": row["source_split"],
                "source_subject_id": row["source_subject_id"],
                "split": assignment[row["participant_key"]],
                "label": row["label"],
                "sds_index": row["sds_index"],
            }
            for row in participant_rows
        ],
        ["participant_key", "source_split", "source_subject_id", "split", "label", "sds_index"],
    )
    (out_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "split_report.json").write_text(
        json.dumps(split_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Participants: {validation['counts']['participants']}")
    print(f"Sessions:     {validation['counts']['sessions']}")
    print(f"Labels:       {validation['counts']['label_counts']}")
    for split, info in split_report["counts"].items():
        print(f"  {split:10s} participants={info['participants']:3d} sessions={info['sessions']:3d} "
              f"labels={info['label_counts']}")
    print(f"Overlap:      {split_report['verification']['participants_in_multiple_splits']} participants in >1 split")
    print(f"Status:       {validation['status']}")
    print(f"Wrote:        {out_dir}")

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"  - {error}")
        return 1
    if warnings:
        print(f"\nWarnings: {len(warnings)} (see validation_report.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
