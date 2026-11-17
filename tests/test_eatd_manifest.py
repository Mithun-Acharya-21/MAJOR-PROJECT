"""
Phase 1 — EATD manifest, alignment and participant-split tests.

These tests read the derived artifacts in ``data/processed/eatd/`` and verify
them against the source files. They never modify data.

Skipped only when the manifest has not been generated yet; run:
    python scripts/build_eatd_manifest.py --seed 42
"""

import csv
import hashlib
import importlib.util
import json
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "eatd"
MANIFEST = OUT_DIR / "manifest.csv"
PARTICIPANTS = OUT_DIR / "participants.csv"
SPLITS = OUT_DIR / "splits.csv"
VALIDATION = OUT_DIR / "validation_report.json"
SPLIT_REPORT = OUT_DIR / "split_report.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="EATD manifest not generated (run scripts/build_eatd_manifest.py)"
)

TOPICS = {"positive", "neutral", "negative"}


def _rows(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_module():
    spec = importlib.util.spec_from_file_location(
        "build_eatd_manifest", PROJECT_ROOT / "scripts" / "build_eatd_manifest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def test_manifest_has_all_verified_participants_and_sessions():
    participants = _rows(PARTICIPANTS)
    manifest = _rows(MANIFEST)

    assert len(participants) == 162
    assert len(manifest) == 486
    assert {row["source_split"] for row in participants} == {"train", "validation"}
    assert all(int(row["n_sessions"]) == 3 for row in participants)
    assert all(row["has_all_transcripts"] == "True" for row in participants)
    assert all(row["has_all_audio"] == "True" for row in participants)


def test_participant_keys_are_partition_scoped():
    """58 numeric ids repeat across partitions but must map to unique keys."""
    participants = _rows(PARTICIPANTS)
    keys = [row["participant_key"] for row in participants]

    assert len(keys) == len(set(keys)) == 162

    by_numeric = defaultdict(set)
    for row in participants:
        by_numeric[row["source_subject_id"]].add(row["participant_key"])

    duplicated_ids = {sid for sid, group in by_numeric.items() if len(group) > 1}
    assert len(duplicated_ids) == 58
    # Same numeric id in different partitions -> distinct participant keys
    for sid in list(duplicated_ids)[:5]:
        assert len(by_numeric[sid]) == 2


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_binary_label_is_exactly_sds_index_ge_50():
    participants = _rows(PARTICIPANTS)

    for row in participants:
        index = float(row["sds_index"])
        assert int(row["label"]) == int(index >= 50.0), row

    class_zero = [float(r["sds_index"]) for r in participants if int(r["label"]) == 0]
    class_one = [float(r["sds_index"]) for r in participants if int(r["label"]) == 1]

    assert max(class_zero) < 50.0 <= min(class_one)
    assert len(class_zero) == 109
    assert len(class_one) == 53


def test_sds_index_equals_raw_times_1_25():
    participants = _rows(PARTICIPANTS)
    for row in participants:
        raw = float(row["sds_raw"])
        index = float(row["sds_index"])
        assert abs(index - raw * 1.25) < 1e-6, row


def test_label_definition_is_documented_as_sds_not_phq():
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))
    definition = report["label_definition"]

    assert "Zung" in definition["instrument"]
    assert definition["binary_rule"] == "label = 1 if index >= 50 else 0"
    assert definition["mismatches_vs_legacy_csv"] == 0
    assert "PHQ" in definition["note"]


# ---------------------------------------------------------------------------
# Text / audio alignment
# ---------------------------------------------------------------------------

def test_every_manifest_file_exists_and_hash_matches():
    manifest = _rows(MANIFEST)

    for row in manifest:
        text_path = PROJECT_ROOT / row["text_path"]
        audio_path = PROJECT_ROOT / row["audio_path"]

        assert text_path.exists(), text_path
        assert audio_path.exists(), audio_path

        text_bytes = text_path.read_text(encoding="utf-8-sig").strip().encode("utf-8")
        assert hashlib.sha256(text_bytes).hexdigest() == row["text_sha256"]

        digest = hashlib.sha256()
        with open(audio_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == row["audio_sha256"], row


def test_audio_is_16khz_and_topic_aligned():
    manifest = _rows(MANIFEST)

    for row in manifest:
        path = PROJECT_ROOT / row["audio_path"]
        with wave.open(str(path), "rb") as handle:
            assert handle.getframerate() == 16000
            assert handle.getnframes() > 0
        assert path.stem in TOPICS
        assert path.stem == row["topic"]
        assert (PROJECT_ROOT / row["text_path"]).stem == row["topic"]


def test_no_duplicate_audio_or_transcripts_across_participants():
    manifest = _rows(MANIFEST)

    audio_hashes = [row["audio_sha256"] for row in manifest]
    text_hashes = [row["text_sha256"] for row in manifest]

    assert len(audio_hashes) == len(set(audio_hashes)) == 486
    assert len(text_hashes) == len(set(text_hashes)) == 486


def test_short_audio_is_flagged_not_dropped_or_fatal():
    """A too-short session is retained for text but excluded from audio models."""
    manifest = _rows(MANIFEST)
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))

    unusable = [row for row in manifest if row["audio_usable"] == "False"]
    assert len(unusable) == 1
    assert unusable[0]["participant_key"] == "eatd_validation_79"
    assert unusable[0]["topic"] == "positive"
    assert "short_audio_lt_1000ms" in unusable[0]["quality_flags"]
    assert float(unusable[0]["audio_duration_sec"]) < 1.0

    # It remains a real session (text still usable) and is not an error.
    assert len(manifest) == 486
    assert report["status"] != "fail"
    assert report["quality_flags"]["short_audio_lt_1000ms"] == 1
    assert report["counts"]["sessions_audio_usable"] == 485


def test_processed_wav_is_not_counted_as_a_sample():
    manifest = _rows(MANIFEST)
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))

    assert report["integrity_checks"]["audio_out_treated_as_separate_sample"] is False
    # One canonical WAV per participant×topic; processed paths are metadata only.
    assert sum(1 for row in manifest if row["audio_path"].endswith("_out.wav")) == 0
    assert all(row["audio_processed_path"].endswith("_out.wav") for row in manifest)


def test_validation_report_passed_integrity_checks():
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))

    assert report["status"] in ("pass", "pass_with_quality_flags")
    assert report["counts"]["participants"] == 162
    assert report["counts"]["sessions"] == 486
    checks = report["integrity_checks"]
    assert checks["all_transcripts_present"] is True
    assert checks["all_audio_present"] is True
    assert checks["all_audio_16khz"] is True
    assert checks["all_text_matches_legacy_csv"] is True
    assert checks["binary_label_reproduces_legacy_csv"] is True
    assert checks["duplicate_audio_content"] == 0
    assert checks["duplicate_transcript_content"] == 0


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_splits_cover_every_participant_exactly_once():
    splits = _rows(SPLITS)
    participants = _rows(PARTICIPANTS)

    assert len(splits) == 162
    assert {row["participant_key"] for row in splits} == {
        row["participant_key"] for row in participants
    }
    assert {row["split"] for row in splits} == {"train", "validation", "test"}


def test_zero_participant_leakage_between_splits():
    splits = _rows(SPLITS)
    per_split = defaultdict(set)
    for row in splits:
        per_split[row["split"]].add(row["participant_key"])

    assert per_split["train"] & per_split["validation"] == set()
    assert per_split["train"] & per_split["test"] == set()
    assert per_split["validation"] & per_split["test"] == set()


def test_every_split_contains_both_classes():
    splits = _rows(SPLITS)
    for split in ("train", "validation", "test"):
        labels = {int(row["label"]) for row in splits if row["split"] == split}
        assert labels == {0, 1}, f"{split} has labels {labels}"


def test_split_is_deterministic_for_seed_42():
    module = _split_module()
    participants = _rows(PARTICIPANTS)
    labels = {row["participant_key"]: int(row["label"]) for row in participants}

    first = module.make_participant_splits(labels, seed=42)
    second = module.make_participant_splits(labels, seed=42)

    assert first == second

    recorded = {row["participant_key"]: row["split"] for row in _rows(SPLITS)}
    assert first == recorded


def test_split_report_documents_legacy_correction():
    report = json.loads(SPLIT_REPORT.read_text(encoding="utf-8"))
    verification = report["verification"]

    assert report["participant_level_split"] is True
    assert verification["participants_in_multiple_splits"] == 0
    assert verification["uncovered_participants"] == []

    legacy = report["legacy_comparison"]
    assert legacy["legacy_participants_kept"] == 81
    assert legacy["new_participants"] == 162
    assert legacy["participants_added_by_corrected_key"] == 81
    assert legacy["legacy_split_was_leak_free"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
