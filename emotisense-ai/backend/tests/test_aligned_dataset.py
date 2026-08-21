"""Tests for Phase 12 — aligned multimodal dataset builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from training.multimodal.aligned_dataset import (
    APPROVED_MULTIMODAL_DATASETS,
    AlignedManifestConfig,
    AlignedMultimodalDataset,
    DISCLAIMER,
    ManifestValidationResult,
    build_aligned_manifest,
    validate_aligned_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_index(tmp_path: Path, rows: list[dict]) -> Path:
    """Write an index CSV (no 'split' column) to tmp_path."""
    path = tmp_path / "index.csv"
    if not rows:
        path.write_text("")
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a manifest CSV (includes 'split' column) for validate tests."""
    path = tmp_path / "manifest.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _base_manifest_rows(n_participants: int = 8) -> list[dict]:
    """Generate minimal valid aligned manifest rows."""
    rows = []
    splits = (["train"] * 6 + ["validation"] * 1 + ["test"] * 1)[:n_participants]
    for i in range(n_participants):
        label = "depressed" if i % 2 == 0 else "control"
        rows.append({
            "participant_id": f"P{i:03d}",
            "session_id": f"S{i:03d}",
            "dataset_name": "daic_woz",
            "split": splits[i] if i < len(splits) else "train",
            "label": label,
            "has_text": "1",
            "has_audio": "1",
            "has_video": "1",
        })
    return rows


def _base_index_rows(n: int = 12) -> list[dict]:
    """Generate minimal valid index rows (no split column)."""
    rows = []
    for i in range(n):
        label = "depressed" if i % 2 == 0 else "control"
        rows.append({
            "participant_id": f"P{i:03d}",
            "session_id": f"S{i:03d}",
            "dataset_name": "daic_woz",
            "label": label,
            "has_text": "1",
            "has_audio": "1",
            "has_video": "0",
        })
    return rows


# ---------------------------------------------------------------------------
# validate_aligned_manifest tests
# ---------------------------------------------------------------------------

class TestValidateAlignedManifest:

    def test_valid_manifest_passes(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path, _base_manifest_rows())
        result = validate_aligned_manifest(path)
        assert result.is_valid, result.errors

    def test_unapproved_dataset_rejected(self, tmp_path: Path) -> None:
        rows = _base_manifest_rows()
        rows[0]["dataset_name"] = "ravdess"  # emotion-only, not allowed
        path = _write_manifest(tmp_path, rows)
        result = validate_aligned_manifest(path)
        assert not result.is_valid
        assert any("approved" in e for e in result.errors)

    def test_participant_in_two_splits_rejected(self, tmp_path: Path) -> None:
        rows = _base_manifest_rows()
        # Put P000 in both train and validation
        rows.append({**rows[0], "split": "validation", "session_id": "S999"})
        path = _write_manifest(tmp_path, rows)
        result = validate_aligned_manifest(path)
        assert not result.is_valid
        assert any("multiple splits" in e for e in result.errors)

    def test_no_modality_flags_rejected(self, tmp_path: Path) -> None:
        rows = _base_manifest_rows()
        rows[0].update({"has_text": "0", "has_audio": "0", "has_video": "0"})
        path = _write_manifest(tmp_path, rows)
        result = validate_aligned_manifest(path)
        assert not result.is_valid
        assert any("no modality" in e for e in result.errors)

    def test_quality_out_of_range_rejected(self, tmp_path: Path) -> None:
        rows = _base_manifest_rows()
        rows[0]["text_quality"] = "1.5"
        path = _write_manifest(tmp_path, rows)
        result = validate_aligned_manifest(path)
        assert not result.is_valid
        assert any("text_quality" in e for e in result.errors)

    def test_missing_required_column_rejected(self, tmp_path: Path) -> None:
        rows = _base_manifest_rows()
        for r in rows:
            del r["label"]
        path = _write_manifest(tmp_path, rows)
        result = validate_aligned_manifest(path)
        assert not result.is_valid

    def test_split_counts_reported(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path, _base_manifest_rows(8))
        result = validate_aligned_manifest(path)
        assert "train" in result.split_counts
        assert result.n_rows == 8

    def test_disclaimer_in_result(self, tmp_path: Path) -> None:
        path = _write_manifest(tmp_path, _base_manifest_rows())
        result = validate_aligned_manifest(path)
        d = result.to_dict()
        assert "disclaimer" in d
        assert "not clinical" in d["disclaimer"].lower()

    def test_empty_manifest_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("")
        result = validate_aligned_manifest(path)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# build_aligned_manifest tests
# ---------------------------------------------------------------------------

class TestBuildAlignedManifest:

    def test_builds_three_split_files(self, tmp_path: Path) -> None:
        index = _write_index(tmp_path, _base_index_rows(12))
        out = tmp_path / "splits"
        cfg = AlignedManifestConfig(seed=0, min_participants_per_split=2)
        result = build_aligned_manifest(index, out, cfg)
        assert (out / "train.csv").exists()
        assert (out / "validation.csv").exists()
        assert (out / "test.csv").exists()
        assert (out / "split-summary.json").exists()

    def test_splits_are_participant_disjoint(self, tmp_path: Path) -> None:
        index = _write_index(tmp_path, _base_index_rows(12))
        out = tmp_path / "splits"
        build_aligned_manifest(index, out, AlignedManifestConfig(seed=7, min_participants_per_split=2))
        def pids(f: Path) -> set[str]:
            return {r["participant_id"] for r in csv.DictReader(f.open())}
        train_p = pids(out / "train.csv")
        val_p = pids(out / "validation.csv")
        test_p = pids(out / "test.csv")
        assert not (train_p & val_p), "train/val overlap"
        assert not (train_p & test_p), "train/test overlap"
        assert not (val_p & test_p), "val/test overlap"

    def test_all_rows_assigned(self, tmp_path: Path) -> None:
        index = _write_index(tmp_path, _base_index_rows(12))
        out = tmp_path / "splits"
        result = build_aligned_manifest(index, out, AlignedManifestConfig(min_participants_per_split=2))
        total_assigned = sum(result["split_row_counts"].values())
        assert total_assigned == 12

    def test_conflicting_participant_labels_rejected(self, tmp_path: Path) -> None:
        rows = _base_index_rows(6)
        # Give P000 two different labels via two sessions
        rows.append({**rows[0], "session_id": "S999", "label": "control"})
        index = _write_index(tmp_path, rows)
        with pytest.raises(ValueError, match="conflicting labels"):
            build_aligned_manifest(index, tmp_path / "out",
                                   AlignedManifestConfig(min_participants_per_split=1))

    def test_unapproved_dataset_in_index_rejected(self, tmp_path: Path) -> None:
        rows = _base_index_rows(6)
        rows[0]["dataset_name"] = "fer_2013"
        index = _write_index(tmp_path, rows)
        with pytest.raises(ValueError, match="unapproved"):
            build_aligned_manifest(index, tmp_path / "out",
                                   AlignedManifestConfig(min_participants_per_split=1))

    def test_summary_contains_disclaimer(self, tmp_path: Path) -> None:
        index = _write_index(tmp_path, _base_index_rows(12))
        out = tmp_path / "splits"
        result = build_aligned_manifest(index, out, AlignedManifestConfig(min_participants_per_split=2))
        assert "disclaimer" in result
        assert "not clinical" in result["disclaimer"].lower()

    def test_split_summary_json_written(self, tmp_path: Path) -> None:
        index = _write_index(tmp_path, _base_index_rows(12))
        out = tmp_path / "splits"
        build_aligned_manifest(index, out, AlignedManifestConfig(min_participants_per_split=2))
        summary = json.loads((out / "split-summary.json").read_text())
        assert "split_row_counts" in summary
        assert "split_participant_counts" in summary


# ---------------------------------------------------------------------------
# AlignedMultimodalDataset tests
# ---------------------------------------------------------------------------

class TestAlignedMultimodalDataset:

    def _build_splits(self, tmp_path: Path) -> Path:
        index = _write_index(tmp_path, _base_index_rows(12))
        out = tmp_path / "splits"
        build_aligned_manifest(index, out, AlignedManifestConfig(min_participants_per_split=2))
        return out

    def test_len_matches_row_count(self, tmp_path: Path) -> None:
        out = self._build_splits(tmp_path)
        ds = AlignedMultimodalDataset(out / "train.csv", "train")
        rows = list(csv.DictReader((out / "train.csv").open()))
        assert len(ds) == len(rows)

    def test_item_has_required_keys(self, tmp_path: Path) -> None:
        out = self._build_splits(tmp_path)
        ds = AlignedMultimodalDataset(out / "train.csv", "train")
        item = ds[0]
        for key in ("participant_id", "session_id", "label", "modality_mask",
                    "quality_scores", "paths", "disclaimer"):
            assert key in item, f"Missing key: {key}"

    def test_modality_mask_values_binary(self, tmp_path: Path) -> None:
        out = self._build_splits(tmp_path)
        ds = AlignedMultimodalDataset(out / "train.csv", "train")
        for i in range(len(ds)):
            mask = ds[i]["modality_mask"]
            for v in mask.values():
                assert v in (0, 1), f"Non-binary modality mask value: {v}"

    def test_wrong_split_rejected(self, tmp_path: Path) -> None:
        out = self._build_splits(tmp_path)
        with pytest.raises(ValueError, match="split"):
            AlignedMultimodalDataset(out / "train.csv", "validation")

    def test_label_distribution_has_both_classes(self, tmp_path: Path) -> None:
        out = self._build_splits(tmp_path)
        ds = AlignedMultimodalDataset(out / "train.csv", "train")
        dist = ds.label_distribution()
        assert len(dist) >= 2, f"Expected ≥2 label classes, got {dist}"

    def test_disclaimer_in_every_item(self, tmp_path: Path) -> None:
        out = self._build_splits(tmp_path)
        ds = AlignedMultimodalDataset(out / "train.csv", "train")
        for i in range(len(ds)):
            assert "disclaimer" in ds[i]
            assert "not clinical" in ds[i]["disclaimer"].lower()
