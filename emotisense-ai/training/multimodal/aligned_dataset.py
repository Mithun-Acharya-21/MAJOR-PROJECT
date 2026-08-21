"""Phase 12 — Aligned multimodal dataset builder.

Scientific constraints enforced here
--------------------------------------
1. Every row must tie a participant_id + session_id to EXACTLY ONE
   depression-risk label sourced from an approved aligned dataset
   (DAIC-WOZ / E-DAIC).  Emotion-dataset rows are never admitted.

2. No cross-dataset label mixing: a DAIC-WOZ transcript + RAVDESS
   audio is an invalid combination and is rejected at validation time.

3. Participant-disjoint splits are mandatory: a participant_id may
   appear in exactly one of train / validation / test.

4. Modality masks track which modalities are genuinely present for
   each sample — missing modalities are represented by a zero mask,
   never fabricated data.

5. Raw file paths are stored in the manifest CSV; raw content is
   never copied or embedded in the manifest.

Schema (one row per aligned session)
--------------------------------------
Required columns:
  participant_id  — stable anonymised identifier
  session_id      — session/interview identifier (may equal participant_id)
  dataset_name    — must be in APPROVED_MULTIMODAL_DATASETS
  split           — train | validation | test
  label           — depression-risk label (dataset-defined, not a diagnosis)
  has_text        — 0 | 1
  has_audio       — 0 | 1
  has_video       — 0 | 1

Optional columns (paths to preprocessed artefacts):
  text_path, audio_path, video_path
  text_quality, audio_quality, video_quality  (all ∈ [0, 1])
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APPROVED_MULTIMODAL_DATASETS = frozenset({"daic_woz", "e_daic"})
VALID_SPLITS = frozenset({"train", "validation", "test"})
REQUIRED_COLUMNS = frozenset({
    "participant_id", "session_id", "dataset_name",
    "split", "label", "has_text", "has_audio", "has_video",
})
MODALITY_COLUMNS = ("has_text", "has_audio", "has_video")

DISCLAIMER = (
    "Labels in this manifest are dataset-defined research scores. "
    "They are not clinical diagnoses of depression or any other condition."
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ManifestValidationResult:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_participants: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    modality_stats: dict[str, int] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "n_rows": self.n_rows,
            "n_participants": self.n_participants,
            "split_counts": self.split_counts,
            "modality_stats": self.modality_stats,
            "disclaimer": DISCLAIMER,
        }


def validate_aligned_manifest(manifest_path: Path) -> ManifestValidationResult:
    """Validate a multimodal manifest CSV against all scientific constraints."""
    result = ManifestValidationResult()

    try:
        with manifest_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:
        result.add_error(f"Cannot read file: {exc}")
        return result

    if not rows:
        result.add_error("Manifest is empty")
        return result

    # Column presence
    missing_cols = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing_cols:
        result.add_error(f"Missing required columns: {sorted(missing_cols)}")
        return result

    result.n_rows = len(rows)

    participant_splits: dict[str, set[str]] = defaultdict(set)
    label_set: set[str] = set()

    for idx, row in enumerate(rows, start=2):  # row 1 = header
        pid = row["participant_id"].strip()
        sid = row["session_id"].strip()
        dataset = row["dataset_name"].strip().lower()
        split = row["split"].strip().lower()
        label = row["label"].strip()

        # Dataset must be approved multimodal source
        if dataset not in APPROVED_MULTIMODAL_DATASETS:
            result.add_error(
                f"Row {idx}: dataset_name '{dataset}' is not an approved "
                f"multimodal depression dataset. "
                f"Approved: {sorted(APPROVED_MULTIMODAL_DATASETS)}"
            )

        # Split validity
        if split not in VALID_SPLITS:
            result.add_error(f"Row {idx}: invalid split '{split}'")

        # Label must be non-empty
        if not label:
            result.add_error(f"Row {idx}: empty label")
        else:
            label_set.add(label)

        # Participant–split tracking
        if pid:
            participant_splits[pid].add(split)

        # At least one modality must be present
        has_any = any(row.get(m, "0").strip() not in ("", "0") for m in MODALITY_COLUMNS)
        if not has_any:
            result.add_error(f"Row {idx}: participant={pid} has no modality flags set")

        # Modality quality values (if present) must be in [0, 1]
        for qual_col in ("text_quality", "audio_quality", "video_quality"):
            val = row.get(qual_col, "").strip()
            if val:
                try:
                    q = float(val)
                    if not (0.0 <= q <= 1.0):
                        result.add_error(
                            f"Row {idx}: {qual_col}={val} must be in [0, 1]"
                        )
                except ValueError:
                    result.add_error(f"Row {idx}: {qual_col}='{val}' is not numeric")

    # Cross-split participant leakage
    for pid, splits in participant_splits.items():
        if len(splits) > 1:
            result.add_error(
                f"Participant '{pid}' appears in multiple splits: {splits}. "
                "Participant-disjoint splits are mandatory."
            )

    result.n_participants = len(participant_splits)

    # Split counts
    split_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        split_counts[row["split"].strip().lower()] += 1
    result.split_counts = dict(split_counts)

    # Modality stats
    for m in MODALITY_COLUMNS:
        result.modality_stats[m] = sum(
            1 for r in rows if r.get(m, "0").strip() not in ("", "0")
        )

    # Warn if any split is empty
    for split in VALID_SPLITS:
        if split_counts.get(split, 0) == 0:
            result.warnings.append(f"Split '{split}' has zero rows")

    # Warn if only one label class
    if len(label_set) < 2:
        result.warnings.append(
            f"Only one label class found: {label_set}. "
            "Binary classification requires at least two classes."
        )

    return result


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

@dataclass
class AlignedManifestConfig:
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify_by_label: bool = True    # Try to keep label balance across splits
    min_participants_per_split: int = 2


def build_aligned_manifest(
    index_path: Path,
    output_dir: Path,
    config: Optional[AlignedManifestConfig] = None,
) -> dict:
    """Build participant-disjoint train/validation/test manifests.

    Parameters
    ----------
    index_path : CSV with columns: participant_id, session_id, dataset_name,
                 label, has_text, has_audio, has_video, and optional quality/path cols.
                 Rows must NOT include a 'split' column (it is assigned here).
    output_dir : Directory to write train.csv, validation.csv, test.csv,
                 split-summary.json.

    Returns
    -------
    dict with split counts, participant counts, and validation results.
    """
    if config is None:
        config = AlignedManifestConfig()

    assert abs(config.train_ratio + config.validation_ratio + config.test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    with index_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError(f"{index_path} is empty")

    # Validate required columns (minus 'split' — we assign it)
    required_no_split = REQUIRED_COLUMNS - {"split"}
    missing = required_no_split - set(rows[0].keys())
    if missing:
        raise ValueError(f"Index missing required columns: {sorted(missing)}")

    # Enforce approved dataset
    bad_datasets = {
        r["dataset_name"].strip().lower() for r in rows
        if r["dataset_name"].strip().lower() not in APPROVED_MULTIMODAL_DATASETS
    }
    if bad_datasets:
        raise ValueError(
            f"Index contains unapproved dataset(s): {bad_datasets}. "
            f"Only {sorted(APPROVED_MULTIMODAL_DATASETS)} are allowed."
        )

    # Group rows by participant
    participant_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        participant_rows[row["participant_id"].strip()].append(row)

    # Participant labels (first label encountered; flag conflicts)
    participant_label: dict[str, str] = {}
    for pid, p_rows in participant_rows.items():
        labels = {r["label"].strip() for r in p_rows}
        if len(labels) > 1:
            raise ValueError(
                f"Participant '{pid}' has conflicting labels: {labels}. "
                "All sessions for a participant must share the same label."
            )
        participant_label[pid] = labels.pop()

    participants = list(participant_rows.keys())
    rng = random.Random(config.seed)
    rng.shuffle(participants)

    # Stratified split by label
    if config.stratify_by_label:
        label_groups: dict[str, list[str]] = defaultdict(list)
        for pid in participants:
            label_groups[participant_label[pid]].append(pid)
        train_pids, val_pids, test_pids = [], [], []
        for label, pids in label_groups.items():
            n = len(pids)
            n_train = max(1, int(n * config.train_ratio))
            n_val = max(1, int(n * config.validation_ratio))
            train_pids.extend(pids[:n_train])
            val_pids.extend(pids[n_train:n_train + n_val])
            test_pids.extend(pids[n_train + n_val:])
    else:
        n = len(participants)
        n_train = max(1, int(n * config.train_ratio))
        n_val = max(1, int(n * config.validation_ratio))
        train_pids = participants[:n_train]
        val_pids = participants[n_train:n_train + n_val]
        test_pids = participants[n_train + n_val:]

    # Enforce minimum participants per split
    for split_name, pids in [("train", train_pids), ("validation", val_pids), ("test", test_pids)]:
        if len(pids) < config.min_participants_per_split:
            raise ValueError(
                f"Split '{split_name}' has only {len(pids)} participant(s); "
                f"minimum is {config.min_participants_per_split}. "
                "Provide more data or reduce min_participants_per_split."
            )

    # Sanity: no participant appears in more than one split
    all_assigned = set(train_pids) | set(val_pids) | set(test_pids)
    overlap = (set(train_pids) & set(val_pids)) | \
              (set(train_pids) & set(test_pids)) | \
              (set(val_pids) & set(test_pids))
    assert not overlap, f"BUG: participant overlap in splits: {overlap}"

    # Build output rows
    split_map = {pid: "train" for pid in train_pids}
    split_map.update({pid: "validation" for pid in val_pids})
    split_map.update({pid: "test" for pid in test_pids})

    fieldnames = list(rows[0].keys()) + ["split"]
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for pid, p_rows in participant_rows.items():
        split = split_map.get(pid)
        if split is None:
            continue
        for row in p_rows:
            split_rows[split].append({**row, "split": split})

    for split_name, s_rows in split_rows.items():
        out_path = output_dir / f"{split_name}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(s_rows)

    summary = {
        "disclaimer": DISCLAIMER,
        "split_row_counts": {k: len(v) for k, v in split_rows.items()},
        "split_participant_counts": {
            "train": len(train_pids),
            "validation": len(val_pids),
            "test": len(test_pids),
        },
        "total_participants": len(participant_rows),
        "total_rows": len(rows),
        "seed": config.seed,
        "note": (
            "Splits are participant-disjoint. "
            "Do not modify after training begins."
        ),
    }
    (output_dir / "split-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# PyTorch Dataset — multimodal aligned records
# ---------------------------------------------------------------------------

class AlignedMultimodalDataset:
    """Read a validated split manifest and yield per-sample feature dicts.

    This class does NOT load raw audio/video/images.  It yields metadata
    dicts with paths, modality masks, quality scores, and the label.
    Actual feature extraction is performed by the modality services
    (text branch, voice branch, vision branch) at inference/training time.

    Parameters
    ----------
    manifest_path : Path to a split CSV produced by build_aligned_manifest().
    expected_split : "train" | "validation" | "test"
    """

    def __init__(self, manifest_path: Path, expected_split: str) -> None:
        if expected_split not in VALID_SPLITS:
            raise ValueError(f"expected_split must be one of {VALID_SPLITS}")

        with manifest_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

        if not rows:
            raise ValueError(f"{manifest_path} is empty")

        # Enforce split column consistency
        bad_splits = [
            r["split"] for r in rows if r.get("split", "").strip() != expected_split
        ]
        if bad_splits:
            raise ValueError(
                f"{manifest_path} contains rows with split != '{expected_split}'"
            )

        self._rows = rows
        self._split = expected_split

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        return {
            "participant_id": row["participant_id"].strip(),
            "session_id": row["session_id"].strip(),
            "label": row["label"].strip(),
            "split": row["split"].strip(),
            "dataset_name": row["dataset_name"].strip(),
            "modality_mask": {
                "text": int(row.get("has_text", "0").strip() or "0"),
                "audio": int(row.get("has_audio", "0").strip() or "0"),
                "video": int(row.get("has_video", "0").strip() or "0"),
            },
            "quality_scores": {
                "text": float(row["text_quality"]) if row.get("text_quality", "").strip() else None,
                "audio": float(row["audio_quality"]) if row.get("audio_quality", "").strip() else None,
                "video": float(row["video_quality"]) if row.get("video_quality", "").strip() else None,
            },
            "paths": {
                "text": row.get("text_path", "").strip() or None,
                "audio": row.get("audio_path", "").strip() or None,
                "video": row.get("video_path", "").strip() or None,
            },
            "disclaimer": DISCLAIMER,
        }

    def label_distribution(self) -> dict[str, int]:
        """Return label → count mapping for this split."""
        dist: dict[str, int] = defaultdict(int)
        for row in self._rows:
            dist[row["label"].strip()] += 1
        return dict(dist)

    def participant_ids(self) -> list[str]:
        return list({r["participant_id"].strip() for r in self._rows})
