"""Create leakage-safe split manifests from approved dataset metadata.

This tool intentionally reads only an index CSV. Raw text, audio, images, and
video stay in their approved local locations; only paths and non-sensitive
identifiers are copied to split manifests.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = {
    "record_id",
    "participant_id",
    "session_id",
    "dataset_id",
    "dataset_version",
    "label",
    "purpose",
    "alignment",
}
SPLITS = ("train", "validation", "test")
MULTIMODAL_PURPOSES = {"multimodal_depression_finetuning", "multimodal_depression_evaluation"}


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        if any(value <= 0 for value in (self.train, self.validation, self.test)):
            raise ValueError("all split ratios must be greater than zero")
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must add up to 1.0")


def read_index(path: Path) -> list[dict[str, str]]:
    """Read and validate approved metadata, failing before writing any output."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"index is missing columns: {', '.join(sorted(missing))}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]

    if not rows:
        raise ValueError("index has no rows")

    record_ids: set[str] = set()
    content_hashes: dict[str, str] = {}
    dataset_versions = {(row["dataset_id"], row["dataset_version"]) for row in rows}
    if len(dataset_versions) != 1:
        raise ValueError("one preparation run must contain exactly one dataset ID and version")

    for line_number, row in enumerate(rows, start=2):
        for column in REQUIRED_COLUMNS:
            if not row[column]:
                raise ValueError(f"line {line_number}: {column} cannot be blank")
        if row["record_id"] in record_ids:
            raise ValueError(f"line {line_number}: duplicate record_id {row['record_id']!r}")
        record_ids.add(row["record_id"])
        if row["purpose"] in MULTIMODAL_PURPOSES and row["alignment"] != "participant_session_aligned":
            raise ValueError(f"line {line_number}: depression fusion requires participant_session_aligned")

        # An optional hash makes duplicate-content leakage detectable without copying input data.
        content_hash = row.get("content_hash", "")
        if content_hash:
            other_record = content_hashes.get(content_hash)
            if other_record:
                raise ValueError(
                    f"line {line_number}: duplicate content_hash also used by record {other_record!r}"
                )
            content_hashes[content_hash] = row["record_id"]
    return rows


def allocate_participants(rows: list[dict[str, str]], ratios: SplitRatios, seed: int) -> dict[str, str]:
    """Allocate each participant once, stratified by their single approved label."""

    ratios.validate()
    labels_by_participant: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_participant[row["participant_id"]].add(row["label"])

    conflicting = [participant for participant, labels in labels_by_participant.items() if len(labels) != 1]
    if conflicting:
        raise ValueError(
            "each participant must have one stable label before splitting; conflicting IDs: "
            + ", ".join(sorted(conflicting)[:10])
        )

    participants_by_label: dict[str, list[str]] = defaultdict(list)
    for participant, labels in labels_by_participant.items():
        participants_by_label[next(iter(labels))].append(participant)

    allocation: dict[str, str] = {}
    rng = random.Random(seed)
    for label, participants in sorted(participants_by_label.items()):
        if len(participants) < 3:
            raise ValueError(f"label {label!r} has fewer than 3 participants; cannot create all three splits")
        participants = sorted(participants)
        rng.shuffle(participants)
        count = len(participants)
        # Every label needs representation in all three partitions. For small
        # cohorts this intentionally prioritizes that guarantee over the exact
        # target ratio (for three people: one person per split).
        train_count = min(max(1, int(count * ratios.train)), count - 2)
        validation_count = min(max(1, int(count * ratios.validation)), count - train_count - 1)
        train_end = train_count
        validation_end = train_count + validation_count
        for participant in participants[:train_end]:
            allocation[participant] = "train"
        for participant in participants[train_end:validation_end]:
            allocation[participant] = "validation"
        for participant in participants[validation_end:]:
            allocation[participant] = "test"
    return allocation


def write_manifests(rows: list[dict[str, str]], allocation: dict[str, str], output_dir: Path) -> None:
    """Write metadata-only split manifests and a reproducibility summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["split"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output_row = dict(row)
        output_row["split"] = allocation[row["participant_id"]]
        grouped[output_row["split"]].append(output_row)

    for split in SPLITS:
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(grouped[split], key=lambda row: row["record_id"]))

    summary = output_dir / "split-summary.txt"
    lines = ["Leakage-safe split summary", "", "This directory contains metadata only."]
    for split in SPLITS:
        participants = {row["participant_id"] for row in grouped[split]}
        labels: dict[str, int] = defaultdict(int)
        for participant in participants:
            participant_label = next(row["label"] for row in grouped[split] if row["participant_id"] == participant)
            labels[participant_label] += 1
        lines.append(f"{split}: {len(grouped[split])} records, {len(participants)} participants, labels={dict(sorted(labels.items()))}")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare participant-disjoint metadata manifests.")
    parser.add_argument("index", type=Path, help="Approved dataset metadata CSV; no raw inputs are read.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_index(args.index)
    allocation = allocate_participants(rows, SplitRatios(), args.seed)
    write_manifests(rows, allocation, args.output_dir)
    print(f"Prepared {len(rows)} records across {len(allocation)} participant-disjoint assignments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
