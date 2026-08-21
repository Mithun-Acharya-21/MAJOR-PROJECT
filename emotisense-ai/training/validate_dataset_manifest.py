"""Fail closed if a planned training manifest violates dataset-compatibility rules.

Expected input is a CSV with columns: record_id, participant_id, session_id,
split, purpose, alignment. This script validates metadata only and never reads
raw participant text, audio, image, or video.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

REQUIRED_COLUMNS = {"record_id", "participant_id", "session_id", "split", "purpose", "alignment"}
VALID_SPLITS = {"train", "validation", "test"}
MULTIMODAL_PURPOSES = {"multimodal_depression_finetuning", "multimodal_depression_evaluation"}


def validate_manifest(path: Path) -> list[str]:
    """Return all violations; an empty list means the metadata is admissible."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        errors = [f"missing required columns: {', '.join(sorted(REQUIRED_COLUMNS - fields))}"] if not REQUIRED_COLUMNS <= fields else []
        rows = list(reader)

    seen_ids: set[str] = set()
    participant_splits: defaultdict[str, set[str]] = defaultdict(set)
    for line_number, row in enumerate(rows, start=2):
        record_id = row.get("record_id", "").strip()
        participant_id = row.get("participant_id", "").strip()
        split = row.get("split", "").strip()
        purpose = row.get("purpose", "").strip()
        alignment = row.get("alignment", "").strip()

        if not record_id or record_id in seen_ids:
            errors.append(f"line {line_number}: record_id must be present and unique")
        seen_ids.add(record_id)
        if not participant_id:
            errors.append(f"line {line_number}: participant_id is required for leakage prevention")
        if split not in VALID_SPLITS:
            errors.append(f"line {line_number}: split must be one of {sorted(VALID_SPLITS)}")
        else:
            participant_splits[participant_id].add(split)
        if purpose in MULTIMODAL_PURPOSES and alignment != "participant_session_aligned":
            errors.append(f"line {line_number}: multimodal depression records require participant_session_aligned")

    for participant_id, splits in participant_splits.items():
        if participant_id and len(splits) > 1:
            errors.append(f"participant {participant_id!r} appears in multiple splits: {sorted(splits)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a non-sensitive dataset split manifest.")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    errors = validate_manifest(args.manifest)
    if errors:
        print("Manifest rejected:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Manifest accepted: participant-disjoint and alignment checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

