"""
Phase 2 dataset access — verified EATD manifest + participant-level splits.

Single source of truth for every Phase 2 experiment:

    data/processed/eatd/manifest.csv      (486 sessions, 162 participants)
    data/processed/eatd/participants.csv  (162 participants, one label each)
    data/processed/eatd/splits.csv        (162 participants → train/validation/test)

Contract enforced here (Phase 1 decisions, not re-decided):
  * participant key  = (source_partition, subject_id)  → ``participant_key``
  * SDS index        = sds_raw * 1.25
  * binary label     = 1 if sds_index >= 50 else 0        (SDS, NOT PHQ-9)
  * splits are participant-level; the three prompted topics of one participant
    always land in the same split.

Nothing in this module writes to data/. It only reads the verified artifacts.

Exports:
    PROJECT_ROOT, MANIFEST_CSV, PARTICIPANTS_CSV, SPLITS_CSV
    load_manifest, load_participants, load_splits
    load_sessions, read_transcript, resolve_path
    assert_split_integrity, participant_groups
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "eatd"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
PARTICIPANTS_CSV = DATA_DIR / "participants.csv"
SPLITS_CSV = DATA_DIR / "splits.csv"

PARTICIPANT_KEY_DEFINITION = "(source_partition, subject_id) — numeric subject ids repeat across the legacy train/validation folders"
LABEL_DEFINITION = "label = 1 if sds_index >= 50 else 0, where sds_index = sds_raw * 1.25 (EATD SDS, not PHQ-9)"
SPLIT_VERSION = "eatd_participant_splits_v1 (seed 42, scripts/build_eatd_manifest.py)"
DATASET_VERSION = "eatd_manifest_v1 (162 participants / 486 sessions)"

_MODALITY_COLUMN = {"text": "text_path", "audio": "audio_path"}


def resolve_path(value: str) -> Path:
    """Manifest paths are project-relative and may use Windows separators."""
    p = Path(str(value).replace("\\", "/"))
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_manifest(path: Path = MANIFEST_CSV) -> List[dict]:
    """Load manifest.csv with typed fields. Raises if the file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} missing — run scripts/build_eatd_manifest.py (Phase 1) first"
        )
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            rows.append(
                {
                    "participant_key": raw["participant_key"],
                    "source_split": raw["source_split"],
                    "source_subject_id": raw["source_subject_id"],
                    "participant_index": int(raw["participant_index"]),
                    "topic": raw["topic"],
                    "text_path": raw["text_path"],
                    "audio_path": raw["audio_path"],
                    "text_sha256": raw["text_sha256"],
                    "audio_sha256": raw["audio_sha256"],
                    "audio_duration_sec": float(raw["audio_duration_sec"]),
                    "audio_usable": _as_bool(raw["audio_usable"]),
                    "quality_flags": raw.get("quality_flags") or "",
                    "sds_raw": float(raw["sds_raw"]),
                    "sds_index": float(raw["sds_index"]),
                    "label": int(raw["label"]),
                }
            )
    return rows


def load_participants(path: Path = PARTICIPANTS_CSV) -> List[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing — run scripts/build_eatd_manifest.py first")
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {
                "participant_key": r["participant_key"],
                "source_split": r["source_split"],
                "source_subject_id": r["source_subject_id"],
                "sds_raw": float(r["sds_raw"]),
                "sds_index": float(r["sds_index"]),
                "label": int(r["label"]),
                "n_sessions": int(r["n_sessions"]),
                "usable_audio_sessions": int(r["usable_audio_sessions"]),
            }
            for r in csv.DictReader(f)
        ]


def load_splits(path: Path = SPLITS_CSV) -> Dict[str, str]:
    """participant_key → 'train' | 'validation' | 'test'."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing — run scripts/build_eatd_manifest.py first")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["participant_key"]: r["split"] for r in csv.DictReader(f)}


def read_transcript(path_value: str) -> str:
    """Read a transcript, tolerating legacy GBK-encoded files."""
    p = resolve_path(path_value)
    data = p.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", data, 0, 1, f"cannot decode transcript {p}")


def load_sessions(
    split: Optional[str] = None,
    modality: str = "text",
    require_usable_audio: bool = False,
    require_file: bool = True,
    manifest: Optional[Sequence[dict]] = None,
) -> List[dict]:
    """Sessions of one split, enriched with split assignment and loaded text.

    Args:
        split: 'train' | 'validation' | 'test' | None (all).
        modality: 'text' or 'audio' — which file must exist.
        require_usable_audio: drop sessions the Phase 1 audit flagged as
            unusable for audio (only 1 session: a 0.48 s recording).
        require_file: verify the modality file exists on disk.
        manifest: pre-loaded manifest rows (avoids re-reading).

    Returns rows with keys: participant_key, participant_label, split, topic,
    text, text_path, audio_path, audio_duration_sec, audio_usable.
    """
    if modality not in _MODALITY_COLUMN:
        raise ValueError(f"modality must be one of {sorted(_MODALITY_COLUMN)}")
    rows = list(manifest) if manifest is not None else load_manifest()
    assignments = load_splits()
    labels = {p["participant_key"]: p["label"] for p in load_participants()}

    out: List[dict] = []
    for r in rows:
        pk = r["participant_key"]
        if pk not in assignments:
            raise KeyError(f"participant {pk} has no split assignment")
        if split is not None and assignments[pk] != split:
            continue
        if require_usable_audio and not r["audio_usable"]:
            continue
        target = r[_MODALITY_COLUMN[modality]]
        if require_file and not resolve_path(target).exists():
            continue
        text = read_transcript(r["text_path"]) if modality == "text" else ""
        out.append(
            {
                **r,
                "split": assignments[pk],
                "participant_label": labels[pk],
                "text": text,
            }
        )
    out.sort(key=lambda r: (r["participant_index"], r["topic"]))
    return out


def participant_groups(rows: Iterable[dict]) -> Dict[str, List[int]]:
    """participant_key → indices of that participant's sessions (for bootstrap)."""
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["participant_key"], []).append(i)
    return groups


def assert_split_integrity(
    splits: Optional[Dict[str, str]] = None,
    participants: Optional[Sequence[dict]] = None,
) -> Dict[str, int]:
    """Verify participant-level split integrity; raise on any violation.

    Returns per-split participant counts. The three checks that matter:
    every participant is assigned exactly once, no participant appears in two
    splits, and every split contains both classes.
    """
    splits = splits or load_splits()
    participants = list(participants or load_participants())
    keys = [p["participant_key"] for p in participants]

    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate participant_key in participants.csv")
    missing = [k for k in keys if k not in splits]
    if missing:
        raise AssertionError(f"participants without split assignment: {missing[:5]}")
    unknown = [k for k in splits if k not in set(keys)]
    if unknown:
        raise AssertionError(f"split rows without a participant record: {unknown[:5]}")

    # A participant key must never be split by topic: check manifest coverage.
    manifest_keys = {r["participant_key"] for r in load_manifest()}
    if manifest_keys != set(keys):
        raise AssertionError("manifest and participants.csv disagree on participant set")

    per_split: Dict[str, int] = {}
    pos: Dict[str, int] = {}
    for p in participants:
        s = splits[p["participant_key"]]
        if s not in {"train", "validation", "test"}:
            raise AssertionError(f"unexpected split name {s!r}")
        per_split[s] = per_split.get(s, 0) + 1
        pos[s] = pos.get(s, 0) + p["label"]

    for s in ("train", "validation", "test"):
        if per_split.get(s, 0) == 0:
            raise AssertionError(f"split {s!r} is empty")
        if pos[s] in (0, per_split[s]):
            raise AssertionError(f"split {s!r} is single-class")

    # The 58 numeric-id collisions must be resolved by source_partition.
    by_id: Dict[str, set] = {}
    for p in participants:
        by_id.setdefault(p["source_subject_id"], set()).add(p["participant_key"])
    collisions = [k for k, v in by_id.items() if len(v) > 1]
    if not collisions:
        raise AssertionError(
            "expected numeric subject-id collisions across source partitions; "
            "the participant key may have been collapsed to subject_id alone"
        )

    return per_split
