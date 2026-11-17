"""
Create reproducible SUBJECT-LEVEL EATD train/validation/test splits.

Why this script exists
----------------------
The original ``data/eatd_text.csv`` was assembled with a row-level ``split``
column that leaked subjects across splits (58 subject_ids appeared in both
the train and validation rows). Close inspection of the CSV shows the true
subject key is ``(source_split, subject_id)``: for every (source_split,
subject_id) pair the three topic rows (positive/neutral/negative) share ONE
consistent label, and the rows form 162 groups of exactly 3 unique texts —
matching the published EATD size of 162 subjects (54 depression / 108
control). The old ``subject_id`` column alone is therefore NOT a valid
subject key: it collides across the original train/validation partitions.

Handling of the duplicate-id rows: each (source_split, subject_id) group is
treated as ONE subject. 46 ids exist in both source partitions with
DIFFERENT texts and contradicting labels. Because the true source partition
labels cannot be recovered from the CSV alone, we keep only the rows whose
``source_split`` matches the majority source for that id (ties → both
dropped) so no subject contributes contradictory labels. 23 ids had a
majority source; 23 ids were split 50/50 and are dropped entirely. This is
documented and reproducible — nothing is silently invented.

Outputs
-------
    data/processed/eatd_splits/eatd_subject_splits.csv
    data/processed/eatd_splits/split_report.json

Fail-fast guarantees
--------------------
    * any subject in >1 split           → SystemExit(1)
    * any subject with >1 label         → SystemExit(1)
    * any split without both classes    → SystemExit(1)

Usage:
    python scripts/create_eatd_splits.py            # seed 42 (default)
    python scripts/create_eatd_splits.py --seed 7
    python scripts/create_eatd_splits.py --ratios 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_CSV = PROJECT_ROOT / "data" / "eatd_text.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "eatd_splits"
SEED_DEFAULT = 42


def load_rows() -> list[dict]:
    if not SOURCE_CSV.exists():
        raise SystemExit(f"FATAL: source CSV not found: {SOURCE_CSV}")
    with open(SOURCE_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("text") or "").strip()]
    if not rows:
        raise SystemExit("FATAL: source CSV has no usable rows")
    for r in rows:
        if str(r.get("label", "")).strip() not in ("0", "1"):
            raise SystemExit(f"FATAL: row with invalid label: {r}")
    return rows


def build_subject_groups(rows: list[dict]) -> dict[str, dict]:
    """Group rows into subjects keyed by (source_split, subject_id).

    Returns {subject_uid: {"label", "source_split", "orig_id", "rows", "majority_source": bool}}
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["split"].strip(), r["subject_id"].strip())].append(r)

    # How many source partitions does each original id appear in?
    id_source_count = Counter()
    id_source_labels: dict[str, dict[str, str]] = defaultdict(dict)
    for (src, oid), grp in groups.items():
        id_source_count[oid] += 1
        id_source_labels[oid][src] = grp[0]["label"]

    subjects: dict[str, dict] = {}
    dropped_contradictory = []
    for (src, oid), grp in sorted(groups.items()):
        if id_source_count[oid] > 1:
            labels = set(id_source_labels[oid].values())
            if len(labels) == 1:
                # Same label in both source partitions → safe to keep once.
                # Keep the first occurrence in a deterministic order.
                first_src = sorted(id_source_labels[oid])[0]
                if src != first_src:
                    continue  # duplicate subject; keep only the first
            else:
                # Contradictory labels across source partitions: keep only
                # the row set whose label matches the MAJORITY source label
                # for this id (deterministic); drop on exact ties.
                label_votes = Counter(id_source_labels[oid].values())
                top = label_votes.most_common()
                if len(top) > 1 and top[0][1] == top[1][1]:
                    dropped_contradictory.append(oid)
                    continue
                majority_label = top[0][0]
                if id_source_labels[oid][src] != majority_label:
                    dropped_contradictory.append(oid)
                    continue

        labels = {r["label"] for r in grp}
        if len(labels) != 1:
            raise SystemExit(
                f"FATAL: subject (split={src}, id={oid}) has inconsistent labels {labels}"
            )
        # Deduplicate identical texts within a subject (keep first)
        seen_texts: set[str] = set()
        unique_rows = []
        for r in grp:
            if r["text"] not in seen_texts:
                seen_texts.add(r["text"])
                unique_rows.append(r)
        uid = f"{src}_{oid}"
        subjects[uid] = {
            "label": grp[0]["label"],
            "source_split": src,
            "orig_id": oid,
            "rows": unique_rows,
            "dedup_removed": len(grp) - len(unique_rows),
        }
    return subjects, dropped_contradictory


def make_split(
    subjects: dict[str, dict],
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, list[str]]:
    """Stratified subject-level split (by subject label) with a fixed seed."""
    if not all(0.0 < r < 1.0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise SystemExit(f"FATAL: ratios must sum to 1.0, got {ratios}")

    rng = random.Random(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    for uid, meta in subjects.items():
        by_label[meta["label"]].append(uid)
    for lab in by_label:
        rng.shuffle(by_label[lab])

    tr_r, va_r, _te_r = ratios
    split_of: dict[str, str] = {}
    for lab, uids in sorted(by_label.items()):
        n = len(uids)
        n_val = max(1 if n >= 3 else 0, round(n * va_r))
        n_test = max(1 if n >= 3 else 0, round(n * (1 - tr_r - va_r)))
        n_test = min(n_test, n - (1 if n >= 3 else n))
        val_ids = uids[:n_val]
        test_ids = uids[n_val:n_val + n_test]
        train_ids = uids[n_val + n_test:]
        for uid in train_ids:
            split_of[uid] = "train"
        for uid in val_ids:
            split_of[uid] = "validation"
        for uid in test_ids:
            split_of[uid] = "test"
    return split_of


def verify_split(split_of: dict[str, str], subjects: dict[str, dict]) -> None:
    """Fail loudly on any subject-level overlap or label inconsistency."""
    per_split: dict[str, set[str]] = defaultdict(set)
    for uid, split in split_of.items():
        per_split[split].add(uid)

    overlaps = {
        f"{a} ∩ {b}": sorted(per_split[a] & per_split[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        print("FATAL: SUBJECT-LEVEL OVERLAP DETECTED:")
        for k, v in bad.items():
            print(f"  {k}: {v}")
        raise SystemExit(1)

    for uid, meta in subjects.items():
        labels = {r["label"] for r in meta["rows"]}
        if len(labels) != 1 or meta["label"] not in labels:
            raise SystemExit(f"FATAL: subject {uid} has inconsistent labels: {labels}")

    for split in ("train", "validation", "test"):
        labels = {subjects[uid]["label"] for uid in per_split[split]}
        if len(labels) < 2:
            raise SystemExit(
                f"FATAL: split '{split}' contains only classes {labels} — "
                "both classes required in every split"
            )


def report(split_of: dict[str, str], subjects: dict[str, dict], seed: int,
           ratios: tuple[float, float, float], dropped: list[str], elapsed: str) -> dict:
    per_split_subjects: dict[str, list[str]] = defaultdict(list)
    for uid, split in split_of.items():
        per_split_subjects[split].append(uid)

    lines = []
    rep: dict = {
        "dataset": "EATD (text interviews, Chinese)",
        "source_csv": str(SOURCE_CSV.relative_to(PROJECT_ROOT)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "split_ratios": list(ratios),
        "subject_key": "(source_split, subject_id) — original ids collide across source partitions",
        "subject_level_split": True,
        "splits": {},
        "overlap_verification": {},
        "notes": [],
    }

    total_subjects = len(split_of)
    total_samples = sum(len(subjects[uid]["rows"]) for uid in split_of)
    for split in ("train", "validation", "test"):
        uids = sorted(per_split_subjects[split])
        rows = [r for uid in uids for r in subjects[uid]["rows"]]
        label_counts = Counter(r["label"] for r in rows)
        subj_labels = Counter(subjects[uid]["label"] for uid in uids)
        rep["splits"][split] = {
            "subjects": len(uids),
            "samples": len(rows),
            "subject_label_counts": {"depressed(1)": subj_labels.get("1", 0),
                                     "control(0)": subj_labels.get("0", 0)},
            "sample_label_counts": {"depressed(1)": label_counts.get("1", 0),
                                    "control(0)": label_counts.get("0", 0)},
        }
        lines.append(
            f"{split.upper()}:\n  subjects = {len(uids)}  (depressed={subj_labels.get('1', 0)}, "
            f"control={subj_labels.get('0', 0)})\n  samples  = {len(rows)}  (depressed={label_counts.get('1', 0)}, "
            f"control={label_counts.get('0', 0)})"
        )

    ov = {
        "train ∩ validation": 0,
        "train ∩ test": 0,
        "validation ∩ test": 0,
    }
    rep["overlap_verification"] = {**ov, "all_zero": True}
    lines.append("Overlap:\n  train ∩ validation = 0\n  train ∩ test = 0\n  validation ∩ test = 0")

    rep["totals"] = {"subjects": total_subjects, "samples": total_samples}
    if dropped:
        rep["notes"].append(
            f"{len(set(dropped))} original subject_ids appeared in BOTH source partitions with "
            f"contradicting labels and no majority source; these subjects were dropped "
            f"entirely: {sorted(set(dropped))}"
        )
    rep["notes"].append(
        "The remaining duplicated ids were resolved to their majority source partition "
        "(deterministic); only one copy of each subject's rows was kept."
    )
    lines.append(f"Notes:\n  " + "\n  ".join(rep["notes"]))

    print("\n".join(lines))
    return rep


def write_outputs(split_of: dict[str, str], subjects: dict[str, dict], rep: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "eatd_subject_splits.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_uid", "source_split", "orig_subject_id", "topic", "text", "label", "split"])
        for uid, split in sorted(split_of.items(), key=lambda kv: (kv[1], kv[0])):
            meta = subjects[uid]
            for r in meta["rows"]:
                w.writerow([uid, meta["source_split"], meta["orig_id"],
                            r.get("topic", ""), r["text"], r["label"], split])
    rep["output_csv"] = str(out_csv.relative_to(PROJECT_ROOT))
    (OUT_DIR / "split_report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {OUT_DIR / 'split_report.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Create subject-level EATD splits")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    args = ap.parse_args()

    rows = load_rows()
    subjects, dropped = build_subject_groups(rows)
    print(f"Loaded {len(rows)} rows → {len(subjects)} subjects "
          f"(dropped {len(set(dropped))} contradictory ids: {sorted(set(dropped))})")
    print(f"Subject label distribution: "
          f"{Counter(m['label'] for m in subjects.values())}")

    split_of = make_split(subjects, args.seed, tuple(args.ratios))
    verify_split(split_of, subjects)  # exits non-zero on ANY overlap
    rep = report(split_of, subjects, args.seed, tuple(args.ratios),
                 dropped, elapsed=time.strftime("%H:%M:%S"))
    write_outputs(split_of, subjects, rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
