"""
Validate the local FER-2013 copy (Phase 7).

Checks class names, per-class image counts, expected FER-2013 structure,
corrupted images (sampled), and duplicates (by file size + sampled hashes).
Writes a machine-readable verdict to data/fer2013/validation_report.json.

The report explicitly marks whether the dataset may be used for training a
seven-class model. An incomplete copy must NEVER be presented as full
FER-2013 training data (research-integrity rule).

Usage:
    python scripts/validate_fer2013.py [--full-hash] [--sample 200]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FER_DIR = PROJECT_ROOT / "data" / "fer2013"
REPORT_PATH = FER_DIR / "validation_report.json"

EXPECTED_CLASSES = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}
# Published FER-2013 training-set counts (Kaggle icml_face_data, train partition)
EXPECTED_TRAIN_COUNTS = {
    "angry": 3995, "disgust": 436, "fear": 4097, "happy": 7215,
    "sad": 4830, "surprise": 3171, "neutral": 4965,
}


def count_images(d: Path) -> tuple[dict, list[str]]:
    counts: dict[str, int] = {}
    missing: list[str] = []
    if not d.is_dir():
        return counts, [f"{d} is missing entirely"]
    present = {p.name for p in d.iterdir() if p.is_dir()}
    for cls in sorted(EXPECTED_CLASSES):
        cls_dir = d / cls
        if not cls_dir.is_dir():
            missing.append(cls)
            counts[cls] = 0
            continue
        counts[cls] = sum(
            1 for f in cls_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
    extra = sorted(present - EXPECTED_CLASSES)
    return counts, missing + [f"unexpected class dir: {e}" for e in extra]


def check_corrupted(cls_dir: Path, sample: int) -> list[str]:
    bad = []
    files = [f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    step = max(1, len(files) // sample) if sample and len(files) > sample else 1
    for f in files[::step]:
        try:
            with Image.open(f) as img:
                img.verify()
        except Exception as e:
            bad.append(f"{f.name}: {e}")
    return bad


def find_duplicates(d: Path, sample: int) -> list[str]:
    """Exact duplicates via (size, hash) over a deterministic sample."""
    seen: dict[str, str] = {}
    dups = []
    files = sorted(d.rglob("*.jpg")) + sorted(d.rglob("*.png"))
    step = max(1, len(files) // sample) if sample and len(files) > sample else 1
    for f in files[::step]:
        try:
            size = f.stat().st_size
            h = hashlib.md5(f.read_bytes()).hexdigest()
            key = f"{size}:{h}"
            if key in seen:
                dups.append(f"{f.relative_to(d)} duplicates {seen[key]}")
            seen[key] = str(f.relative_to(d))
        except Exception:
            continue
    return dups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200,
                    help="max files per class checked for corruption/duplicates")
    ap.add_argument("--full-hash", action="store_true",
                    help="hash every file (slow) instead of sampling")
    args = ap.parse_args()
    sample = 10**9 if args.full_hash else args.sample

    if not FER_DIR.is_dir():
        print(f"FATAL: {FER_DIR} not found")
        return 1

    report: dict = {
        "dataset": "FER-2013 (local copy)",
        "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_classes": sorted(EXPECTED_CLASSES),
        "splits": {},
        "usable_for_seven_class_training": False,
        "integrity_statement": "",
    }

    complete_train = True
    for split in ("train", "test"):
        d = FER_DIR / split
        counts, problems = count_images(d)
        corrupted = []
        duplicates = []
        if d.is_dir():
            for cls in sorted(EXPECTED_CLASSES):
                cls_dir = d / cls
                if cls_dir.is_dir():
                    corrupted += check_corrupted(cls_dir, sample)
            duplicates = find_duplicates(d, sample)
        expected = EXPECTED_TRAIN_COUNTS if split == "train" else None
        missing_classes = [c for c in sorted(EXPECTED_CLASSES) if counts.get(c, 0) == 0]
        split_complete = not missing_classes
        if split == "train" and not split_complete:
            complete_train = False
        report["splits"][split] = {
            "exists": d.is_dir(),
            "class_counts": counts,
            "total_images": sum(counts.values()),
            "missing_classes": missing_classes,
            "expected_train_counts": expected,
            "count_deficits": (
                {c: {"present": counts.get(c, 0), "expected": expected[c]}
                 for c in sorted(EXPECTED_CLASSES)
                 if expected and counts.get(c, 0) < expected[c]}
                if expected else {}
            ),
            "corrupted_sampled": corrupted[:20],
            "corrupted_count_sampled": len(corrupted),
            "duplicates_sampled": duplicates[:20],
            "duplicate_count_sampled": len(duplicates),
            "problems": problems,
        }

    # Verdict
    if complete_train:
        report["usable_for_seven_class_training"] = True
        report["integrity_statement"] = (
            "All seven classes present in train; the copy may be used for "
            "seven-class expression training with the caveat that expression "
            "labels are NOT depression labels (Phase 8)."
        )
    else:
        report["usable_for_seven_class_training"] = False
        missing = report["splits"]["train"]["missing_classes"]
        report["integrity_statement"] = (
            f"INCOMPLETE FER-2013 TRAINING COPY: train is missing classes "
            f"{missing} (and several present classes are far below published "
            f"counts). This copy MUST NOT be used for seven-class FER-2013 "
            f"training, and NO results may be reported as 'trained on full "
            f"FER-2013'. Expression modeling remains a separate, "
            f"clearly-labeled auxiliary experiment (Phase 8): FER-2013 is a "
            f"facial-expression dataset and does not independently establish "
            f"depression classification."
        )

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "train_class_counts": report["splits"]["train"]["class_counts"],
        "train_missing": report["splits"]["train"]["missing_classes"],
        "train_total": report["splits"]["train"]["total_images"],
        "test_total": report["splits"]["test"]["total_images"],
        "corrupted_sampled_train": report["splits"]["train"]["corrupted_count_sampled"],
        "duplicates_sampled_train": report["splits"]["train"]["duplicate_count_sampled"],
        "usable_for_seven_class_training": report["usable_for_seven_class_training"],
    }, indent=2))
    print("\n" + report["integrity_statement"])
    print(f"\nSaved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
