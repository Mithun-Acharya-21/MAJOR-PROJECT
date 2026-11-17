"""Record and verify SHA-256 checksums for acquired dataset files.

Registry of checksummed files lives in data/manifests/dataset_checksums.csv.
Only dataset roots are scanned (never the whole repository), archives and
data files only, and EATD is included because its manifest hashes are content
hashes rather than file hashes.

Usage:
  .venv/Scripts/python scripts/datasets/verify_dataset_checksums.py --record
  .venv/Scripts/python scripts/datasets/verify_dataset_checksums.py
Exit codes: 0 ok, 1 mismatch or missing file.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXIT_FAIL, EXIT_OK, MANIFESTS, PROJECT_ROOT, RAW, DATA,
    get_logger, human_bytes, sha256_file, utcnow, write_csv,
)

CHECKSUMS = MANIFESTS / "dataset_checksums.csv"
COLUMNS = ["dataset_name", "file", "bytes", "sha256", "md5", "recorded_at", "source"]

ROOTS = {
    "harvard_tweets": RAW / "text" / "harvard_tweets",
    "avdd_tcmi": RAW / "avdd_tcmi",
    "eatd_archives": DATA / "eatd",
}
SUFFIXES = {".zip", ".xlsx", ".csv", ".wav", ".txt", ".tar", ".gz", ".7z"}
MAX_FILES = 2000  # safety cap; EATD has ~1.8k files


def iter_files(root: Path):
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUFFIXES:
            yield p


def load() -> dict[str, dict]:
    if not CHECKSUMS.exists():
        return {}
    with CHECKSUMS.open(encoding="utf-8", newline="") as fh:
        return {r["file"]: r for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true", help="hash files and (re)write the registry")
    ap.add_argument("--dataset", choices=sorted(ROOTS), default=None)
    args = ap.parse_args()
    log = get_logger("verify_dataset_checksums")

    rows = load()
    targets = {args.dataset: ROOTS[args.dataset]} if args.dataset else ROOTS
    added = updated = unchanged = 0
    scanned = 0
    problems: list[str] = []

    for name, root in targets.items():
        files = list(iter_files(root))
        if not files:
            log.info("%s: no checksummed files under %s", name, root)
            continue
        if len(files) > MAX_FILES:
            log.warning("%s: %d files exceed the cap of %d; truncating", name, len(files), MAX_FILES)
            files = files[:MAX_FILES]
        for p in files:
            scanned += 1
            rel = str(p.relative_to(PROJECT_ROOT))
            digest = sha256_file(p)
            if not args.record:
                prev = rows.get(rel)
                if prev is None:
                    problems.append(f"unrecorded file: {rel}")
                elif prev["sha256"] != digest:
                    problems.append(f"checksum mismatch: {rel}")
                else:
                    unchanged += 1
                continue
            prev = rows.get(rel)
            rows[rel] = {
                "dataset_name": name,
                "file": rel,
                "bytes": p.stat().st_size,
                "sha256": digest,
                "md5": "",
                "recorded_at": utcnow(),
                "source": prev["source"] if prev else "local acquisition",
            }
            if prev is None:
                added += 1
            elif prev["sha256"] != digest:
                updated += 1
            else:
                unchanged += 1

    if args.record:
        ordered = [rows[k] for k in sorted(rows)]
        write_csv(CHECKSUMS, ordered, COLUMNS)
        log.info("registry: %d files (+%d new, %d changed, %d unchanged) -> %s",
                 len(ordered), added, updated, unchanged, CHECKSUMS)
        return EXIT_OK

    log.info("verified %d files (%d unchanged, %d problems)", scanned, unchanged, len(problems))
    for p in problems[:40]:
        log.error("%s", p)
    return EXIT_OK if not problems else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
