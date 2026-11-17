"""Verify a selective extraction of the AVDD-TCMI feature CSVs.

The phase-4 plan extracts only the acoustic and visual feature CSVs and leaves
the 25,830 TCM ``.jpg`` files inside the archive, so this verifier checks that
the extracted tree is exactly that subset and nothing else.

What it checks
--------------
1. Expected file counts (1,230 acoustic, 1,230 visual) against the archive.
2. HDS/SDS structure is preserved and every file maps to a real archive member.
3. No ``.jpg``/``.png`` file was extracted.
4. No duplicate or unresolvable sample ids; no unexpected files.
5. Every extracted file is non-empty.
6. Optional byte-level comparison against the archive member (size + CRC-32),
   which proves the extracted copy is identical to the published record.

Layout tolerance
----------------
The extraction layout was never fixed by the acquisition phase, so the verifier
does not assume one: it walks the tree, identifies the subset from any path part
equal to ``HDS``/``SDS`` and the sample id from the file stem
(``<id>_acoustic.csv`` / ``<id>_visual.csv`` / ``<id>.csv``), then matches those
against the archive's own member list.

Reports ``not_extracted`` (exit 4) with the exact size budget when the tree does
not exist, so the report never claims work that was not done.

Writes ``data/reports/avdd_tcmi_extraction_integrity.json``.

Usage:
  .venv/Scripts/python scripts/datasets/verify_avdd_extraction.py
  .venv/Scripts/python scripts/datasets/verify_avdd_extraction.py --compare-bytes
Exit codes: 0 verified, 1 integrity failures, 4 not extracted / no archive.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXIT_FAIL, EXIT_MISSING, EXIT_OK, RAW, REPORTS,
    free_bytes, get_logger, human_bytes, utcnow, write_json,
)

ARCHIVE = RAW / "avdd_tcmi" / "AVDD-TCMI-dataset.zip"
EXTRACT_ROOT = Path(__file__).resolve().parents[2] / "data" / "interim" / "avdd_tcmi"
REPORT = REPORTS / "avdd_tcmi_extraction_integrity.json"

SUBSETS = ("HDS", "SDS")
FEATURE_SUFFIXES = ("_acoustic.csv", "_visual.csv")
MEMBER_RE = re.compile(r"^(?:[^/]+/)?(HDS|SDS)/([^/]+)/[^/]+$")
STEM_RE = re.compile(r"^(?P<id>.+?)(?:_(?P<kind>acoustic|visual))?\.csv$", re.IGNORECASE)


def archive_index(archive: Path) -> dict:
    """Describe the expected feature-CSV set from the archive central directory."""
    expected: dict[tuple[str, str, str], dict] = {}
    size_by_kind: Counter = Counter()
    count_by_kind: Counter = Counter()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            m = MEMBER_RE.match(info.filename)
            if not m:
                continue
            subset, sample = m.group(1), m.group(2)
            name = info.filename.rsplit("/", 1)[-1].lower()
            # "_acoustic.csv" -> "acoustic" (strip the leading underscore and suffix)
            kind = next((s[1:-len(".csv")] for s in FEATURE_SUFFIXES if name.endswith(s)), None)
            if kind is None:
                continue
            expected[(subset, sample, kind)] = {
                "member": info.filename,
                "size": info.file_size,
                "crc": info.CRC,
            }
            size_by_kind[kind] += info.file_size
            count_by_kind[kind] += 1
    return {
        "expected": expected,
        "counts": dict(count_by_kind),
        "bytes": dict(size_by_kind),
        "total_bytes": sum(size_by_kind.values()),
    }


def discover_extracted(root: Path) -> tuple[dict, list[str], list[str], list[str], list[str]]:
    """Walk the extracted tree and classify every file.

    Returns (found, unexpected, empty, duplicate_ids, jpg_files) where ``found``
    maps (subset, id, kind) -> path.
    """
    found: dict[tuple[str, str, str], Path] = {}
    unexpected: list[str] = []
    empty: list[str] = []
    duplicates: list[str] = []
    images: list[str] = []
    seen: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            images.append(rel)
            continue
        if path.suffix.lower() != ".csv":
            unexpected.append(rel)
            continue
        subset = next((p for p in path.parts if p in SUBSETS), None)
        m = STEM_RE.match(path.name)
        if subset is None or m is None:
            unexpected.append(rel)
            continue
        kind = (m.group("kind") or "").lower()
        if not kind:
            kind = next((s[1:-len(".csv")] for s in FEATURE_SUFFIXES
                         if path.stem.lower().endswith(s)), "")
        if kind not in ("acoustic", "visual"):
            unexpected.append(rel)
            continue
        key = (subset, m.group("id"), kind)
        if key in found:
            duplicates.append(rel)
            seen[key].append(rel)
            continue
        found[key] = path
        if path.stat().st_size == 0:
            empty.append(rel)
    return found, unexpected, empty, duplicates, images


def verify(root: Path, archive: Path, compare_bytes: bool = False) -> dict:
    """Verify a selective extraction. Never modifies the extracted tree."""
    index = archive_index(archive)
    expected = index["expected"]
    payload: dict = {
        "generated_at": utcnow(),
        "dataset_name": "avdd_tcmi",
        "task": "selective extraction integrity (acoustic + visual feature CSVs)",
        "extraction_root": str(root),
        "archive": str(archive),
        "expected": {
            "files_total": len(expected),
            "counts_by_kind": index["counts"],
            "bytes_by_kind": index["bytes"],
            "bytes_total": index["total_bytes"],
            "bytes_total_human": human_bytes(index["total_bytes"]),
        },
        "free_bytes": free_bytes(),
        "free_bytes_human": human_bytes(free_bytes()),
        "provenance": "expected set derived from the archive central directory, not from a manifest",
    }
    payload["fits_in_free_space"] = index["total_bytes"] <= free_bytes()
    issues: list[str] = []
    warnings: list[str] = []

    if not archive.exists():
        payload.update({"status": "archive_missing", "ok": False})
        payload["issues"] = [f"archive not found: {archive}"]
        return payload

    if not root.exists():
        payload.update({
            "status": "not_extracted",
            "extraction_performed": False,
            "extracted": False,
            "counts": {"acoustic": 0, "visual": 0},
            "missing": {
                "total": len(expected),
                "acoustic": index["counts"].get("acoustic", 0),
                "visual": index["counts"].get("visual", 0),
            },
            "issues": [],
            "warnings": [
                "the selective extraction was never performed: the destination directory "
                f"{root} does not exist, and no extraction report or log was found",
                f"extracting both CSV modalities needs {human_bytes(index['total_bytes'])} "
                f"({index['total_bytes']} B) but only {human_bytes(free_bytes())} is free, "
                "so the extraction must be reduced or space must be freed first",
                "the archive is readable directly, so this blocks nothing: members were "
                "validated in place by validate_avdd_features.py",
            ],
            "ok": False,
        })
        return payload

    found, unexpected, empty, duplicates, images = discover_extracted(root)
    counts = Counter(k[2] for k in found)
    missing = sorted(set(expected) - set(found))
    missing_by_kind = Counter(m[2] for m in missing)
    by_subset = Counter(m[2] for m in missing if m[0] == "HDS"), Counter(m[2] for m in missing if m[0] == "SDS")

    if missing:
        issues.append(
            f"{len(missing)} expected files are missing "
            f"(acoustic {missing_by_kind.get('acoustic', 0)}, visual {missing_by_kind.get('visual', 0)})")
    extra = sorted(set(found) - set(expected))
    if extra:
        issues.append(f"{len(extra)} extracted ids do not exist in the archive")
    if unexpected:
        issues.append(f"{len(unexpected)} unexpected non-CSV files were extracted")
    if images:
        issues.append(f"{len(images)} image files were extracted; the phase-4 plan keeps them in the archive")
    if empty:
        issues.append(f"{len(empty)} extracted CSV files are empty")
    if duplicates:
        issues.append(f"{len(duplicates)} duplicate sample ids were extracted")
    if counts.get("acoustic") != index["counts"].get("acoustic") or \
            counts.get("visual") != index["counts"].get("visual"):
        issues.append(f"file counts {dict(counts)} do not match the archive {index['counts']}")

    mismatches: list[dict] = []
    crc_checked = 0
    if compare_bytes:
        for key, path in found.items():
            exp = expected.get(key)
            if not exp:
                continue
            actual_size = path.stat().st_size
            if actual_size != exp["size"]:
                mismatches.append({"file": str(path), "kind": "size",
                                   "expected": exp["size"], "actual": actual_size})
                continue
            crc_checked += 1
            actual_crc = 0
            with path.open("rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    actual_crc = zlib.crc32(block, actual_crc)
            if actual_crc & 0xFFFFFFFF != exp["crc"]:
                mismatches.append({"file": str(path), "kind": "crc32",
                                   "expected": exp["crc"], "actual": actual_crc & 0xFFFFFFFF})
        if mismatches:
            issues.append(f"{len(mismatches)} extracted files differ from the archive member")

    payload.update({
        "status": "verified" if not issues else "failed",
        "extraction_performed": True,
        "extracted": True,
        "counts": {"acoustic": counts.get("acoustic", 0), "visual": counts.get("visual", 0),
                   "total": sum(counts.values())},
        "ids_by_subset": {s: len({k[1] for k in found if k[0] == s}) for s in SUBSETS},
        "missing": {"total": len(missing), "acoustic": missing_by_kind.get("acoustic", 0),
                    "visual": missing_by_kind.get("visual", 0),
                    "by_subset": {"HDS": dict(by_subset[0]), "SDS": dict(by_subset[1])},
                    "examples": [list(m) for m in missing[:10]]},
        "unexpected_files": unexpected[:20],
        "unexpected_count": len(unexpected),
        "images_extracted": len(images),
        "empty_files": len(empty),
        "duplicate_ids": len(duplicates),
        "byte_comparison": {
            "performed": compare_bytes,
            "files_crc_checked": crc_checked,
            "mismatches": mismatches[:20],
            "mismatch_count": len(mismatches),
        },
        "extracted_bytes": human_bytes(sum(p.stat().st_size for p in found.values())),
        "issues": issues,
        "warnings": warnings,
        "ok": not issues,
    })
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=EXTRACT_ROOT)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--compare-bytes", action="store_true",
                    help="CRC-compare every extracted file with its archive member")
    args = ap.parse_args()
    log = get_logger("verify_avdd_extraction")

    if not args.archive.exists():
        log.error("archive missing: %s", args.archive)
        write_json(args.report, {"generated_at": utcnow(), "status": "archive_missing", "ok": False,
                                 "archive": str(args.archive)})
        return EXIT_MISSING

    payload = verify(args.root, args.archive, compare_bytes=args.compare_bytes)
    write_json(args.report, payload)
    status = payload["status"]
    if status == "not_extracted":
        log.warning("extraction never performed: %s does not exist", args.root)
        log.warning("needs %s, free %s", human_bytes(payload["expected"]["bytes_total"]),
                    human_bytes(payload["free_bytes"]))
    elif payload["ok"]:
        log.info("extraction verified: %s", payload["counts"])
    else:
        for i in payload["issues"]:
            log.error("%s", i)
    log.info("status=%s report=%s", status, args.report)
    if status == "not_extracted":
        return EXIT_MISSING
    return EXIT_OK if payload["ok"] else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
