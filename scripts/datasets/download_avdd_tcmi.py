"""Download the AVDD-TCMI dataset from Hugging Face (CC-BY-NC-4.0).

Source: https://huggingface.co/datasets/edcdd/AVDD-TCMI-dataset

Verified record facts (2026-11-13):
  - one archive: AVDD-TCMI-dataset.zip, 14,529,499,485 bytes (13.5 GiB)
  - public, not gated, licence cc-by-nc-4.0, language zh
  - the archive contains pre-extracted acoustic/visual CSVs and TCM
    facial-region images, not raw audio or raw video

This script refuses to start unless free space covers the archive plus a
configurable extraction allowance. It resumes with HTTP Range and verifies the
archive with zipfile.testzip() before any extraction.

--inspect-remote reads the ZIP central directory over HTTP range requests and
records the real member list, sample counts, label distributions and feature
CSV headers without storing the archive. Use it to answer "what is actually in
there" before authorising a 13.5 GiB download.

Usage:
  .venv/Scripts/python scripts/datasets/download_avdd_tcmi.py --dry-run
  .venv/Scripts/python scripts/datasets/download_avdd_tcmi.py --inspect-remote
  .venv/Scripts/python scripts/datasets/download_avdd_tcmi.py --extract
Exit codes: 0 ok, 1 corrupt archive, 2 refused (space/licence), 4 not retrievable.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXIT_FAIL, EXIT_MISSING, EXIT_OK, EXIT_REFUSED, RAW, REPORTS, USER_AGENT,
    free_bytes, get_logger, human_bytes, parse_avdd_labels, sha256_file, utcnow,
    write_json,
)
from _remote_zip import RemoteZip  # noqa: E402

REPO_ID = "edcdd/AVDD-TCMI-dataset"
ARCHIVE = "AVDD-TCMI-dataset.zip"
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{ARCHIVE}"
API = f"https://huggingface.co/api/datasets/{REPO_ID}"
TREE_API = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
DEST = RAW / "avdd_tcmi"
REPORT = REPORTS / "avdd_tcmi_download.json"
INSPECT_REPORT = REPORTS / "avdd_tcmi_remote_inspection.json"
EXTRACTED_SUBDIR = "AVDD-TCMI-dataset"
SAMPLE_RE = re.compile(r"^AVDD-TCMI-dataset/(HDS|SDS)/([^/]+)/")
DECLARED = {"samples": 1230, "depressed": 261, "non_depressed": 969,
            "HDS": {"samples": 282, "depressed": 123, "non_depressed": 159},
            "SDS": {"samples": 948, "depressed": 138, "non_depressed": 810}}


def remote_meta(session: requests.Session) -> dict:
    r = session.get(API, timeout=60)
    r.raise_for_status()
    meta = r.json()
    tree = session.get(TREE_API, timeout=60).json()
    entry = next((e for e in tree if e.get("path") == ARCHIVE), None)
    if entry is None:
        raise RuntimeError(f"{ARCHIVE} not found in repo tree")
    return {
        "repo_id": REPO_ID,
        "repo_url": f"https://huggingface.co/datasets/{REPO_ID}",
        "license": next((t.split(":", 1)[1] for t in meta.get("tags", []) if t.startswith("license:")), None),
        "language": meta.get("cardData", {}).get("language"),
        "gated": meta.get("gated", False),
        "private": meta.get("private", False),
        "last_modified": meta.get("lastModified"),
        "archive": ARCHIVE,
        "archive_bytes": entry.get("size"),
        "download_url": URL,
        "lfs_sha256": (entry.get("lfs") or {}).get("oid"),
    }


def download(session: requests.Session, url: str, dest: Path, expected: int | None, log) -> None:
    # A 13.5 GiB binary zip must never go through a content decoder. requests
    # advertises gzip/deflate/zstd by default; if the CDN honours that, urllib3
    # has to buffer decoded output while streaming and the run dies with
    # MemoryError part-way through. Ask for identity encoding instead.
    session.headers["Accept-Encoding"] = "identity"
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    if have:
        log.info("resuming at %s", human_bytes(have))
    with session.get(url, headers=headers, stream=True, timeout=300) as r:
        if r.status_code == 416:
            log.warning("server says range unsatisfiable; discarding the partial file")
            r.close()
            part.unlink(missing_ok=True)
            raise RuntimeError("range unsatisfiable; partial file removed, retry to download fresh")
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code}")

        # Only append when the server confirms it is sending exactly the bytes we
        # are missing. A 206 that starts anywhere else (CDN rewrites, differing
        # revisions) would silently produce a corrupt file that still has the
        # right length, so the start offset is checked, not trusted.
        mode, restart_reason = "ab", None
        if not have:
            mode = "wb"
        elif r.status_code == 200:
            mode, restart_reason = "wb", "server ignored the range request"
        else:
            content_range = r.headers.get("Content-Range", "")
            m = re.match(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range)
            if not m:
                mode, restart_reason = "wb", f"missing/unparsable Content-Range {content_range!r}"
            elif int(m.group(1)) != have:
                mode, restart_reason = "wb", f"server started at {m.group(1)} instead of {have}"
        if restart_reason:
            log.warning("restarting download from zero: %s", restart_reason)
            have = 0
        response_len = int(r.headers.get("Content-Length") or 0)
        total = (have + response_len) if response_len else expected
        done, last_log = have, 0
        with part.open(mode) as fh:
            if mode == "wb":
                fh.truncate(0)
            for block in r.raw.stream(1 << 20, decode_content=False):
                if not block:
                    continue
                fh.write(block)
                done += len(block)
                if done - last_log > 500 * 1024 * 1024:
                    last_log = done
                    pct = f"{100 * done / total:.1f}%" if total else "?"
                    log.info("  %s / %s (%s)", human_bytes(done), human_bytes(total or 0), pct)
    size = part.stat().st_size
    if expected and size != expected:
        raise RuntimeError(f"size mismatch {size} != {expected}")
    part.replace(dest)


def inspect_remote(session, meta: dict, log) -> dict:
    """Audit the archive contents without downloading it."""
    log.info("reading ZIP central directory over HTTP range requests (archive not downloaded)")
    rz = RemoteZip(meta["download_url"], session)
    entries = rz.entries()
    files = [e for e in entries if not e.is_dir]

    samples: dict[str, set[str]] = defaultdict(set)
    file_kinds: Counter = Counter()
    kinds_by_subset: dict[str, Counter] = defaultdict(Counter)
    other: list[str] = []
    for e in files:
        m = SAMPLE_RE.match(e.name)
        if m:
            samples[m.group(1)].add(m.group(2))
            kinds_by_subset[m.group(1)][Path(e.name).suffix.lower()] += 1
        else:
            other.append(e.name)
        file_kinds[Path(e.name).suffix.lower() or "<none>"] += 1

    def bytes_for(predicate) -> int:
        return sum(e.uncompressed_size for e in files if predicate(e))

    by_name = {e.name: e for e in entries}
    labels: dict[str, dict] = {}
    for subset in ("HDS", "SDS"):
        name = next((n for n in by_name if n.endswith(f"{subset}_labels.csv")), None)
        if not name:
            log.warning("%s_labels.csv not present in archive", subset)
            continue
        text = rz.read_prefix(by_name[name], 8 << 20).decode("utf-8-sig")
        columns, rows = parse_avdd_labels(text)
        ids = [str(r.get("id", "")).strip() for r in rows]
        label_counts = Counter(str(r.get("label", "")).strip() for r in rows)
        gender_counts = Counter(str(r.get("gender", "")).strip() for r in rows)
        sample_ids = samples.get(subset, set())
        declared = DECLARED[subset]
        labels[subset] = {
            "member": name,
            "rows": len(rows),
            "columns": columns,
            "header_row_present": columns[0] == "id" and rows and "label" in rows[0],
            "label_distribution": dict(sorted(label_counts.items())),
            "gender_distribution": dict(sorted(gender_counts.items())),
            "unique_ids": len({i for i in ids if i}),
            "duplicate_ids": len([i for i in ids if i]) - len({i for i in ids if i}),
            "sample_dirs_in_archive": len(sample_ids),
            "samples_without_label": sorted(sample_ids - {i for i in ids if i}),
            "labels_without_sample": sorted({i for i in ids if i} - sample_ids),
            "declared_by_card": declared,
            "declared_label_counts_match": {
                "depressed": label_counts.get("1", 0) == declared["depressed"],
                "non_depressed": label_counts.get("0", 0) == declared["non_depressed"],
            },
        }

    # feature CSV shape: header line + byte size of one sample per subset
    shapes: dict[str, dict] = {}
    for subset in ("HDS", "SDS"):
        first = sorted(samples.get(subset, []))[0] if samples.get(subset) else None
        if first is None:
            continue
        for kind in ("acoustic", "visual"):
            name = f"AVDD-TCMI-dataset/{subset}/{first}/{first}_{kind}.csv"
            if name not in by_name:
                continue
            head = rz.read_prefix(by_name[name], 1 << 16).decode("utf-8-sig", "replace")
            header = head.splitlines()[0] if head else ""
            cols = next(csv.reader([header])) if header else []
            shapes[f"{subset}/{kind}"] = {
                "sample": first,
                "columns": len(cols),
                "column_names": cols[:24],
                "file_bytes": by_name[name].uncompressed_size,
            }

    tcm_images = [e for e in files if "/_tcm/" in e.name or re.search(r"_tcm/", e.name)]
    raw_audio = [e for e in files if e.name.lower().endswith((".wav", ".flac", ".mp3", ".m4a"))]
    raw_video = [e for e in files if e.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

    return {
        "generated_at": utcnow(),
        "mode": "remote_zip_directory",
        "archive": meta,
        "archive_bytes": meta["archive_bytes"],
        "archive_on_disk": False,
        "member_count": len(entries),
        "file_count": len(files),
        "directory_count": len(entries) - len(files),
        "uncompressed_bytes": sum(e.uncompressed_size for e in files),
        "samples_by_subset": {k: len(v) for k, v in samples.items()},
        "files_by_subset": {k: sum(c.values()) for k, c in kinds_by_subset.items()},
        "files_by_subset_extension": {k: dict(c.most_common()) for k, c in kinds_by_subset.items()},
        "extension_counts": dict(file_kinds.most_common(20)),
        "top_level_entries": [n for n in by_name if n.count("/") <= 1][:20],
        "tcm_region_images": len(tcm_images),
        "tcm_images_bytes": sum(e.uncompressed_size for e in tcm_images),
        "raw_audio_files": len(raw_audio),
        "raw_video_files": len(raw_video),
        "acoustic_feature_csv": sum(1 for e in files if e.name.endswith("_acoustic.csv")),
        "visual_feature_csv": sum(1 for e in files if e.name.endswith("_visual.csv")),
        "labels": labels,
        "feature_shapes": shapes,
        "declared_by_card": DECLARED,
        "declared_matches": {
            "total_samples": sum(len(v) for v in samples.values()) == DECLARED["samples"],
            "HDS_samples": len(samples.get("HDS", ())) == DECLARED["HDS"]["samples"],
            "SDS_samples": len(samples.get("SDS", ())) == DECLARED["SDS"]["samples"],
            "modalities": {"raw_audio_absent": len(raw_audio) == 0, "raw_video_absent": len(raw_video) == 0},
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inspect-remote", action="store_true",
                    help="audit archive contents via range requests, downloading nothing")
    ap.add_argument("--extract", action="store_true", help="extract after a successful download")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extract-allowance-gb", type=float, default=16.0,
                    help="free space to keep beyond the archive for extraction")
    args = ap.parse_args()
    log = get_logger("download_avdd_tcmi")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    meta = remote_meta(session)

    if args.inspect_remote:
        payload = inspect_remote(session, meta, log)
        write_json(INSPECT_REPORT, payload)
        for subset, info in payload["labels"].items():
            log.info("%s: %d label rows %s (archive has %d sample dirs, %d unlabelled %s)",
                     subset, info["rows"], info["label_distribution"], info["sample_dirs_in_archive"],
                     len(info["samples_without_label"]), info["samples_without_label"])
            log.info("%s declared-vs-actual labels: %s", subset, info["declared_label_counts_match"])
        log.info("files=%d images=%d acoustic_csv=%d visual_csv=%d uncompressed=%s",
                 payload["file_count"], payload["tcm_region_images"],
                 payload["acoustic_feature_csv"], payload["visual_feature_csv"],
                 human_bytes(payload["uncompressed_bytes"]))
        log.info("declared-vs-actual: %s", payload["declared_matches"])
        log.info("inspection report -> %s (archive NOT downloaded)", INSPECT_REPORT)
        return EXIT_OK

    need = (meta["archive_bytes"] or 0) + int(args.extract_allowance_gb * 1024**3)
    free = free_bytes()
    log.info("archive %s | licence %s | free %s | required %s",
             human_bytes(meta["archive_bytes"] or 0), meta["license"], human_bytes(free), human_bytes(need))
    if meta["license"] and "nc" in meta["license"].lower():
        log.info("NON-COMMERCIAL licence: academic research use only; attribution required")

    plan = {
        "generated_at": utcnow(),
        "dataset": meta,
        "destination": str(args.dest / meta["archive"]),
        "free_bytes": free,
        "required_bytes": need,
        "extract_allowance_gb": args.extract_allowance_gb,
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        feasible = free >= need
        plan["status"] = "dry_run"
        plan["feasible"] = feasible
        write_json(REPORT, plan)
        log.info("DRY RUN — %s: archive %s + extraction allowance %s vs free %s",
                 "feasible" if feasible else "NOT FEASIBLE on current disk",
                 human_bytes(meta["archive_bytes"] or 0), human_bytes(int(args.extract_allowance_gb * 1024**3)),
                 human_bytes(free))
        return EXIT_OK if feasible else EXIT_REFUSED

    if free < need:
        log.error("refusing: need %s free, have %s (free up disk or lower --extract-allowance-gb)",
                  human_bytes(need), human_bytes(free))
        plan["status"] = "refused_no_space"
        write_json(REPORT, plan)
        return EXIT_REFUSED

    args.dest.mkdir(parents=True, exist_ok=True)
    archive = args.dest / meta["archive"]
    if archive.exists() and not args.force:
        log.info("archive already present (%s); verifying integrity", human_bytes(archive.stat().st_size))
        status = "reused"
    else:
        download(session, meta["download_url"], archive, meta["archive_bytes"], log)
        status = "downloaded"

    expected_sha = meta.get("lfs_sha256")
    actual_sha = sha256_file(archive)
    plan.update({
        "status": status,
        "size_on_disk": archive.stat().st_size,
        "sha256_on_disk": actual_sha,
        "sha256_expected": expected_sha,
        "sha256_matches": (actual_sha == expected_sha) if expected_sha else None,
    })
    log.info("archive sha256 %s (%s)", actual_sha[:16] + "…",
             "matches the Hub LFS oid" if plan["sha256_matches"] else
             "NOT verified against the Hub LFS oid" if plan["sha256_matches"] is None else "MISMATCH")
    if expected_sha and actual_sha != expected_sha:
        log.error("sha256 mismatch: on disk %s != expected %s", actual_sha, expected_sha)
        write_json(REPORT, plan)
        return EXIT_FAIL

    log.info("testing archive integrity (zipfile.testzip) — this reads the whole archive")
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    plan.update({
        "archive_entries": len(names),
        "archive_testzip_bad_member": bad,
        "completed_at": utcnow(),
    })
    if bad:
        log.error("corrupt archive member: %s", bad)
        write_json(REPORT, plan)
        return EXIT_FAIL

    if args.extract:
        out = args.dest
        log.info("extracting to %s", out)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
        plan["extracted_to"] = str(out / EXTRACTED_SUBDIR if (out / EXTRACTED_SUBDIR).exists() else out)
        plan["free_after_extract"] = free_bytes()
        log.info("extracted; free space now %s", human_bytes(free_bytes()))

    write_json(REPORT, plan)
    log.info("%s %s", status, archive)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
