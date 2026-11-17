"""Validate the AVDD-TCMI archive/contents (Hugging Face, CC-BY-NC-4.0).

Two modes:
  archive mode (default when only the .zip exists): reads the ZIP central
  directory, so the 13.5 GiB archive can be audited without extracting it.
  extracted mode (--extracted): walks the extracted tree, checks the label
  CSVs, per-sample feature CSVs and TCM facial-region images.

Documented layout (dataset card, verified 2026-11-13):
  AVDD-TCMI-dataset/
    HDS/<id>/<id>_tcm/*.png, <id>_acoustic.csv, <id>_visual.csv
    SDS/<id>/...
    HDS_labels.csv  (id,label,gender)
    SDS_labels.csv

Writes data/reports/avdd_tcmi_validation.json.

--remote validates the record from data/reports/avdd_tcmi_remote_inspection.json
(the ZIP central directory read over HTTP ranges) when the archive has not been
downloaded. It reports real member and label counts but marks checksum and
per-file readability as unverified.

Usage:
  .venv/Scripts/python scripts/datasets/validate_avdd_tcmi.py
  .venv/Scripts/python scripts/datasets/validate_avdd_tcmi.py --extracted
  .venv/Scripts/python scripts/datasets/validate_avdd_tcmi.py --remote
Exit codes: 0 ok, 1 failures, 4 dataset absent.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXIT_FAIL, EXIT_MISSING, EXIT_OK, RAW, REPORTS,
    count_files, dir_size, get_logger, human_bytes, parse_avdd_labels, read_json,
    utcnow, write_json,
)

DEFAULT_DIR = RAW / "avdd_tcmi"
ARCHIVE_NAME = "AVDD-TCMI-dataset.zip"
REPORT = REPORTS / "avdd_tcmi_validation.json"
INSPECT_REPORT = REPORTS / "avdd_tcmi_remote_inspection.json"
# Real archive members are prefixed with the top-level folder
# ("AVDD-TCMI-dataset/HDS/0/..."), so the prefix must be optional here.
SAMPLE_RE = re.compile(r"^(?:[^/]+/)?(HDS|SDS)/([^/]+)/")
DECLARED = {"samples": 1230, "depressed": 261, "non_depressed": 969}


def read_labels_csv(raw: bytes | str) -> tuple[list[str], list[dict]]:
    """The released label CSVs are header-less `id,label,gender`.

    csv.DictReader assumes a header, which silently dropped the first label row
    and produced no recognisable columns; the shared parser handles both cases.
    """
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    return parse_avdd_labels(text)


def summarise_labels(rows: list[dict], subset: str, columns: list[str]) -> dict:
    label_col = next((c for c in columns if "label" in c.lower()), "label")
    gender_col = next((c for c in columns if "gender" in c.lower()), "gender")
    counts = Counter(str(r.get(label_col, "")).strip() for r in rows)
    genders = Counter(str(r.get(gender_col, "")).strip() for r in rows)
    ids = [str(r.get("id") or r.get("ID") or "").strip() for r in rows]
    return {
        "subset": subset,
        "rows": len(rows),
        "columns": columns,
        "label_distribution": dict(sorted(counts.items())),
        "gender_distribution": dict(sorted(genders.items())),
        "unique_ids": len({i for i in ids if i}),
        "duplicate_ids": len(ids) - len({i for i in ids if i}),
    }


def validate_archive(path: Path, log) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        by_subset: Counter = Counter()
        samples: dict[str, set[str]] = defaultdict(set)
        kinds: Counter = Counter()
        for n in names:
            if n.endswith("/"):
                continue
            m = SAMPLE_RE.match(n)
            if m:
                by_subset[m.group(1)] += 1
                samples[m.group(1)].add(m.group(2))
            kinds[Path(n).suffix.lower() or "<noext>"] += 1

        labels = {}
        for subset in ("HDS", "SDS"):
            member = f"AVDD-TCMI-dataset/{subset}_labels.csv"
            alt = f"{subset}_labels.csv"
            found = next((x for x in names if x.endswith(f"{subset}_labels.csv")), None)
            if found:
                columns, rows = read_labels_csv(zf.read(found))
                labels[subset] = summarise_labels(rows, subset, columns)
                labels[subset]["member"] = found
                sample_ids = samples.get(subset, set())
                label_ids = {str(r.get("id", "")).strip() for r in rows if str(r.get("id", "")).strip()}
                labels[subset]["sample_dirs_in_archive"] = len(sample_ids)
                labels[subset]["samples_without_label"] = sorted(sample_ids - label_ids)
                labels[subset]["labels_without_sample"] = sorted(label_ids - sample_ids)
            else:
                log.warning("label file for %s not found in archive (looked for %s / %s)", subset, member, alt)

        acoustic = sum(1 for n in names if n.endswith("_acoustic.csv"))
        visual = sum(1 for n in names if n.endswith("_visual.csv"))
        tcm_images = sum(1 for n in names
                         if not n.endswith("/") and n.lower().endswith((".png", ".jpg", ".jpeg")))
        raw_audio = sum(1 for n in names if n.lower().endswith((".wav", ".flac", ".mp3", ".m4a")))
        raw_video = sum(1 for n in names if n.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
        # CRC scan on a >6 GiB archive takes minutes; the download step already
        # runs zipfile.testzip() and records the result, so record it as skipped.
        scanned = path.stat().st_size < 6 * 1024**3
        bad = zf.testzip() if scanned else None

    return {
        "member_count": len(names),
        "files_by_subset": dict(by_subset),
        "samples_by_subset": {k: len(v) for k, v in samples.items()},
        "extension_counts": dict(kinds.most_common(12)),
        "acoustic_feature_csv": acoustic,
        "visual_feature_csv": visual,
        "tcm_region_images": tcm_images,
        "raw_audio_files": raw_audio,
        "raw_video_files": raw_video,
        "labels": labels,
        "crc_scan": ("ok" if bad is None else f"bad member: {bad}") if scanned else "skipped (>6 GiB; see download report)",
    }


def validate_extracted(root: Path, log) -> dict:
    base = root / "AVDD-TCMI-dataset" if (root / "AVDD-TCMI-dataset").exists() else root
    out = {"root": str(base), "samples": {}, "labels": {}}
    for subset in ("HDS", "SDS"):
        sub = base / subset
        ids = sorted(p.name for p in sub.iterdir() if p.is_dir()) if sub.exists() else []
        missing_acoustic = [i for i in ids if not (sub / i / f"{i}_acoustic.csv").exists()]
        missing_visual = [i for i in ids if not (sub / i / f"{i}_visual.csv").exists()]
        no_tcm = [i for i in ids if not (sub / i / f"{i}_tcm").exists()]
        tcm_images = count_files(sub / i / f"{i}_tcm") if ids else 0
        out["samples"][subset] = {
            "directories": len(ids),
            "missing_acoustic_csv": len(missing_acoustic),
            "missing_visual_csv": len(missing_visual),
            "missing_tcm_dir": len(no_tcm),
            "tcm_images_in_first_sample": tcm_images,
        }
        lf = base / f"{subset}_labels.csv"
        if lf.exists():
            rows = read_labels_csv(lf.read_text(encoding="utf-8"))
            info = summarise_labels(rows, subset)
            label_ids = {str(r.get("id", "")).strip() for r in rows}
            info["samples_without_label"] = len([i for i in ids if i not in label_ids])
            info["labels_without_sample"] = len([i for i in label_ids if i and i not in set(ids)])
            out["labels"][subset] = info
        else:
            log.warning("%s missing", lf)

    # feature CSV sanity on one sample per subset
    for subset in ("HDS", "SDS"):
        sub = base / subset
        first = next((p for p in sorted(sub.iterdir()) if p.is_dir()), None) if sub.exists() else None
        if not first:
            continue
        for kind in ("acoustic", "visual"):
            f = first / f"{first.name}_{kind}.csv"
            if not f.exists():
                continue
            with f.open(encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.reader(fh))
            out.setdefault("feature_shapes", {})[f"{subset}/{kind}"] = {
                "rows": len(rows),
                "columns": max((len(r) for r in rows), default=0),
                "file": str(f),
            }
    out["size"] = human_bytes(dir_size(base))
    return out


def validate_remote(inspection: dict) -> tuple[dict, list[str], list[str]]:
    """Validate the record from the remote ZIP directory listing only."""
    issues: list[str] = []
    warnings = [
        "archive is not stored locally: SHA-256, per-member CRC and per-file readability are unverified",
        "feature CSV values were not parsed beyond header/column count",
    ]
    labels = inspection.get("labels", {})
    total_label_rows = sum(v.get("rows", 0) for v in labels.values())
    if total_label_rows and total_label_rows != DECLARED["samples"]:
        issues.append(f"label rows total {total_label_rows} != declared {DECLARED['samples']}")
    depressed = sum(v.get("label_distribution", {}).get("1", 0) for v in labels.values())
    non_depressed = sum(v.get("label_distribution", {}).get("0", 0) for v in labels.values())
    if depressed != DECLARED["depressed"]:
        issues.append(f"depressed samples {depressed} != declared {DECLARED['depressed']}")
    if non_depressed != DECLARED["non_depressed"]:
        issues.append(f"non-depressed samples {non_depressed} != declared {DECLARED['non_depressed']}")
    for subset, info in labels.items():
        if info.get("samples_without_label"):
            issues.append(f"{subset}: samples without a label row {info['samples_without_label']}")
        if info.get("labels_without_sample"):
            issues.append(f"{subset}: label rows without a sample directory {info['labels_without_sample']}")
        if info.get("duplicate_ids"):
            issues.append(f"{subset}: {info['duplicate_ids']} duplicate ids")
    if inspection.get("raw_audio_files") or inspection.get("raw_video_files"):
        issues.append("archive contains raw audio/video; modality documentation must be corrected")
    samples = inspection.get("samples_by_subset", {})
    if sum(samples.values()) != DECLARED["samples"]:
        issues.append(f"sample directories {samples} do not total {DECLARED['samples']}")

    payload = {
        "status": "verified_remote",
        "archive_on_disk": False,
        "archive_bytes": inspection.get("archive_bytes"),
        "uncompressed_bytes": inspection.get("uncompressed_bytes"),
        "member_count": inspection.get("member_count"),
        "file_count": inspection.get("file_count"),
        "samples_by_subset": samples,
        "samples_total": sum(samples.values()),
        "tcm_region_images": inspection.get("tcm_region_images"),
        "acoustic_feature_csv": inspection.get("acoustic_feature_csv"),
        "visual_feature_csv": inspection.get("visual_feature_csv"),
        "raw_audio_files": inspection.get("raw_audio_files"),
        "raw_video_files": inspection.get("raw_video_files"),
        "extension_counts": inspection.get("extension_counts"),
        "labels": labels,
        "label_totals": {"depressed": depressed, "non_depressed": non_depressed,
                         "total": depressed + non_depressed},
        "feature_shapes": inspection.get("feature_shapes"),
        "declared_vs_actual": inspection.get("declared_matches"),
    }
    return payload, issues, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--extracted", action="store_true")
    ap.add_argument("--remote", action="store_true",
                    help="validate from the remote ZIP directory listing (no local data)")
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()
    log = get_logger("validate_avdd_tcmi")

    if args.remote:
        if not INSPECT_REPORT.exists():
            log.error("no remote inspection found; run download_avdd_tcmi.py --inspect-remote first")
            return EXIT_MISSING
        inspection = read_json(INSPECT_REPORT) or {}
        payload, issues, warnings = validate_remote(inspection)
        report = {
            "generated_at": utcnow(),
            "dataset_name": "avdd_tcmi",
            "source": "https://huggingface.co/datasets/edcdd/AVDD-TCMI-dataset",
            "license": "CC-BY-NC-4.0 (non-commercial; attribution required)",
            "language_declared": "zh",
            "declared_totals": DECLARED,
            "modalities": {
                "raw_audio": False, "raw_video": False,
                "preextracted_acoustic_features": True, "preextracted_visual_features": True,
                "tcm_facial_region_images": True,
                "note": "verified from the archive directory: no .wav/.mp3/.mp4 members exist",
            },
            **payload,
            "issues": issues,
            "warnings": warnings,
            "ok": not issues,
        }
        write_json(args.report, report)
        log.info("remote validation: samples=%s labels=%s issues=%d -> %s",
                 payload["samples_total"], payload["label_totals"], len(issues), args.report)
        for w in warnings:
            log.warning("%s", w)
        for i in issues:
            log.error("%s", i)
        return EXIT_OK if not issues else EXIT_FAIL

    archive = args.dir / ARCHIVE_NAME
    extracted_dir = args.dir / "AVDD-TCMI-dataset"
    if not archive.exists() and not extracted_dir.exists():
        log.error("nothing to validate in %s (archive or extracted tree required)", args.dir)
        write_json(args.report, {"generated_at": utcnow(), "status": "missing_data",
                                 "directory": str(args.dir), "ok": False})
        return EXIT_MISSING

    issues: list[str] = []
    payload: dict = {
        "generated_at": utcnow(),
        "status": "validated",
        "dataset_name": "avdd_tcmi",
        "source": "https://huggingface.co/datasets/edcdd/AVDD-TCMI-dataset",
        "license": "CC-BY-NC-4.0 (non-commercial; attribution required)",
        "language_declared": "zh",
        "declared_totals": {"samples": 1230, "non_depressed": 969, "depressed": 261,
                            "HDS": 282, "SDS": 948},
        "modalities": {
            "raw_audio": False,
            "raw_video": False,
            "preextracted_acoustic_features": True,
            "preextracted_visual_features": True,
            "tcm_facial_region_images": True,
            "note": "Dataset card documents feature CSVs and TCM region images, not raw recordings.",
        },
    }

    if args.extracted and extracted_dir.exists():
        log.info("validating extracted tree under %s", args.dir)
        payload["mode"] = "extracted"
        payload.update(validate_extracted(args.dir, log))
        for subset, info in payload["labels"].items():
            if info.get("samples_without_label"):
                issues.append(f"{subset}: {info['samples_without_label']} samples without a label row")
            if info.get("duplicate_ids"):
                issues.append(f"{subset}: {info['duplicate_ids']} duplicate ids in label file")
    elif archive.exists():
        log.info("validating archive %s without extracting", archive)
        payload["mode"] = "archive"
        payload["archive"] = {"path": str(archive), "bytes": archive.stat().st_size,
                              "size_human": human_bytes(archive.stat().st_size)}
        payload.update(validate_archive(archive, log))
        total_samples = sum(payload["samples_by_subset"].values())
        for subset, info in payload["labels"].items():
            if info.get("samples_without_label"):
                issues.append(f"{subset}: {len(info['samples_without_label'])} samples without a label row")
            if info.get("labels_without_sample"):
                issues.append(f"{subset}: {len(info['labels_without_sample'])} label rows without a sample")
            if info.get("duplicate_ids"):
                issues.append(f"{subset}: {info['duplicate_ids']} duplicate ids in label file")
        if total_samples != 1230:
            log.warning("archive sample directories = %d (card claims 1230)", total_samples)
        else:
            log.info("archive sample directories = %d (matches the card)", total_samples)
    else:
        log.info("archive absent; validating extracted tree under %s", args.dir)
        payload["mode"] = "extracted"
        payload.update(validate_extracted(args.dir, log))

    if payload.get("raw_audio_files"):
        issues.append("archive unexpectedly contains raw audio; update the modality documentation")

    payload["issues"] = issues
    payload["ok"] = not issues
    write_json(args.report, payload)
    log.info("mode=%s issues=%d report=%s", payload["mode"], len(issues), args.report)
    for i in issues:
        log.warning("%s", i)
    return EXIT_OK if not issues else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
