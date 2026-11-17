"""Generate the dataset inventory from validated artifacts.

Writes data/manifests/dataset_inventory.{csv,json} and, from the same rows,
data/reports/dataset_inventory.json (the research-readiness view).

The inventory is derived, never hand-written: counts come from the validation
reports in data/reports/ plus a filesystem size scan. A dataset with no
validation report is inventoried with status "not_acquired" and null counts —
no numbers are invented. Record counts are never promoted to participant
counts: AVDD-TCMI ships 1,230 samples with no participant metadata, so its
participant field stays null.

Usage:
  .venv/Scripts/python scripts/datasets/generate_dataset_inventory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DATA, EXIT_OK, MANIFESTS, RAW, REPORTS, dir_size, get_logger, human_bytes, read_json, utcnow, write_csv, write_json,
)

OUT_CSV = MANIFESTS / "dataset_inventory.csv"
OUT_JSON = MANIFESTS / "dataset_inventory.json"
OUT_REPORT_JSON = REPORTS / "dataset_inventory.json"

COLUMNS = [
    "dataset_name", "category", "status", "source_url", "license", "language",
    "participants", "sessions", "records", "text_samples", "audio_recordings",
    "visual_samples", "modalities", "data_form", "size_bytes", "size_human",
    "validated", "validation_report", "access_requirement", "label_mapping_verified",
]

CATEGORY = {
    "eatd": "depression text + audio (Chinese interviews)",
    "harvard_tweets": "depression-related text (English tweets)",
    "avdd_tcmi": "depression audio-visual features (Chinese)",
    "daic_woz": "paired text + audio + visual interviews (English)",
    "rsdd": "depression-related text at user level (English Reddit)",
    "smhd": "mental-health text at user level (English Reddit)",
}


def _row_from_validation(name: str, path: Path) -> dict | None:
    report = read_json(path)
    if not report or report.get("status") in (None, "missing_data", "missing_manifest"):
        return None
    return report


def build_eatd() -> dict:
    rep = _row_from_validation("eatd", REPORTS / "eatd_validation.json")
    size = dir_size(DATA / "eatd")
    base = {
        "dataset_name": "eatd",
        "category": CATEGORY["eatd"],
        "source_url": "https://github.com/Fancy-Block/EATD-Corpus",
        "license": "no licence declared upstream (verified: GitHub license endpoint 404, README states no terms)",
        "language": "zh",
        "modalities": "text+audio",
        "data_form": "raw transcripts (txt) + raw 16 kHz stereo WAV",
        "size_bytes": size,
        "size_human": human_bytes(size),
        "access_requirement": "public GitHub release; redistribution terms unconfirmed",
    }
    if not rep:
        return {**base, "status": "present_but_unvalidated", "validated": False}
    counts = rep["counts"]
    return {
        **base,
        "status": "available",
        "participants": counts["participants"],
        "sessions": counts["sessions"],
        "records": counts["sessions"],
        "text_samples": counts["text_files"],
        "audio_recordings": counts["audio_files"],
        "visual_samples": 0,
        "validated": bool(rep.get("ok")),
        "validation_report": "data/reports/eatd_validation.json",
        "access_requirement": base["access_requirement"],
        "label_mapping_verified": True,
    }


def build_harvard() -> dict:
    rep = _row_from_validation("harvard_tweets", REPORTS / "harvard_tweets_validation.json")
    base = {
        "dataset_name": "harvard_tweets",
        "category": CATEGORY["harvard_tweets"],
        "source_url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ISZCSA",
        "license": "CC0 1.0",
        "language": "en",
        "modalities": "text",
        "data_form": "single .xlsx (tweets + binary label)",
        "access_requirement": "none (public, unrestricted download)",
    }
    size = dir_size(RAW / "text" / "harvard_tweets")
    if not rep:
        return {**base, "status": "not_acquired", "size_bytes": size, "size_human": human_bytes(size), "validated": False}
    s = rep["statistics"]
    return {
        **base,
        "status": "available",
        "participants": None,  # no user/tweet id column exists: records != participants
        "sessions": None,
        "records": s["records_with_text"],
        "text_samples": s["records_with_text"],
        "audio_recordings": 0,
        "visual_samples": 0,
        "size_bytes": size,
        "size_human": human_bytes(size),
        "validated": bool(rep.get("ok")),
        "validation_report": "data/reports/harvard_tweets_validation.json",
        "label_mapping_verified": False,
    }


def build_avdd() -> dict:
    rep = _row_from_validation("avdd_tcmi", REPORTS / "avdd_tcmi_validation.json")
    base = {
        "dataset_name": "avdd_tcmi",
        "category": CATEGORY["avdd_tcmi"],
        "source_url": "https://huggingface.co/datasets/edcdd/AVDD-TCMI-dataset",
        "license": "CC-BY-NC-4.0 (non-commercial, attribution)",
        "language": "zh",
        "modalities": "audio-features+visual-features+facial-images",
        "data_form": "pre-extracted acoustic/visual CSVs + TCM facial-region images (no raw recordings)",
        "access_requirement": "public Hugging Face download; 13.5 GiB archive",
    }
    size = dir_size(RAW / "avdd_tcmi")
    if not rep:
        return {**base, "status": "not_acquired", "size_bytes": size, "size_human": human_bytes(size), "validated": False}
    samples = rep.get("samples_by_subset") or {k: v.get("directories") for k, v in (rep.get("samples") or {}).items()}
    total = sum(v for v in samples.values() if isinstance(v, int)) or None
    labels = rep.get("labels") or {}
    remote = rep.get("status") == "verified_remote"
    uncompressed = rep.get("uncompressed_bytes")
    return {
        **base,
        "status": "verified_remote_not_downloaded" if remote else "available",
        # 1,230 samples with no participant metadata in the release: a sample
        # count must not be reported as a participant count.
        "participants": None,
        "participant_count_note": "not stated by the release; 1,230 samples are the unit of observation",
        "sessions": total,
        "records": total,
        "text_samples": 0,
        "audio_recordings": 0,
        "visual_samples": rep.get("tcm_region_images"),
        "size_bytes": size,
        "size_human": (f"{human_bytes(size)} stored locally; {human_bytes(uncompressed)} uncompressed "
                       f"in a {human_bytes(rep.get('archive_bytes') or 0)} archive") if remote and uncompressed
                      else human_bytes(size),
        "validated": bool(rep.get("ok")),
        "validation_report": "data/reports/avdd_tcmi_validation.json",
        "labels": {k: v.get("label_distribution") for k, v in labels.items()},
        "label_mapping_verified": True,
    }


def build_pending(name: str, report_name: str, category_key: str) -> dict:
    rep = read_json(REPORTS / report_name) or {}
    req = rep.get("access_requirements")
    if isinstance(req, list):
        req = "; ".join(str(x) for x in req)
    return {
        "dataset_name": name,
        "category": CATEGORY[category_key],
        "status": rep.get("status", "access_pending"),
        "source_url": rep.get("official_source") or rep.get("source_url"),
        "license": rep.get("license"),
        "language": rep.get("language"),
        "participants": rep.get("participants_reported"),
        "sessions": rep.get("sessions_reported"),
        "records": None,
        "text_samples": None,
        "audio_recordings": None,
        "visual_samples": None,
        "modalities": rep.get("modalities"),
        "data_form": rep.get("data_form"),
        "size_bytes": None,
        "size_human": "unknown (controlled access)",
        "validated": False,
        "validation_report": None,
        "access_requirement": req,
        "label_mapping_verified": False,
    }


def main() -> int:
    log = get_logger("generate_dataset_inventory")
    rows = [
        build_eatd(),
        build_harvard(),
        build_avdd(),
        build_pending("daic_woz", "daic_access_report.json", "daic_woz"),
        build_pending("rsdd", "rsdd_access_report.json", "rsdd"),
        build_pending("smhd", "smhd_access_report.json", "smhd"),
    ]
    for r in rows:
        for c in COLUMNS:
            r.setdefault(c, None)
    write_csv(OUT_CSV, rows, COLUMNS)
    payload = {
        "generated_at": utcnow(),
        "dataset_count": len(rows),
        "available": [r["dataset_name"] for r in rows if r["status"] == "available"],
        "access_pending": [r["dataset_name"] for r in rows if r["status"] != "available"],
        "total_bytes": sum(r["size_bytes"] or 0 for r in rows),
        "datasets": rows,
    }
    write_json(OUT_JSON, payload)
    write_json(OUT_REPORT_JSON, payload)
    for r in rows:
        log.info("%-14s %-20s participants=%s records=%s size=%s",
                 r["dataset_name"], r["status"], r["participants"], r["records"], r["size_human"])
    log.info("inventory: %d datasets, %s on disk -> %s", len(rows), human_bytes(payload["total_bytes"]), OUT_CSV)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
