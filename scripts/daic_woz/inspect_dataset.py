#!/usr/bin/env python3
"""
DAIC-WOZ Dataset Inspector, Manifest Builder, and Quality Auditor
NeuroScreen Major Project - Phase 2 Dataset Inspection

Produces:
    - data/manifests/daic_woz/dataset_inventory.json
    - data/manifests/daic_woz/participant_manifest.csv
    - data/manifests/daic_woz/phq8_label_audit.json
    - results/daic_woz/reports/dataset_quality_report.md
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


def inspect_dataset(raw_dir: Path):
    manifest_dir = Path("data/manifests/daic_woz")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("results/daic_woz/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate participant directories
    participant_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir() and (d.name.endswith("_P") or d.name.isdigit())
    ])

    # 2. Search for label CSVs (AVEC 2016 / 2017 or raw label files)
    label_files = list(raw_dir.glob("*split*Depression*.csv")) + list(raw_dir.glob("*_labels*.csv")) + list(raw_dir.glob("all_labels*.csv"))
    labels_dict = {}

    for lf in label_files:
        try:
            df_labels = pd.read_csv(lf)
            id_col = None
            for c in ["Participant_ID", "participant_id", "ID", "id"]:
                if c in df_labels.columns:
                    id_col = c
                    break
            phq_col = None
            for c in ["PHQ8_Score", "phq8_score", "PHQ_Score", "Score"]:
                if c in df_labels.columns:
                    phq_col = c
                    break
            bin_col = None
            for c in ["PHQ8_Binary", "phq8_binary", "Binary", "label"]:
                if c in df_labels.columns:
                    bin_col = c
                    break

            if id_col:
                for _, row in df_labels.iterrows():
                    pid = int(row[id_col])
                    score = float(row[phq_col]) if phq_col and pd.notna(row[phq_col]) else None
                    binary = int(row[bin_col]) if bin_col and pd.notna(row[bin_col]) else (
                        (1 if score >= 10 else 0) if score is not None else None
                    )
                    labels_dict[pid] = {
                        "score": score,
                        "binary": binary,
                        "source_file": lf.name
                    }
        except Exception as e:
            print(f"[-] Warning: Failed parsing label file {lf.name}: {e}")

    # 3. Analyze each participant directory
    participants_manifest = []
    modality_counts = {"text": 0, "audio": 0, "visual": 0, "complete_triad": 0}

    for pdir in participant_dirs:
        pid_str = pdir.name.replace("_P", "")
        try:
            pid = int(pid_str)
        except ValueError:
            pid = pid_str

        # Look for transcripts
        transcript_files = list(pdir.glob("*TRANSCRIPT*.csv")) + list(pdir.glob("*transcript*.csv"))
        transcript_path = transcript_files[0] if transcript_files else None

        # Look for audio
        audio_files = list(pdir.glob("*AUDIO*.wav")) + list(pdir.glob("*audio*.wav")) + list(pdir.glob("*.wav"))
        audio_path = audio_files[0] if audio_files else None

        # Look for visual / OpenFace features
        visual_files = (
            list(pdir.glob("*CLNF_features*.txt")) +
            list(pdir.glob("*OpenFace*.csv")) +
            list(pdir.glob("*AUs*.csv")) +
            list(pdir.glob("*features*.csv"))
        )
        visual_path = visual_files[0] if visual_files else None

        has_text = transcript_path is not None and transcript_path.stat().st_size > 0
        has_audio = audio_path is not None and audio_path.stat().st_size > 44  # WAV header is 44 bytes
        has_visual = visual_path is not None and visual_path.stat().st_size > 0

        if has_text: modality_counts["text"] += 1
        if has_audio: modality_counts["audio"] += 1
        if has_visual: modality_counts["visual"] += 1
        if has_text and has_audio and has_visual: modality_counts["complete_triad"] += 1

        label_info = labels_dict.get(pid, {"score": None, "binary": None, "source_file": None})

        participants_manifest.append({
            "participant_id": pid,
            "session_id": f"{pid}_P",
            "folder_path": str(pdir.relative_to(raw_dir.parent.parent)),
            "transcript_path": str(transcript_path.relative_to(raw_dir.parent.parent)) if transcript_path else None,
            "audio_path": str(audio_path.relative_to(raw_dir.parent.parent)) if audio_path else None,
            "visual_path": str(visual_path.relative_to(raw_dir.parent.parent)) if visual_path else None,
            "has_text": has_text,
            "has_audio": has_audio,
            "has_visual": has_visual,
            "is_complete_triad": has_text and has_audio and has_visual,
            "phq8_score": label_info["score"],
            "phq8_binary": label_info["binary"],
            "label_source": label_info["source_file"]
        })

    # Save participant manifest CSV
    manifest_csv = manifest_dir / "participant_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_id", "session_id", "folder_path", "transcript_path",
            "audio_path", "visual_path", "has_text", "has_audio", "has_visual",
            "is_complete_triad", "phq8_score", "phq8_binary", "label_source"
        ])
        writer.writeheader()
        writer.writerows(participants_manifest)

    # 4. Audit PHQ-8 Labels
    scores = [p["phq8_score"] for p in participants_manifest if p["phq8_score"] is not None]
    binaries = [p["phq8_binary"] for p in participants_manifest if p["phq8_binary"] is not None]

    phq8_audit = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "total_participants_inspected": len(participants_manifest),
        "participants_with_scores": len(scores),
        "participants_with_binary": len(binaries),
        "binary_label_definition": "Official clinical threshold: PHQ-8 >= 10 indicates Moderate or Severe Depression (1), else 0",
        "class_distribution": {
            "depressed_count_1": binaries.count(1),
            "control_count_0": binaries.count(0),
            "depressed_prevalence": round(binaries.count(1) / len(binaries), 4) if binaries else None
        },
        "score_statistics": {
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "mean_score": round(sum(scores) / len(scores), 2) if scores else None
        },
        "unlabelled_participants": [p["participant_id"] for p in participants_manifest if p["phq8_binary"] is None]
    }

    phq8_audit_path = manifest_dir / "phq8_label_audit.json"
    with open(phq8_audit_path, "w", encoding="utf-8") as f:
        json.dump(phq8_audit, f, indent=2)

    # 5. Inventory JSON
    inventory = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "total_participant_folders": len(participant_dirs),
        "modality_breakdown": modality_counts,
        "eligible_triad_participants": modality_counts["complete_triad"],
        "labels_found": len(labels_dict),
        "inspection_status": "DATA_AUDITED" if len(participants_manifest) > 0 else "NO_RECORDS_ON_DISK"
    }

    inventory_json = manifest_dir / "dataset_inventory.json"
    with open(inventory_json, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    # 6. Markdown Quality Report
    report_md = report_dir / "dataset_quality_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# DAIC-WOZ Dataset Inspection and Quality Audit Report\n\n")
        f.write(f"**Audit Date:** {inventory['audited_at']}\n")
        f.write(f"**Inspection Status:** `{inventory['inspection_status']}`\n\n")
        f.write("## 1. Modality and Participant Census\n\n")
        f.write(f"- Total Participant Sessions Located: `{len(participant_dirs)}`\n")
        f.write(f"- Text Transcripts Available: `{modality_counts['text']}`\n")
        f.write(f"- Vocal Audio Recordings Available: `{modality_counts['audio']}`\n")
        f.write(f"- Visual / Facial Features Available: `{modality_counts['visual']}`\n")
        f.write(f"- Complete Paired Triads (Text + Audio + Visual): `{modality_counts['complete_triad']}`\n\n")
        f.write("## 2. PHQ-8 Label Distribution & Clinical Cut-offs\n\n")
        f.write("- **Primary Assessment Instrument:** Patient Health Questionnaire 8-item (PHQ-8, range 0–24)\n")
        f.write("- **Official Cut-off Criterion:** $\\text{PHQ-8} \\ge 10 \\rightarrow 1$ (Moderate to Severe Depressive Symptoms); $< 10 \\rightarrow 0$\n")
        f.write(f"- **Depressed Participants ($y=1$):** {phq8_audit['class_distribution']['depressed_count_1']}\n")
        f.write(f"- **Control Participants ($y=0$):** {phq8_audit['class_distribution']['control_count_0']}\n\n")
        if inventory['inspection_status'] == "NO_RECORDS_ON_DISK":
            f.write("> [!IMPORTANT]\n")
            f.write("> **Dataset Status: BLOCKED PENDING OFFICIAL ACCESS**\n")
            f.write("> Raw DAIC-WOZ archives have not yet been downloaded to `data/raw/daic_woz/`.\n")
            f.write("> In accordance with non-fabrication research directives, 0 synthetic records were generated.\n")

    print(f"[+] Dataset inspection complete:")
    print(f"    - Manifest: {manifest_csv}")
    print(f"    - PHQ-8 Audit: {phq8_audit_path}")
    print(f"    - Inventory: {inventory_json}")
    print(f"    - Report: {report_md}")


if __name__ == "__main__":
    raw_dir = Path("data/raw/daic_woz")
    inspect_dataset(raw_dir)
