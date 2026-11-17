"""Generate data/manifests/dataset_compatibility.csv.

Every cell is a deliberate research judgement recorded with its evidence; the
compatibility flags are conservative. A dataset is only marked fusion-eligible
when participant/session alignment *and* label comparability were verified.

Usage:
  .venv/Scripts/python scripts/datasets/generate_dataset_compatibility.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EXIT_OK, MANIFESTS, get_logger, utcnow, write_csv  # noqa: E402

OUT = MANIFESTS / "dataset_compatibility.csv"

COLUMNS = [
    "dataset_name", "language", "participant_ids_available", "text_available",
    "audio_available", "raw_audio_available", "visual_available",
    "raw_video_available", "facial_images_available",
    "preextracted_features_available", "depression_labels_available",
    "label_definition", "questionnaire_type", "paired_modalities",
    "compatible_with_eatd", "compatible_with_daic", "compatible_with_avdd",
    "fusion_eligible", "limitations",
]

ROWS = [
    {
        "dataset_name": "eatd",
        "language": "zh",
        "participant_ids_available": "yes - (source_partition, subject_id); 162 participants",
        "text_available": "yes - raw transcripts, 486 sessions",
        "audio_available": "yes - 486 recordings (485 usable)",
        "raw_audio_available": "yes - 16 kHz stereo WAV",
        "visual_available": "no",
        "raw_video_available": "no",
        "facial_images_available": "no",
        "preextracted_features_available": "no (Phase 2 features are derived locally)",
        "depression_labels_available": "yes - binary SDS band at participant level",
        "label_definition": "label = 1 if sds_index >= 50 else 0, sds_index = sds_raw * 1.25 (Zung SDS self-rating)",
        "questionnaire_type": "SDS (self-reported, 3 prompted topics)",
        "paired_modalities": "text + audio (same participant and topic, verified aligned)",
        "compatible_with_eatd": "same dataset",
        "compatible_with_daic": "no - different instrument (SDS vs PHQ-8), language, and recording protocol",
        "compatible_with_avdd": "no - feature-only modalities and different label provenance",
        "fusion_eligible": "yes, within EATD only (text+audio; participant-level splits verified)",
        "limitations": "24 test participants (8 positive); SDS is a screening band, not a diagnosis; licence unclear",
    },
    {
        "dataset_name": "harvard_tweets",
        "language": "en",
        "participant_ids_available": "no - verified: the workbook has only text and label columns (plus a side legend) and no user, tweet or session id, so records cannot be grouped by author",
        "text_available": "yes - 99,590 rows of pre-processed tweet text",
        "audio_available": "no",
        "raw_audio_available": "no",
        "visual_available": "no",
        "raw_video_available": "no",
        "facial_images_available": "no",
        "preextracted_features_available": "no",
        "depression_labels_available": "yes - weak tweet-level binary label; numeric code -> class mapping is ambiguous in the released file",
        "label_definition": "depositor-assigned tweet-level 'depressed' vs 'non-depressed'; no annotation protocol, annotator count or agreement statistic is published. The sheet legend lists Depressed=58,405 / Non-depressed=41,185 but never binds them to codes 0/1, and the counts match code 0 while a depression lexicon matches code 1 (66.0% vs 18.5%)",
        "questionnaire_type": "none",
        "paired_modalities": "none",
        "compatible_with_eatd": "partial - text-only external comparison only; different language, label provenance and text preprocessing",
        "compatible_with_daic": "no - tweet-level weak labels vs interview PHQ-8",
        "compatible_with_avdd": "no - text only",
        "fusion_eligible": "no - single modality and no participant identifiers, so participant-disjoint evaluation cannot be constructed",
        "limitations": "labels are weak social-media proxies, not diagnoses; code-to-class mapping unverified; text is stopword-stripped with 'xexx' apostrophe placeholders; no user id => possible author overlap across any split; no tweet id or timestamp; CC0 applies to the depositor's file, not to Twitter's terms on the underlying content",
    },
    {
        "dataset_name": "avdd_tcmi",
        "language": "zh",
        "participant_ids_available": "yes - verified: 1,230 per-sample directory ids (HDS 0-281, SDS 0-947) with a label row for every sample; ids are only unique within a subset",
        "text_available": "no",
        "audio_available": "features only - 1,230 acoustic CSVs, 1,024 numeric columns each, headerless (per-frame embeddings, ~20 MB per sample)",
        "raw_audio_available": "no - verified: the archive contains no .wav/.mp3/.flac members",
        "visual_available": "features only - 1,230 visual CSVs with 538 named columns (OpenFace-style: frame, face_id, timestamp, confidence, success, gaze, eye landmarks, AU intensities) plus 25,830 TCM facial-region jpgs",
        "raw_video_available": "no - verified: the archive contains no .mp4/.avi/.mov members",
        "facial_images_available": "yes - 25,830 TCM facial inspection region images (289 MB), 21 per sample",
        "preextracted_features_available": "yes - acoustic (1024-d) and visual (538-column frame-level) feature CSVs, archive 13.5 GiB / 34.0 GB uncompressed",
        "depression_labels_available": "yes - verified binary per sample: HDS 159 non-depressed / 123 depressed, SDS 810 / 138, totals 969 / 261",
        "label_definition": "header-less id,label,gender CSV per subset with label 0=non-depressed, 1=depressed as stated on the dataset card; HDS labels come from clinically diagnosed patients vs volunteers, SDS labels from a student screening. The two label provenances are kept separate and never merged",
        "questionnaire_type": "not documented - the card names no screening instrument, cut-off or rater protocol for either subset",
        "paired_modalities": "acoustic + visual features for the same sample id (and 21 facial-region images)",
        "compatible_with_eatd": "no - no raw audio, no text, different language and label provenance",
        "compatible_with_daic": "partial - both audio/visual depression corpora, but AVDD ships features only and DAIC needs approval",
        "compatible_with_avdd": "same dataset",
        "fusion_eligible": "yes within a single subset (feature-level audio+visual, same sample id, labels verified) - never pooled across HDS and SDS",
        "limitations": "no raw recordings and no text; feature extractors are undocumented so representations are not comparable with locally extracted eGeMAPS/OpenFace features; CC-BY-NC-4.0 non-commercial with attribution; HDS and SDS differ in label provenance and 282 vs 948 sample counts; the 13.5 GiB archive is stored but not extracted, so contents are read directly from the ZIP",
    },
    {
        "dataset_name": "daic_woz",
        "language": "en",
        "participant_ids_available": "yes - per-session participant ids",
        "text_available": "yes - interview transcripts",
        "audio_available": "yes - participant audio",
        "raw_audio_available": "yes",
        "visual_available": "yes - facial features and video",
        "raw_video_available": "yes",
        "facial_images_available": "yes (frames via video)",
        "preextracted_features_available": "yes - facial/audio feature files shipped with the corpus",
        "depression_labels_available": "yes - PHQ-8 based",
        "label_definition": "PHQ-8 total score with the corpus's released label convention; not interchangeable with SDS",
        "questionnaire_type": "PHQ-8",
        "paired_modalities": "text + audio + visual (same session)",
        "compatible_with_eatd": "no - different instrument, language and protocol; only cross-dataset generalisation studies",
        "compatible_with_daic": "same dataset",
        "compatible_with_avdd": "partial - both audio/visual depression corpora, no shared participants or labels",
        "fusion_eligible": "pending access approval - alignment is documented, labels must be re-verified after download",
        "limitations": "controlled access; redistribution prohibited; 189 sessions",
    },
    {
        "dataset_name": "rsdd",
        "language": "en",
        "participant_ids_available": "yes - pseudonymous Reddit user ids with official participant-level splits",
        "text_available": "yes - user post collections",
        "audio_available": "no",
        "raw_audio_available": "no",
        "visual_available": "no",
        "raw_video_available": "no",
        "facial_images_available": "no",
        "preextracted_features_available": "no",
        "depression_labels_available": "yes - user-level self-reported diagnosis vs matched controls",
        "label_definition": "diagnosed user = self-reported depression diagnosis verified by crowdsourcing; control = matched user with no MH posts",
        "questionnaire_type": "none (self-report statements, not a questionnaire)",
        "paired_modalities": "none",
        "compatible_with_eatd": "partial - text-only external comparison; self-report vs SDS bands",
        "compatible_with_daic": "no - no PHQ-8, no interviews",
        "compatible_with_avdd": "no - text only",
        "fusion_eligible": "no - single modality",
        "limitations": "DUA required; no contact/deanonymisation/linking; 116,484 users with 9,210 diagnosed",
    },
    {
        "dataset_name": "smhd",
        "language": "en",
        "participant_ids_available": "yes - pseudonymous Reddit user ids with official train/dev/test splits",
        "text_available": "yes - user post collections",
        "audio_available": "no",
        "raw_audio_available": "no",
        "visual_available": "no",
        "raw_video_available": "no",
        "facial_images_available": "no",
        "preextracted_features_available": "no",
        "depression_labels_available": "yes - user-level self-reported diagnosis for nine conditions plus controls",
        "label_definition": "diagnosed user = self-reported diagnosis of one of nine conditions; depression compared one-vs-control",
        "questionnaire_type": "none (self-report statements, not a questionnaire)",
        "paired_modalities": "none",
        "compatible_with_eatd": "partial - text-only external comparison",
        "compatible_with_daic": "no",
        "compatible_with_avdd": "no - text only",
        "fusion_eligible": "no - single modality",
        "limitations": "DUA required; nine conditions are not mutually exclusive categories; 385,476 users total",
    },
]


def main() -> int:
    log = get_logger("generate_dataset_compatibility")
    rows = [{c: r.get(c, "") for c in COLUMNS} for r in ROWS]
    write_csv(OUT, rows, COLUMNS)
    for r in rows:
        log.info("%-14s fusion=%-8s paired=%s", r["dataset_name"],
                 "yes" if str(r["fusion_eligible"]).startswith("yes") else "no", r["paired_modalities"])
    log.info("compatibility matrix (%d datasets, generated %s) -> %s", len(rows), utcnow(), OUT)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
