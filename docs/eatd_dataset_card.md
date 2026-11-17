# EATD Dataset Card — Verified Local Copy

**Status:** Phase 1 audit complete. Source files unchanged. Derived artifacts are
under `data/processed/eatd/`.

## 1. What is in the repository

```
data/eatd/
├── train/<id>/          83 participant folders
└── validation/<id>/     79 participant folders
     ├── label.txt        Zung SDS raw total (20–80)
     ├── new_label.txt    Zung SDS index = raw × 1.25 (25–100)
     ├── positive.txt / neutral.txt / negative.txt   prompted transcripts
     └── positive.wav  / neutral.wav  / negative.wav  canonical audio (16 kHz)
        positive_out.wav / ...                        processed variants (derived)
```

- **162 participants** (83 + 79), **486 prompted sessions** (3 per participant).
- Modalities: **text + audio**. There is **no face/video** in this corpus.

## 2. Participant identity (verified, not assumed)

Numeric folder ids are **partition-local**, not global. 58 ids appear in both
`train/` and `validation/`, giving 104 distinct numeric ids but 162 distinct
participants. Evidence that they are different people:

| Check | Result |
|---|---|
| Shared audio SHA-256 across same-id folders | **0 / 58** |
| Shared transcript text across same-id folders | **0 / 58** |
| Identical SDS index across same-id folders | 2 / 58 (coincidental) |
| Same-id duration examples | id 2: 4.09 s vs 8.46 s; id 5: 2.38 s vs 15.90 s |
| Global duplicate audio content | **0 / 486** |
| Global duplicate transcripts | **0 / 486** |

**Participant key:** `(source_partition, source_subject_id)`, written as
`eatd_train_107`, `eatd_validation_107`.

The legacy split (`data/processed/eatd_splits/`) assumed numeric ids were global
and dropped 23 "contradictory" ids (46 participant folders), leaving 81. That
split is leakage-free but unnecessarily conservative. The new manifest keeps all
162 verified participants.

## 3. Label definition

- Instrument: **Zung Self-rating Depression Scale (SDS)** — a 20-item self-report
  symptom scale, **not** a clinical diagnosis.
- `label.txt` = raw total (observed 20–66 locally; theoretical 20–80).
- `new_label.txt` = SDS index = raw total × 1.25 (verified for all 162;
  observed 25.0–82.5).
- **Binary label:** `1` if SDS index ≥ 50, else `0`.
- This rule reproduces `data/eatd_text.csv` for **162 / 162** participants with
  **zero mismatches**. Thresholds 49 and 50 both reproduce it; the clean
  separation is class 0 max = 48.75, class 1 min = 50.0.
- Local class distribution: **109 control / 53 depressed**. Published EATD
  descriptions report 108 / 54; the one-participant difference per class is
  unexplained in the local copy and is recorded rather than "corrected".

**Do not** mix SDS labels with PHQ-8, PHQ-9, clinical diagnoses, or
social-media-derived labels. They are different instruments with different
thresholds and semantics.

## 4. Text/audio alignment

Alignment is at **participant × prompted topic** level: each `topic.txt` is the
transcript of the same-topic `topic.wav`. There are no word-level timestamps.

Verified:

- All **486** legacy CSV texts are byte-identical to the raw `topic.txt` files.
- Every participant has all three transcripts and all three canonical WAVs.
- All canonical WAVs are 16 kHz; channels are 1 or 2.
- `_out.wav` is a **processed variant**, not a fourth sample. It is shorter than
  the canonical WAV for 228/486 sessions (trimming), so it must not be counted
  separately or paired with a different transcript.

## 5. Derived artifacts

| File | Contents |
|---|---|
| `data/processed/eatd/manifest.csv` | one row per participant×topic, with paths, hashes, durations, SDS, label |
| `data/processed/eatd/participants.csv` | one row per participant |
| `data/processed/eatd/splits.csv` | participant-level train/validation/test assignment |
| `data/processed/eatd/validation_report.json` | integrity checks, statistics, label definition |
| `data/processed/eatd/split_report.json` | split counts, leakage checks, legacy comparison |

## 5b. Audio quality flags

- **485 / 486** sessions are usable for audio modelling (`audio_usable = True`).
- **1 session** is flagged: `validation/79/positive.wav` is **0.48 s**
  (`short_audio_lt_1000ms`). It is retained in the manifest because its text is
  valid, but audio training must filter on `audio_usable == True`.
- 12 sessions are shorter than 2 s (`short_audio_lt_2s` flag) but remain usable.
- Quality flags never change labels or split assignment, and are warnings, not
  integrity errors. The report status is `pass_with_quality_flags`.

Regenerate (idempotent for a fixed seed):

```bash
python scripts/build_eatd_manifest.py --seed 42
```

The script exits non-zero if any integrity check fails. It never writes into
`data/eatd/`.

## 6. Splits

- Stratified by participant label, seed 42, ~70/15/15:
  **train 114 / validation 24 / test 24** participants
  (342 / 72 / 72 sessions; class counts per split are preserved).
- **Zero participant overlap** between train/validation/test; every participant
  assigned exactly once; all three splits contain both classes.
- All sessions of a participant stay in that participant's split.

## 7. Limitations

- Small: 162 participants, 486 sessions; confidence intervals will be wide.
- SDS self-report, not clinical diagnosis.
- Chinese-language interviews with three prompted topics; topic/prompt confound.
- Three sessions per participant are not independent samples.
- No face/video, so this corpus supports text+voice research only.
- Local-copy licence/redistribution terms must be confirmed before publication.
