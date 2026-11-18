# Phase 4 — Dataset Acquisition, Validation and External Generalization Preparation

Date: 2026-11-13 (AVDD-TCMI download, archive validation and registry refresh
completed 2026-11-14)
Scope: acquire, inspect, validate and organise the shortlisted research
datasets. **No model was trained, no fusion was attempted, no backend/frontend
code was touched, no EATD file or split was modified, and no checkpoint was
replaced.** Controlled-access datasets were not requested: no form was
submitted, no agreement accepted, no credential created.

Statements below are tagged **[verified]** (checked directly against the
official record or the local files during this phase), **[reported]** (claimed
by the dataset's own documentation, not yet reproducible locally because the
data is absent), or **[assumption]** (explicitly not verified).

### Execution history and recovery (2026-11-14)

The first AVDD-TCMI download attempt (09:50–10:27) reached **99.0 % (13.4 GiB)**
and then lost its partial file; a second attempt (10:27–10:30) died at 3.6 %
with `MemoryError`. The cause was `requests` advertising
`Accept-Encoding: gzip, deflate, zstd` on a 13.5 GiB binary stream, which pushes
the response through urllib3's content decoder. The download path now requests
`identity` encoding and reads the raw stream with `decode_content=False`.

The surviving partial (886,046,720 B) was verified byte-for-byte against the
remote at six sampled offsets (0, head, middle and three random windows) before
resuming, so no bytes were downloaded twice and no corrupt prefix was appended.
The resume completed at 11:54 with exit code 0. Full evidence in
`data/logs/download_avdd_tcmi.log`.

---

## 1. Environment and Phase A audit

| Item | Result |
|---|---|
| Project root | Repository root **[verified]** |
| Interpreter | `.venv/Scripts/python` — **Python 3.14.0**, torch 2.13.0+cu130, CUDA available **[verified]** |
| Requested interpreter | The brief says Python 3.12. The verified virtualenv is 3.14.0; the CUDA environment was **not** replaced. **[verified discrepancy]** |
| `datasets` package | Not installed; `huggingface_hub` 1.28.0 is, so HF files are fetched via the hub API without adding a package **[verified]** |
| Disk | 378 GB volume; 21.7 GiB free at audit time, 46.0 GiB before the download, **32.0 GiB free now** (Windows reclaimed space after the archive landed) **[verified]** |
| Existing data | `data/eatd` 740.9 MB (1,806 files), `data/fer2013` 24.4 MB, `data/processed/eatd` 247 KB **[verified]** |
| Phase 2 artifacts | 6 result files, 5 model dirs, all intact **[verified]** |

Audit artifacts: `data/manifests/existing_datasets_audit.json`, log in
`data/logs/audit_existing_datasets.log`.

## 2. Dataset roles (kept strictly separate)

| Category | Datasets | Meaning |
|---|---|---|
| 1 — depression-related **text** | Harvard tweets, RSDD, SMHD, EATD text, DAIC-WOZ transcripts | text-only studies |
| 2 — depression-related **audio** | EATD audio, DAIC-WOZ audio, AVDD-TCMI acoustic features | voice studies |
| 3 — depression-related **visual** | DAIC-WOZ visual, AVDD-TCMI visual features + TCM facial images | visual studies |
| 4 — **auxiliary representation** | (none acquired yet; `data/fer2013` is an existing local emotion set, not a depression dataset) | pretraining/aux only |
| 5 — **genuinely paired multimodal** | EATD (text+audio, same participant/topic), AVDD-TCMI (audio+visual features per sample), DAIC-WOZ (text+audio+visual, pending access) | fusion candidates |

No dataset is combined across categories, and no label set is pooled. Text,
audio and visual results from different corpora must be reported as separate
external-generalisation studies.

## 3. Official sources and verification status

| Dataset | Official source | Licence | Verified on |
|---|---|---|---|
| Harvard English depression tweets | `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ISZCSA` | CC0 1.0 **[verified]** | Dataverse API, dataset v1.1 |
| AVDD-TCMI | `https://huggingface.co/datasets/edcdd/AVDD-TCMI-dataset` | CC-BY-NC-4.0 **[verified]** | HF API + dataset card |
| DAIC-WOZ | `https://dcapswoz.ict.usc.edu/` | controlled access **[verified]** | official site |
| RSDD | `https://ir.cs.georgetown.edu/resources/rsdd` (now `https://georgetown-ir-lab.github.io/emnlp17-depression`) | research DUA **[verified]** | official pages |
| SMHD | `https://ir.cs.georgetown.edu/resources/smhd.html` | research DUA **[verified]** | official pages |
| EATD | `https://github.com/Fancy-Block/EATD-Corpus` | not stated locally **[verified absence]** | local copy + Phase 1 manifest |

## 4. Download status

| Dataset | Status | Detail |
|---|---|---|
| EATD (F) | **AVAILABLE, VALIDATED** | already local; not re-downloaded |
| Harvard tweets (A) | **AVAILABLE, VALIDATED** | 1 file, `BinaryLabeled_Depression_English_99590.xlsx`, 4,678,835 B (4.5 MB), CC0, downloaded from the Dataverse file API; MD5 `c925e2fb68d49bd1a1003a79f58d8c10` matches the Dataverse record, SHA-256 `e566fa961cce15a0fc7432aa5c47c6dd3d023ac0bbd7552999465c5c5f95f9ba` |
| AVDD-TCMI (B) | **DOWNLOADED AND VALIDATED (archive only, not extracted)** | `data/raw/avdd_tcmi/AVDD-TCMI-dataset.zip`, **14,529,499,485 B (13.5 GiB)**, exactly the declared size; SHA-256 `85e93fe1…` identical to the Hub LFS oid; `zipfile.testzip` clean over all 34,445 members; 1,230 samples re-confirmed **from the local archive**, not from the record. Extraction is deferred (the full archive expands to 36,472,587,246 B = 33.97 GiB; 32.0 GiB free) — see §14 and §16 |
| DAIC-WOZ (C) | **ACCESS_PENDING** | application form + signed agreement, academic e-mail only |
| RSDD (D) | **ACCESS_PENDING** | signed DUA, request form |
| SMHD (E) | **ACCESS_PENDING** | signed DUA, request form |

Reports: `data/reports/harvard_tweets_download.json`,
`data/reports/harvard_tweets_validation.json`,
`data/reports/avdd_tcmi_download.json`,
`data/reports/avdd_tcmi_remote_inspection.json`,
`data/reports/avdd_tcmi_validation.json`.

Disk at time of writing: 378 GB volume, **32.0 GiB free** (46.0 GiB before the
archive was stored). The archive expands to **33.97 GiB** (36,472,587,246 B) and
the feature CSVs alone to **33.70 GiB** (36,183,442,074 B) — the 25,830 jpgs are
only 0.27 GiB of it — so even a CSV-only extraction does not fit. The archive is
kept un-extracted and read directly from the ZIP at ~250 MB/s; extraction remains
the owner's decision (§16).

Size convention: sizes are shown in GiB, labelled `GB` by the helper that prints
them (`scripts/datasets/_common.py::human_bytes`). Exact byte counts are given
wherever a decision depends on them.

## 5. Verified statistics

### EATD (validated locally, 486/486 text and audio hashes match)

| Quantity | Value |
|---|---|
| Participants | 162 |
| Sessions (3 prompted topics each) | 486 |
| Audio-usable sessions | 485 (one 0.48 s recording) |
| Participants by label | 109 control / 53 depressed **[from Phase 1, re-derived here]** |
| Sessions by label | 327 label 0 / 159 label 1 |
| Splits (participants) | 114 train / 24 validation / 24 test |
| Transcript language | Chinese (CJK characters present; ASCII letters negligible) |
| Text / audio files | 486 `.txt`, 486 `.wav` (+486 processed `_out.wav`) |
| Size | 776,891,936 B (740.9 MB) |

### Harvard tweets — **[verified locally]**

| Quantity | Value |
|---|---|
| Rows | **99,590** (matches the depositor's claim exactly) |
| Code 0 | 58,405 rows |
| Code 1 | 41,185 rows |
| Empty text | 0 |
| Exact duplicate texts | 0 |
| Duplicates after whitespace/case normalisation | 264 |
| Columns | `text`, `label`, plus an unlabelled side column holding a class legend and one padding column |
| User / tweet / session identifiers | **none** (no id, user, author, screen_name or timestamp column) |
| URL / @mention / e-mail / phone-like strings | 0 / 0 / 0 / 0 |
| Language | ASCII English throughout (Latin letters only, 0 CJK characters) |
| Text length | min 4, p50 54, p90 158, mean 71.5, max 566 characters |

Two findings that change how this dataset may be used **[verified]**:

1. **The label code → class mapping is ambiguous.** The dataset description and
the in-sheet legend both state *Depressed = 58,405* and *Non-depressed =
41,185*, but neither binds those classes to the numeric codes. Positionally,
58,405 rows carry code **0** and 41,185 carry code **1**, which would make
0 = depressed. A depression lexicon applied to the text points the other way:
**66.0 %** of code-1 rows contain a depression term versus **18.5 %** of
code-0 rows, which would make 1 = depressed. The original codes are preserved
unchanged; no relabelling was performed, and this ambiguity must be resolved
with the depositor (or by reproducing the depositor's pipeline) before any
binary metric is reported.
2. **The text is pre-processed, not raw tweet text.** Mean 10.1 tokens, 54 % of
rows contain no common stopword, URLs/@mentions/emoji are absent, and 22 % of
rows contain `xexx` where an apostrophe belongs. It behaves like a
stopword-stripped keyword bag, so results are not comparable with results on
raw EATD transcripts.

### AVDD-TCMI — **[verified locally from the downloaded archive]**

| Quantity | Verified | Dataset card |
|---|---|---|
| Total valid samples | **1,230** | 1,230 |
| Non-depressed / depressed | **969 / 261** | 969 / 261 |
| HDS (West China Hospital) | **282** = 159 non-depressed / 123 depressed | 282 |
| SDS (middle school) | **948** = 810 non-depressed / 138 depressed | 948 |
| Sample directories | 1,230 (HDS 282 + SDS 948) | — |
| Label rows | 1,230, every sample directory labelled, 0 duplicates, 0 orphans | — |
| Acoustic feature CSVs | 1,230 — **1,024 numeric columns, headerless** | "audio_features" |
| Visual feature CSVs | 1,230 — **538 named columns** (frame, face_id, timestamp, confidence, success, gaze_*, eye landmarks, AU intensities) | "video_features" |
| TCM facial-region images | **25,830** `.jpg` (21 per sample, 289 MB) | regional features |
| Raw audio files | **0** | — |
| Raw video files | **0** | — |
| Archive / uncompressed size | 14,529,499,485 B / 34,472,587,246 B | ≈14.5 GB |
| Official train/val/test split | none | none |
| Gender field | `f`/`m` present in the label files | documented |

Label files are **header-less** `id,label,gender` CSVs; the card's documented
`0 = non-depressed, 1 = depressed` mapping is consistent with the observed
counts. No screening instrument, cut-off or rater protocol is documented — the
card only says HDS depressed participants were clinically diagnosed in hospital
and SDS depressed students were identified in a school setting. Archive last
modified 2026-11-07.

Per-sample files are large — sample HDS/0 has a 20.06 MB acoustic CSV and a
4.64 MB visual CSV — i.e. these are **frame-level sequences**, not one row of
summary statistics per participant. That is what makes full extraction tight on
this disk.

### DAIC-WOZ — **[reported]**

189 interview sessions, 7–33 minutes (mean ≈16), with transcripts, participant
audio, facial features and questionnaire responses.

### RSDD — **[reported]**

9,210 diagnosed users + 107,274 controls = **116,484 users**; official splits:
training 3,070/35,753, validation 3,070/35,746, test 3,070/35,775.

### SMHD — **[reported]**

385,476 users in total (incl. 335,952 controls). Depression: 14,139 users /
1,272K posts, split 2,662 train / 2,574 dev / 2,611 test / 6,292 RC. Other
conditions: ADHD 10,098; anxiety 8,783; autism 2,911; bipolar 6,434; eating 598;
OCD 2,336; PTSD 2,894; schizophrenia 1,331.

### Record counts are not participant counts

| Dataset | Unit | Participants/users | Records |
|---|---|---|---|
| EATD | session | 162 | 486 sessions (486 text, 486 audio) |
| Harvard | tweet | **unknowable from the file** — no author id exists | 99,590 tweets **[verified]** |
| AVDD | sample | 1,230 sample ids **[verified]**, split across two cohorts | 1,230 samples (1,230 acoustic + 1,230 visual CSVs, 25,830 images) |
| DAIC | session | 189 | 189 interviews |
| RSDD | user | 116,484 | millions of posts (not counted here) |
| SMHD | user | 385,476 | 116M control posts + condition posts |

Where an identifier is unverified, no participant count is claimed.

## 6. Label definitions (preserved, never translated)

| Dataset | Positive label | Source | Weak/proxy/validated |
|---|---|---|---|
| EATD | `sds_index = sds_raw × 1.25 ≥ 50` | Zung SDS self-rating, 3 prompted topics | **weak screening band**, not a diagnosis |
| Harvard tweets | "depressed" tweet classification | depositor's keyword/label methodology | **weak**, tweet-level |
| AVDD-TCMI | HDS: clinically diagnosed depressed patients; SDS: depressed students | separate label files per subset | HDS closest to clinical; SDS screening-like; **never merged** |
| DAIC-WOZ | PHQ-8 based | interview questionnaire | validated instrument, but a screening score, **not SDS** |
| RSDD | self-reported diagnosis statement, crowdsource-verified | Reddit posts | **weak**, self-report at user level |
| SMHD | self-reported diagnosis of one of nine conditions | Reddit posts | **weak**, self-report at user level |

Explicitly recorded: EATD SDS ≠ PHQ-8 ≠ AVDD labels ≠ weak social-media
labels ≠ emotion labels. A common binary mapping is **not** applied anywhere.

## 7. Modalities: raw vs pre-extracted

| Dataset | Text | Raw audio | Raw video | Facial images | Pre-extracted features |
|---|---|---|---|---|---|
| EATD | raw `.txt` | raw 16 kHz stereo WAV | no | no | no (Phase 2 features derived locally) |
| Harvard | tweets in `.xlsx` | no | no | no | no |
| AVDD-TCMI | **no** | **no** | **no** | yes (TCM facial-region images) | yes (acoustic + visual CSVs, one per sample) |
| DAIC-WOZ | transcripts | yes | yes | via video | yes (facial/audio feature files) |
| RSDD | posts | no | no | no | no |
| SMHD | posts | no | no | no | no |

Important correction to an assumption in the brief: **AVDD-TCMI does not
provide raw audio or raw video.** It provides per-sample acoustic and visual
feature CSVs plus TCM facial-region images. Any "raw audio-visual" claim would
be false, and the ResNet-18 image pipeline would only see the small
facial-region crops, not full frames.

## 8. Privacy, licence and redistributon restrictions

| Dataset | Restrictions |
|---|---|
| Harvard | CC0 1.0 (public-domain dedication) **[verified]**. The depositor's file is CC0; the underlying tweets remain subject to Twitter/X terms and may contain third-party content. Treat as sensitive, do not publish raw rows. |
| AVDD-TCMI | **CC-BY-NC-4.0**: non-commercial use only, attribution required, no redistribution of the archive. |
| DAIC-WOZ | Academic/non-profit only; **redistribution prohibited**; signed agreement; institutional e-mail required. |
| RSDD | Signed DUA: no contacting users, no deanonymisation, no cross-dataset linking, **no sharing of any excerpt**. |
| SMHD | Signed DUA: same privacy constraints, nine conditions, large volume of sensitive text. |
| EATD | **Licence not stated in the local copy**; redistribution terms flagged for review before any publication of raw data. |

Additionally, no dataset was uploaded to any third-party service in this phase.

## 9. Validation performed

`data/reports/eatd_validation.json` — full local validation:

- 486/486 text hashes and 486/486 audio hashes match the Phase 1 manifest
  (text hash = hash of BOM-stripped, `.strip()`ed UTF-8 transcript; audio hash =
  raw WAV bytes — this distinction is now encoded in the validator).
- 0 missing/empty transcripts, 0 missing/empty audio files.
- 0 duration mismatches against WAV headers, 0 sample-rate/channel mismatches
  (all 16 kHz stereo).
- 0 label-rule violations (`label == (sds_index >= 50)`), 0 SDS arithmetic
  violations (`sds_index == 1.25 × sds_raw`).
- 0 participants with mixed labels or mixed SDS scores.
- All participant keys are dataset-qualified (`eatd_<split>_<id>`), 0 overlaps
  between train/validation/test.
- 1,783 files checksummed into `data/manifests/dataset_checksums.csv`;
  verification re-run is clean.

`data/reports/harvard_tweets_validation.json` — local validation:

- 99,590 rows parsed from the `.xlsx` with a stdlib reader (no `openpyxl` added
  to the CUDA environment); 99,590 with non-empty text; 0 empty rows.
- 0 exact duplicate texts; 264 duplicates after case/whitespace normalisation
  (recorded, **not** removed).
- Label values are exactly {0, 1}; no missing label; no non-numeric code.
- 0 URLs, 0 @mentions, 0 e-mails, 0 phone-like strings in the released text.
- No user/tweet/session identifier column ⇒ participant-independent evaluation
  is impossible from this file (recorded in the report, not papered over).
- Legend, label-code ambiguity and preprocessing artifacts are recorded in the
  report (`legend`, `label_code_mapping`, `preprocessing_artifacts`), with the
  ambiguity raised as a warning. Exit code stays 0 because no *integrity* check
  failed; the ambiguity is an interpretive finding.

`data/reports/avdd_tcmi_validation.json` — **archive validated locally**
(`mode: archive`, status `validated`, 0 issues):

- Archive downloaded: 14,529,499,485 B, **SHA-256
  `85e93fe18a6b8574705a4f716e89fe363e41d3f9b8fb470d4560ad160ca60859`**, identical
  to the Hub LFS oid ⇒ content is byte-identical to the published record.
- `zipfile.testzip()` over the whole archive returned **no bad member** (per-member
  CRC verified), exit code 0.
- 34,445 ZIP members: 28,292 files + 6,153 directories; 34,472,587,246 B
  uncompressed.
- 1,230 sample directories (**HDS 282 + SDS 948**), 1,230 acoustic CSVs, 1,230
  visual CSVs, 25,830 TCM `.jpg`s, 0 `.wav`/`.mp3`/`.flac`, 0 `.mp4`/`.avi`/`.mov`.
- Label files parsed as header-less `id,label,gender`: 1,230 rows, every sample
  directory has exactly one label row, 0 duplicate ids, 0 orphans, 0 label rows
  without a sample; distributions 159/123 and 810/138 match the card exactly.
- Feature CSV headers/shapes read from one sample per subset (1,024 numeric
  columns acoustic; 538 named columns visual).
- `data/reports/avdd_tcmi_remote_inspection.json` is retained as the
  pre-download record; its counts agree with the local archive on every field.
- Two validator defects were found and fixed while validating the real archive:
  the sample-directory regex assumed members were not prefixed with the
  top-level folder (it matched 0 samples), and the label reader used
  `csv.DictReader` on header-less files (it lost one row per subset and reported
  no columns). Both are corrected in `scripts/datasets/validate_avdd_tcmi.py`.
- **Still unverified**: that every one of the 2,462 feature CSVs parses
  row-by-row *from an extracted tree*; this is now covered in place by
  `validate_avdd_features.py`, which parsed all 2,460 CSVs from the archive.
  **Resolved on 2026-11-14** — see §18.

## 10. Required scripts (all idempotent, `--dry-run` where relevant)

| Script | Purpose |
|---|---|
| `scripts/datasets/audit_existing_datasets.py` | read-only repo/data/disk audit |
| `scripts/datasets/download_harvard_tweets.py` | Dataverse API download, Range resume, MD5 check |
| `scripts/datasets/download_avdd_tcmi.py` | HF download, disk preflight, `zipfile.testzip`, optional extract, `--inspect-remote` audit |
| `scripts/datasets/validate_harvard_tweets.py` | stdlib xlsx parsing, labels, duplicates, PII, language |
| `scripts/datasets/validate_avdd_tcmi.py` | archive-mode audit, extracted-tree audit, or `--remote` record validation |
| `scripts/datasets/validate_avdd_features.py` | parses all 2,460 acoustic/visual CSVs in place, per-column statistics, label + modality alignment, generates the feature inspection doc |
| `scripts/datasets/verify_avdd_extraction.py` | selective-extraction integrity: counts, HDS/SDS structure, id mapping, no jpgs, CRC vs archive members, `not_extracted` when absent |
| `scripts/datasets/validate_eatd.py` | manifest/hash/label/split/audio-header validation |
| `scripts/datasets/verify_dataset_checksums.py` | record + verify SHA-256 registry |
| `scripts/datasets/write_access_reports.py` | DAIC/RSDD/SMHD access reports + guide e-mail draft |
| `scripts/datasets/generate_dataset_inventory.py` | `dataset_inventory.{csv,json}` |
| `scripts/datasets/generate_dataset_compatibility.py` | `dataset_compatibility.csv` |
| `scripts/datasets/_common.py` | shared stdlib-only helpers |
| `scripts/datasets/_remote_zip.py` | read a remote ZIP's central directory over HTTP ranges |

Exit codes are non-zero on validation failure (1), refusal (2) or missing data
(4). No script hardcodes secrets and none deletes data.

## 11. Missing, duplicate and invalid data

- **EATD**: 1 recording unusable (0.48 s), flagged in Phase 1 and preserved in
  the manifest as `audio_usable=False`; not silently dropped.
- **Harvard**: 0 empty rows, 0 exact duplicates, **264** near-duplicate groups
  after case/whitespace normalisation (264 extra rows). **0** of those groups
  carry conflicting labels. Nothing was removed.
- **AVDD-TCMI**: 0 missing labels, 0 duplicate sample ids, 0 orphan label rows
  (verified remotely). Per-file corruption cannot be assessed until download.
- No exclusions were applied to any dataset in this phase.

## 12. Participant identifiers and leakage policy

| Dataset | Identifier | Split policy |
|---|---|---|
| EATD | `(source_partition, subject_id)` | existing participant-level splits **preserved unchanged** |
| Harvard | **none verified** — no user, tweet or session id column exists | participant-independent evaluation **cannot be guaranteed**; any split is tweet-level and may place tweets by the same author on both sides. This must be stated in every result produced from this dataset |
| AVDD-TCMI | per-sample directory id within HDS/SDS | split by sample id; never pool HDS with SDS |
| DAIC-WOZ | participant id | participant-level split, created after access |
| RSDD / SMHD | pseudonymous user id | official participant-level partitions preserved |

## 13. Compatibility summary

`data/manifests/dataset_compatibility.csv` records 6 datasets × 19 columns.
Fusion-eligible:

- **EATD** — yes, within itself (text+audio, participant-level splits verified).
- **AVDD-TCMI** — yes within a single subset (audio+visual features, same
  sample id); never across HDS/SDS and never with EATD.
- **DAIC-WOZ** — *pending*: alignment documented, labels must be re-verified
  after access is granted.
- Harvard / RSDD / SMHD — **no**: single modality, no paired data.

No dataset is marked compatible with EATD or DAIC on label grounds: SDS, PHQ-8,
clinical diagnosis and weak social-media labels are different constructs.

## 14. Research limitations of the current collection

1. **Storage-bound**: the AVDD-TCMI archive is stored but **not extracted** —
   extraction needs 33.97 GiB and only 32.0 GiB is free. The data is fully
   readable from the ZIP (validated in place), so this costs I/O on every access
   rather than blocking work; freeing ~2 GiB or a per-subset extraction is the
   realistic next step.
2. **Label heterogeneity**: no two sources share a label instrument, so
   cross-dataset numbers are generalisation probes, not accuracy improvements.
3. **Weak labels dominate**: three of six sources are social-media self-report.
4. **AVDD-TCMI is feature-only**: it cannot validate a raw-audio or raw-video
   pipeline, only feature-level fusion.
5. **DAIC-WOZ, RSDD and SMHD are unavailable** until the owner signs agreements;
   the project currently has exactly one fully paired raw multimodal corpus
   (EATD) with 24 test participants.
6. **EATD licence** is unconfirmed for redistribution.
7. **Python 3.14** is what the CUDA environment actually uses, not 3.12.

## 15. Experiments each dataset enables

| Dataset | Now possible | After access |
|---|---|---|
| EATD | external text validation of the Phase 2 TF-IDF model; voice re-analysis; text+audio fusion within one corpus | — |
| Harvard | large-scale English tweet classification *with a stated label assumption*; cross-source text generalisation (EATD Chinese text → English tweets is **not** a fair transfer test) | — |
| AVDD-TCMI | audio-feature + visual-feature fusion baselines on HDS and SDS separately; facial-region image experiments | — |
| DAIC-WOZ | — | genuine paired text/audio/visual benchmark; PHQ-8 regression/classification |
| RSDD | — | user-level text classification with official participant splits |
| SMHD | — | large-scale user-level text; depression-vs-control stress test |

## 16. What is awaiting the owner

1. **Harvard label mapping**: ask the depositor whether code 0 or code 1 means
   "depressed" (the file, the description and lexical evidence disagree). Until
   answered, any Harvard result must name the assumed mapping.
2. **AVDD-TCMI extraction decision** (archive already downloaded and
   validated): free ~2 GiB for a full CSV extraction to 33.70 GiB, approve a
   per-subset extraction (e.g. one of HDS/SDS, or visual features only at
   9.06 GiB), or keep reading directly from the ZIP. Nothing is blocked either
   way — the CSVs have already been validated in place.
3. **Sign / route the DAIC-WOZ and RSDD/SMHD agreements** (guide e-mail draft is
   in `data/reports/daic_access_report.json`).
4. **Confirm EATD licence terms** before any publication of raw data.

---

## 17. Final summary

### Status counts

| Status | Count | Datasets |
|---|---|---|
| Acquired and validated | **3** | EATD, Harvard tweets, AVDD-TCMI |
| Newly downloaded this phase | **2** | Harvard tweets (4.5 MB, MD5 identical to the Dataverse record); AVDD-TCMI (13.5 GiB, SHA-256 identical to the Hub LFS oid) |
| Stored but not extracted | **1** | AVDD-TCMI (1,230 samples and all 2,460 CSVs verified inside the ZIP; extraction needs 33.70 GiB vs 32.0 GiB free) |
| Awaiting access approval | **3** | DAIC-WOZ, RSDD, SMHD |
| Awaiting licence review | **1** | EATD (licence not stated in the local copy) |

### Real counts (verified, never inferred)

| Quantity | Total | Detail |
|---|---|---|
| Participants | **162 verified** | EATD only. AVDD-TCMI releases 1,230 **samples** with no participant metadata, and Harvard has **no identity column at all**, so neither yields a participant count. DAIC-WOZ 189, RSDD 116,484 and SMHD 385,476 are *reported* by their distributors and are not held |
| Records | **101,306** | EATD 486 sessions + Harvard 99,590 tweets + AVDD 1,230 samples |
| Sessions | **486** | EATD only (3 prompted sessions per participant). AVDD samples are not sessions |
| Audio recordings | **486** (485 usable) | EATD only. AVDD ships 1,230 acoustic *feature* files (1,024-d, headerless), not recordings |
| Visual samples | **25,830** images + 1,230 visual feature CSVs | AVDD TCM facial-region jpgs and frame-level visual features. No raw video anywhere |
| Text samples | **100,076** | EATD 486 transcripts + Harvard 99,590 processed tweet texts |
| Storage on disk | **14.3 GB** | EATD 740.9 MB + Harvard 4.5 MB + AVDD archive 13.5 GiB |
| Storage if AVDD CSVs are extracted | +33.70 GiB (36,183,442,074 B) | only 32.0 GiB free, so extraction is deferred; the 25,830 jpgs (0.27 GiB) are never extracted |

### Validation results

| Dataset | Result |
|---|---|
| EATD | **validated** — 486/486 text and audio hashes match, 0 missing, 0 label-rule violations, splits intact |
| Harvard | **validated** — 99,590/99,590 rows, 0 empty, 0 exact duplicates, 264 near-duplicates with 0 label conflicts; label-code ambiguity and preprocessing recorded as warnings |
| AVDD-TCMI | **validated (full)** — SHA-256 equals the Hub LFS oid, `testzip` clean over 34,445 members, 1,230/1,230 samples with exactly one label row, 0 duplicate ids, 0 orphans, 0 raw audio/video; **every one of the 2,460 feature CSVs parsed in place**: 2.72 billion values, 0 NaN, 0 inf, 0 missing/ragged rows |
| DAIC-WOZ / RSDD / SMHD | **ACCESS_PENDING** — no data, no fabricated statistics; application requirements recorded |
| Checksum registry | **1,785 files** recorded, 1,785 verified, 0 problems (the AVDD archive was added) |

### Research eligibility

| Research use | Eligible today | Why |
|---|---|---|
| Text research | **EATD text, Harvard tweets** | both local, both validated; Harvard needs a stated label mapping |
| Voice research | **EATD audio only** | AVDD offers embeddings, not audio; DAIC-WOZ is pending |
| Visual research | **AVDD-TCMI** (feature-level and 21 facial-region crops per sample) | no raw video exists in any acquired dataset |
| Genuine paired multimodal | **EATD (text + audio)** and **AVDD-TCMI (acoustic + visual features, within a subset)** | both have per-participant/per-sample alignment and consistent labels; DAIC-WOZ would add text+audio+video once access is granted |

Nothing in this phase changes a Phase 2 artefact, an EATD split, a checkpoint,
the backend or the frontend. No model was trained, no fusion was attempted, and
no restricted dataset was acquired (no form submitted, no agreement accepted,
no credential created, nothing uploaded to a third party).

---

## 18. Phase 4 completion assessment (2026-11-14)

### 18.1 Completed — each backed by an artifact, not by a script existing

| Item | Evidence |
|---|---|
| Repository + dataset audit | `data/manifests/existing_datasets_audit.json`, `data/logs/audit_existing_datasets.log` |
| EATD validation | `eatd_validation.json` — 486/486 text and audio hashes, 0 missing, splits 114/24/24 intact |
| Harvard source verification + download | `harvard_tweets_download.json` — Dataverse v1.1, CC0, MD5 `c925e2fb…` matches the record |
| Harvard validation | `harvard_tweets_validation.json` — 99,590 rows, 0 empty, 0 exact duplicates, 264 near-duplicates, 0 label conflicts |
| Harvard label audit | `docs/harvard_tweets_label_audit.md` — ambiguity documented, codes preserved, use blocked |
| AVDD-TCMI download | 14,529,499,485 B, SHA-256 = Hub LFS oid, `testzip` clean, `data/logs/download_avdd_tcmi.log` |
| AVDD-TCMI archive validation | `avdd_tcmi_validation.json` — `validated`, 0 issues, 1,230 samples, 1,230/1,230 label rows |
| **AVDD-TCMI feature CSV validation** | `avdd_tcmi_acoustic_validation.json` / `avdd_tcmi_visual_validation.json` — 2,460 files parsed, 0 issues |
| **AVDD-TCMI label + alignment verification** | `avdd_tcmi_alignment_report.json` — 0 issues, 1,230 samples, 261/969 labels, acoustic/visual/label ids in agreement |
| **AVDD-TCMI extraction integrity** | `avdd_tcmi_extraction_integrity.json` — status `not_extracted`, exact size budget recorded |
| Access preparation | `docs/phase_4_access_status.md` + three `*_access_report.json` |
| EATD provenance + licence | `docs/eatd_provenance_and_license_audit.md` |
| Compatibility matrix | `docs/dataset_compatibility_matrix.md` |
| Dataset inventory | `data/reports/dataset_inventory.json`, `data/manifests/dataset_inventory.{csv,json}` |
| Checksum registry | 1,785 files recorded, 1,785 verified, 0 problems |
| Tests | `tests/test_avdd_validation.py` — 25 tests, all passing |

**AVDD-TCMI per-modality results (verified, in place, no extraction):**

| | Acoustic | Visual |
|---|---|---|
| Files | 1,230 (HDS 282 / SDS 948) | 1,230 (HDS 282 / SDS 948) |
| Columns | 1,024 (header-less) | 538 (OpenFace-style names, identical in every file) |
| Rows | 1,012,980 (136–1,193 per file) | 3,132,200 (608–6,860 per file) |
| Values | 1,037,291,520 | 1,685,123,600 |
| NaN / inf | 0 / 0 | 0 / 0 |
| Missing / non-numeric / ragged | 0 | 0 |
| Duplicate rows | 4,718 (identical frames within a file) | 0 |
| Constant columns | 0 | 2 (`face_id`, `success`) |
| Uncompressed | 24.6 GB | 9.1 GB |

Acoustic values sit in roughly [-1.4, +1.7] and look z-normalised; visual columns
are pixel coordinates and gaze/pose values, so their ranges are not comparable
and no global normalisation is applied or stored.

### 18.2 Partially completed

| Item | What is done | What is missing |
|---|---|---|
| AVDD-TCMI selective extraction | integrity verifier written and tested; size budget measured from the central directory | **the extraction itself was never performed** — 33.70 GiB needed vs 32.0 GiB free |
| Harvard label semantics | two candidate mappings documented with evidence and counts | the depositor's statement; supervised use stays blocked |
| EATD licensing | provenance chain verified; upstream has no licence file (GitHub API 404) | written terms from the authors |
| EATD label definition | SDS instrument verified; the ≥53 rule reproduces the authors' 30/132 split exactly | a decision on ≥50 (Phase 2 baselines) vs ≥53 (authors) |

### 18.3 Blocked

| Blocker | Datasets | Owner action |
|---|---|---|
| Depositor clarification (numeric label codes) | Harvard tweets | send the drafted e-mail in `docs/harvard_tweets_label_audit.md` §7 |
| Researcher application + signed DUA | DAIC-WOZ, RSDD, SMHD | sign and submit; nothing can be done locally |
| Licence clarification | EATD | ask the authors for written terms |
| Additional storage (~2 GB for the full CSV set, more if jpgs are wanted) | AVDD-TCMI extraction | free space, or approve reading from the ZIP (already works) |
| Missing source documentation | AVDD-TCMI feature dictionary, PHQ-8 label convention | none available; recorded as unverified |

### 18.4 Not started (deliberately out of scope)

- Any model training, fine-tuning, fusion model, or Phase 5 experiment.
- Acquisition of DAIC-WOZ, RSDD or SMHD (no form, no agreement, no download).
- Extraction of the AVDD-TCMI jpgs (explicitly forbidden this phase).
- Any relabelling of Harvard or re-thresholding of EATD.

### 18.5 Scientific limitations

1. **EATD is small.** 162 participants, 24 in the locked test split (8 positive).
   Any single-dataset result carries wide confidence intervals; a 1-point macro-F1
   difference is inside the noise.
2. **Labels are weak everywhere.** SDS is a self-report symptom scale, PHQ-8 a
   screening questionnaire, RSDD/SMHD self-reported diagnoses detected in text,
   Harvard an undocumented tweet-level binary. **None is a clinical diagnosis**,
   and no dataset supports a claim of detecting depression.
3. **Cross-dataset labels are incompatible.** SDS ≠ PHQ-8 ≠ self-reported
   diagnosis ≠ AVDD's undocumented binary ≠ Harvard's ambiguous codes. Datasets
   must not be merged into one training set with a shared label.
4. **AVDD-TCMI has no raw signal.** It ships pre-extracted 1,024-d acoustic and
   538-d visual features plus 21 face crops per sample — no audio, no video, no
   feature dictionary. A pipeline using it cannot be validated end to end, and its
   features are not comparable to features extracted by this project.
5. **AVDD-TCMI is not frame-aligned across modalities.** Acoustic and visual row
   counts are equal for **0 of 1,230** samples and their ratio takes 429 distinct
   values, so frames cannot be joined by index without an explicit resampling
   policy — which this phase deliberately does not choose.
6. **Access restrictions.** Three of six datasets are unusable today, and the one
   that would make the multimodal claim (DAIC-WOZ) is among them. EATD may not be
   redistributed at all.
7. **Participant-level leakage risks.** Harvard has no author identifier, so its
   evaluation can only be tweet-level and can leak the same author across splits.
   AVDD-TCMI releases no participant metadata, so its split can only be
   sample-level. Only EATD and (once acquired) DAIC-WOZ support genuine
   participant-independent evaluation.
8. **AVDD-TCMI HDS/SDS are two different label sources** within one release. They
   are reported separately and must never be pooled without stating so.

### 18.6 Phase 5 readiness: **READY WITH RESTRICTIONS**

*Why not NOT READY:* every locally acquirable dataset is downloaded, checksummed
and fully validated, the label semantics of each are either verified or explicitly
marked unresolved, the compatibility matrix states which combinations are legal,
and the test suite covering the validation utilities passes. The project can start
Phase 5 on EATD (text + audio, participant-level split) and on AVDD-TCMI
(acoustic + visual features, sample-level split) today, with real numbers and
known limits.

*Why not READY:* the primary multimodal benchmark (DAIC-WOZ) is not acquired, so
no text + audio + visual result at participant level is possible; Harvard cannot
be used for supervised work until the label mapping is confirmed; EATD's binary
threshold disagrees with the corpus authors' own split; and the AVDD feature
semantics are undocumented beyond column names.

The restrictions that must travel with any Phase 5 report:

1. State the dataset-specific label rule (EATD threshold, AVDD subset) in every
   table; never present a shared "depression" label across datasets.
2. Never claim clinical validity, diagnosis, or screening capability.
3. Split at the participant level where identity exists (EATD) and at the sample
   level where it does not (AVDD), and say which was used.
4. Do not merge datasets for training; cross-dataset numbers are ranking evidence
   only.
5. Do not redistribute EATD content, and do not quote RSDD/SMHD posts (not held).
6. Report AVDD acoustic and visual fusion as *feature-level* fusion, and resample
   explicitly before any frame-level alignment.

**Verdict: Phase 4 is complete for everything the project owns today, and
incomplete for everything that depends on a third party. Phase 5 may begin under
the six restrictions above.**
