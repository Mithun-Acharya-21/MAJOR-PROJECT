# Dataset compatibility matrix

Audit date: 2026-11-14 · Derived from the validation reports in `data/reports/`
and `data/reports/dataset_inventory.json`. Nothing here is hand-guessed: each
count is tagged with its evidence.

## 1. Units of observation — read this before the table

The six datasets do **not** share a unit of observation, and this is the single
biggest obstacle to combining them:

| Dataset | Unit | Count | Contains |
|---|---|---|---|
| EATD | prompted session (3 per participant) | 486 sessions / 162 participants | session ⊂ participant |
| AVDD-TCMI | **sample** (participant count not released) | 1,230 samples | 1 sample = 1 acoustic CSV + 1 visual CSV + 21 face crops |
| Harvard tweets | **tweet** | 99,590 tweets | **no user id at all** — tweets cannot be grouped |
| DAIC-WOZ | interview session | 189 sessions **[reported]** | 1 session per participant |
| RSDD | **user** | 116,484 users **[reported]** | many posts per user, already aggregated |
| SMHD | **user** | 385,476 users **[reported]** | many posts per user, already aggregated |

Consequences that are enforced in the code and in every report:

- A **tweet is not a user**, a **session is not a participant**, a **frame is not
  a participant**, and a **sample is not a participant**. Counts below are never
  promoted up that ladder.
- Harvard cannot be split by author, so its evaluation is tweet-level only.
- EATD and DAIC can be split at participant level; AVDD can only be split at
  sample level because participant identity is not in the release.

## 2. Matrix

| Dataset | Modality | Language | Unit of observation | Sample count | Participant/user count | Label type | Label mapping verified? | Access status | Intended research role |
|---|---|---|---|---|---|---|---|---|---|
| **EATD** | text + audio (raw 16 kHz WAV, transcripts) | zh | prompted session | 486 sessions (486 audio, 486 transcripts) | **162** participants (verified) | Zung SDS self-report index → binary (`≥50` locally) | **Yes** for the instrument; **no** for the binary threshold (authors ≈ `≥53`) | **Available**, local, licence unresolved | 4 — existing unimodal benchmark |
| **AVDD-TCMI** | acoustic features (1,024-d) + visual features (538-d) + TCM face crops; **no raw audio/video** | zh | sample | **1,230** samples (HDS 282, SDS 948) | **not released** | binary depressed / non-depressed (HDS 159/123, SDS 810/138) | **Yes** — from the dataset card and released label files, not inferred from signal | **Available** (CC-BY-NC-4.0), archive stored | 2 — auxiliary multimodal feature research |
| **Harvard tweets** | text only (pre-processed tokens) | en | tweet | **99,590** tweets | **unknown, not derivable** (no id column) | binary `0`/`1` | **No — unresolved ambiguity** | **Available** (CC0), local | 3 — text-only exploratory research |
| **DAIC-WOZ** | text + audio + facial features (+video under DUA) | en | interview session | 189 sessions **[reported]** | **189** participants **[reported]** | PHQ-8 questionnaire (0–24) | **No** — label files not seen | **ACCESS_PENDING** — not acquired | 1 — primary multimodal benchmark (on acquisition) |
| **RSDD** | text only (post collections) | en | user | 116,484 users **[reported]** | **116,484** users **[reported]** | self-reported diagnosis, crowdsource-verified (weak) | **No** — data not seen | **ACCESS_PENDING** — not acquired | 3 — text-only exploratory / external generalisation |
| **SMHD** | text only (post collections) | en | user | 385,476 users **[reported]** | **385,476** users **[reported]** | self-reported condition, 9 classes (weak) | **No** — data not seen | **ACCESS_PENDING** — not acquired | 3 — text-only exploratory (depression-vs-control only) |

`[reported]` = stated by the dataset's own documentation and **not** reproducible
locally, because the data is not held. Nothing in this table is marked verified
unless the bytes were checked.

## 3. Role classification and reasoning

**1 — Primary multimodal benchmark: DAIC-WOZ.**
It is the only shortlisted corpus that pairs *text + audio + facial* data with a
per-participant questionnaire label, which is exactly the shape the project's
fusion claim needs. It is also the only one that can support participant-level
train/test separation across all three modalities. It is not acquired, which is
why the project currently has **no** primary multimodal benchmark available.

**2 — Auxiliary multimodal feature research: AVDD-TCMI.**
It has both acoustic and visual features for every sample, so it can support
fusion experiments — but the release contains **only pre-extracted features and
face crops**, no raw audio or video, and no participant metadata. It therefore
cannot validate a pipeline end to end (no raw signal in, no speaker-level split),
and it cannot be compared to EATD/DAIC at the feature level because the features
were produced by different extractors. It is genuinely useful for
feature-level fusion and cross-corpus robustness, auxiliary to the primary
benchmark.

**3 — Text-only exploratory research: Harvard tweets, RSDD, SMHD.**
All three are text-only. Harvard is tweet-level with no author id and an
unresolved label mapping; RSDD and SMHD are user-level with self-reported,
weakly labelled diagnoses. None can carry a multimodal claim. RSDD and SMHD have
*stronger* labels than Harvard (self-reported diagnosis vs an undocumented tweet
label) but are gated behind DUAs.

**4 — Existing unimodal benchmark: EATD.**
162 participants, 486 sessions, text + audio, participant-level split
(114/24/24, locked test 24 participants / 8 positive). It is the project's only
validated benchmark today and anchors the unimodal baselines from Phase 2. It is
not multimodal (no face, no video) and its licence does not permit redistribution.

**5 — Access-pending research dataset: DAIC-WOZ, RSDD, SMHD.**
These three are not usable until the owner signs and submits the respective
agreements. Their inclusion in the matrix is a plan, not a capability.

## 4. Label incompatibility — the hard constraint

| Instrument | What it actually measures | Comparable with |
|---|---|---|
| Zung SDS index (EATD) | 20-item **self-report symptom scale**, 25–100 index | nothing else in this set |
| PHQ-8 (DAIC-WOZ) | 8-item **interviewer-administered questionnaire**, 0–24 | not SDS |
| AVDD-TCMI binary label | binary class, instrument **not documented** in the release | not stated |
| Harvard `0`/`1` | binary class, **codes undocumented, meaning unresolved** | nothing |
| RSDD / SMHD | **self-reported diagnosis** detected in text, crowdsource-verified | not a questionnaire |

No two of these measure the same construct, and none is a clinical diagnosis.
Therefore:

- **Do not merge datasets into one training set with a shared binary label.**
- **Do not compare headline metrics across datasets** as if "depressed" meant the
  same thing. Cross-dataset numbers are only meaningful as *ranking* evidence,
  and only with the label definition printed alongside.
- A shared 0/1 target is acceptable **only** within one dataset, with the
  dataset-specific rule stated in the report.

## 5. What a compatible combination would look like

| Combination | Verdict | Condition |
|---|---|---|
| EATD text+audio unimodal baselines | **Allowed now** | state SDS threshold |
| AVDD acoustic+visual feature fusion | **Allowed now** | sample-level split only; no participant claims |
| Harvard text experiments | **Blocked for supervised use** | needs the depositor's code mapping |
| EATD + AVDD joint training | **Not allowed** | different units (session vs sample) and different label instruments |
| EATD + DAIC | Would be the real benchmark | needs DAIC access; still two separate label definitions |
| Any dataset + SMHD/RSDD users | **Never link individuals** | DUAs forbid it explicitly |

## 6. Consistency with the machine-readable inventory

`data/reports/dataset_inventory.json` (regenerated from the validation reports,
not hand-written) carries the same rows plus `label_mapping_verified`. Both files
are produced by `scripts/datasets/generate_dataset_inventory.py`, so the table
above and the JSON cannot drift silently. A test asserts the invariants:
three datasets `available`, three `ACCESS_PENDING`, Harvard
`label_mapping_verified=false`, AVDD `participants=null`.
