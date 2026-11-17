# Phase 5 — Research Experimental Protocol

Status: **design only** (Phase 5A). No model was trained, no checkpoint written, no
manifest, split or label modified in this phase.

Date: 2026-11-15
Supersedes nothing. Phase 2 artifacts remain the frozen reference results.

Evidence tags used throughout:

| Tag | Meaning |
|---|---|
| **[V]** verified | read from a file in this repository during this phase; the path is given |
| **[I]** inferred | follows from verified evidence but is not stated by a source |
| **[R]** recommendation | a design choice made by this protocol, not a fact about the data |
| **[B]** blocked | cannot proceed until an external condition is met |

This system performs **research on depression-related symptom signals**. Nothing in
this document, and nothing produced under it, is a clinical diagnosis, a screening
device, or evidence of clinical validity. Every metric in Phase 5 inherits
`clinical_validation = false`.

---

## 1. Scope

### 1.1 In scope

1. A selection-and-evaluation protocol for the EATD unimodal baselines that does not
   consume the locked test split.
2. A sensitivity analysis for the EATD SDS label threshold.
3. A sample-level protocol for the AVDD-TCMI pre-extracted acoustic and visual
   features, including the conditions under which fusion may be attempted.
4. Explicit restrictions for Harvard tweets, DAIC-WOZ, RSDD and SMHD.
5. A statistical evaluation plan, an experiment matrix, a risk register, and a
   machine-readable inventory.

### 1.2 Out of scope for Phase 5A

No training, no fine-tuning, no feature extraction, no dataset download, no
archive extraction, no JPG extraction, no threshold change, no relabelling, no
frontend or backend change, no PyTorch/CUDA change, no modification of
`data/processed/eatd/*`, `models/experiments/*`, `models/final/*` or `results/phase2/*`.

### 1.3 Preservation guarantees

| Artifact | Guarantee |
|---|---|
| `data/processed/eatd/{manifest,participants,splits}.csv` | read-only; Phase 5 reads and never writes |
| `data/processed/eatd/split_report.json` | read-only; seed 42 assignment unchanged |
| `results/phase2/**` | read-only; Phase 2 numbers are cited, never regenerated |
| `models/experiments/*`, `models/final/*` | read-only; no Phase 2 checkpoint is loaded for training or overwritten |
| `data/raw/avdd_tcmi/*.zip` | read-only; no extraction, no JPG extraction |
| `data/eatd/**` | read-only; raw corpus untouched |
| Git | pre-existing unrelated changes are not staged or reverted |

---

## 2. Verified baseline

Every count below was re-read from the artifact named beside it in Phase 5A. Where a
figure is quoted from an external distributor it is tagged **[reported]** and is not
treated as verified.

### 2.1 EATD

| Quantity | Value | Evidence |
|---|---|---|
| Participants | 162 (83 `train/` + 79 `validation/` source folders) | `data/reports/eatd_validation.json` **[V]** |
| Sessions | 486 (3 prompted topics × 162) | `data/reports/eatd_validation.json` **[V]** |
| Audio-usable sessions | 485 (one 0.48 s file excluded from audio modelling) | `docs/eatd_dataset_card.md` §5b **[V]** |
| Participant key | `(source_partition, subject_id)` — 58 numeric ids repeat across partitions as different people | `docs/eatd_dataset_card.md` §2 **[V]** |
| Split | train 114 / validation 24 / test 24 participants (342 / 72 / 72 sessions), seed 42, stratified by label, zero overlap | `data/processed/eatd/split_report.json` **[V]** |
| Label | `sds_index = sds_raw × 1.25`; `label = 1 if sds_index ≥ 50` | `data/processed/eatd/split_report.json` **[V]** |
| Class counts at ≥ 50 | 109 control / 53 depressed overall; train 77/37, validation 16/8, test 16/8 | `data/processed/eatd/splits.csv` **[V]** |
| Class counts at ≥ 53 | 132 control / 30 depressed overall; train 93/21, validation 18/6, test 21/3 | derived, `data/processed/eatd_phase5/label_threshold_sensitivity.csv` **[V]** |
| Locked test positives | 8 (at ≥ 50); 3 (at ≥ 53) | as above **[V]** |

Note the deviation from the project card: the authors' own README split (19 + 11 = 30
depressed / 132 control) is reproduced **exactly** by SDS ≥ 53, not by ≥ 50, and the
23 participants that differ all sit in the index range 50.00–52.50
(`docs/eatd_provenance_and_license_audit.md` §5, re-verified here) **[V]**. This is a
label-definition discrepancy, not a data discrepancy. It is **not** resolved in
Phase 5A (§5).

### 2.2 Phase 2 results (frozen, locked test, participant level, threshold 0.5)

| ID | Model | Modality | ROC-AUC | F1 | 95 % CI on ROC-AUC |
|---|---|---|---|---|---|
| T1 | TF-IDF + Logistic Regression | text | 0.695 | 0.182 | 0.430–0.925 |
| T2a | Frozen multilingual transformer + head | text | 0.531 | 0.400 | 0.241–0.815 |
| T2b | Transformer top-2-layer fine-tune | text | 0.570 | 0.526 | 0.242–0.874 |
| V1 | eGeMAPS + Logistic Regression | voice | 0.383 | 0.364 | 0.139–0.644 |
| V2 | Wav2Vec2 frozen + head | voice | 0.398 | 0.364 | 0.135–0.667 |

Source: `results/phase2/comparisons/comparison.json` **[V]**; bootstrap = participant
cluster, 2000 resamples, seed 0; n = 24 participants (8 positive).
Every interval contains 0.5. Validation→test ROC-AUC gaps are 0.195–0.414
(`docs/phase_2_unimodal_baselines.md` §4) **[V]**. **These five numbers are the only
EATD test results that exist, and Phase 5 must not generate a sixth at this stage.**

### 2.3 AVDD-TCMI

| Quantity | Value | Evidence |
|---|---|---|
| Samples | 1,230 (HDS 282 / SDS 948) | `data/reports/avdd_tcmi_alignment_report.json` **[V]** |
| Acoustic features | 1,024 columns, header-less (`f0..f1023`) | `docs/avdd_tcmi_feature_inspection.md` §1 **[V]** |
| Visual features | 538 columns, OpenFace-style, identical header in every file | §2 **[V]** |
| Acoustic frames/sample | min 136, median 820.5, max 1,193 (total 1,012,980) | `data/reports/avdd_tcmi_acoustic_validation.json` **[V]** |
| Visual frames/sample | min 608, median 1,601.5, max 6,860 (total 3,132,200) | `data/reports/avdd_tcmi_visual_validation.json` **[V]** |
| Matching sample IDs | 1,230 acoustic / 1,230 visual / 0 orphans | alignment report **[V]** |
| Equal acoustic/visual frame counts | **0 of 1,230** | alignment report **[V]** |
| Distinct acoustic:visual row ratios | 429 | alignment report **[V]** |
| Constant visual columns | 2 (`face_id`, `success`) | visual validation **[V]** |
| Constant acoustic columns | 0 | acoustic validation **[V]** |
| Duplicate acoustic rows | 4,718 within-file frame duplicates | acoustic validation **[V]** |
| Participant metadata | **none** — no speaker/subject id in the release | validation report, `participants: null` **[V]** |
| Labels | HDS 159/123, SDS 810/138 → 969 non-depressed / 261 depressed | alignment report **[V]** |
| Raw audio / raw video | 0 / 0 (25,830 TCM face JPGs only) | `data/reports/avdd_tcmi_validation.json` **[V]** |
| Licence | CC-BY-NC-4.0 | inventory **[V]** |
| Archive / free disk | 13.5 GB archive; CSV extraction needs 33.7 GB; 32.0 GB free → does not fit | `avdd_tcmi_extraction_integrity.json` **[V]** |

### 2.4 Other datasets

| Dataset | Verified state | Blocking condition |
|---|---|---|
| Harvard tweets | 99,590 records, no user/tweet id column, label ∈ {0,1} with 58,405 / 41,185 rows, code→class mapping **unresolved** | depositor statement **[B]** |
| DAIC-WOZ | not acquired, 189 sessions **[reported]** | signed DUA + institutional e-mail **[B]** |
| RSDD | not acquired, 116,484 users **[reported]** | signed DUA **[B]** |
| SMHD | not acquired, 385,476 users **[reported]** | signed DUA **[B]** |

Sources: `data/reports/dataset_inventory.json`, `docs/harvard_tweets_label_audit.md`,
`docs/phase_4_access_status.md` **[V]**.

---

## 3. Research questions

| ID | Research question | Experiment(s) | Primary evaluation unit |
|---|---|---|---|
| RQ1 | Can a text-only model classify EATD SDS band above chance, estimated without touching the locked test set? | E-T1, E-T2, E-T3 | EATD participant (n = 138 dev) |
| RQ2 | Do acoustic features carry EATD SDS-band signal under the same protocol? | E-V1, E-V2 | EATD participant (n = 138 dev) |
| RQ3 | Do AVDD-TCMI acoustic features carry sample-level label signal? | E-A1 | AVDD sample (n = 1,230) |
| RQ4 | Do AVDD-TCMI visual features carry signal beyond constant columns and beyond recording length? | E-A2, E-A3 | AVDD sample (n = 1,230) |
| RQ5 | Are the text and voice branches complementary, i.e. is multimodal fusion scientifically justified? | E-F1 (gated), E-A4 (gated) | participant / sample per dataset |
| RQ6 | How much of Phase 2's apparent signal is split noise and selection optimism? | E-T1…E-V4 vs Phase 2 numbers | participant (in-fold) |
| RQ7 | Which label definition (SDS ≥ 50 vs ≥ 53) is defensible, and how sensitive are conclusions to it? | E-L1 | participant, both label versions |

RQ6 is a **replication/attribution** question, not a leaderboard question: the answer is
expected to be "much of it", and a result that confirms the Phase 2 numbers is not a
success criterion.

---

## 4. EATD experimental protocol

### 4.1 Evaluation pool

Phase 5 model selection and estimation use **train + validation only: 138 participants
(414 sessions), 93 control / 45 depressed at ≥ 50** **[V]**.

The 24 locked test participants are **not loaded** by any Phase 5 selection script.
This is stronger than Phase 2, where selection used 24 validation participants once per
experiment; the ≥ 50 live test set is now reserved for a single, later, pre-registered
confirmation (§4.9).

Consequence stated plainly: the dev pool is 138 participants with 45 positives, so an
outer fold holds ~9 positives. Every point estimate in Phase 5B remains wide. The gain
over Phase 2 is *estimator honesty* (no selection on the reported set, repeated folds,
per-fold dispersion), not statistical power.

### 4.2 Splitting

- **Unit of splitting and resampling:** participant key. Sessions are never split
  independently. `ml/phase2/dataset.py::assert_split_integrity` and
  `participant_groups()` are reused; a fold assignment that separates one participant's
  sessions is a hard error. **[R]**
- **Outer loop:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`,
  groups = participant key, stratify = label. 5 folds × 5 repeats = 25 outer folds. **[R]**
- **Repeats:** seeds 42, 43, 44, 45, 46 (42 is the project seed; the others are
  pre-registered, not chosen after seeing results). **[R]**
- **Inner loop:** `StratifiedGroupKFold(n_splits=4)` inside each outer training part,
  used for hyperparameter grid search and for producing inner out-of-fold (OOF)
  probabilities that feed threshold selection and calibration. **[R]**
- Hyperparameters are the Phase 2 grids in `ml/configs/phase2/*.yaml`, unchanged, so the
  comparison to Phase 2 isolates the estimation protocol rather than the model space. **[V]**

### 4.3 What is fitted where

| Component | Fitted on | Never fitted on |
|---|---|---|
| TF-IDF vocabulary / idf | outer-train (inner folds for grid) | outer-val, locked test |
| Scaler / imputer | outer-train | outer-val, locked test |
| Class weights / resampling | outer-train only | outer-val |
| PCA / feature selection | inner-fold train or outer-train | outer-val |
| Probability calibrator | inner OOF predictions | outer-val |
| Decision threshold | inner OOF predictions | outer-val, locked test |
| Hyperparameters | inner CV | outer-val, locked test |

### 4.4 Participant aggregation

Phase 2 used the mean of the three session probabilities. Phase 5 makes the aggregation
a **declared, fixed analysis choice**, evaluated as its own sensitivity axis:

| Aggregation | Role |
|---|---|
| mean of session probabilities | primary (matches Phase 2, keeps comparability) **[R]** |
| max | secondary, exploratory (E-V3) |
| median | secondary, exploratory (E-V3) |
| session-weighted by audio duration | secondary (E-V3) |

The primary aggregation must be declared before any prediction is inspected. Choosing
`max` because it scores better is prohibited; all four are reported in E-V3 with the
same folds.

### 4.5 Class imbalance

- Reported metrics are imbalance-aware by default (PR-AUC, balanced accuracy,
  sensitivity at fixed specificity, MCC) — see `docs/phase_5_statistical_evaluation_plan.md`.
- Candidate imbalance handling, **all compared inside the same nested CV**: (a) no
  weighting, (b) `class_weight="balanced"`, (c) random undersampling of the majority
  class in the training part only, (d) no resampling + threshold tuned on inner OOF. **[R]**
- SMOTE-style synthetic oversampling is **not** used: the dev pool is 414 sessions of
  correlated prompted speech, and interpolating in a 1,024- or 768-dimensional space
  with 45 positives would manufacture structure, not evidence. This is a documented
  refusal, not an oversight. **[R]**
- Accuracy is never the headline metric: a majority-class predictor already reaches
  ~0.67 participant accuracy on the dev pool.

### 4.6 Threshold selection policy

1. Thresholds are selected on **inner-fold OOF probabilities** by maximising balanced
   accuracy (primary criterion), with Youden's J and "sensitivity at specificity ≥ 0.80"
   reported as alternative policies.
2. The selected threshold is then applied unchanged to the outer fold.
3. The distribution of selected thresholds across the 25 outer folds is reported
   (median, IQR, min, max). Wide dispersion is itself a result: it means the operating
   point is not identifiable at this sample size.
4. Results are reported at **both** the selected threshold and the fixed 0.5 default, so
   a reader can see how much of any F1 comes from the threshold.
5. The locked test set is never used to choose a threshold.

### 4.7 Calibration

- Platt (sigmoid) scaling is the primary calibrator, fitted on inner OOF probabilities.
  Isotonic is reported only as a sensitivity check; with 45 positives it overfits. **[R]**
- Reported: Brier score, expected calibration error (10 equal-frequency bins), and
  calibration slope/intercept from a logistic re-fit of the outcome on the logit of the
  predicted probability.
- ECE with ~28 participants per outer fold is a descriptive statistic only. No bin-level
  calibration claim is made below n = 200. **[R]**
- `scripts/fit_calibration.py` and `models/calibration.json` belong to the serving
  application; Phase 5 calibration is a research artifact and does not replace them. **[V]**

### 4.8 Reporting of folds

Every experiment emits per-fold rows: fold id, repeat seed, outer-train/val participant
ids and class counts, chosen hyperparameters, chosen threshold, all metrics, and the
number of aggregated sessions. Aggregate rows are mean ± SD across the 25 folds plus the
repeat-level means. **A mean without the per-fold spread is not reportable.**

### 4.9 Optional single confirmation on the locked test set

Permitted only after: (a) Phase 5B–5D are complete, (b) one model per modality is
pre-registered with a fixed threshold and fixed hyperparameters selected by inner CV,
(c) the pre-registration file records the model, threshold, metric and analysis script
hash before the test set is read. One pass, one comparison, no post-hoc revision.
The existing 24-participant test set (8 positive) is not enlarged by Phase 5.

---

## 5. EATD label threshold sensitivity analysis (E-L1)

### 5.1 The discrepancy

The corpus README's own class split is 30 depressed / 132 control; SDS ≥ 53 reproduces it
exactly on all 162 participants, and SDS ≥ 50 (the Phase 2 rule) does not
(`docs/eatd_provenance_and_license_audit.md` §5) **[V]**. 23 participants change class
between the two rules, all with SDS index in 50.00–52.50 **[V]** — i.e. every disagreement
sits inside a single 2.5-point band of a self-report scale.

### 5.2 Preservation of the original labels

`data/processed/eatd/{manifest,participants,splits}.csv` and their `label` column are
untouched. No Phase 2 result is recomputed. The ≥ 50 rule remains the definition of the
Phase 2 baselines and remains the definition used by RQ1–RQ6 unless the owner decides
otherwise.

### 5.3 Representation of the alternative threshold

A separate derived file, additive only:

`data/processed/eatd_phase5/label_threshold_sensitivity.csv`

| Column | Meaning |
|---|---|
| `participant_key` | as in `splits.csv` |
| `split` | copied from `splits.csv`, unchanged |
| `sds_raw`, `sds_index` | copied from `participants.csv`, unchanged |
| `label_sds50` | the existing Phase 2 label, unchanged |
| `label_sds53` | derived alternative, `1 if sds_index >= 53` |
| `flip` | `label_sds50 != label_sds53` |
| `sds_band` | `<50`, `50–52.5` (the disagreement band), `>=53` |

162 rows, generated read-only from the two locked CSVs; no new information, no new
identity, no session rows.

### 5.4 How the distributions change

| Rule | All (162) | Train (114) | Validation (24) | Locked test (24) |
|---|---|---|---|---|
| ≥ 50 | 53 / 109 (32.7 % pos) | 37 / 77 | 8 / 16 | **8 / 16** |
| ≥ 53 | 30 / 132 (18.5 % pos) | 21 / 93 | 6 / 18 | **3 / 21** |

Consequences that must be stated in any report using ≥ 53:

- The locked test set would hold **3 positives**. A participant-level ROC-AUC on 3
  positives is not estimable with a usable interval; PR-AUC would be dominated by one
  sample.
- The **locked split assignment stays valid** (it is stratified on the ≥ 50 label), but
  its class balance is no longer optimal for ≥ 53. Re-stratifying would invalidate the
  lock. Keeping the lock and accepting imbalance is the only preservation-compatible
  option. **[R]**
- Phase 2 checkpoints and baselines cannot be compared across rules without retraining.
  A ≥ 53 run is a **new experiment family**, not a correction.

### 5.5 How existing experiments remain reproducible

Phase 2 artifacts store the label definition string (`label = 1 if sds_index >= 50`,
`docs/phase_2_unimodal_baselines.md` §2) **[V]**. Any Phase 5 report must print
`label_rule` next to every metric. The experiment inventory JSON carries
`label_rule: "sds_index_ge_50"` on every EATD experiment and a separate entry for the
≥ 53 family.

### 5.6 Evidence required before selecting a primary threshold

Selection must not be made on model performance. Required, in order:

1. **Source evidence** from the EATD authors: the code or description that produces
   their 30/132 split (README text is insufficient to prove ≥ 53 was used, only that it
   reproduces the totals). Ask for the label-generation script or a statement.
2. **Instrument documentation**: whether the released `new_label.txt` is the SDS index
   (`raw × 1.25`) as assumed, or an already-transformed score, and which Chinese norm
   the authors apply. `docs/eatd_dataset_card.md` §3 notes the index formula is verified
   arithmetically but the norm is not sourced **[V]**.
3. **Reporting-convention evidence**: how comparable published work on this corpus
   defines the positive class.
4. Only then, a decision record stating the primary rule, its rationale (clinical-norm or
   author-match), and the impact on the locked test set's estimability.

A threshold that yields a better ROC-AUC is **not** evidence. If ≥ 53 were chosen because
it makes models look better, that would be a reportable research-integrity failure.

---

## 6. AVDD-TCMI experimental protocol

### 6.1 Unit of observation — non-negotiable

The unit is a **sample**. It has an integer id within its subset (HDS or SDS) and
**no participant identity is released** **[V]**. Therefore:

- The words "participant", "subject" and "speaker" are not used for AVDD results.
- The bootstrap unit is the sample.
- Per-sample confidence intervals may be **optimistic** if several samples come from one
  person. Because that cannot be checked, every AVDD interval carries the caveat
  "sample-level resampling; clustering by speaker unknown". **[R]**
- HDS and SDS are reported as **separate populations** and are never pooled into one
  training set by default. The subsets may differ in label instrument, gender balance
  (HDS f/m 194/88; SDS 517/431) and recording protocol **[V]**; pooling them assumes a
  shared label semantics that the release does not document. A pooled run is a separate
  pre-registered experiment with the assumption written down (E-A5, gated).

### 6.2 Loading

ZIP-backed, read-only streaming from
`data/raw/avdd_tcmi/AVDD-TCMI-dataset.zip` (13.5 GB) **[V]**. Full CSV extraction is
**not** permitted: it needs 33.7 GB against 32.0 GB free **[V]**, and extraction would
duplicate 33.7 GB for no analytic gain. JPG extraction is prohibited outright. Access is
per sample (`acoustic/<subset>/<id>.csv`, `visual/<subset>/<id>.csv`), cached only as
small aggregated matrices inside a Phase 5 working directory. **[R]**

Practical consequence: ~34 GB of decompression per full pass. The protocol therefore
requires a **feature cache** (one row per sample per modality) built once, with its
SHA-256 and member list recorded, and every subsequent experiment reading the cache. **[R]**

### 6.3 Frame-level to sample-level aggregation

Acoustic frame count and visual frame count are **never equal** (0 / 1,230) and the ratio
takes 429 distinct values **[V]**, so no frame-level join exists and none is invented.
No alignment is assumed, interpolated, or resampled across modalities.

Both modalities are reduced independently by the same fixed functionals computed **over
time within a sample**:

| Statistic set | Description |
|---|---|
| mean, std | distributional summary |
| median, IQR (p75−p25) | robustness to silent/occluded frames |
| p05, p95 | dynamic range |
| slope of a least-squares fit against normalised frame index | crude within-sample trend |

Frame index is normalised `0..1` per sample so samples of different length are
comparable without assuming a shared time base. This is a **within-modality temporal
summary**, not alignment. **[R]**

Alternative reductions (percentiles only; Δ-style first differences) are sensitivity
analyses (E-A6), not primary.

### 6.4 Column policy for the visual modality

Decision, documented once and applied identically everywhere (no automatic removal
without a record):

| Column group | Count | Decision | Reason |
|---|---|---|---|
| `face_id`, `success` | 2 | drop | constant across the whole corpus (std = 0), so they carry no information and cannot be standardised **[V]** |
| `frame`, `timestamp` | 2 | exclude from model features; retain for QC | they encode sample duration/frame rate, not behaviour; including them would let the model read recording length |
| `confidence` | 1 | exclude from primary; report mean per sample as a QC covariate | tracking quality; a plausible shortcut for label-correlated recording conditions |
| `gaze_*` | 8 | keep | varies; documented OpenFace semantics |
| `pose_*` | 6 | keep | varies |
| `AU*` | 35 | keep | all 35 have std > 1e-3 corpus-wide; the most behaviourally interpretable block **[V]** |
| `eye_lmk_x_*`, `eye_lmk_y_*` (2 × 56) | 112 | keep, flagged | raw 2-D eye-landmark pixel coordinates: strongly expected to mix head pose, camera distance and identity |
| `eye_lmk_X_*`, `eye_lmk_Y_*`, `eye_lmk_Z_*` (3 × 56) | 168 | keep, flagged | 3-D eye-landmark block; same caveat |
| `X_*`, `Y_*`, `Z_*` (3 × 68) | 204 | keep, flagged | 3-D facial landmark block; same caveat |

So of 538 columns: 2 removed as constant, 3 removed from model features as
metadata/quality, **533 retained**, of which 49 are AU/gaze/pose and **484 / 533 are raw
landmark coordinates** whose absolute values confound identity, camera geometry and head
pose. **E-A3 tests this
directly** (AU-only and AU+pose+gaze ablations) rather than assuming it.

"Meaningful information beyond constant columns" is therefore an empirical question, not
a preprocessing step: constant columns are only 2 of 538, so the honest statement is
that the *non-constant* columns vary — variation is not evidence of signal, and E-A2/A3
measure whether any of it predicts the label.

### 6.5 Splitting (AVDD)

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` on samples, stratified by
  label, within each subset. 5 folds × 5 repeats (seeds 42–46). **[R]**
- Standardisation, any column filtering beyond §6.4, and PCA (if used) are fitted on the
  training part only. With 1,024 × 8 ≈ 8,192 acoustic features and 948 SDS samples,
  in-fold PCA/regularisation is mandatory. **[R]**
- Because speaker identity is unknown, a duplicate/near-duplicate sample check runs
  before any experiment: identical aggregated feature rows across folds would indicate
  repeated recordings. Any duplicate group is kept inside one fold. **[R]**

### 6.6 Fusion gate (E-A4 / E-F1)

Fusion is attempted only if all of the following hold, evaluated on the *same* protocol:

1. **Both** unimodal branches beat chance, with a 95 % CI excluding 0.5 on the primary
   metric (ROC-AUC) in the same subset, under the same folds and repeat seeds.
2. The branches' per-sample/per-participant prediction errors are **not** near-identical:
   Pearson correlation of out-of-fold probabilities < 0.7, and each branch has ≥ 20 % of
   positives that the other misses.
3. Fusion (late: logistic stack on OOF probabilities; early: concatenated features with
   in-fold scaling/PCA) improves PR-AUC by more than the fold-to-fold SD of PR-AUC, with
   a paired bootstrap over the same resampling units giving a CI excluding 0.

If any condition fails, the reported outcome is **"fusion not justified"**, which is a
legitimate and expected result. Fusion is not attempted on EATD (text + voice) only
because "multimodal" is in the project description.

For AVDD the fusion claim is deliberately weaker than for EATD: two independently
reduced feature sets from unknown extractors, with no shared time base, fused at sample
level, is *feature-level fusion of summaries*, not temporal multimodal modelling.
The report must say exactly that.

### 6.7 Pre-training review gate

Before any AVDD model is fitted, these must be reviewed and signed off in a note:

1. The label semantics for HDS and SDS (`1` = depressed is taken from the dataset card
   and label files, **not** inferred from signal) **[V]**, plus what the subsets mean.
2. The absence of a feature dictionary for the 1,024 acoustic columns, and therefore the
   impossibility of interpreting acoustic coefficients **[V]**.
3. The untested nature of the time unit in `timestamp` and the unknown simultaneity of
   acoustic and visual frames **[V]**.
4. The decision in §6.4 and the two documented removals.

---

## 7. Harvard tweets — restrictions

- The numeric code → class mapping is **unresolved** **[V]**. No supervised classification,
  no metric, no threshold, no comparison with EATD or AVDD.
- Permitted: tokenisation/structural analysis, length and vocabulary statistics,
  unsupervised representation work, and *descriptive* reporting of `label ∈ {0,1}` with
  the two counts and the ambiguity stated in the same table.
- If a binary metric is ever produced, the assumed mapping must be printed beside it and
  the complementary-mapping run reported alongside, so the reader can see that the label,
  not the model, produces the difference (`docs/harvard_tweets_label_audit.md` §5) **[V]**.
- Unblocking evidence, any one of: (1) depositor reply stating the mapping; (2) an
  author-released script/notebook using the positive class; (3) a second released file
  with per-class indices. Until then the status is `BLOCKED_SUPERVISED`.

---

## 8. DAIC-WOZ, RSDD, SMHD — future-acquisition protocol

### 8.1 Standing rule

These datasets are **not held**. No Phase 5 document may imply otherwise: no count, no
metric, no split, no example. The inventory marks them `ACCESS_PENDING` and
`validated: false` **[V]**.

### 8.2 Pre-registration template (to be completed on acquisition)

For each dataset, before the first model is fitted, record: verified file inventory and
hashes; the label file's own convention (DAIC: PHQ-8 sum and the released cut-off,
verified from the distributed label file, **not** assumed to be ≥ 10); the unit of
observation; the participant/user key; the split origin (official split vs locally
generated) and its seed; the join policy for modalities; and the licence/DUA constraints
on storage and reporting.

### 8.3 DAIC-WOZ priority

DAIC-WOZ is the only shortlisted corpus with paired text + audio + facial data and a
per-participant questionnaire label, and therefore the only one that could support the
project's genuine multimodal, participant-level claim. It is the highest-priority
acquisition, gated on a signed DUA and an institutional e-mail
(`docs/phase_4_access_status.md` §2) **[V]**. Until then, no multimodal participant-level
result can be produced, and the Phase 5 documents must not report one.

### 8.4 Harmonisation prohibitions

- **Never** merge RSDD or SMHD users with EATD or DAIC participants. Their DUAs forbid
  cross-linking individuals, and the label instruments differ (self-reported diagnosis
  detected in text vs questionnaire score) **[V]**.
- **Never** pool EATD, AVDD, DAIC and social-media corpora into one training set with a
  shared binary label. Only one of those labels (EATD ≥ 50) is even locally verifiable,
  and each measures a different construct (`docs/dataset_compatibility_matrix.md` §4) **[V]**.
- Cross-dataset work is limited to **ranking/robustness** statements with the label rule
  printed for each dataset, and only for datasets whose label mapping is verified.
- SMHD is used as depression-vs-control only, never as nine-way classification **[V]**.

---

## 9. Metrics and statistical reporting

See `docs/phase_5_statistical_evaluation_plan.md` for the full specification. Summary of
the binding rules:

1. No metric is reported without its **evaluation unit and sample count**
   (e.g. "ROC-AUC 0.71, EATD participant level, n = 138 dev, 45 positive, 25 folds").
2. Primary metrics: **ROC-AUC** and **PR-AUC** with 95 % CIs. Secondary: macro F1,
   precision, recall/sensitivity, specificity, balanced accuracy, MCC, confusion matrix at
   the declared threshold, Brier score, ECE, calibration slope.
3. Under class imbalance and small n, PR-AUC and sensitivity-at-fixed-specificity are more
   informative than accuracy or F1 at 0.5; ROC-AUC is retained because it is
   threshold-free and comparable to Phase 2.
4. Confidence intervals: participant-cluster bootstrap for EATD (2,000 resamples),
   sample-level bootstrap for AVDD, explicitly labelled as such.
5. All 25 outer folds are reported; means are accompanied by SD and the per-repeat means.
6. Multiple comparisons: a small pre-registered primary family per dataset, Holm
   correction within the family, and all secondary comparisons labelled exploratory.
7. Seeds: 42 (42–46 for repeats) fixed and recorded; no result may be reported from an
   unrecorded seed.

---

## 10. Leakage prevention

| Leakage channel | Control |
|---|---|
| Participant/speaker crossing folds | group key on participant (EATD) or duplicate-row grouping (AVDD); `assert_split_integrity` reuse; per-fold assertion that train ∩ eval = ∅ |
| Locked test set used for selection | Phase 5B–5D never read `split == "test"`; §4.9 pre-registration before any test read |
| Preprocessing fitted on evaluation data | vectorizer/scaler/imputer/PCA/column filter constructed inside the training part only |
| Threshold tuned on reported set | thresholds from inner OOF only; distribution of chosen thresholds reported |
| Calibrator tuned on reported set | fitted on inner OOF only |
| Feature selection on full data | selection inside folds; per-fold selection sets exported for audit |
| Multiple peeking at outer folds | fixed repeat seeds, fixed metric list, no re-runs after seeing fold results |
| Duplicate content across folds | EATD duplicate transcript/audio check inherited from Phase 1 (0 / 486) **[V]**; AVDD duplicate-row grouping check (§6.5) |
| Label leakage through paths/Filenames | EATD paths contain no label; AVDD label lives in a separate table joined by id |
| Duration/length as a proxy for label | explicit duration/length-only confound baselines (E-L2, E-A3b) |
| Corpus/partition artefacts (EATD `train/` vs `validation/` source folders) | source partition recorded as a covariate; check that folds are not separable by partition |
| Documentation drift | counts in documents are re-verified against artifacts in the final review (§12) |

---

## 11. Artifact contract for Phase 5B onward

Each run writes `results/phase5/<dataset>/<experiment_id>/`:

| File | Contents |
|---|---|
| `config.json` | fully resolved config, grid, seeds, folds |
| `folds.json` | per-fold participant/sample ids, class counts, chosen hyperparameters, chosen threshold |
| `metrics.json` | per-fold and aggregate metrics with CIs, unit and n |
| `predictions_oof.json` | out-of-fold predictions keyed by participant/sample id |
| `metadata.json` | dataset version, label rule, split version, git commit, environment, `clinical_validation: false`, `serving: false` |

`metadata.json` reuses the Phase 2 schema from `ml/phase2/evaluation.py::experiment_metadata`
so Phase 2 and Phase 5 results are machine-comparable **[V]**. No Phase 5 artifact is
written under `models/experiments/` or `models/final/`.

---

## 12. Phase 5B–5F execution order and definition of done

| Stage | Work | Done when |
|---|---|---|
| **5A** (this) | protocol, matrix, statistical plan, risk register, inventory | documents reviewed; counts verified against artifacts |
| **5B** | shared Phase 5 evaluation harness: group K-fold/repeated nested CV, participant aggregation, threshold policy, calibration, CIs, artifact writer; plus E-L1 label table verification | harness unit-tested on synthetic data with known leakage traps; runs without touching `split=="test"` |
| **5C** | EATD text replication/extension: E-T1, E-T2, (E-T3 cost permitting), feasibility baseline E-L2 | per-fold results + dispersion for 25 folds; no test-set access |
| **5D** | EATD voice: E-V1, E-V2, aggregation sensitivity E-V3 | same; includes the negative-result branch honestly |
| **5E** | AVDD: feature cache build, E-A1, E-A2, E-A3, E-A6; fusion gate evaluated | per-subset sample-level results; §6.7 signed off |
| **5F** | gated fusion (E-F1 / E-A4) and, if and only if §4.9 conditions hold, one pre-registered locked-test confirmation | gate decision documented either way |

Nothing in 5B–5F may proceed on a different evaluation unit than the matrix states.

---

## 13. What Phase 5 must not claim

- Not a diagnosis, not a screening tool, not clinically validated, not deployable.
- Not "detects depression from voice/face" — the achievable claims are about
  *classification of a self-report symptom band in a small research corpus* and, for
  AVDD, *sample-level separability of pre-extracted features*.
- Not cross-dataset generalisation, since no second verified-label dataset exists.
- Not multimodal participant-level evidence, since DAIC-WOZ is not held.
- Not a threshold decision, since §5.6 is unmet.
