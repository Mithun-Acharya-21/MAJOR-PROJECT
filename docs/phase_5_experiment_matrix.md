# Phase 5 — Experiment Matrix

Companion to `docs/phase_5_research_protocol.md`. Machine-readable form:
`data/reports/phase_5_experiment_inventory.json` (same experiment IDs).

Readiness values:

| Value | Meaning |
|---|---|
| `READY` | all inputs verified and local; can start in Phase 5B without new data |
| `READY_WITH_CONSTRAINTS` | runnable, but a documented interpretation caveat or decision gate applies |
| `GATED` | runnable only if a stated scientific gate is passed by earlier experiments |
| `BLOCKED` | cannot run: missing source evidence or missing data |
| `NOT_AVAILABLE` | dataset not held |

Every EATD experiment uses `label_rule = sds_index_ge_50` unless the row says otherwise.
Every EATD row's evaluation pool is **train + validation only (138 participants, 414
sessions, 45 positive)**; the locked 24-participant test set is not read.

---

## 1. EATD — text

| Field | E-T1 | E-T2 | E-T3 |
|---|---|---|---|
| Dataset | EATD | EATD | EATD |
| Modality | text (3 prompted transcripts) | text (frozen multilingual transformer) | text (top-2-layer fine-tune) |
| Research question | RQ1: is a lexical baseline above chance under honest estimation? (replicates Phase 2 T1) | RQ1: does a frozen contextual encoder + head carry signal? (replicates T2a) | RQ1: does limited fine-tuning change the picture? (replicates T2b) |
| Evaluation unit | participant (mean of 3 session probabilities) | participant | participant |
| Split strategy | repeated nested stratified group 5-fold × 5 repeats, groups = participant, seeds 42–46; inner 4-fold for grid/threshold/calibration | same | same |
| Baseline models | majority class; SDS-free duration/length-only logistic (E-L2); Phase 2 T1 point estimate cited, not re-fit | majority class; E-T1 as reference | E-T1, E-T2 |
| Candidate models | TF-IDF (CJK char + word/char_wb n-grams) + LogisticRegression (Phase 2 grid, unchanged); imbalance variants: none / balanced / undersample | DistilBERT-multilingual frozen embeddings + linear head | top-2-layer fine-tune, AMP, same grid and epochs as Phase 2 |
| Metrics | ROC-AUC, PR-AUC (primary); macro F1, sensitivity, specificity, balanced accuracy, MCC, confusion matrix at selected threshold; Brier, ECE, calibration slope | same | same |
| Leakage risks | vectorizer/idf fitted across folds; duplicate transcripts; threshold chosen on reported fold | embedding cache keyed wrongly (session↔participant mix-up); pooled normalisation across folds | fine-tune seeing outer-val; early stopping on outer-val |
| Current blockers | none | none | GPU budget (Phase 2: 26 s/fit ⇒ ~25 fits ≈ manageable, but 5 repeats × 5 folds × 4 inner = 100 fits; must be costed) |
| Readiness | `READY` | `READY` | `READY_WITH_CONSTRAINTS` (cost) |

## 2. EATD — voice

| Field | E-V1 | E-V2 | E-V3 | E-V4 |
|---|---|---|---|---|
| Dataset | EATD | EATD | EATD | EATD |
| Modality | voice (openSMILE eGeMAPS v02, 88-d) | voice (frozen Mandarin wav2vec2) | voice (aggregation study) | voice (acoustic QC/confound) |
| Research question | RQ2: do functionals carry SDS-band signal? (replicates V1) | RQ2: do frozen speech embeddings? (replicates V2) | RQ6: is the negative result an artifact of mean pooling? | is any signal explained by duration/energy/quality alone? |
| Evaluation unit | participant | participant | participant | participant |
| Split strategy | same repeated nested group CV | same | same folds as E-V1/E-V2 | same |
| Baseline models | majority class; duration-only logistic | majority class; duration-only logistic | E-V1/E-V2 with mean pooling as reference | transcript length + audio duration only |
| Candidate models | eGeMAPS + LR (median-impute + standardise in-fold); imbalance variants | wav2vec2 mean+std frame pooling, 8 s chunks (≤12), chunk-mean, + linear head | mean / max / median / duration-weighted aggregation of the same per-session probabilities | LR on (duration, mean energy, SNR proxy) |
| Metrics | as E-T1 | as E-T1 | same metrics per aggregation; paired differences across identical folds | ROC-AUC, PR-AUC |
| Leakage risks | eGeMAPS cache built over all sessions and split later; 0.48 s file handled inconsistently between train and eval; chunk overlap across participants | chunking introducing cross-session frames; cache key mismatch (encoder/chunk/len) | selecting the aggregation that looks best | none beyond fold leakage |
| Current blockers | CPU cost: extraction 183 s per pass, LR fit 192 s (Phase 2) ⇒ nested CV is expensive but feasible with the existing cache | none | depends on E-V1/E-V2 completing | none |
| Readiness | `READY_WITH_CONSTRAINTS` (cost, pooled negative result expected) | `READY` | `READY_WITH_CONSTRAINTS` (runs after E-V1/E-V2) | `READY` |

## 3. EATD — label definition and fusion

| Field | E-L1 | E-L2 | E-F1 |
|---|---|---|---|
| Dataset | EATD | EATD | EATD |
| Modality | text + voice, label axis | n/a (metadata only) | text + voice fusion |
| Research question | RQ7: what changes under SDS ≥ 53, and is the threshold choice justified? | RQ6: how much signal is recording length? | RQ5: are branches complementary? |
| Evaluation unit | participant, both label versions | participant | participant |
| Split strategy | **existing** locked split assignment reused unchanged (no re-stratification); distributions reported per rule; no model fitting in 5A | same folds as E-T1/E-V1 | same folds and repeats as E-T1 + E-V1 |
| Baseline models | n/a (label analysis, not a model) | majority class | best unimodal branch |
| Candidate models | n/a | LR on fold-safe metadata only (durations, char counts, topic proportions) | late fusion: logistic stack on inner-OOF probabilities; early fusion: concatenated standardised features |
| Metrics | class counts, positive rate, flip set (n=23, all 50.00–52.50), estimability of test metrics at 3 positives | ROC-AUC, PR-AUC | ΔPR-AUC, ΔROC-AUC vs best unimodal, correlation of OOF probabilities, error overlap |
| Leakage risks | treating the ≥ 53 labels as a correction of Phase 2 results; re-stratifying and thereby breaking the lock | duration differences driven by topic prompt (confound) | stacking fitted on outer-val; fusion chosen after seeing unimodal numbers on the same folds |
| Current blockers | §5.6 evidence (author label script, norm documentation) is external | none | **fusion gate (§6.6)** — currently expected to fail, since no Phase 2 branch has a CI excluding 0.5 |
| Readiness | `BLOCKED` for choosing a rule; the sensitivity table itself is `READY` | `READY` | `GATED` |

## 4. AVDD-TCMI — sample level

Not participant-level. No participant metadata exists in the release (verified).

| Field | E-A1 | E-A2 | E-A3 | E-A6 | E-A4 | E-A5 |
|---|---|---|---|---|---|---|
| Dataset | AVDD-TCMI (HDS and SDS reported separately) | AVDD-TCMI | AVDD-TCMI | AVDD-TCMI | AVDD-TCMI | AVDD-TCMI |
| Modality | acoustic features (1,024-d × frame functionals) | visual features (538 → 533 after documented removals) | visual ablation + confounds | reduction sensitivity | acoustic + visual fusion | HDS+SDS pooled |
| Research question | RQ3: do acoustic summaries separate the sample labels? | RQ4: do visual summaries carry signal? | RQ4: is any visual signal in the behavioural block (AU/gaze/pose) or in length/quality/identity geometry? | RQ3/4: is the answer an artifact of the functional set? | RQ5: is feature-level fusion justified? | does pooling subsets change conclusions? |
| Evaluation unit | sample | sample | sample | sample | sample | sample |
| Split strategy | stratified 5-fold × 5 repeats (seeds 42–46) **within each subset**; in-fold scaling/PCA only | same | same | same | same folds | stratified by subset × label |
| Baseline models | majority class; frame-count-only (duration proxy) logistic | majority class; frame-count-only logistic; constant-column predictor | frame count; mean `confidence`; landmark-centroid-only | mean/std only as reference | best unimodal branch per subset | per-subset results as reference |
| Candidate models | functionals → in-fold standardise → LR (L2), linear SVM, PCA(≤64)+LR, gradient boosting (small depth) | same family on 533 columns | AU-only (35), AU+gaze+pose (49), landmarks-only (484), metadata-only | percentiles-only; Δ-features; mean/std | late stack on OOF probabilities; early concatenation with in-fold PCA | same model family, subset indicator recorded |
| Metrics | ROC-AUC, PR-AUC, balanced accuracy, sensitivity at 90 % specificity; CIs by sample bootstrap | same | same | same | ΔPR-AUC with paired bootstrap; correlation of OOF probabilities | per-subset metrics + pooled metrics, both with subset printed |
| Leakage risks | 1,024 columns with 948 samples (SDS) ⇒ overfitting read as signal; constants/duplicate frames; **unknown speaker clustering across samples** (intervals may be too narrow); subset id correlating with label | landmark columns encoding identity/pose rather than expression; `frame`/`timestamp` leaking duration | ablations chosen after seeing full-set result (must be pre-declared) | functional selection on reported folds | fusion on 1,230 samples with 261 positives across two label conventions | label semantics assumed shared across HDS and SDS |
| Current blockers | feature cache must be built from the 13.5 GB ZIP (~34 GB decompressed per pass, 32 GB free) — ZIP streaming, no extraction | same | same + §6.7 pre-training review sign-off | same | **fusion gate** | subset label-convention review |
| Readiness | `READY_WITH_CONSTRAINTS` (cache + §6.7 review) | `READY_WITH_CONSTRAINTS` | `READY_WITH_CONSTRAINTS` | `READY_WITH_CONSTRAINTS` | `GATED` | `GATED` (needs a defensible shared-label argument) |

Note: HDS has 282 samples with 123 positives (43.6 %) and SDS 948 with 138 (14.6 %).
The same model family is expected to behave very differently on the two subsets; a
pooled number would hide exactly the effect worth reporting.

## 5. Harvard tweets

| Field | E-H1 |
|---|---|
| Dataset | Harvard English tweets (99,590 records) |
| Modality | text (pre-processed token strings, mean 10.1 tokens) |
| Research question | none supervised. Descriptive/structural: tokenisation, length, vocabulary, duplicate structure |
| Evaluation unit | tweet (no author id exists — grouping impossible) |
| Split strategy | not applicable (no supervised evaluation) |
| Baseline models | n/a |
| Candidate models | n/a |
| Metrics | descriptive statistics only; if any binary metric is ever produced, the assumed mapping must be printed and the complementary-mapping run reported alongside |
| Leakage risks | duplicate texts after normalisation (264) with no author grouping ⇒ any supervised split would leak near-duplicates |
| Current blockers | code→class mapping unresolved (58,405 × `0`, 41,185 × `1`); needs depositor statement or author script |
| Readiness | `BLOCKED` for supervised use; descriptive work only |

## 6. Not-acquired datasets

| Field | E-D1 (DAIC-WOZ) | E-D2 (RSDD) | E-D3 (SMHD) |
|---|---|---|---|
| Dataset status | `ACCESS_PENDING`, 189 sessions **[reported]** | `ACCESS_PENDING`, 116,484 users **[reported]** | `ACCESS_PENDING`, 385,476 users **[reported]** |
| Modality | text + audio + facial (video under DUA) | text | text |
| Research question (on acquisition) | RQ5/RQ6: the only genuine paired multimodal participant-level benchmark | external generalisation of the text branch, user-level | depression-vs-control only, never nine-way |
| Evaluation unit | participant/session | user | user |
| Split strategy | official split if released, else participant-level stratified group CV with recorded seed | official split (3,070/…/…) **[reported]** | official split |
| Baseline models | unimodal Phase 5 models transplanted | Phase 5 text model applied zero-shot, English only | same |
| Candidate models | participant-level fusion, only after DAIC-only unimodal results | logistic/TF-IDF, transformer head | same |
| Metrics | PHQ-8 convention read from released files, not assumed ≥ 10 | ROC-AUC, PR-AUC at user level | same |
| Leakage risks | multiple sessions per participant; audio/video sync assumptions; label-file convention assumed | user posts split across folds | user posts split across folds; nine-class confusion |
| Current blockers | signed DUA + institutional e-mail + ethics position | signed DUA | signed DUA |
| Readiness | `NOT_AVAILABLE` | `NOT_AVAILABLE` | `NOT_AVAILABLE` |

## 7. Counts each experiment must print

Every result row in Phase 5B onward must carry: dataset, label rule, evaluation unit,
n units, n positive units, number of folds, repeat seeds, threshold (and its selection
rule), and the metric's CI method. A row missing any of these is not reportable.
