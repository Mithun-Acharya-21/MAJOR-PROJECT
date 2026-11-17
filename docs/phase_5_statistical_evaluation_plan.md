# Phase 5 — Statistical Evaluation Plan

Scope: how every Phase 5 metric is computed, over what unit, with what uncertainty, and
under what decision rules. Applies to EATD (participant level) and AVDD-TCMI (sample
level). The locked EATD test set is not part of model selection at any point.

---

## 1. Metric set and what each one is for

| Metric | Role | Why / caveat |
|---|---|---|
| **ROC-AUC** | primary, threshold-free | Comparable to Phase 2. Insensitive to class prior, which is good for cross-study comparison and bad for deployment relevance |
| **PR-AUC** (average precision) | primary, threshold-free | The only global metric that visibly degrades under imbalance. At 14.6 % prevalence (AVDD SDS) or 32.7 % (EATD), a random classifier's PR-AUC equals the prevalence, so the "chance floor" differs per experiment and must be printed |
| **Balanced accuracy** | secondary | Mean of sensitivity and specificity; interpretable under imbalance; used for threshold selection |
| **Macro F1** | secondary | Reported for continuity with Phase 2, but it is threshold-dependent and, with 9 positives in a fold, dominated by a single participant |
| **Precision** | secondary | Highly unstable at low prevalence: with 8–9 positives per fold, precision moves in steps of ~0.05–0.1 |
| **Recall / sensitivity** | secondary | The clinically salient direction; never reported alone |
| **Specificity** | secondary | Paired with sensitivity at a fixed operating point |
| **Sensitivity at specificity ≥ 0.80 / ≥ 0.90** | secondary, imbalance-aware | The most decision-relevant summary for a screening-style use case, because it fixes the false-positive budget. Reported with CIs from the same resampling unit |
| **MCC** | secondary | Single balanced summary that is not inflated by class skew |
| **Confusion matrix** | mandatory at the declared threshold | The only honest way to show what an F1 of 0.18 actually was (Phase 2 T1: 1 TP / 2 FP / 7 FN / 14 TN) |
| **Brier score** | calibration | Proper scoring rule; comparable across models on the same unit |
| **ECE (10 equal-frequency bins)** | calibration, descriptive | n per fold is ~28 participants, so bin counts are 2–3. Report, do not conclude |
| **Calibration slope / intercept** | calibration | Slope < 1 indicates over-confidence; more informative than ECE at small n |
| **Prevalence / chance floors** | mandatory context | Random ROC-AUC = 0.5; random PR-AUC = prevalence. Both printed per experiment |

### 1.1 Which metrics are most informative here

Under class imbalance **and** small samples (EATD dev: 138 participants / 45 positives;
one outer fold ≈ 28 participants / 9 positives):

1. **ROC-AUC with its CI** — most stable of the threshold-free metrics at this n, and the
   only one directly comparable to the frozen Phase 2 numbers.
2. **PR-AUC with its CI** — because it makes the low-prevalence problem visible; but its
   fold-to-fold SD at n≈28 is large, so differences under ~0.10 are not interpretable.
3. **Sensitivity at fixed specificity** — the metric a reader actually cares about, and
   robust to the "F1 went up because the threshold moved" artefact that produced T2b's
   higher F1 in Phase 2.
4. **Balanced accuracy / MCC** — for a single-number ranking that is not accuracy-inflated.

**Not informative and not to be headlined:** accuracy (majority class ≈ 0.67 on the EATD
dev pool), F1 at 0.5 without the confusion matrix, and any metric quoted without n.

For AVDD (SDS: 948 samples / 138 positives, 14.6 %), the same ranking holds, with the
extra warning that sample-level n = 948 makes CIs look narrow while speaker clustering is
unknown — the interval is narrower than the evidence.

---

## 2. Confidence intervals

### 2.1 Method

Percentile bootstrap (§2.3), 2,000 resamples, fixed seed 0, resampling **units, not rows**.

| Dataset | Resampling unit | Implemented by |
|---|---|---|
| EATD | participant cluster (all 3 sessions travel together) | `ml/phase2/evaluation.py::participant_bootstrap_ci` (reused) |
| AVDD | sample (no participant id exists) | new, same signature |

Stratification: resample within class so both classes survive each replicate; report the
number of valid replicates. If fewer than 95 % of replicates yield a defined metric, the
metric is reported as `null` with a reason rather than as a number (the Phase 2 harness
already does this for undefined AUC).

### 2.2 What the CI covers — stated explicitly in every table

| Reported estimate | CI construction |
|---|---|
| Single repeat, pooled OOF | participant/sample bootstrap on the pooled outer-OOF predictions |
| Mean over 5 repeats | bootstrap **within** repeat, then take the repeat-mean of the bootstrap distribution; the interval thus includes between-fold variability |
| Per-fold metric | no CI; folds are reported with their own n and class counts |
| Difference between two models | **paired** bootstrap: same resampling units, same folds, difference recomputed per replicate; the CI is on the difference, not on each metric separately |

### 2.3 Why not DeLong / t-tests

DeLong assumes independent observations; EATD has three correlated sessions per
participant and AVDD may have repeated speakers. A clustered bootstrap is the defensible
choice. No p-values are computed for model-vs-model comparisons at this n; paired
bootstrap CIs on differences are reported instead, and "overlapping CIs" is never used as
evidence of no difference.

### 2.4 Interpretive rule

An interval containing 0.5 (ROC-AUC) or the prevalence (PR-AUC) is reported as **"not
distinguishable from chance"**, with the point estimate still shown. Phase 2's five models
all fall in this category; Phase 5 must not describe a CI-overlapping-0.5 result as a
trend, a near-significant finding, or a promising direction.

---

## 3. Cross-validation reporting

For every experiment:

1. **Per-fold table** — fold, repeat seed, n train / n eval units, positives in each,
   chosen hyperparameters, chosen threshold, all metrics.
2. **Aggregate table** — mean, SD, min, max across the 25 outer folds, plus the five
   repeat-level means so between-repeat stability is visible.
3. **Selected-hyperparameter table** — how often each grid point won, per modality. A
   hyperparameter that wins in 3 of 25 folds tells you the search is noise.
4. **Selected-threshold distribution** — median, IQR, range. If the IQR spans more than
   ~0.15 in probability, the operating point is not identifiable and must be reported that
   way.
5. **Per-class-count folds** — any fold with fewer than 5 positives is flagged; a metric
   computed on fewer than 5 positives is not pooled into an aggregate claim.

Aggregation across folds: the primary summary is the **mean of per-fold metrics** (equal
fold weight), not a metric computed on pooled predictions, because pooling gives more
weight to folds with more participants. Pooled values are reported as a secondary column
so the difference is visible.

---

## 4. Threshold selection policy

| Stage | Rule |
|---|---|
| Selection data | inner-fold OOF probabilities only (never outer-val, never locked test) |
| Primary criterion | maximise balanced accuracy |
| Alternative criteria (reported, same folds) | Youden's J; sensitivity at specificity ≥ 0.80; fixed 0.5 |
| Application | chosen threshold frozen, applied unchanged to the outer fold |
| Reporting | results at chosen threshold **and** at 0.5, both with confusion matrices |
| Prohibition | no threshold may be chosen, adjusted, or re-selected after any metric on the reporting fold or the locked test set is seen |

Rationale recorded for the record: Phase 2's T2b "best F1" (0.526) was 5 TP / 6 FP at 0.5
while T1's F1 (0.182) was 1 TP / 2 FP. Those numbers describe threshold placement on 24
participants, not model quality; Phase 5 removes that ambiguity by selecting the threshold
inside the CV and reporting both values.

---

## 5. Calibration

| Item | Policy |
|---|---|
| Primary calibrator | Platt/sigmoid, fitted on inner OOF logits |
| Secondary | isotonic (sensitivity check only — 45 positives is too few) |
| Evaluated by | Brier, ECE (10 equal-frequency bins), calibration slope and intercept |
| Data | outer-fold predictions, uncalibrated and calibrated, reported side by side |
| Rule | if calibration slope CI includes 1 and intercept CI includes 0, report "no detectable miscalibration at this n" rather than "well calibrated" |
| Serving boundary | `models/calibration.json` belongs to the application and is **not** modified or replaced by Phase 5 research calibration |

---

## 6. Class imbalance handling

| Option | Status | Note |
|---|---|---|
| No weighting | baseline arm | keeps comparability with Phase 2 |
| `class_weight="balanced"` | candidate arm | changes calibration; must be calibrated afterwards |
| Majority-class undersampling (train part only) | candidate arm | discards data; with 93/45 controls already limited |
| Threshold selection instead of resampling | candidate arm | preferred when the ranking is the useful signal |
| SMOTE / synthetic oversampling | **excluded** | 414 correlated sessions, 45 positives, 1,024-d feature spaces: interpolated points are not evidence. Documented refusal |
| Metric reweighting | n/a | prefer reporting imbalance-aware metrics rather than reweighting a misleading one |

The arm is part of the pre-registered configuration; it is not chosen after seeing outer
results. Arms are compared with paired bootstrap on identical folds.

---

## 7. Multiple comparisons

- Each dataset has a **pre-registered primary family**: at most three primary comparisons
  (e.g. EATD: E-T1 vs E-L2 feasibility baseline, E-V1 vs baseline, textual-vs-acoustic
  difference).
- Within a family: Holm–Bonferroni on paired-bootstrap difference intervals (or on the
  corresponding two-sided bootstrap p-value), α = 0.05.
- Everything outside the primary family is **exploratory** and labelled so: aggregations
  (E-V3), functionals (E-A6), ablations (E-A3), pooled subsets (E-A5), calibration variants.
- No "best of 25 folds" selection. The reported number is the mean over folds, never the
  maximum.
- Number of experiments is fixed by the matrix; adding one after seeing results requires a
  written amendment to this file before running it.

---

## 8. Random seed policy

| Purpose | Seed(s) |
|---|---|
| Project canonical | 42 |
| CV repeats | 42, 43, 44, 45, 46 (declared before running) |
| Bootstrap | 0 (matches Phase 2 comparison report) |
| Model init / fine-tune | 42, recorded per run |
| Hook/transformer caching | deterministic; cache key includes encoder, chunk/length settings |

Rules: every seed is written into `config.json` and `metadata.json`; a result produced
under an unrecorded seed is not reportable; seeds are never changed to improve a number;
single-seed results are labelled "single seed" exactly as Phase 2 did.

---

## 9. Minimum group size and estimability guards

| Guard | Trigger | Action |
|---|---|---|
| Fold positive count | < 5 positives in an evaluation fold | metric reported per fold but excluded from the aggregate claim |
| Undefined metric | bootstrap replicate has one class, or no positive predictions | report `null` + reason; do not impute 0 or 0.5 |
| ECE bins | bin count < 2 observations | ECE suppressed for that fold |
| Locked-test estimability | n positives < 10 | test-set metric is declared **not estimable** (relevant to the SDS ≥ 53 branch: 3 positives) |
| PR-AUC floor | prevalence printed next to every PR-AUC | a PR-AUC is never compared across datasets without its prevalence |

---

## 10. Reporting template (mandatory fields per result row)

```
dataset        : eatd | avdd_tcmi_hds | avdd_tcmi_sds | ...
label_rule     : sds_index_ge_50 | sds_index_ge_53 | released_hds_label | released_sds_label
modality       : text | voice | acoustic | visual | fusion
experiment_id  : E-T1 | E-V2 | E-A3 | ...
evaluation_unit: participant | sample
n_units        : 138
n_positive     : 45
prevalence     : 0.326
folds          : 5 x 5 (seeds 42-46)
aggregation    : mean of 3 session probabilities
threshold      : 0.61 (inner-OOF balanced accuracy) ; also reported at 0.50
ci_method      : participant-cluster percentile bootstrap, 2000 resamples, seed 0
roc_auc        : 0.68 [0.57, 0.78]
pr_auc         : 0.48 [0.34, 0.61]   (prevalence floor 0.33)
...
clinical_validation: false
serving            : false
```

A table row missing `evaluation_unit`, `n_units` and `n_positive` is a defect, not a
summary.

---

## 11. Statistical honesty checks to run in Phase 5B

1. **Leakage probe** — train a model on a deliberately permuted label; every metric must
   land at chance (ROC-AUC ≈ 0.5, PR-AUC ≈ prevalence). If it does not, the pipeline is
   leaking.
2. **Baseline probe** — majority-class and single-feature (duration, frame count) baselines
   must be computed before any modality claim.
3. **Fold symmetry probe** — verify each participant appears exactly once per repeat in
   the evaluation parts, summing to 138.
4. **Cache probe** — recompute one sample's aggregated features and compare to the cache.
5. **Test-set firewall probe** — assert that no Phase 5 selection script opened
   `split == "test"` rows (`assert` on split label inside the loader, mirroring
   `assert_split_integrity`).
