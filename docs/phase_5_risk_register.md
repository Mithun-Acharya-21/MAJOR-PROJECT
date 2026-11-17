# Phase 5 — Risk Register

Scope: risks to the scientific validity, reproducibility, and integrity of Phase 5.
Likelihood/impact: L / M / H. Status: `open`, `mitigated` (control in place), `accepted`
(cannot be removed, disclosed), `blocked` (waiting on an external party).

Risks are ordered by scientific severity, not by likelihood.

---

## A. Validity risks introduced by Phase 2's evidence

| ID | Risk | L | I | Mitigation | Status |
|---|---|---|---|---|---|
| R-01 | **Phase 2 conclusions are noise.** Every 95 % CI on the locked test set includes 0.5; the test set is 24 participants with 8 positives. Repeating the same protocol cannot resolve this. | H | H | Phase 5 does not extend the test-set leaderboard. It estimates from 138 dev participants with repeated nested CV and reports dispersion. Phase 2 numbers are cited as frozen, not re-derived. Expected outcome: wide intervals persist. | `accepted` |
| R-02 | **Selection optimism was the whole validation→test gap** (0.195–0.414 ROC-AUC). | H | H | Selection moves entirely inside the CV; thresholds and calibrators fitted on inner OOF; per-fold dispersion reported; no post-hoc choice of aggregation. | `mitigated` |
| R-03 | **Threshold-driven metric inflation.** T2b's 0.526 F1 was 5 TP / 6 FP; T1's 0.182 F1 was 1 TP / 2 FP. | H | M | Threshold selected on inner OOF, frozen before outer evaluation; results at 0.5 shown alongside; confusion matrix mandatory. | `mitigated` |
| R-04 | **Voice "below chance" is over-interpreted.** V1 (0.383) and V2 (0.398) are reported as negative results; at n = 24 they are also uninformative. | M | H | E-V1/E-V2/E-V3 rerun the branches under repeated nested CV; the report states the negative result and the uninformative result as separate statements. | `mitigated` |
| R-05 | **EATD class-imbalance shortcut** — accuracy 0.67 is available from the majority class. | H | M | Accuracy is never headlined; balanced accuracy/PR-AUC/MCC are primary; prevalence printed next to PR-AUC. | `mitigated` |

## B. Dataset and label risks

| ID | Risk | L | I | Mitigation | Status |
|---|---|---|---|---|---|
| R-06 | **EATD label threshold mismatch.** Project uses SDS ≥ 50 (53/109) while the authors' README totals (30/132) are reproduced exactly by ≥ 53. 23 participants sit in the 50.00–52.50 band. Results are not comparable across rules. | H | H | Labels preserved; alternative threshold documented in a separate derived table; every metric prints `label_rule`; no rule chosen on performance; §5.6 evidence list required. | `open` (owner decision) |
| R-07 | **Switching to SDS ≥ 53 destroys test-set estimability** — the locked test would hold 3 positives, so ROC-AUC/PR-AUC are not estimable within the plan's guard (R-09). | H | H | Lock preserved; re-stratification prohibited; if ≥ 53 is adopted, the test metric is declared not estimable and the phase reports dev-only estimates. | `accepted` |
| R-08 | **AVDD speaker leakage is undetectable.** No participant metadata; several samples may come from one person, so sample-level CIs are overstated. | M | H | Sample-level language throughout; interval caveat string mandated; duplicate/near-duplicate aggregated-row grouping before CV; no participant-level claim. | `accepted` |
| R-09 | **AVDD label semantics are thin.** HDS 159/123 and SDS 810/138 come from released label files; the instrument behind the binary code is not documented, and HDS vs SDS may not share a label rule. | M | H | Subsets reported separately; pooling is a separate gated experiment (E-A5); label codes preserved verbatim; no comparison with EATD SDS or PHQ-8. | `accepted` |
| R-10 | **AVDD feature dictionary absent.** 1,024 acoustic columns are header-less `f0..f1023`; no extractor is documented, so coefficient interpretation is impossible. | H | M | Results limited to predictive performance; no acoustic-parameter interpretation; recorded in the §6.7 review gate. | `accepted` |
| R-11 | **AVDD visual signal may be duration/identity/pose, not expression.** 484 of 533 retained columns are raw landmark coordinates (only 49 are AU/gaze/pose). | M | H | Documented column policy (no automatic removal); metadata/quality/landmark ablations (E-A3); frame-count-only and confidence-only baselines. | `mitigated` |
| R-12 | **Harvard label mapping unresolved** (58,405 × `0` vs 41,185 × `1`; no user id). | H | H | Supervised use blocked outright; descriptive work only; if a binary metric is ever produced, both mapping directions must be reported. | `blocked` (depositor) |
| R-13 | **Cross-dataset merging temptation.** EATD SDS, AVDD HDS/SDS, PHQ-8 (DAIC) and self-reported diagnosis (RSDD/SMHD) measure different constructs with different units. | M | H | Merging prohibited in the protocol; only within-dataset 0/1 with the rule printed; cross-dataset statements limited to ranking with label rules shown. | `mitigated` |
| R-14 | **No genuine multimodal participant-level benchmark exists yet** (DAIC-WOZ not acquired). Any multimodal claim today rests on pre-extracted AVDD features with no participant identity. | H | H | DAIC-WOZ prioritised as the acquisition target; Phase 5 documents state that multimodal participant-level evidence does not exist yet. | `blocked` (DUA) |

## C. Methodological / statistical risks

| ID | Risk | L | I | Mitigation | Status |
|---|---|---|---|---|---|
| R-15 | **Fold leakage via session-level operations** — a vectorizer, scaler or PCA fitted over all sessions before splitting. | M | H | Fit-inside-fold rule with a per-component table; leakage probe on permuted labels in Phase 5B. | `mitigated` |
| R-16 | **Too many comparisons across 25 folds × 5 models × 4 aggregations.** | H | M | Pre-registered primary family per dataset; Holm within family; everything else labelled exploratory; experiment count fixed by the matrix. | `mitigated` |
| R-17 | **Non-identifiable operating point** — thresholds varying widely across folds. | M | M | Threshold distribution reported (median/IQR/range); wide IQR is reported as a result, not hidden. | `mitigated` |
| R-18 | **ECE misleads at small n** (bins of 2–3 participants per fold). | H | M | Calibration slope/intercept preferred; ECE suppressed when a bin holds < 2 observations; no bin-level claim under n = 200. | `mitigated` |
| R-19 | **PR-AUC floors misread across experiments** (prevalence 0.33 on EATD dev vs 0.15/0.44 on AVDD SDS/HDS). | M | M | Prevalence printed with every PR-AUC; cross-dataset PR-AUC comparison prohibited. | `mitigated` |
| R-20 | **Cost-driven shortcuts** — eGeMAPS extraction is CPU-bound (183 s per pass; 192 s per fit in Phase 2), pushing towards fewer folds or reused caches fitted over all data. | M | M | Cache is per-session raw features only (never fit-dependent); grids fixed; if cost forces fewer repeats, the reduction is declared in the report rather than silently applied. | `open` |
| R-21 | **AVDD compute/IO risk** — ~34 GB decompressed per full pass against 32 GB free; ZIP streaming required. | M | M | Build one aggregated feature cache, hash it, and run all experiments from the cache; no extraction, no JPG extraction. | `mitigated` |
| R-22 | **Duration/length confound** in EATD (three prompts of different lengths) and AVDD (frame counts 136–6,860). | M | H | Explicit duration/frame-count-only baselines (E-L2, E-A3); duration features excluded from primary model inputs. | `mitigated` |
| R-23 | **Single-seed results presented as stable.** | M | M | Five declared repeat seeds; single-seed results labelled; seeds recorded in metadata. | `mitigated` |

## D. Process, provenance and integrity risks

| ID | Risk | L | I | Mitigation | Status |
|---|---|---|---|---|---|
| R-24 | **Locked test contamination** — a later convenience run reading `split == "test"`. | M | H | Firewall probe in §11 of the stats plan; §4.9 pre-registration before any test read; the test split stays at 24 participants. | `mitigated` |
| R-25 | **Phase 2 artifacts overwritten** by a Phase 5 script writing into `results/phase2/`, `models/experiments/` or `models/final/`. | L | H | Phase 5 writes only under `results/phase5/`; artifact contract in the protocol §11; existing Phase 2 integrity tests retained. | `mitigated` |
| R-26 | **Documentation drift** — numbers in Phase 5 documents diverging from artifacts. | M | M | Phase 5A numbers re-verified from the named files; the inventory JSON restates dataset counts from `data/reports/dataset_inventory.json`; the final review re-checks each cited count. | `mitigated` |
| R-27 | **Clinical over-claim** in narrative, slides or the app ("AI detects depression"). | M | H | Every artifact carries `clinical_validation: false`, `serving: false`; §13 of the protocol lists prohibited claims; SDS is described as a self-report symptom band. | `mitigated` |
| R-28 | **Licence/redistribution** — EATD has no declared licence (upstream 404); AVDD is CC-BY-NC-4.0 (non-commercial); DAIC/RSDD/SMHD DUAs forbid redistribution. | M | H | Aggregate-only reporting; no corpus content in reports or demos; no audio/transcripts/posts published; NC terms respected. | `open` (EATD author confirmation) |
| R-29 | **Two Python interpreters on the machine** (`requirements.txt` / `docs/environment.md` warn). Results produced by the wrong interpreter are not comparable. | M | M | All Phase 5 runs use `.venv/Scripts/python.exe`; environment recorded in `metadata.json` (Phase 2 schema). | `mitigated` |
| R-30 | **Git hygiene** — pre-existing unrelated working-tree changes being swept into a Phase 5 commit. | M | M | Phase 5A touches only the five new documents plus one derived CSV; unrelated changes are neither staged nor reverted. | `mitigated` |
| R-31 | **Backend/frontend coupling.** A Phase 5 research result leaking into the serving path. | L | M | No Phase 5 artifact path is read by `model_registry`; the existing test asserting Phase 2/5 artifacts live outside `TEXT_MODEL_DIR` applies. | `mitigated` |

---

## E. Residual risk statement

After mitigation, three risks remain materially open and must be disclosed in any Phase 5
report:

1. **R-01/R-02** — the underlying sample size (138 dev participants, 45 positives) cannot
   support more than order-of-magnitude statements about signal.
2. **R-06/R-07** — the project has two defensible label rules whose class distributions
   differ by 23 participants in exactly the band where borderline cases live, and the
   locked test set cannot arbitrate between them.
3. **R-08/R-14** — no participant-level multimodal data is held, so no multimodal
   participant-level claim can be made at all.
