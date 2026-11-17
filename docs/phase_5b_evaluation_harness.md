# Phase 5B — Evaluation Harness

Date: 2026-11-16
Scope: a reproducible, participant-safe evaluation harness for Phase 5, plus its
leakage probes, locked-test firewall, metric guards, bootstrap interfaces and
artifact metadata contract. Infrastructure only.

This phase **produced no predictions and no model performance numbers**. No model
was trained, no checkpoint was written, no Phase 2 result was touched, and the
EATD manifests, labels and locked splits are unchanged. The only file written
outside code and docs is the split audit at
`data/reports/phase_5b_split_audit.json`.

Protocol references: `docs/phase_5_research_protocol.md`,
`docs/phase_5_statistical_evaluation_plan.md`,
`docs/phase_5_experiment_matrix.md`, `docs/phase_5_risk_register.md`.

---

## 1. What the harness is for

Phase 5 asks whether a multimodal model beats text-only and voice-only baselines
on EATD. That question is only answerable if the evaluation cannot leak, cannot
quietly tune on the wrong partition, and cannot report a number whose meaning is
ambiguous. Phase 5B builds exactly those guarantees, so that Phase 5C can plug in
a `predict` callback and get trustworthy numbers without re-deciding anything.

Non-goals, stated explicitly because they were easy to get wrong:

- no model fitting, no hyperparameter search, no checkpoint;
- no probability, metric or interval is invented anywhere in the harness;
- AVDD's sample-level resampling is documented as ignoring participant
  clustering, not silently treated as equivalent to EATD's;
- the locked test set is never read — the firewall holds participant keys only.

---

## 2. Module map

| Path | Purpose |
|---|---|
| `ml/evaluation/provenance.py` | Partition vocabulary (`inner_train`, `inner_validation`, `outer_evaluation`, `locked_test`), operation vocabulary, immutable `Provenance` records, `assert_fit_on_training`, append-only `ProvenanceLog`. |
| `ml/evaluation/splits.py` | `Unit`/`Fold`/`Repetition`, integrity assertions, `repeated_stratified_group_folds`, `inner_group_folds`, EATD development/locked unit builders, `audit_eatd_development_data`. |
| `ml/evaluation/nested_cv.py` | `PredictionRequest`, `OutOfFoldPredictions`, `generate_oof_predictions`, `generate_inner_oof`; enforces the prediction callback contract and stamps provenance on every OOF block. |
| `ml/evaluation/metrics.py` | Phase 2 functions unchanged; appended guarded `MetricResult` layer (`compute_evaluation_metrics` and the individual `*_metric` helpers). |
| `ml/evaluation/thresholds.py` | Threshold policies (`balanced_accuracy` primary; `youden_j`, `sensitivity_at_specificity`, `fixed` alternatives), `select_threshold`, `reference_threshold`, `apply_threshold`. |
| `ml/evaluation/calibration.py` | `CalibrationPlan`/`plan_calibration`/`apply_calibration` interfaces; `fit_calibrator` deliberately raises until Phase 5C. |
| `ml/evaluation/bootstrap.py` | `BootstrapConfig`, `participant_cluster_bootstrap`, `sample_bootstrap`, `bootstrap_for_dataset`, `aggregate_by_participant`. |
| `ml/evaluation/firewall.py` | `LockedTestFirewall`, `LockedTestAccessRecord`, `LockedTestAccessError`. |
| `ml/evaluation/artifacts.py` | `build_experiment_artifact`, `validate_experiment_artifact`, `write_experiment_artifact`, `collect_environment`, `sha256_of_files`, `git_revision`. |
| `ml/evaluation/config.py` | Typed loader for `ml/configs/phase5/evaluation.yaml`. |
| `ml/configs/phase5/evaluation.yaml` | Declared seeds, folds, bootstrap and threshold policy, guards, artifact and locked-test rules. |
| `ml/configs/phase5/experiment_artifact.schema.json` | Published JSON Schema (draft 2020-12) for the artifact contract. |
| `scripts/phase5/build_split_audit.py` | Builds `data/reports/phase_5b_split_audit.json` (refuses to overwrite without `--force`). |

---

## 3. Data and split protocol (Tasks A–C)

Verified against the real manifests, not assumed:

| Quantity | Value |
|---|---|
| EATD participants | 162 |
| Development participants | 138 (114 train + 24 validation) |
| Locked-test participants | 24 (8 positive) |
| Development sessions | 414 (3 per participant) |
| Development positives / negatives | 45 / 93 |

Group key is `(source_partition, subject_id)`, e.g. `eatd_train_58` and
`eatd_validation_58` are different people even though both have `subject_id`
58. 45 numeric subject ids collide across partitions; the audit reports them and
the splitter refuses units whose group key is not partition-qualified.

Outer procedure: five repeats, seeds 42–46, `StratifiedGroupKFold(n_splits=5,
shuffle=True, random_state=seed)` over participant groups. Each fold yields
≈110 training participants / 28 evaluation participants (≈9 positives), with
class stratification at the participant level. Every fold records train and
eval unit ids, group ids, participant/session counts, positive/negative counts
and the realised class distribution.

Inner procedure: four folds, built **only** from the outer training partition,
with the inner seed equal to the outer repeat seed (declared in the config, never
tuned). Outer evaluation participants are excluded structurally: `inner_group_folds`
raises if any inner member belongs to the outer evaluation set.

**No fallback splitter exists.** When `StratifiedGroupKFold` cannot produce a
usable split (for example a single-class evaluation fold), the harness raises
`SplitError` with the exact seed, fold and class counts and a message ending
"No fallback splitter is used." Random or ungrouped splitting is never
substituted.

### Split audit

`data/reports/phase_5b_split_audit.json` records, for all 25 outer folds: the
seed, fold index, both partitions' participant/unit ids, counts, positives,
negatives, overlap checks (empty), inner-fold summaries and named limitations.
Status: `verified`; full coverage; `scikit_learn_version` recorded. Regenerate
with `./.venv/Scripts/python.exe scripts/phase5/build_split_audit.py` (add
`--force` to replace).

---

## 4. Threshold and calibration policy (Task D)

- OOF probabilities come from an explicit callback: `predict(request) ->
  Sequence[float]`. The request carries `train_unit_ids`, `eval_unit_ids`, the
  seed and fold, so the callback cannot pick its own data.
- The callback contract is enforced: wrong length, non-finite values, values
  outside `[0, 1]`, or mutating the request raise `PredictionContractError`.
- Thresholds are selected on **inner** OOF data only. `select_threshold` takes a
  `Provenance`; `outer_evaluation` and `locked_test` provenance is rejected with
  `ProvenanceError`, before any arithmetic happens.
- The primary policy is `balanced_accuracy`. The 0.5 reference is computed by
  `reference_threshold` and reported separately, never as if it were tuned.
- Calibration interfaces exist (`platt`, `isotonic`) but `fit_calibrator` raises
  `CalibrationNotImplementedError`: Phase 5B defines the interface and forbids
  fitting, so no unreported calibration step can slip into a result.
- Every threshold and calibration operation records its data partition,
  operation name, dataset, unit and label digest in the provenance record.

---

## 5. Metrics (Task E)

`compute_evaluation_metrics` returns ROC-AUC, PR-AUC, macro F1, precision,
recall, specificity, balanced accuracy, MCC, the confusion matrix,
sensitivity at a target specificity, and the calibration metrics (Brier, ECE,
logistic slope/intercept). Every entry is a `MetricResult` with `value`, `unit`,
`n`, `n_positive`, `defined` and — when undefined — a machine-readable `reason`.

Rules the layer enforces:

- class 1 is the positive class everywhere, and the aggregate output says so;
- a metric that cannot be computed is `None` with a reason (`single_class`,
  `no_positive_predictions`, `specificity_target_not_attainable`,
  `zero_denominator`, …), never 0.0 or 0.5;
- accessing `.as_float()` on an undefined metric raises `MetricUndefinedError`;
- `n`, `unit` and class counts are attached to every metric so a
  participant-level number can never be mistaken for a session-level one;
- ECE uses equal-frequency bins and is *suppressed* (undefined, reason
  `bin_below_min_observations`) when any bin holds fewer than 2 observations;
- imbalance and small-positive notes are added to the output rather than left
  to the reader.

The Phase 2 helpers (`compute_metrics`, `compute_full_metrics`, `brier_score`,
`expected_calibration_error`, `reliability_curve`) are untouched, so existing
Phase 2 imports keep working.

---

## 6. Bootstrap intervals (Task F)

- EATD: participant-cluster bootstrap, 2000 resamples, seed 0, percentile
  interval, stratified within class, session probabilities aggregated to one
  value per participant (mean) before resampling.
- AVDD: sample-level bootstrap, 2000 resamples, seed 0 — with
  `clustering_respected: false` and an explicit note, because AVDD releases no
  participant identity.
- Degenerate resamples (single class, undefined metric) are counted in
  `n_invalid_resamples` with reason counts; if the valid fraction falls below
  0.95 the interval is withheld (`status: no_valid_resamples` /
  `insufficient_valid_resamples`) instead of being reported thin.
- Seeds, resample counts, alpha, interval type, valid fraction and the point
  estimate are all recorded; identical seeds reproduce identical intervals.

These are interval *machinery* checks on synthetic arrays. No interval in this
phase describes a real model.

---

## 7. Leakage probes (Task G)

`tests/test_phase5_leakage.py` constructs each invalid condition with synthetic
fixtures and asserts the harness rejects it:

| # | Leak | Guard |
|---|---|---|
| 1 | same participant in train and evaluation | `SplitIntegrityError` at `Fold` construction |
| 2 | participant in two outer evaluation folds | `assert_repetition_integrity` (unit- and group-level) |
| 3 | same session in two partitions | `SplitIntegrityError` |
| 4 | locked-test participant inside development units | `SplitIntegrityError` (`locked_group_ids`) |
| 5 | inner fold containing an outer evaluation participant | `inner_group_folds` |
| 6 | threshold selection on outer evaluation | `ProvenanceError` (`threshold_selection`) |
| 7 | calibration fitted on outer evaluation / locked test | `ProvenanceError` (`calibration_fit`) |
| 8 | preprocessing fitted on evaluation data | `ProvenanceError` (`preprocessing_fit`) |
| 9 | feature selection using evaluation labels | `ProvenanceError` (`feature_selection`) |
| 10 | grouping by bare subject id across partitions | `GroupingError` / lazy-key check |

Probes 6–9 also assert the *allowed* case still works (inner OOF, inner train),
so a guard that simply always raised would fail the tests.

---

## 8. Locked-test firewall (Task H)

`LockedTestFirewall.from_eatd_splits()` holds the 24 locked participant keys and
`labels_available: false`; the report it produces contains no participant keys
and no labels. It blocks, by construction:

- locked participants in any fit, selection or inner-CV call
  (`assert_not_locked`, `assert_selection_safe`, `filter_development`,
  `assert_partition_pristine`, and the `firewall=` argument of
  `generate_oof_predictions`);
- declarations that a model-selection partition is `locked_test`.

The single authorized access requires a complete pre-registration
(`experiment_id`, model configuration, threshold rule, primary metric,
evaluation script, `script_sha256`, registration timestamp), a non-empty reason,
and an optional script path whose hash must match. It writes a ledger entry
*before* the body runs, marks it `completed` or `failed`, and refuses a second
authorization in the same process.

### Future final-evaluation procedure (documented, not executed)

1. Freeze the model configuration, threshold rule and primary metric; write the
   pre-registration to `results/phase5/<experiment_id>/preregistration.json`.
2. Hash the evaluation script and record it in the pre-registration.
3. Run the frozen script once under `authorize_final_evaluation(...)`; the ledger
   entry at `results/phase5/locked_test_access.jsonl` is the audit trail.
4. Report the locked-test number once, with its interval, beside the development
   numbers — never in place of them.
5. Any further access requires a new pre-registration and an explicit decision
   recorded in `docs/`.

Phase 5B stops at step 0: the procedure is written down and the guard is tested,
nothing is run.

---

## 9. Artifact contract and configuration (Task I)

`validate_experiment_artifact` enforces the required blocks (`dataset`,
`label_definition`, `evaluation`, `preprocessing`, `model`, `selection`,
`results`, `environment`, `timestamps`), `code_revision`,
`known_limitations`, and the two always-false flags:

- `clinical_validation` must be `false` — this is a research prototype;
- `serving` must be `false` — no Phase 5 model is deployable.

Honesty rules: `results.status` is one of `not_measured` / `partial` /
`measured`; `not_measured` may not carry numbers; `measured`/`partial` require
metrics and a named `results.source`; a `null` metric must state `reason`; every
metric must declare a valid `unit`. `selection.threshold_partition`,
`selection.calibration_partition` and `preprocessing.fit_partition` may not be
`outer_evaluation` or `locked_test`. `evaluation.n_positive` may not exceed
`n_units`, seeds must be recorded, and `dataset.sha256` must identify the
snapshot.

`write_experiment_artifact` validates first, then refuses
`results/phase2`, `models/experiments`, `models/final`, and refuses to overwrite
an existing artifact unless `overwrite=True`. The same rules are published as
`ml/configs/phase5/experiment_artifact.schema.json`; tests validate fixtures
against both the Python validator and the JSON Schema.

`ml/configs/phase5/evaluation.yaml` is the single declared source for seeds
(42–46), outer/inner fold counts (5/4), bootstrap defaults (2000, seed 0, alpha
0.05, 0.95 valid fraction), the threshold policy and forbidden partitions, the
metric set and positive class, the guard thresholds, artifact rules and the
locked-test policy. `load_evaluation_config` refuses an ungrouped splitter, a
session-level EATD unit, calibration fitting in this phase, or a locked-test
read — and the tests assert the config agrees with the module constants.

---

## 10. Running it

```bash
./.venv/Scripts/python.exe scripts/phase5/build_split_audit.py          # split audit only
./.venv/Scripts/python.exe -m pytest tests/test_phase5_splits.py -q
./.venv/Scripts/python.exe -m pytest tests/test_phase5_metrics.py -q
./.venv/Scripts/python.exe -m pytest tests/test_phase5_leakage.py -q
./.venv/Scripts/python.exe -m pytest tests/test_phase5_firewall.py -q
./.venv/Scripts/python.exe -m pytest tests/test_phase5_artifacts.py -q
./.venv/Scripts/python.exe -m pytest -q                                 # full suite
```

No new dependency was added: the harness uses the existing `scikit-learn`,
`numpy`, `scipy`, `PyYAML` and `jsonschema` in `.venv`.

---

## 11. Limitations

- The harness has not produced a single real prediction; every metric,
  interval and artifact in the tests is synthetic. **Phase 5B cannot be cited
  as evidence about model performance.**
- Calibration interfaces are defined but unusable: `fit_calibrator` raises by
  design until Phase 5C.
- AVDD bootstrap intervals ignore participant clustering because the dataset
  releases no participant identity; they are not comparable to EATD intervals.
- The skewed EATD development set (45 positives across 138 participants) means
  several folds hold few positives; the guards report that rather than
  compensating for it, and small-positive estimates remain unreliable.
- The locked-test firewall is an in-process guard, not a filesystem permission:
  it prevents the harness from passing locked keys into fits and selection, but
  a determined caller could read the files outside the harness. The audit trail,
  not enforcement, is the guarantee.
- The split audit's inner-fold summaries are metadata only; no inner training or
  scoring was performed.
