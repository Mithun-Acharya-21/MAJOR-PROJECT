# Phase 15 — Evaluation, Ablation, Fairness and Domain-Shift Checks

## Metric suite

Every model evaluation produces the full set below.
No metric subset is acceptable for a publication-quality report.

| Metric | Function |
|---|---|
| Accuracy | `compute_classification_metrics` |
| Precision | " |
| Recall | " |
| F1 | " |
| ROC-AUC | " |
| PR-AUC | " |
| Brier score | " |
| Sensitivity (TPR) | " |
| Specificity (TNR) | " |
| Confusion matrix | " |
| Bootstrap 95 % CI (F1, ROC-AUC) | `bootstrap_ci=True` |
| ECE / MCE | `expected_calibration_error` (Phase 14) |
| Per-modality subset F1 | `run_modality_ablation` |
| Fusion variant ranking | `compare_fusion_variants` |
| Quality-accuracy correlation | `analyse_gating_impact` |
| Subgroup disparity gap | `compute_subgroup_metrics` |
| Domain shift score (MMD / KL) | `detect_domain_shift` |

## Evaluation protocol

```
1. Validate model selection on the VALIDATION split only.
2. assert_not_final_test("test", locked=False)  ← guard prevents premature test use.
3. Lock model (no further weight changes, no hyperparameter tuning).
4. assert_not_final_test("test", locked=True)   ← now permitted.
5. Run one final evaluation on the TEST split.
6. Record EvaluationReport and save to experiments/.
```

## Modality ablation

`run_modality_ablation(predict_fn, samples)` evaluates under:
- All modalities present
- Each single modality (text only, audio only, video only)
- All pairwise combinations (text+audio, text+video, audio+video)

Results are ranked by F1 to show the marginal value of each modality.

## Fairness checks

`compute_subgroup_metrics(actual, predicted, probs, subgroup_labels)` computes
per-group accuracy, F1, sensitivity, and specificity, then reports:
- `f1_max_gap` — largest F1 difference across groups
- `sensitivity_max_gap` — largest sensitivity difference across groups

A gap > 0.05 warrants investigation before deployment.

## Domain-shift detection

`detect_domain_shift(train_features, eval_features)` uses:
- **MMD (RBF)** — non-parametric, no distribution assumption
- **Gaussian KL** — faster, assumes Gaussian marginals

A `shift_score > 0.1` triggers `high_shift_warning = True`.

## Reliability-gating impact

`analyse_gating_impact(quality_scores, actual, predicted)` stratifies
accuracy by quality quartile. A positive `quality_accuracy_correlation`
confirms the gate is discriminative (high quality → better predictions).

## Important disclaimers

- **Never evaluate on the final test set before the model is locked.**
  The `assert_not_final_test` guard enforces this in code.
- All metrics are on a research dataset. Clinical population performance
  has not been established and may differ substantially.
- Confidence intervals are non-parametric bootstrap CIs; they do not
  account for dataset-level biases or demographic shift.
