# DAIC-WOZ Evaluation Framework & Metrics Specification

---

## 1. Primary Classification Metrics

In accordance with clinical machine learning guidelines, all evaluations report:
- **Balanced Accuracy:** Arithmetic mean of sensitivity and specificity (protects against class imbalance).
- **Macro-F1 & Weighted-F1:** Balanced evaluation across control ($y=0$) and depressed ($y=1$) classes.
- **ROC-AUC & PR-AUC:** Discrimination capacity across continuous decision thresholds.
- **Brier Score:** Mean squared error between predicted probabilities and binary labels ($0.0$ indicates perfect calibration).
- **Participant-Level Bootstrap Confidence Intervals (95% CI):** Derived from 1,000 resamples of the test cohort.

---

## 2. Regression Metrics (Continuous PHQ-8)

When continuous PHQ-8 scores are evaluated:
- **Mean Absolute Error (MAE):** Average absolute point difference on 0–24 scale.
- **Root Mean Squared Error (RMSE):** Penalizes large deviations.
- **Pearson Correlation Coefficient ($r$):** Linear association with clinician-administered scores.
