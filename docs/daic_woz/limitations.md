# DAIC-WOZ Research Limitations & Boundary Conditions

---

## 1. Dataset Limitations

1. **Cohort Size:** $N=189$ participants. Due to limited sample size, deep neural networks with millions of parameters risk severe overfitting; regularized linear baselines and frozen embeddings must be prioritized.
2. **Context Specificity:** Interviews were conducted with a virtual avatar (Ellie) in a laboratory setting. Acoustic and linguistic patterns may differ from natural doctor-patient consultations or spontaneous daily conversations.
3. **Class Imbalance:** Subclinical control participants outnumber clinically depressed participants; class weighting and balanced metrics must always be reported.

---

## 2. Technical and Boundary Limitations

1. **Non-Diagnostic:** NeuroScreen outputs are statistical screening indicators, not diagnostic classifications.
2. **Facial Affect Separation:** Momentary facial emotion detection (e.g. FER-2013) is strictly decoupled from psychiatric depression risk.
3. **Missing Modality Handling:** The system flags offline modalities transparently and evaluates complete triads separately from partial inputs.
