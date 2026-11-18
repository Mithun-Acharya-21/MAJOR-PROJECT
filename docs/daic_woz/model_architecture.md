# DAIC-WOZ Model Architecture Specification

---

## 1. Single-Modality Architectures

```
Text Stream:
  Raw Transcript -> Speaker Filter (Participant) -> TF-IDF (1-2 grams) -> Logistic Regression (L2) / SVM
                                                 -> DistilBERT (Frozen) -> BiLSTM (128) -> Self-Attention -> Dense(2)

Audio Stream:
  WAV Audio -> openSMILE eGeMAPSv02 -> StandardScaler (Train Fit) -> Regularized Logistic Regression / SVC (RBF)

Visual Stream:
  OpenFace Kinematics -> Temporal Aggregation (Mean/Std) -> StandardScaler -> Regularized Logistic Regression
```

---

## 2. Multimodal Late-Fusion Meta-Learner

To prevent optimistic bias and training leakage, fusion models are trained using **5-fold out-of-fold (OOF) cross-validation**:

1. The development training set is partitioned into 5 stratified participant folds.
2. For each fold $k \in \{1, \dots, 5\}$:
   - Base text, audio, and visual models are fitted on the remaining 4 folds.
   - Out-of-fold probability vectors $\hat{p}_{\text{text}}, \hat{p}_{\text{audio}}, \hat{p}_{\text{visual}}$ are generated on fold $k$.
3. The stacked matrix of OOF predictions is used to fit an $L_2$-regularized Logistic Regression meta-learner:
   $$\hat{y}_{\text{fusion}} = \sigma\left(w_0 + w_{\text{text}} \hat{p}_{\text{text}} + w_{\text{audio}} \hat{p}_{\text{audio}} + w_{\text{visual}} \hat{p}_{\text{visual}}\right)$$
4. Final probability calibration is applied via Platt scaling to ensure valid Brier score alignment.
