# Viva Preparation — EmotiSense AI

This document prepares you for common viva questions across architecture,
methodology, safety, explainability, and project limitations.

---

## 1. Architecture questions

**Q: Why DeBERTa-v3 over BERT or RoBERTa for the text branch?**

DeBERTa-v3 uses disentangled attention — it encodes word content and
relative position separately rather than summing them. This lets the model
independently attend to what a word is and where it appears, which improves
performance on tasks where word order and context interact. For depression-risk
screening, negation and sentence structure are particularly important (e.g.
"I don't feel hopeless" vs "I feel hopeless"), making the position-aware
attention mechanism beneficial. DeBERTa-v3 also achieves higher NLU benchmark
scores than BERT/RoBERTa at similar parameter counts.

**Q: Why add BiLSTM after DeBERTa instead of using only the transformer?**

The DeBERTa encoder produces contextual token representations, but the BiLSTM
learns additional sequential patterns over those representations in both
directions. In practice this is justified by running the ablation: the
`deberta_bilstm_attention` variant is compared against `deberta_mean` and
`deberta_attention` on the same frozen split. If the BiLSTM improves validation
F1, its added complexity is justified. If not, the simpler baseline is
preferred. No assumption is made before seeing the evidence.

**Q: Why use MobileNetV2 rather than ResNet-50 for face encoding?**

MobileNetV2 has 3.4 M parameters (vs 25 M for ResNet-50), which allows CPU
inference for demonstration purposes and reduces GPU memory requirements during
training. ResNet-18 is provided as an alternative in `VisionTransferConfig`
for comparison. The recommendation is to run both on the same frozen FER split
and select based on validation accuracy, not assumption.

**Q: How does the fusion gate decide how much to trust each modality?**

The gate MLP takes three inputs per modality: the projected embedding, the
quality score from the validation pipeline (audio SNR/VAD, ASR confidence,
face blur/detection), and the presence mask. The output is a scalar in (0, 1),
hard-multiplied by the presence mask so absent modalities always contribute
zero. This means a poor-quality recording automatically reduces the acoustic
branch's contribution without discarding the text branch. After training, the
gate weights are available as a diagnostic signal in the API response.

**Q: What is temperature scaling and why is it done post-hoc?**

Temperature scaling divides the logits by a learned scalar T before the
softmax. A model with T > 1 is overconfident; T < 1 is underconfident.
Optimising T minimises NLL on the validation set after the model is locked,
using only one extra parameter. It is post-hoc because the model's class
probabilities and rankings do not change — only the confidence level is
adjusted. This is the standard calibration method from Guo et al. (2017)
and is strictly applied to validation data only.

---

## 2. Dataset and scientific methodology questions

**Q: Why can't you use RAVDESS or FER-2013 labels for depression training?**

Because there is no participant-session alignment. A RAVDESS actor performing
"sad" speech and a FER-2013 facial image from a different person have no
connection to any depression label. Pairing them would teach the model to
associate acted sadness with a clinical label from an unrelated person — that
is not a valid causal relationship and produces misleading results. The
metadata validator explicitly rejects any multimodal depression record whose
modalities do not come from the same participant session.

**Q: What is participant-disjoint splitting and why is it necessary?**

If participant P001 appears in both the training and test sets, the model can
learn to recognise P001's personal vocal or linguistic characteristics rather
than generalising depression-risk patterns. This inflates test metrics because
the model has effectively seen the test subject before. Participant-disjoint
splitting guarantees that every person's data is in exactly one partition,
so test performance reflects generalisation to unseen individuals.

**Q: How do you handle class imbalance in DAIC-WOZ?**

DAIC-WOZ has approximately 30 % depressed and 70 % control labels. Three
mechanisms address this: (1) class-balanced logistic regression for the
classical baseline; (2) `class_weight="balanced"` in the baseline pipeline;
(3) in the DeBERTa and fusion training, weighted cross-entropy with inverse
class frequencies, and `modality_dropout` during training so the model learns
missing-modality robustness from the minority class too.

**Q: Why is the test set locked?**

The test set measures generalisation to completely unseen data. Every time you
look at test metrics and adjust your model, the test set becomes part of model
selection. To get an honest estimate of out-of-sample performance, the test
set must be evaluated exactly once, after all development decisions are frozen.
The `assert_not_final_test(locked=False)` guard enforces this in code — the
benchmark script will not run unless `locked=True` is explicitly set.

---

## 3. Explainability questions

**Q: Can attention weights explain the model's decision?**

No. Attention weights show which tokens the model attended to, but high
attention on a token does not prove that token caused the prediction. Multiple
papers (Jain & Wallace 2019; Wiegreffe & Pinter 2019) show that attention
weights can be manipulated without changing predictions. In this system,
attention weights are explicitly labelled as a "supplementary diagnostic
signal, not a causal explanation" and are shown separately from LIME/SHAP.

**Q: What is Grad-CAM and what can it actually tell us?**

Grad-CAM computes the gradient of the target class score with respect to the
last convolutional feature map, then globally-average-pools those gradients to
produce per-channel weights. Weighted feature maps are summed and passed
through ReLU. The result highlights spatial regions that were important for
the prediction. It tells us what the model was looking at — not why the face
region physiologically corresponds to any state. In this system it is
always labelled as an "emotion-model visualisation, not a depression-risk
indicator."

**Q: LIME vs. SHAP — when would you use each?**

LIME perturbs the input locally and fits a simple linear model to the perturbed
outputs. It is fast and model-agnostic but approximations vary between runs
(stochastic). SHAP has a game-theory foundation (Shapley values) and provides
consistent attributions across runs, but KernelSHAP is slower. For a quick
interactive explanation LIME is preferred; for a reproducible attribution used
in a report, SHAP is more reliable. Both are used in this system, with results
displayed side-by-side.

---

## 4. Safety and ethics questions

**Q: How does the crisis-first policy work technically?**

`SupportPolicyEngine.evaluate()` checks all observed signals against
`CRISIS_SIGNALS = {self_harm_language, immediate_danger}` before any other
logic runs. If any crisis signal is present, `_crisis_response()` is called
immediately, which: (1) returns only `CRISIS_RESOURCES` category suggestions,
(2) sets `suppressed_by_crisis = True`, and (3) returns `is_crisis = True`.
No wellness suggestions are shown. Custom jurisdiction-specific crisis lines
can be injected at runtime via `crisis_resources` in `PolicyInput`.

**Q: How do you ensure the system never produces a clinical diagnosis?**

Three layers of enforcement: (1) `FORBIDDEN_DIAGNOSTIC_PHRASES` in
`safety.py` rejects any response containing phrases like "you have depression"
before it reaches the API. (2) All schemas have `disclaimer: str` fields
defaulting to `RESEARCH_DISCLAIMER`, so the disclaimer is structural — it
cannot be accidentally omitted. (3) The suggestion library is tested against
a forbidden-word list (`diagnos`, `prescri`, `medication`) in `test_support_policy.py`.

**Q: What is your approach to privacy for biometric data?**

All audio, image, and video data is processed in memory. The `read_upload_bytes`
function reads the file, validates it, and returns raw bytes — no temporary
file is written. The downstream processing functions (AudioValidator,
FacePreprocessor, WhisperASR) receive numpy arrays or bytes, not paths.
Request IDs are UUID4, with no user identity attached. `raw_input_retention_hours`
defaults to 0 in the settings. Consent is required before any biometric input
is processed.

---

## 5. Evaluation and metrics questions

**Q: Why report both F1 and ROC-AUC?**

F1 depends on the classification threshold (default 0.5) and is affected by
class imbalance. ROC-AUC is threshold-independent and measures ranking
performance across all thresholds. For a screener that might be calibrated to
different sensitivity/specificity trade-offs, ROC-AUC is more informative.
PR-AUC (area under the precision-recall curve) is also reported because for
imbalanced datasets it is more pessimistic and thus a more honest measure.

**Q: What is Expected Calibration Error (ECE)?**

ECE measures whether a model's confidence matches its actual accuracy. The
probability space is divided into bins; for each bin, the gap between mean
confidence and mean accuracy is computed, weighted by bin size. A perfectly
calibrated model has ECE = 0. After temperature scaling on the validation
set, the calibrated ECE should be lower than the raw model ECE.

**Q: How do you test for domain shift?**

The `detect_domain_shift(train_features, eval_features)` function computes
Maximum Mean Discrepancy (MMD) with an RBF kernel on a random subsample.
MMD near 0 suggests the distributions are similar; values above 0.1 trigger
a `high_shift_warning`. A second method (approximate KL divergence assuming
Gaussian marginals) is also available for comparison. These are diagnostic
tools — they indicate when evaluation data looks different from training data,
which prompts investigation rather than automatic disqualification.

---

## 6. Project design decisions

**Q: Why a rule-based support policy rather than a language model?**

A language model producing free-text suggestions introduces unpredictable
outputs that cannot be audited. The rule-based policy maps specific observed
signals to a fixed, approved library of non-prescriptive suggestions. Every
suggestion can be read, reviewed, and approved by a clinical consultant.
The policy is fully auditable: the input state, signal-to-category mapping,
and output suggestions are all inspectable in a single Python dataclass.

**Q: Why FastAPI rather than Flask/Django?**

FastAPI generates OpenAPI documentation automatically, enforces request/response
schemas via Pydantic (which prevents forbidden phrases from slipping through),
and has native async support for handling concurrent upload processing. Its
dependency injection system makes the consent gate and rate limiter easy to
attach as per-route dependencies without modifying business logic.

**Q: How does the missing-modality handling work at inference time?**

Each modality provides a mask tensor (1 = present, 0 = absent). The
`ModalityProjection` layer zeros the projected vector wherever the mask is 0.
The `ModalityGate` hard-multiplies its output by the mask, so an absent
modality always contributes zero — even if the gate network would otherwise
assign it a high weight. The `key_padding_mask` in `CrossModalAttention`
suppresses absent tokens during attention computation. The result is that
the model gracefully degrades: text-only, audio-only, or any combination
all work without any imputation or fabrication.

---

## 7. What to say if you don't have trained model results

"The architecture, training pipeline, evaluation protocol, safety controls,
explainability layer, and frontend are fully implemented and tested with
240+ unit and integration tests. The system is blocked only on obtaining
approved access to DAIC-WOZ, which requires a formal data-use agreement
from USC ICT. I have documented the exact access process in `docs/dataset-strategy.md`,
and the pipeline is ready to accept the dataset the moment access is granted.
All design decisions were made to be scientifically defensible: participant-
disjoint splits, a locked test set, calibration on validation only, and
an honest architecture-only readiness status in the API. No metrics have
been fabricated."
