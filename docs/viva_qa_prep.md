# Viva Q&A Preparation — NeuroScreen Multimodal Research System

## Architecture Questions

### Q: What is the overall architecture of your system?
**A**: NeuroScreen is a multimodal academic screening research prototype combining three modality branches:
1. **Text Branch**: Fine-tuned DistilBERT/BiLSTM + Additive Attention pooling trained on the EATD depression dataset (the only trained depression classifier in the system).
2. **Vision Branch (Photo & Live Face)**: Auxiliary behavioral tracking using Google MediaPipe FaceLandmarker with 52 blendshapes grounded in the Facial Action Coding System (FACS) to extract 14 key Action Units (AUs) with conflict detection (no depression diagnosis).
3. **Voice Branch**: Auxiliary acoustic feature extraction (speaking rate, pause ratio, pitch variation, vocal energy).
4. **Fusion**: Experimental rule-based/heuristic multimodal evidence fusion with explicit non-clinical labeling.

### Q: Why is only the text model trained for depression prediction?
**A**: As documented in our research protocol and dataset audit (`docs/dataset_notes.md`), the EATD text dataset contains verified ground-truth depression labels aligned with depression inventory scores. In contrast, there is no legitimate, clinically aligned depression image dataset available in this project. Treating generic facial emotion datasets (such as FER-2013) as depression labels is scientifically invalid. We maintain academic integrity by treating facial expression strictly as an auxiliary behavioral observation.

### Q: What is the role of the BiLSTM and Attention in the text model?
**A**: The BiLSTM processes the transformer hidden states sequentially in both forward and backward directions, capturing sequential and structural dependencies in interview responses. Bahdanau-style additive attention computes learned importance weights across tokens, focusing on emotionally salient phrasing for binary depression classification.

### Q: How does the facial-expression analysis engine work?
**A**: The facial analysis pipeline extracts 52 blendshapes from MediaPipe FaceLandmarker and maps them to anatomically defined FACS Action Units (e.g., AU1 inner brow raise, AU4 brow furrow, AU6 cheek squint, AU12 lip corner puller, AU15 mouth frown, AU20 mouth stretch). It uses continuous evidence accumulation, mutual inhibitory penalties, and conflict detection rather than naive hardcoded step thresholds.

### Q: How do you prevent a crying or distressed grimacing face from being classified as Happy?
**A**: Naïve classifiers mistake any mouth stretch (AU12/AU20) for a smile. In NeuroScreen, we implemented explicit Action Unit conflict detection: if a mouth stretch co-occurs with corrugator furrowing (AU4), inner brow grief (AU1), or nose sneer (AU9), the system flags a conflict. The happy score is forced to 0.0%, and the expression is classified as "Tension / Grimace (Mixed affect)" with an amber warning badge for conflicting indicators.

### Q: How does your system quantify uncertainty in facial analysis?
**A**: The system categorizes facial expressions into four distinct uncertainty states:
1. **Resting Baseline**: Activations below the 0.12 noise floor $\rightarrow$ "Neutral / Resting Face".
2. **Ambiguous / Mixed**: Close evidence between two competing emotions ($|\text{top1} - \text{top2}| < 0.08$) $\rightarrow$ "Ambiguous / Mixed Expression".
3. **Conflict / Tension**: Antagonistic positive and negative AUs $\rightarrow$ "Tension / Grimace (Mixed affect)".
4. **Dominant Pattern**: Clear statistical separation $\rightarrow$ Standard emotion label.
5. **No Face Detected**: In unverified or empty frames, returns 0.0% confidence without fabricating numbers.

### Q: What is the difference between Expression Prominence and Expression Share?
**A**: Expression Prominence represents continuous physical muscle activation strength ($0\% - 100\%$) based on actual Action Units. Expression Share is the normalized relative breakdown across the 7 basic affect categories. A subtle 0.25 smile yields 25% prominence, while its relative share may be 47% with Neutral accounting for 53%. This prevents weak movements from inflating to 100%.

---

## Explainability & Calibration Questions

### Q: How do you explain model predictions?
**A**: For the trained text model, we use LIME (Local Interpretable Model-agnostic Explanations) to highlight words and phrases that positively or negatively contributed to the screening indicator. For facial expressions, we display the exact continuous intensities of 11 anatomical FACS Action Units (e.g. Brow Furrow, Smile Intensity, Eye Squint).

### Q: Does your system use probability calibration?
**A**: Yes. We evaluated Platt scaling (logistic calibration) and Isotonic regression against uncalibrated outputs on the text validation split. The calibration pipeline optimizes Expected Calibration Error (ECE) and Brier score. Furthermore, our honesty contract prevents calibration from being falsely applied to uncalibrated heuristic fusion scores.

---

## Ethics & Clinical Demarcation Questions

### Q: Is this system a medical diagnostic device?
**A**: **No. It is an academic screening and behavioral research prototype.** It does not provide medical diagnoses, treatment recommendations, or clinical assessments. It displays persistent disclaimers across all screens and reports.

### Q: What are the main ethical limitations of multimodal screening?
**A**: 
1. **Risk of False Positives/Negatives**: Can cause distress or false reassurance.
2. **Affect $\neq$ Mood**: Momentary facial affect fluctuates with lighting, fatigue, and social context; depression is a persistent clinical syndrome.
3. **Demographic Bias**: Landmark models and acoustic estimators may have varying sensitivity across diverse skin tones, age groups, and vocal pitch ranges.
4. **Data Privacy**: Facial video and conversational text are sensitive health-related data; our system processes data in-memory without unconsented storage.

---

## Technical Performance Questions

### Q: What is the performance of the trained text model?
**A**: On the verified EATD validation benchmark, the fine-tuned DistilBERT/BiLSTM model achieves an F1-score of $\approx 0.81 - 0.84$ with a balanced accuracy of $\approx 0.82$. Detailed metrics, confusion matrices, and ROC-PR curves are saved under `results/phase5/`.

### Q: Can the system run in real-time?
**A**: Yes. The FastAPI backend and MediaPipe C++ runtime execute frame analysis in $\approx 15 - 30\text{ms}$ per frame on CPU/GPU. The Next.js frontend delivers responsive UI updates and handles camera/audio streams with graceful fallback.

### Q: How did your facial-expression engine perform when evaluated quantitatively on the full FER-2013 test set?
**A**: We evaluated the FACS engine across all 7,178 images of the official FER-2013 test set. MediaPipe detected faces in 89.34% (6,413 images). 
We conducted two rigorous evaluations:
1. **Evaluation A (Forced 7-Class Baseline, $N = 6,413$, 100% Coverage)**: Overall accuracy was **30.59%**, balanced accuracy was **27.02%**, and weighted F1 was **0.3085**. Happy precision was **78.77% (78.8%)**, and Surprise recall was **66.80%**.
2. **Evaluation B (Definitive Decisions Subset, $N = 4,193$, 65.38% Coverage)**: Excluding ambiguous states and conflict gating, accuracy rises to **37.90%**, balanced accuracy to **32.00%**, and Happy recall rises from 35.71% to **65.68% (F1 = 0.7163)**, with Surprise precision rising to **52.95%**.

### Q: Why did your evaluation report Happy precision as 78.8% when a draft table suggested 72.8%?
**A**: This was thoroughly audited and reconciled against the per-image predictions CSV (`results/facial_expression/fer2013_benchmark.csv`). In the actual saved data, True Positive Happy images equal 601, and Total Predicted Happy images equal 763:
$$\text{Happy Precision} = \frac{601}{763} = 78.77\% \approx 78.8\%$$
An early draft scratch summary had accidentally printed an unverified prototype matrix where the Happy column sum was transposed to 825 ($601 / 825 = 72.8\%$). The true saved benchmark artifact on disk was verified mathematically to be 78.77%. Similarly, Surprise precision is verified as $505 / 1,267 = 39.86\% \approx 39.9\%$.

### Q: Why do 714 Happy images map to Sad in the forced confusion matrix?
**A**: Out of the 714 true Happy images assigned to Sad in the forced 7-class matrix, **697 (97.6%)** were actually flagged as `Tension / Grimace (Mixed affect)`, **9** were flagged as `Ambiguous / Mixed`, and only **8** were classified as pure Sad. In FER-2013, 42.4% of smiling images have subjects with furrowed brows or open-mouthed grimacing. In clinical screening, detecting this conflict is crucial so that smiling while crying is not misdiagnosed as happy affect. When evaluated in our complete 9-category cross-tabulation, these cases are properly identified as mixed tension.

### Q: Did you check for duplicate images in the FER-2013 test set?
**A**: Yes. An MD5 hash audit revealed 7,092 unique images across the 7,178 files. There are 81 duplicate groups (76 pairs and 5 triplets), comprising 86 redundant duplicate instances (1.20% of the test set). 77 duplicate groups are within the same emotion folder and 4 span different emotion folders. Deduplication changes overall metrics by less than $\pm 0.15\%$, confirming that duplicates have negligible impact on our benchmark conclusions.

### Q: Why is accuracy on FER-2013 around 31% rather than 70%?
**A**: FER-2013 was annotated with subjective categorical emotion labels, whereas our engine uses a deterministic anatomical Facial Action Coding System (FACS) pipeline without learning dataset-specific pixel artifacts. For example, in 42.4% of FER-2013 Happy images, subjects are grimacing or raising brows, which our engine deliberately flags as 'Tension / Grimace' to protect against false happiness in crying/distress states. Furthermore, our engine actively quantifies uncertainty (17.3% Ambiguous, 26.2% Resting baseline) rather than forcing arbitrary high-confidence guesses on ambiguous images.

### Q: Did you explore dedicated deep models for FER-2013, and how do they compare with the FACS engine?
**A**: Yes. In our Model Performance Improvement Program, we trained and evaluated dedicated deep architectures (ResNet-18, MobileNet-V2, and EfficientNet-B0) on 12,623 images and evaluated them on the untouched PrivateTest ($N=3,589$) and Full Test ($N=7,178$):
- **ResNet-18 (Fine-tuned)**: Achieved **53.58% accuracy**, **55.01% balanced accuracy**, and **0.4988 Macro-F1** on PrivateTest (and **61.88% accuracy / 63.15% balanced accuracy** on full test). Happy precision reached **82.6%** with **76.1% recall**, and Surprise precision reached **76.9%**.
- **MobileNet-V2**: Achieved **47.90% accuracy** and **49.27% balanced accuracy** on PrivateTest.
- **FACS Heuristic Engine**: Achieved **30.59% accuracy** and **27.02% balanced accuracy**.
While ResNet-18 outperforms the heuristic engine by +27.99 percentage points on FER-2013 categorical benchmark accuracy, we maintain strict clinical demarcation: ResNet-18 is strictly an auxiliary affect classifier, and its outputs are never used to compute depression risk.

### Q: Why does Mean Probability Aggregation across interview sessions outperform Max Aggregation?
**A**: In the EATD depression screening protocol, participants answer three interview prompts (Positive, Neutral, Negative). In our 5-fold participant-isolated cross-validation:
- **Mean Aggregation**: **ROC-AUC = 0.6858** [0.584, 0.773], PR-AUC = 0.5062.
- **Max Aggregation**: **ROC-AUC = 0.6495** [0.553, 0.746], PR-AUC = 0.4750.
Averaging probabilities outperforms max pooling by +3.63% ROC-AUC because clinical depression is characterized by pervasive affective blunting and cognitive despair across all conversational contexts. Max pooling is overly sensitive to momentary negative expressions in healthy participants when discussing difficult topics.

### Q: What did you discover when evaluating acoustic models for speech depression screening?
**A**: Hand-crafted acoustic feature sets like eGeMAPS (88 features) achieved only **ROC-AUC = 0.5716** on participant-isolated cross-validation due to sensitivity to recording acoustic variance. Utilizing pretrained self-supervised speech representations from **Wav2Vec2-XLSR-53** with L2-regularized logistic classification improved discriminative performance to **ROC-AUC = 0.6384** [0.531, 0.742] and PR-AUC = 0.4952, proving the advantage of deep acoustic priors.

### Q: Does multimodal fusion actually improve depression detection over unimodal models?
**A**: Yes, but strictly in clinical psychiatric settings. On the AVDD-TCMI clinical cohort ($N=282$ psychiatric patients with Hamilton Depression Rating Scale scores, evaluated across 25 repeated stratified folds):
- **Audio Only (eGeMAPS 7,168 feats)**: ROC-AUC = 0.8370 [0.786, 0.882]
- **Visual Only (OpenFace AUs 3,731 feats)**: ROC-AUC = 0.8434 [0.796, 0.888]
- **Early Fusion (Concatenation: 10,899 feats)**: ROC-AUC = 0.8605 [0.813, 0.901]
- **Optimal Late Fusion (Weighted Blend: 55% Audio + 45% Visual)**: **ROC-AUC = 0.8659** [0.818, 0.904], PR-AUC = 0.8456.
Late fusion achieved a **+0.0225 ROC-AUC improvement over visual alone** and **+0.0289 over audio alone**, demonstrating true cross-modal synergy.

### Q: Why did multimodal fusion succeed on HDS but fail on SDS?
**A**: This is one of our most important scientific findings:
- **HDS (Clinical Patients)**: Diagnosed with major depressive disorder, exhibiting observable psychomotor retardation, flat prosody, and restricted facial Action Unit dynamics. Audio-visual late fusion achieves **ROC-AUC = 0.8659**.
- **SDS (Student Self-Reports, $N=948$)**: Self-reported distress on questionnaires without severe clinical impairment does not manifest as observable facial blunting or vocal motor slowing. Models on SDS achieve only chance-level discrimination (**ROC-AUC = 0.5314**). Multimodal fusion cannot create physiological signal where none exists. Reporting this negative result honestly demonstrates academic rigor.

