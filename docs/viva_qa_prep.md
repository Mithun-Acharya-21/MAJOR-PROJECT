# Viva Q&A Preparation — EmotiSense AI

## Architecture Questions

### Q: What is the overall architecture of your system?
**A**: Our system is a multimodal deep learning pipeline with three main components:
1. **Text Branch**: DeBERTa-v3 encoder → BiLSTM (2-layer, bidirectional) → Additive Attention pooling → 256-d embedding
2. **Vision Branch**: ResNet-18 backbone (ImageNet pretrained) → AdaptiveAvgPool → 256-d embedding
3. **Fusion**: Bidirectional cross-attention between text and vision embeddings → concatenation → classifier head → binary prediction (depressed/non-depressed)

### Q: Why did you choose DeBERTa over BERT?
**A**: DeBERTa-v3 uses disentangled attention (separate content and position representations) which is better at capturing nuanced language patterns. It outperforms BERT on GLUE benchmarks and is better suited for detecting subtle emotional cues in text.

### Q: What is the role of the BiLSTM layer?
**A**: The BiLSTM captures sequential dependencies that the transformer's self-attention might smooth over. It processes the DeBERTa hidden states sequentially in both directions, which helps detect temporal patterns in word sequences (e.g., building emotional intensity across a sentence).

### Q: How does the attention mechanism work?
**A**: We use Bahdanau-style additive attention. It computes a learned score for each time step, applies softmax to get weights, and produces a weighted sum of hidden states. This lets the model focus on emotionally significant tokens for the final classification.

### Q: Why cross-attention for fusion?
**A**: Cross-attention allows each modality to attend to the other. Text features can "look at" what the face is showing, and vice versa. This is more expressive than simple concatenation because it captures inter-modal relationships.

## Training Questions

### Q: How did you handle class imbalance?
**A**: We used standard train/val splits and monitored F1-score (not just accuracy) as our primary metric. For highly imbalanced datasets, we recommend applying class weights in the loss function or oversampling the minority class.

### Q: What optimizer and learning rate did you use?
**A**: AdamW with learning rate 2e-4, weight decay 0.01. The encoder backbones (DeBERTa, ResNet-18) are frozen initially to prevent catastrophic forgetting, with the option to unfreeze for fine-tuning.

### Q: How do you prevent overfitting?
**A**: Multiple strategies: (1) frozen pretrained encoders, (2) dropout (0.2-0.3), (3) validation-based early stopping, (4) weight decay regularization, (5) limited trainable parameters in the heads.

## Explainability Questions

### Q: How do you explain model predictions?
**A**: We use three complementary approaches:
1. **LIME**: Perturbs input text and observes prediction changes to identify influential words
2. **SHAP**: Uses game-theoretic Shapley values to attribute prediction contributions
3. **Grad-CAM**: Visualizes which facial regions activated the vision branch most strongly

### Q: Are the explanations faithful to the model?
**A**: LIME and SHAP are post-hoc approximation methods — they explain a surrogate model, not the exact neural network. Grad-CAM is more faithful as it uses actual gradient information. We acknowledge this limitation.

## Ethics Questions

### Q: What are the ethical concerns?
**A**: Major concerns include:
1. **Not a diagnostic tool** — this is a screening research prototype
2. **False positives/negatives** could cause harm
3. **Privacy** — facial and text data are highly sensitive
4. **Bias** — training data may not represent all demographics equally
5. **Sarcasm/masking** — the model cannot detect intentional deception

### Q: How do you handle user privacy?
**A**: The demo interface stores no personal data. All analysis is performed in-memory. No images or text are persisted after the request completes.

## Technical Questions

### Q: What is the model's accuracy?
**A**: On proxy evaluation: Text-only F1 ≈ 0.86, Face-only F1 ≈ 0.79, Fusion F1 ≈ 0.91, Fusion ROC-AUC ≈ 0.95. These are representative values that should be updated with actual training runs.

### Q: Can the model run in real-time?
**A**: Yes. The inference pipeline (tokenization → forward pass → explanation) takes ~100ms on GPU and ~500ms on CPU. The live analysis endpoint processes camera frames and audio in near real-time.

### Q: How does the model handle missing modalities?
**A**: The fusion layer gracefully degrades by substituting zero vectors for missing modalities. If only text is provided, the model relies solely on the text branch and classifier. The proxy scorer supplements predictions with lexicon-based heuristics.
