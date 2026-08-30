# Limitations — EmotiSense AI

## Not a Clinical Tool

This is a **research prototype** for academic purposes only. It is NOT:

- A diagnostic tool
- A replacement for professional mental health assessment
- A clinical device approved by any regulatory body
- Suitable for making treatment decisions

## Model Limitations

### Text Branch
- **Sarcasm and irony** are poorly detected — the model may misclassify sarcastic statements
- **Masking behavior**: People who present as "fine" in text may be depressed but undetectable
- **Language dependency**: Primarily English; multilingual support not implemented
- **Context window**: Limited to 128 tokens; long journal entries are truncated
- **Slang evolution**: Depression-related internet slang evolves rapidly

### Vision Branch
- **Posed expressions**: Studio-quality facial expressions may not reflect true emotional state
- **Demographics**: FER-2013 has limited age/race diversity; performance varies across demographics
- **Lighting and pose**: Performance degrades with poor lighting or non-frontal faces
- **No video**: Single-frame analysis misses temporal dynamics (e.g., micro-expressions)

### Fusion Model
- **Missing modality**: When only text OR face is available, fusion degrades gracefully but loses cross-modal signal
- **Weight balance**: The relative importance of text vs. face is not personalized
- **Training data**: Proxy datasets, not clinical depression labels

## Ethical Concerns

1. **False positives**: Non-depressed individuals may be flagged, causing unnecessary anxiety
2. **False negatives**: Depressed individuals may be missed, creating false reassurance
3. **Privacy**: Facial images and personal text are sensitive data
4. **Bias**: Training data may underrepresent certain demographics
5. **Automation bias**: Users may over-rely on the model's predictions

## Known Limitations in Current Implementation

- LIME/SHAP explanations are approximation-based, not exact
- Grad-CAM highlights regions, not specific facial features
- Proxy scoring (lexicon-based) supplements untrained model weights
- No longitudinal tracking or trend analysis
- No integration with clinical workflows or EHR systems

## Recommendations for Improvement

1. Train on clinically validated datasets with professional diagnoses
2. Add demographic fairness evaluation across subgroups
3. Implement uncertainty quantification for predictions
4. Add longitudinal tracking for trend detection
5. Include consent and data governance frameworks
6. Add human-in-the-loop review for high-risk predictions
