"""
EmotiSense AI — Emotion-Aware & Explainable Deep Learning Model
for Early Depression Detection.

This package contains the reusable Python source code organized by pipeline stage:

  src/text_pipeline/  — Text preprocessing, DeBERTa encoder, BiLSTM+Attention
  src/face_pipeline/  — Face detection, preprocessing, CNN model
  src/fusion/         — Feature fusion, classifier head, training
  src/explainability/ — LIME, SHAP, visualization
  src/evaluation/     — Metrics, baseline models
  src/utils/          — Config loader, logger, seed management
"""

__version__ = "1.0.0"
