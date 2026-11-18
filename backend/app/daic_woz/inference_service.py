"""
DAIC-WOZ Multimodal Research Inference Service
NeuroScreen Major Project - Phase 10 Inference Service
"""

from typing import Dict, Any, Optional
import numpy as np

from backend.app.daic_woz.model_loader import DaicWozModelLoader
from backend.app.daic_woz.preprocessing import (
    preprocess_transcript_text,
    validate_visual_feature_vector
)
from backend.app.daic_woz.schemas import (
    DaicWozPredictionResponse,
    ModalityPrediction
)


class DaicWozInferenceService:
    def __init__(self):
        self.loader = DaicWozModelLoader.get_instance()

    def predict_text(self, text: str) -> ModalityPrediction:
        cleaned_text = preprocess_transcript_text(text)
        if not self.loader.text_model or not self.loader.text_vectorizer:
            return ModalityPrediction(
                modality="text",
                available=False,
                model_name="TF-IDF + Logistic Regression",
                status="Model checkpoint awaiting official DAIC-WOZ dataset download"
            )

        X_tfidf = self.loader.text_vectorizer.transform([cleaned_text])
        prob = float(self.loader.text_model.predict_proba(X_tfidf)[0, 1])
        return ModalityPrediction(
            modality="text",
            available=True,
            model_name="TF-IDF + Logistic Regression",
            raw_score=prob,
            calibrated_probability=round(prob, 4),
            status="PREDICTION_COMPLETE"
        )

    def predict_visual(self, features: list) -> ModalityPrediction:
        feat_arr = validate_visual_feature_vector(features).reshape(1, -1)
        if not self.loader.visual_model or not self.loader.visual_scaler:
            return ModalityPrediction(
                modality="visual",
                available=False,
                model_name="OpenFace Kinematics + Logistic Regression",
                status="Model checkpoint awaiting official DAIC-WOZ dataset download"
            )

        X_scaled = self.loader.visual_scaler.transform(feat_arr)
        prob = float(self.loader.visual_model.predict_proba(X_scaled)[0, 1])
        return ModalityPrediction(
            modality="visual",
            available=True,
            model_name="OpenFace Kinematics + Logistic Regression",
            raw_score=prob,
            calibrated_probability=round(prob, 4),
            status="PREDICTION_COMPLETE"
        )

    def analyze_multimodal(
        self,
        transcript_text: Optional[str] = None,
        visual_features: Optional[list] = None,
        audio_features: Optional[np.ndarray] = None
    ) -> DaicWozPredictionResponse:
        modality_breakdown = {}

        # 1. Text inference
        if transcript_text:
            modality_breakdown["text"] = self.predict_text(transcript_text)
        else:
            modality_breakdown["text"] = ModalityPrediction(
                modality="text", available=False, model_name="None", status="NO_INPUT_SUPPLIED"
            )

        # 2. Audio inference
        modality_breakdown["audio"] = ModalityPrediction(
            modality="audio",
            available=False,
            model_name="openSMILE eGeMAPS + Logistic Regression",
            status="Model checkpoint awaiting official DAIC-WOZ dataset download"
        )

        # 3. Visual inference
        if visual_features:
            modality_breakdown["visual"] = self.predict_visual(visual_features)
        else:
            modality_breakdown["visual"] = ModalityPrediction(
                modality="visual", available=False, model_name="None", status="NO_INPUT_SUPPLIED"
            )

        # Check available probabilities
        available_probs = [
            m.calibrated_probability
            for m in modality_breakdown.values()
            if m.available and m.calibrated_probability is not None
        ]

        if available_probs:
            final_prob = float(np.mean(available_probs))
            indicator = "Elevated screening indicators" if final_prob >= 0.50 else "Lower-risk indicators"
            risk_level = "ELEVATED" if final_prob >= 0.50 else "LOW"
            fusion_applied = len(available_probs) > 1
        else:
            final_prob = None
            indicator = "Model training checkpoint pending official dataset download"
            risk_level = "PENDING_CHECKPOINTS"
            fusion_applied = False

        return DaicWozPredictionResponse(
            screening_indicator=indicator,
            calibrated_probability=round(final_prob, 4) if final_prob is not None else None,
            risk_level=risk_level,
            confidence_interval={"lower_bound": 0.0, "upper_bound": 0.0} if final_prob is not None else None,
            modality_breakdown=modality_breakdown,
            fusion_applied=fusion_applied,
            model_provenance={
                "dataset": "DAIC-WOZ (Distress Analysis Interview Corpus)",
                "institution": "USC Institute for Creative Technologies",
                "training_split": "Participant-stratified zero-leakage development partition",
                "target_definition": "PHQ-8 score >= 10 (Moderate to Severe Depressive Symptoms)"
            }
        )
