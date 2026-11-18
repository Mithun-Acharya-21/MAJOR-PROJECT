"""
DAIC-WOZ Safe Model Loader & Registry
NeuroScreen Major Project - Phase 10 Model Loader
"""

from pathlib import Path
from typing import Optional, Dict, Any
import joblib
import torch


class DaicWozModelLoader:
    _instance = None

    def __init__(self):
        self.base_dir = Path("models/daic_woz")
        self.text_model_path = self.base_dir / "text" / "tfidf_logreg_model.joblib"
        self.text_vectorizer_path = self.base_dir / "text" / "tfidf_vectorizer.joblib"
        self.audio_model_path = self.base_dir / "audio" / "egemaps_logreg_model.joblib"
        self.audio_scaler_path = self.base_dir / "audio" / "egemaps_scaler.joblib"
        self.visual_model_path = self.base_dir / "visual" / "visual_kinematics_model.joblib"
        self.visual_scaler_path = self.base_dir / "visual" / "visual_scaler.joblib"
        self.fusion_model_path = self.base_dir / "fusion" / "late_fusion_model.joblib"

        self.text_model = None
        self.text_vectorizer = None
        self.audio_model = None
        self.audio_scaler = None
        self.visual_model = None
        self.visual_scaler = None
        self.fusion_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.load_models()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DaicWozModelLoader()
        return cls._instance

    def load_models(self):
        """Loads available checkpoints into memory safely."""
        if self.text_model_path.exists() and self.text_vectorizer_path.exists():
            try:
                self.text_model = joblib.load(self.text_model_path)
                self.text_vectorizer = joblib.load(self.text_vectorizer_path)
            except Exception as e:
                print(f"[-] Failed loading text model: {e}")

        if self.audio_model_path.exists() and self.audio_scaler_path.exists():
            try:
                self.audio_model = joblib.load(self.audio_model_path)
                self.audio_scaler = joblib.load(self.audio_scaler_path)
            except Exception as e:
                print(f"[-] Failed loading audio model: {e}")

        if self.visual_model_path.exists() and self.visual_scaler_path.exists():
            try:
                self.visual_model = joblib.load(self.visual_model_path)
                self.visual_scaler = joblib.load(self.visual_scaler_path)
            except Exception as e:
                print(f"[-] Failed loading visual model: {e}")

        if self.fusion_model_path.exists():
            try:
                self.fusion_model = joblib.load(self.fusion_model_path)
            except Exception as e:
                print(f"[-] Failed loading fusion model: {e}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "text_model_loaded": self.text_model is not None,
            "audio_model_loaded": self.audio_model is not None,
            "visual_model_loaded": self.visual_model is not None,
            "fusion_model_loaded": self.fusion_model is not None,
            "device": self.device,
            "dataset_acquisition_status": "PENDING_OFFICIAL_DOWNLOAD" if self.text_model is None else "CHECKPOINTS_LOADED",
            "active_checkpoints": {
                "text": str(self.text_model_path) if self.text_model_path.exists() else None,
                "audio": str(self.audio_model_path) if self.audio_model_path.exists() else None,
                "visual": str(self.visual_model_path) if self.visual_model_path.exists() else None,
                "fusion": str(self.fusion_model_path) if self.fusion_model_path.exists() else None,
            }
        }
