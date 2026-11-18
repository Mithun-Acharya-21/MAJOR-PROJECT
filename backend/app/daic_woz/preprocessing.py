"""
DAIC-WOZ Input Preprocessing & Feature Extraction Service
NeuroScreen Major Project - Phase 10 Preprocessing
"""

import re
from pathlib import Path
from typing import List, Optional
import numpy as np

try:
    import opensmile
    HAS_OPENSMILE = True
except ImportError:
    HAS_OPENSMILE = False


def preprocess_transcript_text(raw_text: str) -> str:
    """Cleans and standardizes raw transcript speech while preserving emotional phrasing."""
    if not raw_text or not raw_text.strip():
        return ""
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", raw_text).strip()
    return text


def extract_audio_egemaps_from_buffer(audio_bytes: bytes, temp_wav_path: Path) -> np.ndarray:
    """Writes temporary audio file and extracts eGeMAPSv02 88-dimensional features."""
    if not HAS_OPENSMILE:
        return np.zeros(88, dtype=np.float32)
    try:
        temp_wav_path.write_bytes(audio_bytes)
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        df = smile.process_file(str(temp_wav_path))
        return df.values.flatten()[:88]
    except Exception as e:
        print(f"[-] Preprocessing error for audio: {e}")
        return np.zeros(88, dtype=np.float32)
    finally:
        if temp_wav_path.exists():
            temp_wav_path.unlink()


def validate_visual_feature_vector(features: List[float]) -> np.ndarray:
    """Validates and formats the 40-dimensional OpenFace summary feature vector."""
    arr = np.array(features, dtype=np.float32)
    if len(arr) < 40:
        arr = np.pad(arr, (0, 40 - len(arr)))
    return arr[:40]
