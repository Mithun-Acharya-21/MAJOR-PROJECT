"""Phase 9 — Acoustic feature extraction for speech-emotion modelling.

Extracted features
------------------
- MFCCs (default 40 coefficients) + delta + delta-delta
- Log-mel spectrogram
- Pitch / F0 (via librosa's pyin)
- Energy (frame-level RMS)
- Zero-crossing rate (ZCR)
- Spectral centroid, bandwidth, rolloff, flatness, contrast
- Pauses (silence segments) and speech rate (syllable/second proxy)

Outputs
-------
AcousticFeatures dataclass:
  - mfcc          : (n_mfcc × 3, T) — stacked MFCC + Δ + ΔΔ
  - log_mel       : (n_mels, T)
  - pitch_hz      : (T,) with 0 where unvoiced
  - pitch_voiced  : (T,) boolean mask
  - rms           : (T,) energy per frame
  - zcr           : (T,)
  - spectral_*    : (T,) centroid, bandwidth, rolloff, flatness
  - pause_count   : int — number of silence segments
  - pause_total_s : float — total silence duration
  - speech_rate   : float — voiced-frame ratio (proxy for rate)
  - quality_meta  : dict from AudioValidationResult

These are model-observed acoustic patterns; they must not be labelled
as clinical evidence.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_ACOUSTIC_CONFIG: dict = {
    "sample_rate": 16000,
    "n_mfcc": 40,
    "n_mels": 128,
    "n_fft": 512,
    "hop_length": 160,       # 10 ms at 16 kHz
    "win_length": 400,       # 25 ms at 16 kHz
    "fmin": 60.0,
    "fmax": 8000.0,
    "pitch_fmin": 60.0,
    "pitch_fmax": 400.0,
    "silence_threshold_db": -40.0,
    "silence_min_duration_s": 0.2,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AcousticFeatures:
    """All acoustic features extracted from one audio segment."""

    mfcc: Optional[np.ndarray] = None             # (n_mfcc*3, T)
    log_mel: Optional[np.ndarray] = None          # (n_mels, T)
    pitch_hz: Optional[np.ndarray] = None         # (T,)
    pitch_voiced: Optional[np.ndarray] = None     # (T,) bool
    rms: Optional[np.ndarray] = None              # (T,)
    zcr: Optional[np.ndarray] = None              # (T,)
    spectral_centroid: Optional[np.ndarray] = None
    spectral_bandwidth: Optional[np.ndarray] = None
    spectral_rolloff: Optional[np.ndarray] = None
    spectral_flatness: Optional[np.ndarray] = None
    spectral_contrast: Optional[np.ndarray] = None  # (n_bands, T)
    pause_count: int = 0
    pause_total_s: float = 0.0
    speech_rate: float = 0.0    # voiced_frames / total_frames
    duration_s: float = 0.0
    sample_rate: int = 16000
    extraction_errors: list[str] = field(default_factory=list)

    def to_summary(self) -> dict:
        """Return scalar statistics for logging / metadata storage."""
        def _stats(arr: Optional[np.ndarray], name: str) -> dict:
            if arr is None or arr.size == 0:
                return {}
            a = arr[np.isfinite(arr)]
            if a.size == 0:
                return {}
            return {
                f"{name}_mean": round(float(a.mean()), 4),
                f"{name}_std": round(float(a.std()), 4),
                f"{name}_min": round(float(a.min()), 4),
                f"{name}_max": round(float(a.max()), 4),
            }

        stats: dict = {
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "pause_count": self.pause_count,
            "pause_total_s": round(self.pause_total_s, 3),
            "speech_rate": round(self.speech_rate, 4),
            "extraction_errors": self.extraction_errors,
            "disclaimer": (
                "Acoustic features are model-observed patterns. "
                "They are not clinical evidence of any mental-health condition."
            ),
        }
        # Scalar summaries only — raw arrays are not serialised
        stats.update(_stats(self.mfcc, "mfcc") if self.mfcc is not None else {})
        stats.update(_stats(self.pitch_hz, "pitch_hz"))
        stats.update(_stats(self.rms, "rms"))
        stats.update(_stats(self.spectral_centroid, "spectral_centroid"))
        return stats


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class AcousticFeatureExtractor:
    """Extract all acoustic features from a mono float32 waveform."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = {**DEFAULT_ACOUSTIC_CONFIG, **(config or {})}
        try:
            import librosa  # noqa: F401
        except ImportError as exc:
            raise ImportError("librosa is required. pip install librosa") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_path(self, audio_path: Path | str) -> AcousticFeatures:
        import librosa
        try:
            mono, sr = librosa.load(str(audio_path), sr=self.cfg["sample_rate"], mono=True)
        except Exception as exc:
            feat = AcousticFeatures()
            feat.extraction_errors.append(f"load_failed:{exc}")
            return feat
        return self.extract_from_array(mono, sr)

    def extract_from_array(
        self,
        mono: np.ndarray,
        sample_rate: int,
    ) -> AcousticFeatures:
        """Extract all features from a mono float32 waveform."""
        import librosa

        cfg = self.cfg
        # Resample if needed
        if sample_rate != cfg["sample_rate"]:
            mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=cfg["sample_rate"])
            sample_rate = cfg["sample_rate"]

        feat = AcousticFeatures(sample_rate=sample_rate, duration_s=len(mono) / sample_rate)
        hop = cfg["hop_length"]
        n_fft = cfg["n_fft"]
        win = cfg["win_length"]

        # 1. MFCCs + deltas
        try:
            mfcc = librosa.feature.mfcc(
                y=mono, sr=sample_rate,
                n_mfcc=cfg["n_mfcc"], n_fft=n_fft, hop_length=hop, win_length=win,
            )
            delta1 = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            feat.mfcc = np.concatenate([mfcc, delta1, delta2], axis=0)
        except Exception as exc:
            feat.extraction_errors.append(f"mfcc:{exc}")

        # 2. Log-mel spectrogram
        try:
            mel = librosa.feature.melspectrogram(
                y=mono, sr=sample_rate,
                n_mels=cfg["n_mels"], n_fft=n_fft, hop_length=hop,
                fmin=cfg["fmin"], fmax=cfg["fmax"],
            )
            feat.log_mel = librosa.power_to_db(mel, ref=np.max)
        except Exception as exc:
            feat.extraction_errors.append(f"log_mel:{exc}")

        # 3. Pitch / F0 via pyin
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f0, voiced_flag, _ = librosa.pyin(
                    mono,
                    fmin=cfg["pitch_fmin"],
                    fmax=cfg["pitch_fmax"],
                    sr=sample_rate,
                    hop_length=hop,
                )
            feat.pitch_hz = np.nan_to_num(f0, nan=0.0)
            feat.pitch_voiced = voiced_flag.astype(bool)
        except Exception as exc:
            feat.extraction_errors.append(f"pitch:{exc}")

        # 4. Energy (RMS)
        try:
            feat.rms = librosa.feature.rms(
                y=mono, frame_length=win, hop_length=hop
            ).squeeze()
        except Exception as exc:
            feat.extraction_errors.append(f"rms:{exc}")

        # 5. Zero-crossing rate
        try:
            feat.zcr = librosa.feature.zero_crossing_rate(
                mono, frame_length=win, hop_length=hop
            ).squeeze()
        except Exception as exc:
            feat.extraction_errors.append(f"zcr:{exc}")

        # 6. Spectral features
        try:
            feat.spectral_centroid = librosa.feature.spectral_centroid(
                y=mono, sr=sample_rate, n_fft=n_fft, hop_length=hop
            ).squeeze()
            feat.spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=mono, sr=sample_rate, n_fft=n_fft, hop_length=hop
            ).squeeze()
            feat.spectral_rolloff = librosa.feature.spectral_rolloff(
                y=mono, sr=sample_rate, n_fft=n_fft, hop_length=hop
            ).squeeze()
            feat.spectral_flatness = librosa.feature.spectral_flatness(
                y=mono, n_fft=n_fft, hop_length=hop
            ).squeeze()
            feat.spectral_contrast = librosa.feature.spectral_contrast(
                y=mono, sr=sample_rate, n_fft=n_fft, hop_length=hop
            )
        except Exception as exc:
            feat.extraction_errors.append(f"spectral:{exc}")

        # 7. Pause / VAD analysis
        try:
            feat.pause_count, feat.pause_total_s, feat.speech_rate = (
                self._analyse_pauses(mono, sample_rate)
            )
        except Exception as exc:
            feat.extraction_errors.append(f"pauses:{exc}")

        return feat

    # ------------------------------------------------------------------
    # Pause / speech-rate analysis
    # ------------------------------------------------------------------

    def _analyse_pauses(
        self, mono: np.ndarray, sr: int
    ) -> tuple[int, float, float]:
        """Return (pause_count, pause_total_s, voiced_ratio)."""
        import librosa

        cfg = self.cfg
        hop = cfg["hop_length"]
        win = cfg["win_length"]
        rms = librosa.feature.rms(y=mono, frame_length=win, hop_length=hop).squeeze()
        rms_db = 20 * np.log10(rms + 1e-9)
        silence_mask = rms_db < cfg["silence_threshold_db"]

        frame_duration = hop / sr
        min_silence_frames = max(1, int(cfg["silence_min_duration_s"] / frame_duration))

        pause_count = 0
        pause_total_frames = 0
        in_pause = False
        current_len = 0

        for is_silent in silence_mask:
            if is_silent:
                current_len += 1
                in_pause = True
            else:
                if in_pause and current_len >= min_silence_frames:
                    pause_count += 1
                    pause_total_frames += current_len
                current_len = 0
                in_pause = False
        # Trailing pause
        if in_pause and current_len >= min_silence_frames:
            pause_count += 1
            pause_total_frames += current_len

        pause_total_s = pause_total_frames * frame_duration
        voiced_ratio = float(1.0 - silence_mask.mean())
        return pause_count, round(pause_total_s, 3), round(voiced_ratio, 4)
