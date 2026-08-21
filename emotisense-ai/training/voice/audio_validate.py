"""Phase 9 — Audio validation: format, duration, clipping, noise, sample-rate,
and voice-activity ratio checks.

All checks are non-destructive and run in-memory.  Raw audio is never persisted
by this module.

Outputs a structured AudioValidationResult with:
  - is_valid (bool)
  - quality_score  ∈ [0, 1]
  - rejection_reasons (list[str])
  - metadata (sample_rate, duration_s, n_channels, bit_depth, file_format)

The quality_score is passed to the multimodal fusion gate (Phase 14).
A score of 0.0 means the voice branch contributes zero weight to fusion.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "min_duration_s": 1.0,          # Minimum acceptable duration in seconds
    "max_duration_s": 300.0,        # 5 minutes maximum
    "target_sample_rate": 16000,    # Model-expected sample rate
    "allowed_sample_rates": [8000, 16000, 22050, 44100, 48000],
    "clipping_threshold": 0.99,     # Fraction of max amplitude → clipping
    "clipping_fraction_max": 0.01,  # Max proportion of clipped samples allowed
    "min_vad_ratio": 0.05,          # Minimum proportion of voiced frames
    "snr_floor_db": 5.0,            # Minimum acceptable SNR estimate (dB)
    "vad_frame_length": 0.025,      # VAD frame length in seconds
    "vad_hop_length": 0.010,        # VAD hop length in seconds
    "vad_energy_threshold_db": -40.0,  # Energy below this → silence frame
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AudioValidationResult:
    is_valid: bool = False
    quality_score: float = 0.0       # Composite [0, 1]; fed to fusion gate
    rejection_reasons: list[str] = field(default_factory=list)
    # Audio metadata
    sample_rate: Optional[int] = None
    duration_s: Optional[float] = None
    n_channels: Optional[int] = None
    # Quality indicators
    clipping_fraction: float = 0.0
    vad_ratio: float = 0.0           # Proportion of voiced frames
    snr_db: Optional[float] = None
    rms_db: Optional[float] = None

    def to_metadata(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "quality_score": round(self.quality_score, 4),
            "rejection_reasons": self.rejection_reasons,
            "sample_rate": self.sample_rate,
            "duration_s": round(self.duration_s, 3) if self.duration_s is not None else None,
            "n_channels": self.n_channels,
            "clipping_fraction": round(self.clipping_fraction, 4),
            "vad_ratio": round(self.vad_ratio, 4),
            "snr_db": round(self.snr_db, 2) if self.snr_db is not None else None,
            "rms_db": round(self.rms_db, 2) if self.rms_db is not None else None,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class AudioValidator:
    """Validate and quality-score a raw audio waveform or file."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_path(self, audio_path: Path | str) -> AudioValidationResult:
        """Load an audio file and validate it."""
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError("soundfile is required. pip install soundfile") from exc

        result = AudioValidationResult()
        try:
            info = sf.info(str(audio_path))
            result.sample_rate = info.samplerate
            result.duration_s = info.duration
            result.n_channels = info.channels
            audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        except Exception as exc:
            result.rejection_reasons.append(f"file_unreadable:{exc}")
            return result

        # Mix to mono for analysis
        mono = audio.mean(axis=1)
        return self._validate_waveform(mono, sr, result)

    def validate_bytes(self, data: bytes) -> AudioValidationResult:
        """Validate in-memory audio bytes (any format soundfile supports)."""
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError("soundfile is required") from exc

        result = AudioValidationResult()
        try:
            buf = io.BytesIO(data)
            audio, sr = sf.read(buf, dtype="float32", always_2d=True)
            result.sample_rate = sr
            result.duration_s = len(audio) / sr
            result.n_channels = audio.shape[1]
        except Exception as exc:
            result.rejection_reasons.append(f"decode_failed:{exc}")
            return result

        mono = audio.mean(axis=1)
        return self._validate_waveform(mono, sr, result)

    def validate_array(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> AudioValidationResult:
        """Validate a pre-loaded mono float32 waveform."""
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        result = AudioValidationResult()
        result.sample_rate = sample_rate
        result.duration_s = len(waveform) / sample_rate
        result.n_channels = 1
        return self._validate_waveform(waveform.astype(np.float32), sample_rate, result)

    # ------------------------------------------------------------------
    # Internal validation pipeline
    # ------------------------------------------------------------------

    def _validate_waveform(
        self,
        mono: np.ndarray,
        sr: int,
        result: AudioValidationResult,
    ) -> AudioValidationResult:
        cfg = self.cfg

        # 1. Duration
        dur = result.duration_s or (len(mono) / sr)
        result.duration_s = dur
        if dur < cfg["min_duration_s"]:
            result.rejection_reasons.append("too_short")
        if dur > cfg["max_duration_s"]:
            result.rejection_reasons.append("too_long")

        # 2. Sample rate
        result.sample_rate = sr
        if sr not in cfg["allowed_sample_rates"]:
            result.rejection_reasons.append(f"unsupported_sample_rate:{sr}")

        # 3. Clipping detection — compare against absolute ceiling (1.0), not
        #    relative peak, so a quiet but clean sine wave is never flagged.
        peak = float(np.max(np.abs(mono)))
        if peak > 0:
            clipped = float(np.mean(np.abs(mono) >= cfg["clipping_threshold"]))
            result.clipping_fraction = clipped
            if clipped > cfg["clipping_fraction_max"]:
                result.rejection_reasons.append("clipping")
        else:
            result.rejection_reasons.append("silent_signal")

        # 4. RMS level
        rms = float(np.sqrt(np.mean(mono ** 2)))
        result.rms_db = float(20 * np.log10(rms + 1e-9))

        # 5. SNR estimate (energy of signal vs estimated noise floor)
        result.snr_db = self._estimate_snr(mono, sr)
        if result.snr_db is not None and result.snr_db < cfg["snr_floor_db"]:
            result.rejection_reasons.append("low_snr")

        # 6. Voice-activity ratio
        result.vad_ratio = self._compute_vad_ratio(mono, sr)
        if result.vad_ratio < cfg["min_vad_ratio"]:
            result.rejection_reasons.append("insufficient_voice_activity")

        # 7. Composite quality score
        result.quality_score = self._composite_quality(result)
        result.is_valid = not bool(result.rejection_reasons)
        return result

    # ------------------------------------------------------------------
    # Quality helpers
    # ------------------------------------------------------------------

    def _compute_vad_ratio(self, mono: np.ndarray, sr: int) -> float:
        """Energy-based VAD: fraction of frames above energy threshold."""
        frame_len = max(1, int(self.cfg["vad_frame_length"] * sr))
        hop_len = max(1, int(self.cfg["vad_hop_length"] * sr))
        threshold_db = self.cfg["vad_energy_threshold_db"]

        n_frames = max(1, 1 + (len(mono) - frame_len) // hop_len)
        voiced = 0
        for i in range(n_frames):
            frame = mono[i * hop_len: i * hop_len + frame_len]
            if len(frame) == 0:
                continue
            energy_db = 20 * np.log10(float(np.sqrt(np.mean(frame ** 2))) + 1e-9)
            if energy_db > threshold_db:
                voiced += 1
        return voiced / n_frames

    @staticmethod
    def _estimate_snr(mono: np.ndarray, sr: int, noise_window_s: float = 0.5) -> Optional[float]:
        """Estimate SNR using the first ``noise_window_s`` seconds as noise floor."""
        noise_samples = int(noise_window_s * sr)
        if len(mono) < noise_samples * 2:
            return None
        noise = mono[:noise_samples]
        signal = mono[noise_samples:]
        noise_power = float(np.mean(noise ** 2)) + 1e-12
        signal_power = float(np.mean(signal ** 2)) + 1e-12
        return float(10 * np.log10(signal_power / noise_power))

    def _composite_quality(self, result: AudioValidationResult) -> float:
        """Weighted sub-scores → [0, 1]."""
        scores: list[float] = []

        # SNR score
        if result.snr_db is not None:
            snr_score = min(1.0, max(0.0, (result.snr_db - 0) / 40.0))
            scores.append(snr_score)

        # VAD score
        vad_score = min(1.0, result.vad_ratio / 0.5)
        scores.append(vad_score)

        # Clipping penalty
        clip_score = max(0.0, 1.0 - result.clipping_fraction * 20)
        scores.append(clip_score)

        # Duration normalisation (longer is generally better up to 60 s)
        if result.duration_s is not None:
            dur_score = min(1.0, result.duration_s / 60.0)
            scores.append(dur_score)

        base = float(np.mean(scores)) if scores else 0.0

        # Hard penalties for critical rejections
        critical = {"too_short", "silent_signal", "clipping", "insufficient_voice_activity"}
        penalty = sum(0.25 for r in result.rejection_reasons if r in critical)
        return round(max(0.0, min(1.0, base - penalty)), 4)
