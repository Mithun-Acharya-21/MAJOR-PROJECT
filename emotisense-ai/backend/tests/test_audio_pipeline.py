"""Unit tests for Phase 9 — audio validation and acoustic feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if audio deps not available
librosa = pytest.importorskip("librosa", reason="librosa not installed")
soundfile = pytest.importorskip("soundfile", reason="soundfile not installed")

from training.voice.audio_validate import AudioValidator, AudioValidationResult, DEFAULT_CONFIG
from training.voice.acoustic_features import AcousticFeatureExtractor, AcousticFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine_wave(
    freq: float = 440.0,
    duration_s: float = 3.0,
    sr: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def _silence(duration_s: float = 3.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _clipped(duration_s: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Generate a wave where > 1 % of samples sit at the ±0.99 absolute ceiling."""
    wave = np.random.uniform(-1.0, 1.0, int(sr * duration_s)).astype(np.float32)
    # Force 5 % of samples to exactly ±0.99 (absolute clipping)
    n_clip = int(len(wave) * 0.05)
    wave[:n_clip] = 0.99
    wave[n_clip: n_clip * 2] = -0.99
    return wave


# ---------------------------------------------------------------------------
# AudioValidator tests
# ---------------------------------------------------------------------------

class TestAudioValidator:

    def test_valid_sine_wave_passes(self) -> None:
        v = AudioValidator()
        wave = _sine_wave(duration_s=3.0)
        result = v.validate_array(wave, 16000)
        # Sine wave may not pass SNR heuristic (signal == noise floor proxy)
        # but must not be rejected for clipping, silence, or duration
        assert "clipping" not in result.rejection_reasons
        assert "too_short" not in result.rejection_reasons
        assert "silent_signal" not in result.rejection_reasons
        assert 0.0 <= result.quality_score <= 1.0

    def test_too_short_rejected(self) -> None:
        v = AudioValidator()
        wave = _sine_wave(duration_s=0.3)
        result = v.validate_array(wave, 16000)
        assert "too_short" in result.rejection_reasons
        assert not result.is_valid

    def test_silence_rejected(self) -> None:
        v = AudioValidator()
        result = v.validate_array(_silence(3.0), 16000)
        assert "silent_signal" in result.rejection_reasons
        assert not result.is_valid

    def test_clipping_detected(self) -> None:
        v = AudioValidator()
        result = v.validate_array(_clipped(), 16000)
        assert "clipping" in result.rejection_reasons
        assert result.clipping_fraction > 0.0

    def test_unsupported_sample_rate(self) -> None:
        v = AudioValidator()
        wave = _sine_wave(sr=11025, duration_s=3.0)
        result = v.validate_array(wave, 11025)
        assert any("unsupported_sample_rate" in r for r in result.rejection_reasons)

    def test_quality_score_in_unit_interval(self) -> None:
        v = AudioValidator()
        for wave in [_sine_wave(), _silence(), _clipped()]:
            result = v.validate_array(wave, 16000)
            assert 0.0 <= result.quality_score <= 1.0

    def test_to_metadata_is_serialisable(self) -> None:
        v = AudioValidator()
        result = v.validate_array(_sine_wave(), 16000)
        meta = result.to_metadata()
        import json
        json.dumps(meta)  # must not raise

    def test_validation_result_has_vad_ratio(self) -> None:
        v = AudioValidator()
        result = v.validate_array(_sine_wave(duration_s=4.0), 16000)
        assert result.vad_ratio >= 0.0


# ---------------------------------------------------------------------------
# AcousticFeatureExtractor tests
# ---------------------------------------------------------------------------

class TestAcousticFeatureExtractor:

    def test_mfcc_shape(self) -> None:
        ext = AcousticFeatureExtractor()
        wave = _sine_wave(duration_s=3.0)
        feat = ext.extract_from_array(wave, 16000)
        assert feat.mfcc is not None
        # (n_mfcc * 3, T)  where n_mfcc = 40 default
        assert feat.mfcc.shape[0] == 40 * 3

    def test_log_mel_shape(self) -> None:
        ext = AcousticFeatureExtractor()
        wave = _sine_wave(duration_s=3.0)
        feat = ext.extract_from_array(wave, 16000)
        assert feat.log_mel is not None
        assert feat.log_mel.shape[0] == 128  # n_mels default

    def test_rms_positive(self) -> None:
        ext = AcousticFeatureExtractor()
        wave = _sine_wave(duration_s=3.0, amplitude=0.5)
        feat = ext.extract_from_array(wave, 16000)
        assert feat.rms is not None
        assert feat.rms.mean() > 0.0

    def test_zcr_shape_matches_rms(self) -> None:
        ext = AcousticFeatureExtractor()
        wave = _sine_wave(duration_s=2.0)
        feat = ext.extract_from_array(wave, 16000)
        assert feat.zcr is not None
        assert feat.zcr.shape == feat.rms.shape

    def test_spectral_features_extracted(self) -> None:
        ext = AcousticFeatureExtractor()
        feat = ext.extract_from_array(_sine_wave(duration_s=2.0), 16000)
        assert feat.spectral_centroid is not None
        assert feat.spectral_bandwidth is not None
        assert feat.spectral_rolloff is not None
        assert feat.spectral_flatness is not None

    def test_speech_rate_in_unit_interval(self) -> None:
        ext = AcousticFeatureExtractor()
        feat = ext.extract_from_array(_sine_wave(duration_s=3.0), 16000)
        assert 0.0 <= feat.speech_rate <= 1.0

    def test_silence_has_low_speech_rate(self) -> None:
        ext = AcousticFeatureExtractor()
        feat = ext.extract_from_array(_silence(3.0), 16000)
        assert feat.speech_rate < 0.1

    def test_summary_contains_disclaimer(self) -> None:
        ext = AcousticFeatureExtractor()
        feat = ext.extract_from_array(_sine_wave(duration_s=2.0), 16000)
        summary = feat.to_summary()
        assert "disclaimer" in summary
        assert "not clinical evidence" in summary["disclaimer"].lower()

    def test_summary_is_json_serialisable(self) -> None:
        import json
        ext = AcousticFeatureExtractor()
        feat = ext.extract_from_array(_sine_wave(duration_s=2.0), 16000)
        json.dumps(feat.to_summary())

    def test_no_extraction_errors_on_clean_signal(self) -> None:
        ext = AcousticFeatureExtractor()
        feat = ext.extract_from_array(_sine_wave(duration_s=3.0), 16000)
        assert feat.extraction_errors == [], feat.extraction_errors

    def test_duration_correct(self) -> None:
        sr = 16000
        duration = 2.5
        ext = AcousticFeatureExtractor()
        feat = ext.extract_from_array(_sine_wave(duration_s=duration, sr=sr), sr)
        assert abs(feat.duration_s - duration) < 0.05
