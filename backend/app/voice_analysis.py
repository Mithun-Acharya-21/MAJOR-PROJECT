"""
Voice/Acoustic Analysis — Extract features from raw audio for depression screening.

This module provides purely acoustic/voice features (NOT text transcription).
Separate from Whisper-based speech-to-text.

Features extracted:
  - RMS energy (volume)
  - Silence ratio (fraction of near-silence frames)
  - Speaking rate estimate (based on voiced segments)
  - Duration
  - Intensity (mean absolute amplitude)
  - Pitch estimation (zero-crossing rate based approximation)

These are lightweight features that work without heavy ML models.
For production, consider using Librosa or Parselmouth for richer features.

Exports:
    extract_voice_features — bytes → VoiceFeatures
    score_voice — VoiceFeatures → depression probability proxy
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger("neuroscreen.voice")

# Minimum audio duration for meaningful analysis (seconds)
_MIN_DURATION = 0.3


def _decode_audio_to_numpy(audio_bytes: bytes) -> Optional[tuple]:
    """Decode audio bytes to numpy float32 array.

    Supports:
      - WAV (via Python wave module)
      - WebM/Opus, MP4/AAC, OGG, etc. (via pyav / libavformat)
      - Raw PCM fallback (16-bit, 16kHz)

    Returns:
        (samples_float32_mono, sample_rate) or None if decoding fails.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return None

    # --- Strategy 1: WAV decoding (fast, no dependencies) ---
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                return None

            samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if n_channels > 1:
                samples = samples[::n_channels]  # Take first channel

            max_val = float(2 ** (sample_width * 8 - 1))
            if max_val > 0:
                samples = samples / max_val

            logger.debug("Decoded WAV: %d samples, %d Hz, %d ch", len(samples), sample_rate, n_channels)
            return samples, sample_rate
    except Exception:
        pass

    # --- Strategy 2: pyav decoding (WebM/Opus, MP4, OGG, etc.) ---
    try:
        import av
        buf = io.BytesIO(audio_bytes)
        container = av.open(buf)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            logger.warning("No audio stream found in container")
            container.close()
            return None

        sample_rate = audio_stream.rate
        channels = audio_stream.channels

        # Resample to mono float32
        resampler = av.audio.resampler.AudioResampler(
            format="flt",
            layout="mono",
            rate=sample_rate,
        )

        all_frames = []
        for frame in container.decode(audio=0):
            resampled = resampler.resample(frame)
            for r in resampled:
                arr = r.to_ndarray().flatten()
                all_frames.append(arr)

        container.close()

        if not all_frames:
            logger.warning("pyav decoded 0 frames")
            return None

        samples = np.concatenate(all_frames).astype(np.float32)
        duration = len(samples) / sample_rate
        logger.debug(
            "Decoded via pyav: %d samples, %d Hz, %.2fs, peaks [%.3f, %.3f]",
            len(samples), sample_rate, duration, samples.min(), samples.max(),
        )
        return samples, sample_rate

    except Exception as exc:
        logger.warning("pyav decode failed: %s", exc)

    # --- Strategy 3: Raw PCM fallback ---
    if len(audio_bytes) > 100:
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0
            logger.debug("Decoded as raw PCM: %d samples", len(samples))
            return samples, 16000
        except Exception:
            pass

    logger.warning("All audio decode strategies failed (audio format: %s)",
                   audio_bytes[:4].hex() if len(audio_bytes) >= 4 else "too short")
    return None


def _estimate_rms_energy(samples: np.ndarray) -> float:
    """Compute root mean square energy."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def _estimate_silence_ratio(samples: np.ndarray, threshold: float = 0.01) -> float:
    """Estimate the fraction of near-silence frames."""
    if len(samples) == 0:
        return 1.0
    frame_size = 512
    n_frames = max(1, len(samples) // frame_size)
    silent_count = 0
    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(samples))
        frame = samples[start:end]
        if np.sqrt(np.mean(frame ** 2)) < threshold:
            silent_count += 1
    return silent_count / n_frames


def _estimate_speaking_rate(samples: np.ndarray, sample_rate: int) -> float:
    """Rough speaking rate estimate based on voiced segment detection.

    Uses energy-based voicing detection: counts transitions from
    silence to speech as approximate syllable onsets.
    """
    if len(samples) == 0 or sample_rate == 0:
        return 0.0

    frame_size = int(sample_rate * 0.02)  # 20ms frames
    if frame_size < 1:
        frame_size = 1

    n_frames = len(samples) // frame_size
    if n_frames < 3:
        return 0.0

    # Compute frame energies
    energies = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(samples))
        energies[i] = np.sqrt(np.mean(samples[start:end] ** 2))

    # Adaptive threshold: mean of energies
    threshold = np.mean(energies) * 0.3
    voiced = energies > threshold

    # Count transitions (silence → speech) as syllable onsets
    transitions = 0
    for i in range(1, n_frames):
        if voiced[i] and not voiced[i - 1]:
            transitions += 1

    duration = len(samples) / sample_rate
    if duration < _MIN_DURATION:
        return 0.0

    # Syllables per second (rough: English avg ~4-5 syllables/sec in normal speech)
    return transitions / duration


def _estimate_intensity(samples: np.ndarray) -> float:
    """Mean absolute amplitude as intensity proxy."""
    if len(samples) == 0:
        return 0.0
    return float(np.mean(np.abs(samples)))


def _estimate_pitch_proxy(samples: np.ndarray, sample_rate: int) -> tuple:
    """Rough pitch estimation using zero-crossing rate.

    Returns:
        (estimated_freq_hz, variation_std)

    This is a rough proxy — real pitch detection requires autocorrelation
    or YIN algorithm. Zero-crossing rate correlates with pitch for
    voiced speech segments.
    """
    if len(samples) < sample_rate * 0.1:  # Need at least 100ms
        return 0.0, 0.0

    frame_size = int(sample_rate * 0.03)  # 30ms frames
    if frame_size < 2:
        return 0.0, 0.0

    n_frames = len(samples) // frame_size
    if n_frames < 2:
        return 0.0, 0.0

    zcr_values = []
    for i in range(n_frames):
        start = i * frame_size
        end = min(start + frame_size, len(samples))
        frame = samples[start:end]

        # Only consider voiced frames (energy above threshold)
        energy = np.sqrt(np.mean(frame ** 2))
        if energy < 0.01:
            continue

        # Zero-crossing rate
        crossings = np.sum(np.abs(np.diff(np.sign(frame))) > 0)
        zcr = crossings / (2.0 * len(frame))
        # Approximate frequency from ZCR (rough but directionally useful)
        freq = zcr * sample_rate / 2.0
        zcr_values.append(freq)

    if not zcr_values:
        return 0.0, 0.0

    mean_freq = float(np.mean(zcr_values))
    std_freq = float(np.std(zcr_values))
    return mean_freq, std_freq


def extract_voice_features(audio_bytes: bytes) -> Optional[tuple]:
    """Extract acoustic voice features from raw audio bytes.

    Args:
        audio_bytes: Raw audio file content (WAV preferred, raw PCM fallback).

    Returns:
        (VoiceFeatures, depression_score) or None if audio cannot be decoded.

    The depression_score is a proxy heuristic based on acoustic markers:
      - High silence ratio → elevated risk signal
      - Low speaking rate → elevated risk signal
      - Low energy → elevated risk signal
      - High pitch variation → lower risk signal (expressiveness)
    """
    from backend.app.session_manager import VoiceFeatures

    decoded = _decode_audio_to_numpy(audio_bytes)
    if decoded is None:
        logger.warning("Could not decode audio for voice analysis")
        return None

    samples, sample_rate = decoded
    duration = len(samples) / sample_rate

    if duration < _MIN_DURATION:
        logger.debug("Audio too short for voice analysis: %.2fs", duration)
        return None

    features = VoiceFeatures(
        rms_energy=_estimate_rms_energy(samples),
        speaking_rate=_estimate_speaking_rate(samples, sample_rate),
        silence_ratio=_estimate_silence_ratio(samples),
        pitch_mean=0.0,    # Will be set below
        pitch_variation=0.0,
        duration_sec=duration,
        intensity_mean=_estimate_intensity(samples),
    )

    pitch_mean, pitch_var = _estimate_pitch_proxy(samples, sample_rate)
    features.pitch_mean = pitch_mean
    features.pitch_variation = pitch_var

    score = score_voice(features)
    return features, score


def score_voice(features: "VoiceFeatures") -> float:
    """Compute a depression-probability proxy from voice features.

    This is a heuristic scoring function based on published acoustic
    markers of depression:
      - Slower speech rate
      - More pauses/silence
      - Lower energy/intensity
      - Reduced pitch variability (monotone)

    Returns:
        Float in [0.08, 0.92] representing estimated depression probability.
    """
    score = 0.5  # baseline

    # Silence ratio: more silence → higher risk (max +0.2)
    if features.silence_ratio > 0.3:
        score += 0.15
    elif features.silence_ratio > 0.2:
        score += 0.08
    elif features.silence_ratio < 0.05:
        score -= 0.05

    # Speaking rate: slower → higher risk (max +0.15)
    if features.speaking_rate > 0:
        if features.speaking_rate < 2.0:
            score += 0.12
        elif features.speaking_rate < 3.0:
            score += 0.05
        elif features.speaking_rate > 5.0:
            score -= 0.05

    # RMS energy: lower → higher risk (max +0.1)
    if features.rms_energy < 0.02:
        score += 0.10
    elif features.rms_energy < 0.05:
        score += 0.04
    elif features.rms_energy > 0.15:
        score -= 0.04

    # Pitch variation: less variation → higher risk (max +0.1)
    if features.pitch_variation > 0:
        if features.pitch_variation < 5.0:
            score += 0.08
        elif features.pitch_variation > 30.0:
            score -= 0.06

    # Clamp to reasonable range
    return max(0.08, min(0.92, score))
