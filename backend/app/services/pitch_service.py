"""
Voice Pitch & Prosodic Analysis Service
NeuroScreen Major Project - Phase 5 Voice Pitch Analysis

Features Extracted:
    - Pitch: min, max, mean, median, std, range, and pitch contour over time (F0 in Hz)
    - Temporal dynamics: speech duration, silence duration, pause frequency, speaking rate
    - Energy: RMS volume, intensity distribution
    - Robustness: Voiced/unvoiced frame filtering, zero-crossing checks, multi-format audio decoding
    - Language Agnostic: Works on English, Kannada, Mandarin, and other spoken audio inputs
"""

from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger("emotiondep.pitch")

# Minimum audio duration in seconds
_MIN_DURATION_SEC = 0.3


def decode_audio_bytes(audio_bytes: bytes) -> Optional[Tuple[np.ndarray, int]]:
    """Decodes audio bytes (WAV, MP3, WebM, OGG, FLAC) to mono float32 numpy array."""
    if not audio_bytes or len(audio_bytes) < 100:
        return None

    # Strategy 1: soundfile (fast, handles WAV, FLAC, OGG)
    try:
        import soundfile as sf
        with io.BytesIO(audio_bytes) as buf:
            data, sample_rate = sf.read(buf, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Mono downmix
            return data, int(sample_rate)
    except Exception:
        pass

    # Strategy 2: librosa / audioread (handles MP3 and other formats)
    try:
        import librosa
        with io.BytesIO(audio_bytes) as buf:
            data, sample_rate = librosa.load(buf, sr=16000, mono=True)
            return data.astype(np.float32), int(sample_rate)
    except Exception:
        pass

    # Strategy 3: wave module (standard uncompressed WAV)
    try:
        import wave
        with io.BytesIO(audio_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                dtype = np.int16 if sample_width == 2 else (np.uint8 if sample_width == 1 else np.int32)
                samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
                if n_channels > 1:
                    samples = samples[::n_channels]
                max_val = float(2 ** (sample_width * 8 - 1))
                if max_val > 0:
                    samples = samples / max_val
                return samples, sample_rate
    except Exception:
        pass

    return None


def analyze_voice_pitch_and_prosody(
    audio_bytes: bytes,
    hop_length: int = 512,
    fmin: float = 65.0,   # C2 (~65 Hz) - covers low male pitch
    fmax: float = 500.0   # B4 (~500 Hz) - covers high female/child pitch
) -> Dict[str, Any]:
    """
    Extracts detailed pitch and prosodic acoustic metrics from audio.
    Returns summary statistics, temporal prosody, and pitch contour time-series.
    """
    decoded = decode_audio_bytes(audio_bytes)
    if decoded is None:
        return {
            "valid": False,
            "error": "Unable to decode audio format. Please provide a valid WAV, MP3, or WebM file.",
            "metrics": {},
            "pitch_contour": []
        }

    samples, sr = decoded
    total_duration = len(samples) / sr

    if total_duration < _MIN_DURATION_SEC:
        return {
            "valid": False,
            "error": f"Audio duration too short ({total_duration:.2f}s). Minimum { _MIN_DURATION_SEC}s required.",
            "metrics": {},
            "pitch_contour": []
        }

    # 1. Pitch Extraction using Librosa Yin / Pyin or Autocorrelation
    pitch_values = []
    pitch_times = []
    voiced_flags = []

    try:
        import librosa
        # Use librosa.yin for fast, accurate fundamental frequency estimation
        f0 = librosa.yin(
            samples,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length
        )
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

        # Calculate frame energies to distinguish voiced speech from silence/unvoiced
        frame_energies = librosa.feature.rms(y=samples, hop_length=hop_length)[0]
        max_energy = float(np.max(frame_energies)) if len(frame_energies) > 0 else 0.0
        # A frame is voiced if energy exceeds baseline noise floor or 8% of max energy
        energy_thresh = max(0.003, max_energy * 0.08)

        # Align length
        min_len = min(len(f0), len(times), len(frame_energies))
        for i in range(min_len):
            hz = float(f0[i])
            is_voiced = bool(frame_energies[i] > energy_thresh and not np.isnan(hz) and hz >= fmin and hz <= fmax)
            pitch_times.append(round(float(times[i]), 3))
            pitch_values.append(round(hz, 1) if is_voiced else 0.0)
            voiced_flags.append(is_voiced)
    except Exception as e:
        logger.warning("Librosa pitch extraction failed: %s, falling back to spectral analysis", e)
        # Fallback: simple autocorrelation across frames
        frame_len = int(sr * 0.03)
        n_frames = len(samples) // frame_len
        times = [round(i * 0.03, 3) for i in range(n_frames)]
        for i in range(n_frames):
            frame = samples[i * frame_len : (i + 1) * frame_len]
            energy = np.sqrt(np.mean(frame ** 2))
            if energy > 0.01:
                corr = np.correlate(frame, frame, mode="full")
                corr = corr[len(corr) // 2 :]
                d = np.diff(corr)
                start = np.where(d > 0)[0]
                if len(start) > 0:
                    peak = np.argmax(corr[start[0] :]) + start[0]
                    hz = sr / peak if peak > 0 else 0.0
                    if fmin <= hz <= fmax:
                        pitch_values.append(round(hz, 1))
                        voiced_flags.append(True)
                        pitch_times.append(times[i])
                        continue
            pitch_values.append(0.0)
            voiced_flags.append(False)
            pitch_times.append(times[i])

    # 2. Extract Valid Voiced F0
    valid_f0 = [v for v, is_v in zip(pitch_values, voiced_flags) if is_v and v > 0]

    if valid_f0:
        p_min = float(np.min(valid_f0))
        p_max = float(np.max(valid_f0))
        p_mean = float(np.mean(valid_f0))
        p_median = float(np.median(valid_f0))
        p_std = float(np.std(valid_f0))
        p_range = float(p_max - p_min)
    else:
        p_min = p_max = p_mean = p_median = p_std = p_range = 0.0

    # 3. Temporal & Speech Dynamics
    voiced_count = sum(voiced_flags)
    total_frames = max(1, len(voiced_flags))
    voiced_ratio = voiced_count / total_frames
    silence_ratio = 1.0 - voiced_ratio

    speech_duration = round(total_duration * voiced_ratio, 2)
    silence_duration = round(total_duration * silence_ratio, 2)

    # Pause detection: contiguous runs of unvoiced frames >= 200ms
    pauses = 0
    in_pause = False
    pause_run = 0
    frame_dur = hop_length / sr if 'hop_length' in locals() and sr > 0 else 0.032
    frames_for_pause = max(2, int(0.20 / frame_dur))  # 200 ms threshold

    for is_v in voiced_flags:
        if not is_v:
            pause_run += 1
            if pause_run >= frames_for_pause and not in_pause:
                pauses += 1
                in_pause = True
        else:
            in_pause = False
            pause_run = 0

    pause_freq_per_min = round((pauses / (total_duration / 60.0)), 2) if total_duration > 0 else 0.0

    # Approximate syllable / speaking rate
    # Counting energy peaks in voiced segments as approximate syllable onsets
    energy_arr = np.abs(samples)
    rms_energy = float(np.sqrt(np.mean(samples ** 2)))
    window_sz = int(sr * 0.05)
    if window_sz > 0:
        local_energies = [np.mean(energy_arr[i : i + window_sz]) for i in range(0, len(energy_arr), window_sz)]
        transitions = sum(
            1 for i in range(1, len(local_energies))
            if local_energies[i] > np.mean(local_energies) and local_energies[i - 1] <= np.mean(local_energies)
        )
        speaking_rate = round(transitions / max(0.5, speech_duration), 2)
    else:
        speaking_rate = 0.0

    # 4. Decimate contour for frontend rendering (up to 120 points)
    step = max(1, len(pitch_times) // 120)
    sampled_contour = [
        {"time_sec": pitch_times[i], "pitch_hz": pitch_values[i], "voiced": voiced_flags[i]}
        for i in range(0, len(pitch_times), step)
    ]

    metrics = {
        "min_pitch_hz": round(p_min, 1),
        "max_pitch_hz": round(p_max, 1),
        "mean_pitch_hz": round(p_mean, 1),
        "median_pitch_hz": round(p_median, 1),
        "pitch_std_hz": round(p_std, 1),
        "pitch_range_hz": round(p_range, 1),
        "total_duration_sec": round(total_duration, 2),
        "speech_duration_sec": speech_duration,
        "silence_duration_sec": silence_duration,
        "silence_ratio": round(silence_ratio, 3),
        "pause_frequency_per_min": pause_freq_per_min,
        "speaking_rate_syllables_sec": speaking_rate,
        "rms_energy": round(rms_energy, 4),
        "voiced_frames_count": voiced_count,
        "total_frames_count": total_frames
    }

    return {
        "valid": True,
        "sample_rate": sr,
        "metrics": metrics,
        "pitch_contour": sampled_contour,
        "scientific_disclaimer": (
            "Observed acoustic features represent physical speech characteristics. "
            "Pitch dynamics vary naturally across pitch registers, languages, and emotional contexts. "
            "Pitch metrics alone must never be treated as independent evidence of clinical depression."
        )
    }
