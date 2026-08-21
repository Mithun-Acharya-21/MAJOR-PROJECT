# Audio Pipeline — Phase 9 & 10

## Phase 9 — Validation and acoustic feature extraction

### Audio validation chain

```
Input (file path / bytes / numpy array)
        ↓
  AudioValidator
        ├── Duration check  (min 1 s, max 300 s)
        ├── Sample-rate check  (8/16/22.05/44.1/48 kHz allowed)
        ├── Clipping detection  (absolute ≥ 0.99 ceiling, > 1 % → rejected)
        ├── RMS level measurement
        ├── SNR estimate  (first 0.5 s as noise floor proxy)
        ├── Voice-activity ratio  (energy-based VAD)
        └── Composite quality_score ∈ [0, 1]
        ↓
  AudioValidationResult
        ├── is_valid (bool)
        ├── quality_score → multimodal fusion gate (Phase 14)
        ├── rejection_reasons []
        └── metadata (sr, duration, n_channels, vad_ratio, snr_db, rms_db)
```

### Rejection codes

| Code | Meaning |
|---|---|
| `too_short` | < 1 s |
| `too_long` | > 300 s |
| `unsupported_sample_rate:N` | Not in allowed list |
| `clipping` | > 1 % of samples at ±0.99 absolute ceiling |
| `silent_signal` | Peak = 0 |
| `low_snr` | Estimated SNR < 5 dB |
| `insufficient_voice_activity` | VAD ratio < 5 % |

### Acoustic features extracted

| Feature | Shape | Description |
|---|---|---|
| `mfcc` | (120, T) | 40 MFCCs + Δ + ΔΔ |
| `log_mel` | (128, T) | Log-mel spectrogram |
| `pitch_hz` | (T,) | F0 via pyin; 0 where unvoiced |
| `pitch_voiced` | (T,) bool | Voiced frame mask |
| `rms` | (T,) | Frame-level energy |
| `zcr` | (T,) | Zero-crossing rate |
| `spectral_centroid` | (T,) | Frequency centroid |
| `spectral_bandwidth` | (T,) | Spectral spread |
| `spectral_rolloff` | (T,) | 85 % rolloff frequency |
| `spectral_flatness` | (T,) | Tonality vs. noise |
| `spectral_contrast` | (7, T) | Valley–peak contrast per sub-band |
| `pause_count` | scalar | Number of silence segments ≥ 0.2 s |
| `pause_total_s` | scalar | Total silence duration |
| `speech_rate` | scalar | Voiced-frame ratio |

Frame: 25 ms window, 10 ms hop, 16 kHz target sample rate.

### Important disclaimers

- All acoustic features are **model-observed patterns**.
- They must not be labelled as clinical evidence of any mental-health condition.
- The `to_summary()` method includes a mandatory disclaimer in its output dict.

## Phase 10 — Speech emotion model

See [speech-emotion.md](speech-emotion.md).
