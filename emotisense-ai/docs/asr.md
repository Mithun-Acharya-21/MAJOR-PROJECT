# Phase 11 — ASR Integration and Transcript Quality Handling

## Design

All transcription runs **locally and offline** using OpenAI Whisper.
No audio is sent to any external service.

## Voice branch pipeline

```
Audio (file / bytes / waveform)
        ↓
  AudioValidator (Phase 9)
        ├── is_valid?  No → return {status: "invalid_audio"}
        └── quality_score  → component of combined_quality_score
        ↓
  WhisperASR
        ├── Load audio (whisper.load_audio or resample)
        ├── Transcribe (offline, local weights)
        ├── Extract: transcript, language, language_prob, no_speech_prob,
        │           word_segments (word timestamps), word_count
        ├── Quality checks → rejection_reasons
        └── quality_score → component of combined_quality_score
        ↓
  assess_transcript_quality()
        ├── word_count, sentence_count, avg_word_length
        ├── repetition_ratio (consecutive duplicates)
        └── quality_score → component of combined_quality_score
        ↓
  combined_quality_score = 0.4×audio + 0.4×asr + 0.2×transcript
        → multimodal fusion gate (Phase 14)
```

## ASR rejection codes

| Code | Meaning |
|---|---|
| `low_language_confidence` | language_prob < 0.70 |
| `likely_no_speech` | no_speech_prob > 0.50 |
| `too_few_words` | word_count < 3 |
| `unsupported_language:XX` | Language not in allowed list |
| `empty_transcript` | Whisper returned empty string |
| `transcription_error:…` | Internal Whisper exception |

## Model sizes

| Size | Parameters | Relative speed | Recommended use |
|---|---|---|---|
| `tiny` | 39 M | Fastest | Testing, CPU demo |
| `base` | 74 M | Fast | Development default |
| `small` | 244 M | Moderate | Higher accuracy |
| `medium` | 769 M | Slow | Near-clinical quality |
| `large` | 1550 M | Slowest | Maximum accuracy |

Default is `base`. Change via `ASR_MODEL_SIZE` env var or config dict.

## Privacy controls

- Audio is processed in-memory; no temporary WAV files are written.
- Only the transcript and metadata are returned; the raw waveform is
  not persisted.
- Word timestamps are optional (default: enabled) and support
  explainability (Phase 17) without retaining audio.

## Downstream use

| Output | Consumer |
|---|---|
| `transcript` | DeBERTa text branch (Phase 5) |
| `language` | Safety check (non-target language warning) |
| `combined_quality_score` | Multimodal fusion gate (Phase 14) |
| `word_segments` | Explainability timestamp overlay (Phase 17) |
| `is_usable` | Gate: False → voice text path zeroed out in fusion |
