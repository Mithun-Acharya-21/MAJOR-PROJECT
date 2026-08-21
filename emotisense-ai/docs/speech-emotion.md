# Phase 10 — Speech Emotion Model and Feature Attribution

## Architecture options

Two models are benchmarked on the same frozen emotion-labelled split:

### 1. CNN-BiLSTM (recommended for resource-constrained training)

```
MFCC + Δ + ΔΔ  (120 × T)
        ↓
3 × Conv1d → BN → ReLU → MaxPool2
        ↓
2-layer bidirectional LSTM  (hidden = 128)
        ↓
Trainable additive attention pooling
        ↓
Linear(256-d embedding) → Dropout
        ↓
7-class emotion head  +  256-d acoustic embedding
```

### 2. WavLM fine-tune wrapper (higher accuracy, needs GPU)

```
Raw waveform (16 kHz, ≤ 10 s)
        ↓
Pretrained WavLM/wav2vec2 encoder  (frozen warm-up → unfrozen)
        ↓
Mean-pool over frame dimension
        ↓
Linear projection → 256-d embedding
        ↓
7-class emotion head
```

## Scientific rationale

| Choice | Reason |
|---|---|
| CNN-BiLSTM baseline | Transparent, reproducible, no pretrained weights required |
| WavLM comparison | State-of-the-art speech representation; justifies added compute |
| MFCC + Δ + ΔΔ | Captures both spectral envelope and its temporal dynamics |
| Attention pooling | Learns which frames are most informative; inspectable |

## Feature attribution

Permutation importance shuffles one MFCC row (coefficient index) at a
time across the batch and measures accuracy drop.  Higher drop = more
important coefficient.

**Mandatory disclaimer on every attribution view:**
> *"Permutation importance shows which MFCC rows most affect model
> accuracy. This is a model-observed attribution signal, not causal
> evidence."*

## Training controls

| Control | CNN-BiLSTM | WavLM |
|---|---|---|
| Optimiser | AdamW | AdamW |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Gradient clipping | 1.0 | 1.0 |
| Early stopping patience | 5 | 5 |
| Input length | Pad/truncate to 10 s | Pad/truncate to 10 s |

## Running

```powershell
cd C:\Users\mh129\majorProject\emotisense-ai
backend\.venv\Scripts\python.exe -m training.voice.speech_emotion_model `
  --train path\to\speech_train.csv `
  --validation path\to\speech_val.csv `
  --output-dir models\checkpoints\speech-emotion-v1
```

Each CSV row: `path,label` where label is one of the 7 emotion strings.

## Outputs used downstream

| Output | Consumer |
|---|---|
| 256-d `embedding` | Multimodal fusion gate (Phase 14) |
| `emotion_probabilities` | Explainability view (Phase 17) |
| `attention_weights` | Supplementary frame-level inspection |
| Permutation importance | Acoustic attribution view (Phase 17) |

## Important disclaimers

- Labels are **emotion classes only**, never depression scores.
- Acoustic patterns are model-observed features.
- They must not be interpreted as evidence of any mental-health condition.
- Performance on acted speech datasets (RAVDESS, CREMA-D) may not
  generalise to naturalistic clinical recordings.
