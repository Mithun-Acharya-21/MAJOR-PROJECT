# EmotiSense AI — Full Project Architecture

> Paste this entire document into ChatGPT for context about the project.

---

## 1. Project Overview

**Name:** EmotiSense AI (branded "NeuroScreen" in the backend)
**Purpose:** Multimodal early depression detection using explainable deep learning
**Type:** Academic research prototype (Phase I — 4-person team project)
**Stack:** Python (PyTorch) backend + TypeScript (React) frontend

The system analyzes **text**, **facial images**, **webcam video**, and **voice/audio** to produce a depression screening risk assessment with LIME/SHAP explanations, voice feature analysis, and safety-aware wellness recommendations.

---

## 2. Top-Level Directory Structure

```
├── backend/              # FastAPI inference server (Python)
│   ├── run.py            # Uvicorn entry point
│   └── app/
│       ├── main.py       # FastAPI app, all API endpoints, model lifecycle
│       ├── models/       # PyTorch nn.Module definitions
│       ├── services/     # Per-modality analysis services
│       ├── explainability/  # LIME text + Grad-CAM image explainer
│       ├── baselines.py  # Benchmark comparison data
│       ├── live_stt.py   # Whisper speech-to-text wrapper
│       ├── voice_analysis.py   # Acoustic feature extraction + scoring
│       ├── proxy_scorer.py     # Lexicon-based text + image heuristics (demo mode)
│       ├── session_manager.py  # Live session state, transcript accumulation, dedup
│       ├── recommendations.py  # Wellness recommendation engine + safety keyword detection
│       ├── training.py         # Fine-tuning orchestration
│       └── training_state.py   # Training session state management
│
├── ml/                   # ML research code (training pipelines, evaluation)
│   ├── text_pipeline/    # DeBERTa-v3 tokenizer, BiLSTM-Attention, text training
│   ├── face_pipeline/    # CNN model, face detection, preprocessing, training
│   ├── fusion/           # Feature fusion, classifier head, fusion training
│   ├── explainability/   # LIME, SHAP experiment code
│   ├── evaluation/       # Metrics, baselines
│   ├── configs/          # YAML configs (text, face, fusion, training)
│   └── utils/            # Shared utilities
│
├── src/                  # React frontend (TanStack Start + Vite)
│   ├── routes/           # TanStack Router file-based routes
│   ├── components/       # React components (ScreeningWizard, WebcamCapture, AudioRecorder)
│   ├── lib/              # API client, utilities
│   └── hooks/            # Custom React hooks
│
├── data/                 # Datasets
│   ├── eatd/             # EATD (Emotion-Aware Text Dataset)
│   ├── fer2013/          # FER-2013 facial expression dataset
│   ├── processed/        # Processed face/text splits
│   └── external/         # Slang/emoji dictionaries
│
├── notebooks/            # Jupyter notebooks (01-06: exploration → explainability)
├── tests/                # Pytest tests (face model, text model, fusion, preprocessing)
├── models/               # Saved model checkpoints (best_model.pt, etc.)
├── results/              # Metrics, ROC curves, confusion matrices, XAI outputs
├── docs/                 # Architecture diagrams, dataset notes, limitations, viva prep
└── .venv/                # Python virtual environment
```

---

## 3. ML Architecture (Backend Models)

### 3.1 Text Branch (`backend/app/models/text_branch.py`)
```
Input: Raw text
  → Text preprocessing (URL/mention normalization, NFKC unicode)
  → DistilBERT tokenizer (max_length=128)
  → DistilBERT encoder (frozen backbone, 768-d hidden states)
  → BiLSTM (2-layer, bidirectional, hidden=128, output=256-d)
  → Additive Attention Pooling (learned attention scores → context vector)
  → Linear projection → 256-d text embedding
```
- **Class:** `TextBranch(nn.Module)` — EMBED_DIM = 256
- **Tokenizer:** `distilbert-base-uncased` via HuggingFace Transformers
- **Returns:** `(embedding: (B, 256), attention_weights: (B, T))`

### 3.2 Vision Branch (`backend/app/models/vision_branch.py`)
```
Input: Image bytes
  → Decode + resize to 224×224
  → ImageNet normalization
  → ResNet-18 backbone (pretrained, layers conv1→layer4)
  → AdaptiveAvgPool2d → (512, 1, 1)
  → Linear(512, 256) + ReLU + Dropout
  → 256-d vision embedding
```
- **Class:** `VisionBranch(nn.Module)` — EMBED_DIM = 256
- **Grad-CAM:** Hooks on `layer4` for heatmap generation (224×224 float32 array)
- **Returns:** `embedding: (B, 256)`

### 3.3 Fusion Model (`backend/app/models/fusion_layer.py`)
```
Input: text_features (B, 256) and/or vision_features (B, 256)
  → Reshape to (B, 1, 256) sequences
  → Cross-Attention: text attends to vision (MultiheadAttention, 4 heads)
  → Cross-Attention: vision attends to text (MultiheadAttention, 4 heads)
  → LayerNorm + residual connections
  → Concatenate fused vectors → (B, 512)
  → MLP Classifier: Linear(512→256) → ReLU → Dropout → Linear(256→2)
  → Softmax → (probability_non_depressed, probability_depressed)
```
- **Class:** `MultimodalFusionModel(nn.Module)`
- **Handles missing modalities:** Zero-pads absent branch
- **Returns:** `(logits: (B, 2), probabilities: (B, 2))`

### 3.4 Voice Analysis (`backend/app/voice_analysis.py`)
```
Input: Raw audio bytes (WAV/WebM/MP3/OGG/raw PCM)
  → Decode to float32 mono samples (wav → pyav → raw PCM fallback)
  → Extract acoustic features:
      - RMS energy
      - Silence ratio (fraction of near-silence 512-sample frames)
      - Speaking rate (syllables/sec via energy-based onset detection)
      - Pitch estimation (zero-crossing rate proxy)
      - Intensity (mean absolute amplitude)
  → Heuristic scoring: voice features → depression probability [0.08, 0.92]
```
- **Dataclass:** `VoiceFeatures` (in `session_manager.py`)
- **Scoring:** Rule-based (silence, rate, energy, pitch variation thresholds)

### 3.5 Proxy Scorer (`backend/app/proxy_scorer.py`)
- **Lexicon-based text scoring:** Risk terms (hopeless, worthless, numb...) vs protective terms (happy, grateful, hopeful...) with negation handling
- **Image proxy:** Brightness + contrast heuristics from PIL
- **Blending:** 65% proxy + 35% model output for demo-quality predictions

---

## 4. Backend API Endpoints (`backend/app/main.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check + model loading status |
| GET | `/api/baselines` | Benchmark comparison data |
| POST | `/api/analyze` | Analyze text and/or image (multipart form) |
| POST | `/api/analyze/text` | Text-only analysis |
| POST | `/api/analyze/photo` | Single photo analysis |
| POST | `/api/analyze/face` | Multi-frame webcam analysis |
| POST | `/api/analyze/voice` | Audio-only analysis |
| POST | `/api/analyze/multimodal` | Fuse all available modalities |
| POST | `/live-analyze` | Live analysis: camera frame + audio chunk + session accumulation |
| POST | `/session/start` | Start a new live session |
| POST | `/session/{id}/stop` | Stop session, return final summary |
| POST | `/api/train` | Start fine-tuning run (async, poll /api/training-status) |
| GET | `/api/training-status` | Poll training progress |
| GET | `/` | Serve frontend SPA |

### Analysis Pipeline Flow
```
1. Preprocess inputs (text clean, image decode/resize, audio decode)
2. Text Branch → 256-d text embedding + attention weights
3. Vision Branch → 256-d vision embedding + Grad-CAM heatmap
4. Fusion Model → logits → softmax probabilities
5. Proxy blending (lexicon + image heuristics)
6. Risk level: depressed_prob ≥ 0.7 → high, ≥ 0.4 → moderate, else low
7. Safety keyword check on transcript
8. Signal detection (voice features, text tokens, transcript keywords)
9. Generate recommendations (today / this week / when to seek help)
10. Return JSON: prediction, confidence, explanations, risk_level, recommendations
```

---

## 5. Frontend Architecture

- **Framework:** TanStack Start (React 19) + Vite 8
- **Router:** TanStack Router (file-based routing)
- **UI:** Radix UI primitives + Tailwind CSS 4 + shadcn/ui components
- **State:** React hooks (no global state library)
- **HTTP:** Direct fetch via `src/lib/api-client.ts`

### Key Components
- `ScreeningWizard.tsx` — Step-by-step multimodal screening flow (Text → Photo → Face → Voice)
- `WebcamCapture.tsx` — Live webcam frame capture
- `AudioRecorder.tsx` — Microphone recording with chunked upload

---

## 6. Data & Training

### Datasets
- **EATD:** Emotion-Aware Text Dataset for depression text classification
- **FER-2013:** Facial Expression Recognition (7 classes: angry, disgust, fear, happy, sad, surprise, neutral)
- **External slang/emoji dictionaries** for text preprocessing

### Training Configs (YAML)
- `ml/configs/text_config.yaml` — DeBERTa + BiLSTM-Attention training params
- `ml/configs/face_config.yaml` — CNN face model training params
- `ml/configs/fusion_config.yaml` — Multimodal fusion training params
- `ml/configs/training_config.yaml` — General training settings

### Evaluation Plan
- **Baselines:** Naive Bayes, SVM, plain BERT
- **Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Outputs:** JSON to `results/metrics/`, ROC curves, confusion matrices
- **Reported F1:** Text-only 0.86, Face-only 0.79, Fusion 0.91, Fusion ROC-AUC 0.94

---

## 7. Key Dependencies

### Python (requirements.txt)
- **ML:** torch, torchvision, transformers, numpy
- **API:** FastAPI, uvicorn, python-multipart
- **Explainability:** lime, shap
- **Audio:** pyav, openai-whisper
- **Face detection:** opencv-python
- **Data:** scikit-learn, PyYAML, Pillow

### TypeScript (package.json)
- **Framework:** @tanstack/react-start, @tanstack/react-router
- **UI:** @radix-ui/* (15+ primitives), tailwindcss, lucide-react, recharts
- **Forms:** react-hook-form, zod, @hookform/resolvers
- **Build:** Vite 8, TypeScript 5.8

---

## 8. Safety & Ethics

- **NOT a medical device** — screening research prototype only
- **Safety keyword detection:** Regex patterns for explicit suicidal ideation → immediate crisis resources (988 Lifeline, Crisis Text Line, IASP)
- **Recommendation tiers:** Low Concern → Some Indicators → Elevated Concern → Immediate Support
- **Explainable signals:** Every recommendation maps to a specific detected signal
- **No data storage** — demo interface, no persistence
- **Known limitations:** Sarcasm/masking degrade accuracy; dataset bias across demographics documented in `docs/limitations.md`

---

## 9. Testing

Located in `tests/`:
- `test_text_model.py` — Text branch forward pass, tokenizer
- `test_face_model.py` — Vision branch, image preprocessing
- `test_fusion.py` — Fusion model with various modality combinations
- `test_preprocessing.py` — Text/image preprocessing pipelines
