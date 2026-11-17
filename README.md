# Emotion-Aware and Explainable Deep Learning Model for Early Depression Detection Using Attention-Based Transformer Networks

A research prototype for **multimodal depression detection** using Text, Photo, Live Facial Expression, and Voice/Speech analysis. Combines deep learning models with LIME explainable AI and weighted probability fusion.

> **Disclaimer**: This is a research/educational screening tool and does **NOT** provide a medical diagnosis. If you or someone you know is struggling, please contact a qualified mental health professional.

---

## Architecture

```
Frontend (React — EmotionDep UI in frontend/)
        ↓
    API Layer (FastAPI on :8000)
        ↓
    Service Layer
        ↓
┌───────────────────────────────────────────────┐
│  Text Branch    │  DistilBERT + BiLSTM + Attn │
│  Photo Branch   │  ResNet-18 + Grad-CAM       │
│  Live Face      │  Multi-frame aggregation     │
│  Voice/Speech   │  Whisper STT + Acoustic      │
└───────────────────────────────────────────────┘
        ↓
    Weighted Probability Fusion
        ↓
    LIME Explainability
        ↓
    Final Risk Result + Suggestions
```

## Quick Start (Windows — PowerShell)

```powershell
# Terminal 1 — Backend (FastAPI + CUDA)
cd C:\Users\mh129\majorProject
.\.venv\Scripts\python.exe backend\run.py
# → Uvicorn running on http://localhost:8000
```

```powershell
# Terminal 2 — Frontend (Next.js 16 + Turbopack)
cd C:\Users\mh129\majorProject\frontend
npm run dev
# → Ready on http://localhost:3000
```

`frontend/.env.local` already contains `NEXT_PUBLIC_API_URL=http://localhost:8000`.

Open **http://localhost:3000** → Start Screening → complete the interview.

> See [`docs/presentation_guide.md`](docs/presentation_guide.md) for the full demo flow, API reference, and Q&A prep.

## Project Structure

```
emotiondep/
├── frontend/              # EmotionDep UI (React)
│   ├── app/               # Landing + /screening wizard
│   ├── components/        # Screening steps, LIME, results
│   └── lib/               # API client + session store
│
├── backend/               # FastAPI backend
│   ├── app/
│   │   ├── main.py                   # FastAPI app + routes
│   │   ├── models/                   # Neural network definitions
│   │   │   ├── text_branch.py        # DistilBERT + BiLSTM
│   │   │   ├── vision_branch.py      # ResNet-18
│   │   │   └── fusion_layer.py       # Cross-attention fusion
│   │   ├── services/                 # Business logic
│   │   │   ├── text_service.py       # Text analysis
│   │   │   ├── photo_service.py      # Photo analysis
│   │   │   ├── face_service.py       # Live face analysis
│   │   │   ├── voice_service.py      # Voice analysis
│   │   │   └── fusion_service.py     # Multimodal fusion
│   │   ├── explainability/           # LIME explanations
│   │   │   ├── explainer.py          # Attention + Grad-CAM
│   │   │   └── lime_text.py          # LIME text explainer
│   │   ├── proxy_scorer.py           # Lexicon-based scoring
│   │   ├── voice_analysis.py         # Acoustic features
│   │   ├── live_stt.py               # Whisper transcription
│   │   ├── session_manager.py        # Session state
│   │   ├── recommendations.py        # Wellness suggestions
│   │   ├── baselines.py              # Baseline metrics
│   │   ├── training.py               # Training pipeline
│   │   └── training_state.py         # Training state
│   └── run.py                        # Entry point
│
├── ml/                    # ML training code (separate from backend)
│   ├── text_pipeline/     # Text preprocessing + training
│   ├── face_pipeline/     # Face detection + CNN training
│   ├── fusion/            # Fusion layer training
│   ├── explainability/    # LIME/SHAP experiments
│   ├── evaluation/        # Metrics + baselines
│   ├── utils/             # Config, logging, seeds
│   ├── configs/           # YAML hyperparameters
│   └── app/               # Streamlit demo + inference
│
├── data/                  # Datasets
│   ├── eatd/              # EATD text dataset (487 samples)
│   ├── fer2013/           # FER-2013 facial expressions
│   └── processed/         # Preprocessed splits
│
├── models/                # Trained model artifacts
│   ├── checkpoints/       # Training checkpoints
│   └── final/             # Best model
│
├── results/               # Evaluation outputs
│   ├── metrics/           # JSON metrics
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── explainability_outputs/
│
├── notebooks/             # Jupyter notebooks (01-06)
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── requirements.txt       # Python dependencies
```

## Technologies

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TanStack Start, Tailwind CSS 4, Shadcn UI |
| Backend | FastAPI, Python 3.14, Uvicorn |
| Text Model | DistilBERT + BiLSTM + Additive Attention |
| Vision Model | ResNet-18 + Grad-CAM |
| Fusion | Cross-Attention + Weighted Probability Fusion |
| Speech | Faster-Whisper (STT) + Acoustic Features |
| Explainability | LIME, Attention Weights, Grad-CAM |
| ML Training | PyTorch, Transformers, torchvision |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm

### 1. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Backend

```bash
cd backend
python run.py
# or
python -m uvicorn app.main:app --host localhost --port 8000 --reload
```

### 3. Start Frontend

```bash
npm install
npm run dev
```

### 4. Open

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000 (returns a JSON service descriptor)
- **API Docs**: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/analyze/text` | Text analysis with LIME |
| POST | `/api/analyze/photo` | Photo analysis with Grad-CAM |
| POST | `/api/analyze/face` | Live webcam frame analysis |
| POST | `/api/analyze/voice` | Voice/speech analysis |
| POST | `/api/analyze/multimodal` | Weighted fusion of all modalities |
| POST | `/api/analyze` | Legacy combined endpoint |
| GET | `/api/baselines` | Model benchmark data |
| POST | `/api/train` | Start training run |
| GET | `/api/training-status` | Poll training progress |
| POST | `/session/start` | Start live analysis session |
| POST | `/session/{id}/stop` | End session |

## Multimodal Pipeline

```
Text → analyze_text() → text_prob (weight: 35%)
Photo → analyze_photo() → photo_prob (weight: 15%)
Live Face → analyze_face() → face_prob (weight: 20%)
Voice → analyze_voice() → voice_prob (weight: 30%)
                    ↓
        fuse_multimodal() → Final Prediction
                    ↓
        LIME Explanation → Token/Feature attributions
                    ↓
        Recommendations → Wellness suggestions
```

**Fusion Weights** (configurable in `fusion_service.py`):
- Text: 35% — strongest depression signal from language patterns
- Voice: 30% — acoustic markers well-documented in depression research
- Live Face: 20% — temporal expression patterns
- Photo: 15% — single-image expression analysis

> ⚠️ The fused output is an **experimental heuristic screening score**, not a
> clinically validated probability. Only the trained text model (Chinese EATD)
> has measured, honestly-reported performance; facial/voice/questionnaire
> components are auxiliary or rule-based scores.

## Datasets

| Dataset | Type | Samples (local copy) | Status |
|---------|------|---------|---------|
| EATD | Text + audio (Chinese) | 162 participants / 486 sessions (3 prompted topics each) | ✅ Verified manifest + 162-participant splits (`data/processed/eatd/`, seed 42, zero overlap). Legacy 81-subject text splits retained for the existing checkpoint (`data/processed/eatd_splits/`). See `docs/eatd_dataset_card.md`. Trained text model: `results/metrics/text_metrics.json` |
| FER-2013 | Images | 9,034 train (3 of 7 classes) / 7,178 test | ⚠️ **INCOMPLETE train copy** — NOT usable for 7-class training; see `data/fer2013/validation_report.json` |

**Honesty notes (see `docs/dataset_notes.md`):**
- EATD is Chinese; the trained text model uses a multilingual encoder. The screening interview (English) is scored by a clearly-labeled lexicon proxy, not by the Chinese-trained model.
- FER-2013 is a facial-expression dataset and does **not** independently establish depression classification. Expression/acoustic signals are auxiliary only.
- No aligned multimodal depression dataset exists in this repo, so there is **no multimodal training/validation** — the fusion is an explicitly-labeled experimental heuristic.

## Testing

```bash
python tests/test_preprocessing.py
python tests/test_text_model.py
python tests/test_face_model.py
python tests/test_fusion.py
```

## Limitations

- Research prototype, not a clinical tool; nothing here is clinically validated
- The existing text checkpoint was trained on the legacy 55-participant EATD train split and has weak test performance (see `results/metrics/text_metrics.json`). A corrected 162-participant manifest is now available but has not yet been used for retraining.
- The fusion is an experimental heuristic over mixed signal types — never report it as validated accuracy
- Facial/voice signals are auxiliary expression/acoustic scores, not depression predictions
- FER-2013 train copy is incomplete (3/7 classes); expression-model training is not possible with this copy
- No aligned **face/video** depression dataset exists locally; text+audio ARE aligned in EATD (see `docs/eatd_dataset_card.md`), while face modalities remain separate research experiments
- Sarcasm, masking, and posed expressions degrade accuracy
- Dataset bias across demographics
- No personal data is stored

## License

Academic project — for educational and research purposes only.
