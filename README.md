# NeuroScreen — Multimodal Depression-Risk Screening

A research prototype for **multimodal depression-risk screening** using Text, Photo, Live Facial Expression, and Voice/Speech analysis. Combines deep learning models with LIME explainable AI and weighted probability fusion.

> **Disclaimer**: This is a research/educational screening tool and does **NOT** provide a medical diagnosis. If you or someone you know is struggling, please contact a qualified mental health professional.

---

## Architecture

```
Frontend (Next.js — NeuroScreen v0 UI in frontend/)
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

## Quick start

```bash
# Terminal 1 — backend
python backend/run.py --reload

# Terminal 2 — frontend (Next.js)
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `frontend/.env.local`.

Open http://localhost:3000 → **Start screening** → complete all 5 steps.

## Project Structure

```
neuroscreen/
├── frontend/              # NeuroScreen v0 UI (Next.js)
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
- **Backend API**: http://localhost:8000
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

## Datasets

| Dataset | Type | Samples | Purpose |
|---------|------|---------|---------|
| EATD | Text | 487 | Depression text classification |
| FER-2013 | Images | ~28,709 | Facial expression recognition |

## Testing

```bash
python tests/test_preprocessing.py
python tests/test_text_model.py
python tests/test_face_model.py
python tests/test_fusion.py
```

## Limitations

- Research prototype, not a clinical tool
- Sarcasm, masking, and posed expressions degrade accuracy
- Dataset bias across demographics
- No personal data is stored
- Models use proxy scoring until full training on real datasets

## License

Academic project — for educational and research purposes only.
