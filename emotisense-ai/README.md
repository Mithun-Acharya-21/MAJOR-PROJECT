# EmotiSense AI

**Emotion-Aware and Explainable Multimodal Deep Learning Model for Early Depression-Risk Detection Using Attention-Based Transformer Networks**

EmotiSense AI is a research screening system. It estimates model-observed, depression-related risk signals from optional text, voice, face, and video inputs. It is **not** a medical diagnostic, psychiatric assessment, or treatment system.

## Phase 1 status

This phase establishes the maintainable architecture, safety and privacy defaults, configuration, health/model-information API, Docker development entry point, and automated tests. No trained model, synthetic metric, or prediction endpoint is included yet.

## Quick start

Requires Python 3.11 or later.

```powershell
cd C:\Users\mh129\majorProject\emotisense-ai\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs`. The current API exposes health and model-information endpoints only.

## Safety commitments

- Never presents an output as a diagnosis or certainty.
- Treats facial and vocal emotion as auxiliary signals, not evidence of depression.
- Does not persist raw sensitive inputs by default.
- Requires participant/session-aligned data for multimodal depression training.
- Keeps final test data untouched until model selection is locked.

## Project layout

```text
backend/       FastAPI service, schemas, future ML modules, backend tests
frontend/      React + TypeScript application (introduced in Phase 19)
training/      Reproducible training, evaluation, ablation scripts
data/          Dataset documentation and local, gitignored data partitions
models/        Versioned checkpoints and exported artifacts (gitignored)
experiments/   Reproducible run metadata and reports
configs/       Versioned model and runtime configuration
docs/          Architecture, privacy, datasets, and research documentation
```

## Phase 1 verification

```powershell
cd C:\Users\mh129\majorProject\emotisense-ai\backend
pytest
```

