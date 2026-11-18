# NeuroScreen — Presentation Guide

**Last updated:** 2026-11-16  
**Purpose:** Step-by-step startup, demo flow, and Q&A guidance for the final-year academic presentation.

> [!IMPORTANT]
> NeuroScreen is an **academic research prototype** investigating depression-related signals across text, voice, facial expression, and interview modalities. It is **not** a clinical diagnostic tool and does not replace assessment by a qualified mental-health professional.

---

## 1. Prerequisites

| Requirement | Detail |
|---|---|
| Python | 3.12 (`.venv` at project root) |
| Node.js | 18+ |
| GPU | NVIDIA RTX 3050 (CUDA 13.0) — auto-detected |
| Ports | Backend: 8000 · Frontend: 3000 |

---

## 2. Startup Commands

Open **two separate terminals** at the repository root.

### Terminal 1 — Backend (FastAPI)

```powershell
.\.venv\Scripts\python.exe backend\run.py
```

Expected output:
```
INFO:     Verifying model checkpoints on device: cuda
INFO:     Verified checkpoint: text (trained_metadata, version=text-eatd-1.0.0)
INFO:     Verified servable branches: text
INFO:     Uvicorn running on http://localhost:8000
```

### Terminal 2 — Frontend (Next.js)

```powershell
Set-Location frontend
npm run dev
```

Expected output:
```
▲ Next.js 16.3.3 (Turbopack)
✓ Ready in ~2s
http://localhost:3000
```

### Verify both are running

```powershell
# Backend health check
.\.venv\Scripts\python.exe -c "import urllib.request, json; print(json.loads(urllib.request.urlopen('http://localhost:8000/api/health', timeout=5).read())['status'])"
# Should print: healthy
```

---

## 3. Demo Flow

### A. Home Page (http://localhost:3000)
- Explain the research question: *Can text, voice, facial expression, and interview responses surface depression-related signals?*
- Point to the disclaimer banner — emphasise research prototype, not diagnostic.
- Show the modality cards (text, photo, voice, live-face, interview).

### B. Text Analysis Demo
1. Click **Text Analysis**.
2. Enter: *"I have been feeling very sad and hopeless lately. I can't sleep and nothing brings me joy."*
3. Submit → result appears with:
   - Prediction label (e.g. "Moderate risk")
   - Confidence score
   - Keyword-level LIME explanation panel
   - **Important to say:** English text uses a **lexicon proxy** (keyword scoring), not the trained EATD model, because the EATD model was trained on Chinese interview transcripts. Label this clearly.

### C. Screening Interview Demo
1. Click **Screening Interview**.
2. Walk through 3–5 adaptive questions.
3. Show how answers change the question selection (adaptive logic).
4. Complete → show results with modality contribution bars.

### D. Photo / Live-Face (Optional)
- Photo analysis: upload a neutral or smiling face image.
- Live-face: requires webcam. Shows frame-by-frame expression tracking.
- **Caveat to state:** Facial expression ≠ depression diagnosis. These are auxiliary signals, not trained depression classifiers.

### E. Research Results Page
- Show EATD text model metrics (AUC 0.639, real test set).
- Show AVDD-TCMI HDS E-A1 results (ROC AUC ~0.82 on acoustic features).
- Show the dataset inventory (117,518 total records across 4 datasets).

---

## 4. API Reference (Quick)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend status + checkpoint verification |
| GET | `/api/model/metrics` | Real EATD test-set metrics |
| GET | `/api/baselines` | Literature baseline comparisons |
| GET | `/api/training-status` | Training queue state |
| POST | `/api/analyze/text` | Text depression screening (form-data: `text`) |
| POST | `/api/analyze/photo` | Photo facial analysis (form-data: `photo`) |
| POST | `/api/analyze/voice` | Voice analysis (form-data: `audio`) |
| POST | `/api/analyze/face` | Live-face analysis (form-data: `frames[]`) |
| POST | `/api/analyze/multimodal` | Fused multimodal (form-data: text+photo+audio+frames) |
| POST | `/live-analyze` | Real-time frame+audio analysis (form-data: `image`, `audio`) |
| POST | `/session/start` | Start a session |
| POST | `/session/{id}/stop` | Stop session, return summary |
| GET | `/api/recommendations` | Personalised recommendations |
| GET | `/docs` | FastAPI Swagger UI |

---

## 5. Honest Capability Descriptions

| Modality | Actual Capability | Label to Use |
|---|---|---|
| Text (English) | Keyword lexicon proxy — not a trained model | "Lexicon Proxy (English)" |
| Text (Chinese) | Genuinely trained BiLSTM/transformer on EATD | "Trained EATD Model" |
| Voice | Whisper transcription + librosa acoustic features | "Auxiliary acoustic signals" |
| Photo/Live-Face | Face detection + expression estimation | "Auxiliary expression signals" |
| Screening Interview | Adaptive question logic + response analysis | "Adaptive clinical-analog interview" |
| Fusion | Late probability averaging (heuristic) | "Heuristic fusion — experimental" |

---

## 6. Known Limitations (For Q&A)

1. **English text path:** Routes through keyword proxy, not the trained EATD model. EATD training corpus is Chinese interview transcripts.
2. **Sample size:** EATD has 162 participants (138 dev, 24 test). Small sample — results not generalizable.
3. **Speaker identity not released (AVDD-TCMI):** Cross-validation results assume independence; real uncertainty may be wider.
4. **FER-2013 train split incomplete:** 3 of 7 expression classes missing from training data.
5. **Harvard Tweets:** Label mapping (0/1 semantics) unresolved; dataset treated as blocked for supervised training.
6. **No multimodal aligned dataset:** Fusion is heuristic — no ground-truth multi-modal depression labels available.
7. **In-memory sessions:** Backend session state is not persistent across restarts.
8. **SDS E-A1 (AVDD acoustic):** Still running (25-fold CV on 7168 features × 948 samples). Result pending.

---

## 7. Viva / Examiner Q&A Quick Reference

**Q: Why does the system give a "Depressed" label for clearly non-depressed text?**  
A: English input uses a keyword proxy. The EATD-trained model only covers Chinese interview text. Any English result carries a large uncertainty disclaimer.

**Q: What is your model's accuracy?**  
A: EATD text model on locked test set: AUC 0.639, balanced accuracy 0.574, 39 samples. These are real numbers from genuine training, clearly below clinical thresholds.

**Q: Why not use a larger English dataset?**  
A: Harvard Tweets (99k records) has an ambiguous 0/1 label mapping — we cannot confirm what 0 and 1 mean without author confirmation. Using it without that would invalidate the experiments.

**Q: Is this a diagnostic tool?**  
A: No. This is an academic research prototype. The disclaimer is embedded in the UI and backend responses. We investigate signals, not diagnoses.

**Q: What is the 100k record claim?**  
A: Total raw records across 4 datasets = 117,518 (99,590 Harvard Tweets + 16,212 FER-2013 + 1,230 AVDD-TCMI + 486 EATD). Strictly supervised clinical depression records = 1,716. The honest count is prominently documented in `data/reports/dataset_coverage_100k.json`.

---

## 8. File Locations (Key)

| Artifact | Path |
|---|---|
| Backend entry | `backend/run.py` |
| Frontend entry | `frontend/` (`npm run dev`) |
| API client | `frontend/lib/api.ts` |
| Text model | `models/final/text_model/text_model.pt` |
| EATD model metadata | `models/final/text_model/text_model_metadata.json` |
| Dataset inventory | `data/reports/final_dataset_inventory.md` |
| Phase 5 report | `docs/phase5_final_research_report.md` |
| Research protocol | `docs/phase_5_research_protocol.md` |
| HDS E-A1 results | `results/phase5/avdd_features/HDS/E-A1/metrics.json` |
| Test suite | `tests/` |

---

## 9. Test Commands

```powershell
# Full non-HTTP test suite (excludes TestClient-based tests)
.\.venv\Scripts\python.exe -m pytest tests\test_calibration.py tests\test_text_model.py tests\test_face_model.py tests\test_personalization.py tests\test_fusion.py -v

# Adaptive screening (22 tests)
.\.venv\Scripts\python.exe -m pytest tests\test_adaptive_screening.py -v

# Multimodal contract (34 tests, skips HTTP test)
.\.venv\Scripts\python.exe -m pytest tests\test_multimodal_contract.py --deselect tests/test_multimodal_contract.py::test_research_trace_endpoint -v

# Phase 5 visual cache (7 tests)
.\.venv\Scripts\python.exe -m pytest tests\test_phase5_visual_cache.py -v

# Frontend navigation tests
cd frontend; npm test
```
