# Initial Repository Audit — NeuroScreen

Date: 2026-11-16 (Phase A). Scope: read-only inspection of the repository at
`C:\Users\mh129\majorProject`. No source code, data, model, split or artifact was
modified by this audit. Pre-existing uncommitted working-tree changes (modified and
deleted files) are preserved untouched.

## 1. Scope and method

This audit traces actual execution paths rather than inferring functionality from
filenames. Evidence is taken from files read in this repository. Where a figure is
quoted from a prior report it is tagged **[reported]** and is not treated as
verified by this audit.

The repository is a single Git working tree on branch `main` containing three
distinct application surfaces (a Vite/TanStack frontend at the root, a Next.js
frontend under `frontend/`, and a second full-stack app under `emotisense-ai/`),
one FastAPI backend under `backend/`, and an ML research tree under `ml/`.
Only the root `backend/` + `ml/` + root `src/` + `frontend/` surfaces appear
active; `emotisense-ai/` is a frozen/deleted copy (see §12).

## 2. Repository overview

Top-level layout (selected):

| Path | Role |
|---|---|
| `src/`, `vite.config.ts`, root `package.json` | Vite + TanStack Start frontend ("tanstack_start_ts"), served as the primary UI shell |
| `frontend/` | Next.js 16 app (alternate UI, tests under `frontend/__tests__/`) |
| `emotisense-ai/` | Second full-stack copy (deleted in working tree; not active) |
| `backend/` | FastAPI app (`backend/app/main.py`, `backend/run.py`) |
| `ml/` | ML pipelines, Phase 2 experiments, Phase 5 harness |
| `data/` | Datasets, processed splits, reports |
| `models/` | Checkpoints, experiment artifacts, final text model |
| `results/` | Metrics, curves, explainability outputs, Phase 2 reports |
| `tests/` | Python test suite (pytest) |
| `scripts/` | Dataset audit/download/validation + Phase 2/5 runners |
| `docs/` | Phase 0–5 documentation |
| `notebooks/` | Jupyter notebooks 01–06 |

Git: branch `main`, 11 commits; pre-existing modified files include `backend/app/*`,
`docs/dataset_notes.md`, `README.md`, `.gitignore`; `FINAL_INTEGRATION_REPORT.md`,
`PROJECT_ARCHITECTURE_FOR_CHATGPT.md` and the entire `emotisense-ai/` tree are
deleted in the working tree. These are left untouched by this audit.

## 3. Environment

Verified interpreter: `.venv/Scripts/python.exe` (Python 3.14.0). PyTorch 2.13.0+cu130,
CUDA available, NVIDIA GeForce RTX 3050 6 GB. scikit-learn 1.9.0, transformers 5.15.1,
fastapi 0.141.1, pytest 8.4.2. Disk: 24 GB free on C: (94% used). The system
Python is CPU-only and must not be used (see `docs/environment.md`).

## 4. Frontend architecture

Two UI surfaces exist:

- **Root `src/`** (Vite + TanStack Router/Start, React 19): `src/routes/index.tsx`,
  `src/components/ScreeningWizard.tsx`, `AudioRecorder.tsx`, `WebcamCapture.tsx`,
  `src/lib/api-client.ts`. This is the surface launched by `npm run dev`.
- **`frontend/`** (Next.js 16, React 19, Tailwind, shadcn/ui): `app/`, `components/`,
  `lib/`, plus `frontend/__tests__/` (Vitest).

Both call the FastAPI backend. Camera/microphone capture lives in the root
`src/components/WebcamCapture.tsx` and `AudioRecorder.tsx`; the Next.js copy mirrors
a subset. The root `src/server.ts`/`start.ts` indicate SSR/server functions.

Frontend test status: `frontend/__tests__/` contains `free-navigation.test.tsx` and a
`screening/` directory; full run not executed in this audit.

## 5. Backend architecture

FastAPI app in `backend/app/main.py`. Endpoints actually registered:

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serve frontend dashboard |
| GET | `/api/health` | Health + model loading status |
| GET | `/api/baselines` | Benchmark comparison data |
| GET | `/api/training-status` | Poll training progress |
| POST | `/api/train` | Start a fine-tuning run |
| POST | `/api/analyze` | Legacy combined endpoint |
| POST | `/session/start` | Start live analysis session |
| POST | `/session/{id}/stop` | End session |
| POST | `/live-analyze` | Live frame analysis |
| POST | `/api/analyze/text` | Text analysis + LIME |
| POST | `/api/analyze/photo` | Photo analysis + Grad-CAM |
| POST | `/api/analyze/face` | Live webcam frame analysis |
| POST | `/api/analyze/voice` | Voice/speech analysis |
| POST | `/api/analyze/multimodal` | Weighted fusion |
| GET | `/api/model/metrics` | Model metrics |
| GET | `/api/recommendations` | Wellness suggestions |
| (router) | `/screening/*` | `screening_routes.py` (start/response/state/result/recommendations/research-trace/complete) |

Services: `text_service`, `photo_service`, `face_service`, `voice_service`,
`fusion_service`, `screening_service`, `adaptive_question_service`,
`calibration_service`, `response_analysis_service`, `uncertainty_service`.
Model loading is verification-only (`ModelState.load` calls `model_registry.audit_checkpoints`);
no randomly initialized branches are served. Text weights load lazily via
`TrainedTextClassifier`; no vision/fusion checkpoint is servable.

Database: no external DB observed in the active path; session state is in-memory
(`session_manager`). Persistence claims must be verified (see §12).

## 6. ML architecture

- `ml/text_pipeline/`: preprocessing, tokenizer, BiLSTM+attention, DeBERTa encoder,
  training.
- `ml/face_pipeline/`: face detection, preprocessing, CNN, training.
- `ml/fusion/`: feature fusion, classifier head, training.
- `ml/phase2/`: `dataset.py`, `evaluation.py`, `text_tfidf.py`, `text_transformer.py`,
  `voice_egemaps.py`, `voice_wav2vec2.py` — frozen Phase 2 baselines.
- `ml/evaluation/`: metrics, baselines, plus Phase 5B harness
  (`provenance`, `splits`, `nested_cv`, `thresholds`, `calibration`, `bootstrap`,
  `firewall`, `artifacts`, `config`, `metrics` extensions).
- `ml/explainability/`: LIME/SHAP/visualization.
- `ml/app/`: Streamlit demo + inference.
- `ml/utils/`: config loader, logger, seed.

Phase 2 artifacts under `models/experiments/*` and `results/phase2/*` are protected
(read-only). Phase 5 harness completed (314 tests passing).

## 7. Data and datasets

| Dataset | Location | Status |
|---|---|---|
| EATD | `data/eatd/`, `data/processed/eatd/` | Verified: 162 participants, 486 sessions, locked split seed 42 (114/24/24) |
| FER-2013 | `data/fer2013/` | Incomplete train copy (3/7 classes); not usable for 7-class training |
| Harvard tweets | `data/raw/text/harvard_tweets/` | 99,590 records; label mapping unresolved → supervised training blocked |
| AVDD-TCMI | `data/raw/avdd_tcmi/` (zip, 13.5 GB) | Audited: 1,230 samples; no raw audio/video; CSV feature extraction ZIP-backed |
| DAIC-WOZ | not acquired | Access pending (DUA) |
| RSDD / SMHD | not acquired | Access pending |

See `docs/eatd_dataset_card.md`, `docs/dataset_compatibility_matrix.md`,
`docs/harvard_tweets_label_audit.md`, `docs/phase_4_access_status.md`.

## 8. Models and checkpoints

- `models/experiments/eatd_text_tfidf_v1/`, `eatd_text_transformer_v1/`,
  `eatd_text_transformer_ft_v1/`, `eatd_voice_egemaps_v1/`,
  `eatd_voice_wav2vec2_v1/` — Phase 2 unimodal baselines with metadata + test predictions.
- `models/final/text_model/` — servable text checkpoint (`text_model.pt` + metadata).
- `models/calibration.json`, `models/face_landmarker.task`.
- `models/checkpoints/{face,fusion,text}_model/.gitkeep` — empty placeholders.
- `models/final/.gitkeep`.

No face/fusion checkpoint is trained. The face model training code exists but no
verified artifact.

## 9. Tests

Python: `tests/` (26 files detected) covering preprocessing, text/face models,
fusion, screening API, E2E scenarios, research integrity, AVDD validation, Phase 0/2,
Phase 5 splits/metrics/leakage/firewall/artifacts. Last full run: **314 passed,
10 warnings** (after Phase 5B). Frontend: Vitest under `frontend/__tests__/`
(not executed in this audit). Backend endpoint tests: `scripts/test_api_endpoints.py`.

## 10. Documentation

`docs/` contains architecture diagram, dataset cards, environment, limitations,
Phase 0–5 docs, project status summary (empty), viva Q&A, plus the new
`phase_5b_evaluation_harness.md`. `FINAL_INTEGRATION_REPORT.md` and
`PROJECT_ARCHITECTURE_FOR_CHATGPT.md` are deleted in the working tree.

## 11. Component inventory

| Component | Existing | Tested | Working | Missing | Risk |
|---|---|---|---|---|---|
| Root Vite frontend (src/) | Yes | Partial | Partial | E2E verification | Medium |
| Next.js frontend (frontend/) | Yes | Partial | Partial | Full journey tests | Medium |
| FastAPI backend | Yes | Partial | Yes (health) | DB persistence, auth | Medium |
| Text model serving | Yes | Yes | Yes (verified ckpt) | Calibration versioning | Low |
| Photo/face model serving | Yes | Partial | Heuristic only | Trained checkpoint | High |
| Voice model serving | Yes | Partial | Heuristic only | Trained checkpoint | High |
| Multimodal fusion serving | Yes | Partial | Heuristic | Validated fusion model | High |
| Live webcam capture | Yes | No | Partial | Preprocessing compatibility proof | High |
| ML text pipeline | Yes | Yes | Yes | Retraining on 162-pt manifest | Medium |
| ML face pipeline | Yes | Partial | Code only | Data + trained model | High |
| ML fusion pipeline | Yes | Partial | Code only | Paired data (DAIC) | High |
| Phase 2 baselines | Yes | Yes | Yes | — (frozen) | Low |
| Phase 5B harness | Yes | Yes (114 tests) | Yes | Execution of experiments (5C+) | Low |
| EATD dataset | Yes | Yes | Yes | — | Low |
| FER-2013 | Partial | Yes | Not usable (3/7) | Full dataset | Medium |
| Harvard tweets | Yes | Yes | Blocked (label) | Authoritative mapping | High |
| AVDD-TCMI | Yes | Yes | Audited | Feature cache + models | Medium |
| DAIC-WOZ | No | N/A | Blocked (access) | DUA approval | High |
| RSDD/SMHD | No | N/A | Blocked (access) | DUA approval | High |

## 12. Known issues, risks and technical debt

1. **No external database**: session state is in-memory; "PostgreSQL" claims (if any) are
   unverified. Must be audited before claiming persistence.
2. **Heuristic serving**: photo/face/voice/fusion endpoints return rule-based scores, not
   trained model probabilities. Must be labeled as such in UI and API.
3. **FER-2013 incomplete**: 3/7 classes; 7-class training impossible with this copy.
4. **Harvard label mapping unresolved**: supervised training prohibited.
5. **No paired multimodal dataset locally**: DAIC-WOZ not acquired; fusion is heuristic.
6. **No face/fusion checkpoint**: training code exists but no verified artifact.
7. **`emotisense-ai/` tree deleted** in working tree (pre-existing); appears to be a
   duplicate full-stack copy. Not restored by this audit.
8. **Pre-existing modified files** (backend, docs, README, .gitignore) are not reverted.
9. **Disk pressure**: 24 GB free; AVDD full extraction needs 33.7 GB — prohibited.
10. **Python 3.14** ahead of parts of the ecosystem; torch.jit deprecation warning
    (non-fatal).

## 13. Protected artifacts (do not modify)

- `data/processed/eatd/{manifest,participants,splits}.csv`, `split_report.json`
- `results/phase2/**`, `models/experiments/**`, `models/final/**`
- `data/raw/avdd_tcmi/*.zip`, `data/eatd/**`
- All Phase 5A/B artifacts (`docs/phase_5*`, `ml/configs/phase5/*`,
  `data/reports/phase_5b_split_audit.json`, harness modules, tests)
- Pre-existing unrelated working-tree changes

## 14. Prioritized implementation plan (proposed)

1. **Phase A (this)** — audit complete; report findings to owner. (DONE)
2. **Phase B1 (5C)** — EATD text replication/extension using Phase 5B harness, dev pool
   only, no locked-test access. Prerequisite: harness present (yes).
3. **Phase B2 (5D)** — EATD voice experiments.
4. **Phase B3** — Harvard tweets audit; training blocked until label mapping verified.
5. **Phase B4/B5/B6** — AVDD acoustic/visual/face experiments (feature cache first;
   JPG extraction gated on disk).
6. **Phase B7** — DAIC-WOZ adapter (no data yet; blocked on DUA).
7. **Phase B8** — Multimodal fusion (gated on unimodal significance + paired data).
8. **Phase C** — Full metric reporting per experiment.
9. **Phase D/E** — Backend/frontend audit and completion.
10. **Phase F** — Live webcam inference (only after validated model).
11. **Phase G** — System integration tests.
12. **Phase H** — Final locked-test evaluation (pre-registered, one pass).
13. **Phase I/J/K** — Scorecards, final report, deliverables.

## 15. Stop gates / approvals required

- Any destructive operation (deleting data, overwriting protected artifacts) requires
  explicit approval.
- Final locked-test evaluation requires pre-registration + script hash + firewall
  authorization; not started here.
- DAIC-WOZ/RSDD/SMHD experiments require authorized data acquisition.
- Harvard supervised training requires verified label mapping.
- AVDD JPG extraction requires disk-capacity check and approval.

## 16. Immediate findings summary

- The codebase compiles and the Python test suite passes (314 passed) under the
  verified venv.
- The only servable trained model is the EATD text model; face/voice/fusion serving
  is heuristic and must not be presented as validated.
- No external DB is in the active path; persistence is in-memory.
- Dataset access blockers (Harvard labels, DAIC-WOZ, RSDD, SMHD) prevent several
  research phases from proceeding.
- The Phase 5B evaluation harness is complete and tested; it is the correct vehicle
  for all subsequent EATD/AVDD experiments.
