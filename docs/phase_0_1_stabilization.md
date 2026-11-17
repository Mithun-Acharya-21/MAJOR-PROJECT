# Phase 0 + Phase 1 — Stabilization and Dataset Engineering Report

Date: 2026-11-13
Scope approved by the project owner: Phase 0 (stabilization) and Phase 1
(EATD dataset engineering). No new datasets downloaded; no model training run;
no project rebuild. Source data under `data/eatd/` was never modified.

## 1. What changed and why

| File | Change | Reason |
|---|---|---|
| `backend/app/model_registry.py` (new) | Checkpoint verification registry. A branch is servable only with weights + training metadata + metrics (and, for research checkpoints, a participant-level serving policy). | Prevent randomly initialized or unverified networks from contributing probabilities. |
| `backend/app/main.py` | `ModelState` no longer builds random `TextBranch`/`VisionBranch`/`MultimodalFusionModel`. It verifies checkpoints and holds no neural branches. | The legacy `/api/analyze` path was blending random logits into reported probabilities. |
| `backend/app/main.py` | `_run_analysis` uses `text_service` + `photo_service` + documented weighted fusion; returns `method`, `signal_types`, `fusion_type`, `validated_modalities`, and `model_version` only when a trained model produced the score. | Honest provenance for every number; preserves the endpoint contract. |
| `backend/app/main.py` | Training completion no longer swaps its checkpoint into serving (`_reload_from_training` logs and refuses). | API training uses row-level splits; such a checkpoint cannot demonstrate participant-independent performance. |
| `backend/app/main.py` | `GET /` returns a service descriptor when `frontend/index.html` is absent. | The route previously returned 404; the real UI is the Next.js app on port 3000. |
| `backend/app/main.py` | `/api/health` adds `verified_checkpoints`. | Operators can see exactly which branches are servable and why not. |
| `backend/app/services/text_service.py` | Text results carry explicit `signal_type` (`trained_model_probability` / `lexicon_proxy`) and apply calibration only on the trained path. | Fusion previously could not distinguish a trained score from a lexicon proxy. |
| `backend/app/services/photo_service.py` | Image signal is labeled `auxiliary_expression_score` / `proxy_heuristic`, `model_version: None`, no Grad-CAM. | No image model in the repo is trained for depression. |
| `backend/app/services/fusion_service.py` | Fusion output self-labels as `experimental_heuristic_fusion`, lists `validated_modalities`, renormalizes over available modalities only. | No fabricated or over-claimed multimodal probability. |
| `scripts/build_eatd_manifest.py` (new) | Participant audit, manifest, validation report, participant-level splits. | Phase 1 deliverable; reproducible and fail-fast. |
| `data/processed/eatd/*` (new, derived) | `manifest.csv`, `participants.csv`, `splits.csv`, `validation_report.json`, `split_report.json`. | Audited inventory and leakage-free splits. Source data untouched. |
| `tests/test_phase0_model_integrity.py` (new) | 9 tests for the serving contract. | Proves no unverified model can influence screening probabilities. |
| `tests/test_eatd_manifest.py` (new) | 18 tests for alignment, labels, duplicates, splits. | Proves manifest integrity and zero participant leakage. |
| `requirements.txt` | Pinned to the verified virtualenv; `huggingface_hub==1.28.0` pinned; CUDA 13.0 wheel index and the `.venv` interpreter documented. | Reproducibility without replacing the CUDA install; prevents the transformers/hub mismatch. |
| `docs/eatd_dataset_card.md`, `docs/environment.md` (new), `docs/dataset_notes.md`, `README.md` | Dataset and environment documentation corrected. | Earlier docs claimed no aligned multimodal data existed; text+audio are aligned in EATD. |

Existing functionality deliberately preserved: all screening routes, adaptive
interview, calibration, uncertainty, LIME/attention explainability, safety
pathway, recommendations, session handling, and the modality endpoints.

## 2. Phase 0 verification

- `ModelState` exposes `text_branch = vision_branch = fusion_model = tokenizer = None`.
- Text checkpoint verifies (`text-eatd-1.0.0`) and is served lazily by
  `TrainedTextClassifier`; `load_state_dict` is strict, so a shape-mismatched or
  partial checkpoint raises and the loader returns `None` (lexicon fallback).
- Vision/fusion checkpoints report `available: false` with an explicit reason
  (`models/checkpoints/*` are empty directories).
- English text → `method: lexicon_proxy`, `model_version: null`,
  `signal_types.text == lexicon_proxy`.
- Chinese text → `method: trained_model`, `model_version: text-eatd-1.0.0`,
  `signal_types.text == trained_model_probability`, `validated_modalities == ["text"]`.
- Image-only → `signal_types.photo` in `auxiliary_expression_score` /
  `proxy_heuristic`, `model_version: null`, `image_explanation.gradcam is None`.
- Training completion cannot mutate the serving state.
- `GET /` returns 200 (`status: ok`) with no `frontend/index.html` present.

## 3. Phase 1 findings

- **162 participants**, 486 sessions, 83 `train/` + 79 `validation/` folders.
- **58 numeric ids repeat across partitions**; they share **0** audio hashes and
  **0** transcripts (verified on the `positive.wav` / `positive.txt` of every
  same-id pair). They are distinct participants; the participant key is
  `(source_partition, subject_id)`.
- All **486** legacy CSV texts equal the raw transcripts after line-ending
  normalization (the CSV stores CRLF inside quoted fields).
- SDS: `label.txt` = raw total (20–66 observed); `new_label.txt` = 1.25 × raw
  (verified for 162/162); binary label = `index >= 50`, reproducing
  `data/eatd_text.csv` with **0** mismatches. Classes: **109 control / 53 depressed**.
- Audio: 16 kHz, 1–2 channels. `_out.wav` is a processed variant of the same
  session, not an extra sample.
- Quality: 1 session (`validation/79/positive`, 0.48 s) is unusable for audio
  and flagged; 485/486 audio-usable. 11 sessions are under 2 s.
- Legacy split: 81 participants, leakage-free but over-conservative (dropped 23
  ids after treating numeric ids as global). New split: 162 participants,
  114/24/24, zero overlap, both classes in every split.

### Independent re-verification (this pass)

Recomputed from `data/eatd/` and `data/processed/eatd/` without relying on the
generated reports:

```
participant dirs: train=83 validation=79 total=162
distinct numeric ids: 104   ids in both partitions: 58
same-id folders sharing audio hash: 0/58   sharing transcript: 0/58
SDS index != raw*1.25: 0
labels: control=109 depressed=53
label mismatches vs legacy CSV: 0
manifest rows: 486   unique (participant,topic): 486
manifest missing files: 0   text hash mismatch: 0   non-16kHz audio: 0
audio_usable False: 1
split participants: train=114 validation=24 test=24 (total 162)
overlaps: 0 0 0   uncovered: 0   unknown: 0
train classes [0,1]  validation [0,1]  test [0,1]
```

Regeneration is deterministic: running the builder with `--seed 42` into a
temporary directory produced **byte-identical** `manifest.csv`, `participants.csv`
and `splits.csv`; only `generated_at` differs in the two JSON reports.

## 4. Reproduction

```bash
.venv/Scripts/python scripts/build_eatd_manifest.py --seed 42   # exits 0, status pass_with_quality_flags
.venv/Scripts/python -m pytest tests -q                          # full suite
```

## 5. Test results

| Command | Result |
|---|---|
| `.venv/Scripts/python -m pytest tests -q` | **158 passed**, 8 warnings, exit 0 (44.6 s) |
| `python -m pytest tests -q` (system interpreter) | 14 failed / 144 passed — environment only |

Every failure under the system interpreter is
`ImportError: huggingface-hub>=1.5.0,<2.0 is required ... found 2.0.0`, plus the
CPU-only torch build. The same 14 tests pass under `.venv`. This is why the
interpreter is now documented and `huggingface_hub` is pinned.

Coverage of the new contracts: 9 tests in `tests/test_phase0_model_integrity.py`
(random-branch prohibition, unservable vision/fusion, provenance labels on every
path, health reporting, training-cannot-swap, `GET /`) and 18 tests in
`tests/test_eatd_manifest.py` (participant inventory, partition-scoped keys, SDS
rule, alignment, duplicate content, quality flags, split coverage/leakage/
determinism).

## 6. Remaining issues

- The only trained model (text, legacy 55-participant train split) has test
  ROC-AUC 0.429; it is served but is weak. Do not present it as validated.
- It was trained on the **legacy** split, not `data/processed/eatd/splits.csv`,
  so its reported test score does not correspond to the new participant split.
- No face/video corpus; the local FER-2013 copy is incomplete and has no
  depression labels.
- The API training stack (`backend/app/training.py`) still uses row-level
  splits; it is retained for experimentation but is deliberately not servable.
- The legacy 81-subject split and its checkpoint remain on disk for
  reproducibility; future training must use `data/processed/eatd/splits.csv`.
- Only 53 of 162 participants are in the depressed class, and each participant
  contributes 3 non-independent sessions — the test split has 8 positive
  participants, so metric variance will be large.
- Dataset licence/redistribution terms for the local EATD copy still need
  confirmation before publication.
- Two interpreters with different torch builds exist on this machine; the
  system one will silently produce broken runs if used instead of `.venv`.

## 7. Recommended next phase

**Phase 2 — retrain the text model on the new participant-level split**, then a
voice baseline (eGeMAPS + logistic regression) on the 485 audio-usable sessions
using the same splits. Do not start fusion training before the text and audio
single-modality baselines are honestly evaluated on the 24-participant test
split: the current text checkpoint is below chance and was trained on a
different split.
