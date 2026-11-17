# Phase 2 — Unimodal Model Retraining and Evaluation

Date: 2026-11-13
Scope: text-only and voice-only research baselines on the verified EATD
participant-level splits (`data/processed/eatd/splits.csv`). No multimodal
fusion, no application rebuild, no new datasets, no change to the serving gate,
no overwrite of the legacy checkpoint at `models/final/text_model/`.

Every number below comes from the artifacts in `results/phase2/` and
`models/experiments/`, which were produced by the runners in `scripts/phase2/`
on the local RTX 3050 6 GB (PyTorch 2.13.0+cu130, Python 3.14 in `.venv`).

---

## 1. What was reused, and what is new

Reused unchanged: `backend/app/models/text_branch.py` (encoder choice and
`preprocess_text`), `backend/app/models/text_classifier.py`,
`ml/evaluation/metrics.py` (metric definitions), `openSMILE` 2.6.0,
`data/processed/eatd/*` (manifest, participants, splits), and
`backend/app/model_registry.py` (serving gate).

New code:

| Path | Purpose |
|---|---|
| `ml/phase2/dataset.py` | Single loader for manifest + splits; enforces the `(source_partition, subject_id)` participant key, reads transcripts as UTF-8, asserts no participant crosses splits. |
| `ml/phase2/evaluation.py` | Metric bundle, participant aggregation (mean of session probabilities), participant-cluster bootstrap CIs, calibration (Brier/ECE), artifact metadata writer (config, seeds, provenance, git commit, environment). |
| `ml/phase2/text_tfidf.py` (T1) | CJK-aware TF-IDF (per-CJK-character + Latin-word tokens, word 1–2 gram plus `char_wb` 2–4 gram) + `LogisticRegression(liblinear)`; validation-only grid, top-feature export. |
| `ml/phase2/text_transformer.py` (T2) | Frozen multilingual DistilBERT embeddings (masked mean pooling, cached to `.npz`) + lightweight head (T2a); optional top-2-layer fine-tune with AMP and gradient accumulation (T2b). |
| `ml/phase2/voice_egemaps.py` (V1) | openSMILE eGeMAPS v02 functionals (88 features), 16 kHz mono check, duration/NaN quality rules, median-impute + standardize fitted on train only, logistic regression grid. |
| `ml/phase2/voice_wav2vec2.py` (V2) | Frozen Mandarin wav2vec2 encoder, 8 s chunks (max 12 per session), mean+std frame pooling then mean over chunks, cached embeddings, lightweight head. |
| `scripts/phase2/run_text_tfidf.py`, `run_text_transformer.py`, `run_voice_egemaps.py`, `run_voice_wav2vec2.py` | One runner per experiment. Shared flags: `--seed`, `--n-boot`, `--no-write`. Extra: `--finetune` and `--encoder` (T2), `--encoder {chinese,english}`, `--chunk-sec`, `--max-chunks`, `--batch-size` (V2), `--no-cache` to force re-extraction (caches are reused by default). |
| `scripts/phase2/build_comparison_report.py` | Builds `results/phase2/comparisons/comparison.{json,md}` strictly from saved results; missing experiments are reported as missing, never imputed. |
| `ml/configs/phase2/*.yaml` | Frozen experiment configurations (data, features, grid, training, evaluation). |
| `tests/test_phase2_experiments.py` | 16 tests (see §8). |

### Architectural decisions and their reasons

- **Encoder kept, language fixed.** The EATD transcripts are Chinese; the
  multilingual `distilbert-base-multilingual-cased` already used by the
  application is retained rather than introducing an English encoder. Voice
  uses a Mandarin-pretrained wav2vec2 (`TencentGameMate/chinese-wav2vec2-base`),
  with `facebook/wav2vec2-base-960h` available via `--encoder english`.
- **Transportable artifacts.** T1/V1 store `joblib` vectorizer/scaler plus
  classifier; T2a/V2 store only the small trained head plus cached embeddings
  keyed by encoder/length/chunking, so predictions are exactly reproducible
  without re-encoding. T2b stores only the trainable tensors.
- **Separate directories from serving.** Phase 2 artifacts live in
  `models/experiments/<experiment_id>/`; `model_registry.TEXT_MODEL_DIR`
  remains `models/final/text_model`, so no Phase 2 checkpoint can be served.

## 2. Data and split provenance (unchanged from Phase 1)

- 162 participant keys `(source_partition, subject_id)`; numeric ids repeat
  across source partitions and are never used alone.
- 486 sessions (162 × 3 prompted topics), 485 audio-usable (one 0.48 s file).
- Splits: 114 train / 24 validation / 24 test participants
  (342 / 72 / 72 sessions). Test: 24 participants, 8 positive.
- Label: `sds_index = sds_raw × 1.25`; `label = 1 if sds_index ≥ 50 else 0`.
  This is an EATD SDS band, **not** PHQ-9 and not a clinical diagnosis.
- Test is evaluated once per experiment, after selection on validation only.
  Preprocessing (vectorizer, scaler, imputer) is fitted on train during
  selection and refit on train+validation for the final model; the test rows
  only ever pass through `transform`.
- `ml/phase2/dataset.py::assert_split_integrity` fails fast if a participant
  appears in more than one split.

## 3. Results (locked test split, participant level)

| Model | Modality | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression (T1) | text | 0.625 | 0.182 | 0.695 | 0.579 |
| Transformer, frozen encoder + head (T2a) | text | 0.625 | 0.400 | 0.531 | 0.477 |
| Transformer, top-2-layer fine-tune (T2b) | text | 0.625 | 0.526 | 0.570 | 0.531 |
| eGeMAPS + Logistic Regression (V1) | voice | 0.417 | 0.364 | 0.383 | 0.303 |
| Wav2Vec2, frozen embeddings + head (V2) | voice | 0.708 | 0.364 | 0.398 | 0.430 |

Participant-cluster bootstrap 95% CIs (2000 resamples): ROC-AUC spans
0.430–0.925 (T1), 0.241–0.815 (T2a), 0.242–0.874 (T2b), 0.139–0.644 (V1),
0.135–0.667 (V2). Every interval includes 0.5.

Session-level test performance (72 sessions): ROC-AUC 0.612 (T1), 0.486 (T2a),
0.523 (T2b), 0.384 (V1), 0.392 (V2).

Validation → test gap (participant ROC-AUC): 0.891→0.695 (T1), 0.820→0.531
(T2a), 0.812→0.570 (T2b), 0.633→0.383 (V1), 0.812→0.398 (V2).

Calibration (test, participant level): Brier 0.223 / ECE 0.159 (T1),
0.245 / 0.152 (T2a), 0.260 / 0.255 (T2b), 0.310 / 0.347 (V1),
0.249 / 0.163 (V2). All models are poorly calibrated at this sample size and
are not thresholded for deployment.

Cost: T1 fit 5.0 s, test inference 0.011 s. T2a encode 1.5 s + head 12.4 s,
peak 0.575 GB. T2b 26.3 s fine-tune, peak 0.810 GB. V1 eGeMAPS extraction
182.8 s + fit 192.3 s (CPU-dominated), 0.38 s/session. V2 encode 26.0 s
(485 sessions, 1188 chunks) + head 4.5 s, peak 0.922 GB.

## 4. Honest interpretation

- **Only the text TF-IDF model exceeds chance on the test split, and its F1 is
  0.182** (1 TP, 2 FP, 7 FN at threshold 0.5). Its ROC-AUC of 0.695 is
  indistinguishable from chance given the interval.
- **Both voice models are below chance on the test split** (0.383, 0.398).
  This is reported as a negative result, not hidden. With 24 participants it is
  also not evidence that voice carries no signal — it is evidence that the test
  set is too small to say anything.
- **Validation performance is optimistic for every model** (gaps 0.195–0.414).
  Selecting on 24 validation participants inflates the chosen configuration;
  the test split then reverts toward chance.
- T2b's higher F1 comes with 6 FP vs 5 TP; it is a threshold artifact, not a
  better model.
- No model in this phase is clinically validated, and none is enabled for
  serving (`serving.enabled = false`, `clinical_validation = false` in every
  result file).

## 5. Model versioning

`models/experiments/{eatd_text_tfidf_v1, eatd_text_transformer_v1,
eatd_text_transformer_ft_v1, eatd_voice_egemaps_v1, eatd_voice_wav2vec2_v1}/`
each contain `metadata.json` (identical schema to the result JSON) plus weights
and `test_predictions.json`. Every artifact records: experiment id, model
version, modality, creation time, dataset/split version, participant-key and
label definitions, full config, seeds, training history and timings,
validation and test metrics, checkpoint provenance (SHA-256 of weights),
environment (Python, torch, CUDA device/version, git commit), and the explicit
`serving: {enabled: false}` + `clinical_validation: false` flags.

## 6. Reproducing

```bash
# verify interpreter first (docs/environment.md)
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

.venv/Scripts/python scripts/phase2/run_text_tfidf.py
.venv/Scripts/python scripts/phase2/run_text_transformer.py            # T2a
.venv/Scripts/python scripts/phase2/run_text_transformer.py --finetune # T2b
.venv/Scripts/python scripts/phase2/run_voice_egemaps.py
.venv/Scripts/python scripts/phase2/run_voice_wav2vec2.py
.venv/Scripts/python scripts/phase2/build_comparison_report.py
```

Embeddings are cached under the artifact directories; re-running reuses the
cache and reproduces identical test predictions. The legacy
`models/final/text_model/` checkpoint (legacy row-level split, test ROC-AUC
0.429) was left untouched and must not be cited as a Phase 2 result.

## 7. Dependencies added

One pin: `opensmile==2.6.0` (eGeMAPS extraction for V1), added to
`requirements.txt` under Audio/speech. `joblib` is used only through
scikit-learn's `save`/`load` API and is not pinned separately.

## 8. Tests

`tests/test_phase2_experiments.py` (16 tests) covers:

- data: manifest counts and label bands, participant key retains
  `source_partition`, participant-level splits are leak-free, text/audio session
  alignment;
- text: CJK tokenization is character-level and normalized, Phase 2 tokens are
  single CJK characters, TF-IDF fitting is deterministic with probabilities in
  [0, 1];
- audio: `chunk_audio` respects chunk size and the chunk cap, eGeMAPS extraction
  produces real (non-NaN) features, the unusable session is excluded with a
  recorded reason;
- model: head checkpoint round-trip reproduces probabilities, head runs on CPU;
- evaluation: metric bundle reports undefined AUC as `null`, participant
  aggregation keeps a participant's sessions together, bootstrap resamples
  participants (clusters) not sessions;
- serving: Phase 2 artifact directories are outside the registry's text path.

Full suite: **174 passed** (158 pre-existing + 16 Phase 2), 47 s, under
`.venv/Scripts/python`.

## 9. Remaining issues

1. **Statistical power.** 24 test / 8 positive participants make every estimate
   unstable. Point estimates should not be compared beyond ordering.
2. **Voice below chance.** V1 and V2 need a second look: participant-level
   aggregation by mean may be the wrong pooling for audio, and openSMILE
   functionals discard temporal structure. A within-train cross-validation
   estimate would show whether this is split noise.
3. **No seed sweep.** Single seed (42). Requested "where feasible"; with 342
   training sessions and CPU-bound eGeMAPS extraction (183 s per pass) a sweep
   was not run. Re-running T1/T2a/V2 is cheap and can be swept next.
4. **Selection optimism.** Validation is only 24 participants. The next phase
   should use participant-level cross-validation inside train+validation for
   model selection and keep the test split untouched.
5. **Threshold choice.** All metrics use 0.5; with class imbalance, a
   validation-selected operating point should be reported explicitly.
6. **Two interpreters** still exist on this machine; `requirements.txt` and
   `docs/environment.md` warn, but the system interpreter remains a footgun.
7. **Licence.** EATD redistribution terms are still unconfirmed for publication.

## 10. Recommended next phase

**Phase 3 — model selection by participant-level cross-validation, then a
voice re-analysis.** Concretely: (a) nested participant-level CV inside
train+validation for T1/T2a/V2 hyperparameters, so the test split is not the
only honest estimate; (b) test whether mean-pooling, max-pooling, or
session-count weighting changes the voice result; (c) only then attempt fusion,
and only if at least one unimodal branch beats chance with a CI excluding 0.5 —
which, on current evidence, none does.
