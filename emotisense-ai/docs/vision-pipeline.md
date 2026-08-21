# Vision Pipeline — Phase 7 & 8

## Phase 7 — Face detection, quality scoring, and FER baseline

### Processing chain

```
Input (image / video frame / bytes)
        ↓
  FacePreprocessor
        ├── Haar-cascade face detection
        ├── Multi-face handling (use largest; reduce quality)
        ├── Padding + crop
        ├── Quality scoring: blur (Laplacian variance), brightness
        └── Normalise → (C, H, W) float32 tensor
        ↓
  FaceResult
        ├── detected, n_faces, bbox
        ├── quality_score ∈ [0, 1]
        ├── reliability (True only if 1 adequate face)
        ├── face_tensor (optional, not persisted by default)
        └── rejection_reasons []
```

### Video processing

`process_video()` samples frames at a configurable FPS, runs
`FacePreprocessor` on each frame, and aggregates:
- n_frames_sampled / n_frames_with_face
- mean_quality_score, min_quality_score
- reliable_frame_ratio

### FER baseline model

| Component | Detail |
|---|---|
| Architecture | 4-block Conv→BN→ReLU→MaxPool → FC → softmax |
| Input | 48×48 or 224×224 grayscale |
| Output | 7 emotion classes (FER-2013) + 256-d embedding |
| Dataset | FER-2013 / RAF-DB / AffectNet (emotion labels only) |
| Purpose | Baseline before transfer learning (Phase 8) |

**The model outputs are emotion labels, never depression labels.**

### Quality rejection reasons

| Code | Meaning |
|---|---|
| `no_face_detected` | Cascade found no face |
| `multiple_faces` | More than one face detected |
| `blurry` | Laplacian variance below threshold |
| `under_exposed` | Mean brightness < 30 |
| `over_exposed` | Mean brightness > 225 |
| `face_too_small` | Face area / image area < 2 % |
| `file_unreadable` | OpenCV could not read the file |
| `decode_failed` | In-memory decode failed |

When `reliability = False`, the vision branch contributes zero weight
in the multimodal fusion gate (Phase 14).

## Phase 8 — Transfer learning and Grad-CAM

See [vision-transfer.md](vision-transfer.md).

## Important disclaimers

- Facial expression is an **auxiliary emotional signal**, not a depression indicator.
- Every prediction screen must display: *"These are model signals, not proof of a
  mental-health condition."*
- Grad-CAM visualisations show what spatial regions the model used; they do not
  prove those regions caused the prediction.
