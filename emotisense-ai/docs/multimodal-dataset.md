# Phase 12 — Aligned Multimodal Dataset Builder

## Scientific constraints

| Rule | Enforcement point |
|---|---|
| Only DAIC-WOZ / E-DAIC supply depression-risk labels | `APPROVED_MULTIMODAL_DATASETS` constant; all other dataset names rejected |
| Every sample ties one participant+session to one label | `validate_aligned_manifest()` checks dataset_name per row |
| No cross-dataset label mixing | Index validator rejects unapproved dataset names at build time |
| Participant-disjoint splits mandatory | `build_aligned_manifest()` assigns each participant to exactly one split; validator detects leakage |
| Conflicting labels for same participant rejected | `build_aligned_manifest()` raises `ValueError` on label conflict |
| Modality masks track presence, never fabricate | `AlignedMultimodalDataset.__getitem__` reads `has_text/audio/video` flags; missing → mask=0 |

## Manifest schema

```
participant_id  — stable anonymised identifier        [required]
session_id      — interview/session identifier        [required]
dataset_name    — daic_woz | e_daic                   [required]
split           — train | validation | test            [required in manifest; assigned by builder]
label           — dataset-defined risk score           [required]
has_text        — 0 | 1                               [required]
has_audio       — 0 | 1                               [required]
has_video       — 0 | 1                               [required]
text_path       — path to preprocessed transcript     [optional]
audio_path      — path to audio file                  [optional]
video_path      — path to video file                  [optional]
text_quality    — ∈ [0, 1]                            [optional → fusion gate]
audio_quality   — ∈ [0, 1]                            [optional → fusion gate]
video_quality   — ∈ [0, 1]                            [optional → fusion gate]
```

## Building splits

```powershell
cd C:\Users\mh129\majorProject\emotisense-ai
backend\.venv\Scripts\python.exe -c "
from pathlib import Path
from training.multimodal.aligned_dataset import build_aligned_manifest, AlignedManifestConfig
result = build_aligned_manifest(
    Path('training/examples/aligned-multimodal-index.csv'),
    Path('data/processed/manifests/daic-v1'),
    AlignedManifestConfig(seed=42)
)
import json; print(json.dumps(result, indent=2))
"
```

Output directory contains:
```
train.csv
validation.csv
test.csv
split-summary.json
```

## Validating an existing manifest

```python
from pathlib import Path
from training.multimodal.aligned_dataset import validate_aligned_manifest
result = validate_aligned_manifest(Path("data/processed/manifests/daic-v1/train.csv"))
print(result.to_dict())
```

## AlignedMultimodalDataset

Each item returned by `dataset[i]` contains:
```python
{
  "participant_id": "P001",
  "session_id": "S001",
  "label": "depressed",       # dataset score; never interpret as diagnosis
  "split": "train",
  "dataset_name": "daic_woz",
  "modality_mask": {"text": 1, "audio": 1, "video": 0},
  "quality_scores": {"text": 0.92, "audio": 0.87, "video": None},
  "paths": {"text": "...", "audio": "...", "video": None},
  "disclaimer": "Labels in this manifest are dataset-defined research scores. ..."
}
```

The `modality_mask` is passed directly to the fusion gate (Phase 14).
A mask value of `0` means the branch contributes zero weight, not fabricated features.

## Important disclaimers

- Labels are **dataset-defined research scores**, not clinical diagnoses.
- Emotion datasets (RAVDESS, FER-2013, CREMA-D) must never supply rows to this builder.
- Never modify split manifests after training begins.
- All model performance claims must reference the exact split manifest used.
