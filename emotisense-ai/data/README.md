# Data governance

This directory must not contain committed raw participant data.

- `raw/`, `processed/`, `train/`, `validation/`, and `test/` are local-only and gitignored.
- Maintain a data card recording dataset source, license, consent/access conditions, label definitions, participants, preprocessing, and split identifiers.
- Multimodal training records must join text, audio/video, and depression labels from the same participant/session.
- Split by participant before preprocessing or augmentation. The final test partition is never used for model selection.

