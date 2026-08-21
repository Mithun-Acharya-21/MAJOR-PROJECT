# EmotiSense AI — Final Benchmark Report

> **Disclaimer:** These metrics are computed on a research dataset under controlled participant-disjoint evaluation conditions. Performance on clinical populations has not been established and may differ substantially. These scores are not evidence of clinical validity.

## Run metadata

| Field | Value |
|---|---|
| Timestamp | 2026-08-21T09:05:07.844357+00:00 |
| Hardware | Development machine |
| Python | 3.14.0 |
| Checkpoint | `models\checkpoints\fusion-v1\best.pt` |
| Test manifest | `training\examples\test-manifest-example.csv` |
| Manifest SHA-256 | `d958daf749d2e4c0…` |
| Test samples | 2 |
| Test participants | 2 |
| Status | **no_checkpoint** |

## Notes

- No trained model checkpoint found. Train the model on approved aligned data before running the final benchmark. This is expected during the architecture-only phase.

## Reproducibility

To reproduce this benchmark exactly:
```powershell
python experiments/scripts/run_benchmark.py \
  --checkpoint     models\checkpoints\fusion-v1\best.pt \
  --test-manifest  training\examples\test-manifest-example.csv \
  --output-dir     experiments/reports/benchmark-v1 \
  --hardware       "Development machine"
```

The test manifest SHA-256 must match:
`d958daf749d2e4c015cacd0cd0cf4a20a47c43c7a186321a24a4742fbd817477`

> Participant-disjoint splits were generated with a fixed random seed.
> The test set was not used for model selection or hyperparameter tuning.