# Phase 4 text preprocessing and baseline

## Normalization policy

The normalizer uses Unicode NFKC normalization, replaces URLs and user mentions with visible placeholders, converts `#sleepProblems` into `sleepProblems`, reduces extreme repeated characters to two, and preserves emoji, negation, punctuation, and expressive casing. It does not use stop-word deletion because terms such as “not”, “never”, and “alone” may be meaningful.

## Classical baseline

The baseline is TF–IDF word/bigram features plus class-balanced logistic regression. It is a transparent comparison point for the DeBERTa model planned in the next phases, not the final model.

```powershell
backend\.venv\Scripts\python.exe training\text\baseline.py `
  --train path\to\train-text.csv `
  --validation path\to\validation-text.csv `
  --output-dir models\artifacts\text-baseline-v1
```

Both CSVs require `record_id`, `participant_id`, `label`, and `text`; an optional `split` column must match the requested partition. The trainer has no test input and rejects any filename containing `test`.

The exported pickle may contain learned vocabulary. Treat it as a sensitive model artifact, do not commit it, and document its training data in a model card.

