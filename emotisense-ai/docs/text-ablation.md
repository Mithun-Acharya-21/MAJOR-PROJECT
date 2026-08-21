# Phase 6 — BiLSTM Ablation and Text Head Evaluation

## Purpose
Determine whether the added BiLSTM and trainable-attention layers improve over a
plain DeBERTa mean-pool baseline on the same frozen participant-disjoint split.

## Ablation variants

| Variant | BiLSTM | Attention | Notes |
|---|---|---|---|
| `deberta_mean` | ✗ | ✗ | DeBERTa + mean pool baseline |
| `deberta_attention` | ✗ | ✓ | DeBERTa + trainable attention |
| `deberta_bilstm_mean` | ✓ | ✗ | DeBERTa + BiLSTM + mean pool |
| `deberta_bilstm_attention` | ✓ | ✓ | Full architecture (production candidate) |

## Scientific rationale
- BiLSTM captures long-range sequence dependencies that the attention pooler alone
  cannot expose without attending to all positions.
- Trainable additive attention learns which positions contribute most to the final
  embedding, providing a diagnostic signal (not causal proof).
- The ablation justifies the added complexity with evidence rather than assumption.

## Evaluation protocol
1. Each variant is trained on the **identical frozen train split**.
2. Model selection uses **validation F1 only**.
3. The final test set is **not touched** until the model is locked.
4. A full metric report (accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
   Brier score, sensitivity, specificity, confusion matrix) is produced per variant.

## Running the ablation
```powershell
cd C:\Users\mh129\majorProject\emotisense-ai
backend\.venv\Scripts\python.exe -m training.text.run_ablation `
  --train path\to\train.csv `
  --validation path\to\validation.csv `
  --output-dir models\checkpoints\text-ablation-v1
```

## Running evaluation on a single checkpoint
```powershell
backend\.venv\Scripts\python.exe -m training.text.evaluate_text `
  --checkpoint models\checkpoints\text-ablation-v1\deberta_bilstm_attention\best.pt `
  --eval-csv path\to\validation.csv `
  --output-dir models\checkpoints\text-ablation-v1\eval
```

## Important disclaimers
- Attention weights are a **supplementary diagnostic**, not causal evidence.
- All metrics are on a **research dataset**; performance on real-world clinical
  populations has not been established and may differ substantially.
- These scores must not be interpreted as diagnostic accuracy.
