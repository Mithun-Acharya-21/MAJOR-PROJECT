# Phase 5 — DeBERTa-v3 fine-tuning architecture

The text path is `DeBERTa-v3 → BiLSTM → additive attention → binary classification head`. The standard configuration uses `microsoft/deberta-v3-base`, whose published configuration has a 768-dimensional hidden representation and a 512-token maximum position length. [Model configuration](https://huggingface.co/microsoft/deberta-v3-base/blob/main/config.json)

DeBERTa uses disentangled attention; the project adds BiLSTM to model sequential patterns above the contextual encoder and a trainable additive-attention pooling layer to create one text representation. Attention weights are retained as a supplementary inspection signal only, not claimed as causal explanation. [Hugging Face DeBERTa documentation](https://huggingface.co/docs/transformers/model_doc/deberta)

```powershell
backend\.venv\Scripts\python.exe training\text\train_deberta.py `
  --train path\to\train.csv `
  --validation path\to\validation.csv `
  --output-dir models\checkpoints\deberta-text-v1
```

The runner uses AdamW, linear warmup/decay, gradient clipping, mixed precision when CUDA is available, validation-F1 checkpointing, and early stopping. It rejects test-named input and validates participant separation. Model weights are downloaded from Hugging Face only when training begins.

## BiLSTM ablation

Run the four architectures against exactly the same frozen train/validation manifests:

```powershell
backend\.venv\Scripts\python.exe training\text\run_ablation.py `
  --train path\to\train.csv `
  --validation path\to\validation.csv `
  --output-dir experiments\runs\text-ablation-001
```

The candidates are `deberta_mean`, `deberta_attention`, `deberta_bilstm_mean`, and `deberta_bilstm_attention`. Select the model only by validation performance and calibration in later phases; do not choose it from final-test results.
