"""
Tokenizer — Wraps HuggingFace tokenizers for the text pipeline.

Supports:
  - DeBERTa-v3 (primary)
  - DistilBERT (fallback)
  - BERT (baseline comparison)

Exports:
  get_tokenizer — returns a tokenizer instance
  tokenize_batch — batch tokenization helper
"""

from __future__ import annotations

from typing import Dict, List, Optional

from transformers import AutoTokenizer


# Model name registry
_MODEL_REGISTRY: Dict[str, str] = {
    "deberta": "microsoft/deberta-v3-base",
    "deberta-v3": "microsoft/deberta-v3-base",
    "deberta-large": "microsoft/deberta-v3-large",
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
}

# Cache loaded tokenizers
_tokenizer_cache: Dict[str, object] = {}


def get_tokenizer(
    model_name: str = "deberta-v3",
    max_length: int = 128,
) -> object:
    """Get or create a tokenizer for the specified model.

    Args:
        model_name: Key in the registry or a HuggingFace model name.
        max_length: Default max sequence length.

    Returns:
        A HuggingFace fast tokenizer instance.
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    # Resolve registry names
    resolved = _MODEL_REGISTRY.get(model_name, model_name)

    tokenizer = AutoTokenizer.from_pretrained(resolved)
    _tokenizer_cache[model_name] = tokenizer
    return tokenizer


def tokenize_batch(
    texts: List[str],
    model_name: str = "deberta-v3",
    max_length: int = 128,
    padding: str = "max_length",
    truncation: bool = True,
) -> Dict[str, object]:
    """Tokenize a batch of texts.

    Args:
        texts: List of strings to tokenize.
        model_name: Tokenizer to use.
        max_length: Maximum sequence length.
        padding: Padding strategy.
        truncation: Whether to truncate long sequences.

    Returns:
        Dict with 'input_ids', 'attention_mask', and optionally 'token_type_ids'.
    """
    tokenizer = get_tokenizer(model_name)
    return tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors="pt",
    )
