"""
Tests for the text pipeline model components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_additive_attention_shape():
    """Test that attention pooling produces correct output shapes."""
    import torch
    from ml.text_pipeline.bilstm_attention import AdditiveAttentionPooling

    hidden_size = 256
    batch_size = 4
    seq_len = 32

    attn = AdditiveAttentionPooling(hidden_size)
    hidden = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.ones(batch_size, seq_len)
    mask[:, 10:] = 0  # Pad last 22 positions

    context, weights = attn(hidden, mask)

    assert context.shape == (batch_size, hidden_size)
    assert weights.shape == (batch_size, seq_len)


def test_bilstm_attention_shape():
    """Test BiLSTM + Attention output shapes."""
    import torch
    from ml.text_pipeline.bilstm_attention import BiLSTMAttention

    module = BiLSTMAttention(
        input_size=768,
        hidden_size=128,
        num_layers=2,
        projection_dim=256,
    )

    batch_size = 4
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, 768)
    attention_mask = torch.ones(batch_size, seq_len)

    embedding, attn_weights = module(hidden_states, attention_mask)

    assert embedding.shape == (batch_size, 256)
    assert attn_weights.shape == (batch_size, seq_len)


def test_tokenizer():
    """Test tokenizer produces valid output."""
    from ml.text_pipeline.tokenizer import get_tokenizer, tokenize_batch

    tokenizer = get_tokenizer("distilbert")
    assert tokenizer is not None

    result = tokenize_batch(
        ["I feel happy today", "I feel sad"],
        model_name="distilbert",
    )
    assert "input_ids" in result
    assert "attention_mask" in result
    assert result["input_ids"].shape[0] == 2


def test_text_classifier_forward():
    """Test text classifier forward pass."""
    import torch
    from ml.text_pipeline.train_text_model import TextClassifier

    model = TextClassifier(num_classes=2)
    batch_size = 2
    seq_len = 32

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    logits, attn_weights = model(input_ids, attention_mask)

    assert logits.shape == (batch_size, 2)
    assert attn_weights.shape == (batch_size, seq_len)


if __name__ == "__main__":
    tests = [
        test_additive_attention_shape,
        test_bilstm_attention_shape,
        test_tokenizer,
        test_text_classifier_forward,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
