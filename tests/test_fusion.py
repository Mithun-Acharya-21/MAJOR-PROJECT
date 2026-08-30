"""
Tests for the fusion model components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_concat_fusion():
    """Test concatenation fusion."""
    import torch
    from ml.fusion.feature_fusion import ConcatFusion

    fusion = ConcatFusion(embed_dim=256, output_dim=256)
    text_emb = torch.randn(4, 256)
    vision_emb = torch.randn(4, 256)

    result = fusion(text_emb, vision_emb)
    assert result.shape == (4, 256)


def test_cross_attention_fusion():
    """Test cross-attention fusion."""
    import torch
    from ml.fusion.feature_fusion import CrossAttentionFusion

    fusion = CrossAttentionFusion(embed_dim=256)
    text_emb = torch.randn(4, 256)
    vision_emb = torch.randn(4, 256)

    result = fusion(text_emb, vision_emb)
    assert result.shape == (4, 512)  # Concatenated


def test_cross_attention_fusion_missing_modality():
    """Test cross-attention fusion with missing modality."""
    import torch
    from ml.fusion.feature_fusion import CrossAttentionFusion

    fusion = CrossAttentionFusion(embed_dim=256)

    # Text only
    text_emb = torch.randn(2, 256)
    result = fusion(text_emb, None)
    assert result.shape == (2, 512)

    # Vision only
    vision_emb = torch.randn(2, 256)
    result = fusion(None, vision_emb)
    assert result.shape == (2, 512)


def test_gated_fusion():
    """Test gated fusion."""
    import torch
    from ml.fusion.feature_fusion import GatedFusion

    fusion = GatedFusion(embed_dim=256)
    text_emb = torch.randn(4, 256)
    vision_emb = torch.randn(4, 256)

    result = fusion(text_emb, vision_emb)
    assert result.shape == (4, 256)


def test_classifier_head():
    """Test classifier head."""
    import torch
    from ml.fusion.classifier_head import ClassifierHead

    classifier = ClassifierHead(input_dim=512, hidden_dim=256)
    fused = torch.randn(4, 512)

    logits = classifier(fused)
    assert logits.shape == (4, 2)


def test_metrics():
    """Test evaluation metrics."""
    from ml.evaluation.metrics import compute_metrics

    y_true = [1, 1, 0, 0, 1, 0, 1, 1]
    y_prob = [0.9, 0.8, 0.2, 0.3, 0.7, 0.4, 0.85, 0.6]

    metrics = compute_metrics(y_true, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_confusion_matrix():
    """Test confusion matrix computation."""
    from ml.evaluation.metrics import compute_confusion_matrix

    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1]

    cm = compute_confusion_matrix(y_true, y_pred)

    assert cm["tp"] == 2  # True positives
    assert cm["fp"] == 1  # False positives
    assert cm["fn"] == 1  # False negatives
    assert cm["tn"] == 1  # True negatives


if __name__ == "__main__":
    tests = [
        test_concat_fusion,
        test_cross_attention_fusion,
        test_cross_attention_fusion_missing_modality,
        test_gated_fusion,
        test_classifier_head,
        test_metrics,
        test_confusion_matrix,
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
