import pytest

torch = pytest.importorskip("torch")

from training.text.deberta_model import AdditiveAttentionPooling


def test_attention_respects_padding_mask() -> None:
    layer = AdditiveAttentionPooling(hidden_size=4)
    sequence = torch.ones((1, 3, 4))
    mask = torch.tensor([[1, 1, 0]])

    context, weights = layer(sequence, mask)

    assert context.shape == (1, 4)
    assert torch.isclose(weights[0, :2].sum(), torch.tensor(1.0))
    assert weights[0, 2].item() == 0.0

