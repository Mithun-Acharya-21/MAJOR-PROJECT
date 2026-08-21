import pytest

torch = pytest.importorskip("torch")

from training.text.deberta_model import ContextualSequenceHead, DebertaTextConfig
from training.text.train_deberta import VARIANTS


def test_all_ablation_variants_produce_a_context_vector() -> None:
    sequence = torch.ones((2, 4, 6))
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    for settings in VARIANTS.values():
        head = ContextualSequenceHead(6, DebertaTextConfig(lstm_hidden_size=3, **settings))
        context, weights = head(sequence, mask)
        assert context.shape == (2, head.output_size)
        if weights is not None:
            assert torch.all(weights[mask == 0] == 0)

