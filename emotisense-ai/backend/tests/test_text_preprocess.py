import pytest

from training.text.preprocess import normalize_text, tokenize_for_display


def test_normalization_preserves_emotional_content_and_emoji() -> None:
    normalized = normalize_text("I feel soooo tired 😔!!! #NoSleep https://example.com @friend")

    assert "soo tired 😔!!!" in normalized
    assert "NoSleep" in normalized
    assert "<URL>" in normalized
    assert "<MENTION>" in normalized


def test_display_tokens_include_emoji_and_negation() -> None:
    tokens = tokenize_for_display("I am not okay 😔")

    assert "not" in tokens
    assert "😔" in tokens


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_text("   ")

