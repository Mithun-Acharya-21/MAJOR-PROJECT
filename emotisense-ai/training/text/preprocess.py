"""Social-media-aware text normalization that retains emotional information."""

from __future__ import annotations

import re
import unicodedata

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[\w_]+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#([\w_]+)")
# Restrict this to word characters: elongation noise such as "soooo" is
# normalized, while emphatic punctuation such as "!!!" remains a signal.
REPEATED_CHARACTER_PATTERN = re.compile(r"(\w)\1{2,}", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize noise while retaining words, emoji, negation, and punctuation.

    URLs and user handles carry little emotional content and become transparent
    placeholders. Hashtag words are retained, repeated characters are reduced to
    two (``soooo`` -> ``soo``), and Unicode emoji remain untouched.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).strip()
    if not normalized:
        raise ValueError("text cannot be empty")

    normalized = URL_PATTERN.sub(" <URL> ", normalized)
    normalized = MENTION_PATTERN.sub(" <MENTION> ", normalized)
    normalized = HASHTAG_PATTERN.sub(r" \1 ", normalized)
    normalized = REPEATED_CHARACTER_PATTERN.sub(r"\1\1", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    if not normalized:
        raise ValueError("text cannot be empty after normalization")
    return normalized


def tokenize_for_display(text: str) -> list[str]:
    """Provide an explainability-friendly token view without removing emoji."""

    return re.findall(r"<[^>]+>|\w+(?:['’-]\w+)?|[^\w\s]", normalize_text(text), flags=re.UNICODE)
