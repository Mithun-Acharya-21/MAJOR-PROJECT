"""
Tests for text preprocessing pipeline.
"""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_preprocess_text_basic():
    """Test basic text preprocessing."""
    from ml.text_pipeline.preprocessing import preprocess_text

    result = preprocess_text("  Hello World  ")
    assert result == "hello world"
    assert result == result.strip()


def test_preprocess_text_urls():
    """Test URL replacement."""
    from ml.text_pipeline.preprocessing import preprocess_text

    result = preprocess_text("Check this https://example.com out")
    assert "https://example.com" not in result
    assert "<URL>" in result


def test_preprocess_text_mentions():
    """Test @mention replacement."""
    from ml.text_pipeline.preprocessing import preprocess_text

    result = preprocess_text("Hey @user how are you")
    assert "@user" not in result
    assert "<USER>" in result


def test_preprocess_text_empty():
    """Test empty text handling."""
    from ml.text_pipeline.preprocessing import preprocess_text

    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""
    assert preprocess_text(None) == ""


def test_preprocess_text_slang():
    """Test slang expansion."""
    from ml.text_pipeline.preprocessing import preprocess_text

    result = preprocess_text("idk what to do tbh")
    assert "i do not know" in result
    assert "to be honest" in result


def test_preprocess_text_emoji():
    """Test emoji conversion."""
    from ml.text_pipeline.preprocessing import preprocess_text

    result = preprocess_text("I feel 😊 today")
    assert "😊" not in result


def test_load_slang_dictionary():
    """Test slang dictionary loading."""
    from ml.text_pipeline.preprocessing import load_slang_dictionary

    slang = load_slang_dictionary()
    assert "idk" in slang
    assert "tbh" in slang
    assert slang["idk"] == "i do not know"


def test_preprocess_preserves_emotional_content():
    """Test that emotionally meaningful words are preserved."""
    from ml.text_pipeline.preprocessing import preprocess_text

    text = "I feel so hopeless and worthless today"
    result = preprocess_text(text)
    assert "hopeless" in result
    assert "worthless" in result


if __name__ == "__main__":
    # Simple test runner
    tests = [
        test_preprocess_text_basic,
        test_preprocess_text_urls,
        test_preprocess_text_mentions,
        test_preprocess_text_empty,
        test_preprocess_text_slang,
        test_preprocess_text_emoji,
        test_load_slang_dictionary,
        test_preprocess_preserves_emotional_content,
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
