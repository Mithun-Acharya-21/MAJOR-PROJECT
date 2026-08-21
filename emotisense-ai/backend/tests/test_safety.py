from app.core.safety import is_safe_user_facing_text


def test_diagnostic_language_is_rejected() -> None:
    assert not is_safe_user_facing_text("You have depression.")


def test_safe_research_language_is_allowed() -> None:
    assert is_safe_user_facing_text("The model estimates elevated depression-related risk.")

