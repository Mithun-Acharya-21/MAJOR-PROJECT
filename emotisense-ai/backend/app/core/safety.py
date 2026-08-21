"""Centralized wording and non-diagnostic guardrails for all future outputs."""

RESEARCH_DISCLAIMER = (
    "This is an AI research screening result, not a medical diagnosis, psychiatric "
    "assessment, or treatment recommendation. If you are concerned about your wellbeing, "
    "consider speaking with a qualified mental-health professional."
)

FORBIDDEN_DIAGNOSTIC_PHRASES = (
    "you have depression",
    "you are clinically depressed",
    "you definitely have anxiety",
    "you need medication",
)


def is_safe_user_facing_text(message: str) -> bool:
    """Reject prohibited diagnostic language before it reaches an API response."""

    normalized = message.casefold()
    return not any(phrase in normalized for phrase in FORBIDDEN_DIAGNOSTIC_PHRASES)

