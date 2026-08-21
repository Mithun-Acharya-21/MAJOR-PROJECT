# Phase 16 — Controlled Personalised-Support Policy Engine

## Design principles

| Constraint | Implementation |
|---|---|
| No diagnosis | `POLICY_DISCLAIMER` is present on every output; no diagnostic language in any suggestion |
| No prescriptions | All suggestion texts are reviewed against a forbidden-word list (test enforced) |
| Crisis signals bypass all other logic | `CRISIS_SIGNALS` frozenset; any match → `_crisis_response()` immediately |
| No fabricated crisis contacts | Generic non-location-specific text; caller injects jurisdiction contacts at runtime |
| Fully auditable | Rule-based, no free-text generation; all suggestions drawn from `SUGGESTION_LIBRARY` |
| Max suggestions per session | Configurable; default 4 |

## Risk tiers

| Tier | Score threshold | Base categories |
|---|---|---|
| LOW | < 0.40 | Mindfulness/grounding |
| MODERATE | 0.40 – 0.64 | Stress management, mindfulness |
| HIGH | ≥ 0.65 | Professional support, stress management |
| CRISIS | Any crisis signal OR crisis tier | Crisis resources ONLY |

## Signal → category mapping

| Signal | Categories triggered |
|---|---|
| `sleep_disruption` | Sleep hygiene |
| `social_isolation` | Social connection |
| `stress_fatigue` | Stress management, mindfulness/grounding |
| `low_energy` | Physical activity |
| `hopelessness_language` | Professional support |
| `self_harm_language` | **CRISIS RESOURCES** (bypasses all others) |
| `immediate_danger` | **CRISIS RESOURCES** (bypasses all others) |

## Usage

```python
from app.ml.support_policy import build_policy_input, SupportPolicyEngine

engine = SupportPolicyEngine(max_suggestions_per_session=4)

# From fusion model output
policy_input = build_policy_input(
    risk_score=0.72,
    uncertainty=0.18,
    observed_signals=["sleep_disruption", "hopelessness_language"],
    crisis_resources=None,    # or inject local lines: ["Samaritans: 116 123"]
)
output = engine.evaluate(policy_input)
print(output.to_dict())
```

## Mandatory disclaimer

> *"These are general supportive suggestions based on observed signals in
> this session. They are not medical advice, diagnosis, or treatment.
> Please consult a qualified professional for personal support."*

This text is in `POLICY_DISCLAIMER` and appears on every `PolicyOutput`.
The UI (Phase 19) must render it prominently on every suggestion screen.

## Crisis response

When any crisis signal is detected:
1. All ordinary suggestions are **suppressed** (`suppressed_by_crisis = True`).
2. Only `CRISIS_RESOURCES` category suggestions are shown.
3. Jurisdiction-specific lines replace the generic text if provided.
4. `is_crisis = True` and `crisis_resources_shown = True` in the output.

The UI must render the crisis response with high visual priority and
not allow the user to dismiss it without acknowledgement.

## Suggestion library

All suggestion texts are stored in `SUGGESTION_LIBRARY` (a Python dict).
They are short, non-prescriptive, and validated by test against a
forbidden-word list (`diagnos`, `prescri`, `medication`, `drug`, `dose`,
`treat your`).

No suggestion text is generated at runtime — all come from this library.
