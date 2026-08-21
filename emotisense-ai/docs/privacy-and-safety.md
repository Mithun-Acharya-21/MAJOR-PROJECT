# Privacy and safety baseline

Raw text, voice, facial image, and video are sensitive inputs. By default, the service processes them in memory and does not persist them. The Phase 1 environment template disables analysis metadata storage and sets raw input retention to zero hours.

All future response builders must include the centralized research disclaimer and must pass user-visible text through the non-diagnostic wording guardrail. The application must not state or imply that a user has a condition, needs medication, or has received clinical assessment.

If future text safety classification finds a credible immediate-danger signal, normal recommendations are replaced by a high-priority prompt to seek immediate human and local emergency/professional support. This must be designed and evaluated separately from depression-risk classification.

