# Phase 1 architecture decision record

The application follows a layered dependency direction: API → schemas/services → ML modules. Model code may not access HTTP objects, and recommendation code must consume structured indicators rather than raw uploads.

Future analysis requests will accept any subset of text, audio, image, and video. The orchestrator will emit a modality mask and quality score for each supplied modality. Missing modalities receive no fabricated embedding and a zero availability mask.

The initial fusion experiment sequence is concatenation, quality-weighted gating, then reliability-aware cross-modal attention. A final choice requires participant-disjoint validation results from an aligned depression dataset.

