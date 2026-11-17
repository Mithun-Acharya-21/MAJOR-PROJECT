"""Evaluation — metrics, baseline models, and the Phase 5B harness.

Phase 2 metric functions (``metrics.compute_metrics`` and friends) are kept
unchanged. The Phase 5B modules add the evaluation *infrastructure* only:

- ``provenance``: partitions and fit-safety guards;
- ``splits``: repeated nested group CV over participants, with integrity checks;
- ``nested_cv``: outer/inner OOF generation from an explicit predict callback;
- ``thresholds`` / ``calibration``: selection interfaces, fit-safe by provenance;
- ``bootstrap``: participant-cluster and sample-level intervals;
- ``firewall``: locked-test access control and audit ledger;
- ``artifacts`` / ``config``: metadata contract and declared protocol settings.

None of these modules train a model or report performance; they exist so that a
later phase can produce numbers that are reproducible and leakage-probed.
"""
