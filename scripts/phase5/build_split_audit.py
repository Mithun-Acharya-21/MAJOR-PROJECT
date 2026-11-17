"""Generate the Phase 5B split audit report.

Reads the EATD manifests (read-only), verifies the development-pool counts and
builds the repeated outer/inner grouped CV partitions, then writes a JSON audit
to ``data/reports/phase_5b_split_audit.json``.

It does not train a model, does not read locked-test labels, and does not modify
any file under ``data/processed``. The output is a derived report; an existing
report is only replaced with ``--force``.

Usage:
    ./.venv/Scripts/python.exe scripts/phase5/build_split_audit.py
    ./.venv/Scripts/python.exe scripts/phase5/build_split_audit.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation.splits import (  # noqa: E402
    INNER_FOLDS,
    OUTER_FOLDS,
    REPEAT_SEEDS,
    audit_eatd_development_data,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "reports" / "phase_5b_split_audit.json"


def build_report(
    *,
    seeds: tuple[int, ...] = REPEAT_SEEDS,
    n_splits: int = OUTER_FOLDS,
    inner_splits: int = INNER_FOLDS,
) -> dict:
    """Build the audit payload (no file IO)."""
    report = audit_eatd_development_data(
        seeds=seeds, n_splits=n_splits, inner_splits=inner_splits
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["generated_by"] = "scripts/phase5/build_split_audit.py"
    try:
        import sklearn

        report["scikit_learn_version"] = sklearn.__version__
    except Exception:  # pragma: no cover
        report["scikit_learn_version"] = None
    try:
        import numpy

        report["numpy_version"] = numpy.__version__
    except Exception:  # pragma: no cover
        report["numpy_version"] = None

    expectations = report.get("expectations", {})
    observed = report.get("verified_counts", {})
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expectations.items()
        if observed.get(key) != value
    }
    report["expectation_mismatches"] = mismatches
    report["status"] = "verified" if not mismatches else "count_mismatch"
    return report


def write_report(report: dict, output: Path, *, force: bool = False) -> Path:
    if output.exists() and not force:
        raise SystemExit(
            f"{output} already exists; re-run with --force to replace the audit report"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing audit report"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(REPEAT_SEEDS))
    parser.add_argument("--n-splits", type=int, default=OUTER_FOLDS)
    parser.add_argument("--inner-splits", type=int, default=INNER_FOLDS)
    args = parser.parse_args(argv)

    report = build_report(
        seeds=tuple(args.seeds),
        n_splits=args.n_splits,
        inner_splits=args.inner_splits,
    )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    written = write_report(report, output, force=args.force)

    counts = report["verified_counts"]
    print(f"wrote {written}")
    print(
        "development: {development_participants} participants / "
        "{development_sessions} sessions / {development_positive_participants} "
        "positive; locked test: {locked_test_participants} participants "
        "({locked_test_positive_participants} positive)".format(**counts)
    )
    print(
        f"outer CV: {report['outer_cv']['n_repetitions']} repetitions x "
        f"{report['outer_cv']['n_splits']} folds; status={report['status']}"
    )
    if report["expectation_mismatches"]:
        print(f"mismatches: {report['expectation_mismatches']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
