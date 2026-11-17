"""
Bootstrap confidence intervals for Phase 5 (Phase 5B).

Two resampling units, never mixed:

* **EATD** — participants are the unit. Resampling sessions independently would
  treat three correlated sessions of one person as three independent
  observations and shrink the interval. ``participant_cluster_bootstrap``
  aggregates to one prediction per participant, then resamples participants
  within each class.
* **AVDD** — no participant identity is released, so only sample-level
  resampling is possible. ``sample_bootstrap`` reports that limitation in every
  result instead of implying a cluster-robust interval.

Defaults follow ``docs/phase_5_statistical_evaluation_plan.md``: 2000 resamples,
seed 0, 95 % percentile interval.

A resample on which a metric is undefined (single class, no positive
predictions, zero denominator) is counted as invalid and excluded; if fewer than
95 % of resamples are valid the interval is withheld with the reason recorded.

Exports:
    BootstrapConfig, BootstrapError
    participant_cluster_bootstrap, sample_bootstrap, bootstrap_for_dataset
    aggregate_by_participant
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ml.evaluation import metrics as M

DEFAULT_N_RESAMPLES = 2000
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.05
MIN_VALID_FRACTION = 0.95

DEFAULT_METRICS: Tuple[str, ...] = (
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)

#: metric name -> callable(y, p, threshold, unit) -> MetricResult
_METRIC_FUNCS = {
    "roc_auc": lambda y, p, t, u: M.roc_auc_metric(y, p, unit=u),
    "pr_auc": lambda y, p, t, u: M.pr_auc_metric(y, p, unit=u),
    "balanced_accuracy": lambda y, p, t, u: M.balanced_accuracy_at(y, p, t, unit=u),
    "sensitivity": lambda y, p, t, u: M.recall_at(y, p, t, unit=u),
    "specificity": lambda y, p, t, u: M.specificity_at(y, p, t, unit=u),
    "precision": lambda y, p, t, u: M.precision_at(y, p, t, unit=u),
    "macro_f1": lambda y, p, t, u: M.macro_f1_at(y, p, t, unit=u),
    "mcc": lambda y, p, t, u: M.mcc_at(y, p, t, unit=u),
    "brier": lambda y, p, t, u: M.brier_metric(y, p, unit=u),
}

SUPPORTED_METRICS: Tuple[str, ...] = tuple(_METRIC_FUNCS)


class BootstrapError(RuntimeError):
    """The requested bootstrap cannot be run as specified."""


@dataclass(frozen=True)
class BootstrapConfig:
    """Resampling plan. Defaults are pre-registered, not tuned."""

    n_resamples: int = DEFAULT_N_RESAMPLES
    seed: int = DEFAULT_SEED
    alpha: float = DEFAULT_ALPHA
    metrics: Tuple[str, ...] = DEFAULT_METRICS
    threshold: float = 0.5
    min_valid_fraction: float = MIN_VALID_FRACTION

    def __post_init__(self) -> None:
        if self.n_resamples < 1:
            raise BootstrapError("n_resamples must be >= 1")
        if not 0.0 < self.alpha < 1.0:
            raise BootstrapError("alpha must be in (0, 1)")
        if not 0.0 <= self.threshold <= 1.0:
            raise BootstrapError("threshold must be in [0, 1]")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise BootstrapError("min_valid_fraction must be in (0, 1]")
        unknown = [m for m in self.metrics if m not in _METRIC_FUNCS]
        if unknown:
            raise BootstrapError(
                f"unsupported bootstrap metric(s) {unknown}; supported: "
                f"{list(SUPPORTED_METRICS)}"
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_resamples": self.n_resamples,
            "seed": self.seed,
            "alpha": self.alpha,
            "metrics": list(self.metrics),
            "threshold": self.threshold,
            "min_valid_fraction": self.min_valid_fraction,
        }


def _metric_values(
    y: np.ndarray,
    p: np.ndarray,
    config: BootstrapConfig,
    *,
    unit: str,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for name in config.metrics:
        try:
            res = _METRIC_FUNCS[name](y, p, config.threshold, unit)
        except M.MetricInputError:
            out[name] = None
            continue
        out[name] = float(res.value) if res.defined and res.value is not None else None
    return out


def _invalid_reason(y: np.ndarray, p: np.ndarray) -> str:
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return "no_positive_cases"
    if n_pos == y.size:
        return "no_negative_cases"
    return "undefined_metric"


def aggregate_by_participant(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    participant_ids: Sequence[str],
    *,
    aggregation: str = "mean",
) -> Dict[str, object]:
    """Reduce session-level predictions to one prediction per participant.

    Each participant must carry a single label (a person does not change
    diagnosis between sessions). Session probabilities are averaged by default,
    matching the participant-level evaluation unit.

    Raises:
        BootstrapError: misaligned input, or a participant with mixed labels.
    """
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    ids = np.asarray(list(participant_ids), dtype=object).reshape(-1)
    if not (y.shape == p.shape == ids.shape):
        raise BootstrapError(
            f"misaligned input: y={y.size}, p={p.size}, ids={ids.size}"
        )
    if aggregation not in {"mean", "max", "median"}:
        raise BootstrapError(f"unknown aggregation {aggregation!r}")

    groups: Dict[str, List[int]] = {}
    for i, pid in enumerate(ids.tolist()):
        groups.setdefault(str(pid), []).append(i)

    out_labels: List[int] = []
    out_probs: List[float] = []
    out_groups: List[str] = []
    out_sizes: List[int] = []
    for pid in sorted(groups):
        idx = groups[pid]
        labels = {float(y[i]) for i in idx}
        if len(labels) != 1:
            raise BootstrapError(
                f"participant {pid!r} has mixed labels {sorted(labels)}; the "
                "participant-level label must be unique"
            )
        probs = p[idx]
        if aggregation == "mean":
            value = float(np.mean(probs))
        elif aggregation == "max":
            value = float(np.max(probs))
        else:
            value = float(np.median(probs))
        out_labels.append(int(labels.pop()))
        out_probs.append(value)
        out_groups.append(pid)
        out_sizes.append(len(idx))

    return {
        "labels": np.asarray(out_labels, dtype=np.float64),
        "probs": np.asarray(out_probs, dtype=np.float64),
        "participant_ids": tuple(out_groups),
        "session_counts": tuple(out_sizes),
        "aggregation": aggregation,
        "n_units": len(out_groups),
        "n_observations": int(y.size),
    }


def _resample_indices(
    y: np.ndarray, rng: np.random.Generator, *, stratified: bool
) -> np.ndarray:
    """Indices for one resample; stratified keeps the class ratio fixed."""
    if not stratified:
        return rng.integers(0, y.size, size=y.size)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    picked = np.concatenate(
        [
            rng.choice(pos, size=pos.size, replace=True),
            rng.choice(neg, size=neg.size, replace=True),
        ]
    )
    return picked


def _summarise(
    *,
    method: str,
    unit: str,
    description: str,
    point: Mapping[str, Optional[float]],
    samples: Mapping[str, List[float]],
    invalid_reasons: Mapping[str, int],
    n_resamples: int,
    config: BootstrapConfig,
    extra: Mapping[str, object],
) -> Dict[str, object]:
    threshold_count = max(1, int(np.ceil(config.min_valid_fraction * n_resamples)))
    metric_report: Dict[str, object] = {}
    invalid_total = 0
    for name in config.metrics:
        values = np.asarray(samples.get(name, []), dtype=np.float64)
        invalid = n_resamples - int(values.size)
        invalid_total = max(invalid_total, invalid)
        entry: Dict[str, object] = {
            "point": point.get(name),
            "n_valid_resamples": int(values.size),
            "n_invalid_resamples": int(invalid),
            "ci_method": "percentile",
        }
        if values.size == 0:
            entry["ci_lower"] = None
            entry["ci_upper"] = None
            entry["status"] = "no_valid_resamples"
        elif values.size < threshold_count:
            entry["ci_lower"] = None
            entry["ci_upper"] = None
            entry["status"] = "insufficient_valid_resamples"
        else:
            lo, hi = np.percentile(
                values, [100 * config.alpha / 2, 100 * (1 - config.alpha / 2)]
            )
            entry["ci_lower"] = round(float(lo), 6)
            entry["ci_upper"] = round(float(hi), 6)
            entry["status"] = "ok"
            if name in point and point[name] is not None:
                entry["point_inside_interval"] = bool(
                    entry["ci_lower"] <= point[name] <= entry["ci_upper"]
                )
        metric_report[name] = entry

    notes = [
        description,
        "percentile interval; no bias correction applied",
    ]
    if "participant" not in unit:
        notes.append(
            "resampling ignores participant clustering: intervals are likely "
            "too narrow if any participant contributed several samples"
        )

    result: Dict[str, object] = {
        "method": method,
        "unit": unit,
        "clustering_respected": unit == "participant",
        "config": config.to_dict(),
        "n_resamples": n_resamples,
        "n_valid_resamples": n_resamples - invalid_total,
        "n_invalid_resamples": invalid_total,
        "invalid_reasons": dict(invalid_reasons),
        "metrics": metric_report,
        "notes": notes,
    }
    result.update(extra)
    return result


def _run_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    config: BootstrapConfig,
    *,
    method: str,
    unit: str,
    description: str,
    extra: Mapping[str, object],
) -> Dict[str, object]:
    if y.size < 2:
        raise BootstrapError("bootstrap needs at least two units")
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise BootstrapError(
            f"bootstrap needs both classes (pos={n_pos}, neg={n_neg}); a "
            "single-class interval is not defined"
        )
    if not np.isfinite(p).all():
        raise BootstrapError("probabilities must be finite")
    if (p < 0).any() or (p > 1).any():
        raise BootstrapError("probabilities must lie in [0, 1]")

    point = _metric_values(y, p, config, unit=unit)
    rng = np.random.default_rng(config.seed)
    samples: Dict[str, List[float]] = {name: [] for name in config.metrics}
    invalid_reasons: Dict[str, int] = {}

    for _ in range(config.n_resamples):
        idx = _resample_indices(y, rng, stratified=True)
        yb = y[idx]
        pb = p[idx]
        values = _metric_values(yb, pb, config, unit=unit)
        if all(v is None for v in values.values()):
            reason = _invalid_reason(yb, pb)
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            continue
        for name, value in values.items():
            if value is None:
                reason = f"undefined:{name}"
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                continue
            samples[name].append(value)

    return _summarise(
        method=method,
        unit=unit,
        description=description,
        point=point,
        samples=samples,
        invalid_reasons=invalid_reasons,
        n_resamples=config.n_resamples,
        config=config,
        extra=extra,
    )


def participant_cluster_bootstrap(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    participant_ids: Sequence[str],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    metrics: Sequence[str] = DEFAULT_METRICS,
    threshold: float = 0.5,
    aggregation: str = "mean",
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> Dict[str, object]:
    """Participant-cluster bootstrap for EATD (the correct unit there).

    Sessions are reduced to one prediction per participant (mean by default),
    then participants are resampled with replacement within each class. The
    interval is therefore over *people*, not over sessions.
    """
    config = BootstrapConfig(
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
        metrics=tuple(metrics),
        threshold=threshold,
        min_valid_fraction=min_valid_fraction,
    )
    aggregated = aggregate_by_participant(
        y_true, y_prob, participant_ids, aggregation=aggregation
    )
    extra = {
        "n_participants": aggregated["n_units"],
        "n_observations": aggregated["n_observations"],
        "sessions_per_participant": _session_count_summary(
            aggregated["session_counts"]
        ),
        "aggregation": aggregated["aggregation"],
        "resampling": "participants, stratified within class",
    }
    return _run_bootstrap(
        aggregated["labels"],
        aggregated["probs"],
        config,
        method="participant_cluster_bootstrap",
        unit="participant",
        description=(
            "participant-cluster bootstrap: sessions aggregated per participant, "
            "participants resampled within class"
        ),
        extra=extra,
    )


def sample_bootstrap(
    y_true: Sequence[float],
    y_prob: Sequence[float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    metrics: Sequence[str] = DEFAULT_METRICS,
    threshold: float = 0.5,
    min_valid_fraction: float = MIN_VALID_FRACTION,
    dataset: str = "avdd",
) -> Dict[str, object]:
    """Sample-level bootstrap for datasets without released participant ids.

    Valid for AVDD only. Because resampling is at sample level, any repeated
    observations of one participant can land in the same resample and the
    interval will be optimistic; the result says so explicitly.
    """
    config = BootstrapConfig(
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
        metrics=tuple(metrics),
        threshold=threshold,
        min_valid_fraction=min_valid_fraction,
    )
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    extra = {
        "dataset": dataset,
        "participant_identity_available": False,
        "n_observations": int(y.size),
        "resampling": "samples, stratified within class",
    }
    return _run_bootstrap(
        y,
        p,
        config,
        method="sample_level_bootstrap",
        unit="sample",
        description=(
            "sample-level bootstrap: no participant identity available, so "
            "clustered structure cannot be respected"
        ),
        extra=extra,
    )


def bootstrap_for_dataset(
    dataset: str,
    y_true: Sequence[float],
    y_prob: Sequence[float],
    *,
    participant_ids: Optional[Sequence[str]] = None,
    **kwargs: object,
) -> Dict[str, object]:
    """Dispatch to the resampling unit that the dataset actually supports."""
    key = str(dataset).lower()
    if key == "eatd":
        if participant_ids is None:
            raise BootstrapError(
                "EATD bootstrap requires participant_ids: resampling sessions "
                "independently would understate uncertainty"
            )
        return participant_cluster_bootstrap(
            y_true, y_prob, participant_ids, **kwargs  # type: ignore[arg-type]
        )
    if key == "avdd":
        if participant_ids is not None:
            raise BootstrapError(
                "AVDD has no released participant identity; pass no "
                "participant_ids (sample-level bootstrap only)"
            )
        return sample_bootstrap(
            y_true, y_prob, dataset="avdd", **kwargs  # type: ignore[arg-type]
        )
    raise BootstrapError(f"unknown dataset {dataset!r}; expected 'eatd' or 'avdd'")


def _session_count_summary(counts: Iterable[int]) -> Dict[str, object]:
    values = np.asarray(list(counts), dtype=np.float64)
    if values.size == 0:
        return {"min": 0, "median": 0, "max": 0}
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "max": int(values.max()),
    }
