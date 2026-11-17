"""
Grouped cross-validation for Phase 5 (Phase 5B).

The evaluation unit for EATD is the **participant**, never the session: the
three prompted sessions of one participant are correlated and always travel
together. This module builds repeated stratified *group* K-fold partitions from
unit records, and refuses to return anything it cannot verify.

It does not train models, does not read labels of the locked test participants,
and does not fall back to ungrouped splitting when a grouped split fails.

Verified EATD development pool (train + validation, seed-42 split, unchanged):
138 participants / 414 sessions / 45 positive participants (label = 1 iff
``sds_index >= 50``). The 24 locked test participants are excluded before any
label is read, by filtering on ``splits.csv`` assignment.

Exports:
    Unit, Fold, Repetition, SplitError, GroupingError
    participant_group_key, group_members, group_labels
    assert_group_label_consistency, assert_group_key_qualified
    repeated_stratified_group_folds, inner_group_folds
    assert_fold_integrity, assert_no_locked_groups
    eatd_development_units, eatd_locked_test_group_ids, audit_eatd_development_data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPEAT_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)
OUTER_FOLDS = 5
INNER_FOLDS = 4

EATD_DATASET = "eatd"
EATD_DEVELOPMENT_SPLITS = ("train", "validation")
EATD_LOCKED_SPLIT = "test"


class SplitError(RuntimeError):
    """A valid grouped split could not be produced. Never downgraded."""


class GroupingError(RuntimeError):
    """Unit records cannot be grouped safely (missing or inconsistent keys)."""


class SplitIntegrityError(AssertionError):
    """A produced partition violates participant/session isolation."""


# ---------------------------------------------------------------------------
# Unit records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """One observation and the group it belongs to.

    ``unit_id`` is unique (a session for EATD, a sample for AVDD).
    ``group_id`` is the clustering key used for splitting and resampling
    (``participant_key`` for EATD, ``None`` for AVDD where no participant
    identity is released). ``label`` is the 0/1 target; class 1 is positive.
    """

    unit_id: str
    label: int
    group_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("unit_id must be non-empty")
        if int(self.label) not in (0, 1):
            raise ValueError(f"label must be 0 or 1; got {self.label!r}")
        object.__setattr__(self, "label", int(self.label))
        if self.group_id is not None and not self.group_id:
            raise GroupingError(f"unit {self.unit_id!r} has an empty group_id")


def unit_index(units: Iterable[Unit]) -> Dict[str, Unit]:
    """unit_id → Unit, raising on duplicates."""
    out: Dict[str, Unit] = {}
    for u in units:
        if u.unit_id in out:
            raise SplitIntegrityError(f"duplicate unit_id {u.unit_id!r}")
        out[u.unit_id] = u
    return out


def group_members(units: Iterable[Unit]) -> Dict[str, List[str]]:
    """group_id → ordered list of its unit_ids.

    Raises:
        GroupingError: if any unit has no group_id. Grouped CV is never applied
            to records that cannot be grouped.
    """
    out: Dict[str, List[str]] = {}
    for u in units:
        if u.group_id is None:
            raise GroupingError(
                f"unit {u.unit_id!r} has no group_id; grouped splitting requires "
                "a clustering key (participant for EATD) and must not fall back "
                "to independent rows"
            )
        out.setdefault(u.group_id, []).append(u.unit_id)
    return out


def group_labels(units: Iterable[Unit]) -> Dict[str, int]:
    """group_id → label, raising if a group is internally inconsistent."""
    labels: Dict[str, int] = {}
    for u in units:
        if u.group_id is None:
            raise GroupingError(f"unit {u.unit_id!r} has no group_id")
        prev = labels.get(u.group_id)
        if prev is None:
            labels[u.group_id] = u.label
        elif prev != u.label:
            raise GroupingError(
                f"group {u.group_id!r} has inconsistent labels ({prev} vs {u.label}); "
                "the participant label must be one value"
            )
    return labels


def assert_group_label_consistency(units: Iterable[Unit]) -> Dict[str, int]:
    """Public alias with a clearer name; returns group_id → label."""
    return group_labels(units)


def participant_group_key(source_partition: str, subject_id: object) -> str:
    """The Phase 1 participant key: ``(source_partition, subject_id)``.

    58 numeric subject ids repeat across the legacy ``train/`` and
    ``validation/`` folders and are different people; the partition must be part
    of the key.
    """
    partition = str(source_partition or "").strip()
    subject = str(subject_id).strip()
    if not partition or not subject:
        raise GroupingError("both source_partition and subject_id are required")
    return f"eatd_{partition}_{subject}"


def assert_group_key_qualified(records: Iterable[Mapping[str, object]]) -> None:
    """Reject group keys that collapse (source_partition, subject_id) pairs.

    Each record must carry ``source_partition``, ``subject_id`` and ``group_id``.
    Raises :class:`GroupingError` if one group_id maps to more than one
    (partition, subject) pair, or if one pair maps to more than one group_id.
    """
    pair_to_group: Dict[Tuple[str, str], str] = {}
    group_to_pair: Dict[str, Tuple[str, str]] = {}
    for i, rec in enumerate(records):
        missing = [k for k in ("source_partition", "subject_id", "group_id") if k not in rec]
        if missing:
            raise GroupingError(f"record {i} is missing {missing}")
        pair = (str(rec["source_partition"]), str(rec["subject_id"]))
        group = str(rec["group_id"])
        if pair_to_group.setdefault(pair, group) != group:
            raise GroupingError(
                f"(source_partition, subject_id)={pair} maps to multiple group "
                f"keys ({pair_to_group[pair]!r} and {group!r})"
            )
        if group_to_pair.setdefault(group, pair) != pair:
            expected_key = participant_group_key(*pair)
            raise GroupingError(
                f"group key {group!r} covers more than one participant: {pair} and "
                f"{group_to_pair[group]}. Numeric subject ids repeat across source "
                f"partitions; expected key {expected_key!r}."
            )


# ---------------------------------------------------------------------------
# Fold records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fold:
    """One train/evaluation partition of a repetition."""

    stage: str  # "outer" | "inner"
    repeat_seed: int
    fold: int
    n_splits: int
    train_unit_ids: Tuple[str, ...]
    eval_unit_ids: Tuple[str, ...]
    train_group_ids: Tuple[str, ...]
    eval_group_ids: Tuple[str, ...]
    outer_fold: Optional[int] = None

    def __post_init__(self) -> None:
        if self.stage not in {"outer", "inner"}:
            raise ValueError(f"unknown fold stage {self.stage!r}")
        if not 0 <= self.fold < self.n_splits:
            raise ValueError(f"fold index {self.fold} outside 0..{self.n_splits - 1}")
        if not self.train_unit_ids or not self.eval_unit_ids:
            raise SplitIntegrityError(f"fold {self.fold} has an empty partition")
        if set(self.train_unit_ids) & set(self.eval_unit_ids):
            raise SplitIntegrityError(
                f"fold {self.fold}: unit(s) in both train and evaluation: "
                f"{sorted(set(self.train_unit_ids) & set(self.eval_unit_ids))[:5]}"
            )
        if set(self.train_group_ids) & set(self.eval_group_ids):
            raise SplitIntegrityError(
                f"fold {self.fold}: group(s) in both train and evaluation: "
                f"{sorted(set(self.train_group_ids) & set(self.eval_group_ids))[:5]}"
            )

    @property
    def eval_fold(self) -> int:
        return self.fold

    def to_dict(self, units: Mapping[str, Unit]) -> Dict[str, object]:
        """Serialisable metadata for this fold, with per-partition counts."""
        return fold_metadata(self, units)


def fold_metadata(fold: Fold, units: Mapping[str, Unit]) -> Dict[str, object]:
    """Participant ids, unit ids, class and session counts for one fold."""
    missing = [u for u in (*fold.train_unit_ids, *fold.eval_unit_ids) if u not in units]
    if missing:
        raise SplitIntegrityError(f"fold references unknown units: {missing[:5]}")

    def side(unit_ids: Sequence[str], group_ids: Sequence[str]) -> Dict[str, object]:
        labels = [units[u].label for u in unit_ids]
        n_pos = int(sum(labels))
        n_units = len(unit_ids)
        return {
            "n_units": n_units,
            "n_groups": len(group_ids),
            "n_positive_groups": int(
                sum(1 for g in group_ids if _group_label(units, g) == 1)
            ),
            "n_negative_groups": int(
                sum(1 for g in group_ids if _group_label(units, g) == 0)
            ),
            "n_positive_units": n_pos,
            "n_negative_units": n_units - n_pos,
            "positive_rate_units": round(n_pos / n_units, 4) if n_units else None,
            "group_ids": list(group_ids),
            "unit_ids": list(unit_ids),
        }

    train = side(fold.train_unit_ids, fold.train_group_ids)
    evaluation = side(fold.eval_unit_ids, fold.eval_group_ids)
    return {
        "stage": fold.stage,
        "outer_fold": fold.outer_fold,
        "repeat_seed": fold.repeat_seed,
        "fold": fold.fold,
        "n_splits": fold.n_splits,
        "train": train,
        "eval": evaluation,
        "checks": {
            "unit_overlap": len(set(fold.train_unit_ids) & set(fold.eval_unit_ids)),
            "group_overlap": len(set(fold.train_group_ids) & set(fold.eval_group_ids)),
            "eval_has_both_classes": bool(
                evaluation["n_positive_groups"] and evaluation["n_negative_groups"]
            ),
        },
    }


def _group_label(units: Mapping[str, Unit], group_id: str) -> int:
    found = {units[u].label for u in units if units[u].group_id == group_id}
    if len(found) > 1:
        raise GroupingError(f"group {group_id!r} has inconsistent labels {sorted(found)}")
    return found.pop() if found else 0


@dataclass(frozen=True)
class Repetition:
    """All folds produced by one random seed."""

    seed: int
    stage: str
    n_splits: int
    folds: Tuple[Fold, ...]
    outer_fold: Optional[int] = None

    def to_dict(self, units: Mapping[str, Unit]) -> Dict[str, object]:
        return {
            "repeat_seed": self.seed,
            "stage": self.stage,
            "n_splits": self.n_splits,
            "outer_fold": self.outer_fold,
            "folds": [f.to_dict(units) for f in self.folds],
        }


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def _grouped_folds(
    groups: Sequence[str],
    group_label: Mapping[str, int],
    *,
    n_splits: int,
    seed: int,
    stage: str,
    members: Mapping[str, Sequence[str]],
) -> List[Fold]:
    """StratifiedGroupKFold over groups; no fallback on failure."""
    from sklearn.model_selection import StratifiedGroupKFold

    if n_splits < 2:
        raise SplitError(f"n_splits must be >= 2; got {n_splits}")
    y = np.asarray([group_label[g] for g in groups], dtype=np.int64)
    X = np.zeros((len(groups), 1), dtype=np.uint8)
    group_array = np.asarray(list(groups), dtype=object)

    try:
        raw_splits = list(
            StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            ).split(X, y, groups=group_array)
        )
    except Exception as exc:  # sklearn precondition, or an unsupported case
        raise SplitError(
            f"StratifiedGroupKFold(n_splits={n_splits}, shuffle=True, "
            f"random_state={seed}) could not split {len(groups)} groups with "
            f"class counts {{0: {int((y == 0).sum())}, 1: {int((y == 1).sum())}}}: "
            f"{type(exc).__name__}: {exc}. No fallback splitter is used."
        ) from exc

    if len(raw_splits) != n_splits:
        raise SplitError(
            f"expected {n_splits} folds for seed {seed}, got {len(raw_splits)}"
        )

    eval_group_sets = [set(group_array[idx].tolist()) for _, idx in raw_splits]
    covered: set = set()
    for i, eval_groups in enumerate(eval_group_sets):
        dup = covered & eval_groups
        if dup:
            raise SplitError(
                f"seed {seed}: group(s) assigned to multiple evaluation folds: "
                f"{sorted(dup)[:5]}"
            )
        covered |= eval_groups
    if covered != set(groups):
        raise SplitError(
            f"seed {seed}: {len(set(groups) - covered)} group(s) never evaluated: "
            f"{sorted(set(groups) - covered)[:5]}"
        )

    folds: List[Fold] = []
    for fold_idx, (train_idx, _) in enumerate(raw_splits):
        eval_groups = sorted(eval_group_sets[fold_idx])
        train_groups = sorted(set(groups) - set(eval_groups))
        n_eval_pos = sum(group_label[g] for g in eval_groups)
        n_eval_neg = len(eval_groups) - n_eval_pos
        if n_eval_pos == 0 or n_eval_neg == 0:
            raise SplitError(
                f"seed {seed}, fold {fold_idx}: evaluation part has a single class "
                f"({n_eval_pos} positive / {n_eval_neg} negative); this split is "
                "not usable and no alternative splitter is substituted"
            )
        train_units = tuple(u for g in train_groups for u in members[g])
        eval_units = tuple(u for g in eval_groups for u in members[g])
        folds.append(
            Fold(
                stage=stage,
                repeat_seed=seed,
                fold=fold_idx,
                n_splits=n_splits,
                train_unit_ids=train_units,
                eval_unit_ids=eval_units,
                train_group_ids=tuple(train_groups),
                eval_group_ids=tuple(eval_groups),
            )
        )
    return folds


def repeated_stratified_group_folds(
    units: Iterable[Unit],
    *,
    n_splits: int = OUTER_FOLDS,
    seeds: Sequence[int] = REPEAT_SEEDS,
    stage: str = "outer",
    locked_group_ids: Optional[Iterable[str]] = None,
) -> List[Repetition]:
    """Outer repeated stratified group K-fold (5 folds × 5 seeds by default).

    Args:
        units: development units only. Each group must carry exactly one label.
        n_splits: number of folds.
        seeds: pre-registered repeat seeds.
        stage: ``"outer"`` (or ``"inner"`` for internal reuse).
        locked_group_ids: participant keys of the locked test set; any overlap
            with ``units`` is a hard error.

    Returns:
        One :class:`Repetition` per seed, in the order the seeds were given.

    Raises:
        SplitError: no valid grouped split for a seed.
        SplitIntegrityError: a produced partition leaks a group or a unit.
    """
    unit_list = list(units)
    index = unit_index(unit_list)
    members = group_members(unit_list)
    labels = group_labels(unit_list)
    groups = sorted(members)

    locked = set(locked_group_ids or ())
    if locked:
        leaked = sorted(set(groups) & locked)
        if leaked:
            raise SplitIntegrityError(
                f"development units include {len(leaked)} locked-test participant(s): "
                f"{leaked[:5]}. The locked test set is never split."
            )

    if len(seeds) == 0:
        raise SplitError("at least one repeat seed is required")
    if len(set(seeds)) != len(seeds):
        raise SplitError(f"duplicate repeat seeds: {list(seeds)}")

    repetitions: List[Repetition] = []
    for seed in seeds:
        folds = _grouped_folds(
            groups,
            labels,
            n_splits=n_splits,
            seed=int(seed),
            stage=stage,
            members=members,
        )
        repetition = Repetition(
            seed=int(seed), stage=stage, n_splits=n_splits, folds=tuple(folds)
        )
        assert_repetition_integrity(index, repetition, locked_group_ids=locked)
        repetitions.append(repetition)
    return repetitions


def assert_repetition_integrity(
    units: Mapping[str, Unit],
    repetition: Repetition,
    *,
    locked_group_ids: Optional[Iterable[str]] = None,
) -> None:
    """Verify a repetition covers every unit exactly once in its eval parts."""
    if len(repetition.folds) != repetition.n_splits:
        raise SplitIntegrityError(
            f"repetition {repetition.seed}: expected {repetition.n_splits} folds, "
            f"got {len(repetition.folds)}"
        )
    seen_units: Dict[str, int] = {}
    seen_groups: Dict[str, int] = {}
    locked = set(locked_group_ids or ())
    for fold in repetition.folds:
        assert_fold_integrity(units, fold, locked_group_ids=locked)
        for u in fold.eval_unit_ids:
            seen_units[u] = seen_units.get(u, 0) + 1
        for g in fold.eval_group_ids:
            seen_groups[g] = seen_groups.get(g, 0) + 1

    missing_units = sorted(set(units) - set(seen_units))
    if missing_units:
        raise SplitIntegrityError(
            f"repetition {repetition.seed}: {len(missing_units)} unit(s) never "
            f"evaluated: {missing_units[:5]}"
        )
    repeated = sorted(u for u, c in seen_units.items() if c > 1)
    if repeated:
        raise SplitIntegrityError(
            f"repetition {repetition.seed}: unit(s) evaluated more than once: "
            f"{repeated[:5]}"
        )
    repeated_groups = sorted(g for g, c in seen_groups.items() if c > 1)
    if repeated_groups:
        raise SplitIntegrityError(
            f"repetition {repetition.seed}: participant(s) appear in more than one "
            f"outer evaluation fold: {repeated_groups[:5]}"
        )
    missing_groups = sorted({u.group_id for u in units.values()} - set(seen_groups))
    if missing_groups:
        raise SplitIntegrityError(
            f"repetition {repetition.seed}: participant(s) never evaluated: "
            f"{missing_groups[:5]}"
        )


def assert_fold_integrity(
    units: Mapping[str, Unit],
    fold: Fold,
    *,
    locked_group_ids: Optional[Iterable[str]] = None,
) -> None:
    """Assert unit/group isolation, session integrity and locked-test exclusion."""
    unknown = [
        u for u in (*fold.train_unit_ids, *fold.eval_unit_ids) if u not in units
    ]
    if unknown:
        raise SplitIntegrityError(f"fold references unknown units: {unknown[:5]}")

    # 1. No unit in both partitions.
    both_units = set(fold.train_unit_ids) & set(fold.eval_unit_ids)
    if both_units:
        raise SplitIntegrityError(
            f"fold {fold.fold}: session(s) in both train and evaluation: "
            f"{sorted(both_units)[:5]}"
        )

    # 2. No group in both partitions.
    both_groups = set(fold.train_group_ids) & set(fold.eval_group_ids)
    if both_groups:
        raise SplitIntegrityError(
            f"fold {fold.fold}: participant(s) in both train and evaluation: "
            f"{sorted(both_groups)[:5]}"
        )

    # 3. Every participant's sessions stay together.
    train_groups_from_units = {units[u].group_id for u in fold.train_unit_ids}
    eval_groups_from_units = {units[u].group_id for u in fold.eval_unit_ids}
    if train_groups_from_units != set(fold.train_group_ids):
        raise SplitIntegrityError(
            f"fold {fold.fold}: recorded train groups disagree with unit groups"
        )
    if eval_groups_from_units != set(fold.eval_group_ids):
        raise SplitIntegrityError(
            f"fold {fold.fold}: recorded eval groups disagree with unit groups"
        )
    for group in fold.train_group_ids:
        members = {u for u in units if units[u].group_id == group}
        if not members <= set(fold.train_unit_ids):
            raise SplitIntegrityError(
                f"fold {fold.fold}: participant {group!r} has sessions in both "
                "partitions (its sessions were split across folds)"
            )

    # 4. The locked test set never appears in any fold.
    if locked_group_ids is not None:
        assert_no_locked_groups(
            set(fold.train_group_ids) | set(fold.eval_group_ids),
            locked_group_ids,
            operation=f"fold {fold.fold} of {fold.stage} repetition {fold.repeat_seed}",
        )


def assert_no_locked_groups(
    group_ids: Iterable[str],
    locked_group_ids: Iterable[str],
    *,
    operation: str = "split construction",
) -> None:
    """Raise if any locked-test participant reached a development partition."""
    locked = set(locked_group_ids)
    if not locked:
        return
    leaked = sorted(set(group_ids) & locked)
    if leaked:
        raise SplitIntegrityError(
            f"{operation}: {len(leaked)} locked-test participant(s) present: "
            f"{leaked[:5]}"
        )


def inner_group_folds(
    units: Mapping[str, Unit],
    outer_fold: Fold,
    *,
    n_splits: int = INNER_FOLDS,
    seed: Optional[int] = None,
    locked_group_ids: Optional[Iterable[str]] = None,
) -> Repetition:
    """4-fold stratified group CV *inside* one outer training partition.

    The outer evaluation units are never passed in: only
    ``outer_fold.train_unit_ids`` are grouped, and the function re-verifies that
    no outer evaluation participant appears in the result. Inner folds therefore
    cannot see outer evaluation data, and cannot see the locked test set.

    The inner seed defaults to the outer repetition seed (declared, not tuned).
    """
    if outer_fold.stage != "outer":
        raise SplitError("inner CV must be built from an outer fold")

    outer_eval_groups = set(outer_fold.eval_group_ids)
    train_units = [units[u] for u in outer_fold.train_unit_ids]
    inner_index = unit_index(train_units)
    inner_members = group_members(train_units)
    inner_labels = group_labels(train_units)

    leaked_outer = sorted(set(inner_members) & outer_eval_groups)
    if leaked_outer:
        raise SplitIntegrityError(
            f"inner CV would include outer evaluation participant(s): {leaked_outer[:5]}"
        )
    if locked_group_ids is not None:
        assert_no_locked_groups(
            inner_members.keys(), locked_group_ids, operation="inner CV construction"
        )

    inner_seed = int(outer_fold.repeat_seed if seed is None else seed)
    groups = sorted(inner_members)

    # The outer training part must still hold both classes in every inner fold.
    folds = _grouped_folds(
        groups,
        inner_labels,
        n_splits=n_splits,
        seed=inner_seed,
        stage="inner",
        members=inner_members,
    )
    for fold in folds:
        # Re-assert, explicitly, the two isolation properties of inner CV.
        if set(fold.eval_group_ids) & outer_eval_groups:
            raise SplitIntegrityError(
                f"inner fold {fold.fold}: outer evaluation participant(s) present: "
                f"{sorted(set(fold.eval_group_ids) & outer_eval_groups)[:5]}"
            )
        if set(fold.train_group_ids) & outer_eval_groups:
            raise SplitIntegrityError(
                f"inner fold {fold.fold}: outer evaluation participant(s) in inner "
                f"train: {sorted(set(fold.train_group_ids) & outer_eval_groups)[:5]}"
            )
        if locked_group_ids is not None:
            assert_no_locked_groups(
                set(fold.train_group_ids) | set(fold.eval_group_ids),
                locked_group_ids,
                operation=f"inner fold {fold.fold} (outer {outer_fold.fold}, "
                f"seed {outer_fold.repeat_seed})",
            )

    repetition = Repetition(
        seed=inner_seed,
        stage="inner",
        n_splits=n_splits,
        folds=tuple(
            Fold(
                stage="inner",
                repeat_seed=inner_seed,
                fold=f.fold,
                n_splits=n_splits,
                train_unit_ids=f.train_unit_ids,
                eval_unit_ids=f.eval_unit_ids,
                train_group_ids=f.train_group_ids,
                eval_group_ids=f.eval_group_ids,
                outer_fold=outer_fold.fold,
            )
            for f in folds
        ),
        outer_fold=outer_fold.fold,
    )
    assert_repetition_integrity(
        inner_index, repetition, locked_group_ids=locked_group_ids
    )
    return repetition


# ---------------------------------------------------------------------------
# EATD development-pool access (read-only)
# ---------------------------------------------------------------------------

def eatd_locked_test_group_ids(splits: Optional[Mapping[str, str]] = None) -> frozenset:
    """Participant keys assigned to the locked test split — keys only, no labels."""
    from ml.phase2.dataset import load_splits

    assignment = dict(splits) if splits is not None else load_splits()
    return frozenset(k for k, s in assignment.items() if s == EATD_LOCKED_SPLIT)


def eatd_development_units(
    *,
    manifest: Optional[Sequence[dict]] = None,
    splits: Optional[Mapping[str, str]] = None,
    participants: Optional[Sequence[dict]] = None,
) -> List[Unit]:
    """Development units (train + validation participants only).

    Locked-test rows are filtered out by participant key **before** any label is
    read, so test labels never enter the harness. The canonical participant
    label from ``participants.csv`` is used and cross-checked against the
    manifest row.

    Returns one :class:`Unit` per session (414 units for the verified pool).
    """
    from ml.phase2.dataset import load_manifest, load_participants, load_splits

    assignment = dict(splits) if splits is not None else load_splits()
    rows = list(manifest) if manifest is not None else load_manifest()
    people = list(participants) if participants is not None else load_participants()

    development_keys = {
        key for key, split in assignment.items() if split in EATD_DEVELOPMENT_SPLITS
    }
    labels = {
        p["participant_key"]: int(p["label"])
        for p in people
        if p["participant_key"] in development_keys
    }
    if not labels:
        raise SplitIntegrityError("no development participants found")

    units: List[Unit] = []
    for row in rows:
        key = row["participant_key"]
        if key not in development_keys:
            continue
        label = labels.get(key)
        if label is None:
            raise SplitIntegrityError(f"{key!r} has no development participant record")
        if int(row["label"]) != label:
            raise SplitIntegrityError(
                f"{key!r}: manifest label {row['label']} disagrees with "
                f"participants.csv label {label}"
            )
        units.append(
            Unit(unit_id=f"{key}/{row['topic']}", group_id=key, label=label)
        )
    return units


def audit_eatd_development_data(
    *,
    seeds: Sequence[int] = REPEAT_SEEDS,
    n_splits: int = OUTER_FOLDS,
    inner_splits: int = INNER_FOLDS,
) -> Dict[str, object]:
    """Task A + B + C audit: verify counts, splits and both CV loops.

    Reads the locked split assignment only to exclude it; no locked-test label is
    used. Returns a JSON-serialisable report (written by
    ``scripts/phase5/build_split_audit.py``). No model is trained.
    """
    from ml.phase2 import dataset as phase2

    splits = phase2.load_splits()
    participants = phase2.load_participants()
    manifest = phase2.load_manifest()

    # The Phase 1/Phase 2 integrity assertions must still pass, unchanged.
    per_split = phase2.assert_split_integrity(splits, participants)

    development_keys = {
        k for k, s in splits.items() if s in EATD_DEVELOPMENT_SPLITS
    }
    locked_keys = eatd_locked_test_group_ids(splits)

    dev_people = [p for p in participants if p["participant_key"] in development_keys]
    dev_sessions = [r for r in manifest if r["participant_key"] in development_keys]
    locked_people = [p for p in participants if p["participant_key"] in locked_keys]

    units = eatd_development_units(
        manifest=manifest, splits=splits, participants=participants
    )
    index = unit_index(units)
    members = group_members(units)
    labels = group_labels(units)

    session_partitions: Dict[str, set] = {}
    for key in development_keys | set(locked_keys):
        session_partitions.setdefault(key, set())
    # No session may appear in more than one split: sessions are keyed by
    # (participant_key, topic) and the participant key itself is unique.
    unit_to_group = {u.unit_id: u.group_id for u in units}
    if len(unit_to_group) != len(units):
        raise SplitIntegrityError("duplicate session id in development units")

    unassigned = sorted({p["participant_key"] for p in participants} - set(splits))
    if unassigned:
        raise SplitIntegrityError(f"participants without a split: {unassigned[:5]}")

    # Candidate risk (Task A): grouping by bare subject_id would merge people.
    by_subject: Dict[str, set] = {}
    for p in dev_people:
        by_subject.setdefault(str(p["source_subject_id"]), set()).add(
            p["participant_key"]
        )
    colliding_subject_ids = {
        sid: sorted(keys) for sid, keys in by_subject.items() if len(keys) > 1
    }

    repetitions = repeated_stratified_group_folds(
        units, n_splits=n_splits, seeds=seeds, locked_group_ids=locked_keys
    )

    outer_report = []
    inner_summary = []
    for repetition in repetitions:
        folds = []
        for fold in repetition.folds:
            folds.append(fold.to_dict(index))
        inner_rows = []
        for fold in repetition.folds:
            inner = inner_group_folds(
                index, fold, n_splits=inner_splits, locked_group_ids=locked_keys
            )
            inner_rows.append(
                {
                    "outer_fold": fold.fold,
                    "inner_seed": inner.seed,
                    "n_inner_folds": inner.n_splits,
                    "folds": [
                        {
                            "fold": f.fold,
                            "n_train_groups": len(f.train_group_ids),
                            "n_eval_groups": len(f.eval_group_ids),
                            "n_train_units": len(f.train_unit_ids),
                            "n_eval_units": len(f.eval_unit_ids),
                            "eval_positive_groups": int(
                                sum(labels[g] for g in f.eval_group_ids)
                            ),
                            "eval_negative_groups": int(
                                len(f.eval_group_ids)
                                - sum(labels[g] for g in f.eval_group_ids)
                            ),
                            "group_overlap": len(
                                set(f.train_group_ids) & set(f.eval_group_ids)
                            ),
                            "outer_eval_group_overlap": len(
                                (set(f.train_group_ids) | set(f.eval_group_ids))
                                & set(fold.eval_group_ids)
                            ),
                        }
                        for f in inner.folds
                    ],
                }
            )
        outer_report.append(
            {
                "repeat_seed": repetition.seed,
                "n_folds": repetition.n_splits,
                "folds": folds,
            }
        )
        inner_summary.append({"repeat_seed": repetition.seed, "outer_folds": inner_rows})

    def coverage(repetition: Repetition) -> Dict[str, object]:
        evaluated = [u for f in repetition.folds for u in f.eval_unit_ids]
        groups = [g for f in repetition.folds for g in f.eval_group_ids]
        return {
            "units_evaluated": len(evaluated),
            "units_unique": len(set(evaluated)),
            "groups_evaluated": len(groups),
            "groups_unique": len(set(groups)),
            "each_participant_once": len(groups) == len(set(groups)),
            "covers_all_development_units": set(evaluated) == set(index),
        }

    return {
        "phase": "5B",
        "title": "EATD development data and cross-validation split audit",
        "generated_at": None,  # filled by the writer
        "dataset": EATD_DATASET,
        "status": "audit_only_no_training",
        "training_performed": False,
        "locked_test_evaluated": False,
        "source_artifacts": {
            "manifest": "data/processed/eatd/manifest.csv",
            "participants": "data/processed/eatd/participants.csv",
            "splits": "data/processed/eatd/splits.csv",
            "split_report": "data/processed/eatd/split_report.json",
        },
        "verified_counts": {
            "participants_total": len(participants),
            "development_participants": len(dev_people),
            "locked_test_participants": len(locked_people),
            "development_sessions": len(dev_sessions),
            "development_positive_participants": int(
                sum(p["label"] for p in dev_people)
            ),
            "development_negative_participants": int(
                len(dev_people) - sum(p["label"] for p in dev_people)
            ),
            "locked_test_positive_participants": int(
                sum(p["label"] for p in locked_people)
            ),
            "locked_test_negative_participants": int(
                len(locked_people) - sum(p["label"] for p in locked_people)
            ),
            "development_groups": len(members),
            "development_units": len(units),
            "sessions_per_participant": sorted(
                {len(v) for v in members.values()}
            ),
            "phase2_integrity_counts": per_split,
        },
        "expectations": {
            "participants_total": 162,
            "development_participants": 138,
            "locked_test_participants": 24,
            "development_sessions": 414,
            "development_positive_participants": 45,
            "locked_test_positive_participants": 8,
        },
        "checks": {
            "phase2_assert_split_integrity_passed": True,
            "no_participant_in_multiple_splits": per_split
            == {"train": 114, "validation": 24, "test": 24},
            "no_session_in_multiple_splits": len(unit_to_group) == len(units),
            "every_development_session_maps_to_one_participant": all(
                u.group_id in members for u in units
            ),
            "participant_label_consistent": True,
            "locked_test_excluded_from_development_units": not (
                set(members) & locked_keys
            ),
            "group_key_is_partition_qualified": all(
                key.startswith("eatd_") for key in members
            ),
            "subject_ids_colliding_across_partitions": len(colliding_subject_ids),
        },
        "group_key": {
            "definition": "(source_partition, subject_id) → participant_key",
            "numeric_subject_ids_shared_by_two_development_participants": len(
                colliding_subject_ids
            ),
            "example_collisions": dict(
                list(colliding_subject_ids.items())[:5]
            ),
            "warning": (
                "Grouping by subject_id alone merges distinct people; the "
                "participant_key must keep the source partition."
            ),
        },
        "outer_cv": {
            "splitter": "StratifiedGroupKFold(shuffle=True)",
            "n_splits": n_splits,
            "seeds": list(seeds),
            "n_repetitions": len(repetitions),
            "n_outer_folds": len(repetitions) * n_splits,
            "evaluation_unit": "participant",
            "session_level_splitting": False,
            "coverage": {
                str(r.seed): coverage(r) for r in repetitions
            },
            "repetitions": outer_report,
        },
        "inner_cv": {
            "splitter": "StratifiedGroupKFold(shuffle=True)",
            "n_splits": inner_splits,
            "seed_policy": "inner seed = outer repeat seed (declared, not tuned)",
            "outer_evaluation_data_used": False,
            "locked_test_data_used": False,
            "repetitions": inner_summary,
        },
        "limitations": [
            "No model was trained; these are split partitions only.",
            "Fold membership depends on scikit-learn's StratifiedGroupKFold "
            "implementation; the version is recorded in the harness documentation.",
            "Repetitions differ only by seed; they are not independent samples.",
            "Every outer evaluation fold holds ~9 positive participants, so any "
            "metric estimated per fold is inherently wide.",
        ],
    }
