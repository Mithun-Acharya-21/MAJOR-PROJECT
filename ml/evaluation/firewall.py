"""
Locked-test firewall for Phase 5 (Phase 5B).

The EATD locked test set (24 participants, 8 positive) is the project's only
unbiased estimate, and it is worth exactly one use. Everything here exists to
make that single use deliberate and auditable:

* the firewall knows the locked **participant keys** and nothing else — it is
  constructed from split assignments, never from labels;
* development data can be filtered or validated against it;
* any selection, threshold, calibration, preprocessing or inner-CV operation
  that touches a locked participant is refused;
* the one authorised final evaluation must be registered before it runs, with a
  pre-registration record and an access-ledger entry.

Phase 5B does not evaluate the locked test set, so no access is authorised here.

Exports:
    LockedTestFirewall, LockedTestAccessError, LockedTestAccessRecord
    REQUIRED_PREREGISTRATION_FIELDS
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

LOCKED_SPLIT_NAME = "test"
REQUIRED_PREREGISTRATION_FIELDS = (
    "experiment_id",
    "model_configuration",
    "threshold_rule",
    "primary_metric",
    "evaluation_script",
    "script_sha256",
    "registered_at",
)


class LockedTestAccessError(RuntimeError):
    """An operation tried to read or use locked-test data."""


@dataclass
class LockedTestAccessRecord:
    """Ledger entry for one authorised locked-test evaluation."""

    experiment_id: str
    reason: str
    preregistration: Mapping[str, object]
    code_revision: str
    timestamp: str
    n_locked_participants: int
    dataset: str
    ledger: str = "in-memory"
    status: str = "running"
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "reason": self.reason,
            "code_revision": self.code_revision,
            "authorized_at": self.timestamp,
            "completed_at": self.completed_at,
            "status": self.status,
            "n_locked_participants": self.n_locked_participants,
            "scope": "final_evaluation_only",
            "preregistration": dict(self.preregistration),
            "ledger": self.ledger,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LockedTestFirewall:
    """Participant-key guard around the locked EATD test set.

    Args:
        locked_group_ids: participant keys assigned to the locked test split.
        dataset: dataset name recorded in the ledger.
        ledger_path: optional JSON-lines file; one line per authorised access.
        code_revision: git revision recorded with each access.
    """

    def __init__(
        self,
        locked_group_ids: Iterable[str],
        *,
        dataset: str = "eatd",
        ledger_path: Optional[Path] = None,
        code_revision: str = "unknown",
    ) -> None:
        ids = frozenset(str(x) for x in locked_group_ids)
        if not ids:
            raise LockedTestAccessError(
                "firewall requires at least one locked participant key; an empty "
                "firewall would silently allow everything"
            )
        self._locked = ids
        self.dataset = dataset
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.code_revision = code_revision
        self._access_log: List[LockedTestAccessRecord] = []

    # -- construction -------------------------------------------------------

    @classmethod
    def from_eatd_splits(
        cls,
        splits: Optional[Mapping[str, str]] = None,
        *,
        split_name: str = LOCKED_SPLIT_NAME,
        **kwargs: object,
    ) -> "LockedTestFirewall":
        """Build from ``splits.csv`` assignments only — no labels are read."""
        if splits is None:
            from ml.phase2.dataset import load_splits

            splits = load_splits()
        locked = [key for key, split in splits.items() if split == split_name]
        return cls(locked, dataset="eatd", **kwargs)  # type: ignore[arg-type]

    # -- inspection ---------------------------------------------------------

    @property
    def locked_group_ids(self) -> frozenset:
        return self._locked

    @property
    def n_locked_participants(self) -> int:
        return len(self._locked)

    @property
    def labels_available(self) -> bool:
        """The firewall never holds locked-test labels."""
        return False

    @property
    def access_log(self) -> tuple:
        return tuple(r.to_dict() for r in self._access_log)

    def is_locked(self, group_id: str) -> bool:
        return str(group_id) in self._locked

    def locked_overlap(self, group_ids: Iterable[str]) -> List[str]:
        return sorted({str(g) for g in group_ids} & self._locked)

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset,
            "n_locked_participants": len(self._locked),
            "locked_group_ids_recorded": True,
            "labels_available": False,
            "authorized_accesses": len(self._access_log),
            "scope": (
                "blocks locked-test labels from model selection, locked-test data "
                "from inner CV, locked-test predictions from threshold/calibration "
                "fitting, and locked-test metrics from hyperparameter selection"
            ),
        }

    # -- guards -------------------------------------------------------------

    def assert_not_locked(
        self, group_ids: Iterable[str], *, operation: str = "operation"
    ) -> None:
        """Refuse an operation whose inputs include a locked participant."""
        leaked = self.locked_overlap(group_ids)
        if leaked:
            raise LockedTestAccessError(
                f"{operation}: refusing to use {len(leaked)} locked-test "
                f"participant(s) {leaked[:5]}. The locked test set may only be "
                "used by an authorised final evaluation."
            )

    def assert_selection_safe(
        self, group_ids: Iterable[str], *, operation: str = "selection"
    ) -> None:
        """Alias with selection wording (thresholds, calibration, features)."""
        self.assert_not_locked(group_ids, operation=operation)

    def filter_development(
        self, units: Iterable[object], *, operation: str = "development filtering"
    ) -> tuple:
        """Return the units whose group is not locked; assert none were dropped
        by accident when the caller expected a clean development set.

        Accepts any object with a ``group_id`` attribute (e.g.
        :class:`ml.evaluation.splits.Unit`) or a plain string participant key.
        """
        kept = []
        dropped = []
        for unit in units:
            if isinstance(unit, str):
                group = unit
            else:
                group = getattr(unit, "group_id", None)
                if group is None:
                    raise LockedTestAccessError(
                        f"{operation}: cannot determine the group of {unit!r}"
                    )
            if self.is_locked(group):
                dropped.append(str(group))
            else:
                kept.append(unit)
        if dropped:
            raise LockedTestAccessError(
                f"{operation}: {len(dropped)} locked-test participant(s) were "
                f"present in development input: {sorted(set(dropped))[:5]}. "
                "Callers must exclude the locked split explicitly, not rely on "
                "this filter to clean contaminated input."
            )
        return tuple(kept)

    def assert_partition_pristine(self, partitions: Iterable[str]) -> None:
        """A model-selection run must never declare a locked-test partition."""
        bad = [p for p in partitions if p == "locked_test"]
        if bad:
            raise LockedTestAccessError(
                "model selection cannot declare a 'locked_test' partition; the "
                "locked set is evaluated once, by an authorised final run"
            )

    # -- the single authorised use -----------------------------------------

    @contextmanager
    def authorize_final_evaluation(
        self,
        *,
        experiment_id: str,
        reason: str,
        preregistration: Mapping[str, object],
        code_revision: Optional[str] = None,
        script_path: Optional[Path] = None,
    ) -> Iterator[LockedTestAccessRecord]:
        """Register, then yield, one final locked-test evaluation.

        Nothing runs until the pre-registration is complete; the ledger entry is
        written *before* the body executes so an interrupted run is visible.
        """
        record = self._validate_authorization(
            experiment_id=experiment_id,
            reason=reason,
            preregistration=preregistration,
            code_revision=code_revision,
            script_path=script_path,
        )
        self._access_log.append(record)
        self._append_ledger(record)
        try:
            yield record
        except Exception:
            record.status = "failed"
            record.completed_at = _now()
            self._rewrite_ledger()
            raise
        else:
            record.status = "completed"
            record.completed_at = _now()
            self._rewrite_ledger()

    def _validate_authorization(
        self,
        *,
        experiment_id: str,
        reason: str,
        preregistration: Mapping[str, object],
        code_revision: Optional[str],
        script_path: Optional[Path],
    ) -> LockedTestAccessRecord:
        if not experiment_id or not str(experiment_id).strip():
            raise LockedTestAccessError("final evaluation needs an experiment_id")
        if not reason or not str(reason).strip():
            raise LockedTestAccessError(
                "final evaluation needs a recorded reason (e.g. an experiment id "
                "from the Phase 5 matrix)"
            )
        if len(self._access_log) >= 1:
            raise LockedTestAccessError(
                "the locked test set has already been authorised once in this "
                f"process ({self._access_log[0].experiment_id!r}); repeated "
                "evaluation on the locked set would make it a development set"
            )
        missing = [
            f for f in REQUIRED_PREREGISTRATION_FIELDS
            if f not in preregistration or preregistration[f] in (None, "")
        ]
        if missing:
            raise LockedTestAccessError(
                f"pre-registration is incomplete; missing {missing}. Register the "
                "single confirmatory analysis before touching locked-test data."
            )
        declared = str(preregistration.get("experiment_id"))
        if declared != str(experiment_id):
            raise LockedTestAccessError(
                f"preregistration experiment_id {declared!r} does not match the "
                f"requested {experiment_id!r}"
            )
        if script_path is not None:
            script = Path(script_path)
            if not script.is_file():
                raise LockedTestAccessError(f"evaluation script not found: {script}")
            actual = _file_sha256(script)
            expected = str(preregistration.get("script_sha256"))
            if actual != expected:
                raise LockedTestAccessError(
                    "evaluation script hash does not match the pre-registration "
                    f"(expected {expected[:12]}…, got {actual[:12]}…)"
                )
        return LockedTestAccessRecord(
            experiment_id=str(experiment_id),
            reason=str(reason),
            preregistration=dict(preregistration),
            code_revision=str(code_revision or self.code_revision),
            timestamp=_now(),
            n_locked_participants=len(self._locked),
            dataset=self.dataset,
            ledger=str(self.ledger_path) if self.ledger_path else "in-memory",
        )

    def _append_ledger(self, record: LockedTestAccessRecord) -> None:
        if self.ledger_path is None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def _rewrite_ledger(self) -> None:
        if self.ledger_path is None or not self._access_log:
            return
        lines = [
            json.dumps(record.to_dict(), sort_keys=True)
            for record in self._access_log
        ]
        self.ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
