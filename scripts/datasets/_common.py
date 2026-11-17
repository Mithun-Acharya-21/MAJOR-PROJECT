"""Shared helpers for the Phase 4 dataset acquisition scripts.

Deliberately dependency-light: standard library only, so acquisition and
validation never touch the CUDA PyTorch environment.

All scripts are idempotent: existing verified files are reused, downloads
resume with HTTP Range, and nothing is deleted or overwritten silently.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
MANIFESTS = DATA / "manifests"
REPORTS = DATA / "reports"
LOGS = DATA / "logs"

USER_AGENT = "NeuroScreen-phase4-acquisition/1.0 (academic research; contact project owner)"

# Exit codes: 0 ok, 1 validation failure, 2 refused (space/licence/access), 4 missing data
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_REFUSED = 2
EXIT_MISSING = 4


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_logger(name: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(LOGS / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def free_bytes(path: Path = PROJECT_ROOT) -> int:
    return shutil.disk_usage(path).free


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def parse_avdd_labels(text: str) -> tuple[list[str], list[dict]]:
    """Parse an AVDD-TCMI label file.

    The released files are header-less CSV with the columns documented on the
    dataset card: id,label,gender. A header row is tolerated if one ever ships.
    """
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r and any(c.strip() for c in r)]
    if not rows:
        return [], []
    header = rows[0]
    has_header = not all(c.strip().lstrip("-+.").replace(".", "", 1).isdigit() for c in header[:2])
    columns = header if has_header else ["id", "label", "gender"][:len(header)]
    body = rows[1:] if has_header else rows
    records = [{columns[i]: (r[i] if i < len(r) else "") for i in range(len(columns))} for r in body]
    return columns, records


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def count_files(path: Path, suffix: str | None = None) -> int:
    if not path.exists():
        return 0
    it = path.rglob("*")
    return sum(1 for p in it if p.is_file() and (suffix is None or p.suffix.lower() == suffix))


def fail(logger: logging.Logger, message: str, code: int = EXIT_FAIL) -> int:
    logger.error(message)
    return code
