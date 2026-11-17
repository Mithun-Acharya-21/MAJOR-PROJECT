"""Shared pytest configuration.

The default pytest temporary root (`%LOCALAPPDATA%\\Temp\\pytest-of-<user>`) is
not writable in this environment (`PermissionError: [WinError 5]`), which breaks
every test that uses the `tmp_path` fixture. Point the base temp directory at a
project-local, git-ignored folder instead. An explicit `--basetemp` on the
command line still wins.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config) -> None:
    if getattr(config.option, "basetemp", None) is None:
        config.option.basetemp = str(PROJECT_ROOT / ".pytest_tmp")
