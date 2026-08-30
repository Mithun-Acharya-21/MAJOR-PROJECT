"""
Config Loader — Load and validate YAML configuration files.

Exports:
  load_config — load a YAML config file and return as dict
  merge_configs — merge multiple config dicts (later overrides earlier)
  validate_config — check required keys are present
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Configuration as a dictionary.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    try:
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except ImportError:
        # Fallback: parse basic key: value pairs
        config = _simple_yaml_parse(p)

    return config or {}


def _simple_yaml_parse(path: Path) -> Dict[str, Any]:
    """Minimal YAML parser for simple flat configs (no PyYAML)."""
    config = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Try to parse as number
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() in ("true", "yes"):
                            value = True
                        elif value.lower() in ("false", "no"):
                            value = False
                config[key] = value
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple config dicts. Later configs override earlier ones.

    Args:
        *configs: Config dictionaries to merge.

    Returns:
        Merged configuration dictionary.
    """
    result = {}
    for config in configs:
        result.update(config)
    return result


def validate_config(
    config: Dict[str, Any],
    required_keys: List[str],
    name: str = "config",
) -> None:
    """Validate that required keys are present in the config.

    Args:
        config: Configuration dictionary.
        required_keys: List of keys that must be present.
        name: Name for error messages.

    Raises:
        ValueError: If any required key is missing.
    """
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(
            f"Missing required keys in {name}: {missing}"
        )
