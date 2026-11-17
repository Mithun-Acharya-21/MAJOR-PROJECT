"""
Training session state — real fine-tuning runs.

Runs the real PyTorch training pipeline (see ``backend.app.training``) in a
background thread so the API can return immediately and the frontend can
poll ``/api/training-status`` for live progress, per-epoch history, and
updated benchmark metrics.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from backend.app.baselines import get_baseline_metrics

logger = logging.getLogger("emotiondep.training")

# Mutable training session state (mutated by the background thread)
_session: Dict[str, Any] = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "history": [],
    "last_completed": None,
    "metrics": None,
    "error": None,
    "source": None,
    "modality": None,
    "checkpoint_dir": None,
}


def get_training_status() -> Dict[str, Any]:
    """Return the current training session status for polling."""
    metrics = _session.get("metrics") or get_baseline_metrics()
    return {
        "is_training": _session["is_training"],
        "current_epoch": _session["current_epoch"],
        "total_epochs": _session["total_epochs"],
        "history": list(_session["history"]),
        "last_completed": _session["last_completed"],
        "metrics": metrics,
        "error": _session["error"],
        "source": _session["source"],
        "modality": _session["modality"],
        "checkpoint_dir": _session["checkpoint_dir"],
    }


def start_training(
    config: Dict[str, Any],
    on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Validate config and start real training in a background thread.

    Returns ``{"status": "started", ...}`` immediately, or
    ``{"status": "busy", ...}`` if a run is already in progress.
    Raises ValueError for invalid configuration.
    """
    from backend.app.training import TrainConfig, validate_config

    # Normalize empty-string / missing fields to None  
    cleaned = { 
        k: (v if v not in ("", None) else None)
        for k, v in (config or {}).items()
    }  
    try:
        cfg = TrainConfig(**cleaned)
    except TypeError as e:
        raise ValueError(f"Invalid training configuration: {e}") from e
    validate_config(cfg)

    if _session["is_training"]:
        return {"status": "busy", **get_training_status()}

    _session["is_training"] = True
    _session["current_epoch"] = 0
    _session["total_epochs"] = cfg.epochs
    _session["history"] = []
    _session["last_completed"] = None
    _session["metrics"] = None
    _session["error"] = None
    _session["source"] = None
    _session["modality"] = None
    _session["checkpoint_dir"] = None

    thread = threading.Thread(
        target=_run_training, args=(cfg, on_complete), daemon=True, name="emotiondep-train"
    )
    thread.start()
    return {"status": "started", **get_training_status()}


def _run_training(cfg, on_complete: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Background thread body — load data, train, publish results."""
    from backend.app.training import (
        build_models,
        build_updated_baselines,
        prepare_data,
        resolve_device,
        train,
    )

    try:
        items, modality, source_desc, _label_map = prepare_data(cfg)
        _session["source"] = source_desc
        _session["modality"] = modality

        logger.info("Training on %s samples [%s]", len(items), source_desc)
        device = resolve_device(cfg.device)
        models = build_models(modality, device)

        def _on_epoch(entry: Dict[str, Any]) -> None:
            _session["current_epoch"] = entry["epoch"]
            _session["history"] = list(_session["history"]) + [entry]

        result = train(
            cfg,
            items,
            modality,
            models,
            device,
            on_epoch=_on_epoch,
            log=lambda msg: logger.info(msg),
            source_desc=source_desc,
        )

        updated = build_updated_baselines(result["metrics"], modality, source_desc)
        _session["metrics"] = updated
        _session["history"] = list(result["history"])
        _session["current_epoch"] = result["epochs_run"]
        _session["checkpoint_dir"] = result["checkpoint_dir"]
        _session["last_completed"] = time.time()
        _session["error"] = None

        logger.info(
            "Training complete → %s (val acc %.4f, f1 %.4f)",
            result["checkpoint_dir"],
            result["metrics"]["accuracy"],
            result["metrics"]["f1"],
        )

        if on_complete is not None:
            try:
                on_complete(result)
            except Exception as e:  # never let the status thread die from a hook
                logger.warning("on_complete hook failed: %s", e)

    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        _session["error"] = f"{type(e).__name__}: {e}"
    finally:
        _session["is_training"] = False
