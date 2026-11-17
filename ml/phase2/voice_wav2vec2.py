"""
Experiment V2 — frozen Wav2Vec2 embeddings + lightweight classification head.

Encoder choice (justified, not defaulted):
    The EATD audio is Mandarin interview speech. ``facebook/wav2vec2-base-960h``
    is English-ASR pretrained; ``TencentGameMate/chinese-wav2vec2-base`` is the
    same architecture pretrained on Mandarin audio. The Chinese checkpoint is
    therefore the primary encoder; the English one is available via
    ``--encoder`` for comparison. This is the language check the project
    requires before swapping encoders.

Processing (fits a 6 GB RTX 3050):
    * Audio is read at 16 kHz (EATD native), stereo averaged to mono.
    * Long recordings are cut into non-overlapping 8 s chunks (max 12 chunks
      per session ≈ 96 s) — no sample exceeds the encoder's effective context.
    * The encoder is frozen; each chunk's frame states are pooled with
      mean+std (1536-d), then chunk vectors are averaged into a session vector.
    * Embeddings are cached to disk so the head can be retrained exactly.
    * The head is the same small MLP used in T2a, trained on GPU with AMP and
      selected on validation participant ROC-AUC.

No encoder fine-tuning happens here (Phase 2 scope: frozen baseline first).

Exports:
    ENCODERS, chunk_audio, embed_audio, extract_corpus, run_voice_wav2vec2
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ml.phase2.dataset import load_sessions, resolve_path
from ml.phase2.evaluation import (
    evaluate_predictions,
    experiment_metadata,
    write_json,
)
from ml.phase2.text_transformer import head_probabilities, train_head
from ml.phase2.voice_egemaps import _load_audio

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_ID = "eatd_voice_wav2vec2_v1"
MODEL_VERSION = "voice-eatd-wav2vec2-2.0.0"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "experiments" / EXPERIMENT_ID
RESULT_PATH = PROJECT_ROOT / "results" / "phase2" / "voice" / f"{EXPERIMENT_ID}.json"

ENCODERS = {
    "chinese": "TencentGameMate/chinese-wav2vec2-base",
    "english": "facebook/wav2vec2-base-960h",
}
DEFAULT_ENCODER = ENCODERS["chinese"]
CHUNK_SEC = 8.0
MAX_CHUNKS = 12


def chunk_audio(y: np.ndarray, sr: int, chunk_sec: float = CHUNK_SEC,
                max_chunks: int = MAX_CHUNKS) -> List[np.ndarray]:
    """Non-overlapping chunks; the tail chunk is kept when >= 1 s."""
    size = int(chunk_sec * sr)
    chunks = [y[i : i + size] for i in range(0, len(y), size)]
    chunks = [c for c in chunks if len(c) >= sr]  # drop <1 s tail
    if not chunks:
        chunks = [y]
    return [np.asarray(c, dtype=np.float32) for c in chunks[:max_chunks]]


@torch.no_grad()
def embed_audio(
    rows: Sequence[dict],
    encoder_name: str = DEFAULT_ENCODER,
    chunk_sec: float = CHUNK_SEC,
    max_chunks: int = MAX_CHUNKS,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> Tuple[np.ndarray, List[dict], Dict[str, object]]:
    """Session-level frozen Wav2Vec2 embeddings (mean+std pooled chunks)."""
    from transformers import AutoModel

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(encoder_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    amp_enabled = use_amp and device.type == "cuda"
    vectors: List[np.ndarray] = []
    kept: List[dict] = []
    excluded: List[dict] = []
    t0 = time.time()
    n_chunks_total = 0

    # Build a flat chunk plan so batches are filled across sessions.
    plan: List[Tuple[dict, List[np.ndarray]]] = []
    for row in rows:
        if not row.get("audio_usable", True):
            excluded.append(
                {
                    "participant_key": row["participant_key"],
                    "topic": row["topic"],
                    "reason": "marked_audio_unusable_in_phase1_audit",
                    "audio_duration_sec": row["audio_duration_sec"],
                }
            )
            continue
        try:
            y, sr, _meta = _load_audio(row["audio_path"])
        except Exception as e:
            excluded.append(
                {
                    "participant_key": row["participant_key"],
                    "topic": row["topic"],
                    "reason": f"audio_read_error: {type(e).__name__}",
                }
            )
            continue
        chunks = chunk_audio(y, sr, chunk_sec=chunk_sec, max_chunks=max_chunks)
        plan.append((row, chunks))
        n_chunks_total += len(chunks)

    flat: List[np.ndarray] = []
    owner: List[int] = []
    for i, (_row, chunks) in enumerate(plan):
        for c in chunks:
            flat.append(c)
            owner.append(i)

    pooled: List[np.ndarray] = []
    for start in range(0, len(flat), batch_size):
        batch = flat[start : start + batch_size]
        max_len = max(len(c) for c in batch)
        arr = np.zeros((len(batch), max_len), dtype=np.float32)
        for i, c in enumerate(batch):
            arr[i, : len(c)] = c
        x = torch.from_numpy(arr).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            hidden = model(x).last_hidden_state.float()  # (B, T, D)
        mean = hidden.mean(dim=1)
        std = hidden.std(dim=1)
        pooled.append(torch.cat([mean, std], dim=1).cpu().numpy().astype(np.float32))

    chunk_vectors = np.concatenate(pooled, axis=0) if pooled else np.zeros((0, 0))
    for i, (row, chunks) in enumerate(plan):
        sel = [j for j, o in enumerate(owner) if o == i]
        session_vec = chunk_vectors[sel].mean(axis=0) if sel else np.zeros(
            chunk_vectors.shape[1], dtype=np.float32
        )
        vectors.append(session_vec)
        kept.append(
            {
                "participant_key": row["participant_key"],
                "participant_label": row["participant_label"],
                "topic": row["topic"],
                "split": row["split"],
                "audio_path": row["audio_path"],
                "n_chunks": len(chunks),
            }
        )

    elapsed = time.time() - t0
    stats = {
        "encoder": encoder_name,
        "encoder_frozen": True,
        "pooling": "mean+std over frames per chunk, mean over chunks",
        "chunk_sec": chunk_sec,
        "max_chunks": max_chunks,
        "batch_size": batch_size,
        "device": str(device),
        "amp": bool(amp_enabled),
        "sessions_requested": len(rows),
        "sessions_embedded": len(kept),
        "sessions_excluded": len(excluded),
        "excluded_sessions": excluded,
        "chunks_processed": n_chunks_total,
        "embed_seconds": round(elapsed, 2),
        "seconds_per_session": round(elapsed / max(len(kept), 1), 3),
        "embedding_dim": int(chunk_vectors.shape[1]) if len(chunk_vectors) else None,
        "peak_gpu_memory_gb": (
            round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None
        ),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    X = np.vstack(vectors) if vectors else np.zeros((0, 0))
    return X, kept, stats


def _cache_stem(encoder_name: str, chunk_sec: float, max_chunks: int) -> str:
    tag = encoder_name.replace("/", "__").replace("-", "_")
    return f"wav2vec2_{tag}_chunk{int(chunk_sec)}s_max{max_chunks}"


def run_voice_wav2vec2(
    seed: int = 42,
    encoder_name: str = DEFAULT_ENCODER,
    chunk_sec: float = CHUNK_SEC,
    max_chunks: int = MAX_CHUNKS,
    batch_size: int = 4,
    head_epochs: int = 200,
    head_lr: float = 1e-4,
    n_boot: int = 2000,
    write: bool = True,
    reuse_cache: bool = True,
) -> Dict[str, object]:
    """V2 experiment: frozen Wav2Vec2 session embeddings + trained head."""
    t_start = time.time()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cache = ARTIFACT_DIR / f"{_cache_stem(encoder_name, chunk_sec, max_chunks)}.npz"

    rows = (
        load_sessions("train", modality="audio", require_usable_audio=False)
        + load_sessions("validation", modality="audio", require_usable_audio=False)
        + load_sessions("test", modality="audio", require_usable_audio=False)
    )

    if reuse_cache and cache.exists():
        import json

        data = np.load(cache, allow_pickle=False)
        meta = json.loads(Path(str(cache) + ".json").read_text(encoding="utf-8"))
        X, kept, stats = data["X"], meta["rows"], {**meta["stats"], "cached": True}
    else:
        X, kept, stats = embed_audio(
            rows,
            encoder_name=encoder_name,
            chunk_sec=chunk_sec,
            max_chunks=max_chunks,
            batch_size=batch_size,
        )
        if write:
            np.savez_compressed(cache, X=X)
            write_json(Path(str(cache) + ".json"), {"rows": kept, "stats": stats})

    idx = {s: [i for i, r in enumerate(kept) if r["split"] == s] for s in
           ("train", "validation", "test")}
    for s, ids in idx.items():
        if not ids:
            raise RuntimeError(f"no embedded sessions in split {s!r}")

    train_rows = [kept[i] for i in idx["train"]]
    val_rows = [kept[i] for i in idx["validation"]]
    test_rows = [kept[i] for i in idx["test"]]

    model, history, head_info = train_head(
        X[idx["train"]],
        [r["participant_label"] for r in train_rows],
        X[idx["validation"]],
        [r["participant_label"] for r in val_rows],
        [r["participant_key"] for r in train_rows],
        [r["participant_key"] for r in val_rows],
        seed=seed,
        epochs=head_epochs,
        lr=head_lr,
    )
    val_probs = head_probabilities(model, X[idx["validation"]])
    t_infer = time.time()
    test_probs = head_probabilities(model, X[idx["test"]])
    inference_seconds = round(time.time() - t_infer, 4)

    val_eval = evaluate_predictions(val_rows, val_probs, n_boot=n_boot, seed=seed)
    test_eval = evaluate_predictions(test_rows, test_probs, n_boot=n_boot, seed=seed)

    checkpoint_paths: List[Path] = []
    if write:
        torch.save(model.state_dict(), ARTIFACT_DIR / "head.pt")
        checkpoint_paths = [ARTIFACT_DIR / "head.pt"]
        write_json(
            ARTIFACT_DIR / "test_predictions.json",
            {
                "participant_key": [r["participant_key"] for r in test_rows],
                "topic": [r["topic"] for r in test_rows],
                "label": [r["participant_label"] for r in test_rows],
                "probability": [round(float(p), 6) for p in test_probs],
                "model_version": MODEL_VERSION,
            },
        )

    config = {
        "encoder_name": encoder_name,
        "encoder_frozen": True,
        "chunk_sec": chunk_sec,
        "max_chunks": max_chunks,
        "pooling": "mean+std over encoder frames per chunk, then mean over chunks",
        "embedding_dim": int(X.shape[1]),
        "head": {"hidden": head_info["hidden"], "dropout": head_info["dropout"]},
        "head_training": head_info,
        "session_counts": {s: len(ids) for s, ids in idx.items()},
    }
    metadata = experiment_metadata(
        experiment_id=EXPERIMENT_ID,
        model_version=MODEL_VERSION,
        modality="voice",
        config=config,
        training={
            "seed": seed,
            "embedding": stats,
            "history": history,
            "fit_seconds": round(time.time() - t_start, 2),
            "inference_seconds_test": inference_seconds,
            "refit_on_train_plus_validation": False,
            "selection_metric": "validation participant-level ROC-AUC",
        },
        validation_metrics=val_eval,
        test_metrics=test_eval,
        checkpoint_paths=checkpoint_paths,
        notes=[
            "Encoder pretrained on Mandarin (TencentGameMate/chinese-wav2vec2-base), matching the dataset language; only its frozen representations are used.",
            "Frozen encoder: no wav2vec2 weights were updated.",
            "Sessions with n_chunks > 0 only; exclusions are listed in training.embedding.excluded_sessions.",
        ],
    )
    if write:
        write_json(ARTIFACT_DIR / "metadata.json", metadata)
        write_json(RESULT_PATH, metadata)
    return metadata
