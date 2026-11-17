# Verified Environment (Phase 0)

Recorded on 2026-11-13 from this machine. `requirements.txt` pins these versions.

## Which interpreter (read this first)

This repository has **two** Python installations with a `torch` package, and only
one of them is the verified environment:

| Interpreter | torch | CUDA | test suite |
|---|---|---|---|
| `.venv/Scripts/python` | `2.13.0+cu130` | available (`True`) | **158 passed** |
| system `python` (`AppData/Roaming/Python/Python314`) | `2.13.0+cpu` | unavailable (`False`) | 14 failed (env only) |

Always use the virtualenv:

```bash
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python -m uvicorn backend.app.main:app --reload
```

The system interpreter is CPU-only **and** carries `huggingface_hub==2.0.0`,
which `transformers==5.15.1` rejects at import time
(`huggingface-hub>=1.5.0,<2.0 is required ... found huggingface-hub==2.0.0`).
Running the suite with it fails 14 tests that pass under the virtualenv — those
failures are an environment mismatch, not a code regression. `requirements.txt`
now pins `huggingface_hub==1.28.0` to prevent this.

## Verified versions

| Component | Verified value |
|---|---|
| OS | Windows (bash/Git Bash) |
| Interpreter | `.venv/Scripts/python` — Python 3.14.0 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime | 13.0 |
| torchvision | 0.28.0+cu130 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Reported VRAM | 6144 MiB (`nvidia-smi`), 6.44 GB via torch |
| NVIDIA driver | 581.86 |
| `torch.cuda.is_available()` | `True` |
| transformers | 5.15.1 |
| huggingface_hub | 1.28.0 |
| numpy | 2.5.2 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.0 |
| pandas | 3.0.5 |
| pillow | 12.3.0 |
| mediapipe | 1.0.1 |
| opencv-python | 5.0.0.93 |
| faster-whisper | 1.2.1 |
| librosa | 0.11.0 |
| av / soundfile | 18.1.0 / 0.14.0 |
| lime / shap | 0.2.0.1 / 0.52.0 |
| opensmile | 2.6.0 (Phase 2 eGeMAPS extraction) |
| fastapi / uvicorn | 0.141.1 / 0.52.4 |
| pytest | 8.4.2 |

## Verify CUDA is still the GPU build

```bash
.venv/Scripts/python -c "import torch; print(torch.__version__, torch.version.cuda); \
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); \
print(round(torch.cuda.get_device_properties(0).total_memory/1e9, 2), 'GB')"
```

Expected: `2.13.0+cu130 13.0`, `True`, `NVIDIA GeForce RTX 3050 6GB Laptop GPU`,
`6.44 GB`.

## Install / repair

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

The requirements file carries `--extra-index-url https://download.pytorch.org/whl/cu130`
so the CUDA wheels are preserved. Do **not** install a CPU-only torch build over
this environment.

## Known environment notes

- Python 3.14 is ahead of parts of the ML ecosystem; a `torch.jit` deprecation
  warning (`torch.jit.script is not supported in Python 3.14+`) appears during
  tests but does not affect training or inference.
- `openai-whisper` and `facenet-pytorch` are installed in the virtualenv but are
  not imported anywhere in the codebase (`faster-whisper` is used instead).
  They are intentionally not listed in `requirements.txt`.
- `opencv-contrib-python`, `opencv-python-headless` and `scikit-image` are also
  present in the virtualenv without being imported; `cv2` comes from
  `opencv-python`, which is the pinned one.
- `.DS_Store` files exist under `data/eatd/`; the manifest ignores them (only
  numeric participant directories are read).
- `fastapi.testclient` emits a Starlette deprecation warning about `httpx` vs
  `httpx2`. It is a warning only; the suite passes.
- `opensmile==2.6.0` was added in Phase 2 for eGeMAPS extraction. It pulls in
  `audinterface`, `audformat`, `audresample`, `audiofile`, `audeer`, `audmath`,
  `audobject`, `iso639-lang`, `iso3166` and `oyaml`; none of them are imported
  directly by this repository.
- Hugging Face cache (`~/.cache/huggingface`) holds the encoders used in Phase 2:
  `distilbert-base-multilingual-cased`,
  `TencentGameMate/chinese-wav2vec2-base` and
  `facebook/wav2vec2-base-960h`. They are downloaded on first use, not vendored.
  On Windows without Developer Mode, set `HF_HUB_DISABLE_SYMLINKS=1` if the
  cache fails with `WinError 1314`.
