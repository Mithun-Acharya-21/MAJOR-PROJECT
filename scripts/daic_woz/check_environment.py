import sys
import os
import importlib

pkgs = [
    'torch', 'torchvision', 'transformers', 'sklearn', 'numpy', 'pandas',
    'librosa', 'soundfile', 'cv2', 'mediapipe', 'scipy', 'matplotlib',
    'seaborn', 'shap', 'lime', 'torchaudio', 'captum', 'accelerate',
    'datasets', 'joblib', 'fastapi', 'uvicorn', 'pydantic'
]

results = {}
for p in pkgs:
    try:
        mod = importlib.import_module(p)
        results[p] = getattr(mod, '__version__', 'Installed')
    except ImportError:
        results[p] = 'NOT_INSTALLED'
    except Exception as e:
        results[p] = f'ERROR: {e}'

import torch
cuda_avail = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_avail else 'N/A'
vram = torch.cuda.get_device_properties(0).total_memory / (1024**2) if cuda_avail else 0

import psutil
ram_total = psutil.virtual_memory().total / (1024**3)
ram_avail = psutil.virtual_memory().available / (1024**3)
disk_free = psutil.disk_usage('.').free / (1024**3)

print('=== PACKAGES ===')
for k, v in results.items():
    print(f'{k}: {v}')
print('=== SYSTEM ===')
print(f'CUDA: {cuda_avail}')
print(f'GPU: {gpu_name}')
print(f'VRAM_MB: {vram:.1f}')
print(f'RAM_TOTAL_GB: {ram_total:.2f}')
print(f'RAM_AVAIL_GB: {ram_avail:.2f}')
print(f'DISK_FREE_GB: {disk_free:.2f}')
