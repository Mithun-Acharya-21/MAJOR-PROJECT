# Phase 8 — Facial Emotion Transfer Learning and Grad-CAM

## Architecture

```
Pretrained backbone (MobileNetV2 | ResNet-18, ImageNet weights)
        ↓  [warm-up: backbone frozen, head only — N epochs]
        ↓  [fine-tune: backbone unfrozen at lower LR]
Emotion head: Linear(backbone_out → 256) → ReLU → Dropout → Linear(256 → 7)
        ↓
  7 emotion classes (FER-2013)  +  256-d face embedding
```

## Why transfer learning over training from scratch
- ImageNet-pretrained backbones already learn edges, textures, and shape features.
- Freezing the backbone in early epochs prevents catastrophic forgetting.
- A differential learning rate (backbone LR × 0.1) preserves pretrained knowledge
  while adapting the emotion head quickly.
- Results in higher accuracy with far fewer labelled samples than training from
  scratch on a small dataset.

## Grad-CAM

Gradient-weighted Class Activation Mapping highlights the spatial regions of the
input image that most influenced a specific emotion prediction.

**Computing a heatmap:**
1. Register forward/backward hooks on the target convolutional layer.
2. Forward-pass the image.
3. Back-propagate the score for the target class.
4. Global-average-pool the gradients → per-channel weights.
5. Weighted sum of activations → ReLU → upsample → normalize to [0, 1].

**Mandatory framing:**
> *"Grad-CAM highlights the image regions that influenced this emotion prediction.
> This is an emotion-model visualization.
> It is not clinical evidence and does not indicate a mental-health condition."*

This disclaimer must appear on every Grad-CAM visualization in the UI.

## Backbone options

| Backbone | ImageNet top-1 | Params | Recommended use |
|---|---|---|---|
| MobileNetV2 | 71.9 % | 3.4 M | CPU inference, mobile |
| ResNet-18 | 69.8 % | 11.7 M | GPU training, better CAM resolution |

## Training controls

| Control | Value |
|---|---|
| Warm-up freeze epochs | 3 |
| Backbone LR multiplier | 0.1× |
| Scheduler | CosineAnnealingLR |
| Early stopping patience | 4 epochs |
| Augmentation | HorizontalFlip, ColorJitter |
| Normalization | ImageNet mean/std |

## Running

```powershell
cd C:\Users\mh129\majorProject\emotisense-ai
backend\.venv\Scripts\python.exe -m training.vision.vision_transfer `
  --train path\to\fer_train.csv `
  --validation path\to\fer_val.csv `
  --output-dir models\checkpoints\face-transfer-v1 `
  --backbone mobilenet_v2
```

Each CSV row: `path,label` where label is one of the 7 FER emotion strings.

## Face embedding output

The 256-dimensional `embedding` vector from the encoder is the face branch's
contribution to multimodal fusion (Phase 14). It is produced alongside:
- emotion probability distribution (7 classes)
- quality_score from Phase 7
- Grad-CAM heatmap (on request)

## Limitations

- Trained on acted/posed facial expression datasets (FER-2013 etc.).
  Real-world spontaneous expressions may be classified with lower confidence.
- Cultural and demographic biases in training data affect all outputs.
- Occlusion, poor lighting, and non-frontal poses reduce reliability; the
  quality score from Phase 7 propagates this to the fusion gate.
