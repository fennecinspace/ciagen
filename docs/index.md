---
title: Home
description: Generate high-quality synthetic image data using Stable Diffusion + ControlNet with built-in quality evaluation and filtering.
---

# CIA: Controllable Image Augmentation

**Generate high-fidelity synthetic images** using Stable Diffusion + ControlNet. A complete pipeline for data augmentation with built-in quality metrics.

[Get Started](installation.md#installation){ .md-button .md-button--primary }
[View on GitHub](https://github.com/user/synthetic-augmentation){ .md-button .md-button--icon }

---

## Why CIA?

| Feature | Description |
|---------|-------------|
| **Controlled Generation** | Use edge maps, pose estimation, segmentation, or face meshes as control signals |
| **Quality-Aware** | Evaluate with FID, Inception Score, and Mahalanobis distance |
| **Smart Filtering** | Select best samples using PTD scores (top-k, top-p, threshold) |
| **Multiple Interfaces** | Python API, CLI, or Hydra config, pick what fits your workflow |

## The Pipeline

```
Real Images → Extract Condition → SD + ControlNet → Generated Images
     ↓                                              ↓
  Features ←── Quality Metrics (FID, IS, MLD) ──→ Filtered Set
                                                    ↓
                                              Mixed Dataset → Train
```

## Quick Example

```python
from ciagen import generate, evaluate, filter_generated

# Generate synthetic images
result = generate(
    source="data/real/train/images/",
    output="data/generated/",
    extractor="canny",
    sd_model="fennecinspace/sd-v15",
    cn_model="lllyasviel/sd-controlnet-canny",
)

# Evaluate quality
scores = evaluate(
    real="data/real/train/images/",
    generated="data/generated/",
    metrics=["fid", "mld"],
)

# Filter best images
kept = filter_generated(
    generated="data/generated/",
    method="top-k",
    value=1000,
)
```

## Installation

```bash
pip install -e .
```

## Supported Extractors

| Extractor | Use Case |
|-----------|----------|
| `canny` | Edge detection |
| `openpose` | Human pose estimation |
| `segmentation` | Instance segmentation |
| `mediapipe_face` | Face mesh landmarks |

## Supported Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fid` | DTD | Fréchet Inception Distance |
| `inception_score` | DTD | Inception Score |
| `mld` | PTD | Mahalanobis Distance |

---

**Questions?** Open an [issue](https://github.com/user/synthetic-augmentation/issues) on GitHub.
