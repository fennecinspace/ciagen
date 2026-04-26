# CIA — Controllable Image Augmentation

CIA is a framework for generating high-quality synthetic image data using **Stable Diffusion + ControlNet**. It provides a complete pipeline from real images to augmented training data with built-in quality evaluation and filtering.

## Why CIA?

- **Controlled generation**: Use edge maps, pose estimation, segmentation, or face meshes as control signals
- **Quality-aware**: Automatically evaluate and filter synthetic images using FID, Inception Score, and Mahalanobis distance
- **Pipeline-ready**: Generate → Evaluate → Filter → Mix → Train in one workflow
- **Multiple interfaces**: Python API, CLI, or Hydra config — pick what fits your workflow

## Quick Links

| What | Where |
|------|-------|
| Install | [Installation](installation.md) |
| 5-minute setup | [Quick Start](quickstart.md) |
| How it works | [Pipeline Overview](pipeline.md) |
| Python API | [`generate()`](api/generate.md), [`evaluate()`](api/evaluate.md), [`filter_generated()`](api/filter.md), [`caption()`](api/caption.md) |
| CLI | [CLI Reference](cli.md) |
| Contribute | [Contributing](contributing.md) |
