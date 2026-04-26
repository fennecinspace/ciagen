---
title: CLI Reference
description: Complete reference for the ciagen CLI : generate, evaluate, filter, and caption commands.
keywords: cli, command line, reference
---

# CLI Reference

The `ciagen` command-line tool provides access to all core operations.

## Global

```bash
ciagen --help
```

## `ciagen generate`

Generate synthetic images from real images.

```bash
ciagen generate \
    --source DIR \
    --output DIR \
    --extractor NAME \
    --sd-model MODEL \
    --cn-model MODEL \
    --num N \
    --seed SEED \
    --device DEVICE \
    --prompt TEXT \
    --negative-prompt TEXT \
    --quality N \
    --guidance-scale FLOAT
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--source` | Yes | : | Source images directory |
| `--output` | Yes | : | Output directory |
| `--extractor` | Yes | : | `canny`, `openpose`, `segmentation`, `mediapipe_face` |
| `--sd-model` | Yes | : | Stable Diffusion model ID |
| `--cn-model` | Yes | : | ControlNet model ID |
| `--num` | No | 1 | Images per source image |
| `--seed` | No | 34567 | Random seed |
| `--device` | No | cuda | `cuda` or `cpu` |
| `--prompt` | No | : | Positive prompt |
| `--negative-prompt` | No | : | Negative prompt |
| `--quality` | No | 30 | Inference steps |
| `--guidance-scale` | No | 7.0 | Classifier-free guidance scale |

## `ciagen evaluate`

Compute quality metrics for generated images.

```bash
ciagen evaluate \
    --real DIR \
    --generated DIR \
    --metrics fid mld \
    --feature-extractor vit \
    --batch-size 32 \
    --device cuda
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--real` | Yes | : | Real images directory |
| `--generated` | Yes | : | Generated images directory |
| `--metrics` | No | `fid mld` | Metrics to compute |
| `--feature-extractor` | No | `vit` | `vit` or `inception` |
| `--batch-size` | No | 32 | Batch size |
| `--device` | No | auto | `cuda` or `cpu` |

## `ciagen filter`

Filter generated images by quality score.

```bash
ciagen filter \
    --generated DIR \
    --method METHOD \
    --value VALUE \
    --metric mld \
    --feature-extractor vit
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--generated` | Yes | : | Generated images directory |
| `--method` | Yes | : | `threshold`, `top-k`, `top-p` |
| `--value` | Yes | : | Threshold value |
| `--metric` | No | `mld` | PTD metric name |
| `--feature-extractor` | No | `vit` | Feature extractor name |

## `ciagen caption`

Generate captions for images using a vision model.

```bash
ciagen caption \
    --images DIR \
    --output DIR \
    --engine ENGINE \
    --model MODEL \
    --api-key KEY
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--images` | Yes | : | Images to caption |
| `--output` | Yes | : | Output captions directory |
| `--engine` | No | `openai` | `openai` or `ollama` |
| `--model` | No | `gpt-4o-mini` | Vision model name |
| `--api-key` | No | : | API key (for OpenAI) |
