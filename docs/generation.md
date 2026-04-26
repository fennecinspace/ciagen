---
title: Generation
description: Generate synthetic images from real images using Stable Diffusion + ControlNet with condition extractors (canny, openpose, segmentation, mediapipe_face).
keywords: generation, synthetic images, stable diffusion, controlnet, canny, openpose, segmentation
---

# Generation

Generate synthetic images from real images using Stable Diffusion + ControlNet.

## Condition Extractors

Each extractor transforms a real image into a control signal that guides generation:

| Extractor | Description | Requires |
|-----------|-------------|----------|
| `canny` | Canny edge detection | : |
| `openpose` | Human pose estimation | : |
| `segmentation` | YOLOv8-seg instance segmentation | `ultralytics` |
| `mediapipe_face` | Face mesh landmarks | `mediapipe` |

## Usage

### Python API

```python
from ciagen import generate

result = generate(
    source="data/real/train/images/",
    output="data/generated/",
    extractor="canny",
    sd_model="fennecinspace/sd-v15",
    cn_model="lllyasviel/sd-controlnet-canny",
    num_per_image=3,
    prompt="a person walking",
    seed=42,
    quality=30,
    guidance_scale=7.0,
)
```

### CLI

```bash
ciagen generate \
    --source data/real/train/images/ \
    --output data/generated/ \
    --extractor canny \
    --sd-model fennecinspace/sd-v15 \
    --cn-model lllyasviel/sd-controlnet-canny \
    --num 3 \
    --prompt "a person walking"
```

### Hydra

```bash
python run.py task=gen model.cn_use=lllyasviel_canny prompt.base='["a person walking"]'
```

## Prompts

Three prompt strategies:

1. **Fixed prompt** : pass `prompt="a person in a park"`
2. **Caption-based** : set `use_captions=True` to read per-image `.txt` caption files
3. **Vocabulary-modified** : set `modify_captions=True` with a vocabulary template to generate prompt variations

## Output

Generated images are saved as `{original_name}_{index}.png` in the output directory. A `metadata.yaml` file is created with the generation configuration.
