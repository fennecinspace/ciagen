---
title: generate()
description: Python API reference for ciagen.generate() : generate synthetic images from real images using Stable Diffusion + ControlNet.
keywords: api, generate, python
---

# `generate()`

Generate synthetic images from real images using Stable Diffusion + ControlNet.

```python
from ciagen import generate

result = generate(
    source="data/real/train/images/",
    output="data/generated/",
    extractor="canny",
    sd_model="fennecinspace/sd-v15",
    cn_model="lllyasviel/sd-controlnet-canny",
    num_per_image=3,
    prompt="a person walking in a park",
    seed=42,
    device="cuda",
)
```

## Parameters

!!! note
    All parameters except `source`, `output`, `extractor`, `sd_model`, and `cn_model` are optional.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str \| Path` | **required** | Directory containing real source images |
| `output` | `str \| Path` | **required** | Directory to save generated images |
| `extractor` | `str` | **required** | Condition extractor: `canny`, `openpose`, `segmentation`, `mediapipe_face` |
| `sd_model` | `str` | **required** | HuggingFace model ID for Stable Diffusion |
| `cn_model` | `str` | **required** | HuggingFace model ID for ControlNet |
| `num_per_image` | `int` | `1` | Number of synthetic images per real image |
| `seed` | `int \| list[int]` | `34567` | Random seed for reproducibility |
| `device` | `str` | `"cuda"` | `"cuda"` or `"cpu"` |
| `prompt` | `str \| list[str] \| None` | `None` | Positive prompt(s) |
| `negative_prompt` | `str \| None` | `None` | Negative prompt |
| `quality` | `int` | `30` | Number of inference steps |
| `guidance_scale` | `float` | `7.0` | Classifier-free guidance scale |
| `use_captions` | `bool` | `False` | Use per-image caption files as prompts |
| `captions_dir` | `str \| None` | `None` | Directory with caption `.txt` files |
| `modify_captions` | `bool` | `False` | Vary captions using vocabulary substitution |
| `vocabulary_template` | `str \| None` | `None` | Vocabulary config name for caption modification |
| `generation_size` | `int` | `10` | Number of prompt variations to generate |
| `cn_extra_settings` | `dict \| None` | `None` | Extra kwargs for ControlNet loading |
| `image_formats` | `list[str] \| None` | `["png","jpg","jpeg"]` | Supported image formats |

## Returns

`dict` with:

| Key | Type | Description |
|-----|------|-------------|
| `total_generated` | `int` | Number of images successfully generated |
| `output_path` | `str` | Absolute path to the output directory |
| `source_images` | `int` | Number of source images processed |
| `errors` | `list[tuple]` | `(image_path, error_message)` for failures |
