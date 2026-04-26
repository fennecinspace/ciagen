---
title: Configuration
description: Configure CIA via Hydra YAML config. ControlNet models, extractors, metrics, prompts, and data paths.
keywords: configuration, hydra, yaml, config, setup
---

# Configuration

CIA uses [Hydra](https://hydra.cc/) for configuration management. The main config file is `ciagen/conf/config.yaml`.

## Using Hydra

Override any config value from the command line:

```bash
python run.py task=gen model.cn_use=lllyasviel_canny prompt.base="a person"
```

You can also edit `config.yaml` directly.

## Config Reference

### `task`

Which pipeline step to run.

| Value | Description |
|-------|-------------|
| `help` | Print help message |
| `prepare_data` | Download and prepare a dataset |
| `auto_caption` | Generate captions for images |
| `gen` | Generate synthetic images |
| `dtd` | Compute distribution-to-distribution metrics |
| `ptd` | Compute point-to-distribution metrics |
| `filtering` | Filter generated images by quality |
| `mix` | Create mixed real+synthetic dataset |
| `train` | Train downstream model |

### `data`

```yaml
data:
    base: "coco"
    image_formats: ["png", "jpeg", "jpg"]
    limit_size_real: 2000
    limit_size_syn: 2000
    batch_size: 32
    datatype: "image"
```

### `prompt`

```yaml
prompt:
    use_captions: 1
    caption_per_line: 0
    extra_empty_caption: 0
    modify_captions: 0
    template: vocabulary
    generation_size: 10
    base: ["prompt1", "prompt2"]
    quality: "best quality, ..."
    modifier: "Happy man smiling"
    negative: [...]
    negative_simple: "monochrome, lowres, ..."
```

### `model`

```yaml
model:
    sd_steps:
        - fast: 5
        - medium: 10
        - slow: 20
    cn_use: controlnet_segmentation
    cn:
        - controlnet_segmentation:
            extractor: segmentation
            sd: fennecinspace/sd-v15
            cn: lllyasviel/sd-controlnet-seg
        - lllyasviel_canny:
            extractor: canny
            sd: fennecinspace/sd-v15
            cn: lllyasviel/sd-controlnet-canny
    seed: [34567]
    device: cuda
```

### `metrics`

```yaml
metrics:
    fe:
        - vit
    dtd:
        - fid
        - inception_score
    ptd:
        - mld
```

### `filtering`

```yaml
filtering:
    type: "top-k"
    value: 1000
```

### `auto_captioner`

```yaml
auto_captioner:
    service:
        engine: openai
        model: gpt-4o-mini
        api_key: YOUR-KEY
    custom_images_path: null
    custom_captions_path: null
```

## Adding Custom ControlNet Models

Add a new entry under `model.cn` in `config.yaml`:

```yaml
model:
    cn:
        - my_custom_model:
            extractor: canny
            sd: my-org/my-sd-model
            cn: my-org/my-cn-model
```

Then use it: `python run.py task=gen model.cn_use=my_custom_model`
