---
title: Quick Start
description: Get up and running with CIA in 5 minutes. Generate, evaluate, and filter synthetic images.
keywords: quickstart, tutorial, example
---

# Quick Start

## 1. Generate synthetic images

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
)

print(f"Generated {result['total_generated']} images -> {result['output_path']}")
```

## 2. Evaluate quality

```python
from ciagen import evaluate

scores = evaluate(
    real="data/real/train/images/",
    generated="data/generated/",
    metrics=["fid", "mld"],
)

print(f"FID: {scores['dtd']['fid']}")
for path, distance in scores['ptd']['mld'].items():
    print(f"  {path}: {distance:.4f}")
```

## 3. Filter by quality

```python
from ciagen import filter_generated

kept = filter_generated(
    generated="data/generated/",
    method="top-k",
    value=100,
)

for metric_name, fe_data in kept.items():
    for fe, images in fe_data.items():
        print(f"{metric_name}/{fe}: kept {len(images)} images")
```

## CLI alternative

```bash
ciagen generate \
    --source data/real/train/images/ \
    --output data/generated/ \
    --extractor canny \
    --sd-model fennecinspace/sd-v15 \
    --cn-model lllyasviel/sd-controlnet-canny

ciagen evaluate \
    --real data/real/train/images/ \
    --generated data/generated/

ciagen filter \
    --generated data/generated/ \
    --method top-k \
    --value 100
```
