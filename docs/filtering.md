# Filtering

Select the best synthetic images based on point-to-distribution quality scores.

## Methods

| Method | Description |
|--------|-------------|
| `top-k` | Keep the k images with smallest distances |
| `top-p` | Keep the top proportion (0 ≤ p ≤ 1) of images |
| `threshold` | Keep images with distance ≤ value |

## Usage

```python
from ciagen import filter_generated

kept = filter_generated(
    generated="data/generated/",
    method="top-k",
    value=100,
)
```

## CLI

```bash
ciagen filter \
    --generated data/generated/ \
    --method top-k \
    --value 100
```

## Requirements

The generated directory must contain a `metadata.yaml` file with PTD scores (created by `evaluate()`). If `ptd_scores` is provided directly, the metadata file is not needed.
