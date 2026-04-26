# `filter_generated()`

Filter generated images based on point-to-distribution quality scores.

```python
from ciagen import filter_generated

kept = filter_generated(
    generated="data/generated/",
    method="top-k",
    value=100,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generated` | `str \| Path` | **required** | Directory containing generated images and `metadata.yaml` |
| `ptd_scores` | `dict \| None` | `None` | Pre-computed PTD scores (reads from `metadata.yaml` if None) |
| `method` | `str` | `"top-k"` | Filtering method: `threshold`, `top-k`, `top-p` |
| `value` | `float` | `1000` | Filtering threshold (interpretation depends on `method`) |
| `metric` | `str` | `"mld"` | PTD metric name to filter on |
| `feature_extractor` | `str` | `"vit"` | Feature extractor name whose scores to use |

## Filtering Methods

| Method | `value` meaning |
|--------|-----------------|
| `top-k` | Keep the k images with smallest distances |
| `top-p` | Keep the top proportion (0 ≤ p ≤ 1) of images |
| `threshold` | Keep images with distance ≤ value |

## Returns

`dict` mapping metric names to feature extractor results:

```python
{
    "mld": {
        "vit": {
            "/path/to/best_image.png": 0.45,
            "/path/to/second_best.png": 0.52,
        }
    }
}
```

## Raises

- `FileNotFoundError` — if no `metadata.yaml` exists in the generated directory and `ptd_scores` is None
- `ValueError` — if `method` is not one of the valid options or `value` is out of range
