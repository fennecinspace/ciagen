# `evaluate()`

Evaluate the quality of generated images against real images.

```python
from ciagen import evaluate

scores = evaluate(
    real="data/real/train/images/",
    generated="data/generated/",
    metrics=["fid", "mld"],
    feature_extractor="vit",
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `real` | `str \| Path` | **required** | Directory containing real images |
| `generated` | `str \| Path` | **required** | Directory containing generated images |
| `metrics` | `list[str] \| None` | `["fid", "mld"]` | Metrics to compute: `fid`, `inception_score`, `mld` |
| `feature_extractor` | `str` | `"vit"` | Feature extractor: `vit` or `inception` |
| `batch_size` | `int` | `32` | Batch size for computation |
| `limit_size_real` | `int` | `2000` | Max real images to use |
| `limit_size_syn` | `int` | `2000` | Max synthetic images to use |
| `image_formats` | `list[str] \| None` | `["png","jpg","jpeg"]` | Supported formats |
| `device` | `str \| None` | auto | `"cuda"` or `"cpu"` (auto-detected if None) |

## Returns

`dict` with two optional keys depending on which metrics were computed:

### `dtd` key (Distribution-To-Distribution)

```python
{
    "fid": {
        "vit": 45.23
    },
    "inception_score": {
        "vit": 12.5
    }
}
```

### `ptd` key (Point-To-Distribution)

```python
{
    "mld": {
        "vit": {
            "/path/to/image_1.png": 2.34,
            "/path/to/image_2.png": 3.56,
        }
    }
}
```
