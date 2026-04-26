# Evaluation

Compute quality metrics comparing real and generated image distributions.

## Metric Types

### Distribution-To-Distribution (DTD)

Measure overall similarity between the real and synthetic distributions.

| Metric | Description |
|--------|-------------|
| `fid` | Fréchet Inception Distance — lower is better |
| `inception_score` | Inception Score — higher is better |

### Point-To-Distribution (PTD)

Score each individual synthetic image against the real distribution.

| Metric | Description |
|--------|-------------|
| `mld` | Mahalanobis Distance — lower means more similar to real distribution |

## Usage

```python
from ciagen import evaluate

scores = evaluate(
    real="data/real/train/images/",
    generated="data/generated/",
    metrics=["fid", "mld"],
    feature_extractor="vit",
)
```

## Feature Extractors

Both DTD and PTD metrics operate on deep feature representations, not raw pixels.

| Extractor | Model | Dimensions |
|-----------|-------|------------|
| `vit` | ViT-base (default) | 768 |
| `inception` | Inception v3 | 2048 |

## CLI

```bash
ciagen evaluate \
    --real data/real/train/images/ \
    --generated data/generated/ \
    --metrics fid mld \
    --feature-extractor vit
```
