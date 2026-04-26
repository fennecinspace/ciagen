# Custom Metrics

Quality metrics compare real and synthetic image distributions or score individual images.

## Interface

All metrics subclass `QualityMetric`:

```python
from ciagen.metrics.abc_metric import QualityMetric
import torch
from torch.utils.data import DataLoader, Dataset


class MyMetric(QualityMetric):
    @classmethod
    def allows_for_gpu(cls) -> bool:
        return True

    def name(self) -> str:
        return "my_metric"

    def score(
        self,
        real_samples: torch.Tensor | Dataset | DataLoader,
        synthetic_samples: torch.Tensor | Dataset | DataLoader,
        batch_size: int = 32,
    ) -> float | torch.Tensor:
        return result
```

## Registration

1. Create `ciagen/metrics/my_metric.py`

2. Import and use in `ciagen/api/evaluate.py` by adding to `AVAILABLE_DTD_METRICS` or `AVAILABLE_PTD_METRICS`

3. For distance math, add to `ciagen/metrics/distances/`

## DTD vs PTD

- **DTD metrics** return a single scalar (e.g., FID score). Add to `AVAILABLE_DTD_METRICS`.
- **PTD metrics** return a tensor of per-image scores. Add to `AVAILABLE_PTD_METRICS`.

## Using Accumulators

For streaming computation, use the built-in accumulators:

```python
from ciagen.metrics.accumulators import MeanCalculator, CovCalculator

mean_calc = MeanCalculator()
for batch in dataloader:
    mean_calc(batch)
mean = mean_calc.state()  # torch.Tensor
```

Available accumulators:

| Class | Description |
|-------|-------------|
| `MeanCalculator` | Incremental mean computation |
| `CovCalculator` | Incremental covariance matrix computation |
| `KLISCalculator` | KL divergence for Inception Score |
