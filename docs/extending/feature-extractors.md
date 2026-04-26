# Custom Feature Extractors

Feature extractors produce deep feature representations used by quality metrics. They wrap pretrained models.

## Interface

All feature extractors subclass `FeatureExtractor` (inherits from `ABC` and `torch.nn.Module`):

```python
from ciagen.feature_extractors.abc_feature_extractor import FeatureExtractor
import torch


class MyFeatureExtractor(FeatureExtractor):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.model = load_my_model().to(device)

    @classmethod
    def allows_for_gpu(cls) -> bool:
        return True

    def name(self) -> str:
        return "MyFE"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x.to(self.device))
```

## Transform Function

Each feature extractor needs a corresponding image transform:

```python
from torchvision.transforms import Compose, Resize, ToTensor

def my_transform():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
```

## Registration

1. Create `ciagen/feature_extractors/my_extractor.py`

2. Add to `ciagen/feature_extractors/__init__.py`:

```python
from .my_extractor import MyFeatureExtractor, my_transform

AVAILABLE_FEATURE_EXTRACTORS = (..., "my_fe")

def instance_transform(name, **kwargs):
    # ... add my_fe case
    elif name == "my_fe":
        return my_transform()

def available_feature_extractors():
    return {..., "my_fe": MyFeatureExtractor}

def instance_feature_extractor(name, **kwargs):
    # ... add my_fe case
    elif name == "my_fe":
        return MyFeatureExtractor(**kwargs)
```

3. Add to config: `metrics.fe: [vit, my_fe]`
