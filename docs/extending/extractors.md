---
title: Custom Extractors
description: Add custom condition extractors to CIA. Subclass ExtractorABC and register in the extractors module.
keywords: extending, extractors, custom, condition extractor
---

# Custom Extractors

Condition extractors transform a real image into a control signal that guides ControlNet generation.

## Interface

All extractors subclass `ExtractorABC` and implement `extract()`:

```python
from ciagen.extractors.abc_extractor import ExtractorABC
from PIL.Image import Image


class MyExtractor(ExtractorABC):
    name = "MyExtractor"

    def __init__(self, **kwargs):
        pass

    def extract(self, image: Image) -> Image:
        return processed_image
```

## Registration

1. Create `ciagen/extractors/my_extractor.py` with the class above

2. Add to `ciagen/extractors/__init__.py`:

```python
from .my_extractor import MyExtractor

AVAILABLE_EXTRACTORS = (..., "my_extractor")

def instantiate_extractor(control_model: str, **kwargs):
    extractors = {
        # ... existing
        "my_extractor": MyExtractor,
    }
    return extractors[control_model](**kwargs)
```

3. Add a ControlNet config in `ciagen/conf/config.yaml`:

```yaml
model:
    cn:
        - my_controlnet:
            extractor: my_extractor
            sd: my-org/stable-diffusion
            cn: my-org/controlnet-my-extractor
```

## Guidelines

- `extract()` takes a PIL Image and returns a PIL Image
- Output should be the same size as the input
- Accept `**kwargs` in `__init__` for forward compatibility
- Handle edge cases gracefully (e.g., no face detected)
