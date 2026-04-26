# Installation

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)

## Install

```bash
pip install -e .
```

### Optional dependencies

```bash
pip install -e ".[captioning]"    # OpenAI / Ollama captioning
pip install -e ".[training]"      # YOLOv8, W&B, training utilities
pip install -e ".[datasets]"      # COCO, Flickr30K, FER, MOCS download scripts
pip install -e ".[all]"           # Everything
pip install -e ".[dev]"           # pytest, ruff
```

## Verify

```python
import ciagen
print(ciagen.__all__)
# ['generate', 'evaluate', 'filter_generated', 'caption']
```

## Documentation

To build and preview the docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```
