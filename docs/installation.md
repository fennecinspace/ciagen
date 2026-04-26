---
title: Installation
description: Install CIA and its dependencies. Python 3.10+, CUDA GPU recommended.
keywords: install, pip, python, setup
---

# Installation

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)

## Install

```bash
pip install ciagen
```

### Optional dependencies

```bash
pip install ciagen[captioning]    # OpenAI / Ollama captioning
pip install ciagen[training]     # YOLOv8, W&B, training utilities
pip install ciagen[datasets]      # COCO, Flickr30K, FER, MOCS download scripts
pip install ciagen[all]           # Everything
pip install ciagen[dev]           # pytest, ruff
```

## Development

```bash
git clone https://github.com/fennecinspace/ciagen.git
cd ciagen
pip install -e ".[all,dev]"
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
git clone https://github.com/fennecinspace/ciagen.git
cd ciagen
pip install -e ".[docs]"
mkdocs serve
```
