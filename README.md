# CIA: Controllable Image Augmentation

<!-- Badges -->
[![GitHub Stars](https://img.shields.io/github/stars/fennecinspace/ciagen?style=social)](https://github.com/fennecinspace/ciagen)
[![License](https://img.shields.io/github/license/fennecinspace/ciagen)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/ciagen)](https://pypi.org/project/ciagen/)
[![Docs](https://img.shields.io/readthedocs/ciagen)](https://ciagen.readthedocs.io/en/latest/)
[![arXiv](https://img.shields.io/badge/arXiv-2411.16128-blue)](https://arxiv.org/abs/2411.16128)

<!-- Add tests later -->
<!--[![Tests](https://img.shields.io/github/actions/workflow_status/multitel-ai/CIA/tests.yml?label=tests)](https://github.com/multitel-ai/CIA/actions) -->

**CIA** is a Python library for synthetic data augmentation using Stable Diffusion + ControlNet. Generate high-quality synthetic images from real seed images, evaluate their quality, and use them to improve downstream ML models.

## Features

- **Synthetic image generation** using Stable Diffusion controlled by Canny edges, OpenPose, Segmentation, or MediaPipe face features
- **Quality metrics** -- Frechet Inception Distance (FID), Inception Score (IS), Mahalanobis distance
- **Quality-based filtering** -- keep only the best synthetic images via top-k, top-p, or threshold filtering
- **Auto-captioning** -- generate image captions using OpenAI or Ollama vision models
- **Multiple interfaces** -- Python API, CLI, and Hydra config

## Installation

```bash
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[captioning]"    # OpenAI/Ollama auto-captioning
pip install -e ".[training]"      # YOLO/classifier training
pip install -e ".[datasets]"      # COCO, Flickr30K, FER, MOCS datasets
pip install -e ".[all]"           # Everything
```

### Docker

```bash
./run_and_build_docker_file.sh nvidia
docker exec -it ciagen zsh
```

## Quick Start

### Python API

```python
from ciagen import generate, evaluate, filter_generated

# Generate synthetic images
result = generate(
    source="data/real/train/images/",
    output="data/generated/",
    extractor="canny",
    sd_model="fennecinspace/sd-v15",
    cn_model="lllyasviel/sd-controlnet-canny",
    num_per_image=3,
    prompt="a person walking in a park",
    seed=42,
    device="cuda",
)
print(f"Generated {result['total_generated']} images")

# Evaluate quality
scores = evaluate(
    real="data/real/train/images/",
    generated="data/generated/",
    metrics=["fid", "mld"],
    feature_extractor="vit",
)
print(f"FID: {scores['dtd']['fid']}")

# Filter to keep the best images
kept = filter_generated(
    generated="data/generated/",
    method="top-k",
    value=100,
)
```

### CLI

```bash
# Generate images
ciagen generate \
    --source data/real/train/images/ \
    --output data/generated/ \
    --extractor canny \
    --sd-model fennecinspace/sd-v15 \
    --cn-model lllyasviel/sd-controlnet-canny \
    --num 3 \
    --prompt "a person walking"

# Evaluate quality
ciagen evaluate \
    --real data/real/train/images/ \
    --generated data/generated/ \
    --metrics fid mld

# Filter generated images
ciagen filter \
    --generated data/generated/ \
    --method top-k \
    --value 100

# Auto-caption images
ciagen caption \
    --images data/real/train/images/ \
    --output data/real/train/captions/ \
    --engine ollama \
    --model llava
```

### Hydra (Advanced)

```bash
python run.py task=gen model.cn_use=lllyasviel_canny prompt.base="a person"
python run.py task=dtd
python run.py task=ptd
python run.py task=filtering
python run.py task=mix
python run.py task=train
```

See `ciagen/conf/config.yaml` for all configuration options.

## Pipeline

The recommended workflow:

```
real images ──► condition extraction ──► SD + ControlNet ──► synthetic images
                                                              │
real images ──────────────────────────────────────────────► evaluate ──► filter ──► mix ──► train
```

1. **Generate** -- Extract a control condition (edges, pose, segmentation) from each real image, then generate synthetic variations using Stable Diffusion + ControlNet
2. **Evaluate** -- Compute distribution-level metrics (FID, IS) and per-image metrics (Mahalanobis distance)
3. **Filter** -- Select the best synthetic images based on quality scores
4. **Mix** -- Combine real and filtered synthetic data into a training dataset
5. **Train** -- Train your downstream model (YOLOv8 for detection, InceptionV3 for classification)

## Available Extractors

| Extractor | Description | Use Case |
|-----------|-------------|----------|
| `canny` | Canny edge detection | General purpose, preserves structure |
| `openpose` | Human pose estimation | People, actions, body pose |
| `segmentation` | YOLOv8 semantic segmentation | Object boundaries |
| `mediapipe_face` | MediaPipe face landmarks | Facial emotion, face generation |

## Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fid` | Distribution-to-Distribution | Frechet Inception Distance -- lower is better |
| `inception_score` | Distribution-to-Distribution | Inception Score -- higher is better |
| `mld` | Point-to-Distribution | Mahalanobis distance -- per-image, lower is better |

## Data Structure

```
data/
├── real/{dataset}/
│   ├── train/{images,labels,captions}/
│   ├── val/{images,labels,captions}/
│   └── test/{images,labels,captions}/
├── generated/{dataset}/{controlnet-model}/
│   ├── metadata.yaml
│   └── *.png
└── mixed/{dataset}/
```

## Example Datasets

```bash
python run.py task=prepare_data data.base=coco       # COCO People
python run.py task=prepare_data data.base=flickr30k   # Flickr30K Entities
python run.py task=prepare_data data.base=fer         # Facial Emotion Recognition
python run.py task=prepare_data data.base=mocs        # Construction Sites
```

## Documentation

Full documentation is available in the `docs/` directory and can be built with MkDocs:

```bash
pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR guidelines.

## License

This project is licensed under the [GNU Affero General Public License v3](LICENSE).

Copyright (c) 2024 Universite de Mons, Multitel, Universite Libre de Bruxelles, Universite Catholique de Louvain.
