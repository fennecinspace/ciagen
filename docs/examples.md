---
title: Examples
description: Dataset preparation (COCO, Flickr30K, FER, MOCS), mixing real+synthetic data, and training YOLOv8 and InceptionV3.
keywords: examples, dataset, coco, flickr30k, fer, mocs, yolo, training
---

# Examples

The `examples/` directory contains ready-to-use scripts for dataset preparation, mixing, and training.

## Dataset Preparation

These scripts download and structure public datasets into the format CIA expects.

### COCO People

```bash
python run.py task=prepare_data data.base=coco
```

Downloads COCO people dataset and converts annotations to YOLO format with captions.

### Flickr30K Entities

```bash
python run.py task=prepare_data data.base=flickr30k
```

Downloads Flickr30K, extracts person regions with bounding boxes and captions.

### FER (Facial Emotion Recognition)

```bash
python run.py task=prepare_data data.base=fer_real
```

Downloads FER dataset from Kaggle. Requires `~/.kaggle/kaggle.json` (see [Kaggle API](https://www.kaggle.com/docs/api)).

### MOCS (Moving Objects in Construction Sites)

```bash
python run.py task=prepare_data data.base=mocs
```

Downloads the MOCS dataset with 13 construction-related classes.

## Training

### YOLOv8 Object Detection

```bash
python run.py task=train data.base=coco
```

Trains a YOLOv8n model on the mixed dataset. Logs to Weights & Biases.

### InceptionV3 FER Classification

```bash
python run.py task=train data.base=fer
```

Trains an InceptionV3 classifier on the mixed FER dataset.

## Mixing Datasets

Create a mixed real+synthetic dataset:

```bash
python run.py task=mix data.base=coco ml.augmentation_percent=0.25
```

## Data Structure

All prepared datasets follow this layout:

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
