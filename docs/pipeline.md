# Pipeline Overview

The CIA pipeline follows a linear flow from real images to augmented training data. Each step can be run independently or as part of the full pipeline.

```
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│ Real Images  │───►│  Extract     │───►│  SD + ControlNet │───►│  Generated   │
│              │    │  Condition   │    │  Generation      │    │  Images      │
└─────────────┘    └─────────────┘    └──────────────────┘    └──────┬──────┘
                                                                         │
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐             │
│ Real Images  │───►│  Feature     │───►│  Quality Metrics │◄────────────┘
│              │    │  Extraction  │    │  (FID, IS, MLD)  │
└─────────────┘    └─────────────┘    └────────┬─────────┘
                                                  │
                                           ┌──────▼──────┐
                                           │  Filtering   │
                                           │  (top-k, etc)│
                                           └──────┬──────┘
                                                  │
┌─────────────┐    ┌─────────────┐              │
│ Real Images  │───►│    Mix       │◄─────────────┘
│              │    │  Real+Synth  │
└─────────────┘    └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │    Train     │
                   │  Downstream  │
                   │   Model      │
                   └─────────────┘
```

## Step 1: Generate

Extract a control condition from each real image using one of the available extractors, then generate synthetic variations using Stable Diffusion + ControlNet.

**Key decisions:**

- Which **extractor** to use (canny, openpose, segmentation, mediapipe_face)
- Which **Stable Diffusion** and **ControlNet** models from HuggingFace
- How many synthetic images per real image
- What prompts to use (fixed, from captions, or vocabulary-modified)

**Output:** Generated images in `data/generated/{dataset}/{controlnet}/`

## Step 2: Evaluate (DTD + PTD)

Compute quality metrics comparing the real and synthetic distributions.

- **DTD** (Distribution-To-Distribution): FID, Inception Score — measure overall distribution similarity
- **PTD** (Point-To-Distribution): Mahalanobis distance — scores each individual synthetic image

**Output:** Metric scores saved to `metadata.yaml` alongside generated images

## Step 3: Filter

Select the best synthetic images based on PTD scores.

- **top-k**: Keep the k images with smallest distances
- **top-p**: Keep the top proportion (0 ≤ p ≤ 1) of images
- **threshold**: Keep images with distance below a threshold

**Output:** Filtering results appended to `metadata.yaml`

## Step 4: Mix

Combine real and filtered synthetic data into a training-ready dataset (YOLO format or CSV format).

## Step 5: Train

Train downstream models (YOLOv8 for object detection, InceptionV3 for classification) using the mixed dataset.

---

Steps 1–3 are the **core library** functionality. Steps 4–5 are provided as **example scripts** in the `examples/` directory.
