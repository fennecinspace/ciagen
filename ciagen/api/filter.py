from pathlib import Path
from typing import Dict, Optional

import yaml

from ciagen.utils.io import logger


def filter_generated(
    generated: str | Path,
    ptd_scores: Optional[Dict] = None,
    method: str = "top-k",
    value: float = 1000,
    metric: str = "mld",
    feature_extractor: str = "vit",
) -> Dict:
    """Filter generated images based on point-to-distribution quality scores.

    Reads PTD scores from metadata.yaml in the generated directory, or accepts
    them directly via the ptd_scores parameter.

    Args:
        generated: Directory containing generated images and metadata.yaml.
        ptd_scores: Pre-computed PTD scores. If None, reads from metadata.yaml.
        method: Filtering method ('threshold', 'top-k', or 'top-p').
        value: Filtering threshold value.
            - threshold: keep images with distance <= value
            - top-k: keep the k images with smallest distances
            - top-p: keep the top proportion (0 <= p <= 1) of images
        metric: PTD metric name to use for filtering.
        feature_extractor: Feature extractor name whose scores to use.

    Returns:
        Dictionary mapping metric names to {fe: {image_path: score}} for kept images.
    """
    generated = Path(generated)
    metadata_file = generated / "metadata.yaml"

    if ptd_scores is None:
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"No metadata.yaml found in {generated}. "
                "Run evaluate() with ptd metrics first."
            )
        with open(metadata_file, "r") as f:
            metadata_dict = yaml.safe_load(f)
        ptd_scores = metadata_dict["results"]["metrics"]["ptd"]

    kept_images = {}

    for metric_name in ptd_scores:
        ptd_by_fe = ptd_scores[metric_name]
        kept_images_by_fe = {}

        for fe in ptd_by_fe:
            if fe != feature_extractor:
                continue

            ptd = list(ptd_by_fe[fe].items())
            ptd = [(path, abs(float(score))) for path, score in ptd]

            if method == "threshold":
                kept = [item for item in ptd if item[1] <= value]
            elif method == "top-p":
                if not 0 <= value <= 1:
                    raise ValueError("top-p value must be between 0 and 1")
                ptd_sorted = sorted(ptd, key=lambda a: a[1])
                kept = ptd_sorted[: int(len(ptd) * value)]
            elif method == "top-k":
                k = int(value)
                if not 0 <= k <= len(ptd):
                    raise ValueError(
                        f"top-k value must be between 0 and {len(ptd)}"
                    )
                ptd_sorted = sorted(ptd, key=lambda a: a[1])
                kept = ptd_sorted[:k]
            else:
                raise ValueError(
                    f"Unknown filtering method: {method}. "
                    "Use 'threshold', 'top-k', or 'top-p'"
                )

            kept = sorted(kept, reverse=True, key=lambda a: a[1])
            kept_images_by_fe[fe] = {path: score for path, score in kept}

        kept_images[metric_name] = kept_images_by_fe

    logger.info(
        f"Filtering ({method}={value}): kept {sum(len(v) for v in kept_images_by_fe.values())} images"
    )

    return kept_images
