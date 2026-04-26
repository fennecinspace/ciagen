from pathlib import Path
from typing import Dict, Optional

import yaml

from ciagen.feature_extractors import available_feature_extractors
from ciagen.utils.io import logger

VALID_METHODS = frozenset({"threshold", "top-k", "top-p"})
VALID_PTD_METRICS = frozenset({"mld"})


def _validate_filter(
    generated: Path,
    method: str,
    value: float,
    metric: str,
    feature_extractor: str,
) -> None:
    if not generated.is_dir():
        raise NotADirectoryError(f"Generated directory does not exist: {generated}")

    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method '{method}'. Choose from: {', '.join(sorted(VALID_METHODS))}")

    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")

    fe_registry = available_feature_extractors()
    if feature_extractor not in fe_registry:
        raise ValueError(
            f"Invalid feature_extractor '{feature_extractor}'. Choose from: {', '.join(sorted(fe_registry.keys()))}"
        )

    if metric not in VALID_PTD_METRICS:
        raise ValueError(f"Invalid metric '{metric}'. Choose from: {', '.join(sorted(VALID_PTD_METRICS))}")

    if method == "top-p" and not 0 <= value <= 1:
        raise ValueError(f"top-p value must be between 0 and 1, got {value}")


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

    _validate_filter(generated, method, value, metric, feature_extractor)

    if ptd_scores is None:
        if not metadata_file.exists():
            raise FileNotFoundError(f"No metadata.yaml found in {generated}. Run evaluate() with ptd metrics first.")
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
                ptd_sorted = sorted(ptd, key=lambda a: a[1])
                kept = ptd_sorted[: int(len(ptd) * value)]
            elif method == "top-k":
                k = int(value)
                ptd_sorted = sorted(ptd, key=lambda a: a[1])
                kept = ptd_sorted[:k]
            else:
                raise ValueError(f"Unknown filtering method: {method}. Use 'threshold', 'top-k', or 'top-p'")

            kept = sorted(kept, reverse=True, key=lambda a: a[1])
            kept_images_by_fe[fe] = {path: score for path, score in kept}

        kept_images[metric_name] = kept_images_by_fe

    logger.info(f"Filtering ({method}={value}): kept {sum(len(v) for v in kept_images_by_fe.values())} images")

    return kept_images
