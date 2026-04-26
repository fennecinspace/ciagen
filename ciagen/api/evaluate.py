from pathlib import Path
from typing import Dict, List, Optional

import torch

from ciagen.data.loader import create_dataloader, load_images_from_directory
from ciagen.feature_extractors import (
    available_feature_extractors,
    instance_feature_extractor,
    instance_transform,
)
from ciagen.metrics.fid import FID
from ciagen.metrics.inception_score import IS
from ciagen.metrics.mahalanobis import MLD
from ciagen.utils.io import logger

torch.backends.cudnn.benchmark = False

AVAILABLE_DTD_METRICS = {
    "fid": FID,
    "inception_score": IS,
}

AVAILABLE_PTD_METRICS = {
    "mld": MLD,
}

VALID_FE = frozenset({"vit", "inception"})
VALID_DEVICES = frozenset({"cuda", "cpu"})


def _validate_evaluate(
    real: Path,
    generated: Path,
    metrics: List[str],
    feature_extractor: str,
    batch_size: int,
    limit_size_real: int,
    limit_size_syn: int,
    device: Optional[str],
) -> None:
    if not real.is_dir():
        raise NotADirectoryError(f"Real images directory does not exist: {real}")
    if not generated.is_dir():
        raise NotADirectoryError(f"Generated images directory does not exist: {generated}")

    if not metrics:
        raise ValueError("metrics list cannot be empty")

    fe_registry = available_feature_extractors()
    if feature_extractor not in fe_registry:
        raise ValueError(
            f"Invalid feature_extractor '{feature_extractor}'. "
            f"Choose from: {', '.join(sorted(fe_registry.keys()))}"
        )

    all_valid = set(AVAILABLE_DTD_METRICS) | set(AVAILABLE_PTD_METRICS)
    unknown = set(metrics) - all_valid
    if unknown:
        raise ValueError(
            f"Unknown metric(s): {', '.join(sorted(unknown))}. "
            f"Choose from: {', '.join(sorted(all_valid))}"
        )

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if limit_size_real < 1:
        raise ValueError(f"limit_size_real must be >= 1, got {limit_size_real}")
    if limit_size_syn < 1:
        raise ValueError(f"limit_size_syn must be >= 1, got {limit_size_syn}")

    if device is not None and device not in VALID_DEVICES:
        raise ValueError(f"device must be 'cuda' or 'cpu', got '{device}'")


def evaluate(
    real: str | Path,
    generated: str | Path,
    metrics: Optional[List[str]] = None,
    feature_extractor: str = "vit",
    batch_size: int = 32,
    limit_size_real: int = 2000,
    limit_size_syn: int = 2000,
    image_formats: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> Dict:
    """Evaluate the quality of generated images against real images.

    Computes Distribution-To-Distribution metrics (FID, IS) and/or
    Point-To-Distribution metrics (Mahalanobis distance).

    Args:
        real: Directory containing real images.
        generated: Directory containing generated images.
        metrics: List of metrics to compute. Options: 'fid', 'inception_score', 'mld'.
                 Defaults to ['fid', 'mld'].
        feature_extractor: Feature extractor to use ('vit', 'inception').
        batch_size: Batch size for computation.
        limit_size_real: Maximum number of real images to use.
        limit_size_syn: Maximum number of synthetic images to use.
        image_formats: Supported image formats.
        device: Device to use ('cuda' or 'cpu'). Auto-detected if None.

    Returns:
        Dictionary with metric scores.
    """
    if metrics is None:
        metrics = ["fid", "mld"]
    if image_formats is None:
        image_formats = ["png", "jpeg", "jpg"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    real = Path(real)
    generated = Path(generated)

    _validate_evaluate(
        real, generated, metrics, feature_extractor,
        batch_size, limit_size_real, limit_size_syn, device,
    )

    results = {}

    dtd_metrics = [m for m in metrics if m in AVAILABLE_DTD_METRICS]
    ptd_metrics = [m for m in metrics if m in AVAILABLE_PTD_METRICS]

    fe_registry = available_feature_extractors()
    fe_class = fe_registry[feature_extractor]
    fe_device = device if fe_class.allows_for_gpu() else "cpu"

    if dtd_metrics:
        results["dtd"] = _compute_dtd(
            real=real,
            generated=generated,
            metrics=dtd_metrics,
            feature_extractor_name=feature_extractor,
            fe_device=fe_device,
            batch_size=batch_size,
            limit_size_real=limit_size_real,
            limit_size_syn=limit_size_syn,
            image_formats=image_formats,
        )

    if ptd_metrics:
        results["ptd"] = _compute_ptd(
            real=real,
            generated=generated,
            metrics=ptd_metrics,
            feature_extractor_name=feature_extractor,
            fe_device=fe_device,
            batch_size=batch_size,
            limit_size_real=limit_size_real,
            limit_size_syn=limit_size_syn,
            image_formats=image_formats,
        )

    return results


def _compute_dtd(
    real: Path,
    generated: Path,
    metrics: List[str],
    feature_extractor_name: str,
    fe_device: str,
    batch_size: int,
    limit_size_real: int,
    limit_size_syn: int,
    image_formats: List[str],
) -> Dict:
    """Compute distribution-to-distribution metrics (FID, IS)."""
    transform = instance_transform(feature_extractor_name)
    fe = instance_feature_extractor(feature_extractor_name, device=fe_device)

    paths = {
        "real_images": str(real),
        "generated": str(generated),
        "real_labels": None,
        "real_captions": None,
    }
    cfg_like = {
        "data": {
            "limit_size_real": limit_size_real,
            "limit_size_syn": limit_size_syn,
            "batch_size": batch_size,
            "datatype": "image",
            "image_formats": image_formats,
        },
        "metrics": {"fe": [feature_extractor_name]},
    }
    transform_dict = {feature_extractor_name: transform}

    real_dl = create_dataloader(paths, cfg_like, feature_extractor_name, transform_dict, is_real=True)
    syn_dl = create_dataloader(paths, cfg_like, feature_extractor_name, transform_dict, is_real=False)

    results = {}
    for metric_name in metrics:
        metric_class = AVAILABLE_DTD_METRICS[metric_name]
        metric_device = fe_device if metric_class.allows_for_gpu() else "cpu"

        metric = metric_class(feature_extractor=fe, device=metric_device)
        logger.info(f"Computing {metric_name} with {feature_extractor_name}")

        score = metric.score(
            real_samples=real_dl,
            synthetic_samples=syn_dl,
            batch_size=batch_size,
        )

        results[metric_name] = score
        logger.info(f"{metric_name}[{feature_extractor_name}] = {score}")

    return results


def _compute_ptd(
    real: Path,
    generated: Path,
    metrics: List[str],
    feature_extractor_name: str,
    fe_device: str,
    batch_size: int,
    limit_size_real: int,
    limit_size_syn: int,
    image_formats: List[str],
) -> Dict:
    """Compute point-to-distribution metrics (Mahalanobis)."""
    transform = instance_transform(feature_extractor_name)
    fe = instance_feature_extractor(feature_extractor_name, device=fe_device)

    paths = {
        "real_images": str(real),
        "generated": str(generated),
        "real_labels": None,
        "real_captions": None,
    }
    cfg_like = {
        "data": {
            "limit_size_real": limit_size_real,
            "limit_size_syn": limit_size_syn,
            "batch_size": batch_size,
            "datatype": "image",
            "image_formats": image_formats,
        },
        "metrics": {"fe": [feature_extractor_name]},
    }
    transform_dict = {feature_extractor_name: transform}

    real_dl = create_dataloader(paths, cfg_like, feature_extractor_name, transform_dict, is_real=True)
    syn_dl = create_dataloader(paths, cfg_like, feature_extractor_name, transform_dict, is_real=False)

    syn_images, syn_names = load_images_from_directory(
        generated, formats=image_formats, ptd=True, limit_size=limit_size_syn
    )

    results = {}
    for metric_name in metrics:
        metric_class = AVAILABLE_PTD_METRICS[metric_name]
        metric_device = fe_device if metric_class.allows_for_gpu() else "cpu"

        metric = metric_class(feature_extractor=fe, device=metric_device)
        logger.info(f"Computing {metric_name} with {feature_extractor_name}")

        scores = metric.score(
            real_samples=real_dl,
            synthetic_samples=syn_dl,
            batch_size=batch_size,
        )

        specific = {}
        for i, name in enumerate(syn_names):
            full_path = str(generated.absolute() / name)
            specific[full_path] = float(scores[i])

        results[metric_name] = specific
        logger.info(f"Done computing {metric_name}")

        del real_dl, syn_dl
        torch.cuda.empty_cache()

    return results
