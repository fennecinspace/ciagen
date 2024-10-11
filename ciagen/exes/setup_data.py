from pathlib import Path
from ciagen.feature_extractors import instance_transform


import torch
from omegaconf import DictConfig


from typing import Any, Callable, Dict, List, Tuple

from ciagen.utils.data_loader import create_local_dataloader


def create_transform_dict(cfg: DictConfig) -> Dict[str, Callable[[Any], torch.Tensor]]:
    return {fe: instance_transform(fe) for fe in cfg["metrics"]["fe"]}


def call_dataloader2(
    paths: Dict[str, str | Path],
    cfg: DictConfig,
    feature_extractor_name: str,
    transform_dict: Dict[str, Callable[[Any], torch.Tensor]],
    is_real: bool,
) -> None:
    samples_path = paths["real_images"] if is_real else paths["generated"]
    labels_path = paths["real_labels"] if is_real else None
    captions_path = paths["real_captions"] if is_real else None
    limit_size = (
        cfg["data"]["limit_size_real"] if is_real else cfg["data"]["limit_size_syn"]
    )
    batch_size = cfg["data"]["batch_size"]
    transform = transform_dict[feature_extractor_name]

    return create_local_dataloader(
        samples_path=samples_path,
        labels_path=labels_path,
        captions_path=captions_path,
        limit_size=limit_size,
        datatype=cfg["data"]["datatype"],
        transform=transform,
        batch_size=batch_size,
        sample_formats=cfg["data"]["image_formats"],
    )


def force_device(device: str) -> Tuple:

    def to_device(*args):
        return tuple(x.to(device) for x in args)

    return to_device
