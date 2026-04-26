import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from diffusers.utils import load_image
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ciagen.utils.io import list_files


class NaiveTensorDataset(Dataset):
    """A simple dataset wrapping a single tensor."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]


class ImageLocalDataset(Dataset):
    """A dataset that loads images (and optionally labels/captions) from local file paths."""

    def __init__(
        self,
        path_to_samples: List[str | Path],
        path_to_labels: Optional[List[str | Path]] = None,
        path_to_captions: Optional[List[str | Path]] = None,
        transform=None,
    ) -> None:
        self.transform = transform
        self.path_to_samples = sorted(path_to_samples)

        self.path_to_labels = None
        if path_to_labels is not None:
            if len(path_to_labels) != len(path_to_samples):
                raise ValueError(
                    f"Labels and samples must have the same length. "
                    f"Got {len(path_to_labels)} labels and {len(path_to_samples)} samples"
                )
            self.path_to_labels = sorted(path_to_labels)

        self.path_to_captions = None
        if path_to_captions is not None:
            if len(path_to_captions) != len(path_to_samples):
                raise ValueError(
                    f"Captions and samples must have the same length. "
                    f"Got {len(path_to_captions)} captions and {len(path_to_samples)} samples"
                )
            self.path_to_captions = sorted(path_to_captions)

    def __len__(self):
        return len(self.path_to_samples)

    def __getitem__(self, idx):
        sample = Image.open(self.path_to_samples[idx])

        res = []
        if self.transform:
            sample = self.transform(sample)
            res.append(sample)

        if self.path_to_labels is not None:
            with open(self.path_to_labels[idx], "r") as f:
                label = f.read()
            res.append(label)

        if self.path_to_captions is not None:
            with open(self.path_to_captions[idx], "r") as f:
                caption = f.read()
            res.append(caption)

        if len(res) == 1:
            return res[0]
        return tuple(res)


def get_tensor_from_iterable(x):
    """Extract the first torch.Tensor from an iterable or return x if it is already one."""
    if isinstance(x, torch.Tensor):
        return x
    for item in x:
        if isinstance(item, torch.Tensor):
            return item
    return None


def cast_to_dataloader(
    samples: Union[torch.Tensor, Dataset, DataLoader],
    batch_size: int = 32,
) -> DataLoader:
    """Convert a tensor, Dataset, or DataLoader into a DataLoader."""
    if isinstance(samples, torch.Tensor):
        dataset = NaiveTensorDataset(samples)
        return DataLoader(dataset, batch_size=batch_size)
    elif isinstance(samples, Dataset):
        return DataLoader(samples, batch_size=batch_size)
    elif isinstance(samples, DataLoader):
        return samples
    else:
        raise ValueError(f"Data type not supported: {type(samples)}")


def list_all_files(
    path: Path,
    formats: List[str],
    include: bool = True,
    limit_size: int = 0,
) -> List[str]:
    """List files in a directory filtered by extensions."""
    ff = (
        (lambda x: any(x.endswith(f) for f in formats))
        if include
        else (lambda x: not any(x.endswith(f) for f in formats))
    )
    all_files = list(filter(ff, os.listdir(str(path.absolute()))))
    if limit_size:
        return all_files[:limit_size]
    return all_files


def create_local_dataloader(
    samples_path: str | Path,
    labels_path: Optional[str | Path] = None,
    captions_path: Optional[str | Path] = None,
    limit_size: int = 0,
    transform=None,
    shuffle: bool = True,
    datatype: str = "image",
    batch_size: int = 32,
    sample_formats: Optional[List[str]] = None,
    label_formats: Optional[List[str]] = None,
    caption_formats: Optional[List[str]] = None,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader from local directory paths."""
    if sample_formats is None:
        sample_formats = ["jpg", "jpeg", "png"]
    if label_formats is None:
        label_formats = ["txt"]
    if caption_formats is None:
        caption_formats = ["txt"]

    samples_path = Path(samples_path)
    labels_path = Path(labels_path) if labels_path else None
    captions_path = Path(captions_path) if captions_path else None

    samples_list = list_all_files(samples_path, formats=sample_formats, limit_size=limit_size)
    samples_list = [os.path.join(str(samples_path.absolute()), x) for x in samples_list]

    labels_list = None
    if labels_path is not None:
        labels_list = list_all_files(labels_path, formats=label_formats, limit_size=limit_size)
        labels_list = [os.path.join(str(labels_path.absolute()), x) for x in labels_list]

    captions_list = None
    if captions_path is not None:
        captions_list = list_all_files(captions_path, formats=caption_formats, limit_size=limit_size)
        captions_list = [os.path.join(str(captions_path.absolute()), x) for x in captions_list]

    if datatype == "image":
        dataset = ImageLocalDataset(
            path_to_samples=samples_list,
            path_to_labels=labels_list,
            path_to_captions=captions_list,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown datatype: {datatype}")

    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


def load_images_from_directory(
    directory: str | Path,
    formats: List[str] | None = None,
    to_tensors: bool = False,
    ptd: bool = False,
    limit_size: int = 0,
) -> list | tuple[list, list[str]]:
    """Load images from a directory, optionally as tensors and with filenames."""
    if formats is None:
        formats = ["png", "jpg", "jpeg"]

    directory = Path(directory)
    limit = limit_size or len(list_files(directory, formats))
    images_paths = sorted(list_files(directory, formats)[:limit])

    image_names = [] if ptd else None
    images = []

    transform = transforms.ToTensor() if to_tensors else lambda img: img

    for image_path in images_paths:
        try:
            image = load_image(image_path)
            image = transform(image)

            if ptd:
                image_names.append(image_path.split("/")[-1])
            images.append(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    if ptd:
        return images, image_names
    return images


def create_transform_dict(cfg: DictConfig) -> Dict[str, object]:
    """Create a mapping from feature extractor names to their transforms."""
    from ciagen.feature_extractors import instance_transform

    return {fe: instance_transform(fe) for fe in cfg["metrics"]["fe"]}


def create_dataloader(
    paths: Dict[str, str | Path],
    cfg: DictConfig,
    feature_extractor_name: str,
    transform_dict: Dict[str, object],
    is_real: bool,
) -> DataLoader:
    """Create a DataLoader for real or synthetic data based on config."""
    samples_path = paths["real_images"] if is_real else paths["generated"]
    labels_path = paths["real_labels"] if is_real else None
    captions_path = paths["real_captions"] if is_real else None
    limit_size = cfg["data"]["limit_size_real"] if is_real else cfg["data"]["limit_size_syn"]

    return create_local_dataloader(
        samples_path=samples_path,
        labels_path=labels_path,
        captions_path=captions_path,
        limit_size=limit_size,
        datatype=cfg["data"]["datatype"],
        transform=transform_dict[feature_extractor_name],
        batch_size=cfg["data"]["batch_size"],
        sample_formats=cfg["data"]["image_formats"],
    )


def force_device(device: str):
    """Return a function that moves all its arguments to the given device."""

    def to_device(*args):
        return tuple(x.to(device) for x in args)

    return to_device
