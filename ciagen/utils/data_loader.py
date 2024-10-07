import os
from pathlib import Path
from typing import Dict, List, Union
from omegaconf import DictConfig

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PIL import Image

from ciagen.utils.common import list_files


class NaiveTensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]


class ImageLocalDataset(Dataset):
    def __init__(
        self,
        path_to_samples: List[str | Path],
        path_to_labels: List[str | Path] | None = None,
        path_to_captions: List[str | Path] | None = None,
        transform=None,
    ) -> None:
        self.transform = transform

        self.path_to_samples = path_to_samples[:]
        self.path_to_samples.sort()

        self.path_to_labels = None
        if path_to_labels is not None:
            if len(path_to_labels) != len(path_to_samples):
                raise ValueError(
                    f"Labels and samples must have the same length. Got {len(path_to_labels)} labels and {len(path_to_samples)} samples"
                )

            self.path_to_labels = path_to_labels[:]
            self.path_to_labels.sort()

        self.path_to_captions = None
        if path_to_captions is not None:
            if len(path_to_captions) != len(path_to_samples):
                raise ValueError(
                    f"Captions and samples must have the same length. Got {len(path_to_captions)} captions and {len(path_to_samples)} samples"
                )
            self.path_to_captions = path_to_captions[:]
            self.path_to_captions.sort()

    def __len__(self):
        return len(self.path_to_samples)

    def __getitem__(self, idx):

        # sample = read_image(self.path_to_samples[idx])
        sample = Image.open(self.path_to_samples[idx])  # .convert("RGB")

        res = []
        if self.transform:
            sample = self.transform(sample)
            res += [sample]

        if self.path_to_labels is not None:
            label_file = self.path_to_labels[idx]
            with open(label_file, "r") as f:
                label = f.read()
            res += [label]
        if self.path_to_captions is not None:
            caption_file = self.path_to_captions[idx]
            with open(caption_file, "r") as f:
                caption = f.read()
            res += [caption]

        if len(res) == 1:
            res = res[0]
        else:
            res = tuple(res)
        return res


def get_tensor_from_iterable(x):
    if isinstance(x, torch.Tensor):
        return x
    for item in x:
        if isinstance(item, torch.Tensor):
            return item
    return None


def cast_to_dataloader(
    samples: Union[torch.Tensor, Dataset, DataLoader], batch_size=32
):
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
) -> List[Path]:

    ff = (
        (lambda x: any(x.endswith(f) for f in formats))
        if include
        else (lambda x: not any(x.endswith(f) for f in formats))
    )
    all_files = list(
        filter(
            ff,
            os.listdir(str((path.absolute()))),
        )
    )
    if limit_size:
        return all_files[:limit_size]
    return all_files


def create_local_dataloader(
    samples_path: Path | str,
    labels_path: Path | str | None = None,
    captions_path: Path | str | None = None,
    limit_size: int = 0,
    transform=None,
    shuffle: bool = True,
    datatype: str = "image",
    batch_size: int = 32,
    sample_formats: List[str] | None = None,
    label_formats: List[str] | None = None,
    caption_formats: List[str] | None = None,
    **kwargs,
) -> DataLoader:

    if sample_formats is None:
        sample_formats = ["jpg", "jpeg", "png"]
    if label_formats is None:
        label_formats = ["txt"]
    if caption_formats is None:
        caption_formats = ["txt"]

    samples_path = Path(samples_path)

    if labels_path is not None:
        labels_path = Path(labels_path)

    if captions_path is not None:
        captions_path = Path(captions_path)

    samples_list = list_all_files(
        samples_path,
        formats=sample_formats,
        limit_size=limit_size,
        include=True,
    )
    samples_list = [os.path.join(str(samples_path.absolute()), x) for x in samples_list]

    if labels_path is not None:
        labels_list = list_all_files(
            labels_path,
            formats=label_formats,
            include=True,
            limit_size=limit_size,
        )
        labels_list = [
            os.path.join(str(labels_path.absolute()), x) for x in labels_list
        ]
    else:
        labels_list = None
    labels_list = labels_list or None
    if captions_path is not None:
        captions_list = list_all_files(
            captions_path,
            formats=["txt"],
            include=True,
            limit_size=limit_size,
        )
        captions_list = [
            os.path.join(str(captions_path.absolute()), x) for x in captions_list
        ]
    else:
        captions_list = None
    captions_list = captions_list or None

    if datatype == "image":
        dataset = ImageLocalDataset(
            path_to_samples=samples_list,
            path_to_labels=labels_list,
            path_to_captions=captions_list,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown datatype {datatype}")

    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
