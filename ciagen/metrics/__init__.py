import numpy as np
import torch
from torchvision.transforms import Lambda

TL = torch.Tensor | np.ndarray


def id_transform():
    return Lambda(lambd=lambda x: x)


def to_numpy(x: TL) -> TL:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def to_tensor(x: TL) -> TL:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def cast_to(x: TL, to_type: str) -> TL:
    """Cast a tensor/array between numpy and torch."""
    if to_type == "numpy":
        return to_numpy(x)
    elif to_type == "torch":
        return to_tensor(x)
    else:
        raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")


class VirtualDataloader:
    """Wraps a dataset and an index for sub-sampling without copying data."""

    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        return self.dataset[self.index[index]]

    def as_tensor(self):
        return torch.stack([self.dataset[i] for i in self.index])

    def as_list(self):
        return self.dataset
