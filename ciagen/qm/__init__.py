import numpy as np
import torch
from torchvision.transforms import Lambda

# TODO: because we will be using more tensors that anything else, you should implement a torch version wich should be faster, thus as `torch.nn.Module`.


TL = torch.Tensor | np.ndarray


def id_transform():
    return Lambda(lambd=lambda x: x)


def to_numpy(x: TL):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def to_tensor(x: TL):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return x


class VirtualDataloader:
    """
    Creates a virtual dataloader from a dataset and an index
    """

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

    def as_list(
        self,
    ):  # quick fix for reading data issues caused by tensor transforms between FID and IS
        return self.dataset
