from abc import ABC, abstractmethod

from typing import List
import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader, Dataset

SampleT = Image.Image | np.ndarray | torch.Tensor


class FeatureExtractor(ABC):
    @abstractmethod
    def _extract(
        self, samples: List[SampleT] | SampleT, **kwargs
    ) -> List[SampleT] | SampleT: ...

    def extract(
        self, samples: DataLoader | Dataset | List[SampleT] | SampleT, **kwargs
    ) -> List[SampleT] | SampleT:
        # test from simpler to harder
        if isinstance(samples, SampleT):
            samples = self.single_transform(sample=samples)
        if isinstance(samples, list):
            samples = [self.single_transform(sample=sample) for sample in samples]

        # add dataset and dataloader stuff here
        return self._extract(samples, **kwargs)

    def single_transform(self, sample: SampleT):
        if isinstance(sample, Image.Image):
            sample = self.transform_from_image(sample)
        if isinstance(sample, np.ndarray):
            sample = self.transform_from_array(sample)
        if isinstance(sample, torch.Tensor):
            sample = self.transform_from_tensor(sample)
        else:
            raise ValueError(f"Unknown type for transformation: {type(sample)=}")

        return sample

    def __call__(
        self, samples: List[SampleT] | SampleT, **kwargs
    ) -> List[SampleT] | SampleT:
        return self.extract(samples, **kwargs)

    def transform_from_image(self, image: Image.Image) -> SampleT:
        raise NotImplementedError

    def transform_from_tensor(self, tensor: torch.Tensor) -> SampleT:
        raise NotImplementedError

    def transform_from_array(self, array: np.ndarray) -> SampleT:
        raise NotImplementedError
