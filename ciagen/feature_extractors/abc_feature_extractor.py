from abc import ABC, abstractmethod

from typing import List
import numpy as np
import torch
from PIL import Image

SampleT = Image.Image | np.ndarray | torch.Tensor


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(
        self, samples: List[SampleT] | SampleT, **kwargs
    ) -> List[SampleT] | SampleT: ...

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
