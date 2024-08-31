from abc import ABC, abstractmethod

from typing import List
import numpy as np
import torch
from PIL import Image

SampleT = Image.Image | np.ndarray | torch.Tensor


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, samples: List[SampleT] | SampleT, **kwargs) -> List[SampleT]: ...

    def __call__(self, samples: List[SampleT] | SampleT, **kwargs) -> List[SampleT]:
        return self.extract(samples, **kwargs)
