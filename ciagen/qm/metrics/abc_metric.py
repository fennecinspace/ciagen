from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class QualityMetric(ABC):
    @classmethod
    @abstractmethod
    def allows_for_gpu(cls) -> bool: ...

    @abstractmethod
    def name(self) -> str: ...
