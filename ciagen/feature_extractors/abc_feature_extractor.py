from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class FeatureExtractor(ABC, torch.nn.Module):
    @abstractmethod
    def name(self) -> str: ...
