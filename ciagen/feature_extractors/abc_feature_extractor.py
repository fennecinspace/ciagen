from abc import ABC, abstractmethod

import torch


class FeatureExtractor(ABC, torch.nn.Module):
    @classmethod
    @abstractmethod
    def allows_for_gpu(cls) -> bool: ...

    @abstractmethod
    def name(self) -> str: ...
