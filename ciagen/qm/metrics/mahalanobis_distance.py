from typing import Any, Callable, Union

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


from ciagen.qm.calculator import CovCalculator, MeanCalculator
from ciagen.qm.ptd_distances.mahalanobis import mahalanobis_distance_calc
from ciagen.feature_extractors.inception_extractor import (
    InceptionModel,
    inception_transform,
)
from ciagen.qm import id_transform
from ciagen.utils.common import logger
from ciagen.utils.data_loader import NaiveTensorDataset


class MLD:
    """
    Compute Mahalanobis distance for each generated image with the distribution of real images.
    """

    def __init__(
        self,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        weights: str = "DEFAULT",
        softmaxed: bool = False,
    ):

        self._using_inception = feature_extractor is None
        self._feature_extractor = (
            InceptionModel(weights=weights, softmaxed=softmaxed)
            if feature_extractor is None
            else feature_extractor
        )

        self._mean_calculator = MeanCalculator()
        self._cov_calculator = CovCalculator()

    def update(self, samples: torch.Tensor):
        with torch.no_grad():
            features = self._feature_extractor(samples)
            self._mean_calculator(features)
            self._cov_calculator(features)

    def score(
        self,
        real_samples: Union[torch.Tensor, Dataset, DataLoader],
        synthetic_samples: Union[torch.Tensor, Dataset, DataLoader],
        batch_size=32,
    ):

        self._mean_calculator.reset()
        self._cov_calculator.reset()

        if isinstance(real_samples, torch.Tensor):
            real_dataset = NaiveTensorDataset(real_samples)
            real_dataloader = DataLoader(real_dataset, batch_size=batch_size)
        elif isinstance(real_samples, Dataset):
            real_dataloader = DataLoader(real_samples, batch_size=batch_size)
        elif isinstance(real_samples, DataLoader):
            real_dataloader = real_samples
        else:
            raise ValueError(f"Data type not supported: {type(real_samples)}")

        if isinstance(synthetic_samples, torch.Tensor):
            synthetic_dataset = NaiveTensorDataset(synthetic_samples)
            synthetic_dataloader = DataLoader(synthetic_dataset, batch_size=batch_size)
        elif isinstance(synthetic_samples, Dataset):
            synthetic_dataloader = DataLoader(synthetic_samples, batch_size=batch_size)
        elif isinstance(synthetic_samples, DataLoader):
            synthetic_dataloader = synthetic_samples
        else:
            raise ValueError(f"Data type not supported: {type(synthetic_samples)}")

        # first compute mean and cov for real samples
        for x in tqdm(real_dataloader):
            self.update(x)

        real_mean = self._mean_calculator.state()
        real_cov = self._cov_calculator.state()

        def score_batch(x):
            self._feature_extractor.eval()
            x_features = self._feature_extractor(x)
            return mahalanobis_distance_calc(
                x_features,
                real_mean,
                real_cov,
                to_type="torch",
                distance_squared=True,
            )

        results = []
        for x in tqdm(synthetic_dataloader):
            batch_result = score_batch(x)
            results.append(batch_result)

        synthetic_dataset_size = len(synthetic_dataloader.dataset)
        results = torch.stack(results).reshape(synthetic_dataset_size)
        return results
