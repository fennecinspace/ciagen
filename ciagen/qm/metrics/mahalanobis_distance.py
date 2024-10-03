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
from ciagen.utils.data_loader import (
    NaiveTensorDataset,
    cast_to_dataloader,
    get_tensor_from_iterable,
)


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

        real_dataloader = cast_to_dataloader(real_samples, batch_size=batch_size)
        synthetic_dataloader = cast_to_dataloader(
            synthetic_samples, batch_size=batch_size
        )

        logger.info(
            f"Computing Mahalanobis Distance using {self._feature_extractor.name()} as feature extractor"
        )

        logger.info("Computing distribution from real samples")
        # first compute mean and cov for real samples
        for x in tqdm(real_dataloader):
            x = get_tensor_from_iterable(x)
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

        logger.info("Computing distance for synthetic samples")
        results = []
        for x in tqdm(synthetic_dataloader):
            x = get_tensor_from_iterable(x)
            batch_result = score_batch(x)
            results.append(batch_result)

        results = torch.cat(results, dim=0)
        return results
