from typing import Any, Callable, Union

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from ciagen.exes.setup_data import force_device
from ciagen.qm.calculator import CovCalculator, MeanCalculator
from ciagen.qm.metrics.abc_metric import QualityMetric
from ciagen.qm.ptd_distances.mahalanobis import mahalanobis_distance_calc
from ciagen.feature_extractors.inception_extractor import InceptionModel
from ciagen.utils.common import ciagen_logger
from ciagen.utils.data_loader import (
    cast_to_dataloader,
    get_tensor_from_iterable,
)


class MLD(QualityMetric):
    """
    Compute Mahalanobis distance for each generated image with the distribution of real images.
    """

    def __init__(
        self,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        device: str = "cpu",
        weights: str = "DEFAULT",
        softmaxed: bool = False,
    ):

        self._using_inception = feature_extractor is None
        self._feature_extractor = (
            InceptionModel(weights=weights, softmaxed=softmaxed)
            if feature_extractor is None
            else feature_extractor
        )
        self.device = device

        self._mean_calculator = MeanCalculator()
        self._cov_calculator = CovCalculator()

        self._mean_calculator = self._mean_calculator.to(device)
        self._cov_calculator = self._cov_calculator.to(device)
        self._feature_extractor = self._feature_extractor.to(device)

    def name(self):
        return "MLD"

    def allows_for_gpu(cls):
        return True

    def update(self, samples: torch.Tensor):
        with torch.no_grad():
            samples = samples.to(self.device)
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

        self._mean_calculator.eval()
        self._cov_calculator.eval()
        self._feature_extractor.eval()

        self._feature_extractor, self._mean_calculator, self._cov_calculator = (
            force_device(device=self.device)(
                self._feature_extractor, self._mean_calculator, self._cov_calculator
            )
        )

        real_dataloader = cast_to_dataloader(real_samples, batch_size=batch_size)
        synthetic_dataloader = cast_to_dataloader(
            synthetic_samples, batch_size=batch_size
        )

        ciagen_logger.info(
            f"Computing Mahalanobis Distance using {self._feature_extractor.name()} as feature extractor"
        )

        ciagen_logger.info("Computing distribution from real samples")
        # first compute mean and cov for real samples
        for x in tqdm(real_dataloader):
            x = get_tensor_from_iterable(x)
            self.update(x)

        real_mean = self._mean_calculator.state()
        real_cov = self._cov_calculator.state()

        real_mean, real_cov = real_mean.to(self.device), real_cov.to(self.device)

        fixed_type = real_mean.dtype
        real_mean = real_mean.type(fixed_type)
        real_cov = real_cov.type(fixed_type)

        def score_batch(x, rmean, rcov):
            x = x.to(self.device)
            x_features = self._feature_extractor(x).to(self.device)
            x_features = x_features.type(fixed_type)

            return mahalanobis_distance_calc(
                x_features,
                rmean,
                rcov,
                to_type="torch",
                distance_squared=True,
            )

        ciagen_logger.info("Computing distance for synthetic samples")

        results = torch.zeros(len(synthetic_dataloader.dataset), device=self.device)
        for i, x in tqdm(enumerate(synthetic_dataloader)):

            torch.cuda.empty_cache()

            x = get_tensor_from_iterable(x)
            results[i * batch_size : (i + 1) * batch_size] = score_batch(
                x, real_mean, real_cov
            )
        return results
