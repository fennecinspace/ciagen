from typing import Any, Callable, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ciagen.feature_extractors.inception_extractor import (
    InceptionModel,
    InceptionModelSoftmaxed,
    inception_transform,
)
from ciagen.qm import id_transform
from ciagen.qm.calculator import CovCalculator, MeanCalculator
from ciagen.qm.dtd_distances import frechet_distance_gaussian_version
from ciagen.qm.dtd_distances.wasserstein_distance import (
    wasserstein_distance_gaussian_version,
)

from torch.utils.data import Dataset, DataLoader


class FID:
    """
    Implementation of the Frechet Inception Distance: https://arxiv.org/pdf/1706.08500.pdf
    """

    def __init__(
        self,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        eps: float = 1e-16,
        use_wasserstein: bool = False,
        softmaxed: bool = False,
        weights: str = "DEFAULT",
    ):

        self._using_inception = feature_extractor is None

        self._feature_extractor = (
            (
                InceptionModelSoftmaxed(weights=weights)
                if softmaxed
                else InceptionModel(weights=weights)
            )
            if feature_extractor is None
            else feature_extractor
        )

        self._distribution_distance = (
            (
                lambda umean, ucov, vmean, vcov: wasserstein_distance_gaussian_version(
                    umean=umean,
                    ucov=ucov,
                    vmean=vmean,
                    vcov=vcov,
                    to_type="torch",
                )
            )
            if use_wasserstein
            else (
                lambda umean, ucov, vmean, vcov: frechet_distance_gaussian_version(
                    umean=umean,
                    ucov=ucov,
                    vmean=vmean,
                    vcov=vcov,
                    to_type="torch",
                )
            )
        )

        self._real_mean_calculator = MeanCalculator()
        self._real_cov_calculator = CovCalculator()

        self._synthetic_mean_calculator = MeanCalculator()
        self._synthetic_cov_calculator = CovCalculator()

        self.eps = eps

    def update(self, samples: torch.Tensor, is_real: bool = True):
        with torch.no_grad():
            features = self._feature_extractor(samples)
            if is_real:
                self._real_mean_calculator(features)
                self._real_cov_calculator(features)
            else:
                self._synthetic_mean_calculator(features)
                self._synthetic_cov_calculator(features)

    def score(
        self,
        real_samples: Union[torch.Tensor, Dataset, DataLoader],
        synthetic_samples: Union[torch.Tensor, Dataset, DataLoader],
        batch_size=32,
    ):

        self._real_cov_calculator.reset()
        self._real_mean_calculator.reset()

        self._synthetic_cov_calculator.reset()
        self._synthetic_mean_calculator.reset()

        if isinstance(real_samples, torch.Tensor) and isinstance(
            synthetic_samples, torch.Tensor
        ):
            self.update(real_samples, is_real=True)
            self.update(synthetic_samples, is_real=False)

            return self.instant_score()
        if isinstance(real_samples, Dataset) and isinstance(synthetic_samples, Dataset):
            real_dataloader = DataLoader(real_samples, batch_size=batch_size)
            synthetic_dataloader = DataLoader(synthetic_samples, batch_size=batch_size)

            for rx in tqdm(real_dataloader):
                self.update(rx, is_real=True)
            for sx in tqdm(synthetic_dataloader):
                self.update(sx, is_real=False)

            return self.instant_score()
        if isinstance(real_samples, DataLoader) and isinstance(
            synthetic_samples, DataLoader
        ):
            for rx in tqdm(real_samples):
                self.update(rx, is_real=True)
            for sx in tqdm(synthetic_samples):
                self.update(sx, is_real=False)

            return self.instant_score()

        raise ValueError(
            f"Data type not supported or not the same type: {type(real_samples)=}, {type(synthetic_samples)=}"
        )

    def instant_score(self):

        real_mean = self._real_mean_calculator.state()
        real_cov = self._real_cov_calculator.state()

        synthetic_mean = self._synthetic_mean_calculator.state()
        synthetic_cov = self._synthetic_cov_calculator.state()

        res = self._distribution_distance(
            umean=real_mean,
            ucov=real_cov,
            vmean=synthetic_mean,
            vcov=synthetic_cov,
        )

        res = res.real
        return float(res)
