from typing import Any, Callable

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


class FID:
    """
    Implementation of the Frechet Inception Distance: https://arxiv.org/pdf/1706.08500.pdf
    """

    def __init__(
        self,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        transform: None | Callable[[Any], torch.Tensor] = None,
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

        if self._using_inception:
            self._transform = inception_transform()
        else:
            self._transform = transform or id_transform()

        self._distribution_distance = (
            wasserstein_distance_gaussian_version
            if use_wasserstein
            else frechet_distance_gaussian_version
        )

        self._real_mean_calculator = MeanCalculator()
        self._real_cov_calculator = CovCalculator()

        self._synthetic_mean_calculator = MeanCalculator()
        self._synthetic_cov_calculator = CovCalculator()

        self._composed = lambda x: self._feature_extractor(self._transform(x))

        self.eps = eps

    def update(self, samples: torch.Tensor, is_real: bool = True):
        if is_real:
            self._real_mean_calculator(self._composed(samples))
            self._real_cov_calculator(self._composed(samples))
        else:
            self._synthetic_mean_calculator(self._composed(samples))
            self._synthetic_cov_calculator(self._composed(samples))

    def score(self):

        real_mean = self._real_mean_calculator.state()
        real_cov = self._real_cov_calculator.state()

        synthetic_mean = self._synthetic_mean_calculator.state()
        synthetic_cov = self._synthetic_cov_calculator.state()

        res = frechet_distance_gaussian_version(
            umean=real_mean,
            ucov=real_cov,
            vmean=synthetic_mean,
            vcov=synthetic_cov,
            to_type="torch",
        )

        return float(res)
