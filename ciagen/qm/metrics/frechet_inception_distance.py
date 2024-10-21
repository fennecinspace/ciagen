from typing import Any, Callable, Union

import torch
from tqdm import tqdm

from ciagen.exes.setup_data import force_device
from ciagen.feature_extractors.inception_extractor import (
    InceptionModel,
)
from ciagen.qm.calculator import CovCalculator, MeanCalculator
from ciagen.qm.dtd_distances import frechet_distance_gaussian_version
from ciagen.qm.dtd_distances.wasserstein_distance import (
    wasserstein_distance_gaussian_version,
)

from torch.utils.data import Dataset, DataLoader
from ciagen.qm.metrics.abc_metric import QualityMetric
from ciagen.utils.common import ciagen_logger

from ciagen.utils.data_loader import (
    cast_to_dataloader,
    get_tensor_from_iterable,
)


class FID(QualityMetric):
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
        device: str = "cpu",
    ):
        self.device = device
        self._using_inception = feature_extractor is None

        self._feature_extractor = (
            InceptionModel(weights=weights, softmaxed=softmaxed)
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

    @classmethod
    def allows_for_gpu(cls):
        return True

    def name(self):
        return "fid"

    def update(self, samples: torch.Tensor, is_real: bool = True):
        with torch.no_grad():
            self._feature_extractor.eval()

            features = self._feature_extractor(samples.to(self.device)).to(self.device)
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

        (
            self._feature_extractor,
            self._real_cov_calculator,
            self._real_mean_calculator,
            self._synthetic_cov_calculator,
            self._synthetic_mean_calculator,
        ) = force_device(self.device)(
            self._feature_extractor,
            self._real_cov_calculator,
            self._real_mean_calculator,
            self._synthetic_cov_calculator,
            self._synthetic_mean_calculator,
        )

        real_dataloader = cast_to_dataloader(real_samples, batch_size=batch_size)
        synthetic_dataloader = cast_to_dataloader(
            synthetic_samples, batch_size=batch_size
        )

        ciagen_logger.info(
            f"Computing Frechet Inception Distance using {self._feature_extractor.name()} as feature extractor"
        )

        ciagen_logger.info("Computing distribution from real samples")
        for rx in tqdm(real_dataloader):
            rx = get_tensor_from_iterable(rx)
            self.update(rx, is_real=True)

        ciagen_logger.info("Computing distribution from synthetic samples")
        for sx in tqdm(synthetic_dataloader):
            sx = get_tensor_from_iterable(sx)
            self.update(sx, is_real=False)

        return self.instant_score()

    def instant_score(self):

        real_mean = self._real_mean_calculator.state()
        real_cov = self._real_cov_calculator.state()

        synthetic_mean = self._synthetic_mean_calculator.state()
        synthetic_cov = self._synthetic_cov_calculator.state()

        real_mean, real_cov, synthetic_mean, synthetic_cov = force_device(self.device)(
            real_mean, real_cov, synthetic_mean, synthetic_cov
        )

        # force same dtype
        fixed_type = real_mean.dtype
        real_mean = real_mean.type(fixed_type)
        real_cov = real_cov.type(fixed_type)
        synthetic_mean = synthetic_mean.type(fixed_type)
        synthetic_cov = synthetic_cov.type(fixed_type)

        res = self._distribution_distance(
            umean=real_mean,
            ucov=real_cov,
            vmean=synthetic_mean,
            vcov=synthetic_cov,
        )

        res = res.real
        return float(res)
