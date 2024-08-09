from typing import Any, Callable
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image

from qmeasures import id_transform
from qmeasures.distances import frechet_distance_gaussian_version
from qmeasures.distances.wasserstein_distance import (
    wasserstein_distance_gaussian_version,
)

from .inception import InceptionSoftmax, inception_transform


class FID:
    """
    Implementation of the Frechet Inception Distance: https://arxiv.org/pdf/1706.08500.pdf
    """

    def __init__(
        self,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        transform: None | Callable[[Any], torch.Tensor] = None,
        distribution_distance: None | Callable[[Any, Any], torch.Tensor | float] = None,
        eps: float = 1e-16,
        use_wasserstein: bool = False,
    ):

        # TODO: I'm using the softmaxed version of the inception model.
        # Mathematically the transformation should not affect the distance between distributions,
        # I feel like the paper uses the inception model without the softmaxed layer. For the
        # moment I'm not able to make it work, weights go to infinity after matrix multiplication.

        # self._feature_extractor = (
        #     inception_v3() if feature_extractor is None else feature_extractor
        # )

        self._feature_extractor = (
            InceptionSoftmax() if feature_extractor is None else feature_extractor
        )

        self._using_inception = feature_extractor is None

        if self._using_inception:
            self._transform = inception_transform()
        else:
            self._transform = transform or id_transform()

        if distribution_distance is None:

            def dd(real_samples, synthetic_samples):
                real_mean = real_samples.mean(dim=0)
                synthetic_mean = synthetic_samples.mean(dim=0)
                real_cov = real_samples.T.cov()
                synthetic_cov = synthetic_samples.T.cov()

                if use_wasserstein:
                    return wasserstein_distance_gaussian_version(
                        real_mean, real_cov, synthetic_mean, synthetic_cov
                    )
                else:
                    return frechet_distance_gaussian_version(
                        real_mean, real_cov, synthetic_mean, synthetic_cov
                    )

            self.distribution_distance = dd
        else:
            self.distribution_distance = distribution_distance

    def trasform_and_extract_features(
        self,
        real_samples: torch.Tensor | Image.Image | torch.utils.data.DataLoader,
        synthetic_samples: torch.Tensor | Image.Image | torch.utils.data.DataLoader,
        transform: Callable[[Any], torch.Tensor] | None = None,
        feature_extractor: Callable[[Any], torch.Tensor] | None = None,
    ):

        if feature_extractor is not None:
            transform = id_transform() if transform is None else transform
        else:
            transform = self._transform

        feature_extractor = feature_extractor or self._feature_extractor

        if isinstance(feature_extractor, torch.nn.Module):
            feature_extractor.eval()

        with torch.no_grad():
            if not isinstance(real_samples, torch.Tensor):
                real_transformed = []
                for sample in tqdm(real_samples):
                    real_transformed.append(
                        feature_extractor(transform(sample).unsqueeze(0))
                    )
                real_transformed = torch.stack(real_transformed)
            else:
                real_transformed = feature_extractor(transform(real_samples))

            if not isinstance(synthetic_samples, torch.Tensor):
                synthetic_transformed = []
                for sample in tqdm(synthetic_samples):
                    synthetic_transformed.append(
                        feature_extractor(transform(sample).unsqueeze(0))
                    )
                synthetic_transformed = torch.stack(synthetic_transformed)
            else:
                synthetic_transformed = feature_extractor(transform(synthetic_samples))

        return torch.squeeze(real_transformed), torch.squeeze(synthetic_transformed)

    def instant_score(
        self,
        real_samples: torch.Tensor | Image.Image | torch.utils.data.DataLoader,
        synthetic_samples: torch.Tensor | Image.Image | torch.utils.data.DataLoader,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        distribution_distance: (
            None | Callable[[Any, Any], np.ndarray | torch.Tensor | float]
        ) = None,
        transform: None | Callable[[Any], torch.Tensor] = None,
        already_transformed: bool = False,
    ):

        if already_transformed:
            real_samples, synthetic_samples = real_samples, synthetic_samples
        else:
            real_samples, synthetic_samples = self.trasform_and_extract_features(
                real_samples=real_samples,
                synthetic_samples=synthetic_samples,
                transform=transform,
                feature_extractor=feature_extractor,
            )

        distribution_distance = distribution_distance or self.distribution_distance
        res = distribution_distance(real_samples, synthetic_samples)

        return float(res)
