# ciagen.qm.metrics

import torch
from tqdm import tqdm
from PIL import Image
from typing import Any, Callable
import numpy as np

from ciagen.qm.ptd_distances.mahalanobis import mahalanobis_distance_calc
from ciagen.feature_extractors.inception_extractor import (
    InceptionModel,
    inception_transform,
)
from ciagen.qm import id_transform
from ciagen.utils.common import logger


class MLD:
    """
    Compute Mahalanobis distance for each generated image with the distribution of real images.
    """

    def __init__(
        self,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        transform: None | Callable[[Any], torch.Tensor] = None,
        distribution_distance: None | Callable[[Any, Any], torch.Tensor | float] = None,
    ):

        self._feature_extractor = (
            InceptionModel() if feature_extractor is None else feature_extractor
        )

        self._using_inception = feature_extractor is None

        if self._using_inception:
            self._transform = inception_transform()
        else:
            self._transform = transform or id_transform()

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

        def transform_images(samples):
            im_transformed = torch.empty(0)
            for sample in tqdm(samples):
                im_transformed = torch.cat(
                    (im_transformed, feature_extractor(transform(sample).unsqueeze(0))),
                    dim=0,
                )
            return im_transformed

        with torch.no_grad():
            if not isinstance(real_samples, torch.Tensor):
                real_transformed = transform_images(real_samples)
            else:
                real_transformed = feature_extractor(transform(real_samples))

            if not isinstance(synthetic_samples, torch.Tensor):
                synthetic_transformed = transform_images(synthetic_samples)
            else:
                synthetic_transformed = feature_extractor(transform(synthetic_samples))

            print(
                real_transformed.shape,
                type(real_transformed),
                synthetic_transformed.shape,
                type(synthetic_transformed),
            )

        return torch.squeeze(real_transformed), torch.squeeze(synthetic_transformed)

    def get_mahal_distance(
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
            real_features, synthetic_features = self.trasform_and_extract_features(
                real_samples=real_samples,
                synthetic_samples=synthetic_samples,
                transform=transform,
                feature_extractor=feature_extractor,
            )

        real_distrib_mean = torch.mean(real_features, dim=0)
        real_distrib_cov = torch.cov(real_features.T, correction=0)

        ptd_mahal_distance = []
        logger.info(
            f"Calculating Mahalanobis distance for {len(synthetic_features)} synthetic images"
        )
        for i in tqdm(range(len(synthetic_features))):
            mahal_dist = mahalanobis_distance_calc(
                synthetic_features[i], real_distrib_mean, real_distrib_cov
            )
            ptd_mahal_distance.extend([mahal_dist])

        return ptd_mahal_distance
