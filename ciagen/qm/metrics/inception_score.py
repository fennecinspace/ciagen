# See A Note on the Inceptoin Score (https://arxiv.org/pdf/1801.01973) for an in-depth
# discussion.


from typing import Any, Callable, Collection

import numpy as np
import torch
import torch.utils
from PIL import Image
from tqdm import tqdm

from ciagen.feature_extractors.inception_extractor import (
    InceptionModelSoftmaxed,
    inception_transform,
)
from ciagen.qm import TL, VirtualDataloader, id_transform
from ciagen.qm.divergences import kl_divergence


class IS:
    """
    Implementation of the Inception Score:
    Original paper: https://arxiv.org/abs/1606.03498
    In depth discussion: https://arxiv.org/pdf/1801.01973.pdf
    """

    def __init__(
        self,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        distribution_distance: (
            None | Callable[[Any, Any], np.ndarray | torch.Tensor | float]
        ) = None,
        transform: None | Callable[[Any], torch.Tensor] = None,
        eps: float = 1e-16,
    ):
        self._feature_extractor = (
            InceptionModelSoftmaxed()
            if feature_extractor is None
            else feature_extractor
        )
        self._eps = eps
        self._using_inception = feature_extractor is None

        if self._using_inception:
            self._transform = inception_transform()
        else:
            self._transform = transform or id_transform()

        if distribution_distance is None:

            def dd(p, q):
                score = kl_divergence(p, q, as_expectance=True)
                return torch.exp(score)

            self.distribution_distance = dd
        else:
            self.distribution_distance = distribution_distance

    def transform_and_extract_features(
        self,
        samples: Collection[torch.Tensor] | torch.utils.data.DataLoader,
        transform: Callable[[Any], torch.Tensor] | None = None,
        feature_extractor: Callable[[Any], torch.Tensor] | None = None,
    ) -> torch.Tensor:

        if feature_extractor is not None:
            transform = id_transform() if transform is None else transform
        else:
            transform = self._transform

        feature_extractor = feature_extractor or self._feature_extractor

        if isinstance(feature_extractor, torch.nn.Module):
            feature_extractor.eval()

        with torch.no_grad():
            if not isinstance(samples, torch.Tensor):
                transformed = []
                for sample in tqdm(samples):
                    transformed.append(feature_extractor(transform(sample)))
                transformed = torch.stack(transformed)
            else:
                transformed = feature_extractor(transform(samples))

        return torch.squeeze(transformed)

    def instant_score(
        self,
        samples: torch.Tensor | Image.Image | torch.utils.data.DataLoader,
        as_float: bool = True,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        distribution_distance: (
            None | Callable[[Any, Any], np.ndarray | torch.Tensor | float]
        ) = None,
        transform: None | Callable[[Any], torch.Tensor] = None,
        already_transformed: bool = False,
    ):

        if already_transformed:
            pconditional = samples
        else:
            pconditional = self.transform_and_extract_features(
                samples=samples,
                transform=transform,
                feature_extractor=feature_extractor,
            )

        pmarginal = torch.mean(pconditional, dim=0)

        distribution_distance = distribution_distance or self.distribution_distance
        res = distribution_distance(pconditional, pmarginal)

        if as_float:
            res = float(res)

        return res

    def run_score(
        self,
        synthetic_samples: torch.Tensor | Image.Image,
        already_transformed: bool = False,
        as_float: bool = True,
        times: int = 10,
        sampling_size: float = 0.5,
        info: bool = False,
        **kwargs,
    ):
        """
        Following the original paper (https://arxiv.org/pdf/1606.03498), it is recommended
        to run the experience 10 times with 5000 samples as batch.
        """

        sampling_size = int(len(synthetic_samples) * sampling_size)
        accum = 0

        for i in tqdm(range(times)):
            index = np.random.choice(
                len(synthetic_samples), sampling_size, replace=False
            )
            virtual_samples = VirtualDataloader(dataset=synthetic_samples, index=index)

            accum += self.instant_score(
                samples=virtual_samples.as_list(),
                as_float=as_float,
                already_transformed=already_transformed,
            )

            if info:
                print(f"Current score: {accum / (i + 1)}, for iteration {i + 1}")

        return accum / times
