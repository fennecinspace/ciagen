# See A Note on the Inceptoin Score (https://arxiv.org/pdf/1801.01973) for an in-depth
# discussion.


from typing import Any, Callable, Collection

import numpy as np
import torch
import torch.utils
from PIL import Image
from tqdm import tqdm

from ciagen.feature_extractors.inception_extractor import (
    InceptionModel,
    inception_transform,
)

from torch.utils.data import DataLoader, Dataset
from ciagen.qm import TL, VirtualDataloader, id_transform
from ciagen.qm.divergences import kl_divergence
from ciagen.qm.calculator import KLISCalculator


class IS:
    """
    Implementation of the Inception Score:
    Original paper: https://arxiv.org/abs/1606.03498
    In depth discussion: https://arxiv.org/pdf/1801.01973.pdf
    """

    def __init__(
        self,
        feature_extractor: None | Callable[[Any], torch.Tensor] = None,
        eps: float = 1e-16,
        softmaxed: bool = True,
        weights: str = "DEFAULT",
    ):
        self._feature_extractor = (
            InceptionModel(softmaxed=False, weights=weights)
            if feature_extractor is None
            else feature_extractor
        )
        self._eps = eps
        self._kl_calculator = KLISCalculator(force_probability=True)

    def update(
        self,
        synthetic_samples: torch.Tensor | Image.Image | DataLoader | Dataset,
    ):
        with torch.no_grad():
            probabilities = self._feature_extractor(synthetic_samples)
            self._kl_calculator(probabilities)

    def score(
        self,
        real_samples: torch.Tensor | Dataset | DataLoader,
        synthetic_samples: torch.Tensor | Dataset | DataLoader,
        batch_size=32,
    ):
        self._kl_calculator.reset()

        if isinstance(synthetic_samples, torch.Tensor):
            self.update(synthetic_samples)
            return self.instant_score()
        if isinstance(synthetic_samples, Dataset):
            dataloader = DataLoader(synthetic_samples, batch_size=batch_size)

            for x in tqdm(dataloader):
                self.update(x)

            return self.instant_score()

        if isinstance(synthetic_samples, DataLoader):
            for x in tqdm(synthetic_samples):
                self.update(x)

            return self.instant_score()

        raise ValueError(f"Data type not supported: {type(synthetic_samples)}")

    def instant_score(
        self,
    ):

        res = self._kl_calculator.state(return_exp_expectation=True)
        res = res.real
        return float(res)

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
