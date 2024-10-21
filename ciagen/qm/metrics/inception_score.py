# See A Note on the Inceptoin Score (https://arxiv.org/pdf/1801.01973) for an in-depth
# discussion.


from typing import Any, Callable

import numpy as np
import torch
import torch.utils
from PIL import Image
from tqdm import tqdm

from ciagen.feature_extractors.inception_extractor import InceptionModel

from torch.utils.data import DataLoader, Dataset
from ciagen.qm import VirtualDataloader
from ciagen.qm.calculator import KLISCalculator
from ciagen.qm.metrics.abc_metric import QualityMetric
from ciagen.utils.data_loader import cast_to_dataloader, get_tensor_from_iterable


class IS(QualityMetric):
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
        device: str = "cpu",
    ):
        self.device = device
        self._feature_extractor = (
            InceptionModel(softmaxed=softmaxed, weights=weights)
            if feature_extractor is None
            else feature_extractor
        )
        self._eps = eps
        self._kl_calculator = KLISCalculator(force_probability=True)

    @classmethod
    def allows_for_gpu(cls):
        return True

    def name(self) -> str:
        return "IS"

    def update(
        self,
        synthetic_samples: torch.Tensor | Image.Image | DataLoader | Dataset,
    ):
        with torch.no_grad():
            self._feature_extractor.eval()
            probabilities = self._feature_extractor(
                synthetic_samples.to(self.device)
            ).to(self.device)
            self._kl_calculator(probabilities)

    def score(
        self,
        real_samples: (
            torch.Tensor | Dataset | DataLoader
        ),  # this remains here, I'm hoping for a reformat after
        synthetic_samples: torch.Tensor | Dataset | DataLoader,
        batch_size=32,
    ):
        self._kl_calculator.reset()
        self._kl_calculator.to(self.device)
        self._kl_calculator.eval()

        synthetic_dataloader = cast_to_dataloader(
            synthetic_samples, batch_size=batch_size
        )
        for sx in tqdm(synthetic_dataloader):
            sx = get_tensor_from_iterable(sx)
            self.update(sx)

        return self.instant_score()

    def instant_score(self):
        return float(self._kl_calculator.state(return_exp_expectation=True).real)

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
