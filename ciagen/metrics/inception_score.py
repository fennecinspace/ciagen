import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ciagen.data.loader import cast_to_dataloader, get_tensor_from_iterable
from ciagen.feature_extractors.inception_extractor import InceptionModel
from ciagen.metrics import VirtualDataloader
from ciagen.metrics.abc_metric import QualityMetric
from ciagen.metrics.accumulators import KLISCalculator
from ciagen.utils.io import logger


class IS(QualityMetric):
    """Inception Score.

    Original paper: https://arxiv.org/abs/1606.03498
    In-depth discussion: https://arxiv.org/pdf/1801.01973.pdf
    """

    def __init__(
        self,
        feature_extractor=None,
        eps: float = 1e-16,
        softmaxed: bool = True,
        weights: str = "DEFAULT",
        device: str = "cpu",
    ):
        self.device = device
        self._feature_extractor = (
            InceptionModel(softmaxed=softmaxed, weights=weights) if feature_extractor is None else feature_extractor
        )
        self._eps = eps
        self._kl_calculator = KLISCalculator(force_probability=True)

    @classmethod
    def allows_for_gpu(cls) -> bool:
        return True

    def name(self) -> str:
        return "is"

    def update(self, synthetic_samples: torch.Tensor):
        with torch.no_grad():
            self._feature_extractor.eval()
            probabilities = self._feature_extractor(synthetic_samples.to(self.device)).to(self.device)
            self._kl_calculator(probabilities)

    def score(
        self,
        real_samples: torch.Tensor | Dataset | DataLoader,
        synthetic_samples: torch.Tensor | Dataset | DataLoader,
        batch_size: int = 32,
    ) -> float:
        self._kl_calculator.reset()
        self._kl_calculator.to(self.device)
        self._kl_calculator.eval()

        synthetic_dataloader = cast_to_dataloader(synthetic_samples, batch_size=batch_size)
        for sx in tqdm(synthetic_dataloader):
            sx = get_tensor_from_iterable(sx)
            self.update(sx)

        return self.instant_score()

    def instant_score(self) -> float:
        return float(self._kl_calculator.state(return_exp_expectation=True).real)

    def run_score(
        self,
        synthetic_samples,
        already_transformed: bool = False,
        as_float: bool = True,
        times: int = 10,
        sampling_size: float = 0.5,
        info: bool = False,
        **kwargs,
    ) -> float:
        """Run IS multiple times with subsampling for robust estimation."""
        sampling_size = int(len(synthetic_samples) * sampling_size)
        accum = 0

        for i in tqdm(range(times)):
            index = np.random.choice(len(synthetic_samples), sampling_size, replace=False)
            _virtual_samples = VirtualDataloader(dataset=synthetic_samples, index=index)

            accum += self.instant_score()

            if info:
                logger.info(f"Current score: {accum / (i + 1)}, for iteration {i + 1}")

        return accum / times
