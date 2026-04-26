from abc import ABC

import torch


class ABCAccum(ABC, torch.nn.Module):
    """Abstract base class for online streaming accumulators."""

    @property
    def _accums(self):
        return [x for x in self.__dict__ if x.endswith("_accum")]

    @property
    def _composed_accums(self):
        return [x for x in self.__dict__ if isinstance(getattr(self, x, None), ABCAccum)]

    def reset(self):
        for accum in self._accums:
            setattr(self, accum, None)

    def verify_integrity(self) -> bool:
        accums = self._accums
        caccums = self._composed_accums

        if not accums:
            return True

        first_value = getattr(self, accums[0])
        first_is_none = first_value is None

        for accum in accums[1:]:
            value = getattr(self, accum)
            if (first_is_none and value is not None) or (not first_is_none and value is None):
                return False

        for caccum in caccums:
            if not getattr(self, caccum).verify_integrity():
                return False

        return True

    def forward(self, x):
        if not self.verify_integrity():
            raise RuntimeError("Accumulator is not in a valid state.")

        for caccum in self._composed_accums:
            getattr(self, caccum).forward(x)

        for accum in self._accums:
            getattr(self, f"{accum}_update")(x)

        return self.state()


class MeanCalculator(torch.nn.Module):
    """Incrementally computes the mean of batches of samples."""

    def __init__(self):
        super().__init__()
        self._samples_computed: int | None = None
        self._current_sum: torch.Tensor | None = None

    def reset(self):
        self._samples_computed = None
        self._current_sum = None

    def _verify_integrity(self) -> bool:
        return (self._samples_computed is None and self._current_sum is None) or (
            self._samples_computed is not None and self._current_sum is not None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._verify_integrity():
            raise RuntimeError("MeanCalculator is not in a valid state.")

        with torch.no_grad():
            n = x.shape[0]
            if self._current_sum is None:
                self._current_sum = torch.sum(x, axis=0)
                self._samples_computed = n
            else:
                self._current_sum += torch.sum(x, axis=0)
                self._samples_computed += n

        return self.state()

    def state(self) -> torch.Tensor:
        return self._current_sum / self._samples_computed


class CovCalculator(torch.nn.Module):
    """Incrementally computes the covariance matrix of batches of samples."""

    def __init__(self):
        super().__init__()
        self._mean_calculator = MeanCalculator()
        self._samples_computed: int | None = None
        self._cov_accum: torch.Tensor | None = None

    def reset(self):
        self._samples_computed = None
        self._cov_accum = None
        self._mean_calculator.reset()

    def _verify_integrity(self) -> bool:
        if not self._mean_calculator._verify_integrity():
            return False
        return (self._samples_computed is None and self._cov_accum is None) or (
            self._samples_computed is not None and self._cov_accum is not None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._verify_integrity():
            raise RuntimeError("CovCalculator is not in a valid state.")

        with torch.no_grad():
            self._mean_calculator(x)
            n = x.shape[0]

            if self._cov_accum is None:
                self._cov_accum = torch.matmul(x.mT, x)
                self._samples_computed = n
            else:
                self._cov_accum += torch.matmul(x.mT, x)
                self._samples_computed += n

        return self.state()

    def state(self) -> torch.Tensor:
        current_mean = self._mean_calculator.state().unsqueeze(0)
        mean_matrix = torch.matmul(current_mean.T, current_mean)
        current_cov = self._cov_accum - self._samples_computed * mean_matrix
        return current_cov / (self._samples_computed - 1)


class KLISCalculator(torch.nn.Module):
    """Computes KL divergence for Inception Score with optional softmax."""

    def __init__(self, eps: float = 1e-16, force_probability: bool = False):
        super().__init__()
        self._eps = eps
        self._samples_computed: int | None = None
        self._conditional_probability_accumulator: torch.Tensor | None = None
        self._total_probability_accumulator: torch.Tensor | None = None
        self._softmax_layer = torch.nn.Softmax(dim=1) if force_probability else None

    def reset(self):
        self._samples_computed = None
        self._conditional_probability_accumulator = None
        self._total_probability_accumulator = None

    def _verify_integrity(self) -> bool:
        return (
            self._samples_computed is None and self._conditional_probability_accumulator is None
        ) or (
            self._samples_computed is not None and self._conditional_probability_accumulator is not None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._verify_integrity():
            raise RuntimeError("KLCalculator is not in a valid state.")

        with torch.no_grad():
            x_is_positive = torch.all(x >= 0)

            if not x_is_positive and self._softmax_layer is not None:
                x = self._softmax_layer(x)
            if not x_is_positive and self._softmax_layer is None:
                raise ValueError("x must be positive or use `force_probability`=True")

            n = x.shape[0]
            x_log = torch.log(x + self._eps)

            conditional_accumulator = torch.sum(x * x_log, dim=0)
            total_accumulator = torch.sum(x, dim=0)

            if self._samples_computed is None:
                self._samples_computed = n
                self._conditional_probability_accumulator = conditional_accumulator
                self._total_probability_accumulator = total_accumulator
            else:
                self._samples_computed += n
                self._conditional_probability_accumulator += conditional_accumulator
                self._total_probability_accumulator += total_accumulator

        return self.state()

    def state(self, return_exp_expectation: bool = False) -> torch.Tensor:
        total_probability = self._total_probability_accumulator / self._samples_computed
        total_probability_log = torch.log(total_probability + self._eps)

        kl_difference = (
            self._conditional_probability_accumulator
            - self._total_probability_accumulator * total_probability_log
        )

        if return_exp_expectation:
            res = torch.sum(kl_difference) / self._samples_computed
            return torch.exp(res)

        return kl_difference
