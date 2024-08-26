from typing import List
import torch
import numpy as np

from abc import abstractmethod, ABC

from ciagen.qm import TL


class DistributionScore(ABC):

    def __init__(
        self,
        p_initial_values: TL | None = None,
        q_initial_values: TL | None = None,
        **kwargs,
    ):

        self.pvalues = None
        self.qvalues = None

        self.update(p_initial_values, q_initial_values)

    @abstractmethod
    def update(pvalues: TL | None, qvalues: TL | None, **kwargs): ...

    def score(self, t: str = "numpy", **kwargs) -> TL | float:
        allowed_t = ("numpy", "float", "torch")
        if t not in allowed_t:
            raise ValueError(f"Type `t` unkown: {t}")

        self._score(self, t, kwargs)

    @abstractmethod
    def _score(self, t: str = "numpy", **kwargs) -> TL | float: ...

    def instant_score(
        self, pvalues: TL, qvalues: TL, t: str = "numpy", **kwargs
    ) -> TL | float:
        allowed_t = ("numpy", "float", "torch")
        if t not in allowed_t:
            raise ValueError(f"Type `t` unkown: {t}")

        self._instant_score(pvalues, qvalues, t, kwargs)

    @abstractmethod
    def _instant_score(
        self, pvalues: TL, qvalues: TL, t: str = "numpy", **kwargs
    ) -> TL | float: ...


# TODO: implement the transformation to torch, numpy and float
class KLDistributionScore(DistributionScore):
    def __init__(
        self,
        p_initial_values: torch.Tensor | np.ndarray | None = None,
        q_initial_values: torch.Tensor | np.ndarray | None = None,
        **kwargs,
    ):
        self.p_nbr_samples = (
            len(p_initial_values) if p_initial_values is not None else 0
        )
        self.q_nbr_samples = (
            len(q_initial_values) if q_initial_values is not None else 0
        )

        self.kl_accum = 0

        super().__init__(p_initial_values, q_initial_values, **kwargs)

    def _instant_score(
        self,
        pvalues: torch.Tensor | np.ndarray,
        qvalues: torch.Tensor | np.ndarray,
        t: str = "numpy",
        **kwargs,
    ) -> torch.Tensor | np.ndarray | float:
        if isinstance(pvalues, torch.Tensor):
            pvalues = pvalues.numpy()

        if isinstance(qvalues, torch.Tensor):
            qvalues = qvalues.numpy()

        as_expectance = kwargs.get("as_expectance", True)
        eps = kwargs.get("eps", 1e-16)

        res = kl_divergence(pvalues, qvalues, as_expectance=as_expectance, eps=eps)
        return res

    def update(
        self,
        pvalues: torch.Tensor | np.ndarray | None,
        qvalues: torch.Tensor | np.ndarray | None,
        **kwargs,
    ):
        self.kl_accum += kl_divergence(pvalues, qvalues, return_raw=True)
        self.p_nbr_samples += len(pvalues)
        self.q_nbr_samples += len(qvalues)

    def _score(self, t: str = "numpy", **kwargs) -> torch.Tensor | np.ndarray | float:
        is_expectance = kwargs.get("is_expectance", True)
        values = self.kl_accum

        if is_expectance:
            values = np.sum(values, axis=1)
            values = np.mean(values)
        else:
            values = np.sum(values)

        if t == "numpy":
            return values
        elif t == "torch":
            return torch.tensor(values)
        elif t == "float":
            return float(values)


def clog(x: TL, eps: float = 1e-16):
    v = x + eps
    return np.log(v) if isinstance(v, np.ndarray) else torch.log(v)


def kl_divergence(
    p: torch.Tensor | np.ndarray,
    q: torch.Tensor | np.ndarray,
    as_expectance: bool = False,
    eps: float = 1e-16,
    return_raw: bool = False,
):
    """Kullback-Leibler divergence"""

    t = "numpy" if isinstance(p, np.ndarray) else "torch"

    if isinstance(p, torch.Tensor):
        p = p.numpy()
    if isinstance(q, torch.Tensor):
        q = q.numpy()

    if len(p.shape) > 2:
        raise ValueError(
            "Can only compute for a vector of probabilities of a full distribution, "
            f"found unknown shape: {p.shape=}"
        )

    if p.shape != q.shape and q.shape != (1, p.shape[1]) and q.shape != (p.shape[1],):
        raise ValueError(
            "Could not find a strategy to compute the divergence due to mismatched "
            f"shapes: {p.shape=}, {q.shape=}"
        )

    values = p * (clog(p, eps) - clog(q, eps))
    if return_raw:
        return values

    if as_expectance:
        values = np.sum(values, axis=1)
        values = np.mean(values)
    else:
        values = np.sum(values)

    values = float(values)

    if t == "torch":
        if isinstance(values, float):
            values = torch.Tensor([values])
        else:
            values = torch.Tensor(values)

    return values


def js_divergence(
    ps: List[torch.Tensor | np.ndarray],
    weights: List[float] | None = None,
    as_float: bool = True,
    normalize: bool = False,
    as_metric: bool = False,
    as_expectance: bool = False,
    eps: float = 1e-16,
):
    """Jense-Shannon divergence"""

    nbr_p = len(ps)

    ps = np.array([(p if isinstance(p, np.ndarray) else p.numpy) for p in ps])

    if weights is None:
        weights = (1 / nbr_p) * np.ones(len(nbr_p))

    weights = np.array(weights) if not isinstance(weights, np.ndarray) else weights
    if np.sum(weights) != 1:
        raise ValueError("Weights must sum to 1.")

    m = np.average(ps, weights=weights)
    kl_divs = [kl_divergence(p, m, as_expectance=as_expectance, eps=eps) for p in ps]

    res = np.average(kl_divs, weights=weights)

    # Normalization and value metric should be exclusive as there is no interpretability for
    # both at the same time.
    if normalize:
        constant = np.log(nbr_p)
        res = res / constant
    elif as_metric:
        res = np.sqrt(res)

    if as_float:
        res = float(res)

    return res
