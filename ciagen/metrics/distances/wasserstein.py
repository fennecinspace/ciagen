import numpy as np
import scipy
import scipy.linalg
import torch
from scipy.linalg import sqrtm as matrix_sqrt

from ciagen.metrics import TL, cast_to


def wasserstein_distance_multi_dimensional(u: TL, v: TL, as_expectance: bool = True):
    """Compute Wasserstein distance per dimension."""
    if isinstance(u, torch.Tensor):
        u = u.numpy()
    if isinstance(v, torch.Tensor):
        v = v.numpy()

    if len(u.shape) == 1 and len(v.shape) == 1:
        return np.array([scipy.stats.wasserstein_distance(u, v)])

    if len(u.shape) == 2 and len(v.shape) == 2:
        if len(u) != len(v):
            raise ValueError(
                f"Both `u` and `v` must have the same length. Got {len(u)=} and {len(v)=}."
            )
        res = np.array(
            [scipy.stats.wasserstein_distance(u[i], v[i]) for i in range(len(u))]
        )
        if as_expectance:
            res = np.mean(res)
        return res

    raise ValueError(
        f"Both `u` and `v` must be 1D or 2D arrays. Got {u.shape=} and {v.shape=}."
    )


def wasserstein_distance_gaussian_version(
    umean: TL,
    ucov: TL,
    vmean: TL,
    vcov: TL,
    to_type: str = "numpy",
) -> TL:
    """Compute the Wasserstein distance between two Gaussian distributions.

    Reference: https://www.sciencedirect.com/science/article/pii/0024379582901124
    """
    umean, vmean = cast_to(umean, to_type), cast_to(vmean, to_type)
    ucov, vcov = cast_to(ucov, to_type), cast_to(vcov, to_type)

    if to_type == "numpy":
        norm_square = np.linalg.norm(umean - vmean, ord=2) ** 2
        vvar_sqrt = matrix_sqrt(vcov)
        big_matmul = np.matmul(vvar_sqrt, np.matmul(ucov, vvar_sqrt))

        trace_part1 = np.trace(ucov + vcov)
        trace_part2 = -2 * np.trace(matrix_sqrt(big_matmul))

        res = np.sqrt(norm_square + trace_part1 + trace_part2)
    elif to_type == "torch":
        norm_square = torch.linalg.norm(umean - vmean, ord=2) ** 2
        vvar_sqrt = torch.linalg.matrix_power(vcov, 0.5)
        big_matmul = torch.matmul(vvar_sqrt, torch.matmul(ucov, vvar_sqrt))

        trace_part1 = torch.trace(ucov + vcov)
        trace_part2 = -2 * torch.trace(matrix_sqrt(big_matmul))

        res = torch.sqrt(norm_square + trace_part1 + trace_part2)
    else:
        raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

    return res
