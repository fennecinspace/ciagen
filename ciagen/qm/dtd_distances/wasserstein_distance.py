"""
Implementation of the Wassersteing distance.

See On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests
(https://arxiv.org/pdf/1509.02237) for a in-depth discussion.

Or: https://en.wikipedia.org/wiki/Wasserstein_metric
"""

import numpy as np
import scipy
import scipy.linalg
import torch
from scipy.linalg import sqrtm as matrix_sqrt

from ciagen.qm import TL
from ciagen.qm.dtd_distances import cast_to


def wasserstein_distance_multi_dimensional(u: TL, v: TL, as_expectance=True):
    if isinstance(u, torch.Tensor):
        u = u.numpy()
    if isinstance(v, torch.Tensor):
        v = v.numpy()

    if len(u.shape) == 1 and len(v.shape) == 1:
        return np.array([scipy.stats.wasserstein_distance(u, v)])
    if len(u.shape) == 2 and len(v.shape) == 2:
        if len(u) != len(v):
            raise ValueError(
                f"Both `u` and `v` must have the same length. Found {len(u)=} and {len(v)=}."
            )
        res = np.array(
            [scipy.stats.wasserstein_distance(u[i], v[i]) for i in range(len(u))]
        )
        if as_expectance:
            res = np.mean(res)
        return res
    else:
        raise ValueError(
            f"Both `u` and `v` must be 1D or 2D arrays. Found {u.shape=} and {v.shape=}."
        )


def wasserstein_distance_gaussian_version(
    umean: TL, ucov: TL, vmean: TL, vcov: TL, to_type="numpy"
):
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

    return res
