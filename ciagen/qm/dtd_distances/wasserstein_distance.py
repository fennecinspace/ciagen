"""
Implementation of the Wassersteing distance.

See On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests
(https://arxiv.org/pdf/1509.02237) for a in-depth discussion.

Or: https://en.wikipedia.org/wiki/Wasserstein_metric
"""

import scipy.linalg
import torch
import numpy as np
import scipy

from scipy.linalg import sqrtm as matrix_sqrt

from qmeasures import TL


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


def wasserstein_distance_gaussian_version(umean: TL, ucov: TL, vmean: TL, vcov: TL):
    if isinstance(umean, torch.Tensor):
        umean = umean.numpy()
    if isinstance(ucov, torch.Tensor):
        ucov = ucov.numpy()
    if isinstance(vmean, torch.Tensor):
        vmean = vmean.numpy()
    if isinstance(vcov, torch.Tensor):
        vcov = vcov.numpy()

    norm_square = np.linalg.norm(umean - vmean, ord=2) ** 2

    vvar_sqrt = matrix_sqrt(vcov)
    big_matmul = np.matmul(vvar_sqrt, np.matmul(ucov, vvar_sqrt))

    trace_part1 = np.trace(ucov + vcov)
    trace_part2 = -2 * np.trace(matrix_sqrt(big_matmul))

    return np.sqrt(norm_square + trace_part1 + trace_part2)
