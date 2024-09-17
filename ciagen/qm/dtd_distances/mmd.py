"""
Several implementations of the Maximum Mean Discrepancy (MMD) distance:
https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions
"""

from typing import Callable

import numpy as np
import torch

from ciagen.qm import TL, to_numpy
from ciagen.qm.dtd_distances import cast_to


def mmd_unbiased_estimator(
    x: TL,
    y: TL,
    kernel: Callable[[TL, TL], TL | float],
    to_float: bool = False,
    to_type: str = "torch",
) -> TL | float:
    """
    MMD unbiased estimator: https://dl.acm.org/doi/10.5555/2188385.2188410
    """

    x, y = cast_to(x, to_type), cast_to(y, to_type)

    m = len(x)
    x_normalized = 1 / (m * (m - 1))

    n = len(y)
    y_normalized = 1 / (n * (n - 1))

    xy_normalized = 2 / (m * n)

    if to_type == "numpy":
        kxx = x_normalized * np.sum(
            [kernel(x[i], x[j]) for i in range(m - 1) for j in range(i + 1, m)]
        )
        kyy = y_normalized * np.sum(
            [kernel(y[i], y[j]) for i in range(n - 1) for j in range(i + 1, n)]
        )
        kxy = xy_normalized * np.sum(
            [kernel(x[i], y[j]) for i in range(m) for j in range(n)]
        )
    elif to_type == "torch":
        kxx = x_normalized * torch.sum(
            [kernel(x[i], x[j]) for i in range(m - 1) for j in range(i + 1, m)]
        )
        kyy = y_normalized * torch.sum(
            [kernel(y[i], y[j]) for i in range(n - 1) for j in range(i + 1, n)]
        )
        kxy = xy_normalized * torch.sum(
            [kernel(x[i], y[j]) for i in range(m) for j in range(n)]
        )
    else:
        raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

    res = kxx + kyy - 2 * kxy

    if to_float:
        return float(res)

    return res
