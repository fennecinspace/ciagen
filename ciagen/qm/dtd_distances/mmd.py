"""
Several implementations of the Maximum Mean Discrepancy (MMD) distance:
https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions
"""

from typing import Callable
import numpy as np

from qmeasures import TL, to_numpy


def mmd_unbiased_estimator(
    x: TL, y: TL, kernel: Callable[[TL, TL], TL | float], to_float: bool = False
) -> TL | float:
    """
    MMD unbiased estimator: https://dl.acm.org/doi/10.5555/2188385.2188410
    """

    x, y = to_numpy(x), to_numpy(y)

    m = len(x)
    x_normalized = 1 / (m * (m - 1))

    n = len(y)
    y_normalized = 1 / (n * (n - 1))

    xy_normalized = 2 / (m * n)

    kxx = x_normalized * np.sum(
        [kernel(x[i], x[j]) for i in range(m - 1) for j in range(i + 1, m)]
    )
    kyy = y_normalized * np.sum(
        [kernel(y[i], y[j]) for i in range(n - 1) for j in range(i + 1, n)]
    )
    kxy = xy_normalized * np.sum(
        [kernel(x[i], y[j]) for i in range(m) for j in range(n)]
    )

    res = kxx + kyy - 2 * kxy

    if to_float:
        return float(res)

    return res
