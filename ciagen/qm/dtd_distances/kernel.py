from typing import Callable
import numpy as np

from qmeasures import TL, to_numpy


def simple_dot_kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
    """
    Implementation of the dot product between two vectors as a kernel.
    """
    x, y = to_numpy(x), to_numpy(y)
    res = np.dot(x, y)

    if to_float:
        return float(res)

    return res


def rbf_kernel_generator(sigma: float) -> Callable[[TL, TL], TL | float]:
    """
    Implementation of the RBF kernel: https://en.wikipedia.org/wiki/Radial_basis_function
    """

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
        x, y = to_numpy(x), to_numpy(y)

        res = np.exp(-(1 / (2 * sigma**2)) * np.linalg.norm(x - y, ord=2))

        if to_float:
            return float(res)

        return res

    return kernel


def rq_kernel_generator(alpha: float) -> Callable[[TL, TL], TL | float]:
    """
    Implementation of the Rational Quadratic kernel: https://en.wikipedia.org/wiki/Rational_quadratic_covariance_function
    """

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
        x, y = to_numpy(x), to_numpy(y)

        res = (1 + (x - y) ** 2) ** (-alpha)

        if to_float:
            return float(res)

        return res

    return kernel


def basic_polynomial_kernel_generator(d: int, exp: int) -> TL | float:
    """
    Implementation of a basic polynomial kernel used originally in KID: https://arxiv.org/pdf/1801.01401
    """

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:

        x, y = to_numpy(x), to_numpy(y)
        res = ((1 / d) * np.dot(x, y)) ** exp

        if to_float:
            return float(res)

        return res

    return kernel


def distance_induced_kernel_generator(
    z0: TL,
    rho_distance: Callable[[TL, TL], TL] | None = None,
):
    """
    Implementation of the distance induced kernel: https://arxiv.org/abs/1207.6076

    Note that this is a kernel generator, not a kernel. It depends on the rho_distance and fixed
    value z0.
    """

    if rho_distance is None:
        rho_distance = lambda x, y: np.linalg.norm(x, y, ord=2)

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
        x, y = to_numpy(x), to_numpy(y)

        res = (1 / 2) * (rho_distance(x, z0) + rho_distance(y, z0) - rho_distance(x, y))

        if to_float:
            return float(res)

        return res

    return kernel
