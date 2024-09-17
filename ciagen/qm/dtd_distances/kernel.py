from typing import Callable

import numpy as np
import torch

from ciagen.qm import TL, to_numpy, to_tensor
from ciagen.qm.dtd_distances import cast_to


def simple_dot_kernel(
    x: TL, y: TL, to_float: bool = False, to_type: str = "torch"
) -> TL | float:
    """
    Implementation of the dot product between two vectors as a kernel.
    """
    x, y = cast_to(x, to_type), cast_to(y, to_type)

    if to_type == "numpy":
        res = np.dot(x, y)
    elif to_type == "torch":
        res = torch.dot(x, y)
    else:
        raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

    if to_float:
        return float(res)

    return res


def rbf_kernel_generator(
    sigma: float, to_type: str = "torch"
) -> Callable[[TL, TL], TL | float]:
    """
    Implementation of the RBF kernel: https://en.wikipedia.org/wiki/Radial_basis_function
    """

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
        x, y = cast_to(x, to_type), cast_to(y, to_type)

        if to_type == "numpy":
            res = np.exp(-(1 / (2 * sigma**2)) * np.linalg.norm(x - y, ord=2))
        elif to_type == "torch":
            res = torch.exp(-(1 / (2 * sigma**2)) * torch.linalg.norm(x - y, ord=2))
        else:
            raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

        if to_float:
            return float(res)

        return res

    return kernel


def rq_kernel_generator(
    alpha: float, to_type: str = "torch"
) -> Callable[[TL, TL], TL | float]:
    """
    Implementation of the Rational Quadratic kernel: https://en.wikipedia.org/wiki/Rational_quadratic_covariance_function
    """

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
        x, y = cast_to(x, to_type), cast_to(y, to_type)

        res = (1 + (x - y) ** 2) ** (-alpha)

        if to_float:
            return float(res)

        return res

    return kernel


def basic_polynomial_kernel_generator(
    d: int, exp: int, to_type: str = "torch"
) -> TL | float:
    """
    Implementation of a basic polynomial kernel used originally in KID: https://arxiv.org/pdf/1801.01401
    """

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:

        x, y = cast_to(x, to_type), cast_to(y, to_type)
        if to_type == "numpy":
            res = ((1 / d) * np.dot(x, y)) ** exp
        elif to_type == "torch":
            res = ((1 / d) * torch.dot(x, y)) ** exp
        else:
            raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

        if to_float:
            return float(res)

        return res

    return kernel


def distance_induced_kernel_generator(
    z0: TL,
    rho_distance: Callable[[TL, TL], TL] | None = None,
    to_type: str = "torch",
):
    """
    Implementation of the distance induced kernel: https://arxiv.org/abs/1207.6076

    Note that this is a kernel generator, not a kernel. It depends on the rho_distance and fixed
    value z0.
    """

    if rho_distance is None:
        if to_type == "numpy":
            rho_distance = lambda x, y: np.linalg.norm(x, y, ord=2)
        elif to_type == "torch":
            rho_distance = lambda x, y: torch.linalg.norm(x, y, ord=2)
        else:
            raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
        x, y = cast_to(x, to_type), cast_to(y, to_type)

        res = (1 / 2) * (rho_distance(x, z0) + rho_distance(y, z0) - rho_distance(x, y))

        if to_float:
            return float(res)

        return res

    return kernel
