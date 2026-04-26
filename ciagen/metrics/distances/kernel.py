from typing import Callable

import numpy as np
import torch

from ciagen.metrics import TL, cast_to


def simple_dot_kernel(x: TL, y: TL, to_float: bool = False, to_type: str = "torch") -> TL | float:
    """Dot product kernel."""
    x, y = cast_to(x, to_type), cast_to(y, to_type)

    if to_type == "numpy":
        res = np.dot(x, y)
    elif to_type == "torch":
        res = torch.dot(x, y)
    else:
        raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

    return float(res) if to_float else res


def rbf_kernel_generator(sigma: float, to_type: str = "torch") -> Callable[[TL, TL], TL | float]:
    """Create an RBF kernel with the given sigma."""

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
        x, y = cast_to(x, to_type), cast_to(y, to_type)
        if to_type == "numpy":
            res = np.exp(-(1 / (2 * sigma**2)) * np.linalg.norm(x - y, ord=2))
        elif to_type == "torch":
            res = torch.exp(-(1 / (2 * sigma**2)) * torch.linalg.norm(x - y, ord=2))
        else:
            raise ValueError(f"Invalid to_type: {to_type}")
        return float(res) if to_float else res

    return kernel


def rq_kernel_generator(alpha: float, to_type: str = "torch") -> Callable[[TL, TL], TL | float]:
    """Create a Rational Quadratic kernel."""

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
        x, y = cast_to(x, to_type), cast_to(y, to_type)
        res = (1 + (x - y) ** 2) ** (-alpha)
        return float(res) if to_float else res

    return kernel


def basic_polynomial_kernel_generator(d: int, exp: int, to_type: str = "torch") -> Callable[[TL, TL], TL | float]:
    """Create a polynomial kernel (used in KID).

    Reference: https://arxiv.org/pdf/1801.01401
    """

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
        x, y = cast_to(x, to_type), cast_to(y, to_type)
        if to_type == "numpy":
            res = ((1 / d) * np.dot(x, y)) ** exp
        elif to_type == "torch":
            res = ((1 / d) * torch.dot(x, y)) ** exp
        else:
            raise ValueError(f"Invalid to_type: {to_type}")
        return float(res) if to_float else res

    return kernel


def distance_induced_kernel_generator(
    z0: TL,
    rho_distance: Callable[[TL, TL], TL] | None = None,
    to_type: str = "torch",
) -> Callable[[TL, TL], TL | float]:
    """Create a distance-induced kernel.

    Reference: https://arxiv.org/abs/1207.6076
    """
    if rho_distance is None:
        if to_type == "numpy":

            def rho_distance(x, y):
                return np.linalg.norm(x, y, ord=2)
        elif to_type == "torch":

            def rho_distance(x, y):
                return torch.linalg.norm(x, y, ord=2)
        else:
            raise ValueError(f"Invalid to_type: {to_type}")

    def kernel(x: TL, y: TL, to_float: bool = False) -> TL | float:
        x, y = cast_to(x, to_type), cast_to(y, to_type)
        res = 0.5 * (rho_distance(x, z0) + rho_distance(y, z0) - rho_distance(x, y))
        return float(res) if to_float else res

    return kernel
