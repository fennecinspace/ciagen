# ciagen.qm.ptd_distances

import numpy as np
import torch

from ciagen.qm import TL, to_numpy
from ciagen.qm import cast_to


def mahalanobis_distance_calc(
    x: TL,
    mean: TL,
    cov: TL | None = None,
    inv_cov: TL | None = None,
    to_float: bool = False,
    to_type: str = "numpy",
    distance_squared: bool = False,
) -> TL | float:
    """Implementation of the mahalanobis distance: https://en.wikipedia.org/wiki/Mahalanobis_distance"""

    if to_type not in ["numpy", "torch"]:
        raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

    if cov is None and inv_cov is None:
        raise ValueError("Either cov or inv_cov must be provided")

    x, mean, cov = cast_to(x, to_type), cast_to(mean, to_type), cast_to(cov, to_type)

    if to_type == "torch":
        is_batch = len(x.size()) > 1
    else:
        is_batch = len(x.shape) > 1

    if inv_cov is None:
        if to_type == "torch":
            inv_cov = torch.linalg.pinv(cov, hermitian=True)
        else:
            inv_cov = np.linalg.pinv(cov, hermitian=True)
    else:
        inv_cov = cast_to(inv_cov, to_type)

    diff = x - mean
    diff_T = diff.T if is_batch else diff

    if to_type == "torch":

        res = torch.matmul(diff, inv_cov)
        res = torch.matmul(res, diff_T)

        if is_batch:
            res = res.diag()

        if not distance_squared:
            res = torch.sqrt(res)
    else:
        res = np.matmul(diff, inv_cov)
        res = np.matmul(res, diff_T)

        if is_batch:
            res = res.diagonal()

        if not distance_squared:
            res = np.sqrt(res)

    if to_float:
        res = float(res)

    return res
