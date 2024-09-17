# ciagen.qm.ptd_distances

import numpy as np
import torch

from ciagen.qm import TL, to_numpy
from ciagen.qm.dtd_distances import cast_to


def mahalanobis_distance_calc(
    x: TL,
    mean: TL,
    cov: TL | None = None,
    inv_cov: TL | None = None,
    to_float: bool = False,
    to_type: str = "numpy",
) -> TL | float:
    """Implementation of the mahalanobis distance: https://en.wikipedia.org/wiki/Mahalanobis_distance"""
    if cov is None and inv_cov is None:
        raise ValueError("Either cov or inv_cov must be provided")

    x, mean = cast_to(x, to_type), cast_to(mean, to_type)

    if inv_cov is None:
        cov = to_numpy(cov)
        if to_type == "torch":
            inv_cov = torch.linalg.pinv(cov, hermitian=True)
        elif to_type == "numpy":
            inv_cov = np.linalg.pinv(cov, hermitian=True)
        else:
            raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")
    else:
        inv_cov = cast_to(inv_cov, to_type)

    if to_type == "torch":
        res = torch.matmul(x - mean, inv_cov)
        res = torch.matmul(res, x - mean)
    elif to_type == "numpy":
        res = np.matmul(x - mean, inv_cov)
        res = np.matmul(res, x - mean)
    else:
        raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

    if to_float:
        res = float(res)

    return res
