import numpy as np

from ciagen.qm import TL, to_numpy


def mahalanobis_distance(
    x: TL,
    mean: TL,
    cov: TL | None = None,
    inv_cov: TL | None = None,
    to_float: bool = False,
) -> TL | float:
    """Implementation of the mahalanobis distance: https://en.wikipedia.org/wiki/Mahalanobis_distance"""
    if cov is None and inv_cov is None:
        raise ValueError("Either cov or inv_cov must be provided")

    x, mean = to_numpy(x), to_numpy(mean)

    if inv_cov is None:
        cov = to_numpy(cov)
        # `slogdet` returns logdet and sign, the sign is zero if the determinant is zero, this
        # routine is more robust to overflow and underflow issues
        cov_is_invertible = np.linalg.slogdet(cov)[0] != 0

        if cov_is_invertible:
            inv_cov = np.linalg.inv(cov)
        else:
            inv_cov = np.linalg.pinv(cov)
    else:
        inv_cov = to_numpy(inv_cov)

    res = np.matmul(x - mean, inv_cov)
    res = np.matmul(res, x - mean)

    if to_float:
        res = float(res)

    return res
