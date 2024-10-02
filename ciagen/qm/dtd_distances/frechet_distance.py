"""
Note that the frechet distance and wasserstein distance are NOT the same, somewhat related and
sometimes giving the same results nevertheless.

Frechet distance: https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance
Wasserstein distance: https://en.wikipedia.org/wiki/Wasserstein_metric

Frechet distance in gaussian case: https://www.sciencedirect.com/science/article/pii/0047259X8290077X?ref=pdf_download&fr=RR-2&rr=8a41fb45cf10a596
Wasserstein distance in gaussian case: https://www.sciencedirect.com/science/article/pii/0024379582901124?ref=pdf_download&fr=RR-2&rr=8a41fc48fc2ea596
"""

import numpy as np
import torch
from scipy.linalg import sqrtm

from ciagen.qm import TL
from ciagen.qm import cast_to


def matrix_sqrt(x: TL, to_type="numpy") -> TL:
    y = cast_to(x, to_type="numpy")
    y = sqrtm(y)

    return cast_to(y, to_type)


def frechet_distance_gaussian_version(
    umean: TL, ucov: TL, vmean: TL, vcov: TL, to_type="numpy"
):

    if to_type == "torch":
        norm_square = torch.linalg.norm(umean - vmean, ord=2) ** 2
        covar_matmul = torch.matmul(ucov, vcov)
        covar_matmul = torch.matmul(ucov, vcov)

        trace_part1 = torch.trace(ucov + vcov)
        trace_part2 = -2 * torch.trace(matrix_sqrt(covar_matmul, to_type=to_type))

        res = torch.sqrt(norm_square + trace_part1 + trace_part2)
    elif to_type == "numpy":
        norm_square = np.linalg.norm(umean - vmean, ord=2) ** 2
        covar_matmul = np.matmul(ucov, vcov)
        covar_matmul = np.matmul(ucov, vcov)

        trace_part1 = np.trace(ucov + vcov)
        trace_part2 = -2 * np.trace(matrix_sqrt(covar_matmul, to_type=to_type))

        res = np.sqrt(norm_square + trace_part1 + trace_part2)
    else:
        raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

    return res
