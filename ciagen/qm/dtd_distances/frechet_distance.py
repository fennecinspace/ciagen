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
from scipy.linalg import sqrtm as matrix_sqrt

from ciagen.qm import TL
from ciagen.qm.dtd_distances import cast_to


def frechet_distance_gaussian_version(
    umean: TL, ucov: TL, vmean: TL, vcov: TL, to_type="numpy"
):
    umean, vmean, ucov, vcov = (
        cast_to(umean, to_type),
        cast_to(vmean, to_type),
        cast_to(ucov, to_type),
    )

    if to_type == "torch":
        norm_square = torch.linalg.norm(umean - vmean, ord=2) ** 2
        covar_matmul = torch.matmul(ucov, vcov)
        covar_matmul = torch.matmul(ucov, vcov)

        trace_part1 = torch.trace(ucov + vcov)
        trace_part2 = -2 * torch.trace(matrix_sqrt(covar_matmul))

        res = torch.sqrt(norm_square + trace_part1 + trace_part2)
    elif to_type == "numpy":
        norm_square = np.linalg.norm(umean - vmean, ord=2) ** 2
        covar_matmul = np.matmul(ucov, vcov)
        covar_matmul = np.matmul(ucov, vcov)

        trace_part1 = np.trace(ucov + vcov)
        trace_part2 = -2 * np.trace(matrix_sqrt(covar_matmul))

        res = np.sqrt(norm_square + trace_part1 + trace_part2)
    else:
        raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")

    return res
