"""
Note that the frechet distance and wasserstein distance are NOT the same, somewhat related and
sometimes giving the same results nevertheless.

Frechet distance: https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance
Wasserstein distance: https://en.wikipedia.org/wiki/Wasserstein_metric

Frechet distance in gaussian case: https://www.sciencedirect.com/science/article/pii/0047259X8290077X?ref=pdf_download&fr=RR-2&rr=8a41fb45cf10a596
Wasserstein distance in gaussian case: https://www.sciencedirect.com/science/article/pii/0024379582901124?ref=pdf_download&fr=RR-2&rr=8a41fc48fc2ea596
"""

import torch
import numpy as np

from scipy.linalg import sqrtm as matrix_sqrt

from qmeasures import TL


def frechet_distance_gaussian_version(umean: TL, ucov: TL, vmean: TL, vcov: TL):
    if isinstance(umean, torch.Tensor):
        umean = umean.numpy()
    if isinstance(ucov, torch.Tensor):
        ucov = ucov.numpy()
    if isinstance(vmean, torch.Tensor):
        vmean = vmean.numpy()
    if isinstance(vcov, torch.Tensor):
        vcov = vcov.numpy()

    norm_square = np.linalg.norm(umean - vmean, ord=2) ** 2
    covar_matmul = np.matmul(ucov, vcov)
    covar_matmul = np.dot(ucov, vcov)

    trace_part1 = np.trace(ucov + vcov)
    trace_part2 = -2 * np.trace(matrix_sqrt(covar_matmul))

    return np.sqrt(norm_square + trace_part1 + trace_part2)
