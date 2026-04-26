from .frechet import frechet_distance_gaussian_version
from .wasserstein import (
    wasserstein_distance_gaussian_version,
    wasserstein_distance_multi_dimensional,
)

__all__ = [
    "frechet_distance_gaussian_version",
    "wasserstein_distance_gaussian_version",
    "wasserstein_distance_multi_dimensional",
]
