"""
Distribution to distribution distances.

From a machine learning perspective, the distribution to distribution distance is from a group of
samples to another group of samples, and the distribution is the empiric distribution given by the
samples.

In the case of generative AI, the use case is to compare a dataset REAL to a dataset GENERATED.

The following distances are available:

- Frechet distance
- Wasserstein distance
- MMD (maximum mean discrepancy) distance

Note that you can use them as is. But also some metrics implementing the them already are available:

- Inception score
- Frechet inception distance
"""

from ciagen.qm import TL, to_numpy, to_tensor

from .frechet_distance import frechet_distance_gaussian_version
from .wasserstein_distance import (
    wasserstein_distance_gaussian_version,
    wasserstein_distance_multi_dimensional,
)


def cast_to(x: TL, to_type: str) -> TL:
    if to_type == "numpy":
        return to_numpy(x)
    elif to_type == "torch":
        return to_tensor(x)
    else:
        raise ValueError(f"Invalid to_type: {to_type}. Must be 'numpy' or 'torch'")
