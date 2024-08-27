"""
Metrics in this module are meant to be used with between two datasets, not sample vs dataset.
In the case of generative AI they are used to compare a dataset REAL to a dataset GENERATED and thus
measure the quality of the genertive model/method.

They (hopefully) measure how near the GENERATED dataset is to the REAL one.
The current list of metrics is:

- Inception score
- Frechet inception distance
"""

from .inception_score import IS
from .frechet_inception_distance import FID
