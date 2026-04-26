"""
CIA (Controllable Image Augmentation) Framework.

A data generation framework using Stable Diffusion + ControlNet to perform
synthetic data augmentation for downstream ML tasks.
"""

from ciagen.api.caption import caption
from ciagen.api.evaluate import evaluate
from ciagen.api.filter import filter_generated
from ciagen.api.generate import generate

__all__ = [
    "generate",
    "evaluate",
    "filter_generated",
    "caption",
]
