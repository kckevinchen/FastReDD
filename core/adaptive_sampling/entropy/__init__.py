"""
Entropy-based Adaptive Sampling.

This submodule implements the entropy-based adaptive sampling strategy
that uses schema entropy tracking and stability streaks to determine
when to stop processing documents.
"""

from .sampler import AdaptiveSampler
from .document_selector import DocumentSelector

__all__ = [
    "AdaptiveSampler",
    "DocumentSelector",
]
