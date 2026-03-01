"""
DDGT (Diversity-Driven Good-Turing) Adaptive Sampling.

This submodule implements the DDGT adaptive sampling strategy
that uses diversity-driven document selection and Good-Turing stopping
condition for probabilistic coverage guarantees.
"""

from .sampler import DDGTSampler
from .document_selector import DDGTDocumentSelector

__all__ = [
    "DDGTSampler",
    "DDGTDocumentSelector",
]
