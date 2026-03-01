"""
Schema Generation Module.

This module contains classes and utilities for generating database schemas
from unstructured documents using LLMs.
"""

from .base import SchemaGenerator
from .schemagen import SchemaGen

# For backward compatibility
SchemaGenUnified = SchemaGen
SchemaGenGPT = SchemaGen
SchemaGenBasic = SchemaGenerator

__all__ = [
    "SchemaGenerator",
    "SchemaGen",
    "SchemaGenUnified",  # Backward compatibility
    "SchemaGenGPT",       # Backward compatibility
    "SchemaGenBasic",     # Backward compatibility
]
