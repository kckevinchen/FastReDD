"""
Data Population Module

This module contains classes and utilities for populating database tables
with data generated from unstructured documents using LLMs.

Architecture:
- DataPopulator (base.py): Abstract base class defining the interface
- DataPop (datapop.py): Unified API-based implementation (cgpt, deepseek, together, siliconflow, gemini)
- DataPopLocal (datapop_local.py): Local GPU-based implementation with hidden states extraction

Legacy classes (deprecated, will be removed):
- DataPopBasic, DataPopAPI, DataPopDeepSeek, DataPopGPT, DataPopTogether, 
  DataPopSiliconFlow, DataPopGemini, DataPopUnified
"""

# New unified architecture
from .base import DataPopulator
from .datapop import DataPop
from .datapop_local import DataPopLocal

__all__ = [
    # New unified classes
    "DataPopulator",  # Abstract base class
    "DataPop",        # Unified API implementation (supports: cgpt, deepseek, together, siliconflow, gemini)
    "DataPopLocal",   # Local GPU implementation
]
