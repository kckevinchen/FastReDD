"""
ReDD: Relational Deep Dive - Error-Aware Queries Over Unstructured Data

A research project focused on:
- Schema generation from unstructured documents
- Data population using LLMs  
- Error-aware query processing over unstructured data
- Integration with various LLM providers (OpenAI GPT, DeepSeek, TogetherAI, SiliconFlow, Local Models)
"""

__version__ = "1.0.0"
__author__ = "ReDD Team"
__description__ = "Relational Deep Dive: Error-Aware Queries Over Unstructured Data"

# Import main components for easy access
from core import *
from dataset import *

__all__ = []
