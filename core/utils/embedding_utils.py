"""Embedding utilities for query-aware chunk filtering.

This module provides embedding services for computing dense representations
of queries and documents, enabling similarity-based filtering with conformal
guarantees.

Design Philosophy
-----------------
The embedding service abstracts the underlying embedding API (OpenAI, etc.) 
and provides a consistent interface for computing embeddings and similarities.

Key Features:
- Support for multiple embedding providers (OpenAI, etc.)
- Single document embedding for stream processing
- Cosine similarity computation (query vs document)
- Memory efficient for large datasets

Example Usage:
    ```python
    from core.utils.embedding_utils import EmbedderOpenAI
    
    # Initialize embedder
    embedder = EmbedderOpenAI(model="text-embedding-3-small")
    
    # Embed query
    query_emb = embedder.embed_query("Find all albums by Metallica")
    
    # Embed a document
    doc_emb = embedder.embed_document("Album: Master of Puppets")
    
    # Compute similarity
    sim = embedder.cosine_similarity(query_emb, doc_emb)
    print(f"Similarity: {sim:.4f}")
    ```
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple

import numpy as np
from openai import OpenAI

__all__ = [
    "EmbedderBase",
    "EmbedderOpenAI",
    "cosine_similarity",
]


def cosine_similarity(
    query_embedding: Union[List[float], np.ndarray],
    doc_embedding: Union[List[float], np.ndarray]
) -> float:
    """Compute cosine similarity between a query embedding and a document embedding."""
    query_emb = np.array(query_embedding)
    doc_emb = np.array(doc_embedding)
    
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norm = doc_emb / np.linalg.norm(doc_emb)
    
    similarity = np.dot(query_norm, doc_norm)
    return float(similarity)


class EmbedderBase(ABC):
    def __init__(self, model: str):
        self.model = model
        self.embedding_dim = None  # Should be set by subclass
        logging.info(f"[{self.__class__.__name__}:__init__] Initialized with model: {model}")
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        pass
    
    @abstractmethod
    def embed_document(self, document: str) -> List[float]:
        pass
    
    @staticmethod
    def cosine_similarity(query_embedding, doc_embedding) -> float:
        return cosine_similarity(query_embedding, doc_embedding)
 

class EmbedderOpenAI(EmbedderBase):
    """OpenAI embedding service.
    
    Supports OpenAI's text embedding models:
    - text-embedding-3-small (1536 dims, $0.020/1M tokens)
    - text-embedding-3-large (3072 dims, $0.130/1M tokens)
    - text-embedding-ada-002 (1536 dims, $0.100/1M tokens)
    
    Args:
        model: OpenAI embedding model name
        api_key: OpenAI API key (optional, defaults to OPENAI_API_KEY env var)
    
    Example:
        >>> embedder = EmbedderOpenAI(model="text-embedding-3-small")
        >>> query_emb = embedder.embed_query("Find albums")
        >>> doc_emb = embedder.embed_document("Album by Metallica")
        >>> sim = embedder.cosine_similarity(query_emb, doc_emb)
        >>> print(f"{sim:.4f}")
    """
    
    # Model dimensions (for reference)
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        super().__init__(model)
        
        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"OpenAI API key is required. Provide via api_key parameter or "
                         f"OPENAI_API_KEY environment variable.")
            raise ValueError(
                "OpenAI API key is required. Provide via api_key parameter or "
                "OPENAI_API_KEY environment variable."
            )
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key)
        
        # Set embedding dimension
        self.embedding_dim = self.MODEL_DIMS.get(model)
        if self.embedding_dim is None:
            logging.warning(
                f"[{self.__class__.__name__}:__init__] Unknown model {model}, "
                "embedding_dim will be inferred from first embedding"
            )
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query using OpenAI API.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            input=[query],
            model=self.model
        )
        
        embedding = response.data[0].embedding
        
        # Infer dimension if not set
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
            logging.info(
                f"[{self.__class__.__name__}:embed_query] "
                f"Inferred embedding_dim={self.embedding_dim}"
            )
        
        return embedding
    
    def embed_document(self, document: str) -> List[float]:
        """Embed a single document using OpenAI API.
        
        Args:
            document: Document text
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            input=[document],
            model=self.model
        )
        
        embedding = response.data[0].embedding
        
        # Infer dimension if not set
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
            logging.info(
                f"[{self.__class__.__name__}:embed_document] "
                f"Inferred embedding_dim={self.embedding_dim}"
            )
        
        return embedding

