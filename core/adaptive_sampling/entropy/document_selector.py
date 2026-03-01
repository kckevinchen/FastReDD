"""
Document Selector for Adaptive Sampling.

This module implements efficient document selection using pre-generated embeddings.
Uses farthest-from-mean strategy with an indexed embedding structure for O(1) lookups.
All embeddings must be pre-generated (e.g., in notebook PART 0.5).
"""

import logging
import random
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set

import numpy as np


class DocumentSelector:
    """
    Efficiently selects documents for processing based on embedding similarity.
    Uses pre-generated embeddings with an indexed structure for fast lookups.
    Implements incremental farthest-from-mean selection during adaptive sampling.
    """
    
    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        """
        Initialize the document selector.
        
        Args:
            config: Configuration dictionary containing adaptive_sampling settings
            api_key: API key (not used, embeddings are pre-generated, kept for compatibility)
        """
        adaptive_config = config.get("adaptive_sampling", {})
        self.enabled = adaptive_config.get("use_embedding_selection", True)
        self.embedding_model = adaptive_config.get("embedding_model", "text-embedding-3-small")
        
        # Path to load pre-generated embeddings
        self.embedding_file = adaptive_config.get("embedding_file")
        
        # Embedding index structure for efficient lookups
        # embedding_matrix: (N, D) numpy array of all embeddings
        # doc_id_to_idx: dict mapping doc_id (str) -> index in embedding_matrix
        # idx_to_doc_id: dict mapping index -> doc_id (str) for reverse lookup
        self.embedding_matrix = None  # np.ndarray of shape (N, D)
        self.doc_id_to_idx = {}  # doc_id -> row index in matrix
        self.idx_to_doc_id = {}  # row index -> doc_id
        self.embedding_dim = None
        
        self._index_built = False
        
        if self.enabled:
            logging.info(f"[{self.__class__.__name__}:__init__] Initialized with model: {self.embedding_model}")

    def load_embeddings(self, path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load embeddings from JSON file.
        
        Args:
            path: Path to JSON file containing embeddings
            
        Returns:
            Dictionary mapping doc_id to embedding vector
        """
        path = Path(path)
        if not path.exists():
            logging.warning(f"[{self.__class__.__name__}:load_embeddings] Embedding file not found: {path}")
            return {}
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            embeddings = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
            logging.info(f"[{self.__class__.__name__}:load_embeddings] Loaded {len(embeddings)} embeddings from {path}")
            return embeddings
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}:load_embeddings] Failed to load embeddings from {path}: {e}")
            return {}

    def build_index(self, doc_dict: Dict[str, Any]) -> bool:
        """
        Build an efficient index structure from pre-generated embeddings.
        Must be called before using incremental selection.
        
        Args:
            doc_dict: Dictionary of documents {doc_id: [doc_text, ...]}
                     Used to get list of all document IDs
            
        Returns:
            True if index built successfully, False otherwise
        """
        if not self.enabled:
            return False
            
        if self._index_built:
            logging.debug(f"[{self.__class__.__name__}:build_index] Index already built, skipping")
            return True
        
        # Load embeddings from cache
        cache_path = None
        if self.embedding_file:
            cache_path = Path(self.embedding_file)
        else:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] No embedding_file specified in config. "
                f"Embeddings must be pre-generated. Falling back to random selection."
            )
            return False
        
        embeddings_dict = self.load_embeddings(cache_path)
        
        if not embeddings_dict:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] No embeddings loaded. "
                f"Embeddings should be pre-generated (e.g., in notebook PART 0.5). "
                f"Falling back to random selection."
            )
            return False
        
        # Filter embeddings to only include documents in doc_dict
        all_doc_ids = set(doc_dict.keys())
        filtered_embeddings = {did: emb for did, emb in embeddings_dict.items() if did in all_doc_ids}
        
        if not filtered_embeddings:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] No matching embeddings found for documents in doc_dict. "
                f"Expected {len(all_doc_ids)} documents, found {len(embeddings_dict)} embeddings."
            )
            return False
        
        missing_ids = all_doc_ids - set(filtered_embeddings.keys())
        if missing_ids:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] {len(missing_ids)}/{len(all_doc_ids)} documents missing embeddings. "
                f"Will use random selection for those documents."
            )
        
        # Build index structure: create matrix and mappings
        doc_ids = list(filtered_embeddings.keys())
        embeddings_list = [filtered_embeddings[did] for did in doc_ids]
        
        # Stack into matrix: shape (N, D) where N=num_docs, D=embedding_dim
        self.embedding_matrix = np.stack(embeddings_list, axis=0).astype(np.float32)
        self.embedding_dim = self.embedding_matrix.shape[1]
        
        # Build bidirectional mappings
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.idx_to_doc_id = {idx: doc_id for idx, doc_id in enumerate(doc_ids)}
        
        self._index_built = True
        
        logging.info(
            f"[{self.__class__.__name__}:build_index] Built index for {len(doc_ids)} documents "
            f"(embedding_dim={self.embedding_dim})"
        )
        return True

    def select_next_farthest_from_mean(
        self,
        available_doc_ids: Set[str],
        selected_doc_ids: List[str]
    ) -> Optional[str]:
        """
        Select the next document that is farthest from the mean of already selected documents.
        This is an incremental operation - called each time we need the next document.
        
        Args:
            available_doc_ids: Set of document IDs that haven't been selected yet
            selected_doc_ids: List of document IDs already selected (in order)
            
        Returns:
            Document ID of the next document to select, or None if index not built or no candidates
        """
        if not self._index_built or self.embedding_matrix is None:
            # Fallback to random if index not available
            if available_doc_ids:
                return random.choice(list(available_doc_ids))
            return None
        
        # If no documents selected yet, pick random from available
        if not selected_doc_ids:
            return random.choice(list(available_doc_ids)) if available_doc_ids else None
        
        # Get indices of selected documents in the embedding matrix
        selected_indices = [self.doc_id_to_idx[did] for did in selected_doc_ids if did in self.doc_id_to_idx]
        
        if not selected_indices:
            # Selected documents not in index, fallback to random
            return random.choice(list(available_doc_ids)) if available_doc_ids else None
        
        # Compute mean of selected embeddings efficiently using numpy
        # selected_embeddings: (M, D) where M=len(selected_indices), D=embedding_dim
        selected_embeddings = self.embedding_matrix[selected_indices, :]  # Shape: (M, D)
        mean_embedding = np.mean(selected_embeddings, axis=0)  # Shape: (D,)
        
        # Get indices of available documents that are in our index
        available_indices = []
        available_doc_ids_in_index = []
        for doc_id in available_doc_ids:
            if doc_id in self.doc_id_to_idx:
                idx = self.doc_id_to_idx[doc_id]
                available_indices.append(idx)
                available_doc_ids_in_index.append(doc_id)
        
        if not available_indices:
            # No available documents in index, fallback to random
            return random.choice(list(available_doc_ids)) if available_doc_ids else None
        
        # Compute distances from mean to all available embeddings efficiently
        # available_embeddings: (K, D) where K=len(available_indices)
        available_embeddings = self.embedding_matrix[available_indices, :]  # Shape: (K, D)
        
        # Compute L2 distances: ||emb - mean|| for each available embedding
        # Use broadcasting: (K, D) - (D,) -> (K, D), then norm along axis=1 -> (K,)
        distances = np.linalg.norm(available_embeddings - mean_embedding, axis=1)  # Shape: (K,)
        
        # Find index of maximum distance
        max_dist_idx = np.argmax(distances)
        farthest_doc_id = available_doc_ids_in_index[max_dist_idx]
        
        return farthest_doc_id

    def get_document_order(self, doc_dict: Dict[str, Any]) -> List[str]:
        """
        DEPRECATED: Pre-orders all documents. 
        For adaptive sampling, use build_index() and select_next_farthest_from_mean() incrementally.
        
        This method is kept for backward compatibility but is inefficient for adaptive sampling
        since it orders all documents upfront even if we might stop early.
        
        Args:
            doc_dict: Dictionary of documents
            
        Returns:
            Pre-ordered list of document IDs
        """
        all_ids = list(doc_dict.keys())
        
        if not self.enabled:
            logging.info(f"[{self.__class__.__name__}] Embedding selection disabled, using random order")
            random.shuffle(all_ids)
            return all_ids
        
        # Try to build index
        if not self.build_index(doc_dict):
            # Fallback to random
            logging.warning(f"[{self.__class__.__name__}:get_document_order] Index build failed, using random order")
            random.shuffle(all_ids)
            return all_ids
        
        # Pre-order all documents (inefficient for adaptive sampling, but kept for compatibility)
        ordered_ids = []
        available_ids = set(all_ids)
        
        logging.info(f"[{self.__class__.__name__}:get_document_order] Pre-ordering {len(all_ids)} documents using Farthest-From-Mean strategy...")
        
        while available_ids:
            next_id = self.select_next_farthest_from_mean(available_ids, ordered_ids)
            if next_id is None:
                # Should not happen, but handle gracefully
                next_id = random.choice(list(available_ids))
            
            ordered_ids.append(next_id)
            available_ids.remove(next_id)
            
            if len(ordered_ids) % 50 == 0:
                logging.debug(f"[{self.__class__.__name__}:get_document_order] Pre-ordered {len(ordered_ids)}/{len(all_ids)} documents")
        
        return ordered_ids
