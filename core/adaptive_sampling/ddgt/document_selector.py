"""
Document Selector for DDGT (Diversity-Driven Good-Turing) Sampling.

This module implements diversity-driven batch selection using the max-min
(k-center) algorithm. Documents are selected to maximize coverage of the
embedding space using greedy farthest point sampling.

References:
    Algorithm 2 "SelectMaxMinDiversity" from DDGT specification
"""

import logging
import random
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set

import numpy as np


class DDGTDocumentSelector:
    """
    Selects batches of documents using max-min diversity (k-center algorithm).
    
    Implements greedy farthest point sampling where each new document is selected
    to maximize the minimum distance to all previously selected documents.
    This provides better coverage of the embedding space compared to mean-based methods.
    """
    
    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        """
        Initialize the DDGT document selector.
        
        Args:
            config: Configuration dictionary containing adaptive_sampling settings
            api_key: API key (not used, embeddings are pre-generated)
        """
        adaptive_config = config.get("adaptive_sampling", {})
        self.enabled = adaptive_config.get("use_embedding_selection", True)
        self.embedding_model = adaptive_config.get("embedding_model", "models/embedding-001")
        
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
        
        logging.info(
            f"[{self.__class__.__name__}:__init__] Initialized DDGT selector with model: {self.embedding_model}"
        )

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
            logging.warning(
                f"[{self.__class__.__name__}:load_embeddings] Embedding file not found: {path}"
            )
            return {}
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            embeddings = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
            logging.info(
                f"[{self.__class__.__name__}:load_embeddings] "
                f"Loaded {len(embeddings)} embeddings from {path}"
            )
            return embeddings
        except Exception as e:
            logging.warning(
                f"[{self.__class__.__name__}:load_embeddings] "
                f"Failed to load embeddings from {path}: {e}"
            )
            return {}

    def build_index(self, doc_dict: Dict[str, Any]) -> bool:
        """
        Build an efficient index structure from pre-generated embeddings.
        Must be called before using batch selection.
        
        Args:
            doc_dict: Dictionary of documents {doc_id: [doc_text, ...]}
                     Used to get list of all document IDs
            
        Returns:
            True if index built successfully, False otherwise
        """
        if not self.enabled:
            logging.info(
                f"[{self.__class__.__name__}:build_index] "
                "Embedding selection disabled"
            )
            return False
            
        if self._index_built:
            logging.debug(
                f"[{self.__class__.__name__}:build_index] "
                "Index already built, skipping"
            )
            return True
        
        # Load embeddings from cache
        cache_path = None
        if self.embedding_file:
            cache_path = Path(self.embedding_file)
        else:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] "
                f"No embedding_file specified in config. "
                f"Embeddings must be pre-generated. Falling back to random selection."
            )
            return False
        
        embeddings_dict = self.load_embeddings(cache_path)
        
        if not embeddings_dict:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] "
                f"No embeddings loaded. "
                f"Embeddings should be pre-generated (e.g., in notebook PART 0.5). "
                f"Falling back to random selection."
            )
            return False
        
        # Filter embeddings to only include documents in doc_dict
        all_doc_ids = set(doc_dict.keys())
        filtered_embeddings = {
            did: emb for did, emb in embeddings_dict.items() 
            if did in all_doc_ids
        }
        
        if not filtered_embeddings:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] "
                f"No matching embeddings found for documents in doc_dict. "
                f"Expected {len(all_doc_ids)} documents, found {len(embeddings_dict)} embeddings."
            )
            return False
        
        missing_ids = all_doc_ids - set(filtered_embeddings.keys())
        if missing_ids:
            logging.warning(
                f"[{self.__class__.__name__}:build_index] "
                f"{len(missing_ids)}/{len(all_doc_ids)} documents missing embeddings. "
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
            f"[{self.__class__.__name__}:build_index] "
            f"Built index for {len(doc_ids)} documents (embedding_dim={self.embedding_dim})"
        )
        return True

    def select_batch_maxmin(
        self,
        sampled_doc_ids: Set[str],
        batch_size: int,
        distance_metric: str = "euclidean"
    ) -> List[str]:
        """
        Select a batch of documents using max-min diversity (Algorithm 2).
        
        Implements greedy k-center selection:
        1. If first batch: select 1 random document
        2. While batch size < k:
           - For each candidate: compute min distance to all selected docs
           - Select candidate with maximum min_dist (farthest point)
        
        Args:
            sampled_doc_ids: Set of document IDs already processed (S)
            batch_size: Number of documents to select (k)
            distance_metric: "euclidean" or "cosine"
        
        Returns:
            List of document IDs for the selected batch
        """
        if not self._index_built or self.embedding_matrix is None:
            # Fallback to random selection
            logging.warning(
                f"[{self.__class__.__name__}:select_batch_maxmin] "
                "Index not built, falling back to random selection"
            )
            all_doc_ids = set(self.idx_to_doc_id.values()) if self.idx_to_doc_id else set()
            available = list(all_doc_ids - sampled_doc_ids)
            random.shuffle(available)
            return available[:batch_size]
        
        # Identify candidates (documents not yet sampled)
        all_doc_ids = set(self.doc_id_to_idx.keys())
        candidates = list(all_doc_ids - sampled_doc_ids)
        
        if not candidates:
            logging.info(
                f"[{self.__class__.__name__}:select_batch_maxmin] "
                "No candidates remaining (corpus exhausted)"
            )
            return []
        
        # Initialize batch
        batch = []
        
        # Step 1: If first batch (S is empty), select 1 random document
        if not sampled_doc_ids:
            first_doc = random.choice(candidates)
            batch.append(first_doc)
            candidates.remove(first_doc)
            logging.debug(
                f"[{self.__class__.__name__}:select_batch_maxmin] "
                f"First batch: randomly selected {first_doc}"
            )
        
        # Step 2: Greedy k-center selection
        # Already selected = sampled_doc_ids ∪ batch
        selected_doc_ids = sampled_doc_ids.copy()
        selected_doc_ids.update(batch)
        
        while len(batch) < batch_size and candidates:
            # Get indices of already selected documents
            selected_indices = [
                self.doc_id_to_idx[did] for did in selected_doc_ids 
                if did in self.doc_id_to_idx
            ]
            
            if not selected_indices:
                # Shouldn't happen, but handle gracefully
                selected_doc = random.choice(candidates)
                batch.append(selected_doc)
                candidates.remove(selected_doc)
                selected_doc_ids.add(selected_doc)
                continue
            
            # Get embeddings of selected documents
            selected_embeddings = self.embedding_matrix[selected_indices, :]  # (M, D)
            
            # Compute min_dist for each candidate
            best_doc_id = None
            best_min_dist = -1
            
            for candidate_id in candidates:
                if candidate_id not in self.doc_id_to_idx:
                    continue
                
                candidate_idx = self.doc_id_to_idx[candidate_id]
                candidate_emb = self.embedding_matrix[candidate_idx, :]  # (D,)
                
                # Compute distances to all selected documents
                if distance_metric == "cosine":
                    # Cosine distance = 1 - cosine_similarity
                    # cosine_sim = dot(a, b) / (||a|| * ||b||)
                    norms = np.linalg.norm(selected_embeddings, axis=1)  # (M,)
                    candidate_norm = np.linalg.norm(candidate_emb)
                    
                    if candidate_norm == 0:
                        distances = np.ones(len(selected_indices))
                    else:
                        dots = np.dot(selected_embeddings, candidate_emb)  # (M,)
                        cosine_sims = dots / (norms * candidate_norm + 1e-8)
                        distances = 1 - cosine_sims
                else:
                    # Euclidean distance
                    distances = np.linalg.norm(
                        selected_embeddings - candidate_emb, axis=1
                    )  # (M,)
                
                # Min distance to any selected document
                min_dist = np.min(distances)
                
                # Track candidate with maximum min_dist (farthest point)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_doc_id = candidate_id
            
            # Add best candidate to batch
            if best_doc_id is None:
                # No valid candidate found, pick random
                best_doc_id = random.choice(candidates)
            
            batch.append(best_doc_id)
            candidates.remove(best_doc_id)
            selected_doc_ids.add(best_doc_id)
            
            logging.debug(
                f"[{self.__class__.__name__}:select_batch_maxmin] "
                f"Selected {best_doc_id} with min_dist={best_min_dist:.4f}"
            )
        
        logging.info(
            f"[{self.__class__.__name__}:select_batch_maxmin] "
            f"Selected batch of {len(batch)} documents"
        )
        
        return batch

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the document selector.
        
        Returns:
            Dictionary with selector statistics
        """
        return {
            "enabled": self.enabled,
            "index_built": self._index_built,
            "total_documents": len(self.idx_to_doc_id) if self._index_built else 0,
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model,
            "embedding_file": str(self.embedding_file) if self.embedding_file else None
        }
