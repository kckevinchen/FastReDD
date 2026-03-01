"""
Query-Aware Chunk Filtering with Conformal Guarantees

Filters document chunks based on query similarity while guaranteeing
statistical recall using conformal prediction.
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from ..doc_clustering.vectorizer import DocVectorizer, llm_embeddings


@dataclass
class ChunkFilterConfig:
    """
    Configuration for chunk filtering.
    
    Args:
        embedding_model: Embedding model to use. Options:
            OpenAI models:
            - "text-embedding-3-small" (default): Fast and cost-effective
            - "text-embedding-3-large": Higher quality embeddings
            - "text-embedding-ada-002": Older model, still supported
            Google Gemini models:
            - "models/embedding-001": Gemini embedding model
            - "embedding-001": Alternative Gemini model name
    """
    enabled: bool = False
    recall_target: float = 0.95
    chunk_size: int = 500  # characters per chunk
    chunk_overlap: int = 50  # overlap between chunks
    threshold_path: Optional[str] = None
    evidence_augmentation: bool = False
    evidence_aug_max_terms: int = 10
    embedding_model: str = "text-embedding-3-small"


class ConformalChunkFilter:
    """
    Filters document chunks using conformal prediction guarantees.
    """
    
    def __init__(self, config: Dict, api_key=None):
        """
        Initialize conformal chunk filter.
        
        Args:
            config: Configuration dictionary with chunk_filter settings
            api_key: Optional API key for embedding models. If None, will try to get from config or environment.
        """
        filter_config = config.get("chunk_filter", {})
        self.config = ChunkFilterConfig(
            enabled=filter_config.get("enabled", False),
            recall_target=filter_config.get("recall_target", 0.95),
            chunk_size=filter_config.get("chunk_size", 500),
            chunk_overlap=filter_config.get("chunk_overlap", 50),
            threshold_path=filter_config.get("threshold_path"),
            evidence_augmentation=filter_config.get("evidence_augmentation", {}).get("enabled", False),
            evidence_aug_max_terms=filter_config.get("evidence_augmentation", {}).get("max_terms", 10),
            embedding_model=filter_config.get("embedding_model", "text-embedding-3-small")
        )
        
        if not self.config.enabled:
            return
        
        # Determine API key for embedding model
        # Priority: provided api_key > config api_key > environment variable
        embedding_api_key = api_key
        if embedding_api_key is None:
            embedding_api_key = config.get("api_key")
        
        # Check if embedding model is Gemini (needs GEMINI_API_KEY/GOOGLE_API_KEY)
        is_gemini_embedding = (
            self.config.embedding_model.startswith("models/embedding-") or 
            self.config.embedding_model.startswith("embedding-") or
            self.config.embedding_model.startswith("gemini-embedding-")
        )
        
        # If it's a Gemini embedding model, try to get Gemini API key if not provided
        if is_gemini_embedding and embedding_api_key is None:
            embedding_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        # Initialize vectorizer with API key
        self.embedding_api_key = embedding_api_key
        self.vectorizer = DocVectorizer(
            embedder=lambda s: llm_embeddings(s, model=self.config.embedding_model, api_key=self.embedding_api_key)
        )
        
        # Load or compute threshold
        self.threshold = self._load_or_compute_threshold()
        
        logging.info(f"[{self.__class__.__name__}:__init__] Chunk filtering enabled: "
                    f"recall_target={self.config.recall_target}, "
                    f"embedding_model={self.config.embedding_model}, "
                    f"threshold={self.threshold:.4f}")
    
    def _load_or_compute_threshold(self) -> float:
        """
        Load threshold from file or compute from calibration data.
        
        Returns:
            Similarity threshold value
        """
        if self.config.threshold_path and os.path.exists(self.config.threshold_path):
            try:
                with open(self.config.threshold_path, "r") as f:
                    threshold_data = json.load(f)
                    threshold = threshold_data.get("threshold", 0.5)
                    logging.info(f"[{self.__class__.__name__}:_load_or_compute_threshold] "
                               f"Loaded threshold from {self.config.threshold_path}: {threshold}")
                    return threshold
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}:_load_or_compute_threshold] "
                               f"Failed to load threshold: {e}, using default")
        
        # Default threshold (should be calibrated in practice)
        return 0.5
    
    def chunk_document(self, document: str) -> List[str]:
        """
        Split document into overlapping chunks.
        
        Args:
            document: Full document text
            
        Returns:
            List of chunk strings
        """
        if len(document) <= self.config.chunk_size:
            return [document]
        
        chunks = []
        start = 0
        
        while start < len(document):
            end = start + self.config.chunk_size
            chunk = document[start:end]
            chunks.append(chunk)
            
            if end >= len(document):
                break
            
            # Move start forward with overlap
            start = end - self.config.chunk_overlap
        
        return chunks
    
    def augment_query(self, query: str, schema_features: Optional[List[str]] = None) -> str:
        """
        Augment query with schema features if query is vague.
        
        Args:
            query: Original query
            schema_features: List of schema feature names/terms
            
        Returns:
            Augmented query string
        """
        if not self.config.evidence_augmentation:
            return query
        
        # Simple heuristic: consider query vague if it's short (< 20 chars) or has few words
        query_words = query.split()
        is_vague = len(query) < 20 or len(query_words) < 5
        
        if not is_vague or not schema_features:
            return query
        
        # Add top schema features
        top_features = schema_features[:self.config.evidence_aug_max_terms]
        augmented = f"{query}\n\nContext: {', '.join(top_features)}"
        
        logging.debug(f"[{self.__class__.__name__}:augment_query] "
                     f"Augmented query with {len(top_features)} features")
        
        return augmented
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def filter_chunks(self, query: str, document: str, 
                     schema_features: Optional[List[str]] = None) -> Tuple[List[str], Dict]:
        """
        Filter document chunks based on query similarity.
        
        Args:
            query: User query
            document: Document text to filter
            schema_features: Optional schema features for augmentation
            
        Returns:
            (filtered_chunks, stats) tuple where stats contains filtering metrics
        """
        if not self.config.enabled:
            return [document], {"filtered": False, "num_chunks": 1, "num_kept": 1}
        
        # Augment query if needed
        augmented_query = self.augment_query(query, schema_features)
        
        # Chunk document
        chunks = self.chunk_document(document)
        
        if len(chunks) == 1:
            # Single chunk, no filtering needed
            return chunks, {"filtered": False, "num_chunks": 1, "num_kept": 1}
        
        # Get embeddings
        try:
            query_emb = np.array(self.vectorizer.embedder(augmented_query))
            chunk_embs = [np.array(self.vectorizer.embedder(chunk)) for chunk in chunks]
        except Exception as e:
            error_msg = str(e) if e else repr(e)
            error_type = type(e).__name__
            logging.warning(f"[{self.__class__.__name__}:filter_chunks] "
                           f"Failed to get embeddings ({error_type}): {error_msg}. "
                           f"Returning all chunks without filtering.")
            import traceback
            logging.debug(f"[{self.__class__.__name__}:filter_chunks] "
                         f"Embedding error traceback: {traceback.format_exc()}")
            return chunks, {"filtered": False, "num_chunks": len(chunks), "num_kept": len(chunks), 
                          "error": error_msg, "error_type": error_type}
        
        # Calculate similarities
        similarities = [self.cosine_similarity(query_emb, chunk_emb) for chunk_emb in chunk_embs]
        
        # Filter chunks above threshold
        filtered_chunks = [chunk for chunk, sim in zip(chunks, similarities) if sim >= self.threshold]
        
        stats = {
            "filtered": True,
            "num_chunks": len(chunks),
            "num_kept": len(filtered_chunks),
            "num_dropped": len(chunks) - len(filtered_chunks),
            "min_similarity": min(similarities) if similarities else 0.0,
            "max_similarity": max(similarities) if similarities else 0.0,
            "avg_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "threshold": self.threshold
        }
        
        logging.info(f"[{self.__class__.__name__}:filter_chunks] "
                     f"Filtered {len(chunks)} chunks: kept {len(filtered_chunks)}, "
                     f"avg_sim={stats['avg_similarity']:.4f}")
        
        return filtered_chunks, stats
    
    def calibrate_threshold(self, calibration_data: List[Tuple[str, str, bool]]) -> float:
        """
        Calibrate threshold from labeled calibration data.
        
        Args:
            calibration_data: List of (query, chunk, is_relevant) tuples
            
        Returns:
            Calibrated threshold value
        """
        if not calibration_data:
            logging.warning(f"[{self.__class__.__name__}:calibrate_threshold] "
                           f"No calibration data provided, using default threshold")
            return 0.5
        
        # Get embeddings and similarities for relevant chunks
        similarities_relevant = []
        
        for query, chunk, is_relevant in calibration_data:
            try:
                query_emb = np.array(self.vectorizer.embedder(query))
                chunk_emb = np.array(self.vectorizer.embedder(chunk))
                sim = self.cosine_similarity(query_emb, chunk_emb)
                
                if is_relevant:
                    similarities_relevant.append(sim)
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}:calibrate_threshold] "
                               f"Failed to process calibration sample: {e}")
                continue
        
        if not similarities_relevant:
            logging.warning(f"[{self.__class__.__name__}:calibrate_threshold] "
                           f"No relevant chunks found in calibration data")
            return 0.5
        
        # Set threshold to (1 - recall_target) percentile of relevant similarities
        # This ensures recall_target fraction of relevant chunks are kept
        percentile = (1 - self.config.recall_target) * 100
        threshold = np.percentile(similarities_relevant, percentile)
        
        logging.info(f"[{self.__class__.__name__}:calibrate_threshold] "
                    f"Calibrated threshold: {threshold:.4f} "
                    f"(from {len(similarities_relevant)} relevant chunks, "
                    f"target recall={self.config.recall_target})")
        
        # Save threshold
        if self.config.threshold_path:
            threshold_dir = os.path.dirname(self.config.threshold_path)
            if threshold_dir:
                os.makedirs(threshold_dir, exist_ok=True)
            with open(self.config.threshold_path, "w") as f:
                json.dump({"threshold": float(threshold), "recall_target": self.config.recall_target}, f, indent=2)
        
        self.threshold = threshold
        return threshold

