"""
DDGT Sampler: Diversity-Driven Good-Turing Adaptive Sampling.

This module implements the Good-Turing stopping condition for adaptive schema
generation. It tracks feature frequencies and uses the Good-Turing estimator
to determine when to stop sampling based on probabilistic coverage guarantees.

References:
    DDGT Specification - Section 4: Algorithm Logic
    Good-Turing Frequency Estimation
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from ..schema_entropy import SchemaEntropyCalculator


class DDGTSampler:
    """
    DDGT adaptive sampler with Good-Turing stopping condition.
    
    Implements:
    - Feature frequency tracking (multiset Phi)
    - Unique feature set tracking (F_current)
    - Good-Turing missing mass estimation
    - Probabilistic stopping criterion
    
    State Variables (Section 3 of spec):
    - Phi: Counter tracking frequency of every observed feature
    - F_current: Set of unique features observed
    - S: Set of sampled document IDs
    - n_samples: Total count of processed documents
    """
    
    def __init__(
        self,
        delta: float = 0.05,
        n_min: int = 50,
        batch_size: int = 5,
        feature_extractor: Optional[SchemaEntropyCalculator] = None
    ):
        """
        Initialize DDGT sampler.
        
        Args:
            delta: Allowed failure probability (δ) - probability of missing a feature
            n_min: Minimum documents to process before checking stopping condition
            batch_size: Number of documents to select per batch (k)
            feature_extractor: Optional custom feature extractor
        """
        self.delta = delta
        self.n_min = n_min
        self.batch_size = batch_size
        
        # State variables (Section 3 of spec)
        self.Phi = Counter()           # Feature frequency multiset
        self.F_current = set()         # Unique features observed
        self.S = set()                 # Sampled document IDs
        self.n_samples = 0             # Total processed documents
        
        # Feature extractor (reuse from entropy calculator)
        self.feature_extractor = feature_extractor or SchemaEntropyCalculator()
        
        # Statistics tracking
        self.N1_history = []           # Track N1 (singletons) over time
        self.G_est_history = []        # Track G_est (missing mass) over time
        self.should_stop = False
        self.stop_reason = None
        
        # Enable adaptive sampling
        self.enabled = True
        
        logging.info(
            f"[{self.__class__.__name__}:__init__] Initialized with parameters: "
            f"delta={delta}, n_min={n_min}, batch_size={batch_size}"
        )
    
    def extract_features_from_schema(self, schema: Any) -> Set[str]:
        """
        Extract features from schema representation.
        
        Reuses the feature extraction logic from SchemaEntropyCalculator.
        
        Args:
            schema: Schema in list or dict format
        
        Returns:
            Set of feature strings (table names and attributes)
        """
        # Use the existing feature extraction logic
        # The feature_extractor is a SchemaEntropyCalculator instance
        # which has a feature_extractor callable, not an extract_features method
        features = self.feature_extractor.feature_extractor(schema)
        return features
    
    def update_features(self, extracted_features: Set[str], doc_id: str):
        """
        Update feature counters after processing a document (Step B).
        
        Implements Algorithm 1, Lines 13-20:
        - For each feature: increment count in Phi
        - Update F_current (unique features)
        - Mark document as processed
        
        Args:
            extracted_features: Set of features from LLM_Extract
            doc_id: ID of the processed document
        """
        # Update feature frequencies
        for feature in extracted_features:
            self.Phi[feature] += 1
        
        # Update unique feature set
        self.F_current.update(extracted_features)
        
        # Mark document as processed
        self.S.add(doc_id)
        self.n_samples += 1
        
        logging.debug(
            f"[{self.__class__.__name__}:update_features] "
            f"Processed doc {doc_id}: {len(extracted_features)} features, "
            f"total unique: {len(self.F_current)}"
        )
    
    def check_stopping_condition(self) -> bool:
        """
        Check Good-Turing stopping condition (Step C / Algorithm 1 lines 24-25).
        
        Implements:
        1. Stability check: If n_samples < n_min, continue
        2. Calculate N1: number of features observed exactly once
        3. Calculate G_est: Good-Turing missing mass estimate = N1 / n_samples
        4. Decision: If G_est < delta, STOP; else CONTINUE
        
        Returns:
            True if should STOP, False if should CONTINUE
        """
        # Step C.1: Stability Check
        if self.n_samples < self.n_min:
            logging.debug(
                f"[{self.__class__.__name__}:check_stopping_condition] "
                f"n_samples={self.n_samples} < n_min={self.n_min}, continuing"
            )
            return False
        
        # Step C.2: Calculate Singletons (N1)
        # Count features observed exactly once
        N1 = sum(1 for count in self.Phi.values() if count == 1)
        
        # Step C.3: Good-Turing Estimate
        # G_est = N1 / n_samples (missing mass probability)
        G_est = N1 / self.n_samples if self.n_samples > 0 else 1.0
        
        # Track history for visualization
        self.N1_history.append(N1)
        self.G_est_history.append(G_est)
        
        logging.info(
            f"[{self.__class__.__name__}:check_stopping_condition] "
            f"n={self.n_samples}, N1={N1}, G_est={G_est:.6f}, delta={self.delta}"
        )
        
        # Step C.4: Decision
        if G_est < self.delta:
            self.should_stop = True
            self.stop_reason = (
                f"Good-Turing: G_est={G_est:.6f} < delta={self.delta} "
                f"(N1={N1}, n={self.n_samples})"
            )
            logging.info(
                f"[{self.__class__.__name__}:check_stopping_condition] "
                f"STOPPING: {self.stop_reason}"
            )
            return True
        
        logging.debug(
            f"[{self.__class__.__name__}:check_stopping_condition] "
            f"CONTINUING: G_est={G_est:.6f} >= delta={self.delta}"
        )
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Return statistics for analysis and visualization.
        
        Returns:
            Dictionary containing sampling statistics
        """
        # Calculate current N1
        N1_current = sum(1 for count in self.Phi.values() if count == 1)
        G_est_current = N1_current / self.n_samples if self.n_samples > 0 else 1.0
        
        # Feature frequency distribution
        freq_dist = Counter(self.Phi.values())
        
        return {
            "n_samples": self.n_samples,
            "n_features_unique": len(self.F_current),
            "n_sampled_docs": len(self.S),
            "N1_current": N1_current,
            "G_est_current": G_est_current,
            "N1_history": self.N1_history.copy(),
            "G_est_history": self.G_est_history.copy(),
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason,
            "parameters": {
                "delta": self.delta,
                "n_min": self.n_min,
                "batch_size": self.batch_size
            },
            "feature_freq_distribution": dict(freq_dist),
            "Phi_size": len(self.Phi)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Alias for get_statistics() for compatibility with existing code.
        
        Returns:
            Dictionary containing sampling statistics
        """
        stats = self.get_statistics()
        
        # Add backward compatibility fields
        stats.update({
            "num_docs_processed": self.n_samples,
            "num_unique_features": len(self.F_current),
            "avg_entropy": 0.0,  # Not applicable for DDGT
            "stability_streak": 0  # Not applicable for DDGT
        })
        
        return stats
    
    def reset(self):
        """
        Reset the sampler to initial state.
        """
        self.Phi = Counter()
        self.F_current = set()
        self.S = set()
        self.n_samples = 0
        
        self.N1_history = []
        self.G_est_history = []
        self.should_stop = False
        self.stop_reason = None
        
        # Reset feature extractor
        if hasattr(self.feature_extractor, 'reset'):
            self.feature_extractor.reset()
        
        logging.info(f"[{self.__class__.__name__}:reset] Sampler reset to initial state")
    
    def get_stop_reason(self) -> Optional[str]:
        """
        Get the reason for stopping (if stopped).
        
        Returns:
            String describing the stop reason, or None if not stopped
        """
        return self.stop_reason
    
    def get_feature_count(self) -> int:
        """
        Get the number of unique features observed.
        
        Returns:
            Number of unique features in F_current
        """
        return len(self.F_current)
    
    def get_entropy_history(self) -> List[float]:
        """
        Get entropy history (for compatibility with existing code).
        
        For DDGT, we return G_est_history instead of entropy.
        
        Returns:
            List of G_est values over time
        """
        return self.G_est_history.copy()
