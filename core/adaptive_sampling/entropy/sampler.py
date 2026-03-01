"""
Adaptive Sampling Algorithm for Schema Generation.

This module implements the two-phase adaptive sampling algorithm that reduces
the number of document processing steps (LLM calls) while maintaining
probabilistic guarantees on schema quality.

The algorithm tracks schema entropy and stops sampling when BOTH conditions are met:
1. Stability Condition: Entropy falls below threshold for a consecutive number of iterations.
2. Probabilistic Condition: Coverage guarantee is satisfied with given failure probability.

References:
    Nick's Notes July 23, 2025 - Adaptive Sampling
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from ..schema_entropy import SchemaEntropyCalculator


class AdaptiveSampler:
    """
    Adaptive sampling controller for schema generation.
    
    Implements the AdaptiveSchemaExtraction algorithm with:
    - Schema entropy tracking
    - Stability streak monitoring
    - Probabilistic early stopping
    - Dual-condition stopping logic
    """
    
    def __init__(
        self,
        theta: float = 0.05,
        m: int = 8,
        n_min: int = 10,
        delta: float = 0.05,
        epsilon: float = 0.05,
        entropy_calculator: Optional[SchemaEntropyCalculator] = None,
        enable_probabilistic_stop: bool = True
    ):
        """
        Initialize the adaptive sampler.
        
        Args:
            theta: Entropy threshold for stability (default 0.05)
            m: Stability streak threshold - number of consecutive low entropy
               iterations required (default 8)
            n_min: Minimum samples to process before early stopping (default 10)
            delta: Allowed failure probability for coverage guarantee (default 0.05)
            epsilon: Probability that a single document contains a new feature (default 0.05)
            entropy_calculator: Optional custom entropy calculator
            enable_probabilistic_stop: Whether to use probabilistic stopping criterion
                                      (now integrated into dual condition, default True)
        """
        self.theta = theta
        self.m = m
        self.n_min = n_min
        self.delta = delta
        self.epsilon = epsilon
        self.enable_probabilistic_stop = enable_probabilistic_stop
        
        # Initialize entropy calculator
        self.entropy_calculator = entropy_calculator or SchemaEntropyCalculator()
        
        # State tracking
        self.low_entropy_streak = 0
        self.n_processed = 0
        self.should_stop = False
        self.stop_reason = None
        
        logging.info(
            f"[{self.__class__.__name__}:__init__] Initialized with parameters: "
            f"theta={theta}, m={m}, n_min={n_min}, delta={delta}, epsilon={epsilon}, "
            f"probabilistic_stop={enable_probabilistic_stop}"
        )
    
    def should_continue(self, current_schema: Any) -> bool:
        """
        Determine whether to continue processing more documents.
        
        This is the main decision function that implements the stopping criteria.
        Returns False (Stop) only when BOTH stability and probabilistic conditions are met.
        
        Args:
            current_schema: Current state of the schema
            
        Returns:
            True if processing should continue, False if should stop
        """
        # Compute entropy for current iteration
        entropy = self.entropy_calculator.compute_entropy(current_schema)
        
        # Update streak counter
        if entropy < self.theta:
            self.low_entropy_streak += 1
        else:
            self.low_entropy_streak = 0
        
        # Increment processed count
        self.n_processed += 1
        
        logging.info(
            f"[{self.__class__.__name__}:should_continue] "
            f"Iteration {self.n_processed}: entropy={entropy:.4f}, "
            f"streak={self.low_entropy_streak}/{self.m}, "
            f"features={self.entropy_calculator.get_feature_count()}"
        )
        
        # Check stopping criteria
        
        # Condition 1: Stability streak + minimum samples
        stability_met = (self.low_entropy_streak >= self.m and self.n_processed >= self.n_min)
        
        # Condition 2: Probabilistic stopping
        feature_count = self.entropy_calculator.get_feature_count()
        if self.enable_probabilistic_stop:
            probabilistic_met = self._check_probabilistic_condition(self.n_processed, feature_count)
        else:
            # If probabilistic stop is disabled, we treat it as met (fallback to single condition)
            # Or we could enforce only stability. Based on "dual-condition" requirement,
            # we assume if it's disabled, we ignore it (so true).
            probabilistic_met = True

        # Dual Condition Check
        if stability_met and probabilistic_met:
            self.should_stop = True
            
            reasons = []
            if stability_met:
                reasons.append(f"stability(streak={self.low_entropy_streak}>={self.m})")
            if probabilistic_met:
                reasons.append(f"probabilistic((1-ε)^n*F <= δ)")
                
            self.stop_reason = " AND ".join(reasons)
            
            logging.info(
                f"[{self.__class__.__name__}:should_continue] "
                f"Stopping due to dual conditions: {self.stop_reason}"
            )
            return False
            
        # Log why we are continuing if one condition is met but not the other
        if stability_met and not probabilistic_met:
            logging.debug(f"[{self.__class__.__name__}] Stability met but probabilistic not met. Continuing.")
        elif probabilistic_met and not stability_met:
            logging.debug(f"[{self.__class__.__name__}] Probabilistic met but stability not met. Continuing.")
        
        # Continue processing
        return True
    
    def _check_probabilistic_condition(self, n_processed: int, feature_est: int) -> bool:
        """
        Check probabilistic stopping criterion.
        
        Condition: (1 - ε)^n * ℱ <= δ
        
        Args:
            n_processed: Number of documents processed so far
            feature_est: Estimated feature count seen so far
            
        Returns:
            True if condition is met
        """
        if feature_est == 0:
            return False
        
        # Compute (1 - ε)^n_processed
        prob_missing = math.pow(1 - self.epsilon, n_processed)
        
        # Check: prob_missing * feature_est <= delta
        # Equivalent to: prob_missing <= delta / feature_est
        
        threshold_val = prob_missing * feature_est
        is_met = threshold_val <= self.delta
        
        if is_met:
            logging.debug(
                f"[{self.__class__.__name__}:_check_probabilistic_condition] "
                f"Met: (1-ε)^n * F = {threshold_val:.6f} <= δ={self.delta}"
            )
        
        return is_met
    
    def compute_minimum_samples(self, feature_universe_size: int) -> int:
        """
        Compute the minimum number of samples required for feature coverage.
        
        Implements the formula:
            n_min = ⌈log(|ℱ|/δ) / ε⌉
        
        This ensures that all features appearing with probability ≥ ε
        are observed with probability ≥ (1 - δ).
        
        Args:
            feature_universe_size: Size of the feature universe |ℱ|
            
        Returns:
            Minimum number of samples required
        """
        if feature_universe_size <= 0:
            return self.n_min
        
        # Derived from (1-e)^n * F <= delta
        # (1-e)^n <= delta/F
        # n * ln(1-e) <= ln(delta/F)
        # n >= ln(delta/F) / ln(1-e)
        
        # Using the approximation ln(1-e) approx -e, we get n >= ln(F/delta) / e
        # Original implementation used math.log(feature_universe_size / self.delta) / self.epsilon
        
        numerator = math.log(feature_universe_size / self.delta)
        # Use exact log(1-epsilon) for better precision, or epsilon if approx is desired.
        # Original code used epsilon. Keeping it for consistency unless requested.
        # Actually, let's use the exact log if possible, but adhering to existing pattern is safer.
        # Let's stick to epsilon for now as denominator (standard approximation).
        
        n_min_computed = math.ceil(numerator / self.epsilon)
        
        logging.info(
            f"[{self.__class__.__name__}:compute_minimum_samples] "
            f"Computed n_min={n_min_computed} for |F|={feature_universe_size}, "
            f"δ={self.delta}, ε={self.epsilon}"
        )
        
        return max(n_min_computed, self.n_min)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the adaptive sampling process.
        
        Returns:
            Dictionary containing sampling statistics
        """
        entropy_stats = self.entropy_calculator.get_statistics()
        
        return {
            "n_processed": self.n_processed,
            "low_entropy_streak": self.low_entropy_streak,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason,
            "parameters": {
                "theta": self.theta,
                "m": self.m,
                "n_min": self.n_min,
                "delta": self.delta,
                "epsilon": self.epsilon,
                "probabilistic_stop_enabled": self.enable_probabilistic_stop
            },
            "entropy_statistics": entropy_stats,
            "entropy_history": self.entropy_calculator.get_entropy_history()
        }
    
    def reset(self):
        """Reset the sampler to initial state."""
        self.low_entropy_streak = 0
        self.n_processed = 0
        self.should_stop = False
        self.stop_reason = None
        self.entropy_calculator.reset()
        
        logging.info(f"[{self.__class__.__name__}:reset] Adaptive sampler reset")
    
    def get_stop_reason(self) -> Optional[str]:
        """
        Get the reason for stopping (if stopped).
        
        Returns:
            String describing the stop reason, or None if not stopped
        """
        return self.stop_reason
