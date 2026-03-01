"""
Schema Entropy Calculator for Adaptive Sampling.

This module implements the schema entropy calculation as described in the
adaptive sampling algorithm. Schema entropy measures the stability of schema
evolution by tracking feature changes between successive schema states.

Note: This is NOT Shannon entropy from information theory, but rather a
normalized divergence ratio that reflects structural volatility.
"""

import logging
from typing import Dict, List, Set, Any, Optional


class SchemaEntropyCalculator:
    """
    Calculates schema entropy to measure schema stability during extraction.
    
    Schema Entropy H(i) measures how much a document changes the current
    understanding of a data schema. It is defined as:
    
        H(i) = |Δ_i| / |F(S_i) ∪ F(S_{i-1})|
    
    where:
        - F(S) is the feature set of schema S
        - Δ_i = F(S_i) ⊕ F(S_{i-1}) is the symmetric difference
        - Returns 1.0 if denominator is 0 (complete instability)
    
    Lower entropy indicates schema stabilization, higher entropy suggests
    the schema is still evolving significantly.
    """
    
    def __init__(self, feature_extractor: Optional[callable] = None):
        """
        Initialize the entropy calculator.
        
        Args:
            feature_extractor: Optional custom function to extract features
                             from schema. If None, uses default extraction.
        """
        self.feature_extractor = feature_extractor or self._default_feature_extractor
        self.prev_features: Optional[Set[str]] = None
        self.entropy_history: List[float] = []
        
    def _default_feature_extractor(self, schema: Any) -> Set[str]:
        """
        Default feature extraction from schema representation.
        
        Extracts features from the ReDD schema format which is typically:
        [
            {
                "Schema Name": <table_name>,
                "Attributes": [
                    {<attribute_name>: <explanation>},
                    ...
                ]
            },
            ...
        ]
        
        Or from the log format which is a list of schema entries.
        
        Args:
            schema: Schema representation (list or dict)
            
        Returns:
            Set of feature strings representing the schema structure
        """
        features = set()
        
        if not schema:
            return features
            
        # Handle list format (typical ReDD schema format)
        if isinstance(schema, list):
            for table_entry in schema:
                if isinstance(table_entry, dict):
                    # Add table name as feature
                    table_name = table_entry.get("Schema Name", table_entry.get("Table Name", ""))
                    if table_name:
                        features.add(f"table:{table_name}")
                    
                    # Add attributes as features
                    attributes = table_entry.get("Attributes", [])
                    for attr in attributes:
                        if isinstance(attr, dict):
                            # Each attribute is like {attr_name: description}
                            for attr_name, attr_desc in attr.items():
                                if table_name:
                                    # Feature format: "table_name.attribute_name"
                                    features.add(f"{table_name}.{attr_name}")
                                else:
                                    features.add(f"attr:{attr_name}")
                        elif isinstance(attr, str):
                            # Sometimes attributes might be simple strings
                            if table_name:
                                features.add(f"{table_name}.{attr}")
                            else:
                                features.add(f"attr:{attr}")
        
        # Handle dict format
        elif isinstance(schema, dict):
            # If it's a single table entry
            if "Schema Name" in schema or "Table Name" in schema:
                table_name = schema.get("Schema Name", schema.get("Table Name", ""))
                if table_name:
                    features.add(f"table:{table_name}")
                
                attributes = schema.get("Attributes", [])
                for attr in attributes:
                    if isinstance(attr, dict):
                        for attr_name, attr_desc in attr.items():
                            if table_name:
                                features.add(f"{table_name}.{attr_name}")
                            else:
                                features.add(f"attr:{attr_name}")
                    elif isinstance(attr, str):
                        if table_name:
                            features.add(f"{table_name}.{attr}")
                        else:
                            features.add(f"attr:{attr}")
            
            # If it's a nested structure with multiple tables
            else:
                for key, value in schema.items():
                    if isinstance(value, list):
                        features.add(f"table:{key}")
                        for item in value:
                            if isinstance(item, str):
                                features.add(f"{key}.{item}")
                            elif isinstance(item, dict):
                                for k, v in item.items():
                                    features.add(f"{key}.{k}")
        
        return features
    
    def compute_entropy(self, current_schema: Any, update_history: bool = True) -> float:
        """
        Compute the schema entropy between current and previous schema states.
        
        Args:
            current_schema: Current schema representation
            update_history: Whether to update internal state (default True)
            
        Returns:
            Schema entropy value in [0, 1], where:
                - 0.0 = no change (perfect stability)
                - 1.0 = complete change or initial state
        """
        # Extract features from current schema
        current_features = self.feature_extractor(current_schema)
        
        # First iteration - no previous state
        if self.prev_features is None:
            if update_history:
                self.prev_features = current_features
                self.entropy_history.append(1.0)
            return 1.0
        
        # Compute symmetric difference (features changed)
        delta = current_features.symmetric_difference(self.prev_features)
        
        # Compute union (all features observed)
        union = current_features.union(self.prev_features)
        
        # Calculate entropy
        if len(union) == 0:
            entropy = 1.0  # No features at all - complete instability
        else:
            entropy = len(delta) / len(union)
        
        # Update state if requested
        if update_history:
            self.prev_features = current_features
            self.entropy_history.append(entropy)
            
            logging.debug(
                f"[{self.__class__.__name__}:compute_entropy] "
                f"Features: curr={len(current_features)}, prev={len(self.prev_features)}, "
                f"delta={len(delta)}, union={len(union)}, entropy={entropy:.4f}"
            )
        
        return entropy
    
    def get_entropy_history(self) -> List[float]:
        """
        Get the history of computed entropy values.
        
        Returns:
            List of entropy values in chronological order
        """
        return self.entropy_history.copy()
    
    def get_current_features(self) -> Optional[Set[str]]:
        """
        Get the current feature set (from most recent schema).
        
        Returns:
            Set of feature strings, or None if no schema processed yet
        """
        return self.prev_features.copy() if self.prev_features else None
    
    def get_feature_count(self) -> int:
        """
        Get the number of features in the current schema.
        
        Returns:
            Number of features, or 0 if no schema processed yet
        """
        return len(self.prev_features) if self.prev_features else 0
    
    def reset(self):
        """Reset the entropy calculator to initial state."""
        self.prev_features = None
        self.entropy_history = []
        logging.info(f"[{self.__class__.__name__}:reset] Entropy calculator reset")
    
    def compute_stability_streak(self, threshold: float) -> int:
        """
        Compute the current stability streak (consecutive low entropy values).
        
        Args:
            threshold: Entropy threshold below which schema is considered stable
            
        Returns:
            Number of consecutive entropy values below threshold
        """
        if not self.entropy_history:
            return 0
        
        streak = 0
        for entropy in reversed(self.entropy_history):
            if entropy < threshold:
                streak += 1
            else:
                break
        
        return streak
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the entropy calculation process.
        
        Returns:
            Dictionary containing entropy statistics
        """
        if not self.entropy_history:
            return {
                "num_iterations": 0,
                "mean_entropy": None,
                "min_entropy": None,
                "max_entropy": None,
                "final_entropy": None,
                "feature_count": 0
            }
        
        return {
            "num_iterations": len(self.entropy_history),
            "mean_entropy": sum(self.entropy_history) / len(self.entropy_history),
            "min_entropy": min(self.entropy_history),
            "max_entropy": max(self.entropy_history),
            "final_entropy": self.entropy_history[-1],
            "feature_count": self.get_feature_count()
        }

