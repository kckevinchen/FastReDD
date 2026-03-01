"""
Adaptive Sampling for Schema Extraction

Implements early stopping during schema discovery by tracking schema stability
using a custom entropy metric.
"""
import logging
from typing import Dict, List, Set, Tuple, Optional


class AdaptiveSampling:
    """
    Tracks schema evolution and implements early stopping conditions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize adaptive sampling with configuration.
        
        Args:
            config: Configuration dictionary with adaptive_sampling settings
        """
        self.enabled = config.get("adaptive_sampling", {}).get("enabled", False)
        if not self.enabled:
            return
        
        adaptive_config = config.get("adaptive_sampling", {})
        self.entropy_threshold = adaptive_config.get("entropy_threshold", 0.05)
        self.streak_limit = adaptive_config.get("streak_limit", 5)
        self.min_docs = adaptive_config.get("min_docs", 10)
        self.coverage_confidence = adaptive_config.get("coverage_confidence", 0.95)
        self.min_feature_count = adaptive_config.get("min_feature_count", 2)
        
        # State tracking
        self.reset()
        
        logging.info(f"[{self.__class__.__name__}:__init__] Adaptive sampling enabled: "
                    f"entropy_threshold={self.entropy_threshold}, "
                    f"streak_limit={self.streak_limit}, "
                    f"min_docs={self.min_docs}")
    
    def reset(self):
        """Reset state for a new dataset."""
        self.prev_features: Set[Tuple[str, str]] = set()  # (table_name, attr_name) tuples
        self.curr_features: Set[Tuple[str, str]] = set()
        self.feature_counts: Dict[Tuple[str, str], int] = {}
        self.stability_streak = 0
        self.num_docs_processed = 0
        self.entropy_history: List[float] = []
    
    def extract_features_from_log(self, log: List[Dict]) -> Set[Tuple[str, str]]:
        """
        Extract schema features (tables and attributes) from log.
        
        Args:
            log: List of schema dictionaries with "Schema Name" and "Attributes"
            
        Returns:
            Set of (table_name, attribute_name) tuples
        """
        features = set()
        
        for schema in log:
            schema_name = schema.get("Schema Name", "")
            if not schema_name:
                continue
            
            attrs = schema.get("Attributes", [])
            if isinstance(attrs, list):
                for attr in attrs:
                    if isinstance(attr, dict):
                        # Handle different attribute dict formats:
                        # 1. {"Attribute Name": "name", ...} - explicit format
                        if "Attribute Name" in attr:
                            attr_name = attr.get("Attribute Name", "")
                            if attr_name:
                                features.add((schema_name, attr_name))
                        else:
                            # 2. {"attr_name": "description"} - attribute name is the key
                            # This is the format used in the actual results
                            for attr_name in attr.keys():
                                if attr_name and attr_name.strip():
                                    features.add((schema_name, attr_name))
                    elif isinstance(attr, str):
                        # Attribute is just a string name
                        if attr:
                            features.add((schema_name, attr))
        
        return features
    
    def calculate_schema_entropy(self, prev_features: Set[Tuple[str, str]], 
                                curr_features: Set[Tuple[str, str]]) -> float:
        """
        Calculate custom Schema Entropy metric.
        
        Entropy = (changed_features) / (total_unique_features)
        where changed_features = features added or removed
        
        Args:
            prev_features: Features from previous step
            curr_features: Features from current step
            
        Returns:
            Schema entropy value (0.0 to 1.0)
        """
        if not prev_features and not curr_features:
            return 0.0
        
        union_features = prev_features | curr_features
        intersection_features = prev_features & curr_features
        
        # Changed features = features that are in union but not in intersection
        changed_features = union_features - intersection_features
        
        if not union_features:
            return 0.0
        
        entropy = len(changed_features) / len(union_features)
        return entropy
    
    def update(self, log: List[Dict]) -> Dict[str, float]:
        """
        Update state with new log and calculate metrics.
        
        Args:
            log: Current log state from schema generation
            
        Returns:
            Dictionary with metrics: entropy, stability_streak, num_docs_processed
        """
        if not self.enabled:
            return {}
        
        self.num_docs_processed += 1
        
        # Extract current features
        self.curr_features = self.extract_features_from_log(log)
        
        # Update feature counts
        for feature in self.curr_features:
            self.feature_counts[feature] = self.feature_counts.get(feature, 0) + 1
        
        # Calculate entropy
        entropy = self.calculate_schema_entropy(self.prev_features, self.curr_features)
        self.entropy_history.append(entropy)
        
        # Update stability streak
        if entropy < self.entropy_threshold:
            self.stability_streak += 1
        else:
            self.stability_streak = 0
        
        # Update previous features for next iteration
        self.prev_features = self.curr_features.copy()
        
        metrics = {
            "entropy": entropy,
            "stability_streak": self.stability_streak,
            "num_docs_processed": self.num_docs_processed,
            "num_unique_features": len(self.curr_features),
            "num_total_features_seen": len(self.feature_counts)
        }
        
        logging.info(f"[{self.__class__.__name__}:update] Doc {self.num_docs_processed}: "
                    f"entropy={entropy:.4f}, streak={self.stability_streak}, "
                    f"features={len(self.curr_features)}")
        
        return metrics
    
    def should_stop_stability(self) -> bool:
        """
        Check stopping condition A: Stability streak.
        
        Returns:
            True if should stop due to stability streak
        """
        if not self.enabled:
            return False
        
        if self.num_docs_processed < self.min_docs:
            return False
        
        return self.stability_streak >= self.streak_limit
    
    def should_stop_coverage(self) -> bool:
        """
        Check stopping condition B: Probabilistic safety stop.
        
        Returns:
            True if should stop due to sufficient coverage
        """
        if not self.enabled:
            return False
        
        if self.num_docs_processed < self.min_docs:
            return False
        
        if not self.feature_counts:
            return False
        
        # Count features that have been seen at least min_feature_count times
        frequent_features = sum(1 for count in self.feature_counts.values() 
                               if count >= self.min_feature_count)
        total_features = len(self.feature_counts)
        
        if total_features == 0:
            return False
        
        coverage_ratio = frequent_features / total_features
        
        should_stop = coverage_ratio >= self.coverage_confidence
        
        if should_stop:
            logging.info(f"[{self.__class__.__name__}:should_stop_coverage] "
                        f"Coverage stop: {frequent_features}/{total_features} = {coverage_ratio:.4f}")
        
        return should_stop
    
    def should_stop(self) -> Tuple[bool, str]:
        """
        Check if should stop processing (either condition).
        
        Returns:
            (should_stop, reason) tuple
        """
        if not self.enabled:
            return False, ""
        
        if self.should_stop_stability():
            return True, f"stability_streak (streak={self.stability_streak})"
        
        if self.should_stop_coverage():
            return True, f"coverage (ratio={len([c for c in self.feature_counts.values() if c >= self.min_feature_count])}/{len(self.feature_counts)})"
        
        return False, ""
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        if not self.enabled:
            return {}
        
        frequent_features = sum(1 for count in self.feature_counts.values() 
                               if count >= self.min_feature_count)
        
        return {
            "num_docs_processed": self.num_docs_processed,
            "num_unique_features": len(self.curr_features),
            "num_total_features_seen": len(self.feature_counts),
            "frequent_features": frequent_features,
            "stability_streak": self.stability_streak,
            "avg_entropy": sum(self.entropy_history) / len(self.entropy_history) if self.entropy_history else 0.0,
            "last_entropy": self.entropy_history[-1] if self.entropy_history else 0.0
        }

