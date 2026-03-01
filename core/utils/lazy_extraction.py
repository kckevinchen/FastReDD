"""
Lazy Attribute Extraction with Cost-Based Ordering

Optimizes data population by extracting attributes in order that fails fast.
"""
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from .selectivity_estimator import SelectivityEstimator


@dataclass
class AttributeFilter:
    """Represents an attribute filter with extraction function."""
    name: str
    extract_fn: Callable  # Function that extracts attribute value
    filter_condition: Callable  # Function that returns True if row should be kept
    cost: float = 1.0  # Extraction cost
    selectivity: float = 0.5  # Estimated selectivity (probability of passing)
    
    def priority(self) -> float:
        """Calculate priority = selectivity * cost."""
        return self.selectivity * self.cost


class LazyAttributeExtractor:
    """
    Manages lazy extraction of attributes with cost-based ordering.
    """
    
    def __init__(self, config: Dict, selectivity_estimator: Optional[SelectivityEstimator] = None):
        """
        Initialize lazy attribute extractor.
        
        Args:
            config: Configuration dictionary with lazy_attr settings
            selectivity_estimator: Optional selectivity estimator (for Experiment 4)
        """
        lazy_config = config.get("lazy_attr", {})
        self.enabled = lazy_config.get("enabled", False)
        self.use_runtime_selectivity = lazy_config.get("use_runtime_selectivity", False)
        self.default_costs = lazy_config.get("costs", {})
        
        self.selectivity_estimator = selectivity_estimator
        
        if not self.enabled:
            return
        
        logging.info(f"[{self.__class__.__name__}:__init__] Lazy extraction enabled: "
                    f"use_runtime_selectivity={self.use_runtime_selectivity}")
    
    def create_attribute_filters(self, schema: List[Dict], 
                                 extract_fns: Dict[str, Callable],
                                 filter_conditions: Dict[str, Callable]) -> List[AttributeFilter]:
        """
        Create attribute filters from schema.
        
        Args:
            schema: Schema list with attributes
            extract_fns: Dictionary mapping attribute names to extraction functions
            filter_conditions: Dictionary mapping attribute names to filter functions
            
        Returns:
            List of AttributeFilter objects
        """
        filters = []
        
        for table_schema in schema:
            table_name = table_schema.get("Schema Name", "")
            attrs = table_schema.get("Attributes", [])
            
            for attr in attrs:
                if isinstance(attr, dict):
                    attr_name = attr.get("Attribute Name", "")
                elif isinstance(attr, str):
                    attr_name = attr
                else:
                    continue
                
                if not attr_name:
                    continue
                
                # Get extraction function and filter condition
                extract_fn = extract_fns.get(attr_name)
                filter_cond = filter_conditions.get(attr_name)
                
                if not extract_fn or not filter_cond:
                    continue
                
                # Get cost
                cost = self.default_costs.get(attr_name, 1.0)
                
                # Get selectivity
                selectivity = 0.5  # Default
                if self.selectivity_estimator and self.use_runtime_selectivity:
                    selectivity = self.selectivity_estimator.get_mean_selectivity(attr_name)
                
                filter_obj = AttributeFilter(
                    name=attr_name,
                    extract_fn=extract_fn,
                    filter_condition=filter_cond,
                    cost=cost,
                    selectivity=selectivity
                )
                
                filters.append(filter_obj)
        
        return filters
    
    def sort_by_priority(self, filters: List[AttributeFilter], 
                       use_thompson: bool = False) -> List[AttributeFilter]:
        """
        Sort filters by priority (selectivity * cost), descending.
        
        Args:
            filters: List of AttributeFilter objects
            use_thompson: If True, use Thompson Sampling for selectivity
            
        Returns:
            Sorted list of filters
        """
        if not self.enabled:
            return filters
        
        # Update selectivities if using runtime estimation
        if self.selectivity_estimator and self.use_runtime_selectivity:
            attr_names = [f.name for f in filters]
            selectivities = self.selectivity_estimator.get_all_selectivities(
                attr_names, use_thompson=use_thompson
            )
            
            for filter_obj in filters:
                filter_obj.selectivity = selectivities.get(filter_obj.name, filter_obj.selectivity)
        
        # Sort by priority (descending)
        sorted_filters = sorted(filters, key=lambda f: f.priority(), reverse=True)
        
        logging.debug(f"[{self.__class__.__name__}:sort_by_priority] "
                     f"Sorted {len(sorted_filters)} filters by priority")
        
        return sorted_filters
    
    def extract_lazy(self, row_context: Dict, filters: List[AttributeFilter]) -> Tuple[Dict, Dict]:
        """
        Extract attributes lazily, stopping early if filter fails.
        
        Args:
            row_context: Context dictionary for the row (e.g., document, schema)
            filters: List of AttributeFilter objects (should be pre-sorted)
            
        Returns:
            (extracted_values, stats) tuple where:
            - extracted_values: Dictionary of extracted attribute values
            - stats: Statistics about extraction (num_extracted, stopped_early, etc.)
        """
        if not self.enabled:
            # Extract all attributes
            extracted_values = {}
            for filter_obj in filters:
                try:
                    value = filter_obj.extract_fn(row_context)
                    extracted_values[filter_obj.name] = value
                except Exception as e:
                    logging.warning(f"[{self.__class__.__name__}:extract_lazy] "
                                   f"Failed to extract {filter_obj.name}: {e}")
                    extracted_values[filter_obj.name] = None
            
            return extracted_values, {
                "num_extracted": len(extracted_values),
                "stopped_early": False
            }
        
        extracted_values = {}
        stats = {
            "num_extracted": 0,
            "stopped_early": False,
            "stopped_at": None
        }
        
        # Extract attributes in priority order
        for i, filter_obj in enumerate(filters):
            try:
                # Extract value
                value = filter_obj.extract_fn(row_context)
                extracted_values[filter_obj.name] = value
                stats["num_extracted"] += 1
                
                # Check filter condition
                if not filter_obj.filter_condition(value):
                    # Filter failed, stop early
                    stats["stopped_early"] = True
                    stats["stopped_at"] = filter_obj.name
                    
                    # Update selectivity estimator if available
                    if self.selectivity_estimator:
                        self.selectivity_estimator.update_selectivity(filter_obj.name, passed=False)
                    
                    logging.debug(f"[{self.__class__.__name__}:extract_lazy] "
                                 f"Stopped early at {filter_obj.name} (filter failed)")
                    break
                
                # Update selectivity estimator if available
                if self.selectivity_estimator:
                    self.selectivity_estimator.update_selectivity(filter_obj.name, passed=True)
                    
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}:extract_lazy] "
                               f"Failed to extract {filter_obj.name}: {e}")
                extracted_values[filter_obj.name] = None
                # Continue to next attribute even if extraction fails
        
        return extracted_values, stats

