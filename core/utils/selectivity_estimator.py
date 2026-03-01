"""
Selectivity Estimation & Runtime Tuning

Provides selectivity estimates for attribute filters using:
1. LLM cold-start estimation
2. Thompson Sampling for runtime tuning
"""
import json
import logging
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from scipy.stats import beta

from ..utils.prompt_utils import PromptGPT, PromptDeepSeek, get_api_key


@dataclass
class SelectivityConfig:
    """Configuration for selectivity estimation."""
    enabled: bool = False
    init_mode: str = "llm"  # "llm" or "uniform"
    prior_strength: float = 3.0
    thompson_enabled: bool = False
    cache_path: Optional[str] = None
    mode: str = "cgpt"  # LLM mode for estimation


class SelectivityEstimator:
    """
    Estimates selectivity of attribute filters using LLM and Thompson Sampling.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize selectivity estimator.
        
        Args:
            config: Configuration dictionary with selectivity_estimation settings
        """
        est_config = config.get("selectivity_estimation", {})
        self.config = SelectivityConfig(
            enabled=est_config.get("enabled", False),
            init_mode=est_config.get("init_mode", "llm"),
            prior_strength=est_config.get("prior_strength", 3.0),
            thompson_enabled=est_config.get("thompson", {}).get("enabled", False),
            cache_path=est_config.get("cache_path"),
            mode=config.get("mode", "cgpt")
        )
        
        if not self.config.enabled:
            return
        
        # Initialize LLM prompt for cold-start estimation
        if self.config.init_mode == "llm":
            try:
                api_key = get_api_key(config, self.config.mode, None)
                if self.config.mode == "deepseek":
                    self.llm_prompt = PromptDeepSeek(
                        self.config.mode,
                        "prompts/selectivity_estimation.txt",  # TODO: create this prompt
                        llm_model=config.get("llm_model", "deepseek-chat"),
                        api_key=api_key
                    )
                else:
                    self.llm_prompt = PromptGPT(
                        self.config.mode,
                        "prompts/selectivity_estimation.txt",
                        llm_model=config.get("llm_model", "gpt-4o"),
                        api_key=api_key
                    )
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}:__init__] "
                               f"Failed to initialize LLM prompt: {e}, using uniform init")
                self.config.init_mode = "uniform"
        
        # Per-attribute Beta distributions: (alpha, beta) for Beta(alpha, beta)
        self.beta_params: Dict[str, Tuple[float, float]] = {}
        
        # Load cached parameters if available
        self._load_cache()
        
        logging.info(f"[{self.__class__.__name__}:__init__] Selectivity estimation enabled: "
                    f"init_mode={self.config.init_mode}, "
                    f"thompson={self.config.thompson_enabled}")
    
    def _load_cache(self):
        """Load cached Beta parameters from file."""
        if not self.config.cache_path or not self.config.cache_path:
            return
        
        try:
            with open(self.config.cache_path, "r") as f:
                cache_data = json.load(f)
                for attr_name, params in cache_data.get("beta_params", {}).items():
                    self.beta_params[attr_name] = (params["alpha"], params["beta"])
            logging.info(f"[{self.__class__.__name__}:_load_cache] "
                        f"Loaded {len(self.beta_params)} cached parameters")
        except FileNotFoundError:
            logging.debug(f"[{self.__class__.__name__}:_load_cache] "
                         f"Cache file not found: {self.config.cache_path}")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}:_load_cache] "
                          f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save Beta parameters to cache file."""
        if not self.config.cache_path:
            return
        
        try:
            cache_data = {
                "beta_params": {
                    attr_name: {"alpha": alpha, "beta": beta}
                    for attr_name, (alpha, beta) in self.beta_params.items()
                }
            }
            with open(self.config.cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
            logging.debug(f"[{self.__class__.__name__}:_save_cache] "
                         f"Saved {len(self.beta_params)} parameters to cache")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}:_save_cache] "
                          f"Failed to save cache: {e}")
    
    def estimate_llm_cold_start(self, attribute_name: str, filter_condition: str) -> float:
        """
        Estimate selectivity using LLM cold-start.
        
        Args:
            attribute_name: Name of the attribute
            filter_condition: Description of the filter condition
            
        Returns:
            Estimated selectivity (0.0 to 1.0)
        """
        if self.config.init_mode != "llm" or not hasattr(self, "llm_prompt"):
            return 0.5  # Default uniform
        
        try:
            prompt_input = {
                "Attribute": attribute_name,
                "Filter Condition": filter_condition,
                "Task": "Estimate what fraction of records will pass this filter (selectivity between 0.0 and 1.0)"
            }
            
            result_str = self.llm_prompt(json.dumps(prompt_input)).strip()
            
            # Try to parse as JSON or extract number
            try:
                result = json.loads(result_str)
                selectivity = float(result.get("selectivity", result.get("value", 0.5)))
            except (json.JSONDecodeError, KeyError, ValueError):
                # Try to extract number from text
                import re
                numbers = re.findall(r'\d+\.?\d*', result_str)
                if numbers:
                    selectivity = float(numbers[0])
                    if selectivity > 1.0:
                        selectivity = selectivity / 100.0  # Assume percentage
                else:
                    # Try keyword matching
                    result_lower = result_str.lower()
                    if "high" in result_lower:
                        selectivity = 0.8
                    elif "low" in result_lower:
                        selectivity = 0.2
                    elif "medium" in result_lower:
                        selectivity = 0.5
                    else:
                        selectivity = 0.5
            
            # Clamp to [0, 1]
            selectivity = max(0.0, min(1.0, selectivity))
            
            logging.info(f"[{self.__class__.__name__}:estimate_llm_cold_start] "
                        f"Attribute {attribute_name}: selectivity={selectivity:.4f}")
            
            return selectivity
            
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}:estimate_llm_cold_start] "
                           f"LLM estimation failed: {e}, using default 0.5")
            return 0.5
    
    def initialize_attribute(self, attribute_name: str, filter_condition: str, 
                            initial_selectivity: Optional[float] = None) -> float:
        """
        Initialize Beta distribution for an attribute.
        
        Args:
            attribute_name: Name of the attribute
            filter_condition: Description of the filter condition
            initial_selectivity: Optional initial selectivity estimate
            
        Returns:
            Initial selectivity estimate
        """
        if attribute_name in self.beta_params:
            # Already initialized, return current mean
            alpha, beta = self.beta_params[attribute_name]
            return alpha / (alpha + beta)
        
        # Get initial estimate
        if initial_selectivity is not None:
            mu0 = initial_selectivity
        elif self.config.init_mode == "llm":
            mu0 = self.estimate_llm_cold_start(attribute_name, filter_condition)
        else:
            mu0 = 0.5  # Uniform prior
        
        # Initialize Beta parameters
        # Beta(alpha, beta) with mean = alpha / (alpha + beta) = mu0
        # Use prior_strength as total "pseudo-counts"
        alpha = mu0 * self.config.prior_strength
        beta_param = (1 - mu0) * self.config.prior_strength
        
        self.beta_params[attribute_name] = (alpha, beta_param)
        
        logging.debug(f"[{self.__class__.__name__}:initialize_attribute] "
                     f"Initialized {attribute_name}: Beta({alpha:.2f}, {beta_param:.2f}), "
                     f"mean={mu0:.4f}")
        
        return mu0
    
    def sample_selectivity(self, attribute_name: str) -> float:
        """
        Sample selectivity from Beta distribution (Thompson Sampling).
        
        Args:
            attribute_name: Name of the attribute
            
        Returns:
            Sampled selectivity value
        """
        if not self.config.thompson_enabled:
            # Return mean instead of sample
            return self.get_mean_selectivity(attribute_name)
        
        if attribute_name not in self.beta_params:
            return 0.5  # Default if not initialized
        
        alpha, beta_param = self.beta_params[attribute_name]
        sampled = beta.rvs(alpha, beta_param)
        
        return max(0.0, min(1.0, sampled))
    
    def get_mean_selectivity(self, attribute_name: str) -> float:
        """
        Get mean selectivity (expected value) for an attribute.
        
        Args:
            attribute_name: Name of the attribute
            
        Returns:
            Mean selectivity
        """
        if attribute_name not in self.beta_params:
            return 0.5  # Default
        
        alpha, beta_param = self.beta_params[attribute_name]
        return alpha / (alpha + beta_param)
    
    def update_selectivity(self, attribute_name: str, passed: bool):
        """
        Update Beta distribution after observing filter result.
        
        Args:
            attribute_name: Name of the attribute
            passed: True if filter passed (row kept), False if dropped
        """
        if attribute_name not in self.beta_params:
            # Initialize if not exists
            self.beta_params[attribute_name] = (
                self.config.prior_strength * 0.5,
                self.config.prior_strength * 0.5
            )
        
        alpha, beta_param = self.beta_params[attribute_name]
        
        if passed:
            alpha += 1
        else:
            beta_param += 1
        
        self.beta_params[attribute_name] = (alpha, beta_param)
        
        # Periodically save cache
        if len(self.beta_params) % 10 == 0:
            self._save_cache()
    
    def get_all_selectivities(self, attribute_names: List[str], 
                             use_thompson: bool = False) -> Dict[str, float]:
        """
        Get selectivity estimates for multiple attributes.
        
        Args:
            attribute_names: List of attribute names
            use_thompson: If True, use Thompson Sampling; else use mean
            
        Returns:
            Dictionary mapping attribute names to selectivity estimates
        """
        selectivities = {}
        for attr_name in attribute_names:
            if use_thompson and self.config.thompson_enabled:
                selectivities[attr_name] = self.sample_selectivity(attr_name)
            else:
                selectivities[attr_name] = self.get_mean_selectivity(attr_name)
        
        return selectivities

