"""
Abstract Base Class for Data Population.

This module defines the abstract interface that all data population
implementations must follow.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class DataPopulator(ABC):
    """
    Abstract base class for data population from unstructured documents.
    
    This class defines the interface and common utilities for data population.
    Subclasses must implement __call__ and __str__ methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data populator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.loader = None  # Data loader will be set during processing
    
    @abstractmethod
    def __call__(self, dataset_task_list: Optional[List[str]] = None):
        """
        Main entry point for data population.
        
        Args:
            dataset_task_list: Optional list of dataset/task paths to process
        """
        logging.error(f"[{self.__class__.__name__}:__call__] Subclasses must implement __call__")
        raise NotImplementedError("Subclasses must implement __call__")
    
    @abstractmethod
    def __str__(self) -> str:
        """
        String representation of the data populator.
        
        Returns:
            String describing the data populator
        """
        logging.error(f"[{self.__class__.__name__}:__str__] Subclasses must implement __str__")
        raise NotImplementedError("Subclasses must implement __str__")
    
    # Common utility methods
    
    def save_results(self, res_path: str, res_dict: Dict[str, Any], encoding: str = "utf-8"):
        """
        Save results to JSON file.
        
        Args:
            res_path: Path to save results
            res_dict: Dictionary of results to save
            encoding: File encoding (default: utf-8)
        """
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        with open(res_path, "w", encoding=encoding) as f:
            json.dump(res_dict, f, indent=2)
        logging.debug(f"[{self.__class__.__name__}:save_results] Saved results to {res_path}")
    
    def load_json(self, file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            file_path: Path to JSON file
            encoding: File encoding (default: utf-8)
            
        Returns:
            Dictionary loaded from JSON file
        """
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
    
    def load_processed_res(self, res_path: str) -> Dict[str, Any]:
        """
        Load processed results from file.
        
        Args:
            res_path: Path to results file
            
        Returns:
            Dictionary of results, or empty dict if file doesn't exist
        """
        res_dict = dict()
        if os.path.exists(res_path):
            res_dict = self.load_json(res_path)
        return res_dict
