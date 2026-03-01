"""Basic evaluation framework for ReDD project.

This module provides the abstract base class for all evaluation tasks,
defining the common interface and utility methods.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod

from ..utils.utils import is_null
from ..data_loader import create_data_loader, DataLoaderBase


class EvalBasic(ABC):
    """Abstract base class for evaluation tasks.
    
    This class provides common functionality for evaluation including:
    - Data validation
    - Basic statistics computation (TP, FP, FN, TN)
    - Metrics calculation (precision, recall, F1)
    - Result persistence (save/load JSON)
    
    Subclasses must implement:
    - __call__: Main evaluation entry point
    - compute_stat: Task-specific statistics computation (optional override)
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        data_loader: Optional[DataLoaderBase] = None
    ):
        """Initialize the basic evaluator.
        
        Args:
            config: Configuration dictionary containing evaluation parameters.
                   Expected keys depend on the specific evaluation task.
            data_loader: Optional data loader instance. If not provided,
                        subclasses should create task-specific loaders.
        """
        self.config = config
        self.data_loader = data_loader
        
        # Data storage for evaluation
        self.prediction_data: Optional[List[Dict[str, Any]]] = None
        self.gt_data: Optional[List[Dict[str, Any]]] = None
        
        logging.info(f"[{self.__class__.__name__}:__init__] Initialized evaluator with config keys: {list(config.keys())}")
    
    @abstractmethod
    def __call__(self, *args, **kwargs) -> None:
        """Main evaluation method to be implemented by subclasses.
        
        This method should:
        1. Load or receive datasets to evaluate
        2. Prepare prediction and ground truth data
        3. Compute evaluation metrics
        4. Display and save results
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        logging.error(f"[{self.__class__.__name__}:__call__] "
                     f"{self.__class__.__name__}.__call__ must be implemented by subclasses")
        raise NotImplementedError(f"{self.__class__.__name__}.__call__ must be implemented by subclasses")

    def compute_stat(self) -> Optional[Tuple[int, int, int, int]]:
        """Compute evaluation statistics.
        
        This method computes basic classification metrics:
        - True Positives (TP): Prediction is non-null and matches ground truth
        - False Positives (FP): Prediction is non-null but doesn't match ground truth
        - False Negatives (FN): Prediction is null but ground truth is non-null
        - True Negatives (TN): Prediction is null and ground truth is also null
        
        Subclasses should override this method for task-specific evaluation logic
        (e.g., attribute-level evaluation, semantic comparison).
        
        Returns:
            Tuple of (true_positives, false_positives, false_negatives, true_negatives)
            or None if data validation fails
        """
        if not self._validate_data():
            return None
        
        return self._compute_basic_stats()
    
    def _validate_data(self) -> bool:
        """Validate that prediction and ground truth data are properly loaded.
        
        Checks:
        1. Both prediction_data and gt_data are not None/empty
        2. Both have the same length (paired evaluation)
        
        Returns:
            True if data is valid and ready for evaluation, False otherwise
        """
        if not self.prediction_data or not self.gt_data:
            logging.error(f"[{self.__class__.__name__}:_validate_data] No data loaded. "
                         f"prediction_data: {bool(self.prediction_data)}, gt_data: {bool(self.gt_data)}")
            return False
            
        pred_len = len(self.prediction_data)
        gt_len = len(self.gt_data)
        
        if pred_len != gt_len:
            logging.error(f"[{self.__class__.__name__}:_validate_data] Data length mismatch. "
                         f"Predictions: {pred_len}, Ground truth: {gt_len}")
            return False
            
        logging.info(f"[{self.__class__.__name__}:_validate_data] Data validation passed. "
                    f"Evaluating {pred_len} instances.")
        return True
    
    def _compute_basic_stats(self) -> Tuple[int, int, int, int]:
        """Compute basic TP, FP, FN, TN statistics.
        
        This is a simple implementation that compares table names using
        exact string matching. Subclasses should override compute_stat()
        for more sophisticated evaluation logic (e.g., attribute-level
        comparison, semantic matching, fuzzy matching).
        
        The logic:
        - TP: Both pred and gt are non-null AND table names match
        - FP: Pred is non-null BUT (gt is null OR table names don't match)
        - FN: Pred is null BUT gt is non-null
        - TN: Both pred and gt are null
        
        Returns:
            Tuple of (tp, fp, fn, tn)
        """
        true_positives = false_positives = false_negatives = true_negatives = 0
        
        for predicted, ground_truth in zip(self.prediction_data, self.gt_data):
            pred_table = predicted.get("table")
            gt_table = ground_truth.get("table")
            
            is_pred_non_null = not is_null(pred_table)
            is_gt_non_null = not is_null(gt_table)
            
            if is_pred_non_null and is_gt_non_null:
                # Both have predictions: check if they match
                if pred_table == gt_table:
                    true_positives += 1
                else:
                    false_positives += 1
                    
            elif is_pred_non_null and not is_gt_non_null:
                # Predicted something for irrelevant document
                false_positives += 1
                
            elif not is_pred_non_null and is_gt_non_null:
                # Missed a relevant document
                false_negatives += 1
                
            else:
                # Both null: correctly identified irrelevant document
                true_negatives += 1
        
        logging.info(f"[{self.__class__.__name__}:_compute_basic_stats] "
                    f"TP={true_positives}, FP={false_positives}, FN={false_negatives}, TN={true_negatives}")
        
        return true_positives, false_positives, false_negatives, true_negatives
    
    def compute_recall_precision_f1(
        self, 
        tp: int, 
        fp: int, 
        fn: int
    ) -> Tuple[float, float, float]:
        """Compute recall, precision, and F1 score from confusion matrix counts.
        
        Formulas:
        - Recall = TP / (TP + FN)       [Coverage: how many relevant items were found]
        - Precision = TP / (TP + FP)    [Accuracy: how many predictions were correct]
        - F1 = 2 * (Precision * Recall) / (Precision + Recall)  [Harmonic mean]
        
        Handles edge cases:
        - If TP + FN = 0: recall = 0.0 (no relevant items)
        - If TP + FP = 0: precision = 0.0 (no predictions made)
        - If both = 0: F1 = 0.0
        
        Args:
            tp: True positives count
            fp: False positives count
            fn: False negatives count
            
        Returns:
            Tuple of (recall, precision, f1_score)
        """
        # Compute recall
        recall_denominator = tp + fn
        recall = tp / recall_denominator if recall_denominator > 0 else 0.0
        
        # Compute precision
        precision_denominator = tp + fp
        precision = tp / precision_denominator if precision_denominator > 0 else 0.0
        
        # Compute F1 score
        f1_denominator = recall + precision
        f1_score = 2 * recall * precision / f1_denominator if f1_denominator > 0 else 0.0
        
        logging.debug(f"[{self.__class__.__name__}:compute_recall_precision_f1] "
                     f"Recall={recall:.4f}, Precision={precision:.4f}, F1={f1_score:.4f}")
        
        return recall, precision, f1_score
    
    def compute_accuracy(self, correct: int, total: int) -> float:
        """Compute accuracy as the ratio of correct predictions to total predictions.
        
        Formula:
        - Accuracy = Correct / Total
        
        Args:
            correct: Number of correct predictions
            total: Total number of predictions
            
        Returns:
            Accuracy as a float between 0.0 and 1.0
        """
        if total <= 0:
            logging.warning(f"[{self.__class__.__name__}:compute_accuracy] Total count is {total}, returning 0.0")
            return 0.0
        
        accuracy = correct / total
        logging.debug(f"[{self.__class__.__name__}:compute_accuracy] "
                     f"Accuracy={accuracy:.4f} ({correct}/{total})")
        
        return accuracy

    def save_results(
        self, 
        result_path: Union[str, Path], 
        result_dict: Dict[str, Any], 
        encoding: str = "utf-8"
    ) -> None:
        """Save evaluation results to JSON file.
        
        The file is saved with indentation for readability and with
        ensure_ascii=False to preserve Unicode characters.
        Parent directories are created automatically if they don't exist.
        
        Args:
            result_path: Path to save the results (relative or absolute)
            result_dict: Results dictionary to save
            encoding: File encoding (default: utf-8)
        """
        try:
            result_path = Path(result_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(result_path, "w", encoding=encoding) as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
                
            file_size = result_path.stat().st_size
            logging.info(f"[{self.__class__.__name__}:save_results] "
                        f"Results saved to {result_path} ({file_size} bytes, {len(result_dict)} entries)")
                        
        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:save_results] "
                         f"Failed to save results to {result_path}: {e}")
            raise
    
    def load_json(
        self, 
        file_path: Union[str, Path], 
        encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """Load JSON data from file.
        
        Args:
            file_path: Path to the JSON file (relative or absolute)
            encoding: File encoding (default: utf-8)
            
        Returns:
            Loaded JSON data as dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logging.error(f"[{self.__class__.__name__}:load_json] File not found: {file_path}")
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        try:
            with open(file_path, "r", encoding=encoding) as f:
                data = json.load(f)
                
            entry_count = len(data) if isinstance(data, (dict, list)) else "N/A"
            logging.info(f"[{self.__class__.__name__}:load_json] "
                        f"Loaded {file_path} ({entry_count} entries)")
            
            return data
            
        except json.JSONDecodeError as e:
            logging.error(f"[{self.__class__.__name__}:load_json] "
                         f"Invalid JSON in file {file_path}: {e}")
            raise
        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:load_json] "
                         f"Failed to load {file_path}: {e}")
            raise
    
    def display_metrics(
        self,
        title: str,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        correct: Optional[int] = None,
        total: Optional[int] = None,
        width: int = 80
    ) -> None:
        """Display evaluation metrics in a formatted table.
        
        Args:
            title: Title to display (e.g., "Query Q1 Results")
            tp: True positives
            fp: False positives
            fn: False negatives
            tn: True negatives
            correct: Optional number of correct predictions for accuracy
            total: Optional total number of predictions for accuracy
            width: Width of the display table (default: 80)
        """
        recall, precision, f1 = self.compute_recall_precision_f1(tp, fp, fn)
        
        print("\n" + "=" * width)
        print(f"{title}")
        print("-" * width)
        print(f"{'Metric':<20}{'Value'}")
        print("-" * width)
        print(f"{'True Positives':<20}{tp}")
        print(f"{'False Positives':<20}{fp}")
        print(f"{'False Negatives':<20}{fn}")
        print(f"{'True Negatives':<20}{tn}")
        print(f"{'Recall':<20}{recall:.4f}")
        print(f"{'Precision':<20}{precision:.4f}")
        print(f"{'F1 Score':<20}{f1:.4f}")
        
        if correct is not None and total is not None:
            accuracy = self.compute_accuracy(correct, total)
            print(f"{'Accuracy':<20}{correct}/{total} = {accuracy:.4f}")
            
        print("=" * width)
