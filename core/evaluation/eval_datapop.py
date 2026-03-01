import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils import constants
from ..utils.constants import (
    PREDICTION_KEY, GROUND_TRUTH_KEY,
    ATTRIBUTE_NAME_KEY, ATTRIBUTE_VALUE_KEY,
    PATH_TEMPLATES,
)
from ..utils.utils import is_null
from .eval_basic import EvalBasic
from ..utils.prompt_utils import PromptGPT, PromptDeepSeek, get_api_key
from ..data_loader import create_data_loader, DataLoaderBase


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    correct_count: int = 0
    total_count: int = 0
    doc_stats: Dict[str, Dict[str, Any]] = None
    attr_stats: Dict[str, Dict[str, int]] = None  # per-attribute statistics

    def __post_init__(self):
        if self.doc_stats is None:
            self.doc_stats = {}
        if self.attr_stats is None:
            self.attr_stats = {}

    def to_tuple(self) -> Tuple[int, int, int, int, int, int, Dict[str, Any], Dict[str, Dict[str, int]]]:
        """Convert to tuple format for backward compatibility."""
        return (
            self.true_positives,
            self.false_positives,
            self.false_negatives,
            self.true_negatives,
            self.correct_count,
            self.total_count,
            self.doc_stats,
            self.attr_stats
        )


class EvalDataPop(EvalBasic):
    """Data population evaluation with comprehensive metrics and LLM-based semantic comparison."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        data_loader: DataLoaderBase = None, 
        api_key: Optional[str] = None
    ):
        """Initialize data population evaluator.
        
        Args:
            config: Configuration dictionary containing evaluation parameters
            data_loader: Optional data loader instance (usually not provided, created per dataset)
            api_key: Optional API key for LLM-based evaluation

        TODO: name map generation not implemented yet.
        """
        super().__init__(config, data_loader)
        
        # Data loader configuration
        self.loader_type = config.get("data_loader_type", "sqlite")
        self.loader_config = config.get("data_loader_config", {})

        # Evaluation configuration
        eval_config = config.get("eval", {})
        self.res_param_str = config.get("res_param_str", "default")
        self.name_map: Optional[Dict[str, Any]] = None
        
        # Support both single LLM and committee voting
        self.committee_prompts: List[Dict[str, Any]] = []
        self.prompts: Dict[str, Any] = {}
        
        if "committee" in eval_config:
            # Committee voting mode
            self._initialize_committee(eval_config, api_key)
        else:
            # Single LLM mode (backward compatible)
            self.eval_mode = eval_config.get("mode", "deepseek")
            self.eval_api_key = get_api_key(eval_config, self.eval_mode, api_key)
            if "prompts" in eval_config:
                self._initialize_prompts(eval_config)
    
    def _initialize_prompts(self, eval_config: Dict[str, Any]) -> None:
        """ Initialize LLM prompts for semantic comparison (single LLM mode). """
        try:
            for prompt_name in eval_config["prompts"]:
                if self.eval_mode == "deepseek":
                    self.eval_llm_model = eval_config.get("llm_model", "deepseek-chat")
                    self.prompts[prompt_name] = PromptDeepSeek(
                        self.eval_mode,
                        eval_config["prompts"][prompt_name],
                        llm_model=self.eval_llm_model,
                        api_key=self.eval_api_key
                    )
                elif self.eval_mode == "cgpt":
                    self.eval_llm_model = eval_config.get("llm_model", "gpt-4o")
                    self.prompts[prompt_name] = PromptGPT(
                        self.eval_mode,
                        eval_config["prompts"][prompt_name],
                        llm_model=self.eval_llm_model,
                        api_key=self.eval_api_key
                    )
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}:_initialize_prompts] Failed to initialize prompts: {e}")
            self.prompts = {}
    
    def _initialize_committee(self, eval_config: Dict[str, Any], api_key: Optional[str] = None) -> None:
        """Initialize committee voting with multiple LLMs."""
        committee_config = eval_config["committee"]
        prompt_paths = eval_config["prompts"]
        
        for member_config in committee_config:
            mode = member_config["mode"]
            llm_model = member_config["llm_model"]
            member_api_key = get_api_key(member_config, mode, api_key)
            
            member_prompts = {}
            for prompt_name, prompt_path in prompt_paths.items():
                if mode == "cgpt":
                    member_prompts[prompt_name] = PromptGPT(mode, prompt_path, llm_model, member_api_key)
                elif mode == "deepseek":
                    member_prompts[prompt_name] = PromptDeepSeek(mode, prompt_path, llm_model, member_api_key)
            
            self.committee_prompts.append({
                "mode": mode,
                "llm_model": llm_model,
                "prompts": member_prompts
            })
        
        logging.info(f"[{self.__class__.__name__}:_initialize_committee] Committee: {len(self.committee_prompts)} members")
            
    def __call__(self, dataset_task_list: Optional[List[str]] = None) -> None:
        """Evaluate multiple dataset tasks.
        
        Args:
            dataset_task_list: List of dataset tasks to evaluate. If None, uses config or default.
        """
        if dataset_task_list is None:
            if "exp_dataset_task_list" in self.config:
                dataset_task_list = self.config["exp_dataset_task_list"]
            else:
                logging.warning(f"[{self.__class__.__name__}:__call__] No dataset tasks specified, using default SPIDER_DATASET_TASK_LIST")
                dataset_task_list = constants.SPIDER_DATASET_TASK_LIST
                
        total_tasks = len(dataset_task_list)
        logging.info(f"[{self.__class__.__name__}:__call__] Evaluating {total_tasks} tasks: {dataset_task_list}")
        
        for dataset_name in dataset_task_list:
            # Separate data path and output path
            data_path = Path(self.config.get("data_main", self.config["out_main"])) / dataset_name
            out_root = Path(self.config["out_main"]) / dataset_name
            logging.info(f"[{self.__class__.__name__}:__call__] Start evaluating dataset: data_path={data_path}, out_root={out_root}")
            self._evaluate_dataset(dataset_name, data_path, out_root)
                
    def _evaluate_dataset(self, dataset_name: str, data_path: str | Path, out_root: str | Path) -> None:
        """Evaluate a single dataset.
        
        Args:
            dataset_name: Name of the dataset
            data_path: Path to dataset directory
            out_root: Path to output directory
        """
        self.data_path = Path(data_path)
        self.out_root = Path(out_root)
        
        if not self.data_path.exists():
            logging.error(f"[{self.__class__.__name__}:_evaluate_dataset] Data directory not found: {self.data_path}")
            return
        
        if not self.out_root.exists():
            logging.error(f"[{self.__class__.__name__}:_evaluate_dataset] Output directory not found: {self.out_root}")
            return
            
        # Create data loader based on configuration
        loader = create_data_loader(
            data_path=self.data_path,
            loader_type=self.loader_type,
            loader_config=self.loader_config
        )
        logging.info(f"[{self.__class__.__name__}:_evaluate_dataset] Created {loader.__class__.__name__} for {self.data_path}")
        
        try:
            query_dict = loader.load_query_dict()
        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:_evaluate_dataset] Failed to load query dict for {dataset_name}: {e}")
            return
        
        if not query_dict:
            logging.warning(f"[{self.__class__.__name__}:_evaluate_dataset] No queries found in dataset {self.data_path}")
            return
            
        logging.info(f"[{self.__class__.__name__}:_evaluate_dataset] Evaluating {len(query_dict)} queries in dataset {self.data_path.name}")
            
        for query_id in query_dict:
            try:
                self._evaluate_query(loader, dataset_name, query_id, query_dict[query_id])
            except Exception as e:
                logging.error(f"[{self.__class__.__name__}:_evaluate_dataset] Failed to evaluate query {query_id} in dataset {dataset_name}: {e}")
                continue
                
    def _evaluate_query(
        self, 
        loader: DataLoaderBase, 
        dataset_name: str, 
        query_id: str, 
        query_info: Dict[str, Any]
    ) -> None:
        """Evaluate a single query.
        
        Args:
            loader: Data loader instance
            dataset_name: Dataset name
            query_id: Query ID
            query_info: Query information dictionary
        """
        query = query_info.get("query", "")
        result_path = self.out_root / PATH_TEMPLATES.data_population_result(query_id, self.res_param_str)
        
        try:
            result_dict = self.load_json(result_path)
        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:_evaluate_query] Failed to load results from {result_path}: {e}")
            return
            
        logging.info(f"[{self.__class__.__name__}:_evaluate_query] Start evaluating query {query_id}: {query}")
        
        # Load name mapping
        self.name_map = self._load_or_generate_mapping(query_id)
        
        # Prepare evaluation data
        self.prediction_data, self.gt_data = self._prepare_evaluation_data(loader, result_dict)
        
        # Compute and save results
        stats = self.compute_statistics()
        if stats is None:
            logging.error(f"[{self.__class__.__name__}:_evaluate_query] Failed to compute statistics for query {query_id}")
            return
            
        tp, fp, fn, tn, correct, total, doc_stats, attr_stats = stats
        
        # Display results
        self._display_results(dataset_name, query_id, tp, fp, fn, tn, correct, total, attr_stats)
        
        # Save detailed results with attribute statistics
        eval_results = {
            "doc_stats": doc_stats,
            "attr_stats": attr_stats
        }
        eval_path = self.out_root / PATH_TEMPLATES.eval_result(query_id, self.res_param_str)
        self.save_results(eval_path, eval_results)
        logging.info(f"[{self.__class__.__name__}:_evaluate_query] Evaluation results saved to {eval_path}")

    def _prepare_evaluation_data(
        self,
        loader: DataLoaderBase,
        result_dict: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Prepare prediction and ground truth data for evaluation.
        
        Args:
            loader: Data loader instance
            result_dict: Dictionary of prediction results
            
        Returns:
            Tuple of (prediction_data, ground_truth_data)
        """
        prediction_data = []
        ground_truth_data = []
        
        for doc_id in result_dict:
            cur_info = loader.get_doc_info(doc_id)
            if not cur_info:
                logging.warning(f"[{self.__class__.__name__}:_prepare_evaluation_data] doc-{doc_id} not found in ground truth, skipping")
                continue
                
            prediction_data.append({
                "doc_id": doc_id,
                "table": result_dict[doc_id].get("res"),
                "data": result_dict[doc_id].get("data", {})
            })
            
            # Extract ground truth table and data (assuming primary/first record)
            gt_table = None
            gt_data = {}
            
            mappings = cur_info.get("mappings")
            if mappings and len(mappings) > 0:
                gt_table = mappings[0].get("table_name")
                
            # Extract ground truth data from data_records
            records = cur_info.get("data_records")
            if isinstance(records, list) and len(records) > 0:
                # Use the 'data' field from the first record
                gt_data = records[0].get("data", {})
            
            ground_truth_data.append({
                "doc_id": doc_id,
                "table": gt_table,
                "data": gt_data
            })
        
        return prediction_data, ground_truth_data

    def _display_results(
        self,
        dataset_name: str,
        query_id: str,
        tp: int,
        fp: int,
        fn: int,
        tn: int,
        correct: int,
        total: int,
        attr_stats: Dict[str, Dict[str, int]]
    ) -> None:
        """Display evaluation results in formatted output.
        
        Args:
            dataset_name: Dataset name
            query_id: Query ID
            tp: True positives
            fp: False positives
            fn: False negatives
            tn: True negatives
            correct: Number of correct predictions
            total: Total number of predictions
            attr_stats: Per-attribute statistics
        """
        title = f"Dataset: {dataset_name} | Query ID: {query_id}"
        self.display_metrics(title, tp, fp, fn, tn, correct, total)
        self._display_attribute_accuracy(attr_stats)

    def _display_attribute_accuracy(
        self, 
        attr_stats: Dict[str, Dict[str, int]],
        width: int = 80
    ) -> None:
        """Display attribute-level accuracy statistics.
        
        Args:
            attr_stats: Per-attribute statistics with format:
                {
                    "table_name.attr_name": {
                        "correct": int,
                        "total": int,
                        "table": str,
                        "attr": str
                    }
                }
            width: Width of the display table (default: 80)
        """
        if not attr_stats:
            logging.warning(f"[{self.__class__.__name__}:_display_attribute_accuracy] No attribute statistics to display")
            return
        
        # Group attributes by table
        table_groups = {}
        for attr_key, stats in attr_stats.items():
            table_name = stats.get("table", "unknown")
            if table_name not in table_groups:
                table_groups[table_name] = {}
            table_groups[table_name][attr_key] = stats
        
        # Calculate overall attribute-level accuracy
        total_correct = sum(stats["correct"] for stats in attr_stats.values())
        total_attrs = sum(stats["total"] for stats in attr_stats.values())
        overall_accuracy = total_correct / total_attrs if total_attrs > 0 else 0.0
        
        # Display header
        print("\n" + "=" * width)
        print("Attribute-Level Accuracy")
        print("=" * width)
        
        # Display per-table statistics
        for table_name in sorted(table_groups.keys()):
            table_attrs = table_groups[table_name]
            
            # Calculate table-level statistics
            table_correct = sum(stats["correct"] for stats in table_attrs.values())
            table_total = sum(stats["total"] for stats in table_attrs.values())
            table_accuracy = table_correct / table_total if table_total > 0 else 0.0
            
            # Display table name
            print(f"\nTable: {table_name}")
            print("-" * width)
            attr_col_width = width - 32  # Reserve space for numbers
            print(f"{'Attribute':<{attr_col_width}}{'Correct':>10}{'Total':>10}{'Accuracy':>10}")
            print("-" * width)
            
            # Display per-attribute accuracy for this table
            for attr_key in sorted(table_attrs.keys(), key=lambda x: table_attrs[x].get("attr", "")):
                stats = table_attrs[attr_key]
                attr_name = stats.get("attr", attr_key)
                accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
                print(f"{attr_name:<{attr_col_width}}{stats['correct']:>10}{stats['total']:>10}{accuracy:>10.2%}")
            
            # Display table subtotal
            print("-" * width)
            subtotal_label = f"Subtotal ({table_name})"
            print(f"{subtotal_label:<{attr_col_width}}{table_correct:>10}{table_total:>10}{table_accuracy:>10.2%}")
        
        # Display overall accuracy
        print("=" * width)
        overall_label = "Overall (All Tables & Attributes)"
        attr_col_width = width - 32
        print(f"{overall_label:<{attr_col_width}}{total_correct:>10}{total_attrs:>10}{overall_accuracy:>10.2%}")
        print("=" * width)

    def compute_statistics(self) -> Optional[Tuple[int, int, int, int, int, int, Dict[str, Any], Dict[str, Dict[str, int]]]]:
        """Compute detailed evaluation statistics for data population.
        
        Compares prediction and ground truth data to compute:
        - True Positives (TP): Correct attribute extractions
        - False Positives (FP): Incorrect or extra extractions
        - False Negatives (FN): Missing extractions
        - True Negatives (TN): Correctly identified irrelevant documents
        - Accuracy: Document-level correctness
        
        Expected format for pred and gt:
        {
            "doc_id": str,
            "table": str (table name or null),
            "data": {
                "column_name": "value",
                ...
            }
        }
        
        Returns:
            Tuple of (tp, fp, fn, tn, correct_count, total_count, doc_stats)
            or None if validation fails
        """
        if not self._validate_data():
            return None
        
        metrics = EvaluationMetrics()

        for pred, gt in zip(self.prediction_data, self.gt_data):
            doc_id = pred["doc_id"]
            metrics.total_count += 1
            metrics.doc_stats[doc_id] = {"table": True, "attr": {}, "final": False}
            
            # Evaluate document
            is_doc_correct = self._evaluate_document(pred, gt, metrics)
            
            # Update document-level correctness
            if is_doc_correct:
                metrics.correct_count += 1
            metrics.doc_stats[doc_id]["final"] = is_doc_correct

        return metrics.to_tuple()

    def _evaluate_document(
        self,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        metrics: EvaluationMetrics
    ) -> bool:
        """Evaluate a single document prediction.
        
        Args:
            pred: Prediction dictionary
            gt: Ground truth dictionary
            metrics: Metrics accumulator
            
        Returns:
            True if document is correctly predicted, False otherwise
        """
        doc_id = pred["doc_id"]
        
        # Case 1: Document is irrelevant to query (gt table is null)
        if is_null(gt["table"]):
            return self._evaluate_irrelevant_document(pred, metrics)
        
        # Case 2: Document is relevant to query
        return self._evaluate_relevant_document(pred, gt, metrics)

    def _evaluate_irrelevant_document(
        self,
        pred: Dict[str, Any],
        metrics: EvaluationMetrics
    ) -> bool:
        """Evaluate prediction for an irrelevant document.
        
        Args:
            pred: Prediction dictionary
            metrics: Metrics accumulator
            
        Returns:
            True if correctly identified as irrelevant, False otherwise
        """
        doc_id = pred["doc_id"]
        
        if not is_null(pred["table"]):
            # False positive: predicted data for irrelevant document
            non_null_attrs = [attr for attr in pred["data"] if not is_null(pred["data"][attr])]
            metrics.false_positives += len(non_null_attrs)
            metrics.doc_stats[doc_id]["table"] = False
            logging.info(f"[{self.__class__.__name__}:_evaluate_irrelevant_document] false_positives (doc irrelevant): {pred}")
            return True
        else:
            # True negative: correctly identified as irrelevant
            metrics.true_negatives += 1
            return True

    def _evaluate_relevant_document(
        self,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        metrics: EvaluationMetrics
    ) -> bool:
        """Evaluate prediction for a relevant document.
        
        Args:
            pred: Prediction dictionary
            gt: Ground truth dictionary
            metrics: Metrics accumulator
            
        Returns:
            True if all attributes are correct, False otherwise
        """
        doc_id = pred["doc_id"]
        gt_table = gt["table"]
        
        # Get attribute mapping for this table
        attr_map_dict = self.name_map.get("attribute", {}).get(gt_table, {})
        
        # Check table assignment
        table_correct = self._check_table_assignment(pred, gt, metrics)
        
        if not table_correct:
            # OLD VERSION LOGIC: Even if table is wrong, we need to:
            # 1. Check for extra attributes (for FP counting)
            # 2. Mark expected attributes as FN (only if GT value is non-null)
            # 3. Return True only if all GT attributes are null
            
            # First, check for extra attributes
            self._check_extra_attributes(pred, attr_map_dict, metrics)
            
            # Mark attributes as FN (but only if GT value is non-null)
            all_gt_null = self._mark_attributes_as_fn_if_gt_nonnull(attr_map_dict, gt, pred, metrics)
            
            return all_gt_null
        
        # Evaluate each expected attribute
        all_attrs_correct = self._evaluate_attributes(pred, gt, attr_map_dict, metrics)
        
        # Check for extra attributes not in ground truth
        self._check_extra_attributes(pred, attr_map_dict, metrics)
        
        return all_attrs_correct

    def _check_table_assignment(
        self,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        metrics: EvaluationMetrics
    ) -> bool:
        """Check if table assignment is correct.
        
        Args:
            pred: Prediction dictionary
            gt: Ground truth dictionary
            metrics: Metrics accumulator
            
        Returns:
            True if table assignment is correct, False otherwise
        """
        doc_id = pred["doc_id"]
        pred_table = pred["table"]
        gt_table = gt["table"]
        
        # Check if prediction has no table assignment
        if is_null(pred_table):
            metrics.doc_stats[doc_id]["table"] = False
            return False
        
        # Check if predicted table name matches ground truth
        table_map = self.name_map.get("table", {})
        if pred_table not in table_map or table_map[pred_table] != gt_table:
            metrics.doc_stats[doc_id]["table"] = False
            return False
        
        return True

    def _mark_attributes_as_fn_if_gt_nonnull(
        self,
        attr_map_dict: Dict[str, Any],
        gt: Dict[str, Any],
        pred: Dict[str, Any],
        metrics: EvaluationMetrics
    ) -> bool:
        """Mark ground truth attributes as false negatives only if GT value is non-null.
        
        This follows OLD VERSION LOGIC: when table is wrong/null, only mark FN for
        non-null GT attributes. If all GT attributes are null, the document can still
        be considered "correct".
        
        NOTE: In the old version, when table is wrong, attr stats remain True (initialized value),
        and only the table-level flag is set to False. This implementation maintains that behavior.
        
        Args:
            attr_map_dict: Attribute mapping dictionary
            gt: Ground truth dictionary
            pred: Prediction dictionary
            metrics: Metrics accumulator
            
        Returns:
            True if all GT attributes are null, False otherwise
        """
        doc_id = pred["doc_id"]
        gt_table = gt["table"]
        all_gt_null = True
        
        for attr in attr_map_dict:
            attr_mapped = attr_map_dict[attr]
            # OLD VERSION: Initialize attr as True (not evaluated when table is wrong)
            metrics.doc_stats[doc_id]["attr"][attr] = True
            
            # Use table_name.attr_name as key to avoid collision across tables
            attr_key = f"{gt_table}.{attr}"
            
            # Initialize attribute statistics if not exists
            if attr_key not in metrics.attr_stats:
                metrics.attr_stats[attr_key] = {"correct": 0, "total": 0, "table": gt_table, "attr": attr}
            
            # Get ground truth value
            gt_value = self._get_gt_value(attr, attr_mapped, gt)
            
            # Update attribute statistics
            metrics.attr_stats[attr_key]["total"] += 1
            
            # OLD VERSION LOGIC: Only mark FN and set table=False if GT is non-null
            # Note: attr[attr] remains True, only table flag is set to False
            if not is_null(gt_value):
                all_gt_null = False
                metrics.false_negatives += 1
                metrics.doc_stats[doc_id]["table"] = False
                # Attribute is incorrect due to wrong table
                # (attr_stats already incremented total, correct remains 0)
                
                # Determine the log message based on whether pred table is null or wrong
                if is_null(pred["table"]):
                    logging.info(f"[{self.__class__.__name__}:_mark_attributes_as_fn_if_gt_nonnull] false_negatives (doc null) of {attr}: {pred}; {gt}")
                else:
                    logging.info(f"[{self.__class__.__name__}:_mark_attributes_as_fn_if_gt_nonnull] false_negatives (table name incorrect) of {attr}: {pred}; {gt}")
            else:
                # GT attribute is null, so we consider it "correct" (no extraction expected)
                metrics.attr_stats[attr_key]["correct"] += 1
        
        return all_gt_null

    def _evaluate_attributes(
        self,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        attr_map_dict: Dict[str, Any],
        metrics: EvaluationMetrics
    ) -> bool:
        """Evaluate all expected attributes for a document.
        
        Args:
            pred: Prediction dictionary
            gt: Ground truth dictionary
            attr_map_dict: Attribute mapping dictionary
            metrics: Metrics accumulator
            
        Returns:
            True if all attributes are correct, False otherwise
        """
        doc_id = pred["doc_id"]
        gt_table = gt["table"]
        all_correct = True
        
        for attr in attr_map_dict:
            metrics.doc_stats[doc_id]["attr"][attr] = True
            
            # Use table_name.attr_name as key to avoid collision across tables
            attr_key = f"{gt_table}.{attr}"
            
            # Initialize attribute statistics if not exists
            if attr_key not in metrics.attr_stats:
                metrics.attr_stats[attr_key] = {"correct": 0, "total": 0, "table": gt_table, "attr": attr}
            
            # Get ground truth value and attribute name
            attr_mapped = attr_map_dict[attr]
            gt_value = self._get_gt_value(attr, attr_mapped, gt)
            gt_attr = self._get_gt_attr_name(attr_mapped)
            
            # Evaluate this attribute
            attr_correct = self._evaluate_single_attribute(
                pred, gt, attr, gt_attr, gt_value, metrics
            )
            
            # Update attribute statistics
            metrics.attr_stats[attr_key]["total"] += 1
            if attr_correct:
                metrics.attr_stats[attr_key]["correct"] += 1
            else:
                all_correct = False
                metrics.doc_stats[doc_id]["attr"][attr] = False
        
        return all_correct

    def _get_gt_value(self, attr: str, attr_mapped: Any, gt: Dict[str, Any]) -> str:
        """Get ground truth value for an attribute.
        
        Args:
            attr: Attribute name in prediction
            attr_mapped: Mapped attribute name(s) in ground truth
            gt: Ground truth dictionary
            
        Returns:
            Ground truth value as lowercase string
        """
        if isinstance(attr_mapped, list):
            # One-to-many mapping: concatenate multiple values
            gt_value = " ".join([
                str(gt["data"].get(i)) 
                for i in attr_mapped 
                if gt["data"].get(i) is not None
            ])
        else:
            # One-to-one mapping
            gt_value = str(gt["data"].get(attr_mapped, ""))
        
        return gt_value.lower()

    def _get_gt_attr_name(self, attr_mapped: Any) -> str:
        """Get ground truth attribute name.
        
        Args:
            attr_mapped: Mapped attribute name(s)
            
        Returns:
            Attribute name string
        """
        if isinstance(attr_mapped, list):
            return "-".join(attr_mapped)
        else:
            return attr_mapped

    def _evaluate_single_attribute(
        self,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        attr: str,
        gt_attr: str,
        gt_value: str,
        metrics: EvaluationMetrics
    ) -> bool:
        """Evaluate a single attribute prediction.
        
        Args:
            pred: Prediction dictionary
            gt: Ground truth dictionary
            attr: Attribute name
            gt_attr: Ground truth attribute name
            gt_value: Ground truth value
            metrics: Metrics accumulator
            
        Returns:
            True if attribute is correct, False otherwise
        """
        doc_id = pred["doc_id"]
        
        # Check if attribute exists in prediction
        if attr not in pred["data"]:
            if not is_null(gt_value):
                metrics.false_negatives += 1
                logging.info(f"[{self.__class__.__name__}:_evaluate_single_attribute] false_negatives (attribute not in prediction) of {attr}: {pred}; {gt}")
                return False
            return True
        
        # Get predicted value
        pred_value = str(pred["data"][attr]).lower()
        
        # Compare values
        return self._compare_values(pred_value, gt_value, attr, gt_attr, pred, gt, metrics)

    def _compare_values(
        self,
        pred_value: str,
        gt_value: str,
        pred_attr: str,
        gt_attr: str,
        pred: Dict[str, Any],
        gt: Dict[str, Any],
        metrics: EvaluationMetrics
    ) -> bool:
        """Compare predicted and ground truth values.
        
        Args:
            pred_value: Predicted value
            gt_value: Ground truth value
            pred_attr: Predicted attribute name
            gt_attr: Ground truth attribute name
            pred: Prediction dictionary (for logging)
            gt: Ground truth dictionary (for logging)
            metrics: Metrics accumulator
            
        Returns:
            True if values match, False otherwise
        """
        # Case 1: Ground truth is null
        if is_null(gt_value):
            if is_null(pred_value):
                metrics.true_positives += 1
                return True
            else:
                metrics.false_positives += 1
                logging.info(f"[{self.__class__.__name__}:_compare_values] false_positives (gt_value null) of {pred_attr}: {pred}; {gt}")
                return False
        
        # Case 2: Ground truth is not null
        if is_null(pred_value):
            metrics.false_negatives += 1
            logging.info(f"[{self.__class__.__name__}:_compare_values] false_negatives (pred_value null) of {pred_attr}: {pred}; {gt}")
            return False
        
        # Case 3: Both values are non-null, compare them
        if pred_value == gt_value or gt_value in pred_value:
            metrics.true_positives += 1
            return True
        
        # Case 4: Values differ, try semantic comparison
        semantically_equal = self._semantic_comparison(pred_attr, pred_value, gt_attr, gt_value)
        if semantically_equal:
            metrics.true_positives += 1
            return True
        else:
            metrics.false_positives += 1
            logging.info(f"[{self.__class__.__name__}:_compare_values] false_positives (semantic mismatch) of {pred_attr}: {pred}; {gt}")
            return False

    def _check_extra_attributes(
        self,
        pred: Dict[str, Any],
        attr_map_dict: Dict[str, Any],
        metrics: EvaluationMetrics
    ) -> None:
        """Check for extra attributes in prediction that are not in ground truth.
        
        Args:
            pred: Prediction dictionary
            attr_map_dict: Attribute mapping dictionary
            metrics: Metrics accumulator
        """
        doc_id = pred["doc_id"]
        
        for attr in pred["data"]:
            # Check if attribute is not in expected attributes and has non-null value
            if attr not in attr_map_dict and not is_null(pred["data"][attr]):
                metrics.false_positives += 1
                logging.info(f"[{self.__class__.__name__}:_check_extra_attributes] false_positives (attribute not in gt) of {attr}: {pred}")

    def _semantic_comparison(
        self,
        pred_attr: str,
        pred_value: str,
        gt_attr: str,
        gt_value: str
    ) -> bool:
        """Compare prediction and ground truth using LLM-based semantic comparison."""
        # Check if committee or single LLM mode
        if self.committee_prompts:
            return self._committee_voting(pred_attr, pred_value, gt_attr, gt_value)
        else:
            return self._single_llm_comparison(pred_attr, pred_value, gt_attr, gt_value)
    
    def _single_llm_comparison(
        self,
        pred_attr: str,
        pred_value: str,
        gt_attr: str,
        gt_value: str
    ) -> bool:
        """Single LLM comparison (backward compatible)."""
        if not self.prompts or "datapop_cmp_str" not in self.prompts:
            return False
            
        # Load comparison cache
        cmp_cache_path = Path(self.config["out_main"]) / PATH_TEMPLATES.eval_comparison_cache()
        cmp_cache = self.load_json(cmp_cache_path) if cmp_cache_path.exists() else {}

        # Check cache
        cache_key = " -- ".join([pred_value, gt_value])
        if cache_key in cmp_cache:
            cached_result = cmp_cache[cache_key]
            if isinstance(cached_result, dict):
                return cached_result.get("result", False)
            else:
                return cached_result

        # Prepare input for LLM comparison
        cmp_input = {
            PREDICTION_KEY: {ATTRIBUTE_NAME_KEY: pred_attr, ATTRIBUTE_VALUE_KEY: pred_value}, 
            GROUND_TRUTH_KEY: {ATTRIBUTE_NAME_KEY: gt_attr, ATTRIBUTE_VALUE_KEY: gt_value}
        }
        cmp_message = json.dumps(cmp_input)
        
        # Call LLM with retry logic
        comparison_result = self._call_llm_comparison(cmp_message)
        
        # Save to cache with metadata
        cmp_cache[cache_key] = {
            "result": comparison_result.get("Result", False),
            "reasoning": comparison_result.get("Reasoning", ""),
            "pred_attr": pred_attr,
            "pred_value": pred_value,
            "gt_attr": gt_attr,
            "gt_value": gt_value,
            "llm_model": self.eval_llm_model
        }
        self.save_results(cmp_cache_path, cmp_cache)
        logging.info(f"[{self.__class__.__name__}:_semantic_comparison] Comparison of `{cmp_input}`: {comparison_result}")        
        
        return bool(comparison_result.get("Result", False))
    
    def _committee_voting(
        self,
        pred_attr: str,
        pred_value: str,
        gt_attr: str,
        gt_value: str
    ) -> bool:
        """Committee voting with multiple LLMs."""
        # Load cache
        cmp_cache_path = Path(self.config["out_main"]) / PATH_TEMPLATES.eval_comparison_cache()
        cmp_cache = self.load_json(cmp_cache_path) if cmp_cache_path.exists() else {}

        # Check cache
        cache_key = " -- ".join([pred_value, gt_value])
        if cache_key in cmp_cache:
            return cmp_cache[cache_key].get("result", False)

        # Prepare input
        cmp_input = {
            PREDICTION_KEY: {ATTRIBUTE_NAME_KEY: pred_attr, ATTRIBUTE_VALUE_KEY: pred_value}, 
            GROUND_TRUTH_KEY: {ATTRIBUTE_NAME_KEY: gt_attr, ATTRIBUTE_VALUE_KEY: gt_value}
        }
        cmp_message = json.dumps(cmp_input)
        
        # Collect votes from all LLMs
        votes, results = [], []
        for member in self.committee_prompts:
            try:
                comparison_result = self._call_llm_comparison(cmp_message, prompt=member["prompts"]["datapop_cmp_str"])
                votes.append(comparison_result.get("Result", False))
                results.append({
                    "llm_model": member["llm_model"],
                    "vote": comparison_result.get("Result", False),
                    "reasoning": comparison_result.get("Reasoning", "")
                })
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}:_committee_voting] {member['mode']}-{member['llm_model']} failed: {e}")
        
        # Majority voting
        true_count = sum(1 for v in votes if v)
        false_count = len(votes) - true_count
        final_result = true_count >= false_count
        
        # Check if there's any disagreement
        has_disagreement = true_count > 0 and false_count > 0
        
        # Save to cache
        cache_entry = {
            "result": final_result,
            "pred_attr": pred_attr,
            "pred_value": pred_value,
            "gt_attr": gt_attr,
            "gt_value": gt_value
        }
        
        # Only save detailed results if there's disagreement
        if has_disagreement:
            cache_entry["votes"] = votes
            cache_entry["results"] = results
        
        cmp_cache[cache_key] = cache_entry
        self.save_results(cmp_cache_path, cmp_cache)
        logging.info(f"[{self.__class__.__name__}:_committee_voting] Result: {final_result} ({true_count} true, {false_count} false){' - unanimous' if not has_disagreement else ''}")
        
        return final_result

    def _call_llm_comparison(self, message: str, prompt: Any = None, max_retries: int = 3) -> Dict[str, Any]:
        """Call LLM for semantic comparison with retry logic.
        
        Args:
            message: JSON message for LLM
            prompt: Prompt instance to use (if None, use self.prompts["datapop_cmp_str"])
            max_retries: Maximum number of retry attempts
            
        Returns:
            Comparison result dictionary with 'Result' and 'Reasoning' keys
        """
        # Use provided prompt or default to self.prompts
        prompt_to_use = prompt if prompt is not None else self.prompts["datapop_cmp_str"]
        
        for attempt in range(max_retries):
            try:
                response = prompt_to_use(msg=message).strip()
                result = json.loads(response)
                if result.get("Result") in [True, False]:
                    return result
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}:_call_llm_comparison] Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logging.error(f"[{self.__class__.__name__}:_call_llm_comparison] All {max_retries} attempts failed, defaulting to False")
                    return {"Result": False, "Reasoning": "LLM comparison failed"}
        
        return {"Result": False, "Reasoning": "LLM comparison failed"}

    def _load_or_generate_mapping(self, query_id: str) -> Dict[str, Any]:
        """Load or generate mapping between prediction and ground truth names.
        
        Loading priority:
        1. Load from self.out_root if exists
        2. Load from self.data_path if exists
        3. Raise NotImplementedError (generation not yet implemented)
        
        Mapping structure:
        - Generated Table Name -> Ground Truth Table Name
        - Generated Attribute Name -> Ground Truth Attribute Name
        - Mapping saved using PATH_TEMPLATES.eval_name_mapping() in the directory
        
        Format:
        {
            "table": {
                "prediction_table_name": "ground_truth_table_name",
                ...
            },
            "attribute": {
                "ground_truth_table_name": {
                    "prediction_attribute_name": "ground_truth_attribute_name",
                    ...
                },
                ...
            }
        }
        
        Args:
            query_id: Query ID
            
        Returns:
            Mapping dictionary with table and attribute mappings
            
        Raises:
            NotImplementedError: If mapping file not found and generation not implemented
        """
        mapping_filename = PATH_TEMPLATES.eval_name_mapping(query_id)
        
        # Priority 1: Load from out_root if exists
        out_mapping_path = self.out_root / mapping_filename
        if out_mapping_path.exists():
            try:
                logging.info(f"[{self.__class__.__name__}:_load_or_generate_mapping] Loading mapping from {out_mapping_path}")
                return self.load_json(out_mapping_path)
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}:_load_or_generate_mapping] Failed to load mapping from {out_mapping_path}: {e}")
        
        # Priority 2: Load from data_path if exists
        data_mapping_path = self.data_path / mapping_filename
        if data_mapping_path.exists():
            try:
                logging.info(f"[{self.__class__.__name__}:_load_or_generate_mapping] Loading mapping from dataset directory: {data_mapping_path}")
                return self.load_json(data_mapping_path)
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}:_load_or_generate_mapping] Failed to load mapping from {data_mapping_path}: {e}")
        
        # Priority 3: Generate mapping (not yet implemented)
        logging.error(f"[{self.__class__.__name__}:_load_or_generate_mapping] Mapping file not found in {out_mapping_path} or {data_mapping_path}")
        raise NotImplementedError("Mapping generation not implemented")
        