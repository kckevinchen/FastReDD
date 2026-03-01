"""
Unified Schema Generation Implementation.

This module provides a single unified implementation of schema generation
that supports multiple LLM providers and adaptive sampling strategies.
"""

import os
import json
import logging
import traceback
import random
from pathlib import Path
from typing import Dict, Any, Optional, List

from tqdm import tqdm

from .base import SchemaGenerator
from core.adaptive_sampling import AdaptiveSamplingMixin
from ..utils import constants
from ..utils.constants import PATH_TEMPLATES
from ..utils.prompt_utils import (
    PromptGPT, PromptDeepSeek, PromptTogether, PromptSiliconFlow, PromptGemini,
    get_api_key
)
from ..utils import output_utils
from ..doc_clustering.doc_clustering import DocumentClustering
from ..data_loader import create_data_loader


class SchemaGen(AdaptiveSamplingMixin, SchemaGenerator):
    """
    Unified Schema Generator supporting multiple LLM providers and adaptive sampling.
    
    Supported modes:
    - "cgpt": OpenAI GPT models
    - "deepseek": DeepSeek models
    - "together": Together AI models
    - "siliconflow": SiliconFlow models
    - "gemini": Google Gemini models
    
    The mode is determined by config["mode"].

    TODO: 
    - [ ] remove doc_dict and use data loader directly. 之后所有模块都移除 doc_dict, 改成 loader 直接load documents. 如果想shuffle 可以shuffle doc_id list 然后通过loader 的 get_doc 函数通过doc_id获得 doc
    """
    
    # Map mode to Prompt class
    PROMPT_CLASS_MAP = {
        "cgpt": PromptGPT,
        "deepseek": PromptDeepSeek,
        "together": PromptTogether,
        "siliconflow": PromptSiliconFlow,
        "gemini": PromptGemini,
    }
    
    # Modes that support adaptive sampling
    ADAPTIVE_SAMPLING_MODES = {"deepseek", "gemini", "cgpt", "together", "siliconflow"}
    
    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        """
        Initialize the schema generator.
        
        Args:
            config: Configuration dictionary
            api_key: Optional API key (can also be provided in config or environment)
        """
        # Initialize base class
        SchemaGenerator.__init__(self, config)
        
        self.mode = config.get("mode", "cgpt")
        
        # Data loader configuration
        self.loader_type = config.get("data_loader_type", "standard")
        self.loader_config = config.get("data_loader_config", {})
        
        if self.mode not in self.PROMPT_CLASS_MAP:
            logging.error(
                f"[{self.__class__.__name__}:__init__] Invalid mode: {self.mode}. "
                f"Supported modes: {list(self.PROMPT_CLASS_MAP.keys())}"
            )
            raise ValueError(f"Invalid mode: {self.mode}")
        
        try:
            # Handle different config structures
            if self.mode == "cgpt":
                # OpenAI GPT uses nested "cgpt" config
                if "cgpt" not in config:
                    logging.error(f"[{self.__class__.__name__}:__init__] "
                                 f"cgpt mode requires 'cgpt' config section")
                    raise ValueError("cgpt mode requires 'cgpt' config section")
                prompt_path = config["cgpt"]["prompt_path"]
                self.log_init_file = config["cgpt"].get("log_init_file")
                self.temperature = config["cgpt"].get("temperature", 1.0)
                self.top_p = config["cgpt"].get("top_p", 1.0)
            else:
                # Other providers use "prompt" config
                prompt_config = config.get("prompt", {})
                if isinstance(prompt_config, dict):
                    prompt_path = prompt_config.get("prompt_path")
                    if not prompt_path:
                        logging.error(f"[{self.__class__.__name__}:__init__] "
                                     f"prompt.prompt_path is required for {self.mode} mode")
                        raise ValueError(f"prompt.prompt_path is required for {self.mode} mode")
                elif isinstance(prompt_config, str):
                    prompt_path = prompt_config
                else:
                    logging.error(f"[{self.__class__.__name__}:__init__] "
                                 f"Invalid prompt config format for {self.mode} mode")
                    raise ValueError(f"Invalid prompt config format for {self.mode} mode")
                self.log_init_file = config.get("log_init_file")
                self.temperature = None
                self.top_p = None
            
            self.doc_cluster_file = config.get("doc_cluster_file")
            self.param_str = config.get("res_param_str")
            self.general_param_str = config.get("general_param_str")
            self.shuffle_documents = config.get("shuffle_documents", True)
            
            # Get API key
            resolved_api_key = get_api_key(config, self.mode, api_key)
            
            # Initialize prompt with appropriate class
            PromptClass = self.PROMPT_CLASS_MAP[self.mode]
            self.prompt = PromptClass(
                self.mode,
                prompt_path,
                llm_model=config.get("llm_model", "gpt-4o"),
                api_key=resolved_api_key
            )
            
            # Initialize adaptive sampling if supported
            if self.mode in self.ADAPTIVE_SAMPLING_MODES:
                self.init_adaptive_sampling(config, api_key=resolved_api_key)
                
                # For backward compatibility
                if hasattr(self, 'adaptive_sampler') and self.adaptive_sampler:
                    self.adaptive_sampling = self.adaptive_sampler
                else:
                    self.adaptive_sampling = None
            else:
                self.adaptive_enabled = False
                self.adaptive_sampler = None
                self.adaptive_sampling = None
                
        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:__init__] Error initializing: {e}")
            traceback.print_exc()
            raise
    
    def __call__(self, dataset_task_list: Optional[List[str]] = None):
        """
        Extract attributes from documents for specified dataset tasks.

        Args:
            dataset_task_list: Optional list of dataset/task paths to process
        """
        if dataset_task_list is None:
            if "exp_dataset_task_list" in self.config:
                dataset_task_list = self.config["exp_dataset_task_list"]
            else:
                logging.warning(
                    f"[{self.__class__.__name__}:__call__] No dataset tasks specified, "
                    f"using default SPIDER_DATASET_TASK_LIST"
                )
                dataset_task_list = constants.SPIDER_DATASET_TASK_LIST
        
        total_tasks = len(dataset_task_list)
        logging.info(
            f"[{self.__class__.__name__}:__call__] Schema Generation for "
            f"{total_tasks} tasks: {dataset_task_list}"
        )

        for dataset_task in dataset_task_list:
            # Separate data path and output path
            data_main = self.config.get("data_main", "dataset/")
            if not data_main:
                data_main = "dataset/"
            data_path = Path(data_main) / dataset_task
            if not data_path.is_absolute():
                data_path = data_path.resolve()
            
            out_root = Path(self.config["out_main"]) / dataset_task
            out_root.mkdir(parents=True, exist_ok=True)
            logging.info(
                f"[{self.__class__.__name__}:__call__] Start processing dataset: "
                f"data_path={data_path}, out_root={out_root}"
            )
            
            if "query" in self.config["in_fields"]:
                self._process_dataset(data_path, out_root)
            else:
                self._process_dataset_no_query(data_path, out_root)
    
    def _process_dataset(self, data_path: Path, out_root: Path):
        """Process all queries and documents in a dataset."""
        self.data_path = Path(data_path)
        self.out_root = Path(out_root)
        
        # Ensure loader_config includes data_main if available
        loader_config = self.loader_config.copy() if self.loader_config else {}
        if "data_main" not in loader_config and "data_main" in self.config:
            loader_config["data_main"] = self.config["data_main"]
        
        # Create data loader based on configuration
        self.loader = create_data_loader(
            data_path=self.data_path,
            loader_type=self.loader_type,
            loader_config=loader_config
        )
        logging.info(
            f"[{self.__class__.__name__}:_process_dataset] Created "
            f"{self.loader.__class__.__name__} for {self.data_path}"
        )
        
        # Load queries and general schema using data loader
        query_dict = self.loader.load_query_dict()
        
        if not query_dict:
            logging.warning(
                f"[{self.__class__.__name__}:_process_dataset] No queries found "
                f"in dataset {self.data_path}"
            )
            return
        
        # Build doc_dict for backward compatibility
        doc_dict = self._build_doc_dict()
        
        # Prepare document order (shuffle or use original order)
        doc_indices = list(range(len(doc_dict)))
        if self.shuffle_documents:
            random.shuffle(doc_indices)
            logging.info(
                f"[{self.__class__.__name__}:_process_dataset] Shuffled document order: "
                f"{doc_indices[:10]}... (showing first 10)"
            )
        else:
            logging.info(
                f"[{self.__class__.__name__}:_process_dataset] Using original document order "
                f"(shuffle_documents=False)"
            )
        
        # Create doc_dict with indices matching processing order
        # If shuffled, doc_indices contains shuffled indices; otherwise sequential [0, 1, 2, ...]
        shuffled_doc_dict = {}
        for new_idx, original_idx in enumerate(doc_indices):
            shuffled_doc_dict[str(new_idx)] = doc_dict[str(original_idx)]
        
        general_schema = self.get_general_schema(self.out_root, doc_dict)
        doc_cluster = self.doc_clustering(self.out_root, doc_dict)
        
        logging.info(
            f"[{self.__class__.__name__}:_process_dataset] Processing "
            f"{len(query_dict)} queries in dataset {self.data_path.name}"
        )
        
        for qid in query_dict:
            query = query_dict[qid]["query"]
            res_path = self.out_root / PATH_TEMPLATES.schema_gen_result_query(qid, self.param_str)
            res_path_current = self.out_root / PATH_TEMPLATES.schema_query_current(qid, self.param_str)
            res_dict = self.load_processed_res(res_path)
            log_init = self.load_log_init(self.out_root, qid)
            general_schema_q = self.get_general_schema(self.out_root, doc_dict, qid, query)
            pgbar_name = f"{self.data_path.name}-{qid}"

            # Use adaptive sampling if enabled
            if self.adaptive_enabled:
                self.process_documents_adaptive(
                    shuffled_doc_dict, query, res_dict, log_init, 
                    general_schema_q, res_path, pgbar_name, doc_indices, res_path_current
                )
            else:
                self.process_documents(
                    shuffled_doc_dict, query, res_dict, log_init, 
                    general_schema_q, res_path, pgbar_name, doc_indices, res_path_current
                )
            self.tailor_schema(self.out_root, doc_dict, qid, query)

    def _process_dataset_no_query(self, data_path: Path, out_root: Path):
        """Process all documents in a dataset without queries."""
        self.data_path = Path(data_path)
        self.out_root = Path(out_root)
        
        # Ensure loader_config includes data_main if available
        loader_config = self.loader_config.copy() if self.loader_config else {}
        if "data_main" not in loader_config and "data_main" in self.config:
            loader_config["data_main"] = self.config["data_main"]
        
        # Create data loader based on configuration
        self.loader = create_data_loader(
            data_path=self.data_path,
            loader_type=self.loader_type,
            loader_config=loader_config
        )
        logging.info(
            f"[{self.__class__.__name__}:_process_dataset_no_query] Created "
            f"{self.loader.__class__.__name__} for {self.data_path}"
        )
        
        # Build doc_dict for backward compatibility
        doc_dict = self._build_doc_dict()
        
        # Prepare document order (shuffle or use original order)
        doc_indices = list(range(len(doc_dict)))
        if self.shuffle_documents:
            random.shuffle(doc_indices)
            logging.info(
                f"[{self.__class__.__name__}:_process_dataset_no_query] Shuffled document order: "
                f"{doc_indices[:10]}... (showing first 10)"
            )
        else:
            logging.info(
                f"[{self.__class__.__name__}:_process_dataset_no_query] Using original document order "
                f"(shuffle_documents=False)"
            )
        
        # Create doc_dict with indices matching processing order
        # If shuffled, doc_indices contains shuffled indices; otherwise sequential [0, 1, 2, ...]
        shuffled_doc_dict = {}
        for new_idx, original_idx in enumerate(doc_indices):
            shuffled_doc_dict[str(new_idx)] = doc_dict[str(original_idx)]
        
        res_path = self.out_root / PATH_TEMPLATES.schema_gen_result_general(self.param_str)
        res_path_current = self.out_root / PATH_TEMPLATES.schema_general_current(self.param_str)
        res_dict = self.load_processed_res(res_path)
        pgbar_name = f"{self.data_path.name}"

        # Use adaptive sampling if enabled
        if self.adaptive_enabled:
            self.process_documents_adaptive(
                shuffled_doc_dict, "", res_dict, dict(), None, res_path, pgbar_name, doc_indices, res_path_current
            )
        else:
            self.process_documents(
                shuffled_doc_dict, "", res_dict, dict(), None, res_path, pgbar_name, doc_indices, res_path_current
            )
    
    def _build_doc_dict(self) -> Dict[str, List]:
        """
        Build doc_dict from data loader for backward compatibility.
        
        Format: {doc_id: [doc_text, source_info], ...}
        Keys are normalized to sequential string integers "0", "1", "2", ...
        """
        doc_dict = {}
        for idx, (doc_text, doc_id, metadata) in enumerate(self.loader.iter_docs()):
            # Get source info (could be filename, table name, etc.)
            source_info = metadata.get("source_file") or metadata.get("table_name") or ""
            # Normalize keys to sequential integers as strings
            doc_dict[str(idx)] = [doc_text, source_info]
        
        logging.info(
            f"[{self.__class__.__name__}:_build_doc_dict] Built doc_dict with "
            f"{len(doc_dict)} documents"
        )
        return doc_dict

    def process_documents(
        self, 
        doc_dict: Dict[str, List], 
        query: str, 
        res_dict: Dict[str, Any], 
        log_init: Dict[str, Any], 
        general_schema: Any, 
        res_path: Path, 
        pgbar_name: str, 
        original_indices: Optional[List[int]] = None,
        current_schema_path: Optional[Path] = None
    ):
        """
        Process individual documents in the dataset.
        
        Args:
            doc_dict: Dictionary of documents (may be shuffled)
            query: Query string (empty for no-query mode)
            res_dict: Dictionary to store results
            log_init: Initial log state
            general_schema: General schema
            res_path: Path to save results
            pgbar_name: Name for progress bar
            original_indices: Optional list mapping shuffled indices to original indices
        """
        num_doc = len(doc_dict)
        i, cnt, progress_bar = 0, 0, tqdm(total=num_doc, desc=f"Processing {pgbar_name}")

        logging.info(f"[{self.__class__.__name__}:process_documents] Start processing query: {query}")
        logging.info(f"[{self.__class__.__name__}:process_documents] Start processing documents: {res_path}")
        logging.info(f"[{self.__class__.__name__}:process_documents] Processed documents: {len(res_dict)} / {num_doc}")
        if original_indices:
            logging.info(f"[{self.__class__.__name__}:process_documents] Documents are shuffled - processing in random order")
        
        # Reset adaptive sampling at the start of processing
        if hasattr(self, 'adaptive_sampling') and self.adaptive_sampling is not None and self.adaptive_sampling.enabled:
            self.adaptive_sampling.reset()
            logging.info(f"[{self.__class__.__name__}:process_documents] Adaptive sampling enabled - will stop early if schema stabilizes")
            
            # First, process all existing results to build up adaptive sampling state
            check_indices = original_indices if original_indices else list(range(num_doc))
            for existing_shuffled_idx, existing_original_idx in enumerate(check_indices):
                if str(existing_original_idx) in res_dict:
                    existing_log = res_dict[str(existing_original_idx)].get("log", [])
                    if existing_log:
                        self.adaptive_sampling.update(existing_log)
                        should_stop, reason = self.adaptive_sampling.should_stop()
                        if should_stop:
                            logging.info(
                                f"[{self.__class__.__name__}:process_documents] "
                                f"Adaptive sampling: Early stop triggered based on existing results ({reason})"
                            )
                            logging.info(
                                f"[{self.__class__.__name__}:process_documents] "
                                f"All documents already processed or schema stabilized - no new processing needed"
                            )
                            progress_bar.update(num_doc - existing_shuffled_idx - 1)
                            progress_bar.close()
                            return
        
        while i < num_doc:
            # Map shuffled index to original index for result storage
            original_idx = original_indices[i] if original_indices else i
            
            if str(original_idx) in res_dict:
                i, cnt = i + 1, 0
                progress_bar.update(1)
                continue

            # Get previous log state
            if i == 0:
                log = log_init
            else:
                prev_original_idx = original_indices[i-1] if original_indices else (i-1)
                log = res_dict[str(prev_original_idx)].get("log", log_init) if str(prev_original_idx) in res_dict else log_init
            
            # Process document at shuffled index i
            input_json = self.prepare_input_json(doc_dict, i, query, log, general_schema)
            out_dict = self.process_single_document(input_json, cnt, i)
            result_data = self.extract_result_data(out_dict)

            if not result_data or len(result_data["log"]) < len(log):
                cnt += 1
                if cnt > 10:
                    if not result_data:
                        logging.error(
                            f"[{self.__class__.__name__}:process_documents] "
                            f"Failed to process document {i} (original: {original_idx}) after {cnt} retries!"
                        )
                        raise RuntimeError(f"Failed to process document {i} (original: {original_idx}) after {cnt} retries")
                    else:
                        logging.error(
                            f"[{self.__class__.__name__}:process_documents] "
                            f"Schema num decrease, retry_count {cnt}, doc_index {i} (original: {original_idx})"
                        )
                        raise RuntimeError(f"Schema num decrease, retry_count {cnt}, doc_index {i} (original: {original_idx})")
                continue

            # Store result with original index
            res_dict[str(original_idx)] = result_data
            self.save_results(res_path, res_dict)
            
            # Update current schema result
            if current_schema_path:
                self._update_current_schema(res_dict, current_schema_path)
            
            # Update adaptive sampling and check if should stop early
            if hasattr(self, 'adaptive_sampling') and self.adaptive_sampling is not None and self.adaptive_sampling.enabled:
                self.adaptive_sampling.update(result_data["log"])
                should_stop, reason = self.adaptive_sampling.should_stop()
                if should_stop:
                    logging.info(f"[{self.__class__.__name__}] Adaptive sampling: Early stop triggered ({reason})")
                    logging.info(
                        f"[{self.__class__.__name__}] Stopped at document {i+1} / {num_doc} "
                        f"(original index: {original_idx}, processed {self.adaptive_sampling.num_docs_processed} documents)"
                    )
                    break

            i, cnt = i + 1, 0
            progress_bar.update(1)
            logging.info(f"[{self.__class__.__name__}:process_documents] Processed document {i} / {num_doc} (original index: {original_idx})")

        progress_bar.close()
        if hasattr(self, 'adaptive_sampling') and self.adaptive_sampling is not None and self.adaptive_sampling.enabled and self.adaptive_sampling.num_docs_processed > 0:
            stats = self.adaptive_sampling.get_stats()
            logging.info(
                f"[{self.__class__.__name__}:process_documents] Adaptive sampling stats: "
                f"{stats.get('num_docs_processed', 0)} docs processed, "
                f"entropy={stats.get('avg_entropy', 0):.4f}, streak={stats.get('stability_streak', 0)}"
            )
        logging.info(f"[{self.__class__.__name__}:process_documents] Finished processing documents in {res_path}")
    
    def process_single_document(self, input_json: Dict[str, Any], retry_count: int, doc_index: int) -> Optional[Dict[str, Any]]:
        """Process a single document and handle errors."""
        attr_msg = "New Input:\n" + json.dumps(input_json)
        result_str = self.apply_prompt(attr_msg)
        try:
            return json.loads(result_str)
        except json.JSONDecodeError:
            logging.warning(
                f"[{self.__class__.__name__}:process_single_document] JSON LOAD ERROR, "
                f"retry_count {retry_count}, doc_index {doc_index}, {repr(result_str)}"
            )
            return None
    
    def apply_prompt(self, attr_msg: str) -> str:
        """Apply prompt and return response."""
        return self.prompt(msg=attr_msg).strip()

    def prepare_input_json(
        self, 
        doc_dict: Dict[str, List], 
        doc_index: int, 
        query: str, 
        log: Dict[str, Any], 
        general_schema: Any
    ) -> Dict[str, Any]:
        """Prepare the input JSON for a single document."""
        input_info = {
            "doc": doc_dict[str(doc_index)][0], 
            "query": query,
            "log": log,
            "general_schema": general_schema,
            "doc_cluster": None,
            "all_clusters": None
        }
        input_json = {}
        for info_key, json_field in self.config["in_fields"].items():
            if info_key not in input_info:
                error_msg = f"Error: input info key <{info_key}> not supported!"
                logging.error(f"[{self.__class__.__name__}:prepare_input_json] {error_msg}")
                raise ValueError(error_msg)
            input_json[json_field] = input_info[info_key]
        return input_json
        
    def extract_result_data(self, out_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract result data from output dictionary."""
        if not out_dict:
            return None
        result_data = dict()
        for res_key, json_field in self.config["out_fields"].items():
            if json_field not in out_dict:
                logging.warning(
                    f"[{self.__class__.__name__}:extract_result_data] "
                    f"output field <{json_field}> not found in out_dict!"
                )
                return None
            result_data[res_key] = out_dict[json_field]
        return result_data
    
    def get_general_schema(
        self, 
        out_root: Path, 
        doc_dict: Dict[str, List], 
        qid: Optional[str] = None, 
        query: Optional[str] = None
    ) -> Any:
        """Get the general schema (optionally query-specific)."""
        out_root = Path(out_root)
        if "general_schema" not in self.config["in_fields"]:
            return None
        schema_path = out_root / PATH_TEMPLATES.schema_general(self.general_param_str, qid)
        if not schema_path.exists():
            logging.info(f"[{self.__class__.__name__}:get_general_schema] General Schema Not Found in {schema_path}")
            try:
                res_dict = self.load_json(out_root / PATH_TEMPLATES.schema_gen_result_general(self.general_param_str))
            except FileNotFoundError:
                logging.error(
                    f"[{self.__class__.__name__}:get_general_schema] "
                    f"{PATH_TEMPLATES.schema_gen_result_general(self.general_param_str)} Not Found!"
                )
                exit(2)
            logging.info(f"[{self.__class__.__name__}:get_general_schema] Start Extracting General Schema ...")
            output_utils.create_general_schema(self.config, res_dict, doc_dict, str(out_root), self.general_param_str, qid, query)
            logging.info(f"[{self.__class__.__name__}:get_general_schema] General Schema Extracted in {schema_path}")
        logging.info(f"[{self.__class__.__name__}:get_general_schema] General Schema Loaded from {schema_path}")
        return self.load_json(schema_path)
    
    def _update_current_schema(self, res_dict: Dict[str, Any], current_schema_path: Path):
        """
        Update the current schema result based on the latest document processed.
        
        This saves the schema from the last processed document's log.
        
        Args:
            res_dict: Dictionary of results from schema generation
            current_schema_path: Path to save the current schema
        """
        if not res_dict:
            return
        
        # Get the schema from the last processed document
        sorted_keys = sorted(res_dict.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        if not sorted_keys:
            return
        
        last_key = sorted_keys[-1]
        last_result = res_dict[last_key]
        
        if "log" not in last_result:
            logging.warning(
                f"[{self.__class__.__name__}:_update_current_schema] "
                f"log not found in result for doc_id={last_key}"
            )
            return
        
        current_schema = last_result["log"]
        self.save_results(str(current_schema_path), current_schema)
        logging.debug(
            f"[{self.__class__.__name__}:_update_current_schema] "
            f"Updated current schema from doc {last_key} to {current_schema_path.name}"
        )
    
    def tailor_schema(self, out_root: Path, doc_dict: Dict[str, List], qid: str, query: str):
        """Generate the tailored query-specific schema."""
        out_root = Path(out_root)
        schema_path = out_root / PATH_TEMPLATES.schema_query_tailored(qid, self.param_str)
        if not schema_path.exists():
            try:
                res_dict = self.load_json(out_root / PATH_TEMPLATES.schema_gen_result_query(qid, self.param_str))
            except FileNotFoundError:
                logging.error(
                    f"[{self.__class__.__name__}:tailor_schema] "
                    f"{PATH_TEMPLATES.schema_gen_result_query(qid, self.param_str)} Not Found!"
                )
                exit(2)
            logging.info(f"[{self.__class__.__name__}:tailor_schema] Start Tailoring Schema ...")
            output_utils.create_tailored_schema(self.config, res_dict, doc_dict, str(out_root), self.param_str, qid, query)
            logging.info(f"[{self.__class__.__name__}:tailor_schema] Tailored Schema Created in {schema_path}")
        else:
            logging.info(f"[{self.__class__.__name__}:tailor_schema] Tailored Schema Already Exists in {schema_path}")
    
    def doc_clustering(self, out_root: Path, doc_dict: Dict[str, List]) -> Any:
        """Conduct document clustering for the dataset."""
        out_root = Path(out_root)
        if self.doc_cluster_file:
            doc_cluster_path = out_root / self.doc_cluster_file
            if not doc_cluster_path.exists():
                logging.warning(f"[{self.__class__.__name__}:doc_clustering] Document Cluster Not Found in {self.doc_cluster_file}")
                logging.info(f"[{self.__class__.__name__}:doc_clustering] Start Document Clustering ...")
                documents = []
                for doc_id in doc_dict:
                    documents.append(doc_dict[doc_id][0])
                doc_clustering = DocumentClustering(documents, 2, None, None)
                cluster_ids = doc_clustering.cluster()
                # TODO: save doc cluster to `out_root/<doc_cluster_file>`
                logging.info(f"[{self.__class__.__name__}:doc_clustering] Document Cluster Generated in {self.doc_cluster_file}")
        return self.load_doc_cluster(out_root)

    def load_log_init(self, out_root: Path, qid: str) -> Dict[str, Any]:
        """Load Log Init from file."""
        out_root = Path(out_root)
        log_init = dict()
        if self.log_init_file is not None:
            try:
                log_init = self.load_json(out_root / self.log_init_file.format(qid))
                logging.info(
                    f"[{self.__class__.__name__}:load_log_init] Query-specific Prompt Log Init "
                    f"({self.log_init_file}, len={len(log_init)}) Imposed for Log Init!"
                )
            except FileNotFoundError:
                logging.warning(
                    f"[{self.__class__.__name__}:load_log_init] Query-specific Prompt Log Init "
                    f"({self.log_init_file}) Not Found!"
                )
        else:
            logging.info(f"[{self.__class__.__name__}:load_log_init] No Query-specific Prompt Log Init!")
        return log_init
    
    def load_doc_cluster(self, out_root: Path) -> Dict[str, Any]:
        """Load the document cluster results from file."""
        out_root = Path(out_root)
        doc_cluster = dict()
        if self.doc_cluster_file is not None:
            try:
                doc_cluster = self.load_json(out_root / self.doc_cluster_file)
                logging.info(
                    f"[{self.__class__.__name__}:load_doc_cluster] Document Cluster "
                    f"({self.doc_cluster_file}, len={len(doc_cluster)}) Loaded!"
                )
            except FileNotFoundError:
                logging.warning(
                    f"[{self.__class__.__name__}:load_doc_cluster] Document Cluster "
                    f"({self.doc_cluster_file}) Not Found!"
                )
        else:
            logging.info(f"[{self.__class__.__name__}:load_doc_cluster] No Document Cluster!")
        return doc_cluster

    def __str__(self) -> str:
        """String representation of the schema generator."""
        return f"{self.__class__.__name__} (mode={self.mode}): \n{self.param_str}\n{self.prompt}"
