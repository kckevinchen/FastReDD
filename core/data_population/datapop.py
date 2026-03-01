"""
Unified Data Population Implementation.

This module provides a single unified implementation of data population
that supports multiple LLM providers.

Output format: `<out_root>/res_tabular_data_{qid}_{param_str}.json`
Structure:
{
    <doc_id>: {
        "res": <table_name>,
        "data": {<attribute_name>: <value>, ...},
        "reason": "...",  # Optional reasoning
    },
    ...
}
"""

from __future__ import annotations

import os
import json
import logging
import traceback
import re
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .base import DataPopulator
from ..utils import constants
from ..utils.constants import (
    NULL_VALUE,
    SCHEMA_NAME_KEY, ATTRIBUTES_KEY, ATTRIBUTE_NAME_KEY,
    DOCUMENT_KEY, SCHEMA_KEY, TARGET_ATTRIBUTE_KEY,
    TABLE_ASSIGNMENT_KEY,
    RESULT_TABLE_KEY, RESULT_DATA_KEY,
    DEFAULT_MAX_TABLE_RETRIES, DEFAULT_MAX_ATTR_RETRIES,
    MAX_ATTRIBUTE_VALUE_LENGTH,
    PATH_TEMPLATES,
)
from ..data_loader import create_data_loader
from ..utils.utils import is_none_value
from ..utils.conformal_filter import ConformalChunkFilter
from ..utils.prompt_utils import (
    PromptGPT, PromptDeepSeek, PromptTogether, PromptSiliconFlow, PromptGemini,
    get_api_key
)

__all__ = ["DataPop"]


class DataPop(DataPopulator):
    """
    Unified Data Population class supporting multiple LLM providers.
    
    Supported modes:
    - "cgpt": OpenAI GPT models
    - "deepseek": DeepSeek models
    - "together": Together AI models
    - "siliconflow": SiliconFlow models
    - "gemini": Google Gemini models
    
    The mode is determined by config["mode"].
    """
    
    # Map mode to Prompt class
    PROMPT_CLASS_MAP = {
        "cgpt": PromptGPT,
        "deepseek": PromptDeepSeek,
        "together": PromptTogether,
        "siliconflow": PromptSiliconFlow,
        "gemini": PromptGemini,
    }
    
    def __init__(self, config: Dict[str, Any], api_key: Optional[str] = None):
        """
        Initialize the data populator.
        
        Args:
            config: Configuration dictionary
            api_key: Optional API key (can also be provided in config or environment)
        """
        # Initialize base class
        super().__init__(config)
        
        self.mode = config.get("mode", "deepseek")
        
        # Validate mode
        if self.mode not in self.PROMPT_CLASS_MAP:
            logging.error(
                f"[{self.__class__.__name__}:__init__] Invalid mode: {self.mode}. "
                f"Supported modes: {list(self.PROMPT_CLASS_MAP.keys())}"
            )
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Validate required keys
        required_keys = ["res_param_str", "llm_model", "prompts"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"Missing required configuration keys: {missing_keys}")
            raise KeyError(f"Missing required configuration keys: {missing_keys}")
        
        try:
            self.param_str = config["res_param_str"]
            
            # Data loader configuration
            self.loader_type = config.get("data_loader_type", "sqlite")
            self.loader_config = config.get("data_loader_config", {})
            
            # API rate limit retry configuration
            self.retry_params = {}
            if "max_retries" in config:
                self.retry_params["max_retries"] = config["max_retries"]
            if "wait_time" in config:
                self.retry_params["wait_time"] = config["wait_time"]
            if self.retry_params:
                logging.info(f"[{self.__class__.__name__}:__init__] Rate limit retry enabled: {self.retry_params}")
            
            # Get API key and store it in config for chunk filter
            resolved_api_key = get_api_key(config, self.mode, api_key)
            self.api_key = resolved_api_key
            config["api_key"] = resolved_api_key
            
            # Get prompt paths from config
            prompts_config = config.get("prompts", {})
            if not prompts_config:
                logging.error(f"[{self.__class__.__name__}:__init__] "
                             f"prompts config is required (must contain 'prompt_table' and 'prompt_attr')")
                raise ValueError("prompts config is required (must contain 'prompt_table' and 'prompt_attr')")
            
            prompt_table_path = prompts_config.get("prompt_table")
            prompt_attr_path = prompts_config.get("prompt_attr")
            
            if not prompt_table_path or not prompt_attr_path:
                logging.error(f"[{self.__class__.__name__}:__init__] "
                             f"prompts.prompt_table and prompts.prompt_attr are required")
                raise ValueError("prompts.prompt_table and prompts.prompt_attr are required")
            
            # Initialize prompts with appropriate class
            PromptClass = self.PROMPT_CLASS_MAP[self.mode]
            self.prompt_table = PromptClass(
                self.mode,
                prompt_table_path,
                llm_model=config.get("llm_model", "deepseek-chat"),
                api_key=resolved_api_key
            )
            self.prompt_attr = PromptClass(
                self.mode,
                prompt_attr_path,
                llm_model=config.get("llm_model", "deepseek-chat"),
                api_key=resolved_api_key
            )
            
            # Initialize chunk filter (Experiment 2: Query-Aware Chunk Filtering)
            chunk_filter_config = config.get("chunk_filter", {})
            if chunk_filter_config.get("enabled", False):
                self.chunk_filter = ConformalChunkFilter(config, api_key=resolved_api_key)
            else:
                self.chunk_filter = None
            
            logging.info(f"[{self.__class__.__name__}:__init__] Initialized with mode: {self.mode}, model: {config.get('llm_model', 'N/A')}")
                
        except Exception as e:
            logging.error(f"[{self.__class__.__name__}:__init__] Error initializing: {e}")
            traceback.print_exc()
            raise

    def __call__(self, dataset_task_list: Optional[List[str]] = None) -> None:
        """
        Extract tabular data from documents in the specified dataset tasks.
        
        Args:
            dataset_task_list: Dataset tasks to process. If None, uses config or default.
                Examples: ["store_1/customers-invoices", "wine_1/wine-appellations"]
        """
        if dataset_task_list is None:
            if "exp_dataset_task_list" in self.config:
                dataset_task_list = self.config["exp_dataset_task_list"]
            else:
                logging.warning(f"[{self.__class__.__name__}:__call__] No dataset tasks specified, using default SPIDER_DATASET_TASK_LIST")
                dataset_task_list = constants.SPIDER_DATASET_TASK_LIST
        
        total_tasks = len(dataset_task_list)
        logging.info(f"[{self.__class__.__name__}:__call__] Processing {total_tasks} tasks: {dataset_task_list}")
        
        for dataset_task in dataset_task_list:
            # Separate data path and output path
            data_path = Path(self.config.get("data_main", self.config["out_main"])) / dataset_task
            out_root = Path(self.config["out_main"]) / dataset_task
            out_root.mkdir(parents=True, exist_ok=True)
            logging.info(f"[{self.__class__.__name__}:__call__] Start processing dataset: data_path={data_path}, out_root={out_root}")
            self._process_dataset(data_path, out_root)

    def _process_dataset(self, data_path: str | Path, out_root: str | Path):
        """Process all queries and documents in a dataset."""
        self.data_path = Path(data_path)
        self.out_root = Path(out_root)
        
        # Create data loader based on configuration
        self.loader = create_data_loader(
            data_path=self.data_path,
            loader_type=self.loader_type,
            loader_config=self.loader_config
        )
        logging.info(f"[{self.__class__.__name__}:_process_dataset] Created {self.loader.__class__.__name__} for {self.data_path}")
        
        query_dict = self.loader.load_query_dict()
        self.schema_general = self.loader.load_schema_general()

        if not query_dict:
            logging.warning(f"[{self.__class__.__name__}:_process_dataset] No queries found in dataset {self.data_path}")
            return
            
        logging.info(f"[{self.__class__.__name__}:_process_dataset] Processing {len(query_dict)} queries in dataset {self.data_path.name}")

        for qid in query_dict:
            query = query_dict[qid]
            query_text = query.get("query", "") if isinstance(query, dict) else str(query)
            schema_query = self.loader.load_schema_query(qid)
            res_path = self.out_root / PATH_TEMPLATES.data_population_result(qid, self.param_str)
            res_data = self.load_processed_res(res_path)
            pgbar_name = f"{self.data_path.name}-{qid}"

            logging.info(f"[{self.__class__.__name__}:_process_dataset] Start processing query-{qid} -> {res_path}")
            self._process_documents(
                schema_query=schema_query,
                query=query_text,
                res_data=res_data,
                res_path=res_path,
                pgbar_name=pgbar_name,
            )

    def _extract_schema_features(self, schema_query: List[Dict[str, Any]]) -> List[str]:
        """
        Extract schema feature names from schema_query for query augmentation.
        
        Args:
            schema_query: Schema structure (list of dicts with "Schema Name" and "Attributes")
            
        Returns:
            List of attribute names (schema features)
        """
        if not schema_query:
            return []
        
        features = []
        for schema_entry in schema_query:
            if not isinstance(schema_entry, dict):
                continue
            attributes = schema_entry.get(ATTRIBUTES_KEY, [])
            for attr in attributes:
                if isinstance(attr, dict):
                    attr_name = attr.get(ATTRIBUTE_NAME_KEY) or attr.get("attribute_name")
                    if attr_name:
                        features.append(attr_name)
                elif isinstance(attr, str):
                    features.append(attr)
        return features
    
    def _process_documents(
        self,
        schema_query: List[Dict[str, Any]],
        query: str,
        res_data: Dict[str, Any],
        res_path: Path,
        pgbar_name: str,
        max_table_retries: int = DEFAULT_MAX_TABLE_RETRIES,
        max_attr_retries: int = DEFAULT_MAX_ATTR_RETRIES,
    ) -> None:
        """Iterate over documents and populate table/attribute data."""
        all_tables = [s[SCHEMA_NAME_KEY] for s in self.schema_general]
        table2schema = {s[SCHEMA_NAME_KEY]: s for s in self.schema_general}
        table2attr: Dict[str, List[str]] = {
            s[SCHEMA_NAME_KEY]: [a[ATTRIBUTE_NAME_KEY] for a in s[ATTRIBUTES_KEY]]
            for s in schema_query
            if s[SCHEMA_NAME_KEY] in all_tables
        }

        total_docs = self.loader.num_docs
        already_processed = len(res_data)
        progress_bar = tqdm(
            total=total_docs, initial=0,
            desc=f"Processing {pgbar_name} ({already_processed}/{total_docs})"
        )
        
        # Extract schema features for query augmentation if chunk filtering is enabled
        schema_features = None
        if self.chunk_filter and self.chunk_filter.config.enabled:
            schema_features = self._extract_schema_features(schema_query)
        
        for did, (doc_text, *_rest) in zip(self.loader.doc_ids, self.loader.iter_docs()):
            if did in res_data:
                progress_bar.update(1)
                continue

            # Apply chunk filtering if enabled (Experiment 2)
            filtered_doc_text = doc_text
            if self.chunk_filter and self.chunk_filter.config.enabled:
                filtered_chunks, filter_stats = self.chunk_filter.filter_chunks(
                    query=query,
                    document=doc_text,
                    schema_features=schema_features
                )
                # Combine filtered chunks back into a single document
                filtered_doc_text = "\n\n".join(filtered_chunks)
                
                # Log filtering statistics (only for first few documents to avoid spam)
                if filter_stats.get("filtered") and already_processed == 0:
                    logging.info(f"[{self.__class__.__name__}:_process_documents] "
                               f"Chunk filtering: {filter_stats.get('num_chunks', 0)} chunks -> "
                               f"{filter_stats.get('num_kept', 0)} kept, "
                               f"avg_similarity={filter_stats.get('avg_similarity', 0):.4f}")

            # -- table assignment -----------------------------------------------------
            table_attempt = 0
            table_failed = False
            while True:
                if table_attempt > max_table_retries:
                    logging.info(f"[{self.__class__.__name__}:_process_documents] Table fail "
                                 f">{max_table_retries}x for doc {did}. Skipping.")
                    table_failed = True
                    break
                tbl_input = {DOCUMENT_KEY: filtered_doc_text, SCHEMA_KEY: self.schema_general}
                tbl_input = json.dumps(tbl_input, ensure_ascii=False)
                raw_text = self.prompt_table(msg=tbl_input, response_format="text", **self.retry_params).strip()
                res_tbl = self._extract_json_block(raw_text)
                if not res_tbl or TABLE_ASSIGNMENT_KEY not in res_tbl:
                    table_attempt += 1
                    continue
                table_assigned = res_tbl[TABLE_ASSIGNMENT_KEY]
                if table_assigned not in all_tables and not is_none_value(table_assigned):
                    table_attempt += 1
                    continue
                break
            
            if table_failed:
                progress_bar.update(1)
                continue

            table_assigned = NULL_VALUE if is_none_value(table_assigned) else table_assigned
            result_entry = {RESULT_TABLE_KEY: table_assigned, RESULT_DATA_KEY: {}}

            # -- attribute extraction -------------------------------------------------
            for attr in table2attr.get(table_assigned, []):
                attr_attempt = 0
                attr_val = None
                while True:
                    if attr_attempt > max_attr_retries:
                        logging.info(f"[{self.__class__.__name__}:_process_documents] Attr fail "
                                     f">{max_attr_retries}x for doc {did} attr {attr}. Skipping attr.")
                        break
                    attr_input = {
                        DOCUMENT_KEY: filtered_doc_text,
                        SCHEMA_KEY: table2schema[table_assigned],
                        TARGET_ATTRIBUTE_KEY: attr,
                    }
                    attr_input = json.dumps(attr_input, ensure_ascii=False)
                    raw_text = self.prompt_attr(msg=attr_input, response_format="text", **self.retry_params).strip()
                    res_attr = self._extract_json_block(raw_text)
                    if not res_attr or attr not in res_attr:
                        attr_attempt += 1
                        continue
                    attr_val = res_attr[attr]
                    break

                attr_val = NULL_VALUE if is_none_value(attr_val) else attr_val
                if isinstance(attr_val, str) and len(attr_val) > MAX_ATTRIBUTE_VALUE_LENGTH:
                    logging.info(f"[{self.__class__.__name__}:_process_documents] Attr too long "
                                    f"(>{len(attr_val)}): doc {did} attr {attr}")
                result_entry[RESULT_DATA_KEY][attr] = attr_val

            res_data[did] = result_entry
            self.save_results(res_path, res_data)
            progress_bar.update(1)

        progress_bar.close()
        logging.info(f"[{self.__class__.__name__}:_process_documents] Done docs -> {res_path}")

    def get_res_schema(self, res_schema_path: str) -> Dict[str, Any]:
        """Load Result Schema from <res_schema_path>."""
        if not os.path.exists(res_schema_path):
            logging.error(f"[{self.__class__.__name__}:get_res_schema] Result Schema not found: {res_schema_path}")
            exit()
        return self.load_json(res_schema_path)
    
    @staticmethod
    def _extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from raw text.
        
        Args:
            raw_text: Raw text that may contain JSON content
            
        Returns:
            Parsed JSON object as dictionary, or None if no valid JSON found
        """
        if not raw_text or not raw_text.strip():
            return None
            
        def _try_parse_json_str(text: str) -> Optional[Dict[str, Any]]:
            """Try to parse JSON using multiple parsers."""
            for parser in (json.loads, ast.literal_eval):
                try:
                    obj = parser(text)
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue
            return None
        
        # Strategy 1: Look for JSON code blocks
        json_patterns = [
            r'```json\s*(.*?)\s*```',  # ```json ... ```
            r'```\s*(\{.*?\})\s*```',  # ``` {...} ```
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                json_candidate = match.group(1).strip()
                result = _try_parse_json_str(json_candidate)
                if result:
                    return result
        
        # Strategy 2: Look for JSON-like objects in the text
        # Use non-greedy matching to find the first complete JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        for match in re.finditer(json_pattern, raw_text, re.DOTALL):
            json_candidate = match.group(0)
            result = _try_parse_json_str(json_candidate)
            if result:
                return result
        
        # Strategy 3: Try parsing the entire text as JSON
        result = _try_parse_json_str(raw_text.strip())
        if result:
            return result
            
        return None

    def __str__(self) -> str:
        """String representation of the data populator."""
        return f"{self.__class__.__name__} (mode={self.mode}): \n{self.param_str}"
