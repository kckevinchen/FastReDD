"""
GPU / Local LLM model for data population.

This module provides data population using locally-hosted LLM models
with GPU inference and hidden states extraction.
"""

from __future__ import annotations

import os
import re
import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

import torch
if torch.cuda.is_available():
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from .base import DataPopulator
from ..data_loader import create_data_loader, DataLoaderBase
from ..utils import constants
from ..utils.constants import (
    SCHEMA_NAME_KEY, ATTRIBUTES_KEY, ATTRIBUTE_NAME_KEY,
    DOCUMENT_KEY, SCHEMA_KEY, TARGET_ATTRIBUTE_KEY,
    TABLE_ASSIGNMENT_KEY,
    PATH_TEMPLATES,
)
from ..utils.conformal_filter import ConformalChunkFilter

__all__ = ["DataPopLocal"]


class DataPopLocal(DataPopulator):
    """
    Data population using locally-hosted LLM models with GPU inference.
    
    This class handles:
    - Local model loading and inference
    - Hidden states extraction for downstream tasks
    - Table assignment and attribute extraction
    
    Note: Requires CUDA-enabled GPU and transformers library.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the local data populator.
        
        Args:
            config: Configuration dictionary containing:
                - mode: Must be "local"
                - llm_model_path: Path to local model weights
                - llm_model: Model name (e.g., "deepseek-ai/deepseek-llm-7b-chat")
                - res_param_str: Parameter string for output naming
                - prompts: Dict with "prompt_table" and "prompt_attr" paths
        """
        super().__init__(config)

        if not torch.cuda.is_available():
            logging.error(f"[{self.__class__.__name__}:__init__] CUDA unavailable. Exiting...")
            raise RuntimeError("CUDA is required for DataPopLocal but is not available")

        self.mode = config.get("mode", "local")
        if self.mode != "local":
            logging.warning(f"[{self.__class__.__name__}:__init__] Mode is '{self.mode}', expected 'local'")
        
        # Validate required keys
        required_keys = ["llm_model_path", "llm_model", "res_param_str", "prompts"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"Missing required configuration keys: {missing_keys}")
            raise KeyError(f"Missing required configuration keys: {missing_keys}")
        
        self.llm_model_path = config["llm_model_path"]
        self.llm_model_name = config["llm_model"]
        self.param_str = config["res_param_str"]
        
        # Data loader configuration
        self.loader_type = config.get("data_loader_type", "sqlite")
        self.loader_config = config.get("data_loader_config", {})

        # Load model
        self.tokenizer = self.model = None
        self._load_model()

        # Load prompts
        self.prompt_table = self._load_prompt(config["prompts"]["prompt_table"])
        self.prompt_attr = self._load_prompt(config["prompts"]["prompt_attr"])
        
        # Initialize chunk filter (optional)
        chunk_filter_config = config.get("chunk_filter", {})
        if chunk_filter_config.get("enabled", False):
            self.chunk_filter = ConformalChunkFilter(config)
        else:
            self.chunk_filter = None
        
        logging.info(f"[{self.__class__.__name__}:__init__] Initialized with model: {self.llm_model_name}")

    def _load_model(self):
        """Load the LLM weights from disk or download if missing."""
        if not os.path.exists(self.llm_model_path):
            logging.info(f"[{self.__class__.__name__}:_load_model] Downloading model ...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_name, trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True, 
            ).cuda()
            # Cache for further runs
            self.tokenizer.save_pretrained(self.llm_model_path)
            self.model.save_pretrained(self.llm_model_path)
        else:
            logging.info(f"[{self.__class__.__name__}:_load_model] Loading model from local ...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.llm_model_path, trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True, 
            ).cuda()
        
        # Configure generation for specific models
        if self.llm_model_name in ["deepseek-ai/deepseek-llm-7b-chat", "deepseek-ai/DeepSeek-V2-Lite-Chat"]:
            self.model.generation_config = GenerationConfig.from_pretrained(self.llm_model_name)
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def __call__(self, dataset_task_list: Optional[List[str]] = None) -> None:
        """
        Extract tabular data from documents in the specified dataset tasks.
        
        Args:
            dataset_task_list: Dataset tasks to process. If None, uses config or default.
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
            data_path = Path(self.config.get("data_main", self.config["out_main"])) / dataset_task
            out_root = Path(self.config["out_main"]) / dataset_task
            out_root.mkdir(parents=True, exist_ok=True)
            logging.info(f"[{self.__class__.__name__}:__call__] Start processing dataset: data_path={data_path}, out_root={out_root}")
            self._process_dataset(data_path, out_root)

    def _process_dataset(self, data_path: str | Path, out_root: str | Path):
        """Process all queries for one dataset."""
        data_path = Path(data_path)
        out_root = Path(out_root)
        
        # Create data loader based on configuration
        loader = create_data_loader(
            data_path=data_path,
            loader_type=self.loader_type,
            loader_config=self.loader_config
        )
        logging.info(f"[{self.__class__.__name__}:_process_dataset] Created {loader.__class__.__name__} for {data_path}")
        
        query_dict = loader.load_query_dict()
        schema_general = loader.load_schema_general()

        for qid in query_dict:
            query = query_dict[qid]
            query_text = query.get("query", "") if isinstance(query, dict) else str(query)
            schema_query = loader.load_schema_query(qid)
            res_path = out_root / PATH_TEMPLATES.data_population_result(qid, self.param_str)
            hs_dir = out_root / PATH_TEMPLATES.hidden_states_dir(qid, self.param_str)
            hs_dir.mkdir(parents=True, exist_ok=True)

            res_data = self.load_processed_res(res_path)
            pgbar_name = f"{data_path.name}-{qid}"

            logging.info(f"[{self.__class__.__name__}:_process_dataset] Start processing query-{qid}")
            logging.info(f"[{self.__class__.__name__}:_process_dataset] Writing results to: {res_path}")
            self._process_documents(
                loader, schema_general, schema_query, query_text, res_data, res_path, hs_dir, pgbar_name
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
            loader: DataLoaderBase,
            schema_general: List[Dict[str, Any]],
            schema_query: List[Dict[str, Any]],
            query: str,
            res_data: Dict[str, Any],
            res_path: Path,
            hs_dir: Path,
            pgbar_name: str,
            max_table_retries: int = 10,
            max_attr_retries: int = 10,
        ):
        """Iterate over documents and populate table/attribute data."""
        all_tables = [s[SCHEMA_NAME_KEY] for s in schema_general]
        table2schema = {s[SCHEMA_NAME_KEY]: s for s in schema_general}
        table2attr: Dict[str, List[str]] = {
            s[SCHEMA_NAME_KEY]: [a[ATTRIBUTE_NAME_KEY] for a in s[ATTRIBUTES_KEY]]
            for s in schema_query
            if s[SCHEMA_NAME_KEY] in all_tables
        }
        attr_general = []
        for tn in table2attr:
            attrs = [a[ATTRIBUTE_NAME_KEY] for a in table2schema[tn][ATTRIBUTES_KEY]]
            attr_general.append({SCHEMA_NAME_KEY: tn, ATTRIBUTES_KEY: attrs})

        num_doc = loader.num_docs
        progress_bar = tqdm(total=num_doc, desc=f"Processing {pgbar_name}")
        
        # Extract schema features for query augmentation if chunk filtering is enabled
        schema_features = None
        if self.chunk_filter and self.chunk_filter.config.enabled:
            schema_features = self._extract_schema_features(schema_query)

        for did, (doc_text, *_rest) in zip(loader.doc_ids, loader.iter_docs()):
            if did in res_data:
                progress_bar.update(1)
                continue

            # Apply chunk filtering if enabled
            filtered_doc_text = doc_text
            if self.chunk_filter and self.chunk_filter.config.enabled:
                filtered_chunks, filter_stats = self.chunk_filter.filter_chunks(
                    query=query,
                    document=doc_text,
                    schema_features=schema_features
                )
                filtered_doc_text = "\n\n".join(filtered_chunks)
                
                if filter_stats.get("filtered") and len(res_data) == 0:
                    logging.info(f"[{self.__class__.__name__}:_process_documents] "
                               f"Chunk filtering: {filter_stats.get('num_chunks', 0)} chunks -> "
                               f"{filter_stats.get('num_kept', 0)} kept, "
                               f"avg_similarity={filter_stats.get('avg_similarity', 0):.4f}")

            failed = False
            # ----------- Table assignment -------------------------
            table_attempt = 0
            while True:
                if table_attempt > max_table_retries:
                    logging.info(f"[{self.__class__.__name__}:_process_documents] Table fail "
                                 f">{max_table_retries}x for doc {did}. Skipping.")
                    failed = True
                    break
                tbl_input = {
                    DOCUMENT_KEY: filtered_doc_text,
                    SCHEMA_KEY: attr_general,
                }
                tbl_input = json.dumps(tbl_input, ensure_ascii=False)
                raw_text, token_info = self._llm_generate(self.prompt_table, tbl_input)
                res_tbl, ts, te = self._extract_json_block(raw_text, token_info)
                if not res_tbl or TABLE_ASSIGNMENT_KEY not in res_tbl:
                    table_attempt += 1
                    continue
                table_name = res_tbl[TABLE_ASSIGNMENT_KEY]
                if table_name not in all_tables:
                    table_attempt += 1
                    continue
                # Try to save span hidden-states; retry generation if span mismatch
                if self._save_span_hs(table_name, token_info[ts:te], hs_dir / f"doc-{did}-table.pt"):
                    break  # success
                table_attempt += 1
            
            if failed:
                progress_bar.update(1)
                continue
            
            result_entry: Dict[str, Any] = {"res": table_name, "data": {}}

            # ----------- Attribute extraction per attribute -------
            for attr in table2attr.get(table_name, []):
                attr_attempt = 0
                while True:
                    if attr_attempt > max_attr_retries:
                        logging.info(f"[{self.__class__.__name__}:_process_documents] Attr {attr} fail "
                                     f">{max_attr_retries}x doc {did}")
                        failed = True
                        break
                    attr_input = {
                        DOCUMENT_KEY: filtered_doc_text,
                        SCHEMA_KEY: table2schema[table_name],
                        TARGET_ATTRIBUTE_KEY: attr,
                    }
                    attr_input = json.dumps(attr_input, ensure_ascii=False)
                    raw_text, token_info = self._llm_generate(self.prompt_attr, attr_input)
                    res_attr, ts, te = self._extract_json_block(raw_text, token_info)
                    if not res_attr or attr not in res_attr:
                        logging.info(f"[{self.__class__.__name__}:_process_documents] Attr {attr} not found in "
                                     f"response. raw_text: {raw_text}")
                        attr_attempt += 1
                        continue
                    attr_val = str(res_attr[attr])
                    if len(attr_val) > 100:
                        logging.info(f"[{self.__class__.__name__}:_process_documents] Attribute data too long: "
                                     f"{attr_val}, please check it")
                    # Try save span; regenerate if mismatch
                    if self._save_span_hs(attr_val, token_info[ts:te], hs_dir / f"doc-{did}-attr-{attr}.pt"):
                        result_entry["data"][attr] = attr_val
                        break
                    logging.info(f"[{self.__class__.__name__}:_process_documents] Span mismatch for attr {attr}. "
                                 f"raw_text: {raw_text}; attr_val: {attr_val}")
                    attr_attempt += 1
                if attr_attempt > max_attr_retries:
                    break  # abandon rest attrs
            
            if failed:
                progress_bar.update(1)
                continue

            # Executed only if *no* break occurred – all attrs succeeded
            res_data[did] = result_entry
            self.save_results(res_path, res_data)
            progress_bar.update(1)

        progress_bar.close()
        logging.info(f"[{self.__class__.__name__}:_process_documents] Done -> {res_path}")
    
    def _llm_generate(self, prompt: str, msg: str) -> tuple:
        """Generate text using the LLM model with hidden states extraction."""
        messages = [{"role": "user", "content": prompt + "\n\n" + msg}]
        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        
        # Create attention mask (all 1s since we don't have padding in single sequence)
        # TODO: Check if this attention_mask handling is correct for other models (DeepSeek, etc.)
        # Some models may have different pad_token/eos_token configurations
        attention_mask = torch.ones_like(input_tensor) 
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=1000,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
            )
        
        gen_tokens = outputs.sequences[0][input_tensor.shape[1]:]
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # Get hidden states
        all_token_hiddens = torch.stack([
            torch.stack([layer_h[:, -1, :].cpu()
                        for layer_h in step_hiddens], 0)
            for step_hiddens in outputs.hidden_states
        ])  # -> (num_tokens, num_layers, hidden)

        # Get scores
        token_scores = torch.stack([s.squeeze(0).cpu() for s in outputs.scores])
        token_probs = torch.softmax(token_scores, dim=-1)

        token_info_pairs = []
        for step, (tok_id, hidden_states) in enumerate(zip(gen_tokens, all_token_hiddens)):
            token_text = self.tokenizer.decode(tok_id, skip_special_tokens=True)
            info = {
                "token_id": tok_id.item(),
                "token_text": token_text.encode("utf-8"),
                "hidden_states": hidden_states,  # shape: (num_layers, hidden_size)
                "prob": token_probs[step, tok_id].item(),
            }
            token_info_pairs.append(info)
                    
        return gen_text, token_info_pairs

    # ------------------ Utils -------------------

    def _save_span_hs(
            self, 
            span_text: str, 
            token_info_pairs: List[Dict[str, Any]], 
            out_path: Path
        ) -> bool:
        """
        Attempt to save the hidden states corresponding to span_text.
        First try exact token-ID matching; if that fails, fall back to
        concatenated token_text matching to handle merged-punctuation cases.
        """
        # 1) Exact token-ID matching
        target_ids = self.tokenizer.encode(span_text, add_special_tokens=False)
        gen_ids = [info["token_id"] for info in token_info_pairs]
        for idx in range(len(gen_ids) - len(target_ids) + 1):
            if gen_ids[idx : idx + len(target_ids)] == target_ids:
                torch.save(token_info_pairs[idx : idx + len(target_ids)], out_path)
                return True

        # 2) Concatenated token_text matching
        t_texts = [info["token_text"].decode("utf-8") for info in token_info_pairs]
        n = len(t_texts)
        for start in range(n):
            acc = ""
            for end in range(start, n):
                acc += t_texts[end]
                if self._compare_values(acc, span_text):
                    torch.save(token_info_pairs[start : end + 1], out_path)
                    return True
                if len(self._strip_punct(acc)) > len(self._strip_punct(span_text)):
                    break

        return False
    
    def _compare_values(self, val1: str, val2: str) -> bool:
        """Compare two values for equality."""
        def is_null(val):
            return val is None or val in ["", "null", "NULL", "None", "none"]

        val1 = self._strip_punct(val1)
        val2 = self._strip_punct(val2)
        if is_null(val1) and is_null(val2):
            return True
        return val1 == val2
    
    @staticmethod
    def _strip_punct(text: str) -> str:
        """Strip punctuation and whitespace from text."""
        punct = "\"'\n\t "
        res = text.lstrip(punct).rstrip(punct)
        res = res.replace("\\\"", "")
        res = res.replace("\"", "")
        return res

    @staticmethod
    def _load_prompt(path: str) -> str:
        """Load prompt template from file."""
        with open(path, "r") as fp:
            return fp.read()

    @staticmethod
    def _save_tensor_dict(file_path: str, data_dict: Dict[str, Any]):
        """Save tensor dictionary to file."""
        torch.save(data_dict, file_path)

    @staticmethod
    def _extract_json_block(raw_text: str, token_info_pairs: list) -> tuple:
        """
        Extract JSON object from raw_text and locate its start and end index in token_info_pairs.

        Returns:
            json_obj: parsed dictionary object (or None if not found)
            start_idx: index in token_info_pairs where JSON starts
            end_idx: index in token_info_pairs where JSON ends (exclusive)
        """
        # Step 1: Find JSON block in raw_text
        match = re.search(r'```json(.*?)```', raw_text, re.S | re.I)
        if not match:
            match = re.search(r'```(.*?)```', raw_text, re.S)

        json_str = None
        start_char, end_char = None, None

        if match:
            json_str = match.group(1).strip()
            start_char, end_char = match.start(1), match.end(1)
        else:
            # Step 2: Find JSON-like patterns in raw_text
            for m in re.finditer(r'\{.*?\}', raw_text, re.S):
                candidate = m.group(0)
                for parser in (json.loads, ast.literal_eval):
                    try:
                        obj = parser(candidate)
                        if isinstance(obj, dict):
                            json_str = candidate
                            start_char, end_char = m.start(), m.end()
                            break
                    except Exception:
                        continue
                if json_str:
                    break

        if not json_str:
            return None, None, None

        # Step 3: Parse JSON string
        json_obj = None
        for parser in (json.loads, ast.literal_eval):
            try:
                json_obj = parser(json_str)
                if isinstance(json_obj, dict):
                    break
            except Exception:
                continue

        if json_obj is None:
            return None, None, None

        # Step 4: Map start and end characters to token_info_pairs
        char_count = 0
        start_idx = end_idx = None
        for idx, tok in enumerate(token_info_pairs):
            char_count += len(tok["token_text"])
            if start_idx is None and char_count > start_char:
                start_idx = idx
            if start_idx is not None and char_count >= end_char:
                end_idx = idx + 1
                break

        return json_obj, start_idx, end_idx

    def __str__(self) -> str:
        """String representation of the local data populator."""
        return f"{self.__class__.__name__} (model={self.llm_model_name}): \n{self.param_str}"
