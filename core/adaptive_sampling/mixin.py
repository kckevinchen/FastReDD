"""
Mixin class for integrating adaptive sampling into schema generators.

This module provides a mixin class that can be added to existing schema
generator classes to enable adaptive sampling functionality without
major refactoring.

Supports two adaptive sampling algorithms:
- Entropy-based: Uses schema entropy and stability streaks
- DDGT: Diversity-Driven Good-Turing with probabilistic coverage
"""

import logging
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, List
from .entropy.sampler import AdaptiveSampler
from .schema_entropy import SchemaEntropyCalculator
from .entropy.document_selector import DocumentSelector
from .ddgt.sampler import DDGTSampler
from .ddgt.document_selector import DDGTDocumentSelector


class AdaptiveSamplingMixin:
    """
    Mixin class to add adaptive sampling capabilities to schema generators.
    
    This mixin provides:
    - Adaptive sampler initialization from config
    - Modified document processing loop with early stopping
    - Statistics tracking and reporting
    - Embedding-based document selection
    - Support for both entropy-based and DDGT algorithms
    
    Usage:
        class SchemaGenWithAdaptive(AdaptiveSamplingMixin, SchemaGenGPT):
            pass
    
    Or simply add to existing class:
    class SchemaGenGPT(SchemaGenBasic, AdaptiveSamplingMixin):
        ...
    """
    
    def init_adaptive_sampling(self, config: Dict[str, Any], api_key: Optional[str] = None):
        """
        Initialize adaptive sampling from configuration.
        
        Supports two algorithms:
        - "entropy" (default): Entropy-based with stability streaks
        - "ddgt": Diversity-Driven Good-Turing
        
        Expected config structure:
            adaptive_sampling:
                enabled: true
                algorithm: "ddgt"            # or "entropy" (default)
                
                # For DDGT:
                failure_probability: 0.05    # delta
                min_docs: 50                 # n_min
                batch_size: 5                # k
                
                # For entropy-based:
                entropy_threshold: 0.05      # theta
                streak_limit: 8              # m
                epsilon: 0.05                # epsilon
                
                # Common:
                use_embedding_selection: true
                embedding_file: "path/to/embeddings.json"
        
        Args:
            config: Configuration dictionary
            api_key: Optional API key for embedding generation
        """
        adaptive_config = config.get("adaptive_sampling", {})
        self.adaptive_enabled = adaptive_config.get("enabled", False)
        
        if not self.adaptive_enabled:
            logging.info(
                f"[{self.__class__.__name__}:init_adaptive_sampling] "
                "Adaptive sampling disabled"
            )
            self.adaptive_sampler = None
            self.document_selector = None
            self.adaptive_mode = None
            return
        
        # Determine algorithm type
        algorithm = adaptive_config.get("algorithm", "entropy").lower()
        self.adaptive_mode = algorithm
        
        if algorithm == "ddgt":
            # Initialize DDGT components
            delta = adaptive_config.get("failure_probability", adaptive_config.get("delta", 0.05))
            n_min = adaptive_config.get("min_docs", adaptive_config.get("n_min", 50))
            batch_size = adaptive_config.get("batch_size", 5)
            
            self.adaptive_sampler = DDGTSampler(
                delta=delta,
                n_min=n_min,
                batch_size=batch_size
            )
            
            self.document_selector = DDGTDocumentSelector(config, api_key=api_key)
            
            logging.info(
                f"[{self.__class__.__name__}:init_adaptive_sampling] "
                f"DDGT adaptive sampling enabled with parameters: "
                f"delta={delta}, n_min={n_min}, batch_size={batch_size}"
            )
        else:
            # Initialize entropy-based components (default)
            theta = adaptive_config.get("entropy_threshold", adaptive_config.get("theta", 0.05))
            m = adaptive_config.get("streak_limit", adaptive_config.get("m", 8))
            n_min = adaptive_config.get("min_docs", adaptive_config.get("n_min", 10))
            delta = adaptive_config.get("failure_probability", adaptive_config.get("delta", 0.05))
            epsilon = adaptive_config.get("epsilon", 0.05)
            probabilistic_stop = adaptive_config.get("probabilistic_stop", True)
            
            self.adaptive_sampler = AdaptiveSampler(
                theta=theta,
                m=m,
                n_min=n_min,
                delta=delta,
                epsilon=epsilon,
                enable_probabilistic_stop=probabilistic_stop
            )
            
            self.document_selector = DocumentSelector(config, api_key=api_key)
            
            logging.info(
                f"[{self.__class__.__name__}:init_adaptive_sampling] "
                f"Entropy-based adaptive sampling enabled with parameters: "
                f"theta={theta}, m={m}, n_min={n_min}, delta={delta}, epsilon={epsilon}"
            )
    
    def process_documents_adaptive(
        self, 
        doc_dict, 
        query, 
        res_dict, 
        log_init, 
        general_schema, 
        res_path, 
        pgbar_name,
        original_indices=None,
        current_schema_path=None
    ):
        """
        Process documents with adaptive sampling (replacement for process_documents).
        
        Dispatches to appropriate method based on algorithm mode:
        - "ddgt": Batch-based processing with Good-Turing stopping
        - "entropy": Incremental processing with entropy-based stopping
        
        Args:
            doc_dict: Dictionary of documents {doc_id: [doc_text, source_info]} 
            query: Query string (if applicable)
            res_dict: Existing results dictionary
            log_init: Initial log/schema state
            general_schema: General schema context
            res_path: Path to save results
            pgbar_name: Name for progress bar
            original_indices: Optional list mapping shuffled indices to original indices.
                            If None, assumes doc_dict is in original order.
        """
        from tqdm import tqdm
        
        if not self.adaptive_enabled or self.adaptive_sampler is None:
            # Fall back to regular processing
            logging.warning(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                "Adaptive sampling not enabled, using standard processing"
            )
            return self.process_documents(
                doc_dict, query, res_dict, log_init, general_schema, 
                res_path, pgbar_name, original_indices, current_schema_path
            )
        
        # Dispatch to appropriate processing method
        if self.adaptive_mode == "ddgt":
            return self._process_documents_ddgt(
                doc_dict, query, res_dict, log_init, general_schema,
                res_path, pgbar_name, original_indices, current_schema_path
            )
        else:
            return self._process_documents_entropy(
                doc_dict, query, res_dict, log_init, general_schema,
                res_path, pgbar_name, original_indices, current_schema_path
            )
    
    def _process_documents_entropy(
        self, 
        doc_dict, 
        query, 
        res_dict, 
        log_init, 
        general_schema, 
        res_path, 
        pgbar_name,
        original_indices=None,
        current_schema_path=None
    ):
        """
        Process documents with entropy-based adaptive sampling (original method).
        
        This implements incremental document selection with entropy-based stopping.
        
        Args:
            doc_dict: Dictionary of documents {doc_id: [doc_text, source_info]} 
            query: Query string (if applicable)
            res_dict: Existing results dictionary
            log_init: Initial log/schema state
            general_schema: General schema context
            res_path: Path to save results
            pgbar_name: Name for progress bar
            original_indices: Optional list mapping shuffled indices to original indices.
        """
        from tqdm import tqdm
        # Reset adaptive sampler for this query/dataset
        self.adaptive_sampler.reset()
        
        # Build embedding index if enabled (for efficient incremental selection)
        if self.document_selector and self.document_selector.enabled:
            logging.info(f"[{self.__class__.__name__}:process_documents_adaptive] Building embedding index for incremental farthest-from-mean selection...")
            index_built = self.document_selector.build_index(doc_dict)
            if not index_built:
                logging.warning(
                    f"[{self.__class__.__name__}:process_documents_adaptive] "
                    "Embedding index build failed, falling back to sequential processing"
                )
        
        # Track available and selected documents for incremental selection
        all_doc_ids = set(doc_dict.keys())
        available_doc_ids = all_doc_ids.copy()  # Documents not yet processed
        selected_doc_ids = []  # Documents selected in order (for farthest-from-mean calculation)
        
        # Remove already processed documents from available set
        already_processed = set(res_dict.keys())
        available_doc_ids -= already_processed
        
        num_doc = len(all_doc_ids)
        num_remaining = len(available_doc_ids)
        cnt = 0
        progress_bar = tqdm(total=num_doc, desc=f"Processing {pgbar_name} (Adaptive)")
        
        logging.info(
            f"[{self.__class__.__name__}:process_documents_adaptive] "
            f"Start adaptive processing: query={query}"
        )
        logging.info(
            f"[{self.__class__.__name__}:process_documents_adaptive] "
            f"Documents: total={num_doc}, already_processed={len(already_processed)}, remaining={num_remaining}"
        )
        
        stopped_early = False
        docs_processed_count = len(already_processed)  # Count already processed
        current_log = log_init  # Track cumulative schema state
        
        # Initialize current_log from existing results if resuming
        if res_dict:
            # Use the most complete schema from existing results as starting point
            best_result = max(res_dict.values(), key=lambda r: len(r.get("log", [])))
            current_log = best_result.get("log", log_init)
            logging.info(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                f"Resuming with existing schema from {len(res_dict)} processed documents"
            )
        
        # Process documents incrementally until early stop or all processed
        i = 0
        while available_doc_ids:
            # Select next document: use farthest-from-mean if enabled, otherwise random
            if self.document_selector and self.document_selector.enabled and self.document_selector._index_built:
                next_doc_id = self.document_selector.select_next_farthest_from_mean(available_doc_ids, selected_doc_ids)
                if next_doc_id is None:
                    # Fallback to random if selection fails
                    next_doc_id = random.choice(list(available_doc_ids))
            else:
                # No embedding selection: use random
                next_doc_id = random.choice(list(available_doc_ids))
            
            doc_id_str = next_doc_id
            
            # Map doc_id to original index for result storage
            try:
                shuffled_idx = int(doc_id_str)
                original_idx = original_indices[shuffled_idx] if original_indices else shuffled_idx
            except (ValueError, KeyError, TypeError):
                original_idx = doc_id_str
            
            # current_log is maintained as cumulative schema state
            # It's updated after each successful processing
            
            # Prepare and process document
            input_json = self.prepare_input_json(
                doc_dict, doc_id_str, 
                query, current_log, general_schema
            )
            
            out_dict = self.process_single_document(input_json, cnt, doc_id_str)
            result_data = self.extract_result_data(out_dict)
            
            # Handle processing errors with retry
            if not result_data or (result_data and len(result_data.get("log", [])) < len(current_log)):
                cnt += 1
                if cnt > 10:
                    if not result_data:
                        logging.error(
                            f"[{self.__class__.__name__}:process_documents_adaptive] "
                            f"Failed to process document {doc_id_str} (original: {original_idx}) after {cnt} retries!"
                        )
                        raise RuntimeError(f"Failed to process document {doc_id_str} (original: {original_idx}) after {cnt} retries")
                    else:
                        logging.error(
                            f"[{self.__class__.__name__}:process_documents_adaptive] "
                            f"Schema num decrease, retry_count {cnt}, doc_index {doc_id_str} (original: {original_idx})"
                        )
                        raise RuntimeError(f"Schema num decrease, retry_count {cnt}, doc_index {doc_id_str} (original: {original_idx})")
                
                # Retry same document (don't advance, don't remove from available)
                continue
            
            # Successfully processed document
            cnt = 0
            current_log = result_data["log"]
            
            # Save result with original index
            res_dict[str(original_idx)] = result_data
            self.save_results(res_path, res_dict)
            
            # Update current schema result
            if current_schema_path and hasattr(self, '_update_current_schema'):
                self._update_current_schema(res_dict, current_schema_path)
            
            # Update tracking: add to selected list and remove from available
            selected_doc_ids.append(doc_id_str)  # Track for farthest-from-mean calculation
            available_doc_ids.remove(doc_id_str)  # Remove from available set
            docs_processed_count += 1
            i += 1
            
            # Check adaptive stopping criterion
            current_schema = result_data["log"]
            if not self.adaptive_sampler.should_continue(current_schema):
                stopped_early = True
                progress_bar.update(1)
                logging.info(
                    f"[{self.__class__.__name__}:process_documents_adaptive] "
                    f"Early stopping triggered at document {i}/{num_doc} (original index: {original_idx})"
                )
                break
            
            progress_bar.update(1)
            logging.debug(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                f"Processed document {i}/{num_doc} (original index: {original_idx})"
            )
        
        progress_bar.close()
        
        # Save adaptive sampling statistics
        stats = self.adaptive_sampler.get_statistics()
        stats["total_documents"] = num_doc
        stats["stopped_early"] = stopped_early
        stats["documents_processed"] = docs_processed_count
        stats["documents_saved"] = num_doc - (docs_processed_count + len([k for k in res_dict if k in doc_dict])) if stopped_early else 0
        
        self._save_adaptive_stats(res_path, stats)
        
        # Log summary
        if stopped_early:
            logging.info(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                f"Finished with early stopping: processed {docs_processed_count} new documents "
            )
            logging.info(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                f"Stop reason: {self.adaptive_sampler.get_stop_reason()}"
            )
        else:
            logging.info(
                f"[{self.__class__.__name__}:process_documents_adaptive] "
                f"Finished processing all documents (no early stopping)"
            )
        
        logging.info(
            f"[{self.__class__.__name__}:_process_documents_entropy] "
            f"Final entropy statistics: {stats['entropy_statistics']}"
        )
    
    def _process_documents_ddgt(
        self,
        doc_dict,
        query,
        res_dict,
        log_init,
        general_schema,
        res_path,
        pgbar_name,
        original_indices=None,
        current_schema_path=None
    ):
        """
        Process documents using DDGT batch sampling with Good-Turing stopping.
        
        Implements Phase 2 loop from DDGT specification:
        1. Select batch using diversity sampling
        2. Process batch and extract features
        3. Check Good-Turing stopping condition
        4. Repeat until stopped or corpus exhausted
        
        Args:
            doc_dict: Dictionary of documents {doc_id: [doc_text, source_info]} 
            query: Query string (if applicable)
            res_dict: Existing results dictionary
            log_init: Initial log/schema state
            general_schema: General schema context
            res_path: Path to save results
            pgbar_name: Name for progress bar
            original_indices: Optional list mapping shuffled indices to original indices.
        """
        from tqdm import tqdm
        
        # Reset sampler for this query/dataset
        self.adaptive_sampler.reset()
        
        # Build embedding index for diversity sampling
        if self.document_selector and self.document_selector.enabled:
            logging.info(
                f"[{self.__class__.__name__}:_process_documents_ddgt] "
                "Building embedding index for max-min diversity selection..."
            )
            index_built = self.document_selector.build_index(doc_dict)
            if not index_built:
                logging.warning(
                    f"[{self.__class__.__name__}:_process_documents_ddgt] "
                    "Embedding index build failed, falling back to random selection"
                )
        
        # Track sampled documents
        all_doc_ids = set(doc_dict.keys())
        sampled_doc_ids = set(res_dict.keys())  # Already processed
        
        num_doc = len(all_doc_ids)
        num_remaining = len(all_doc_ids - sampled_doc_ids)
        cnt = 0
        
        progress_bar = tqdm(total=num_doc, desc=f"Processing {pgbar_name} (DDGT)")
        progress_bar.update(len(sampled_doc_ids))  # Update for already processed
        
        logging.info(
            f"[{self.__class__.__name__}:_process_documents_ddgt] "
            f"Start DDGT processing: query={query}"
        )
        logging.info(
            f"[{self.__class__.__name__}:_process_documents_ddgt] "
            f"Documents: total={num_doc}, already_processed={len(sampled_doc_ids)}, remaining={num_remaining}"
        )
        
        stopped_early = False
        docs_processed_count = len(sampled_doc_ids)
        current_log = log_init
        
        # Initialize current_log from existing results if resuming
        if res_dict:
            best_result = max(res_dict.values(), key=lambda r: len(r.get("log", [])))
            current_log = best_result.get("log", log_init)
            logging.info(
                f"[{self.__class__.__name__}:_process_documents_ddgt] "
                f"Resuming with existing schema from {len(res_dict)} processed documents"
            )
        
        # Phase 2: Adaptive Sampling Loop
        while True:
            # Step A: Diversity Sampling (select batch)
            if self.document_selector and self.document_selector.enabled and self.document_selector._index_built:
                batch_doc_ids = self.document_selector.select_batch_maxmin(
                    sampled_doc_ids=sampled_doc_ids,
                    batch_size=self.adaptive_sampler.batch_size
                )
            else:
                # Fallback to random selection
                available = list(all_doc_ids - sampled_doc_ids)
                random.shuffle(available)
                batch_doc_ids = available[:self.adaptive_sampler.batch_size]
            
            # Handle corpus exhaustion
            if not batch_doc_ids:
                logging.info(
                    f"[{self.__class__.__name__}:_process_documents_ddgt] "
                    "Corpus exhausted, no more documents to process"
                )
                break
            
            logging.info(
                f"[{self.__class__.__name__}:_process_documents_ddgt] "
                f"Selected batch of {len(batch_doc_ids)} documents"
            )
            
            # Step B: Feature Extraction (process batch)
            for doc_id_str in batch_doc_ids:
                # Map doc_id to original index for result storage
                try:
                    shuffled_idx = int(doc_id_str)
                    original_idx = original_indices[shuffled_idx] if original_indices else shuffled_idx
                except (ValueError, KeyError, TypeError):
                    original_idx = doc_id_str
                
                # Prepare and process document
                input_json = self.prepare_input_json(
                    doc_dict, doc_id_str,
                    query, current_log, general_schema
                )
                
                out_dict = self.process_single_document(input_json, cnt, doc_id_str)
                result_data = self.extract_result_data(out_dict)
                
                # Handle processing errors with retry
                if not result_data or (result_data and len(result_data.get("log", [])) < len(current_log)):
                    cnt += 1
                    if cnt > 10:
                        if not result_data:
                            logging.error(
                                f"[{self.__class__.__name__}:_process_documents_ddgt] "
                                f"Failed to process document {doc_id_str} (original: {original_idx}) after {cnt} retries!"
                            )
                            raise RuntimeError(f"Failed to process document {doc_id_str} after {cnt} retries")
                        else:
                            logging.error(
                                f"[{self.__class__.__name__}:_process_documents_ddgt] "
                                f"Schema num decrease, retry_count {cnt}, doc_index {doc_id_str} (original: {original_idx})"
                            )
                            raise RuntimeError(f"Schema num decrease, retry_count {cnt}, doc_index {doc_id_str}")
                    
                    # Retry: skip updating features, retry same document next iteration
                    continue
                
                # Successfully processed document
                cnt = 0
                current_log = result_data["log"]
                
                # Save result with original index
                res_dict[str(original_idx)] = result_data
                self.save_results(res_path, res_dict)
                
                # Update current schema result
                if current_schema_path and hasattr(self, '_update_current_schema'):
                    self._update_current_schema(res_dict, current_schema_path)
                
                # Extract features and update DDGT sampler
                features = self.adaptive_sampler.extract_features_from_schema(current_log)
                self.adaptive_sampler.update_features(features, doc_id_str)
                
                # Mark as sampled
                sampled_doc_ids.add(doc_id_str)
                docs_processed_count += 1
                
                progress_bar.update(1)
                
                logging.debug(
                    f"[{self.__class__.__name__}:_process_documents_ddgt] "
                    f"Processed document {doc_id_str} (original: {original_idx}): "
                    f"{len(features)} features, total unique: {len(self.adaptive_sampler.F_current)}"
                )
            
            # Step C: Check Good-Turing stopping condition
            if self.adaptive_sampler.check_stopping_condition():
                stopped_early = True
                logging.info(
                    f"[{self.__class__.__name__}:_process_documents_ddgt] "
                    f"Early stopping triggered at {docs_processed_count}/{num_doc} documents"
                )
                break
        
        progress_bar.close()
        
        # Save adaptive sampling statistics
        stats = self.adaptive_sampler.get_statistics()
        stats["total_documents"] = num_doc
        stats["stopped_early"] = stopped_early
        stats["documents_processed"] = docs_processed_count
        stats["documents_saved"] = num_doc - docs_processed_count if stopped_early else 0
        
        self._save_adaptive_stats(res_path, stats)
        
        # Log summary
        if stopped_early:
            logging.info(
                f"[{self.__class__.__name__}:_process_documents_ddgt] "
                f"Finished with early stopping: processed {docs_processed_count} documents"
            )
            logging.info(
                f"[{self.__class__.__name__}:_process_documents_ddgt] "
                f"Stop reason: {self.adaptive_sampler.get_stop_reason()}"
            )
        else:
            logging.info(
                f"[{self.__class__.__name__}:_process_documents_ddgt] "
                f"Finished processing all documents (no early stopping)"
            )
        
        logging.info(
            f"[{self.__class__.__name__}:_process_documents_ddgt] "
            f"Final DDGT statistics: {stats}"
        )
    
    def _save_adaptive_stats(self, res_path: Path, stats: Dict[str, Any]):
        """
        Save adaptive sampling statistics alongside results.
        
        Args:
            res_path: Path to results file
            stats: Statistics dictionary
        """
        res_path = Path(res_path)
        stats_path = res_path.parent / f"{res_path.stem}_adaptive_stats.json"
        
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            
            logging.info(
                f"[{self.__class__.__name__}:_save_adaptive_stats] "
                f"Saved adaptive statistics to {stats_path}"
            )
        except Exception as e:
            logging.error(
                f"[{self.__class__.__name__}:_save_adaptive_stats] "
                f"Failed to save adaptive statistics: {e}"
            )
    
    def get_adaptive_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get current adaptive sampling statistics.
        
        Returns:
            Statistics dictionary, or None if adaptive sampling not enabled
        """
        if not self.adaptive_enabled or self.adaptive_sampler is None:
            return None
        
        return self.adaptive_sampler.get_statistics()
