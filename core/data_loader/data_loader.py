"""
Unified Data Loader

A single, standardized data loader that works with a consistent JSON-based format.
This simplifies dataset management and eliminates the need for multiple loader types.

Standard Dataset Format:
------------------------
A dataset directory should contain the following files:

1. documents.json (required)
   Format: {
       "doc_id": {
           "doc": "document text content",
           "table_name": "table or category name",
           "data": {...},  # optional structured data
           ...  # any additional metadata
       }
   }

2. queries.json (optional)
   Format: {
       "query_id": {
           "query": "natural language question",
           "attributes": ["attr1", "attr2", ...],  # optional
           "sql": "SQL query",  # optional
           ...  # any additional fields
       }
   }

3. schema_general.json (optional)
   Format: [
       {
           "Schema Name": "schema_name",
           "Attributes": [
               {
                   "Attribute Name": "attr_name",
                   "Description": "attribute description"
               }
           ]
       }
   ]

4. schema_query_{query_id}.json (optional)
   Format: Same as schema_general.json but query-specific

Backward Compatibility:
-----------------------
The loader also supports legacy formats:
- doc_info.json (maps to documents.json)
- doc_dict.json (Spider format, converted automatically)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

from .data_loader_basic import DataLoaderBase


class DataLoaderKC(DataLoaderBase):
    """
    Unified data loader for standardized JSON-based datasets.
    
    This loader works with a consistent format while maintaining backward
    compatibility with existing dataset structures.
    """
    
    # Standard file names
    DOCUMENTS_FILE = "documents.json"
    DOC_INFO_FILE = "doc_info.json"  # Legacy support
    DOC_DICT_FILE = "doc_dict.json"  # Legacy Spider format
    QUERIES_FILE = "queries.json"
    SCHEMA_GENERAL_FILE = "schema_general.json"
    SCHEMA_QUERY_PREFIX = "schema_query_"
    
    def __init__(
        self,
        data_root: str | Path,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize unified data loader.
        
        Args:
            data_root: Path to dataset directory (can be relative or absolute)
                     Examples: "college_2", "spider_sqlite/college_2",
                              "dataset/spider_sqlite/college_2"
            config: Optional configuration dict with:
                - data_main: Base data directory (default: "dataset/")
                - strict: If True, only accept standard format (default: False)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.config = config or {}
        self.strict = self.config.get("strict", False)
        
        # Resolve dataset root path
        self.data_root = self._resolve_dataset_path(data_root)
        
        # Initialize base class
        super().__init__(self.data_root)
        
        # Load documents
        self._documents = self._load_documents()
        
        # Cache for queries and schemas (loaded lazily)
        self._queries: Optional[Dict[str, Any]] = None
        self._schema_general: Optional[List[Dict[str, Any]]] = None
        self._schema_query_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        logging.info(f"[{self.__class__.__name__}:__init__] "
                    f"Initialized loader with {len(self._documents)} documents")
        logging.info(f"[{self.__class__.__name__}:__init__] "
                    f"Dataset root: {self.data_root}")
    
    def _resolve_dataset_path(self, dataset_path: str | Path) -> Path:
        """
        Resolve dataset path by trying multiple common locations.
        
        Args:
            dataset_path: Dataset path (can be relative or absolute)
        
        Returns:
            Resolved absolute path to dataset root
        """
        dataset_path = Path(dataset_path)
        
        # If absolute path and exists, use it directly
        if dataset_path.is_absolute() and dataset_path.exists():
            return dataset_path
        
        # Get base directories from config
        data_main = Path(self.config.get("data_main", "dataset/"))
        # Ensure data_main is resolved to absolute path
        if not data_main.is_absolute():
            data_main = (Path.cwd() / data_main).resolve()
        
        # Try different base paths
        search_paths = [
            # Direct path
            dataset_path,
            # Relative to current directory
            Path.cwd() / dataset_path,
            # Relative to data_main
            data_main / dataset_path,
            # Try with common prefixes in dataset folder
            data_main / "spider_sqlite" / dataset_path,
            data_main / "quest_sqlite" / dataset_path,
            data_main / "spider_update" / dataset_path,
            data_main / "spider" / dataset_path,
            data_main / "bird" / dataset_path,
            data_main / "galois" / dataset_path,
        ]
        
        # Try each path
        resolved_path = None
        for path in search_paths:
            path = path.resolve()
            if path.exists() and path.is_dir():
                # Check if it looks like a dataset directory
                # Standard format: documents.json
                if (path / self.DOCUMENTS_FILE).exists():
                    return path
                # Legacy formats
                if (path / self.DOC_INFO_FILE).exists():
                    return path
                if (path / self.DOC_DICT_FILE).exists():
                    return path
        
        # If no valid path found, use the first one (will raise error in base class)
        return Path(dataset_path).resolve()
    
    def _load_documents(self) -> Dict[str, Any]:
        """
        Load documents from the dataset.
        Supports multiple formats for backward compatibility.
        
        Returns:
            Dictionary mapping doc_id -> document info
        """
        # Try standard format first
        documents_path = self.data_root / self.DOCUMENTS_FILE
        if documents_path.exists():
            return self._read_json(documents_path)
        
        # Try legacy doc_info.json format
        doc_info_path = self.data_root / self.DOC_INFO_FILE
        if doc_info_path.exists():
            return self._read_json(doc_info_path)
        
        # Try legacy Spider doc_dict.json format
        doc_dict_path = self.data_root / self.DOC_DICT_FILE
        if doc_dict_path.exists():
            doc_dict = self._read_json(doc_dict_path)
            # Convert Spider format to standard format
            return self._convert_spider_format(doc_dict)
        
        # If strict mode, raise error
        if self.strict:
            logging.error(f"[{self.__class__.__name__}:_load_documents] "
                         f"Required file not found. Expected one of: "
                         f"{self.DOCUMENTS_FILE}, {self.DOC_INFO_FILE}, {self.DOC_DICT_FILE}")
            raise FileNotFoundError(
                f"[{self.__class__.__name__}:_load_documents] "
                f"Required file not found. Expected one of: "
                f"{self.DOCUMENTS_FILE}, {self.DOC_INFO_FILE}, {self.DOC_DICT_FILE}"
            )
        
        # Return empty dict if nothing found
        logging.warning(f"[{self.__class__.__name__}:_load_documents] "
                       f"No document file found in {self.data_root}")
        return {}
    
    def _convert_spider_format(self, doc_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Spider format (doc_dict.json) to standard format.
        
        Spider format: doc_id -> (doc_text, table_name, raw_id) or dict
        Standard format: doc_id -> {"doc": ..., "table_name": ..., ...}
        
        Args:
            doc_dict: Dictionary in Spider format
            
        Returns:
            Dictionary in standard format
        """
        converted = {}
        for doc_id, doc_value in doc_dict.items():
            if isinstance(doc_value, tuple) and len(doc_value) >= 2:
                # Tuple format: (doc_text, table_name, raw_id)
                converted[doc_id] = {
                    "doc": doc_value[0],
                    "table_name": doc_value[1],
                    "raw_id": doc_value[2] if len(doc_value) > 2 else doc_id,
                }
            elif isinstance(doc_value, dict):
                # Already in dict format, ensure it has "doc" key
                converted[doc_id] = doc_value.copy()
                if "doc" not in converted[doc_id]:
                    converted[doc_id]["doc"] = converted[doc_id].get("doc_text", "")
            else:
                # String format
                converted[doc_id] = {
                    "doc": str(doc_value),
                    "table_name": "",
                }
        return converted
    
    # ============ CORE DOCUMENT ACCESS ============
    
    @property
    def num_docs(self) -> int:
        """Total number of documents."""
        return len(self._documents)
    
    @property
    def doc_ids(self) -> List[str]:
        """List of all document IDs."""
        return sorted(self._documents.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    def iter_docs(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """
        Iterate over all documents.
        
        Yields:
            Tuple of (doc_text, doc_id, metadata_dict)
        """
        for doc_id in self.doc_ids:
            yield self.get_doc(doc_id)
    
    def get_doc(self, doc_id: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Get a single document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Tuple of (doc_text, doc_id, metadata_dict)
        """
        if doc_id not in self._documents:
            logging.error(f"[{self.__class__.__name__}:get_doc] Document not found: {doc_id}")
            raise KeyError(f"[{self.__class__.__name__}:get_doc] Document not found: {doc_id}")
        
        doc_info = self._documents[doc_id]
        
        # Extract document text
        doc_text = doc_info.get("doc", doc_info.get("doc_text", ""))
        
        # Extract table name (support multiple field names)
        table_name = doc_info.get("table_name", doc_info.get("fn", ""))
        
        # Build metadata dict
        metadata = {
            "table_name": table_name,
            "doc_id": doc_id,
        }
        
        # Add any additional fields
        for key, value in doc_info.items():
            if key not in ["doc", "doc_text"]:
                metadata[key] = value
        
        return (doc_text, doc_id, metadata)
    
    def get_doc_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary with document info, or None if not found
        """
        if doc_id not in self._documents:
            return None
        
        doc_info = self._documents[doc_id].copy()
        doc_info["doc_id"] = doc_id
        return doc_info
    
    # ============ QUERY ACCESS ============
    
    @property
    def num_queries(self) -> int:
        """Total number of queries."""
        if self._queries is None:
            self._queries = self._load_queries()
        return len(self._queries)
    
    @property
    def query_ids(self) -> List[str]:
        """List of all query IDs."""
        if self._queries is None:
            self._queries = self._load_queries()
        return sorted(self._queries.keys())
    
    def _load_queries(self) -> Dict[str, Any]:
        """Load queries from queries.json."""
        queries_path = self.data_root / self.QUERIES_FILE
        if queries_path.exists():
            return self._read_json(queries_path)
        return {}
    
    def load_query_dict(self) -> Dict[str, Any]:
        """Load all queries as a dictionary."""
        if self._queries is None:
            self._queries = self._load_queries()
        return self._queries.copy()
    
    def get_query_info(self, qid: str | int) -> Optional[Dict[str, Any]]:
        """Get information for a specific query."""
        qid_str = str(qid)
        return self.load_query_dict().get(qid_str)
    
    # ============ SCHEMA ACCESS ============
    
    def load_schema_general(self) -> List[Dict[str, Any]]:
        """Load dataset-wide schema information."""
        if self._schema_general is None:
            schema_path = self.data_root / self.SCHEMA_GENERAL_FILE
            if schema_path.exists():
                self._schema_general = self._read_json(schema_path)
                if not isinstance(self._schema_general, list):
                    self._schema_general = []
            else:
                self._schema_general = []
        return self._schema_general.copy()
    
    def load_schema_query(self, qid: str | int) -> List[Dict[str, Any]]:
        """Load query-specific schema."""
        qid_str = str(qid)
        
        # Check cache first
        if qid_str in self._schema_query_cache:
            return self._schema_query_cache[qid_str]
        
        # Try to load schema_query_{qid}.json
        schema_query_path = self.data_root / f"{self.SCHEMA_QUERY_PREFIX}{qid_str}.json"
        if schema_query_path.exists():
            schema = self._read_json(schema_query_path)
            if isinstance(schema, list):
                self._schema_query_cache[qid_str] = schema
                return schema.copy()
        
        # Return empty list if not found
        return []
    
    # ============ CONVENIENCE METHODS ============
    
    def get_doc_dict(self) -> Dict[str, Any]:
        """
        Get doc_dict in the format expected by schema generation.
        
        Returns:
            Dictionary mapping doc_id -> tuple(doc_text, table_name, raw_id)
        """
        doc_dict = {}
        for doc_id in self.doc_ids:
            doc_text, _, metadata = self.get_doc(doc_id)
            table_name = metadata.get("table_name", "")
            raw_id = metadata.get("raw_id", doc_id)
            doc_dict[doc_id] = (doc_text, table_name, raw_id)
        return doc_dict
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information summary.
        
        Returns:
            Dictionary with dataset metadata
        """
        return {
            "data_root": str(self.data_root),
            "num_docs": self.num_docs,
            "num_queries": self.num_queries,
            "has_schemas": self.has_schemas(),
            "has_queries": self.has_queries(),
        }
    
    def find_file(self, filename: str) -> Optional[Path]:
        """
        Find a file in the dataset directory.
        
        Args:
            filename: Name of file to find
        
        Returns:
            Path to file if found, None otherwise
        """
        file_path = self.data_root / filename
        if file_path.exists():
            return file_path
        return None

