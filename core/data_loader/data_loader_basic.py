# data_loader/data_loader_basic.py
"""Base *interface* for dataset loaders.

This file contains **no concrete I/O logic** - it only specifies *what* a
loader must provide so that the rest of the pipeline can work with any
back-end implementation (JSON eager read, streaming folder read, HTML
scraping, SQLite, etc.).

Design Philosophy
-----------------
This interface is designed to be **maximally flexible** to accommodate various
data sources and formats:
- Documents can be stored in JSON, individual files, databases, etc.
- Queries and schemas are **optional** (not all datasets have them)
- Metadata format is flexible and extensible
- Both eager and lazy loading patterns are supported

Core Abstractions
-----------------

**Documents** - The fundamental unit of data:
    - Required: `doc_id` (unique identifier) and `doc_text` (content)
    - Optional: any metadata (source file, table name, parent doc, etc.)
    - Access via: `iter_docs()`, `get_doc()`, `get_doc_info()`

**Queries** (Optional) - Structured questions over documents:
    - Format: dict keyed by query_id with flexible structure
    - Minimum: `{"query": "..."}` (natural language question)
    - Optional fields: `attributes`, `sql`, `answer`, etc.
    - Access via: `load_query_dict()`, `get_query_info()`

**Schemas** (Optional) - Structural metadata about data:
    - Can be dataset-wide or query-specific
    - Flexible format supporting various schema representations
    - Access via: `load_schema_general()`, `load_schema_query()`

**Metadata** (Optional) - Additional document information:
    - Ground truth data, tags, categories, timestamps, etc.
    - Format is implementation-dependent
    - Access via: `get_doc_info()`, `get_doc_metadata()`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Iterator, Tuple, Optional

# ---------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------

class DataLoaderBase(ABC):
    """A minimal contract every dataset reader must fulfill.
    
    This interface supports both in-memory (eager) and streaming (lazy) loading
    patterns. Implementations should choose the appropriate strategy based on
    dataset size and usage patterns.
    
    Attributes:
        data_root: Root directory of the dataset
    """

    def __init__(
        self, 
        data_root: str | Path
    ):
        """Initialize the dataset loader.
        
        Args:
            data_root: Path to the dataset root directory
            
        Raises:
            FileNotFoundError: If data_root does not exist
        """
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.exists():
            logging.error(f"[{self.__class__.__name__}:__init__] Dataset root not found: {self.data_root}")
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")

    # ============ CORE DOCUMENT ACCESS (REQUIRED) ============

    @property
    @abstractmethod
    def num_docs(self) -> int:
        """Total number of documents (should be cheap to compute).
        
        Returns:
            Number of documents in the dataset
        """

    @property
    @abstractmethod
    def doc_ids(self) -> List[str]:
        """List of all document IDs in stable order.
        
        Returns:
            List of document IDs (strings)
        """

    @abstractmethod
    def iter_docs(self) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """Iterate over all documents in the dataset.
        
        This method should support streaming - implementations must NOT assume
        all documents fit in memory. 
        
        Yields:
            Tuple of (doc_text, doc_id, metadata_dict) where metadata_dict contains
            flexible metadata fields such as:
            - "source_file": source file name
            - "table_name": table name (for Spider-style datasets)
            - "parent_doc_id": parent document ID (for chunked documents)
            - "chunk_index": chunk index (for chunked documents)
            - Any other custom fields
        """

    @abstractmethod
    def get_doc(self, doc_id: str) -> Tuple[str, str, Dict[str, Any]]:
        """Get a single document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Tuple of (doc_text, doc_id, metadata_dict)
        """

    @abstractmethod
    def get_doc_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary with document info, or None if not found.
            
            The structure is flexible and all fields are optional:
            - "doc": document text (optional)
            - "mappings": list of table mappings (table_name, row_id)
            - "data_records": detailed list with table/row context and data dict
            - additional metadata fields (flexible)
            
            Implementations should handle missing fields gracefully using .get() method.
        """

    # ============ QUERY ACCESS (OPTIONAL) ============
    # Note: Not all datasets have queries. Implementations should return
    # appropriate empty values if queries are not applicable.

    @property
    def num_queries(self) -> int:
        """Total number of queries (cheap to compute).
        
        Returns:
            Number of queries, or 0 if dataset has no queries
            
        Note:
            Default implementation returns 0. Override if dataset has queries.
        """
        return 0

    @property
    def query_ids(self) -> List[str]:
        """List of all query IDs.
        
        Returns:
            List of query IDs, or empty list if no queries
            
        Note:
            Default implementation returns empty list. Override if dataset has queries.
        """
        return []

    def load_query_dict(self) -> Dict[str, Any]:
        """Load all queries as a dictionary.
        
        Returns:
            Dictionary mapping query_id -> query_info. Empty dict if no queries.
            Query info format is flexible but typically includes:
            - "query": natural language question
            - "attributes": list of expected attributes (optional)
            - "sql": SQL query (optional, for datasets with SQL)
            - "answer": ground truth answer (optional)
            
        Note:
            Default implementation returns empty dict. Override if dataset has queries.
        """
        return {}

    def get_query_info(self, qid: str | int) -> Optional[Dict[str, Any]]:
        """Get information for a specific query.
        
        Args:
            qid: Query identifier (string or int)
            
        Returns:
            Query information dict, or None if not found
            
        Note:
            Default implementation uses load_query_dict(). Override for lazy loading.
        """
        qid_str = str(qid)
        return self.load_query_dict().get(qid_str)

    # ============ SCHEMA ACCESS (OPTIONAL) ============
    # Note: Not all datasets have schemas. Implementations should return
    # appropriate empty values if schemas are not applicable.

    def load_schema_general(self) -> List[Dict[str, Any]]:
        """Load dataset-wide schema information.
        
        Returns:
            List of schema/table definitions. Empty list if no schema.
            Schema format is flexible but commonly includes:
            - "Schema Name" or "table_name": name of the schema/table
            - "Attributes" or "columns": list of attribute definitions
            
        Note:
            Default implementation returns empty list. Override if dataset has schema.
        """
        return []

    def load_schema_query(self, qid: str | int) -> List[Dict[str, Any]]:
        """Load query-specific schema (subset relevant to a query).
        
        Args:
            qid: Query identifier
            
        Returns:
            List of schema definitions relevant to the query. Empty list if not available.
            
        Note:
            Default implementation returns empty list. Override if dataset has query-specific schemas.
        """
        return []

    # ============ HIGH-LEVEL CONVENIENCE METHODS ============
    # These provide a cleaner interface abstracting away internal data structures.
    # Implementations can override these for better performance or custom behavior.

    def get_doc_text(self, doc_id: str) -> str:
        """Get just the text content of a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document text, or empty string if not found
        """
        doc_tuple = self.get_doc(doc_id)
        return doc_tuple[0] if doc_tuple else ""

    def get_doc_metadata(self, doc_id: str, key: str, default: Any = None) -> Any:
        """Get a specific metadata field from document info.
        
        Args:
            doc_id: Document identifier
            key: Metadata key to retrieve
            default: Default value if key not found
            
        Returns:
            Metadata value, or default if not found
        """
        doc_info = self.get_doc_info(doc_id)
        if doc_info is None:
            return default
        return doc_info.get(key, default)

    def get_query_text(self, qid: str | int) -> Optional[str]:
        """Get the natural language query text.
        
        Args:
            qid: Query identifier
            
        Returns:
            Query text, or None if not found
        """
        query_info = self.get_query_info(qid)
        if query_info is None:
            return None
        return query_info.get("query")

    def get_query_attributes(self, qid: str | int) -> List[str]:
        """Get the list of attributes expected for a query.
        
        Args:
            qid: Query identifier
            
        Returns:
            List of attribute names, or empty list if not available
        """
        query_info = self.get_query_info(qid)
        if query_info is None:
            return []
        return query_info.get("attributes", [])

    def get_query_sql(self, qid: str | int) -> Optional[str]:
        """Get the SQL query (if available).
        
        Args:
            qid: Query identifier
            
        Returns:
            SQL query string, or None if not available
            
        Note:
            Only applicable for datasets with SQL annotations (e.g., Spider, Bird)
        """
        query_info = self.get_query_info(qid)
        if query_info is None:
            return None
        return query_info.get("sql")

    def get_schema_by_name(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific schema by name.
        
        Args:
            schema_name: Schema/table name
            
        Returns:
            Schema definition dict, or None if not found
        """
        schemas = self.load_schema_general()
        for schema in schemas:
            # Support both "Schema Name" and "table_name" keys
            name = schema.get("Schema Name") or schema.get("table_name")
            if name == schema_name:
                return schema
        return None

    def get_schema_attributes(self, schema_name: str) -> List[Dict[str, Any]]:
        """Get attributes for a specific schema.
        
        Args:
            schema_name: Schema/table name
            
        Returns:
            List of attribute definitions, or empty list if not found
        """
        schema = self.get_schema_by_name(schema_name)
        if schema is None:
            return []
        # Support both "Attributes" and "columns" keys
        return schema.get("Attributes") or schema.get("columns") or []

    # ============ ADVANCED DOCUMENT ACCESS (OPTIONAL) ============
    # These methods provide dict-based access for more flexible metadata handling.

    def iter_doc_dicts(self) -> Iterator[Dict[str, Any]]:
        """Iterate over documents as dictionaries (more flexible than tuples).
        
        Yields:
            Dict with at least "doc_id" and "doc_text" keys, plus metadata fields
            
        Note:
            Default implementation converts from iter_docs(). Override for native dict support.
        """
        for doc_text, doc_id, metadata_dict in self.iter_docs():
            result = {
                "doc_id": doc_id,
                "doc_text": doc_text,
            }
            # Merge metadata fields into the result
            result.update(metadata_dict)
            yield result

    def get_doc_dict(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document as a dictionary (more flexible than tuple).
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dict with document data, or None if not found
            
        Note:
            Default implementation uses get_doc_info(). Override for custom behavior.
        """
        return self.get_doc_info(doc_id)

    # ============ DATASET INTROSPECTION ============

    def has_queries(self) -> bool:
        """Check if this dataset has queries.
        
        Returns:
            True if dataset has queries, False otherwise
        """
        return self.num_queries > 0

    def has_schemas(self) -> bool:
        """Check if this dataset has schema information.
        
        Returns:
            True if dataset has schemas, False otherwise
        """
        return len(self.load_schema_general()) > 0

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the dataset.
        
        Returns:
            Dictionary with dataset statistics and metadata
        """
        return {
            "data_root": str(self.data_root),
            "num_docs": self.num_docs,
            "num_queries": self.num_queries,
            "has_schemas": self.has_schemas(),
            "loader_type": self.__class__.__name__,
        }

    # ============ HELPER METHODS ============

    def _path(self, key: str, **fmt) -> Path:
        """Return absolute path for a logical resource key.
        
        Args:
            key: Logical name of the resource (e.g., "doc_dict", "queries")
            **fmt: Format arguments for the path pattern (e.g., qid="1")
            
        Returns:
            Absolute Path object
            
        Note:
            Requires subclass to set self._filemap attribute with path patterns.
            
        Example:
            If _filemap = {"schema_query": "schema_query_{qid}.json"}
            Then _path("schema_query", qid="1") returns data_root / "schema_query_1.json"
        """
        if not hasattr(self, '_filemap'):
            logging.error(f"[{self.__class__.__name__}:_path] {self.__class__.__name__} must define _filemap attribute")
            raise AttributeError(f"{self.__class__.__name__} must define _filemap attribute")
        
        pattern = self._filemap[key]
        path = self.data_root / pattern.format(**fmt)
        return path

    @staticmethod
    def _read_json(path: Path | str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read and parse a JSON file.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            Dictionary parsed from JSON, or empty dict if file not found or invalid
            
        Note:
            This is a common helper method used by all loader implementations.
            Errors are logged but do not raise exceptions.
        """
        path = Path(path)
        
        if not path.exists():
            logging.warning(f"[{path.parent.name}:_read_json] File not found: {path}, returning empty dict")
            return {}
        
        try:
            with path.open(encoding=encoding) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"[{path.parent.name}:_read_json] Failed to parse JSON file {path}: {e}")
            return {}
        except Exception as e:
            logging.error(f"[{path.parent.name}:_read_json] Error reading file {path}: {e}")
            return {}

    # ============ PYTHON DUNDER METHODS ============

    def __iter__(self):
        """Iterate over documents (stream-friendly).
        
        Yields:
            Tuple of (doc_text, doc_id, metadata_dict)
        """
        return self.iter_docs()

    def __len__(self):
        """Return number of documents."""
        return self.num_docs

    def __repr__(self):
        """Developer-friendly representation."""
        return (
            f"<{self.__class__.__name__} "
            f"root='{self.data_root}' "
            f"docs={self.num_docs} "
            f"queries={self.num_queries}>"
        )
    
    def __str__(self):
        """User-friendly string representation."""
        return (
            f"{self.__class__.__name__}("
            f"{self.num_docs} docs, "
            f"{self.num_queries} queries)"
        )
