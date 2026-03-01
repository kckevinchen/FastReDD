"""
SQLite Data Loader

This module implements the DataLoaderSQLite class for reading datasets in the
standard SQLite format defined in dataset/README.md.

The format consists of:
1. gt_data.db: Ground truth data tables
2. {task_name}.db: Input data (documents, queries, schemas, mappings)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator

from .data_loader_basic import DataLoaderBase


class DataLoaderSQLite(DataLoaderBase):
    """
    Data loader for SQLite-based datasets.
    
    Loads data from:
    - Input Database ({task_name}.db): documents, queries, schemas, mappings
    - Ground Truth Database (gt_data.db): Actual data rows linked via mappings
    """
    
    def __init__(
        self,
        data_path: str | Path | None = None,
        data_root: str | Path | None = None,
        task_db_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize SQLite data loader.
        
        Args:
            data_path: Path to the data - .db file path (e.g., "bike_1/default_task.db")
                       If `data_path` is given, you do not need to provide `data_root` or `task_db_name`.
            data_root: Path to dataset directory (containing .db files).
                       Use this together with `task_db_name`.
            task_db_name: Specific name of the task database.
                          If not provided, finds the first non-gt_data .db file.
            config: Optional configuration dict with:
                - data_main: Base data directory (default: "dataset/")
            **kwargs: Additional arguments (ignored)
        """
        self.config = config or {}
        resolved_root: Path
        input_db_path: Optional[Path] = None

        # Enforce a single entrypoint
        if data_path is not None and data_root is not None:
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"Both data_path and data_root provided. Provide either "
                         f"`data_root` (with optional `task_db_name`) or `data_path`, not both.")
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] Provide either "
                f"`data_root` (with optional `task_db_name`) or `data_path`, not both."
            )
        if data_path is None and data_root is None:
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"Neither data_path nor data_root provided. Provide `data_root` or `data_path`.")
            raise ValueError(
                f"[{self.__class__.__name__}:__init__] Provide `data_root` or `data_path`."
            )
            
        if data_path is not None:
            data_path = Path(data_path).expanduser()
            if not data_path.is_absolute():
                # First try relative to current directory
                candidate_path = (Path.cwd() / data_path).resolve()
                if candidate_path.exists():
                    data_path = candidate_path
                else:
                    # If not found, try resolving using data_main from config
                    data_path = self._resolve_data_path(data_path)
            
            if not data_path.exists():
                logging.error(f"[{self.__class__.__name__}:__init__] "
                             f"data_path not found: {data_path}")
                raise FileNotFoundError(
                    f"[{self.__class__.__name__}:__init__] data_path not found: {data_path}"
                )

            if data_path.is_file():
                # Direct DB file path provided
                resolved_root = data_path.parent
                input_db_path = data_path
            else:
                # Directory provided; resolve like data_root
                resolved_root = self._resolve_dataset_path(data_path)
        else:
            # Resolve data_root path using data_main from config
            resolved_root = self._resolve_dataset_path(data_root)
        
        super().__init__(resolved_root)
        
        self.gt_db_path = self.data_root / "gt_data.db"
        self.input_db_path = input_db_path
        
        # Resolve input database path
        if self.input_db_path is None:
            if task_db_name:
                self.input_db_path = self.data_root / task_db_name
                if not self.input_db_path.exists():
                    # Try adding .db extension
                    self.input_db_path = self.data_root / f"{task_db_name}.db"
            else:
                # Auto-detect input database
                db_files = list(self.data_root.glob("*.db"))
                for db_file in db_files:
                    if db_file.name != "gt_data.db":
                        self.input_db_path = db_file
                        break
        
        if not self.input_db_path or not self.input_db_path.exists():
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"Input database not found in {self.data_root}")
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] Input database not found in {self.data_root}"
            )

        logging.info(f"[{self.__class__.__name__}:__init__] "
                    f"Loading dataset from: {self.input_db_path.name}")
        
        # Open connections
        self._input_conn = sqlite3.connect(str(self.input_db_path), check_same_thread=False)
        self._input_conn.row_factory = sqlite3.Row
        
        # Attach Ground Truth DB if it exists
        if self.gt_db_path.exists():
            try:
                # Attach gt_data.db as schema 'gt_db'
                self._input_conn.execute(f"ATTACH DATABASE '{str(self.gt_db_path)}' AS gt_db")
                logging.info(f"[{self.__class__.__name__}:__init__] "
                            f"Attached ground truth DB: {self.gt_db_path.name}")
            except sqlite3.Error as e:
                 logging.error(f"[{self.__class__.__name__}:__init__] "
                             f"Failed to attach ground truth DB: {e}")
        else:
            logging.error(f"[{self.__class__.__name__}:__init__] "
                         f"Ground truth DB not found at {self.gt_db_path}")

        # Cache counts
        self._num_docs = self._count_rows("documents")
        self._num_queries = self._count_rows("queries")

    def _resolve_data_path(self, data_path: str | Path) -> Path:
        """
        Resolve data path (file or directory) by trying multiple common locations.
        
        This method is used when data_path is provided but doesn't exist at the
        initial location. It tries to find the path relative to data_main from config.
        
        Args:
            data_path: Data path (can be relative, file or directory)
        
        Returns:
            Resolved absolute path (may not exist)
        """
        data_path = Path(data_path)
        
        # If absolute path, return as-is
        if data_path.is_absolute():
            return data_path
        
        # Get base directories from config
        data_main = Path(self.config.get("data_main", "dataset/"))
        # Ensure data_main is resolved to absolute path
        if not data_main.is_absolute():
            data_main = (Path.cwd() / data_main).resolve()
        
        # Try different base paths
        # Note: data_main might already include subfolder (e.g., "dataset/spider_sqlite/")
        # or just be the base directory (e.g., "dataset/")
        known_subfolders = ["spider_sqlite", "quest_sqlite", "spider_update", "spider", "bird", "galois"]
        
        if data_main.name in known_subfolders:
            # data_main already contains a known subfolder (e.g., "dataset/spider_sqlite/")
            # Just try data_main / data_path
            search_paths = [
                data_main / data_path,
            ]
        else:
            # data_main is likely "dataset/", try data_main first, then try adding subfolders
            search_paths = [
                data_main / data_path,  # First try directly under data_main
                data_main / "spider_sqlite" / data_path,
                data_main / "quest_sqlite" / data_path,
                data_main / "spider_update" / data_path,
                data_main / "spider" / data_path,
                data_main / "bird" / data_path,
                data_main / "galois" / data_path,
            ]
        
        # Try each path
        for path in search_paths:
            resolved_path = path.resolve()
            if resolved_path.exists():
                return resolved_path
        
        # If none found, return the most likely path (data_main / data_path)
        return (data_main / data_path).resolve()
    
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
                resolved_path = path
                break
        
        if resolved_path:
            return resolved_path
        
        # If none found, return the resolved path anyway (let base class handle the error)
        # But try data_main / dataset_path as the most likely option
        final_path = (data_main / dataset_path).resolve()
        return final_path

    def __del__(self):
        """Close database connections."""
        if hasattr(self, '_input_conn') and self._input_conn:
            self._input_conn.close()

    def _count_rows(self, table_name: str) -> int:
        """Count rows in a table in the input DB."""
        try:
            cursor = self._input_conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        except sqlite3.OperationalError:
            return 0

    # ============ CORE DOCUMENT ACCESS ============

    @property
    def num_docs(self) -> int:
        return self._num_docs

    @property
    def doc_ids(self) -> List[str]:
        cursor = self._input_conn.cursor()
        cursor.execute("SELECT doc_id FROM documents ORDER BY rowid")
        return [row[0] for row in cursor.fetchall()]

    def _resolve_doc_text(self, doc_text: Optional[str], metadata: Dict[str, Any]) -> str:
        """
        Helper to resolve document text.
        If doc_text is empty and 'source_file' is in metadata, load from file.
        """
        if doc_text:
            return doc_text
            
        source_file = metadata.get("source_file")
        if source_file:
            try:
                # Try to resolve path relative to data_root
                file_path = self.data_root / source_file
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}] Failed to read source file {source_file}: {e}")
        
        logging.warning(f"[{self.__class__.__name__}] Failed to resolve doc_text for doc_id: "
                       f"{metadata.get('doc_id')} from source file: {source_file}")
        return ""

    def iter_docs(self, batch_size: int = 1000) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
        """
        Iterate over documents in batches to control memory usage.
        """
        cursor = self._input_conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY rowid")
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
                
            for row in rows:
                doc_data = dict(row)
                doc_id = doc_data.pop("doc_id")
                doc_text = doc_data.pop("doc_text", "")
                
                # Get basic metadata
                metadata = doc_data.copy()
                metadata["doc_id"] = doc_id
                
                # Resolve text from file if needed
                doc_text = self._resolve_doc_text(doc_text, metadata)

                yield (doc_text, doc_id, metadata)

    def get_doc(self, doc_id: str) -> Tuple[str, str, Dict[str, Any]]:
        cursor = self._input_conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        
        if not row:
            logging.error(f"[{self.__class__.__name__}:get_doc] Document not found: {doc_id}")
            raise KeyError(f"[{self.__class__.__name__}:get_doc] Document not found: {doc_id}")
            
        doc_data = dict(row)
        doc_text = doc_data.pop("doc_text", "")
        
        metadata = doc_data.copy()
        metadata["doc_id"] = doc_id
        
        # Resolve text from file if needed
        doc_text = self._resolve_doc_text(doc_text, metadata)
            
        return (doc_text, doc_id, metadata)

    def get_doc_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full document info, including structured data from ground truth DB.
        """
        try:
            doc_text, _, metadata = self.get_doc(doc_id)
        except KeyError:
            return None

        mappings = self._get_mappings(doc_id)
        
        # Base info
        doc_info = metadata.copy()
        doc_info["doc"] = doc_text
        
        # Always set mappings key for consistency
        doc_info["mappings"] = mappings
        
        # Fetch structured data from GT DB if linked
        data_records = []
        if mappings:
            for mapping in mappings:
                table_name = mapping.get("table_name")
                row_id = mapping.get("row_id")
                
                if table_name and row_id:
                    try:
                        # Need to handle table names with spaces/special chars safely
                        safe_table = table_name.replace('"', '""')
                        
                        # Query attached GT database directly via input connection
                        # Use gt_db. prefix to access the attached database
                        cursor = self._input_conn.cursor()
                        cursor.execute(f'SELECT * FROM gt_db."{safe_table}" WHERE row_id = ?', (row_id,))
                        gt_row = cursor.fetchone()
                        
                        if gt_row:
                            # Convert row to dict, excluding row_id
                            data = dict(gt_row)
                            if "row_id" in data:
                                del data["row_id"]
                            data_records.append({
                                "table_name": table_name,
                                "row_id": row_id,
                                "data": data
                            })
                    except sqlite3.OperationalError as e:
                        logging.error(f"[{self.__class__.__name__}:get_doc_info] "
                                     f"Error fetching GT data for {table_name}:{row_id} - {e}")
            
        doc_info["data_records"] = data_records
        
        return doc_info
        
    def _get_mappings(self, doc_id: str) -> List[Dict[str, str]]:
        """Helper to fetch all mapping info for a document."""
        try:
            cursor = self._input_conn.cursor()
            cursor.execute(
                """
                SELECT doc_id, table_name, row_id, match_type 
                FROM mapping 
                WHERE doc_id = ? 
                """, 
                (doc_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.OperationalError:
            pass
        return []

    # ============ QUERY ACCESS ============

    @property
    def num_queries(self) -> int:
        return self._num_queries

    @property
    def query_ids(self) -> List[str]:
        try:
            cursor = self._input_conn.cursor()
            cursor.execute("SELECT query_id FROM queries ORDER BY rowid")
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []

    def load_query_dict(self) -> Dict[str, Any]:
        queries = {}
        try:
            cursor = self._input_conn.cursor()
            cursor.execute("SELECT * FROM queries ORDER BY rowid")
            for row in cursor:
                q_data = dict(row)
                qid = q_data.pop("query_id")
                self._parse_json_field(q_data, "attributes")
                queries[qid] = q_data
        except sqlite3.OperationalError:
            pass
        return queries

    def get_query_info(self, qid: str | int) -> Optional[Dict[str, Any]]:
        qid_str = str(qid)
        try:
            cursor = self._input_conn.cursor()
            cursor.execute("SELECT * FROM queries WHERE query_id = ?", (qid_str,))
            row = cursor.fetchone()
            if row:
                q_data = dict(row)
                self._parse_json_field(q_data, "attributes")
                return q_data
        except sqlite3.OperationalError:
            pass
        return None

    # ============ SCHEMA ACCESS ============

    def load_schema_general(self) -> List[Dict[str, Any]]:
        return self._load_schemas(query_id=None)

    def load_schema_query(self, qid: str | int) -> List[Dict[str, Any]]:
        return self._load_schemas(query_id=str(qid))

    def _load_schemas(self, query_id: Optional[str]) -> List[Dict[str, Any]]:
        schemas = []
        try:
            cursor = self._input_conn.cursor()
            
            # Select schemas
            if query_id is None:
                query = "SELECT schema_name FROM schemas WHERE query_id IS NULL ORDER BY rowid"
                params = ()
            else:
                query = "SELECT schema_name FROM schemas WHERE query_id = ? ORDER BY rowid"
                params = (query_id,)
                
            cursor.execute(query, params)
            schema_names = [row[0] for row in cursor.fetchall()]
            
            for schema_name in schema_names:
                # Get attributes
                if query_id is None:
                    attr_query = """
                        SELECT attribute_name, description 
                        FROM schema_attributes 
                        WHERE schema_name = ? AND query_id IS NULL
                        ORDER BY rowid
                    """
                    attr_params = (schema_name,)
                else:
                    attr_query = """
                        SELECT attribute_name, description 
                        FROM schema_attributes 
                        WHERE schema_name = ? AND query_id = ?
                        ORDER BY rowid
                    """
                    attr_params = (schema_name, query_id)
                
                cursor.execute(attr_query, attr_params)
                
                attributes = []
                for attr_row in cursor.fetchall():
                    attributes.append({
                        "Attribute Name": attr_row[0],
                        "Description": attr_row[1]
                    })
                
                schemas.append({
                    "Schema Name": schema_name,
                    "Attributes": attributes
                })
                
        except sqlite3.OperationalError:
            pass
            
        return schemas

    def _parse_json_field(self, data: Dict[str, Any], field: str):
        """Helper to parse JSON string fields in place."""
        if field in data and data[field]:
            try:
                data[field] = json.loads(data[field])
            except (json.JSONDecodeError, TypeError):
                pass

