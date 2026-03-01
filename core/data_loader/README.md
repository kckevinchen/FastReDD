# Data Loader Module

This module provides a unified interface for loading and processing datasets. We recommend using the **factory function** `create_data_loader` rather than instantiating specific loader classes directly.

## Quick Start

```python
from core.data_loader import create_data_loader

# Create a loader (defaults to standard SQLite format)
# The factory automatically resolves paths and sets up the correct loader
loader = create_data_loader(
    'spider_sqlite/college_2', 
    loader_config={'data_main': 'dataset/'}
)

# Access documents
print(f"Number of documents: {loader.num_docs}")
doc_text, doc_id, metadata = loader.get_doc('0')
print(f"Table name: {metadata.get('table_name')}")

# Access queries
if loader.has_queries():
    query_info = loader.get_query_info('Q1')
    print(f"Query: {query_info.get('query')}")

# Access schemas
schemas = loader.load_schema_general()
```

## Loader Types

The `create_data_loader` function supports different loader types via the `loader_type` argument:

- **`"standard"` (Default)**: Uses `DataLoaderSQLite`. Best for modern SQLite-based datasets (`gt_data.db` + `{task_name}.db`).
- **`"sqlite"`**: Explicitly requests the SQLite loader.

```python
# Explicitly requesting a specific loader type
loader = create_data_loader('my_dataset', loader_type="sqlite")
```

## Supported Formats

### 1. SQLite (Standard)
This is the recommended format for all new datasets.
- **gt_data.db**: Contains ground truth tables
- **{task_name}.db**: Contains documents, queries, and schema definitions

See [DATA_FORMAT.md](DATA_FORMAT.md) for the detailed specification.

## API Reference

The loaders return an object adhering to the `DataLoaderBase` interface.

**Key Methods:**
- `get_doc(doc_id)`: Get a document tuple `(text, id, metadata)`
- `iter_docs()`: Iterate over all documents
- `get_doc_info(doc_id)`: Get detailed info including Ground Truth data.
  - Returns a dict with:
    - `doc`: Document text
    - `mappings`: List of mappings `{table_name, row_id}`
    - `data_records`: Detailed records including table context and data dict
- `get_query_info(qid)`: Get query information dict
- `load_schema_general()`: Load dataset-wide schemas
- `load_schema_query(qid)`: Load schemas relevant to a specific query

**Key Properties:**
- `num_docs`: Total number of documents
- `doc_ids`: List of all document IDs
- `num_queries`: Total number of queries

### Handling External Files

The `DataLoaderSQLite` supports loading document text from external files to reduce database size.

- If `doc_text` in the `documents` table is empty or NULL:
- The loader checks the `source_file` field in metadata.
- It attempts to read the text from the file path relative to the dataset root.
- This process is automatic and transparent when calling `get_doc()` or `iter_docs()`.

See `data_loader_basic.py` for the complete interface definition.
