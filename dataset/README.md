# Dataset Format (SQLite)

## Overview

All datasets are encouraged to use SQLite format. Each dataset consists of:

1. **One Ground Truth Database** (`gt_data.db`) - Validation data (shared across all tasks)
2. **Multiple Input Databases** (`{task_name}.db`) - System input data (one per task)

**Note:** For datasets in other formats, you need to implement a custom `DataLoader` to load them.

## File Structure

```
dataset/{output_dir}/{dataset_name}/
  ├── gt_data.db              # Ground truth database (one per dataset)
  ├── {task_name_1}.db        # Input database for task 1
  ├── {task_name_2}.db        # Input database for task 2
  └── ...                     # More task databases
```

## Ground Truth Database (`gt_data.db`)

Contains individual data tables. Each table represents a ground truth data structure.

### Table Structure

Each table has:
- `row_id` (TEXT, PRIMARY KEY) - Sequential row ID (format: `0`, `1`, `2`, ...)
- Additional columns based on dataset attributes

### Example

For a dataset with `wine` and `appellations` tables:

**Table: `wine`**
| row_id | Name | Vintage | ... |
|--------|------|---------|-----|
| 0 | ... | ... | ... |
| 1 | ... | ... | ... |

**Table: `appellations`**
| row_id | Appellation | County | State | ... |
|--------|-------------|--------|-------|-----|
| 0 | ... | ... | ... | ... |
| 1 | ... | ... | ... | ... |

## Input Database (`{task_name}.db`)

### Tables

#### 1. `queries`
Query definitions.

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | TEXT | Primary key |
| `query` | TEXT | Natural language query |
| `attributes` | TEXT | JSON array of attributes |
| `sql` | TEXT | SQL query |
| `difficulty` | TEXT | Query difficulty level |

#### 2. `documents`
Document chunks.

| Column | Type | Description |
|--------|------|-------------|
| `doc_id` | TEXT | Primary key (format: `{count}-0`) |
| `doc_text` | TEXT | Document content (Optional if `source_file` is provided) |
| `source_file` | TEXT | Source file path (Relative to dataset root, used if `doc_text` is empty) |
| `parent_doc_id` | TEXT | Original document ID |
| `chunk_index` | INTEGER | Chunk index (default: 0) |

#### 3. `mapping`
Links ground truth table rows to documents.

| Column | Type | Description |
|--------|------|-------------|
| `table_name` | TEXT | Ground truth table name |
| `row_id` | TEXT | Row ID in GT table |
| `doc_id` | TEXT | Document ID (format: `{count}-0`) |
| `match_type` | TEXT | Match type (default: 'full') |

**Primary Key:** `(table_name, row_id, doc_id)`

**Indexes:**
- `idx_mapping_table_row` on `(table_name, row_id)`
- `idx_mapping_doc` on `(doc_id)`

#### 4. `schemas`
Schema definitions.

| Column | Type | Description |
|--------|------|-------------|
| `schema_name` | TEXT | Schema/table name |
| `query_id` | TEXT | Query ID (NULL for general schema) |

**Primary Key:** `(schema_name, query_id)`

#### 5. `schema_attributes`
Schema attribute definitions.

| Column | Type | Description |
|--------|------|-------------|
| `schema_name` | TEXT | Schema/table name |
| `query_id` | TEXT | Query ID (NULL for general schema) |
| `attribute_name` | TEXT | Attribute name |
| `description` | TEXT | Attribute description |

**Primary Key:** `(schema_name, query_id, attribute_name)`

## Tools

Convert JSON datasets to SQLite format:

```bash
python dataset/convert_spider_to_sqlite.py <dataset_name> <task_name>
```

Batch convert all datasets:

```bash
bash dataset/convert_spider_to_sqlite.sh
```
