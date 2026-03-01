"""Test that converted SQLite data matches original Spider DRC data.

Usage Examples:
    python scripts/checks/check_sqlite_conversion.py wine_1 wine-appellations
    python scripts/checks/check_sqlite_conversion.py apartment_rentals default_task
    python scripts/checks/check_sqlite_conversion.py bike_1 default_task
    python scripts/checks/check_sqlite_conversion.py college_2 course-teaches-instructor
    python scripts/checks/check_sqlite_conversion.py flight_4 routes-airports-airlines
    python scripts/checks/check_sqlite_conversion.py soccer_1 default_task
"""
import sqlite3
import json
import os
import argparse
import math

parser = argparse.ArgumentParser(description="Test SQLite conversion accuracy")
parser.add_argument("dataset_name", nargs="?", default="wine_1", help="Dataset name")
parser.add_argument("task_name", nargs="?", default="wine-appellations", help="Task name")
args = parser.parse_args()

src_path = f"dataset/spider_drc/{args.dataset_name}/{args.task_name}"
dst_path = f"dataset/spider_sqlite/{args.dataset_name}"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def test_queries():
    """Test queries match."""
    print("Testing queries...")
    queries_json = load_json(os.path.join(src_path, "queries.json"))
    conn = sqlite3.connect(os.path.join(dst_path, f"{args.task_name}.db"))
    c = conn.cursor()
    c.execute("SELECT query_id, query, attributes, sql, difficulty FROM queries")
    db_queries = {row[0]: {"query": row[1], "attributes": json.loads(row[2]) if row[2] else [], 
                          "sql": row[3], "difficulty": row[4]} for row in c.fetchall()}
    conn.close()
    
    errors = []
    for qid, q_json in queries_json.items():
        if qid not in db_queries:
            errors.append(f"Query {qid} missing in DB")
            continue
        q_db = db_queries[qid]
        if q_json.get("query") != q_db["query"]:
            errors.append(f"Query {qid} text mismatch")
        if q_json.get("attributes", []) != q_db["attributes"]:
            errors.append(f"Query {qid} attributes mismatch")
        if q_json.get("sql", "") != q_db["sql"]:
            errors.append(f"Query {qid} sql mismatch")
    
    if errors:
        print(f"  ✗ {len(errors)} errors")
        for e in errors[:5]:
            print(f"    - {e}")
        return False
    print(f"  ✓ {len(queries_json)} queries match")
    return True

def test_documents():
    """Test documents match."""
    print("Testing documents...")
    doc_dict = load_json(os.path.join(src_path, "doc_dict.json"))
    conn = sqlite3.connect(os.path.join(dst_path, f"{args.task_name}.db"))
    c = conn.cursor()
    c.execute("SELECT doc_id, doc_text, source_file, parent_doc_id FROM documents")
    db_docs = {row[0]: {"doc_text": row[1], "source_file": row[2], "parent_doc_id": row[3]} 
               for row in c.fetchall()}
    conn.close()
    
    errors = []
    for count, (key, (doc_text, source_file, actual_doc_id)) in enumerate(doc_dict.items()):
        doc_id = f"{count}-0"
        if doc_id not in db_docs:
            errors.append(f"Document {doc_id} missing")
            continue
        db_doc = db_docs[doc_id]
        if doc_text != db_doc["doc_text"]:
            errors.append(f"Document {doc_id} text mismatch")
        if source_file != db_doc["source_file"]:
            errors.append(f"Document {doc_id} source_file mismatch")
        if actual_doc_id != db_doc["parent_doc_id"]:
            errors.append(f"Document {doc_id} parent_doc_id mismatch")
    
    if errors:
        print(f"  ✗ {len(errors)} errors")
        for e in errors[:5]:
            print(f"    - {e}")
        return False
    print(f"  ✓ {len(doc_dict)} documents match")
    return True

def is_float_equal(v1, v2, tol=1e-5):
    try:
        f1 = float(v1)
        f2 = float(v2)
        return math.isclose(f1, f2, rel_tol=tol, abs_tol=tol)
    except (ValueError, TypeError):
        return False

def test_gt_data():
    """Test GT data matches doc_info."""
    print("Testing GT data...")
    doc_info = load_json(os.path.join(src_path, "doc_info.json"))
    conn = sqlite3.connect(os.path.join(dst_path, "gt_data.db"))
    c = conn.cursor()
    
    # Get all tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in c.fetchall()]
    
    errors = []
    table_data = {}
    for key, info in doc_info.items():
        if not (table_name := info.get("fn")) or not (data := info.get("data")):
            continue
        if table_name not in table_data:
            table_data[table_name] = []
        table_data[table_name].append((key, data))
    
    for table_name, rows in table_data.items():
        safe_name = table_name.replace("-", "_").replace(" ", "_")
        if safe_name not in tables:
            errors.append(f"Table {table_name} missing")
            continue
        
        # Get column names (excluding row_id)
        c.execute(f'PRAGMA table_info("{safe_name}")')
        all_columns = [row[1] for row in c.fetchall()]
        columns = [col for col in all_columns if col != "row_id"]
        
        # Select with explicit column names to ensure correct order
        col_list = ", ".join([f'"{col}"' for col in columns])
        c.execute(f'SELECT row_id, {col_list} FROM "{safe_name}"')
        db_rows = {}
        for row in c.fetchall():
            row_id = row[0]
            db_rows[row_id] = {col: val for col, val in zip(columns, row[1:])}
        
        for row_idx, (key, data) in enumerate(rows):
            row_id = str(row_idx)
            if row_id not in db_rows:
                errors.append(f"Table {table_name} row {row_id} missing")
                continue
            
            db_row = db_rows[row_id]
            for attr, val in data.items():
                if attr not in db_row:
                    errors.append(f"Table {table_name} row {row_id} missing attr {attr}")
                    continue
                
                # Check for equality: try float first, then string
                val_str = str(val)
                db_val_str = str(db_row[attr])
                
                if val_str != db_val_str:
                    # If strings don't match, check if they are equal as floats
                    if not is_float_equal(val, db_row[attr]):
                        errors.append(f"Table {table_name} row {row_id} attr {attr} mismatch: {val} != {db_row[attr]}")
    
    conn.close()
    
    if errors:
        print(f"  ✗ {len(errors)} errors")
        for e in errors[:5]:
            print(f"    - {e}")
        return False
    print(f"  ✓ GT data matches")
    return True

def test_mapping():
    """Test mapping correctness."""
    print("Testing mapping...")
    doc_dict = load_json(os.path.join(src_path, "doc_dict.json"))
    doc_info = load_json(os.path.join(src_path, "doc_info.json"))
    conn = sqlite3.connect(os.path.join(dst_path, f"{args.task_name}.db"))
    c = conn.cursor()
    
    doc_key_to_count = {key: i for i, key in enumerate(doc_dict.keys())}
    c.execute("SELECT table_name, row_id, doc_id FROM mapping")
    mappings = c.fetchall()
    conn.close()
    
    errors = []
    table_data = {}
    for key, info in doc_info.items():
        if not (table_name := info.get("fn")) or not (data := info.get("data")):
            continue
        if table_name not in table_data:
            table_data[table_name] = []
        table_data[table_name].append((key, data))
    
    for table_name, rows in table_data.items():
        for row_idx, (key, data) in enumerate(rows):
            expected_doc_id = f"{doc_key_to_count[key]}-0"
            found = False
            for m_table, m_row, m_doc in mappings:
                if m_table == table_name and m_row == str(row_idx) and m_doc == expected_doc_id:
                    found = True
                    break
            if not found:
                errors.append(f"Mapping missing: {table_name} row {row_idx} -> doc {expected_doc_id}")
    
    if errors:
        print(f"  ✗ {len(errors)} errors")
        for e in errors[:5]:
            print(f"    - {e}")
        return False
    print(f"  ✓ {len(mappings)} mappings correct")
    return True

if __name__ == "__main__":
    print(f"Testing {args.dataset_name}/{args.task_name}")
    print("=" * 60)
    
    results = []
    results.append(test_queries())
    results.append(test_documents())
    results.append(test_gt_data())
    results.append(test_mapping())
    
    print("=" * 60)
    if all(results):
        print("✓ All tests passed!")
        exit(0)
    else:
        print("✗ Some tests failed")
        exit(1)

