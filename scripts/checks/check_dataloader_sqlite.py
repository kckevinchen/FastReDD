
import json
from pathlib import Path

from core.data_loader.data_loader_sqlite import DataLoaderSQLite

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

def json_print(data):
    print(json.dumps(data, indent=2, default=str))

def check_wine_dataloader():
    # Path to wine_1 dataset
    # Assuming standard project structure: dataset/spider_sqlite/wine_1/
    data_path = Path("dataset") / "spider_sqlite" / "wine_1"
    
    print(f"Checking dataset at: {data_path}")
    
    if not data_path.exists():
        print(f"Error: Dataset path does not exist: {data_path}")
        return

    try:
        # Initialize Loader
        loader = DataLoaderSQLite(data_path=data_path, task_db_name="wine-appellations")
        print_separator("Initialization")
        print(f"Loader initialized: {loader.__class__.__name__}")
        print(f"Input DB: {loader.input_db_path}")
        print(f"GT DB Path: {loader.gt_db_path}")
        
        # 1. Basic Stats
        print_separator("Basic Statistics")
        print(f"Number of Documents: {loader.num_docs}")
        print(f"Number of Queries: {loader.num_queries}")
        
        doc_ids = loader.doc_ids
        print(f"First 5 Doc IDs: {doc_ids[:5]}")
        
        query_ids = loader.query_ids
        print(f"First 5 Query IDs: {query_ids[:5]}")

        # 2. Check Document Iteration
        print_separator("Document Iteration (First 2)")
        count = 0
        for doc_text, doc_id, meta in loader.iter_docs():
            print(f"[{count+1}] Doc ID: {doc_id}")
            print(f"    Text Preview: {doc_text[:100]}...")
            print(f"    Metadata: {meta}")
            
            # 3. Check Deep Info (Mapping + GT Data)
            # This verifies if ATTACH DATABASE works correctly
            print(f"    --- Fetching Detailed Info (get_doc_info) ---")
            full_info = loader.get_doc_info(doc_id)
            
            if full_info:
                mappings = full_info.get("mappings", [])
                print(f"    Mappings Found: {len(mappings)}")
                for m in mappings:
                    print(f"      -> Table: {m.get('table_name')}, Row ID: {m.get('row_id')}")
                
                data_records = full_info.get("data_records", [])
                if data_records:
                    print(f"    GT Data Records: {len(data_records)}")
                    print(f"      First Record:")
                    print(f"        Table: {data_records[0].get('table_name')}")
                    print(f"        Row ID: {data_records[0].get('row_id')}")
                    print(f"        Data: {data_records[0].get('data')}")
                else:
                    print("    [WARNING] No GT Data found (Mapping might be empty or GT lookup failed)")
            
            count += 1
            if count >= 2:
                break
        
        # 4. Check Query Info
        if query_ids:
            print_separator("Query Info (First Query)")
            qid = query_ids[0]
            q_info = loader.get_query_info(qid)
            print(f"Query ID: {qid}")
            json_print(q_info)

            # 5. Check Schema for this Query
            print_separator(f"Schema for Query {qid}")
            schemas = loader.load_schema_query(qid)
            json_print(schemas)
            
    except Exception as e:
        print(f"Failed to run check: {e}")

if __name__ == "__main__":
    check_wine_dataloader()

