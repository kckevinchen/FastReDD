import json
import sqlite3

import pytest

from core.data_loader.data_loader_sqlite import DataLoaderSQLite


@pytest.fixture()
def sqlite_dataset(tmp_path):
    data_root = tmp_path / "sqlite_dataset"
    data_root.mkdir()
    input_db_path = data_root / "example_task.db"
    gt_db_path = data_root / "gt_data.db"

    conn = sqlite3.connect(input_db_path)
    conn.execute(
        "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, doc_text TEXT, source_file TEXT)"
    )
    conn.execute("CREATE TABLE mapping (doc_id TEXT, table_name TEXT, row_id TEXT, match_type TEXT)")
    conn.execute(
        "CREATE TABLE queries (query_id TEXT, query TEXT, attributes TEXT, sql TEXT, difficulty TEXT)"
    )
    conn.execute("CREATE TABLE schemas (schema_name TEXT, query_id TEXT)")
    conn.execute(
        "CREATE TABLE schema_attributes (schema_name TEXT, attribute_name TEXT, description TEXT, query_id TEXT)"
    )

    conn.executemany(
        "INSERT INTO documents (doc_id, doc_text, source_file) VALUES (?, ?, ?)",
        [("doc1", "Doc text 1", "file_a.txt"), ("doc2", "Doc text 2", "file_b.txt")],
    )
    conn.executemany(
        "INSERT INTO mapping (doc_id, table_name, row_id, match_type) VALUES (?, ?, ?, ?)",
        [
            ("doc1", "items", "1", "full"),
            ("doc1", "items", "2", "full"),  # doc1 maps to two rows
            ("doc2", "items", "2", "full"),  # doc2 shares one row with doc1 (many-to-many example)
        ],
    )
    conn.executemany(
        "INSERT INTO queries (query_id, query, attributes, sql, difficulty) VALUES (?, ?, ?, ?, ?)",
        [
            ("q1", "first question", json.dumps(["name", "price"]), "SELECT * FROM items", "easy"),
            ("q2", "second question", json.dumps(["price"]), "SELECT price FROM items", "easy"),
        ],
    )
    conn.executemany(
        "INSERT INTO schemas (schema_name, query_id) VALUES (?, ?)",
        [("items", None), ("items", "q1"), ("items", "q2")],
    )
    conn.executemany(
        "INSERT INTO schema_attributes (schema_name, attribute_name, description, query_id) VALUES (?, ?, ?, ?)",
        [
            ("items", "name", "Item name", None),
            ("items", "price", "Item price", None),
            ("items", "name", "Item name", "q1"),
            ("items", "price", "Item price", "q1"),
            ("items", "price", "Item price", "q2"),
        ],
    )

    conn.commit()
    conn.close()

    gt_conn = sqlite3.connect(gt_db_path)
    gt_conn.execute(
        'CREATE TABLE "items" (row_id TEXT PRIMARY KEY, name TEXT, price REAL)'
    )
    gt_conn.executemany(
        'INSERT INTO "items" (row_id, name, price) VALUES (?, ?, ?)',
        [("1", "Widget", 9.99), ("2", "Gadget", 4.5)],
    )
    gt_conn.commit()
    gt_conn.close()

    return data_root


@pytest.fixture()
def loader(sqlite_dataset):
    loader = DataLoaderSQLite(sqlite_dataset, task_db_name="example_task.db")
    try:
        yield loader
    finally:
        if hasattr(loader, '_input_conn') and loader._input_conn:
            loader._input_conn.close()


def test_counts_and_doc_ids(loader):
    assert loader.num_docs == 2
    assert loader.doc_ids == ["doc1", "doc2"]


def test_iter_docs_metadata(loader):
    docs = {doc_id: (doc_text, metadata) for doc_text, doc_id, metadata in loader.iter_docs()}

    assert set(docs.keys()) == {"doc1", "doc2"}
    doc_text, metadata = docs["doc1"]
    assert doc_text == "Doc text 1"
    assert metadata["doc_id"] == "doc1"
    assert metadata["source_file"] == "file_a.txt"


def test_get_doc_and_missing(loader):
    doc_text, doc_id, metadata = loader.get_doc("doc2")

    assert doc_id == "doc2"
    assert doc_text == "Doc text 2"
    assert metadata["source_file"] == "file_b.txt"
    with pytest.raises(KeyError):
        loader.get_doc("unknown-id")


def test_get_doc_info_includes_gt_data(loader):
    doc_info = loader.get_doc_info("doc1")

    assert doc_info is not None
    assert doc_info["doc"] == "Doc text 1"
    assert doc_info["mappings"] == [
        {"doc_id": "doc1", "table_name": "items", "row_id": "1", "match_type": "full"},
        {"doc_id": "doc1", "table_name": "items", "row_id": "2", "match_type": "full"},
    ]
    # Multi-mapping should surface data as a list aligned with mappings order
    assert len(doc_info["data_records"]) == 2
    # Detailed records retain table/row context
    assert [rec["row_id"] for rec in doc_info["data_records"]] == ["1", "2"]
    assert [rec["table_name"] for rec in doc_info["data_records"]] == [
        "items",
        "items",
    ]
    # Check data content inside records
    assert [rec["data"]["name"] for rec in doc_info["data_records"]] == ["Widget", "Gadget"]
    assert [rec["data"]["price"] for rec in doc_info["data_records"]] == [
        pytest.approx(9.99),
        pytest.approx(4.5),
    ]

    # Single-mapping now behaves consistently (always list)
    doc2_info = loader.get_doc_info("doc2")
    assert doc2_info["mappings"] == [
        {"doc_id": "doc2", "table_name": "items", "row_id": "2", "match_type": "full"}
    ]
    assert len(doc2_info["data_records"]) == 1
    assert doc2_info["data_records"][0]["data"]["name"] == "Gadget"
    assert doc2_info["data_records"][0]["data"]["price"] == pytest.approx(4.5)

    assert loader.get_doc_info("missing") is None


def test_queries_are_loaded_and_parsed(loader):
    queries = loader.load_query_dict()

    assert loader.num_queries == 2
    assert loader.query_ids == ["q1", "q2"]
    assert queries["q1"]["attributes"] == ["name", "price"]
    q2_info = loader.get_query_info("q2")
    assert q2_info is not None
    assert q2_info["query"] == "second question"
    assert q2_info["attributes"] == ["price"]


def test_schema_loading(loader):
    general_schema = loader.load_schema_general()
    assert general_schema == [
        {
            "Schema Name": "items",
            "Attributes": [
                {"Attribute Name": "name", "Description": "Item name"},
                {"Attribute Name": "price", "Description": "Item price"},
            ],
        }
    ]

    query_schema = loader.load_schema_query("q1")
    assert query_schema == [
        {
            "Schema Name": "items",
            "Attributes": [
                {"Attribute Name": "name", "Description": "Item name"},
                {"Attribute Name": "price", "Description": "Item price"},
            ],
        }
    ]

