"""
Extract all documents from doc_dict.json to a single txt file.

This script reads doc_dict.json and writes all document texts to a txt file.
"""

import json
from pathlib import Path


def extract_docs_to_txt():
    """Extract all documents from doc_dict.json and save to txt file."""
    # Get paths
    script_dir = Path(__file__).parent
    doc_dict_path = script_dir / "default" / "doc_dict.json"
    output_path = script_dir / "all_documents.txt"
    
    # Load doc_dict
    print(f"[extract_docs_to_txt] Loading doc_dict from: {doc_dict_path}")
    with open(doc_dict_path, 'r', encoding='utf-8') as f:
        doc_dict = json.load(f)
    
    # Extract all documents
    print(f"[extract_docs_to_txt] Extracting {len(doc_dict)} documents...")
    documents = []
    for key in sorted(doc_dict.keys(), key=int):  # Sort numerically
        doc_content = doc_dict[key][0]  # First element is the document text
        documents.append(doc_content)
    
    # Write to txt file
    print(f"[extract_docs_to_txt] Writing to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(documents):
            f.write(f"=== Document {i} ===\n")
            f.write(doc)
            f.write("\n\n")
    
    print(f"[extract_docs_to_txt] Successfully extracted {len(documents)} documents to {output_path}")


if __name__ == "__main__":
    extract_docs_to_txt()

