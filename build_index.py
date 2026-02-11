"""
MITRE ATT&CK Vector Store Builder
==================================
This script builds a FAISS vector store from MITRE ATT&CK data.
Run this ONCE locally to generate the vector store for production use.

Requirements:
- torch
- sentence-transformers
- langchain
- langchain-community
- langchain-huggingface
- faiss-cpu

Usage:
    python build_index.py

Output:
    vector_store/ directory with FAISS index files
"""

import json
import os
from pathlib import Path

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def main():
    print("=" * 60)
    print("MITRE ATT&CK Vector Store Builder")
    print("=" * 60)
    
    # Step 1: Load MITRE ATT&CK data
    print("\n[1/4] Loading MITRE ATT&CK data...")
    data_path = Path("data/mitre_clean_rag_ready.json")
    
    if not data_path.exists():
        print(f"❌ Error: {data_path} not found!")
        return
    
    with open(data_path, "r", encoding="utf-8") as f:
        mitre_data = json.load(f)
    
    print(f"✅ Loaded {len(mitre_data)} MITRE techniques")
    
    # Step 2: Create LangChain Documents
    print("\n[2/4] Creating LangChain documents...")
    documents = []
    
    for technique in mitre_data:
        # Combine technique info into content
        content = f"""Technique ID: {technique['technique_id']}
Name: {technique['name']}
Description: {technique['description']}"""
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "technique_id": technique["technique_id"],
                "name": technique["name"],
                "description": technique["description"],
                "mitigations": technique.get("mitigations", [])
            }
        )
        documents.append(doc)
    
    print(f"✅ Created {len(documents)} documents")
    
    # Step 3: Initialize embeddings model (local)
    print("\n[3/4] Initializing embedding model...")
    print("⏳ Downloading sentence-transformers/all-MiniLM-L6-v2...")
    print("   (This may take a few minutes on first run)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("✅ Embedding model loaded")
    
    # Step 4: Build FAISS vector store
    print("\n[4/4] Building FAISS vector store...")
    print("⏳ Generating embeddings for all documents...")
    print(f"   Processing {len(documents)} documents...")
    
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save to disk
    output_dir = "vector_store"
    vector_store.save_local(output_dir)
    
    print(f"✅ Vector store saved to: {output_dir}/")
    
    # Verify files were created
    index_file = Path(output_dir) / "index.faiss"
    pkl_file = Path(output_dir) / "index.pkl"
    
    if index_file.exists() and pkl_file.exists():
        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"\nGenerated files:")
        print(f"  - {index_file} ({index_file.stat().st_size / 1024:.1f} KB)")
        print(f"  - {pkl_file} ({pkl_file.stat().st_size / 1024:.1f} KB)")
        print("\nNext steps:")
        print("  1. Commit the vector_store/ directory to your repo")
        print("  2. Deploy to Streamlit Cloud")
        print("  3. Run: streamlit run streamlit_app.py")
        print()
    else:
        print("\n❌ Error: Vector store files not created!")

if __name__ == "__main__":
    main()
