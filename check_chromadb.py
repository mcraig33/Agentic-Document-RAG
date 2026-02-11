#!/usr/bin/env python3
"""
Diagnostic script to check what's in ChromaDB and test retrieval.
"""
import chromadb
from pathlib import Path

CHROMA_DB_PATH = Path("./chroma_db")
COLLECTION_NAME = "ade_documents"

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

print("="*60)
print("ChromaDB Diagnostic")
print("="*60)

# Count total chunks
total_count = collection.count()
print(f"\nTotal chunks in ChromaDB: {total_count}")

if total_count == 0:
    print("\n⚠️  WARNING: ChromaDB is empty!")
    print("   Run 'python app.py --once' to process documents first.")
    exit(1)

# Get sample chunks
print("\n" + "-"*60)
print("Sample chunks (first 5):")
print("-"*60)
samples = collection.get(limit=5)

for i, (doc, metadata) in enumerate(zip(samples['documents'], samples['metadatas']), 1):
    source_file = metadata.get('source_file', 'unknown')
    page = metadata.get('page', '?')
    chunk_type = metadata.get('chunk_type', 'unknown')
    preview = doc[:150].replace('\n', ' ')
    print(f"\n{i}. File: {source_file}")
    print(f"   Page: {page}, Type: {chunk_type}")
    print(f"   Preview: {preview}...")

# Check unique source files
print("\n" + "-"*60)
print("Unique source files in ChromaDB:")
print("-"*60)
all_data = collection.get()
source_files = set()
for metadata in all_data['metadatas']:
    source_file = metadata.get('source_file', 'unknown')
    source_files.add(source_file)

for sf in sorted(source_files):
    count = sum(1 for m in all_data['metadatas'] if m.get('source_file') == sf)
    print(f"  - {sf}: {count} chunks")

# Test a query
print("\n" + "-"*60)
print("Testing retrieval with sample queries:")
print("-"*60)

from langchain_openai import OpenAIEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

test_queries = [
    "What companies are mentioned?",
    "Apple",
    "Rivian"
]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    docs = retriever.invoke(query)
    print(f"   Retrieved {len(docs)} chunks:")
    for i, doc in enumerate(docs, 1):
        source_file = doc.metadata.get('source_file', 'unknown')
        page = doc.metadata.get('page', '?')
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"   {i}. [{source_file}, p{page}]: {preview}...")

print("\n" + "="*60)
