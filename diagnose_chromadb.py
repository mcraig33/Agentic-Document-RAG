#!/usr/bin/env python3
"""Quick diagnostic to check ChromaDB contents"""
import chromadb

client = chromadb.PersistentClient(path='./chroma_db')
coll = client.get_collection('ade_documents')

print(f"Total chunks in ChromaDB: {coll.count()}\n")

# Get all metadata to see source files
all_data = coll.get()
source_files = {}
for metadata in all_data['metadatas']:
    source_file = metadata.get('source_file', 'unknown')
    source_files[source_file] = source_files.get(source_file, 0) + 1

print("Chunks by source file:")
for k, v in sorted(source_files.items()):
    print(f"  {k}: {v} chunks")

# Sample a few chunks from each file
print("\n" + "="*60)
print("Sample chunks from each file:")
print("="*60)

for source_file in sorted(source_files.keys()):
    print(f"\n{source_file}:")
    # Get chunks for this file
    results = coll.get(
        where={"source_file": source_file},
        limit=3
    )
    for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas']), 1):
        page = metadata.get('page', '?')
        preview = doc[:100].replace('\n', ' ')
        print(f"  {i}. [Page {page}]: {preview}...")
