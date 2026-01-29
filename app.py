import os
import json
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
import document_parser

from IPython.display import display, Image, IFrame, Markdown, JSON 

import helper

# OpenAI & ChromaDB - Embedding + Vector Store
import openai
import chromadb

# Langchain 
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables from .env
_ = load_dotenv(override=True)

# Folder configuration
INPUT_DIR = Path("./input")
PROCESSED_DIR = Path("./processed")
OUTPUT_DIR = Path("./ade_outputs")

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ChromaDB configuration
CHROMA_DB_PATH = Path("./chroma_db")
COLLECTION_NAME = "ade_documents"
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize ChromaDB client (reused across files)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def process_document(doc_path: Path) -> bool:
    """
    Process a single document: parse it, extract chunks, and add to ChromaDB.
    
    Args:
        doc_path: Path to the document file to process
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {doc_path.name}")
        print(f"{'='*60}")
        
        # Validate file exists and is a PDF
        if not doc_path.exists():
            print(f"Error: File not found: {doc_path}")
            return False
        
        if doc_path.suffix.lower() != '.pdf':
            print(f"Warning: Skipping non-PDF file: {doc_path.name}")
            return False
        
        # Display document preview
        helper.print_document(doc_path)
        
        # Build output file paths
        base_name = doc_path.stem
        ade_json_path = OUTPUT_DIR / f"{base_name}_chunks.json"
        ade_md_path = OUTPUT_DIR / f"{base_name}.md"
        
        # Check if already processed (files exist and are non-empty)
        needs_parsing = False
        if not ade_md_path.exists() or not ade_json_path.exists():
            needs_parsing = True
        elif os.path.getsize(ade_md_path) == 0 or os.path.getsize(ade_json_path) == 0:
            needs_parsing = True
        
        # Parse document if needed
        if needs_parsing:
            print("Parsing document with Landing.AI ADE API...")
            document_parser.parse_document(doc_path, OUTPUT_DIR)
        else:
            print("Using existing parsed output files...")
        
        # Load parsed chunks
        if not ade_json_path.exists() or os.path.getsize(ade_json_path) == 0:
            print(f"Error: Failed to generate chunks for {doc_path.name}")
            return False
        
        with open(ade_json_path, "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)
        
        print(f"Loaded {len(loaded_chunks)} chunks from {doc_path.name}")
        
        # Show sample chunk structure
        if loaded_chunks:
            print(f"\nSample chunk structure:")
            print(json.dumps(loaded_chunks[0], indent=2)[:400] + "...")
        
        # Check for existing chunks in ChromaDB
        existing_result = collection.get(ids=[chunk["id"] for chunk in loaded_chunks])
        existing_ids = set(existing_result.get("ids", []))
        print(f"Found {len(existing_ids)} existing chunks in ChromaDB collection")
        
        # Add new chunks to ChromaDB
        print(f"Inserting new chunks into ChromaDB...")
        added_count = 0
        
        for i, chunk in enumerate(loaded_chunks):
            chunk_id = chunk["id"]
            
            # Skip if already exists
            if chunk_id in existing_ids:
                continue
            
            text = chunk.get("text", "")
            
            # Skip empty chunks
            if not text or not text.strip():
                continue
            
            # Generate embeddings
            try:
                emb = openai.embeddings.create(
                    input=text,
                    model=EMBEDDING_MODEL
                ).data[0].embedding
            except Exception as e:
                print(f"Error generating embedding for chunk {chunk_id}: {e}")
                continue
            
            # Prepare metadata
            metadata = {
                "chunk_type": chunk.get("type", "unknown"),
                "page": chunk.get("page", 0),
                "source_file": doc_path.name,
            }
            
            # Add bbox coordinates to metadata
            box = chunk.get("box")
            if box:
                metadata["bbox_left"] = float(box.get("left", 0))
                metadata["bbox_top"] = float(box.get("top", 0))
                metadata["bbox_right"] = float(box.get("right", 0))
                metadata["bbox_bottom"] = float(box.get("bottom", 0))
            
            # Store in ChromaDB
            try:
                collection.add(
                    documents=[text],
                    ids=[chunk_id],
                    metadatas=[metadata],
                    embeddings=[emb]
                )
                added_count += 1
                
                # Progress indicator
                if added_count % 20 == 0:
                    print(f"   Processed {added_count} new chunks...")
            except Exception as e:
                print(f"Error adding chunk {chunk_id} to ChromaDB: {e}")
                continue
        
        print(f"\n✓ Successfully processed {doc_path.name}")
        print(f"  - Added {added_count} new chunks to ChromaDB")
        print(f"  - Skipped {len(existing_ids)} existing chunks")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing {doc_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def move_to_processed(doc_path: Path) -> bool:
    """
    Move a successfully processed file to the processed folder.
    
    Args:
        doc_path: Path to the file to move
        
    Returns:
        bool: True if move was successful, False otherwise
    """
    try:
        processed_path = PROCESSED_DIR / doc_path.name
        
        # Handle filename conflicts by appending a number
        counter = 1
        while processed_path.exists():
            stem = doc_path.stem
            suffix = doc_path.suffix
            processed_path = PROCESSED_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
        
        shutil.move(str(doc_path), str(processed_path))
        print(f"✓ Moved {doc_path.name} to processed folder")
        return True
    except Exception as e:
        print(f"✗ Error moving {doc_path.name} to processed folder: {e}")
        return False


def process_all_files_in_input():
    """
    Process all PDF files currently in the input folder.
    """
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in input folder")
        return
    
    print(f"\nFound {len(pdf_files)} PDF file(s) in input folder")
    
    for pdf_file in pdf_files:
        success = process_document(pdf_file)
        
        if success:
            move_to_processed(pdf_file)
        else:
            print(f"⚠ Skipping move to processed folder due to processing errors")


def monitor_input_folder(poll_interval: int = 5):
    """
    Monitor the input folder for new files and process them.
    Uses polling instead of file system events for simplicity.
    
    Args:
        poll_interval: Seconds between folder checks
    """
    print(f"\n{'='*60}")
    print(f"Starting folder monitor")
    print(f"  Input folder: {INPUT_DIR.absolute()}")
    print(f"  Processed folder: {PROCESSED_DIR.absolute()}")
    print(f"  Polling interval: {poll_interval} seconds")
    print(f"{'='*60}")
    print("\nMonitoring for new files... (Press Ctrl+C to stop)\n")
    
    processed_files = set()
    
    try:
        while True:
            # Get all PDF files in input folder
            current_files = set(INPUT_DIR.glob("*.pdf"))
            
            # Find new files (not yet processed)
            new_files = current_files - processed_files
            
            if new_files:
                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found {len(new_files)} new file(s)")
                
                for pdf_file in new_files:
                    success = process_document(pdf_file)
                    
                    if success:
                        move_to_processed(pdf_file)
                        processed_files.add(pdf_file)
                    else:
                        print(f"⚠ File {pdf_file.name} will be retried on next check")
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print("\n\nStopping folder monitor...")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Process all files once and exit
        print("Running in single-pass mode...")
        process_all_files_in_input()
    else:
        # Process existing files first
        process_all_files_in_input()
        
        # Then start monitoring
        monitor_input_folder(poll_interval=5)
