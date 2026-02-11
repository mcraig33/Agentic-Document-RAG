import os
import json
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
import document_parser

import helper

# OpenAI & ChromaDB - Embedding + Vector Store
import openai
import chromadb

# Langchain 
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to langchain_community if langchain_chroma not available
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

# LLM configuration
LLM_MODEL = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo", etc.

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
        
        # Display document preview (works in both notebook and CLI)
        helper.print_document(doc_path, show_in_notebook=False)
        
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


def setup_retrieval_chain():
    """
    Set up LangChain retrieval chain for querying ChromaDB.
    
    Returns:
        Retrieval chain ready for querying, or None if setup fails
    """
    try:
        # Create LangChain vector store from ChromaDB collection
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings
        )
        
        # Create the retriever with more chunks for better coverage
        # 10-K documents often have company info spread across multiple chunks
        # For comparison queries, we need more chunks to ensure both companies are represented
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})  # Retrieve top 15 chunks
        
        # Create the LLM
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        
        # Create a prompt template for the RAG chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context from SEC 10-K filings and other financial documents.

The context includes excerpts from one or more documents. When answering:

1. For questions about companies: Look for company names, legal entity names, registrant names, and trading symbols (e.g., "Apple Inc.", "Registrant", "AAPL", "Rivian", "RIVN"). Company information is typically found in the cover page, business description, or throughout the document.

2. For questions about document types: Look for phrases like "FORM 10-K", "Annual Report", filing dates, fiscal year information, and SEC identifiers.

3. Extract information directly from the text - don't say information isn't available if it appears in the context.

4. If multiple companies or documents are mentioned in the context, list all of them.

5. Be specific and cite relevant details from the context.

Use ALL the provided context to answer comprehensively."""),
            ("human", "Context from documents:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
        
        # Create the retrieval chain using LangChain's standard RAG pattern
        # Since create_retrieval_chain API may vary, use a more standard approach
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        def format_docs(docs):
            """Format retrieved documents into a context string."""
            # Just return the content - no source metadata needed
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Build the RAG chain: retrieve -> format -> prompt -> LLM -> parse
        retrieval_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # If the chain supports custom prompts, we can set it after creation
        # Otherwise, the default prompt will be used
        
        print("✓ Retrieval chain set up successfully")
        return retrieval_chain, vectorstore, retriever
        
    except Exception as e:
        print(f"✗ Error setting up retrieval chain: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def query_documents(question: str, retrieval_chain=None, retriever=None, verbose=False):
    """
    Query the document database using RAG.
    
    Args:
        question: The question to ask
        retrieval_chain: Optional pre-initialized retrieval chain. If None, creates one.
        retriever: Optional retriever for getting source documents. If None, creates one.
        verbose: If True, print debug information about retrieved chunks
    
    Returns:
        Tuple of (answer, source_docs)
    """
    if retrieval_chain is None:
        retrieval_chain, vectorstore, retriever = setup_retrieval_chain()
        if retrieval_chain is None:
            return "Error: Could not set up retrieval chain. Check your OpenAI API key and ChromaDB setup.", []
    
    try:
        # Get source documents first to see what we're retrieving
        if retriever is None:
            # Create a new retriever if not provided
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        
        # For comparison questions, try to ensure we get chunks from multiple documents
        # by doing a hybrid retrieval: original query + company-specific queries if detected
        question_lower = question.lower()
        source_docs = retriever.invoke(question)
        
        # If question mentions multiple companies and we're only getting one source, 
        # try additional retrievals for the other company
        companies_mentioned = []
        if "apple" in question_lower or "aapl" in question_lower:
            companies_mentioned.append("Apple")
        if "rivian" in question_lower or "rivn" in question_lower:
            companies_mentioned.append("Rivian")
        
        # If multiple companies mentioned but only one source file in results, 
        # do additional retrieval for the missing company
        if len(companies_mentioned) > 1:
            retrieved_sources = set(doc.metadata.get("source_file", "") for doc in source_docs)
            if len(retrieved_sources) == 1:
                # Try to get chunks from the other company
                for company in companies_mentioned:
                    company_query = f"{company} revenue financial"
                    additional_docs = retriever.invoke(company_query)
                    # Add chunks from different source files
                    for doc in additional_docs:
                        if doc.metadata.get("source_file", "") not in retrieved_sources:
                            source_docs.append(doc)
                            retrieved_sources.add(doc.metadata.get("source_file", ""))
                            if len(source_docs) >= 20:  # Limit total chunks
                                break
                    if len(retrieved_sources) > 1:
                        break
        
        # Debug: Show what chunks were retrieved
        if verbose:
            print(f"\n🔍 Retrieved {len(source_docs)} chunks:")
            for i, doc in enumerate(source_docs, 1):
                source_file = doc.metadata.get("source_file", "unknown")
                page = doc.metadata.get("page", "?")
                preview = doc.page_content[:150].replace("\n", " ")
                # Check if chunk contains company indicators
                content_lower = doc.page_content.lower()
                has_company = any(indicator in content_lower for indicator in 
                                ["inc.", "corporation", "company", "registrant", "trading symbol"])
                indicator = "✓" if has_company else " "
                print(f"  {indicator} {i}. [{source_file}, page {page}]: {preview}...")
        
        # Invoke the retrieval chain with the question
        answer = retrieval_chain.invoke(question)
        
        return answer, source_docs
        
    except Exception as e:
        return f"Error querying documents: {e}", []


def interactive_query_mode():
    """
    Start an interactive query session.
    """
    print("\n" + "="*60)
    print("INTERACTIVE QUERY MODE")
    print("="*60)
    print("\nAsk questions about the documents in ChromaDB.")
    print("Type 'quit', 'exit', or 'q' to stop.\n")
    
    # Set up retrieval chain once
    retrieval_chain, vectorstore, retriever = setup_retrieval_chain()
    if retrieval_chain is None:
        print("Failed to set up retrieval chain. Exiting.")
        return
    
    while True:
        try:
            question = input("\n💬 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', '']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            print("\n🔍 Searching documents...")
            answer, sources = query_documents(question, retrieval_chain, retriever, verbose=True)
            
            print(f"\n📝 Answer:\n{answer}")
            
            if sources:
                print(f"\n📚 Retrieved {len(sources)} relevant chunk(s) from:")
                # Show unique source files
                source_files = set()
                for doc in sources:
                    source_file = doc.metadata.get("source_file", "unknown")
                    source_files.add(source_file)
                for sf in source_files:
                    print(f"  - {sf}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def check_chromadb_contents():
    """
    Diagnostic function to check what's actually in ChromaDB.
    """
    print("\n" + "="*60)
    print("ChromaDB Contents Check")
    print("="*60)
    
    total = collection.count()
    print(f"\nTotal chunks in ChromaDB: {total}")
    
    if total == 0:
        print("\n⚠️  WARNING: ChromaDB is empty!")
        print("   Run 'python app.py --once' to process documents first.")
        return
    
    # Get all metadata to see source files
    all_data = collection.get()
    source_files = {}
    for metadata in all_data['metadatas']:
        source_file = metadata.get('source_file', 'unknown')
        source_files[source_file] = source_files.get(source_file, 0) + 1
    
    print("\nChunks by source file:")
    for k, v in sorted(source_files.items()):
        print(f"  {k}: {v} chunks")
    
    # Check for potential issues
    print("\n" + "-"*60)
    if len(source_files) == 1:
        print("⚠️  WARNING: Only one source file found in ChromaDB!")
        print("   If you processed multiple documents, they may not have been added.")
        print("   Check the processing logs for errors.")
    else:
        print(f"✓ Found {len(source_files)} different source files")
    
    print("="*60 + "\n")


def query_once(question: str):
    """
    Query once and print the answer.
    
    Args:
        question: The question to ask
    """
    print(f"\n💬 Question: {question}")
    print("\n🔍 Searching documents...")
    
    answer, sources = query_documents(question)
    
    print(f"\n📝 Answer:\n{answer}")
    
    if sources:
        print(f"\n📚 Retrieved {len(sources)} relevant chunk(s)")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--once":
            # Process all files once and exit
            print("Running in single-pass mode...")
            process_all_files_in_input()
        elif sys.argv[1] == "--query":
            # Interactive query mode
            process_all_files_in_input()  # Process any new files first
            check_chromadb_contents()  # Show what's in ChromaDB
            interactive_query_mode()
        elif sys.argv[1] == "--ask" and len(sys.argv) > 2:
            # Query with a single question
            question = " ".join(sys.argv[2:])
            process_all_files_in_input()  # Process any new files first
            query_once(question)
        elif sys.argv[1] == "--check":
            # Check ChromaDB contents
            check_chromadb_contents()
        else:
            print("Usage:")
            print("  python app.py              # Process files and monitor folder")
            print("  python app.py --once       # Process files once and exit")
            print("  python app.py --query      # Interactive query mode")
            print("  python app.py --ask 'question'  # Ask a single question")
            print("  python app.py --check      # Check ChromaDB contents")
    else:
        # Process existing files first
        process_all_files_in_input()
        
        # Then start monitoring
        monitor_input_folder(poll_interval=5)
