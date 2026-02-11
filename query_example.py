#!/usr/bin/env python3
"""
Example script showing how to query documents using the RAG chain.

This demonstrates the basic usage of querying ChromaDB with an LLM.
"""

from pathlib import Path
from dotenv import load_dotenv
from app import setup_retrieval_chain, query_documents

# Load environment variables
load_dotenv()

def main():
    """Example query session"""
    print("="*60)
    print("Document Query Example")
    print("="*60)
    
    # Set up the retrieval chain
    print("\nSetting up retrieval chain...")
    retrieval_chain = setup_retrieval_chain()
    
    if retrieval_chain is None:
        print("Failed to set up retrieval chain. Make sure:")
        print("  1. ChromaDB has documents (run app.py first to process documents)")
        print("  2. OPENAI_API_KEY is set in your .env file")
        return
    
    # Example questions
    questions = [
        "What is the main topic of this document?",
        "What are the key financial figures mentioned?",
        "Who is the author or company mentioned in the document?",
    ]
    
    print("\n" + "="*60)
    print("Example Queries")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}] Question: {question}")
        print("-" * 60)
        
        answer, sources = query_documents(question, retrieval_chain)
        
        print(f"Answer: {answer}")
        if sources:
            print(f"\nRetrieved {len(sources)} relevant chunk(s) from the document")
    
    print("\n" + "="*60)
    print("Try your own questions!")
    print("="*60)
    print("\nYou can also use the interactive mode:")
    print("  python app.py --query")
    print("\nOr ask a single question:")
    print("  python app.py --ask 'Your question here'")


if __name__ == "__main__":
    main()
