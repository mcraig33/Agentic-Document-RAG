import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import document_parser;

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

DOC_PATH = Path("apple_10k.pdf")
helper.print_document(DOC_PATH)

# Setting Directories and Paths
OUTPUT_DIR = Path("./ade_outputs")
# Build filenames as strings, then join with the output directory
ADE_JSON_PATH = OUTPUT_DIR / f"{Path(DOC_PATH).stem}_chunks.json"
ADE_MD_PATH = OUTPUT_DIR / f"{Path(DOC_PATH).stem}.md"

# Load and display markdown preview
print("\n Parsed Output (Page 1):")

if os.path.getsize(ADE_MD_PATH) == 0 or os.path.getsize(ADE_JSON_PATH) == 0:
    print("No markdown found. Calling API to create markdown...")
    document_parser.parse_document(DOC_PATH, OUTPUT_DIR)

with open(ADE_MD_PATH, "r", encoding="utf-8") as f:
    markdown_content = f.read()
    # Find first page content (up to first page break or 500 chars)
    first_page = markdown_content[:500]
    print(first_page + "...")

with open(ADE_JSON_PATH, "r", encoding="utf-8") as f:
    loaded_chunks = json.load(f)

    print(f"Loaded {len(loaded_chunks)} saved chunks.")

    # Show first chunk structure
    print(f"\n Sample chunk structure:")
    print(json.dumps(loaded_chunks[0], indent=2)[:400] + "...")

print("\n Ready to query!")