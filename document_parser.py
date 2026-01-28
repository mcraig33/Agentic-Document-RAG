"""
Parse a document with ADE and Landing.AI API
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from landingai_ade import LandingAIADE

# Load environment variables
load_dotenv()

def parse_document(document_path: str | Path, output_dir: str | Path = "./ade_outputs"):
    """
    Parse a document using Landing.AI ADE API and save chunks and markdown.
    
    Args:
        document_path: Path to the document file (e.g., "apple_10k.pdf")
        output_dir: Directory to save output files (default: "./ade_outputs")
    
    Returns:
        tuple: (chunks_json_path, markdown_path)
    """
    # Convert to Path objects
    doc_path = Path(document_path)
    output_dir_path = Path(output_dir)
    
    # Validate document exists
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")
    
    # Create output directory if it doesn't exist
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get base filename without extension (e.g., "apple_10k" from "apple_10k.pdf")
    base_name = doc_path.stem
    
    # Define output file paths
    chunks_json_path = output_dir_path / f"{base_name}_chunks.json"
    markdown_path = output_dir_path / f"{base_name}.md"
    
    # Initialize Landing.AI ADE client
    # Check for API key in environment variables
    api_key = os.getenv("LANDINGAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Landing.AI API key not found."
        )
    
    # Initialize client - pass apikey parameter to constructor
    ade = LandingAIADE(apikey=api_key)
    
    # Parse the document
    print(f"Parsing document: {doc_path}")
    parse_response = ade.parse(document=doc_path)
    
    # Extract markdown from the response
    markdown_content = parse_response.markdown
    
    # Extract chunks/grounding information
    # Convert grounding to a serializable format
    chunks_data = []
    if hasattr(parse_response, 'grounding') and parse_response.grounding:
        for chunk_id, grounding in parse_response.grounding.items():
            chunk_info = {
                "id": chunk_id,
                "type": grounding.type if hasattr(grounding, 'type') else None,
                "page": grounding.page if hasattr(grounding, 'page') else None,
                "box": {
                    "left": grounding.box.left if hasattr(grounding, 'box') else None,
                    "top": grounding.box.top if hasattr(grounding, 'box') else None,
                    "right": grounding.box.right if hasattr(grounding, 'box') else None,
                    "bottom": grounding.box.bottom if hasattr(grounding, 'box') else None,
                } if hasattr(grounding, 'box') else None,
                "text": grounding.text if hasattr(grounding, 'text') else None,
            }
            chunks_data.append(chunk_info)
    
    # Save markdown file
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Saved markdown to: {markdown_path}")
    
    # Save chunks JSON file
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    print(f"Saved chunks to: {chunks_json_path}")
    print(f"Total chunks: {len(chunks_data)}")
    
    return chunks_json_path, markdown_path


if __name__ == "__main__":
    # Example usage
    DOCUMENT_PATH = Path("apple_10k.pdf")
    
    if DOCUMENT_PATH.exists():
        parse_document(DOCUMENT_PATH)
    else:
        print(f"Document not found: {DOCUMENT_PATH}")
