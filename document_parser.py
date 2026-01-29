"""
Parse a document with ADE and Landing.AI API
"""
import os
import json
import sys
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
    # Convert grounding to a serializable format, and populate text using PDF + bbox
    chunks_data = []
    if hasattr(parse_response, "grounding") and parse_response.grounding:
        # Open the PDF once so we can extract text for each bounding box
        doc = fitz.open(str(doc_path))

        try:
            for chunk_id, grounding in parse_response.grounding.items():
                # Base fields from grounding
                page_index = grounding.page if hasattr(grounding, "page") else None
                box = grounding.box if hasattr(grounding, "box") else None

                # Default text from grounding if present (often None)
                text = None
                if hasattr(grounding, "text") and grounding.text:
                    text = grounding.text
                elif hasattr(grounding, "content") and grounding.content:
                    text = grounding.content
                elif hasattr(grounding, "value") and grounding.value:
                    text = grounding.value

                # If we still don't have text but have page + bbox, extract from PDF
                if (not text) and (page_index is not None) and box is not None:
                    try:
                        page = doc[page_index]
                        page_rect = page.rect

                        # ADE bbox is in normalized coordinates [0,1]; convert to PDF points
                        x0 = float(box.left) * page_rect.width
                        y0 = float(box.top) * page_rect.height
                        x1 = float(box.right) * page_rect.width
                        y1 = float(box.bottom) * page_rect.height

                        clip_rect = fitz.Rect(x0, y0, x1, y1)
                        extracted = page.get_text("text", clip=clip_rect) or ""
                        text = extracted.strip() or None
                    except Exception:
                        # Fail silently for this chunk; text will remain None
                        text = text or None

                chunk_info = {
                    "id": chunk_id,
                    "type": grounding.type if hasattr(grounding, "type") else None,
                    "page": page_index,
                    "box": {
                        "left": float(box.left) if box is not None else None,
                        "top": float(box.top) if box is not None else None,
                        "right": float(box.right) if box is not None else None,
                        "bottom": float(box.bottom) if box is not None else None,
                    }
                    if box is not None
                    else None,
                    "text": text,
                }
                chunks_data.append(chunk_info)
        finally:
            doc.close()
    
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
