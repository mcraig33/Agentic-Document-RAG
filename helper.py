# Add your utilities or helper functions to this file.
from pathlib import Path
from PIL import Image as PILImage, ImageDraw
import pymupdf
from typing import Union, Dict, List, Any, Optional, Tuple
from functools import lru_cache
import fitz
import io

# Try to import IPython display functions, but don't fail if not in notebook
try:
    from IPython.display import display, Image as DisplayImage, IFrame
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False
    # Create dummy functions for non-notebook environments
    def display(*args, **kwargs):
        pass
    DisplayImage = None
    IFrame = None


## Based on the code sample provided at https://docs.landing.ai/ade/ade-python#visualize-parsed-chunks:-draw-bounding-boxes

# Define colors for each chunk type
CHUNK_TYPE_COLORS = {
    "chunkText": (40, 167, 69),        # Green
    "chunkTable": (0, 123, 255),       # Blue
    "chunkMarginalia": (111, 66, 193), # Purple
    "chunkFigure": (255, 0, 255),      # Magenta
    "chunkLogo": (144, 238, 144),      # Light green
    "chunkCard": (255, 165, 0),        # Orange
    "chunkAttestation": (0, 255, 255), # Cyan
    "chunkScanCode": (255, 193, 7),    # Yellow
    "chunkForm": (220, 20, 60),        # Red
    "tableCell": (173, 216, 230),      # Light blue
    "table": (70, 130, 180),           # Steel blue
}


# ============================================================================
# Environment Detection and Display Helpers
# ============================================================================

def is_notebook() -> bool:
    """Check if running in an IPython/Jupyter notebook environment."""
    return IN_NOTEBOOK


def show_image_in_notebook(image: PILImage.Image) -> None:
    """Display a PIL Image in a notebook (only works in notebook environment)."""
    if IN_NOTEBOOK:
        display(image)
    else:
        print(f"Image size: {image.size}, mode: {image.mode}")


def show_pdf_iframe(path: Union[str, Path], width: int = 800, height: int = 600) -> None:
    """Display a PDF in a notebook using IFrame (only works in notebook environment)."""
    if IN_NOTEBOOK and IFrame:
        display(IFrame(src=str(path), width=width, height=height))
    else:
        print(f"PDF file: {path}")


# ============================================================================
# Document Loading and Preview Functions
# ============================================================================

def load_document_preview(file_path: Union[str, Path], page_num: int = 0) -> Optional[PILImage.Image]:
    """
    Load a document preview as a PIL Image.
    
    Args:
        file_path: Path to the document file (PDF or image)
        page_num: For PDFs, which page to render (0-indexed)
    
    Returns:
        PIL Image object, or None if file not found or unsupported
    """
    path = Path(file_path)
    if not path.exists():
        return None
    
    if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
        img = PILImage.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    elif path.suffix.lower() == '.pdf':
        try:
            pdf = pymupdf.open(path)
            if page_num >= len(pdf):
                page_num = 0
            page = pdf[page_num]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x scaling
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf.close()
            return img
        except Exception as e:
            print(f"Error loading PDF preview: {e}")
            return None
    else:
        return None


def print_document(file_path: Union[str, Path], show_in_notebook: Optional[bool] = None):
    """
    Display a PDF or image file. Works in both notebook and non-notebook environments.
    
    Args:
        file_path: The path to the document file.
        show_in_notebook: If True, use notebook display. If None, auto-detect.
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"File not found: {file_path}")
        return
    
    # Auto-detect notebook if not specified
    if show_in_notebook is None:
        show_in_notebook = is_notebook()
    
    if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
        if show_in_notebook and DisplayImage:
            display(DisplayImage(filename=str(path)))
        else:
            img = load_document_preview(path)
            if img:
                print(f"Image: {path} ({img.size[0]}x{img.size[1]} pixels)")
            else:
                print(f"Image file: {path}")
    elif path.suffix.lower() == '.pdf':
        if show_in_notebook:
            show_pdf_iframe(path)
        else:
            print(f"PDF file: {path}")
            # Optionally show first page preview
            preview = load_document_preview(path, page_num=0)
            if preview:
                print(f"  First page preview: {preview.size[0]}x{preview.size[1]} pixels")
    else:
        print(f"Unsupported file type: {path.suffix}")


# ============================================================================
# Bounding Box Visualization Functions
# ============================================================================

def create_annotated_image(image: PILImage.Image, groundings: Dict, page_num: int = 0) -> Optional[PILImage.Image]:
    """
    Create an annotated image with grounding boxes and labels.
    
    Args:
        image: PIL Image to annotate
        groundings: Dictionary of grounding objects with chunk locations
        page_num: Page number to filter groundings (for multi-page PDFs)
    
    Returns:
        Annotated PIL Image, or None if no groundings found for this page
    """
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    img_width, img_height = image.size
    groundings_found = 0
    
    for gid, grounding in groundings.items():
        # Check if grounding belongs to this page
        if hasattr(grounding, 'page') and grounding.page != page_num:
            continue
        
        groundings_found += 1
        box = grounding.box
        
        # Extract normalized coordinates from box
        left, top, right, bottom = box.left, box.top, box.right, box.bottom
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(left * img_width)
        y1 = int(top * img_height)
        x2 = int(right * img_width)
        y2 = int(bottom * img_height)
        
        # Draw bounding box with color based on chunk type
        color = CHUNK_TYPE_COLORS.get(grounding.type, (128, 128, 128))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background and text
        label = f"{grounding.type}:{gid}"
        label_y = max(0, y1 - 20)
        draw.rectangle([x1, label_y, x1 + len(label) * 8, y1], fill=color)
        draw.text((x1 + 2, label_y + 2), label, fill=(255, 255, 255))
    
    if groundings_found == 0:
        return None
    return annotated_img


def draw_bounding_boxes_2(
    groundings: Dict, 
    document_path: Union[str, Path], 
    base_path: Union[str, Path] = ".", 
    save: bool = True,
    return_images: bool = False
) -> Union[List[PILImage.Image], List[Path], None]:
    """
    Draw bounding boxes on document images to visualize parsed chunks.
    
    Args:
        groundings: Dictionary of grounding objects with chunk locations
        document_path: Path to the original document
        base_path: Directory to save annotated images
        save: Whether to save annotated images to disk
        return_images: If True, return list of PIL Images; if False, return list of file paths
    
    Returns:
        List of PIL Images if return_images=True, list of file paths if return_images=False and save=True,
        or None if no images were created
    """
    document_path = Path(document_path)
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    annotated_images = []
    annotated_paths = []
    
    if document_path.suffix.lower() == '.pdf':
        pdf = pymupdf.open(document_path)
        total_pages = len(pdf)
        
        for page_num in range(total_pages):
            page = pdf[page_num]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x scaling
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Create annotated image for this page
            annotated_img = create_annotated_image(img, groundings, page_num)
            if annotated_img is not None:
                annotated_path = base_path / f"page_{page_num + 1}_annotated.png"
                
                if save:
                    annotated_img.save(annotated_path)
                    print(f"Annotated image saved to: {annotated_path}")
                    annotated_paths.append(annotated_path)
                
                if return_images:
                    annotated_images.append(annotated_img)
        
        pdf.close()
    else:
        # Handle image files directly
        img = PILImage.open(document_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Create annotated image
        annotated_img = create_annotated_image(img, groundings)
        if annotated_img is not None:
            annotated_path = base_path / "page_annotated.png"
            
            if save:
                annotated_img.save(annotated_path)
                print(f"Annotated image saved to: {annotated_path}")
                annotated_paths.append(annotated_path)
            
            if return_images:
                annotated_images.append(annotated_img)
    
    if return_images:
        return annotated_images if annotated_images else None
    elif save:
        return annotated_paths if annotated_paths else None
    else:
        return None


def draw_bounding_boxes(
    parse_response, 
    document_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    save: bool = False,
    show_in_notebook: Optional[bool] = None
) -> Optional[PILImage.Image]:
    """
    Draw bounding boxes for all grounded chunks on the given document.
    
    Args:
        parse_response: ADE ParseResponse containing grounding info.
        document_path: Path to the original PDF or image.
        output_dir: Directory to save annotated images (if save=True)
        save: Whether to save annotated images to disk
        show_in_notebook: If True, display in notebook. If None, auto-detect.
    
    Returns:
        The last annotated PIL Image object (for single-page docs, this is the only page),
        or None if no annotations were created
    """
    document_path = Path(document_path)
    
    if show_in_notebook is None:
        show_in_notebook = is_notebook()
    
    annotated_images = []
    
    if document_path.suffix.lower() == '.pdf':
        pdf = pymupdf.open(document_path)
        total_pages = len(pdf)
        base_name = document_path.stem
        
        for page_num in range(total_pages):
            page = pdf[page_num]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # 2x scaling
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Create annotated image
            annotated_img = create_annotated_image(img, parse_response.grounding, page_num)
            if annotated_img is not None:
                annotated_images.append(annotated_img)
                
                if save and output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    annotated_path = output_path / f"page_{page_num + 1}_annotated.png"
                    annotated_img.save(annotated_path)
                    print(f"Annotated image saved to: {annotated_path}")
        
        pdf.close()
    else:
        # Load image file directly
        img = PILImage.open(document_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Create annotated image
        annotated_img = create_annotated_image(img, parse_response.grounding)
        if annotated_img is not None:
            annotated_images.append(annotated_img)
            
            if save and output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                annotated_path = output_path / "page_annotated.png"
                annotated_img.save(annotated_path)
                print(f"Annotated image saved to: {annotated_path}")
    
    # Return the last annotated image (or first if only one)
    result = annotated_images[-1] if annotated_images else None
    
    # Display in notebook if requested
    if show_in_notebook and result:
        show_image_in_notebook(result)
    
    return result


# ============================================================================
# Chunk Image Extraction Functions
# ============================================================================

def create_cropped_chunk_images(
    parse_result, 
    extraction_metadata: Dict, 
    document_path: Union[str, Path], 
    first_page: int, 
    doc_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    save: bool = False
) -> Dict[str, Dict[str, Union[PILImage.Image, Path]]]:
    """
    Create cropped images of individual chunks and full page with chunks outlined.
    
    Args:
        parse_result: ADE ParseResponse containing grounding info
        extraction_metadata: Dictionary mapping field names to metadata with chunk references
        document_path: Path to the original PDF
        first_page: Page number to process (0-indexed)
        doc_name: Document name for file naming
        output_dir: Directory to save images (if save=True)
        save: Whether to save images to disk
    
    Returns:
        Dictionary mapping field_name -> {
            "crop": PIL.Image or Path,
            "outlined": PIL.Image or Path
        }
    """
    document_path = Path(document_path)
    pdf = pymupdf.open(document_path)
    page = pdf[first_page]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    full_page_img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf.close()
    
    img_width, img_height = full_page_img.size
    field_images = {}
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    for field_name, metadata in extraction_metadata.items():
        # Get the first chunk reference
        chunk_id = metadata['references'][0]
        
        if chunk_id not in parse_result.grounding:
            continue
        
        grounding = parse_result.grounding[chunk_id]
        
        # Only process if it's on the first page
        if grounding.page != first_page:
            continue
        
        box = grounding.box
        left, top, right, bottom = box.left, box.top, box.right, box.bottom
        
        # Convert normalized coordinates to pixels
        x1 = int(left * img_width)
        y1 = int(top * img_height)
        x2 = int(right * img_width)
        y2 = int(bottom * img_height)
        
        # Add padding for better visibility
        padding = 10
        x1_crop = max(0, x1 - padding)
        y1_crop = max(0, y1 - padding)
        x2_crop = min(img_width, x2 + padding)
        y2_crop = min(img_height, y2 + padding)
        
        # Create cropped image
        cropped = full_page_img.crop((x1_crop, y1_crop, x2_crop, y2_crop))
        
        # Create outlined version (full page with just this chunk highlighted)
        outlined = full_page_img.copy()
        draw = ImageDraw.Draw(outlined)
        color = (231, 76, 60)  # Red
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        
        # Add label
        label = field_name
        label_y = max(0, y1 - 25)
        draw.rectangle([x1, label_y, x1 + len(label) * 10, y1], fill=color)
        draw.text((x1 + 5, label_y + 5), label, fill=(255, 255, 255))
        
        result = {"crop": cropped, "outlined": outlined}
        
        # Save if requested
        if save and output_dir:
            crop_path = output_path / f"{doc_name}_{field_name}_crop.png"
            outlined_path = output_path / f"{doc_name}_{field_name}_outlined.png"
            cropped.save(crop_path)
            outlined.save(outlined_path)
            result = {"crop": crop_path, "outlined": outlined_path}
        
        field_images[field_name] = result
    
    return field_images


@lru_cache(maxsize=20)
def get_pdf_page_cached(pdf_path_str: str, page_num: int, dpi: int = 150) -> Tuple[PILImage.Image, float, float]:
    """
    Cache PDF pages for faster repeated access.
    
    Args:
        pdf_path_str: Path to PDF file (as string for caching)
        page_num: Page number (0-indexed)
        dpi: Resolution for rendering
    
    Returns:
        Tuple of (PIL Image, page_width, page_height)
    """
    doc = fitz.open(pdf_path_str)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
    page_width, page_height = page.rect.width, page.rect.height
    doc.close()
    return img, page_width, page_height


def extract_chunk_image(
    pdf_path: Union[str, Path], 
    page_num: int, 
    bbox: Optional[List[float]] = None, 
    highlight: bool = True, 
    padding: int = 10
) -> Optional[bytes]:
    """
    Dynamically extract and crop a specific chunk from PDF.
    
    Args:
        pdf_path: Path to PDF
        page_num: Page number (0-indexed)
        bbox: [x0, y0, x1, y1] in NORMALIZED coordinates (0-1 range) or None for full page
        highlight: Add red border around chunk
        padding: Extra pixels around bbox (default 10)
    
    Returns:
        PNG image bytes or None
    """
    pdf_path = Path(pdf_path)
    
    # Get cached page
    img, page_width, page_height = get_pdf_page_cached(str(pdf_path), page_num)
    
    if img is None:
        return None
    
    # If no bbox, return full page
    if not bbox or len(bbox) != 4:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    # Extract normalized bbox coordinates (0-1 range)
    norm_x0, norm_y0, norm_x1, norm_y1 = bbox
    
    # Convert normalized coordinates to PDF points
    pdf_x0 = norm_x0 * page_width
    pdf_y0 = norm_y0 * page_height
    pdf_x1 = norm_x1 * page_width
    pdf_y1 = norm_y1 * page_height
    
    # Scale PDF points to image pixels
    scale_x = img.width / page_width
    scale_y = img.height / page_height
    
    # Apply scaling and padding
    crop_x0 = max(0, int(pdf_x0 * scale_x) - padding)
    crop_y0 = max(0, int(pdf_y0 * scale_y) - padding)
    crop_x1 = min(img.width, int(pdf_x1 * scale_x) + padding)
    crop_y1 = min(img.height, int(pdf_y1 * scale_y) + padding)
    
    # Crop to chunk region
    chunk_img = img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    
    # Add red border highlight
    if highlight:
        draw = ImageDraw.Draw(chunk_img)
        draw.rectangle(
            [padding, padding, 
             chunk_img.width - padding - 1, 
             chunk_img.height - padding - 1],
            outline="red",
            width=3
        )
    
    # Convert to PNG bytes
    img_bytes = io.BytesIO()
    chunk_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()
