#!/usr/bin/env python3
"""
Test script for grounding visualization functions in helper.py

This script tests the bounding box visualization and chunk extraction functions
using the parsed chunks from apple_10k_chunks.json.
"""

import json
from pathlib import Path
from typing import Dict, Any

# Import helper functions
import helper


class MockGrounding:
    """Mock grounding object to simulate ADE parse_response.grounding items"""
    def __init__(self, chunk_data: Dict[str, Any]):
        self.type = chunk_data.get("type", "unknown")
        self.page = chunk_data.get("page", 0)
        
        # Create a mock box object
        box_data = chunk_data.get("box", {})
        self.box = type('Box', (), {
            'left': box_data.get("left", 0),
            'top': box_data.get("top", 0),
            'right': box_data.get("right", 1),
            'bottom': box_data.get("bottom", 1),
        })()


def load_chunks_as_groundings(chunks_json_path: Path) -> Dict[str, MockGrounding]:
    """
    Load chunks from JSON and convert to mock grounding objects.
    
    Args:
        chunks_json_path: Path to chunks JSON file
    
    Returns:
        Dictionary mapping chunk_id -> MockGrounding object
    """
    with open(chunks_json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    groundings = {}
    for chunk in chunks:
        chunk_id = chunk.get("id")
        if chunk_id:
            groundings[chunk_id] = MockGrounding(chunk)
    
    return groundings


def test_document_preview():
    """Test document preview loading"""
    print("\n" + "="*60)
    print("TEST 1: Document Preview Loading")
    print("="*60)
    
    pdf_path = Path("processed/apple_10k.pdf")
    if not pdf_path.exists():
        pdf_path = Path("input/apple_10k.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found. Please ensure apple_10k.pdf is in input/ or processed/")
        return False
    
    print(f"Loading preview from: {pdf_path}")
    preview = helper.load_document_preview(pdf_path, page_num=0)
    
    if preview:
        print(f"✓ Preview loaded successfully")
        print(f"  - Size: {preview.size[0]}x{preview.size[1]} pixels")
        print(f"  - Mode: {preview.mode}")
        
        # Save preview for verification
        output_path = Path("ade_outputs/test_preview.png")
        preview.save(output_path)
        print(f"  - Saved to: {output_path}")
        return True
    else:
        print("❌ Failed to load preview")
        return False


def test_bounding_boxes_visualization():
    """Test bounding box visualization"""
    print("\n" + "="*60)
    print("TEST 2: Bounding Box Visualization")
    print("="*60)
    
    # Load chunks and convert to groundings
    chunks_path = Path("ade_outputs/apple_10k_chunks.json")
    if not chunks_path.exists():
        print(f"❌ Chunks JSON not found: {chunks_path}")
        print("   Run document_parser.py first to generate chunks")
        return False
    
    print(f"Loading chunks from: {chunks_path}")
    groundings = load_chunks_as_groundings(chunks_path)
    print(f"✓ Loaded {len(groundings)} chunks")
    
    # Find PDF
    pdf_path = Path("processed/apple_10k.pdf")
    if not pdf_path.exists():
        pdf_path = Path("input/apple_10k.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found")
        return False
    
    print(f"Visualizing bounding boxes for: {pdf_path.name}")
    
    # Test draw_bounding_boxes_2
    output_dir = Path("ade_outputs/test_annotations")
    print(f"\nTesting draw_bounding_boxes_2 (saving to {output_dir})...")
    
    try:
        annotated_paths = helper.draw_bounding_boxes_2(
            groundings=groundings,
            document_path=pdf_path,
            base_path=output_dir,
            save=True,
            return_images=False
        )
        
        if annotated_paths:
            print(f"✓ Successfully created {len(annotated_paths)} annotated page(s)")
            for path in annotated_paths[:3]:  # Show first 3
                print(f"  - {path}")
            if len(annotated_paths) > 3:
                print(f"  ... and {len(annotated_paths) - 3} more")
            return True
        else:
            print("⚠ No annotated images were created (no groundings matched pages?)")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunk_image_extraction():
    """Test chunk image extraction"""
    print("\n" + "="*60)
    print("TEST 3: Chunk Image Extraction")
    print("="*60)
    
    # Load chunks
    chunks_path = Path("ade_outputs/apple_10k_chunks.json")
    if not chunks_path.exists():
        print(f"❌ Chunks JSON not found")
        return False
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Find PDF
    pdf_path = Path("processed/apple_10k.pdf")
    if not pdf_path.exists():
        pdf_path = Path("input/apple_10k.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found")
        return False
    
    # Get a few chunks from the first page with text
    first_page_chunks = [
        chunk for chunk in chunks 
        if chunk.get("page") == 0 and chunk.get("text") and chunk.get("text").strip()
    ][:3]  # Take first 3
    
    if not first_page_chunks:
        print("⚠ No chunks with text found on first page")
        return False
    
    print(f"Extracting images for {len(first_page_chunks)} chunks from page 0...")
    
    success_count = 0
    for i, chunk in enumerate(first_page_chunks):
        chunk_id = chunk.get("id", f"chunk_{i}")
        box = chunk.get("box", {})
        
        if not box:
            continue
        
        bbox = [
            box.get("left", 0),
            box.get("top", 0),
            box.get("right", 1),
            box.get("bottom", 1)
        ]
        
        print(f"\n  Chunk {i+1}: {chunk_id[:20]}...")
        print(f"    Type: {chunk.get('type')}")
        print(f"    Text preview: {chunk.get('text', '')[:50]}...")
        
        try:
            chunk_bytes = helper.extract_chunk_image(
                pdf_path=pdf_path,
                page_num=0,
                bbox=bbox,
                highlight=True,
                padding=10
            )
            
            if chunk_bytes:
                # Save extracted chunk
                output_path = Path(f"ade_outputs/test_chunk_{i+1}_{chunk_id[:8]}.png")
                with open(output_path, 'wb') as f:
                    f.write(chunk_bytes)
                print(f"    ✓ Extracted and saved to: {output_path}")
                success_count += 1
            else:
                print(f"    ⚠ No image extracted")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    if success_count > 0:
        print(f"\n✓ Successfully extracted {success_count} chunk image(s)")
        return True
    else:
        print(f"\n❌ Failed to extract any chunk images")
        return False


def test_cropped_chunk_images():
    """Test cropped chunk images creation"""
    print("\n" + "="*60)
    print("TEST 4: Cropped Chunk Images")
    print("="*60)
    
    # This function requires a parse_result object with grounding
    # For testing, we'll create a minimal mock
    chunks_path = Path("ade_outputs/apple_10k_chunks.json")
    if not chunks_path.exists():
        print(f"❌ Chunks JSON not found")
        return False
    
    groundings = load_chunks_as_groundings(chunks_path)
    
    # Create mock parse_result
    class MockParseResult:
        def __init__(self, groundings):
            self.grounding = groundings
    
    parse_result = MockParseResult(groundings)
    
    # Create simple extraction metadata (using first few chunks as "fields")
    extraction_metadata = {}
    for i, (chunk_id, grounding) in enumerate(list(groundings.items())[:5]):
        if grounding.page == 0:  # Only first page
            extraction_metadata[f"field_{i}"] = {
                'references': [chunk_id]
            }
    
    if not extraction_metadata:
        print("⚠ No chunks found on first page for testing")
        return False
    
    pdf_path = Path("processed/apple_10k.pdf")
    if not pdf_path.exists():
        pdf_path = Path("input/apple_10k.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found")
        return False
    
    print(f"Creating cropped images for {len(extraction_metadata)} fields...")
    
    try:
        field_images = helper.create_cropped_chunk_images(
            parse_result=parse_result,
            extraction_metadata=extraction_metadata,
            document_path=pdf_path,
            first_page=0,
            doc_name="test",
            output_dir=Path("ade_outputs/test_crops"),
            save=True
        )
        
        if field_images:
            print(f"✓ Successfully created images for {len(field_images)} field(s)")
            for field_name, images in field_images.items():
                crop_path = images.get("crop")
                outlined_path = images.get("outlined")
                print(f"  - {field_name}:")
                if isinstance(crop_path, Path):
                    print(f"    Crop: {crop_path}")
                if isinstance(outlined_path, Path):
                    print(f"    Outlined: {outlined_path}")
            return True
        else:
            print("⚠ No field images created")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GROUNDING VISUALIZATION TEST SUITE")
    print("="*60)
    print("\nThis script tests the helper functions for visualizing")
    print("ADE grounding data (bounding boxes, chunk extraction, etc.)")
    print("\nMake sure you have:")
    print("  - apple_10k.pdf in input/ or processed/")
    print("  - apple_10k_chunks.json in ade_outputs/")
    
    results = []
    
    # Run tests
    results.append(("Document Preview", test_document_preview()))
    results.append(("Bounding Box Visualization", test_bounding_boxes_visualization()))
    results.append(("Chunk Image Extraction", test_chunk_image_extraction()))
    results.append(("Cropped Chunk Images", test_cropped_chunk_images()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Visualization functions are working correctly.")
        print("\nCheck the following directories for output images:")
        print("  - ade_outputs/test_preview.png")
        print("  - ade_outputs/test_annotations/ (annotated pages)")
        print("  - ade_outputs/test_chunk_*.png (extracted chunks)")
        print("  - ade_outputs/test_crops/ (cropped field images)")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
