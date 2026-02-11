## Agentic Document RAG

This project demonstrates an **agentic RAG (Retrieval-Augmented Generation)** workflow over long documents using:

- **Landing.AI ADE** to parse PDFs into structured chunks and markdown
- **LangChain + OpenAI** for retrieval and question-answering
- A small orchestration script (`app.py`) that ties document parsing and querying together

The example document is `apple_10k.pdf`, but the pipeline is designed to work with any similar PDF.

---

### Project Structure

- `app.py`  
  Orchestrates the workflow:
  - Displays the source PDF
  - Ensures parsed outputs exist (markdown + chunks JSON via `document_parser.py`)
  - Loads the parsed chunks and prints a preview
  - (You can extend this to build a full RAG chain over the chunks.)

- `document_parser.py`  
  Uses **LandingAIADE** to parse a document into:
  - `ade_outputs/<file_stem>_chunks.json` – structured chunk metadata / grounding
  - `ade_outputs/<file_stem>.md` – markdown representation of the document

- `helper.py`  
  Utility functions for:
  - Loading and displaying PDFs/images (works in both CLI and notebook environments)
  - Visualizing ADE chunks as bounding boxes or cropped images
  - Extracting chunk images from PDFs

- `ade_outputs/`  
  Generated outputs (markdown + chunks JSON) for each parsed document.

- `requirements.txt`  
  Pinned dependencies for the project (LangChain, Landing.AI ADE, OpenAI, etc.).

---

### Prerequisites

- Python **3.11** (recommended to match the existing virtualenv)
- Git

You should also have:

- An **OpenAI API key** (for LangChain + OpenAI usage)
- A **Landing.AI API key** (for ADE document parsing)

---

### Setup

1. **Clone the repository**

```bash
git clone git@github.com:mcraig33/Agentic-Document-RAG.git
cd Agentic-Document-RAG
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows (PowerShell/CMD)
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Set environment variables**

Create a `.env` file in the project root:

```bash
touch .env
```

Add the following keys (adjust names/values as needed):

```env
OPENAI_API_KEY=your_openai_key_here
LANDINGAI_API_KEY=your_landingai_key_here  # used by document_parser.py
```

If your Landing.AI account uses a different key name (e.g. `VISION_AGENT_API_KEY`), you can set it instead and update `document_parser.py` accordingly.

---

### Parsing a Document with ADE

`document_parser.py` exposes a function `parse_document` that:

- Takes a document path (e.g. `apple_10k.pdf`)
- Calls the Landing.AI ADE API
- Writes:
  - `ade_outputs/apple_10k_chunks.json`
  - `ade_outputs/apple_10k.md`

You can run it directly:

```bash
source .venv/bin/activate
python document_parser.py
```

By default it looks for `apple_10k.pdf` in the project root. To parse a different document, call it from Python:

```python
from pathlib import Path
from document_parser import parse_document

parse_document(Path("path/to/your_doc.pdf"))
```

---

### Running the End-to-End Script

`app.py` ties together document display, parsing, and preview.

```bash
source .venv/bin/activate
python app.py
```

What it does:

1. Displays `apple_10k.pdf` using the helpers in `helper.py`
2. Checks whether:
   - `ade_outputs/apple_10k.md`
   - `ade_outputs/apple_10k_chunks.json`
   exist and are non-empty
3. If either is missing/empty, calls `document_parser.parse_document` to generate them
4. Prints the first ~500 characters of the markdown
5. Loads the chunks JSON, prints the count and a sample chunk

You can then extend `app.py` to:

- Build a vector store from the chunks (LangChain + Chroma)
- Create a retrieval chain with `ChatOpenAI`
- Implement agentic behaviors (e.g. tool-calling, multi-step reasoning over chunks)

---

### Customizing for Other Documents

To use another PDF:

1. Place the PDF in the project root (or elsewhere).
2. Update `DOC_PATH` in `app.py`:

```python
DOC_PATH = Path("your_doc.pdf")
```

3. Run:

```bash
python app.py
```

The outputs will be written to:

- `ade_outputs/your_doc_chunks.json`
- `ade_outputs/your_doc.md`

---

### Using Helper Functions

The `helper.py` module provides visualization and image extraction utilities that work in both **CLI/service mode** and **Jupyter notebook environments**.

#### Document Display

```python
from helper import print_document, load_document_preview

# In CLI mode (default behavior)
print_document("document.pdf")  # Prints file info, no display

# In notebook mode (auto-detected)
print_document("document.pdf")  # Displays PDF in IFrame or image inline

# Explicitly control behavior
print_document("document.pdf", show_in_notebook=False)  # CLI mode
print_document("document.pdf", show_in_notebook=True)   # Notebook mode

# Get preview as PIL Image (works everywhere)
preview = load_document_preview("document.pdf", page_num=0)
if preview:
    preview.save("preview.png")  # Save to file
```

#### Bounding Box Visualization

```python
from helper import draw_bounding_boxes_2, draw_bounding_boxes
from pathlib import Path

# Save annotated images to disk (CLI mode)
annotated_paths = draw_bounding_boxes_2(
    groundings=parse_response.grounding,
    document_path="document.pdf",
    base_path="./annotations",
    save=True,
    return_images=False
)

# Get PIL Images instead of saving (for web API, etc.)
annotated_images = draw_bounding_boxes_2(
    groundings=parse_response.grounding,
    document_path="document.pdf",
    save=False,
    return_images=True
)

# Draw and optionally display in notebook
annotated_img = draw_bounding_boxes(
    parse_response=parse_response,
    document_path="document.pdf",
    output_dir="./annotations",  # Optional: save to disk
    save=True,
    show_in_notebook=None  # Auto-detect notebook environment
)
```

#### Chunk Image Extraction

```python
from helper import create_cropped_chunk_images, extract_chunk_image

# Create cropped images for extraction fields
field_images = create_cropped_chunk_images(
    parse_result=parse_response,
    extraction_metadata=metadata,
    document_path="document.pdf",
    first_page=0,
    doc_name="document",
    output_dir="./chunk_images",  # Optional
    save=True  # Save to disk
)

# Access images
for field_name, images in field_images.items():
    crop_img = images["crop"]  # PIL Image or Path
    outlined_img = images["outlined"]  # PIL Image or Path

# Extract specific chunk as PNG bytes (for API responses)
chunk_bytes = extract_chunk_image(
    pdf_path="document.pdf",
    page_num=0,
    bbox=[0.1, 0.2, 0.5, 0.6],  # Normalized coordinates
    highlight=True,
    padding=10
)
```

#### Environment Detection

The helper functions automatically detect if you're running in a notebook:

```python
from helper import is_notebook, show_image_in_notebook, show_pdf_iframe

if is_notebook():
    # Use notebook display functions
    show_image_in_notebook(pil_image)
    show_pdf_iframe("document.pdf")
else:
    # CLI mode - work with files/images directly
    pil_image.save("output.png")
    print(f"PDF: document.pdf")
```

All helper functions are designed to work without IPython dependencies in CLI mode, making them suitable for production services, APIs, and automated workflows.

---

### Notes & Troubleshooting

- **Unrelated Git histories**  
  If you initialized local Git separately and later added a remote, you may need:

  ```bash
  git pull origin main --allow-unrelated-histories
  ```

- **Landing.AI API errors**  
  If you see errors mentioning the API key:
  - Confirm the key in `.env`
  - Ensure the environment is loaded (via `load_dotenv()` and/or your shell)

- **Large outputs**  
  `ade_outputs/apple_10k.md` and `apple_10k_chunks.json` can be large for long documents. They are safe to regenerate via `document_parser.py`.

---

### Next Steps

Some ideas for extending this project:

- Build a full **agentic RAG pipeline** over ADE chunks (with LangChain tools/agents)
- Add a **CLI or web UI** for uploading PDFs and querying them
- Enhance **chunk visualization** using `helper.py` functions and ADE grounding data

