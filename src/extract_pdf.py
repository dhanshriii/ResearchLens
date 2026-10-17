"""
extract_pdf.py - Component 1 of ResearchLens: PDF Document Ingestion & Text Extraction.

In a RAG (Retrieval-Augmented Generation) pipeline, document ingestion is the foundation.
Extracting text per page while preserving page numbers and source metadata enables:
1. Grounded citations: Directing users to the exact page where information originated.
2. Fine-grained chunking: Allowing chunking strategies to respect page boundaries.
3. Provenance tracking: Maintaining trace back to original source files.
"""

from pathlib import Path
import json
import sys

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        print(
            "Error: PyMuPDF is not installed. Please install it using 'pip install pymupdf'.",
            file=sys.stderr,
        )
        sys.exit(1)


# Define base paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "papers" / "restormer.pdf"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "restormer_pages.json"


def extract_pdf_pages(
    pdf_path: str | Path = DEFAULT_PDF_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> list[dict] | None:
    """
    Extracts text page-by-page from a PDF file and saves the result to a JSON file.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path to the destination JSON file.

    Returns:
        A list of dictionary records containing page text and metadata,
        or None if an error occurred.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    # Step 1: Validate that the PDF file exists (Requirement 7)
    if not pdf_path.is_file():
        # Check relative to PROJECT_ROOT in case a relative path was provided
        if (PROJECT_ROOT / pdf_path).is_file():
            pdf_path = PROJECT_ROOT / pdf_path
        else:
            print(
                f"Error: PDF file not found at '{pdf_path}'. Please check that the file exists.",
                file=sys.stderr,
            )
            return None

    # Step 2: Open the PDF document using PyMuPDF (Requirement 1)
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Error: Could not open PDF '{pdf_path}': {e}", file=sys.stderr)
        return None

    extracted_pages = []
    total_pages = len(doc)
    pages_with_text = 0

    try:
        # Step 3: Iterate through each page preserving 1-based page numbering (Requirements 2 & 3)
        for page_num, page in enumerate(doc, start=1):
            # Extract plain text from the current page
            page_text = page.get_text()

            # Check if the page contains non-whitespace text
            if page_text.strip():
                pages_with_text += 1

            # Step 4: Structure into standard record schema (Requirement 5)
            record = {
                "source": pdf_path.name,
                "page": page_num,
                "text": page_text,
            }
            extracted_pages.append(record)

    finally:
        # Step 5: Ensure document handle is closed cleanly
        doc.close()

    # Step 6: Print extraction summary (Requirement 6)
    print(f"Total pages: {total_pages}")
    print(f"Pages containing text: {pages_with_text}")

    # Step 7: Ensure destination directory exists and save to JSON (Requirement 4)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_pages, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved extracted pages to '{output_path}'.")
    except Exception as e:
        print(f"Error saving output to '{output_path}': {e}", file=sys.stderr)
        return None

    return extracted_pages


if __name__ == "__main__":
    result = extract_pdf_pages()
    if result is None:
        sys.exit(1)
