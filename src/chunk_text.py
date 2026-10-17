"""
chunk_text.py - Component 2 of ResearchLens: Boundary-Aware Semantic Text Chunking.

Improvements over naive character slicing:
1. Normalization integration: Cleans ligatures, joins line-wrap hyphens, and cleans page noise.
2. Boundary-aware grouping: Chunks respect paragraph and sentence boundaries rather than
   arbitrarily slicing at character offsets.
3. Sentence-level packing: Accumulates complete sentences targeting 600-800 characters.
4. Sentence-level overlap: Carries forward complete sentences between consecutive chunks.
5. Provenance preservation: Keeps 'id', 'chunk_id', 'source', 'page', and 'text' metadata,
   restarting chunk IDs at c001 for each page.
"""

from pathlib import Path
import json
import re
import sys
from typing import Any

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.normalize_text import clean_page_text
except ImportError:
    from normalize_text import clean_page_text

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "restormer_pages.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "restormer_chunks.json"

# Target character lengths
DEFAULT_MIN_CHUNK_SIZE = 400
DEFAULT_TARGET_CHUNK_SIZE = 700
DEFAULT_MAX_CHUNK_SIZE = 850
DEFAULT_MAX_OVERLAP_SIZE = 220


def split_paragraph_into_sentences(text: str) -> list[str]:
    """
    Splits text into complete grammatical sentences using boundary punctuation.
    """
    if not text or not text.strip():
        return []

    # Match sentence endings followed by whitespace and an uppercase letter, digit, quote, or bullet
    pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'(•])")
    raw_sentences = pattern.split(text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


def handle_long_sentence(sentence: str, max_size: int = DEFAULT_MAX_CHUNK_SIZE) -> list[str]:
    """
    Splits an unusually long sentence (e.g., long equations or citations) safely
    at clause boundaries (semicolons, colons, or commas) without cutting words.
    """
    if len(sentence) <= max_size:
        return [sentence]

    sub_parts: list[str] = []
    # Split on semicolons or colons first
    clauses = re.split(r"(?<=[;:])\s+", sentence)
    for clause in clauses:
        if len(clause) <= max_size:
            sub_parts.append(clause)
        else:
            # Further split on commas if still too long
            comma_parts = re.split(r"(?<=,)\s+", clause)
            curr = ""
            for part in comma_parts:
                if len(curr) + len(part) + 1 <= max_size:
                    curr = f"{curr} {part}" if curr else part
                else:
                    if curr:
                        sub_parts.append(curr)
                    curr = part
            if curr:
                sub_parts.append(curr)

    return sub_parts if sub_parts else [sentence]


def chunk_page_text(
    text: str,
    source: str,
    page_num: int,
    target_size: int = DEFAULT_TARGET_CHUNK_SIZE,
    max_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """
    Cleans and splits a single page's text into boundary-aware chunks.

    Args:
        text: Raw page text.
        source: Source document filename (e.g. 'restormer.pdf').
        page_num: 1-based page number.
        target_size: Desired chunk character length.
        max_size: Upper limit before splitting.

    Returns:
        List of structured chunk dictionaries.
    """
    # Step 1: Clean and normalize page text
    cleaned_text = clean_page_text(text, page_num=page_num)
    if not cleaned_text:
        return []

    # Step 2: Segment into paragraphs
    paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]

    # Step 3: Extract sentences, grouping short standalone headings with the following text
    units: list[str] = []
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        # If paragraph is a brief heading (e.g. '1. Introduction', 'Abstract'), combine with next
        if len(p) < 50 and not p.endswith((".", ":", "!", "?")) and i + 1 < len(paragraphs):
            combined = f"{p}: {paragraphs[i+1]}"
            for s in split_paragraph_into_sentences(combined):
                units.extend(handle_long_sentence(s, max_size))
            i += 2
        else:
            for s in split_paragraph_into_sentences(p):
                units.extend(handle_long_sentence(s, max_size))
            i += 1

    # Step 4: Accumulate sentences into chunks with sentence-level overlap
    chunks: list[dict[str, Any]] = []
    chunk_idx = 1
    doc_stem = Path(source).stem

    current_sentences: list[str] = []
    current_length = 0

    for s in units:
        s_len = len(s)
        space_overhead = 1 if current_sentences else 0

        # If adding this sentence would exceed max_size and we have enough content
        if (current_length + s_len + space_overhead > max_size) and (current_length >= DEFAULT_MIN_CHUNK_SIZE):
            chunk_content = " ".join(current_sentences).strip()
            if chunk_content:
                cid = f"{doc_stem}_p{page_num:02d}_c{chunk_idx:03d}"
                chunks.append({
                    "chunk_id": cid,
                    "id": cid,
                    "source": source,
                    "page": page_num,
                    "text": chunk_content,
                })
                chunk_idx += 1

            # Sentence-level overlap: carry over the last sentence if under limit
            if current_sentences and len(current_sentences[-1]) <= DEFAULT_MAX_OVERLAP_SIZE:
                last_s = current_sentences[-1]
                current_sentences = [last_s, s]
                current_length = len(last_s) + 1 + s_len
            else:
                current_sentences = [s]
                current_length = s_len
        else:
            current_sentences.append(s)
            current_length += s_len + space_overhead

    # Flush any remaining sentences
    if current_sentences:
        chunk_content = " ".join(current_sentences).strip()
        if chunk_content:
            cid = f"{doc_stem}_p{page_num:02d}_c{chunk_idx:03d}"
            chunks.append({
                "chunk_id": cid,
                "id": cid,
                "source": source,
                "page": page_num,
                "text": chunk_content,
            })

    return chunks


def process_chunks(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    target_size: int = DEFAULT_TARGET_CHUNK_SIZE,
    max_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[dict[str, Any]] | None:
    """
    Reads extracted page records, applies text normalization and boundary-aware
    chunking, and saves the resulting chunks to JSON.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input file existence
    if not input_path.is_file():
        if (PROJECT_ROOT / input_path).is_file():
            input_path = PROJECT_ROOT / input_path
        else:
            print(f"Error: Input file not found at '{input_path}'.", file=sys.stderr)
            return None

    # Load and parse JSON
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            pages_data = json.load(f)
    except json.JSONDecodeError as err:
        print(f"Error: Failed to parse JSON from '{input_path}': {err}", file=sys.stderr)
        return None
    except Exception as err:
        print(f"Error: Could not read file '{input_path}': {err}", file=sys.stderr)
        return None

    if not isinstance(pages_data, list):
        print(f"Error: Expected JSON list, got {type(pages_data).__name__}.", file=sys.stderr)
        return None

    if not pages_data:
        print(f"Error: No pages found in '{input_path}'.", file=sys.stderr)
        return None

    all_chunks: list[dict[str, Any]] = []
    valid_pages = 0

    for record in pages_data:
        if not isinstance(record, dict):
            continue

        source = record.get("source", "unknown")
        page_num = record.get("page", 0)
        page_text = record.get("text", "")

        page_chunks = chunk_page_text(
            text=page_text,
            source=source,
            page_num=page_num,
            target_size=target_size,
            max_size=max_size,
        )
        all_chunks.extend(page_chunks)
        valid_pages += 1

    # Calculate statistics
    total_chunks = len(all_chunks)
    avg_len = (
        sum(len(c["text"]) for c in all_chunks) / total_chunks
        if total_chunks > 0
        else 0.0
    )

    # Save to JSON
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved boundary-aware chunks to '{output_path}'.")
    except Exception as err:
        print(f"Error saving chunks to '{output_path}': {err}", file=sys.stderr)
        return None

    print(f"Input pages: {valid_pages}")
    print(f"Output chunks: {total_chunks}")
    print(f"Average chunk length: {avg_len:.2f}")

    return all_chunks


if __name__ == "__main__":
    result = process_chunks()
    if result is None:
        sys.exit(1)
