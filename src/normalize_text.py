"""
normalize_text.py - Text Cleaning and Normalization for ResearchLens.

This module cleans extracted PDF text before chunking:
1. Normalizes Unicode ligatures and full-width/special characters via NFKC.
2. De-hyphenates words broken across line wraps (e.g. 'com- \n putational' -> 'computational')
   while preserving legitimate compound prefixes (e.g. 'multi-head', 'cross-channel', 'U-Net').
3. Strips arXiv watermark stamps and standalone page numbers.
4. Normalizes whitespace and reconstructs paragraph breaks.
5. Preserves technical terminology, equations, numbers, and punctuation.
"""

from pathlib import Path
import json
import re
import sys
import unicodedata

# Common compound prefixes in scientific and machine learning literature
# that should retain their hyphen when wrapped across lines.
COMPOUND_PREFIXES = {
    "multi", "cross", "high", "large", "single", "self", "co", "non",
    "well", "state", "dual", "real", "pixel", "scale", "feed", "end",
    "to", "of", "pre", "sub", "re", "post", "zero", "few"
}


def dehyphenate_match(match: re.Match) -> str:
    """
    Decides whether a hyphenated line wrap should be joined as a single word
    or kept as a hyphenated compound word.
    """
    prefix = match.group(1)
    suffix = match.group(2)

    # If the prefix is a single capital letter (e.g., U-Net) or a known compound prefix
    if len(prefix) == 1 or prefix.lower() in COMPOUND_PREFIXES:
        return f"{prefix}-{suffix}"

    # Otherwise, it is a single word broken across lines (e.g. 'restora-\ntion' -> 'restoration')
    return f"{prefix}{suffix}"


def clean_page_text(text: str, page_num: int | None = None) -> str:
    """
    Applies comprehensive, non-destructive text cleaning to extracted page text.

    Args:
        text: Raw text string from a single PDF page.
        page_num: 1-based page number (optional, used to verify footer page numbers).

    Returns:
        Cleaned, normalized text string.
    """
    if not text or not text.strip():
        return ""

    # Step 1: Unicode NFKC normalization (converts ligatures like \ufb01 -> 'fi', \ufb02 -> 'fl')
    cleaned = unicodedata.normalize("NFKC", text)

    # Step 2: Remove arXiv distribution header / watermark stamps
    # Matches patterns like: 'arXiv:2111.09881v2  [cs.CV]  11 Mar 2022'
    cleaned = re.sub(
        r"arXiv:\d+\.\d+(?:v\d+)?\s*\[.*?\]\s*\d+\s*[A-Za-z]+\s*\d+",
        "",
        cleaned,
    )

    # Step 3: Join words split across line wraps with a hyphen
    cleaned = re.sub(r"(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)", dehyphenate_match, cleaned)

    # Step 4: Line-by-line filtering for standalone page numbers and clean spacing
    lines = cleaned.split("\n")
    processed_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Remove standalone page numbers (e.g. line containing only '1' or '2')
        if stripped.isdigit():
            if page_num is None or int(stripped) == page_num or len(stripped) <= 3:
                continue

        processed_lines.append(stripped)

    # Step 5: Paragraph reconstruction
    # In multi-column PDFs, lines within a paragraph are broken by single newlines.
    # We join lines into cohesive paragraphs while preserving headings, lists, and captions.
    paragraphs: list[str] = []
    current_para: list[str] = []

    def is_structural_boundary(l: str) -> bool:
        """Identifies headings, bullet points, table/figure captions, or section titles."""
        if not l:
            return False
        # Headings like '1. Introduction', '3.1. MDTA', 'Figure 2.', 'Table 1.'
        if re.match(r"^(?:\d+\.|\d+\.\d+\.?|[A-Z][a-zA-Z\s]{2,30}$|•|\([a-z]\)|Figure\s+\d+|Table\s+\d+)", l):
            return True
        lower_l = l.lower()
        if lower_l in {
            "abstract", "introduction", "related work", "proposed method",
            "experimental results", "conclusion", "acknowledgements.", "references"
        }:
            return True
        return False

    for line in processed_lines:
        if not line:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            continue

        if is_structural_boundary(line):
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            current_para.append(line)
            # If it's a short heading, flush it as its own block
            if len(line) < 45 and not line.endswith((".", ":", ",")):
                paragraphs.append(" ".join(current_para))
                current_para = []
            continue

        # If previous line was short and ended in terminal punctuation, flush paragraph
        if current_para:
            prev = current_para[-1]
            if (len(prev) < 45 and prev.endswith((".", ":", "!", "?"))) or is_structural_boundary(prev):
                paragraphs.append(" ".join(current_para))
                current_para = []

        current_para.append(line)

    if current_para:
        paragraphs.append(" ".join(current_para))

    # Join paragraphs with double newlines
    result = "\n\n".join(p for p in paragraphs if p.strip())

    # Step 6: Collapse 3+ newlines to 2, and collapse horizontal spaces
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def normalize_pages_file(input_file: Path, output_file: Path | None = None) -> list[dict]:
    """
    Loads extracted pages JSON, cleans each page text, and optionally saves the output.
    """
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        pages = json.load(f)

    if not isinstance(pages, list):
        raise ValueError(f"Expected a list of page records, got {type(pages).__name__}")

    cleaned_pages = []
    for record in pages:
        page_num = record.get("page")
        raw_text = record.get("text", "")
        cleaned_text = clean_page_text(raw_text, page_num=page_num)
        cleaned_pages.append({
            "source": record.get("source", "unknown"),
            "page": page_num,
            "text": cleaned_text,
        })

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_pages, f, indent=2, ensure_ascii=False)

    return cleaned_pages


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    in_path = project_root / "data" / "processed" / "restormer_pages.json"
    out_path = project_root / "data" / "processed" / "restormer_pages_cleaned.json"

    try:
        pages = normalize_pages_file(in_path, out_path)
        print(f"Successfully normalized {len(pages)} pages.")
        print(f"Saved cleaned pages to '{out_path}'.")
    except Exception as err:
        print(f"Error during normalization: {err}", file=sys.stderr)
        sys.exit(1)
