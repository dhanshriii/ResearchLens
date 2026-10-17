import hashlib
import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from retriever import ResearchPaperRetriever
from answer_generator import answer_question
from extract_pdf import pymupdf
from chunk_text import chunk_page_text


st.set_page_config(
    page_title="ResearchLens",
    page_icon="📚",
    layout="wide",
)

st.title("ResearchLens")
st.caption("An evidence-grounded assistant for multiple research documents.")

with st.sidebar:
    st.header("Research workspace")
    st.markdown("**Retrieval:** BM25 + Cross-Encoder")
    st.markdown("**Answer generation:** Gemini")
    st.info(
        "Upload multiple PDFs to ask questions across your documents. "
        "Answers use retrieved evidence and include source-page citations."
    )

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    st.warning(
        "Gemini API key not found. Set GEMINI_API_KEY before starting the app."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def process_uploaded_pdf(pdf_bytes: bytes, filename: str):
    """Extract and chunk one PDF while preserving source and page metadata."""
    pages = []

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text()

                if text.strip():
                    pages.append({
                        "source": filename,
                        "page": page_number,
                        "text": text,
                    })
    except Exception as exc:
        raise ValueError(f"Could not read '{filename}': {exc}") from exc

    if not pages:
        raise ValueError(
            f"'{filename}' contains no selectable text. "
            "Scanned PDFs require OCR, which is not enabled yet."
        )

    chunks = []

    for page in pages:
        chunks.extend(
            chunk_page_text(
                text=page["text"],
                source=filename,
                page_num=page["page"],
            )
        )

    if not chunks:
        raise ValueError(
            f"No usable text chunks could be created from '{filename}'."
        )

    return chunks, len(pages)


@st.cache_resource(show_spinner="Building the combined retrieval system...")
def load_retriever(chunks_json: str, document_id: str):
    """Build one retriever over all selected documents."""
    cache_dir = Path("/tmp/researchlens_uploads")
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = cache_dir / f"{document_id}_chunks.json"
    chunks_path.write_text(chunks_json, encoding="utf-8")

    return ResearchPaperRetriever(
        chunks_path=chunks_path,
        candidate_k=50,
        final_k=5,
        device="cpu",
    )


uploaded_files = st.file_uploader(
    "Upload research papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select multiple PDFs. ResearchLens will search across all selected documents.",
)

if not uploaded_files:
    st.info("Upload one or more PDFs to begin.")
    st.stop()


all_chunks = []
document_summaries = []
document_errors = []

# Process each document independently, then combine its chunks for retrieval.
for uploaded_file in uploaded_files:
    filename = Path(uploaded_file.name).name
    pdf_bytes = uploaded_file.getvalue()

    try:
        with st.spinner(f"Processing {filename}..."):
            chunks, page_count = process_uploaded_pdf(pdf_bytes, filename)

        all_chunks.extend(chunks)
        document_summaries.append({
            "name": filename,
            "pages": page_count,
            "chunks": len(chunks),
            "hash": hashlib.sha256(pdf_bytes).hexdigest(),
        })

    except Exception as exc:
        document_errors.append(str(exc))

for error in document_errors:
    st.error(error)

if not all_chunks:
    st.warning("No usable documents are available. Upload a text-based PDF.")
    st.stop()

# Identify the selected collection by document contents and filenames.
fingerprint_data = sorted(
    (doc["name"], doc["hash"]) for doc in document_summaries
)
document_id = hashlib.sha256(
    json.dumps(fingerprint_data, ensure_ascii=False).encode("utf-8")
).hexdigest()

chunks_json = json.dumps(all_chunks, ensure_ascii=False)

try:
    retriever = load_retriever(chunks_json, document_id)
except Exception as exc:
    st.error(f"Could not build the retrieval system: {exc}")
    st.stop()

st.success(
    f"{len(document_summaries)} document(s) ready | "
    f"{sum(doc['pages'] for doc in document_summaries)} pages with text | "
    f"{len(all_chunks)} total chunks"
)

with st.expander("View selected documents", expanded=True):
    for doc in document_summaries:
        st.markdown(
            f"- **{doc['name']}** — "
            f"{doc['pages']} pages with text, {doc['chunks']} chunks"
        )

# Clear chat history when the selected document collection changes.
if st.session_state.get("active_document_id") != document_id:
    st.session_state.active_document_id = document_id
    st.session_state.messages = []

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for index, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**Source {index}** — "
                        f"{source['source']}, page {source['page']}"
                    )
                    st.caption(f"Chunk: {source['chunk_id']}")

question = st.chat_input(
    "Ask a question across your uploaded documents..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                result = answer_question(retriever, question)

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            if sources:
                with st.expander("View retrieved sources"):
                    for index, source in enumerate(sources, start=1):
                        st.markdown(
                            f"**Source {index}** — "
                            f"{source['source']}, page {source['page']}"
                        )
                        st.caption(f"Chunk: {source['chunk_id']}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as exc:
            st.error(f"Unable to answer this question: {exc}")
