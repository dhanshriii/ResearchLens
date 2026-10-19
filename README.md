# ResearchLens

**An evidence-grounded assistant for research documents.**

ResearchLens lets users upload PDF documents, generate summaries, and ask questions using retrieved passages as evidence.

## Features

- Upload and process multiple PDF documents.
- Extract text and preserve source-page metadata.
- Split text into overlapping chunks.
- Hybrid retrieval using BM25 and ChromaDB semantic search.
- Reciprocal Rank Fusion (RRF) and cross-encoder reranking.
- Gemini-powered question answering with source citations.
- Hierarchical document summarization with checkpoints.
- Expandable retrieved passages for evidence inspection.

## Architecture

1. PyMuPDF extracts text from uploaded PDFs.
2. Text is split into chunks with source and page metadata.
3. BM25 and vector search retrieve candidate passages.
4. RRF combines keyword and semantic rankings.
5. A cross-encoder reranks the candidates.
6. Gemini generates answers using retrieved evidence.
7. The summarizer processes chunks in groups and synthesizes a final summary.

## Requirements

- Python version compatible with the installed ML dependencies.
- A Gemini API key.
- Internet access for Gemini API requests and initial model downloads.

## Installation

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Set your Gemini API key in the terminal. Never commit the key to Git.

```bash
read -rsp "Gemini API key: " GEMINI_API_KEY
echo
export GEMINI_API_KEY
export GEMINI_MODEL="gemini-3.8-flash"
```

Use a Gemini model available to your account.

## Run the application

```bash
streamlit run app.py \
  --server.headless true \
  --server.fileWatcherType none \
  --server.address 127.0.0.1 \
  --server.port 8501
```

Open http://localhost:8501 in your browser.

Upload PDFs, generate a summary, and ask questions. Expand retrieved-source panels to inspect supporting passages.

## Run automated tests

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py src/*.py tests/*.py
```

## Limitations

- Generated answers and summaries can contain errors; verify important claims against the original documents.
- Citations identify retrieved passages but do not guarantee every claim is supported.
- Scanned PDFs may require OCR, which is not guaranteed by the current extraction pipeline.
- Initial model downloads and CPU-based reranking can take time.

## Project structure

```text
ResearchLens_Streamlit/
├── app.py
├── README.md
├── requirements.txt
├── src/
│   ├── answer_generator.py
│   ├── chunk_text.py
│   ├── extract_pdf.py
│   ├── hybrid_retriever.py
│   ├── normalize_text.py
│   ├── retriever.py
│   ├── summarizer.py
│   └── vector_store.py
└── tests/
    └── test_answer_generator.py
```
