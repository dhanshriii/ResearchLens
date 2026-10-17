"""
ResearchLens: BM25 retrieval followed by cross-encoder reranking.

The BM25 implementation uses the same tokenization and scoring formula
as our corrected retrieval evaluation.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder


PROJECT_ROOT = Path.home() / "ResearchLens"
DEFAULT_CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "restormer_chunks.json"
)

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text: str) -> list[str]:
    """Tokenize text using the evaluation pipeline's original method."""
    return re.findall(r"\b[a-z0-9]+\b", (text or "").lower())


def load_chunks(path: Path = DEFAULT_CHUNKS_PATH) -> list[dict[str, Any]]:
    """Load and validate the processed paper chunks."""
    if not path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("The chunk file must contain a non-empty list.")

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not chunk.get("text", "").strip():
            raise ValueError(f"Chunk {index} is invalid or has empty text.")

        if "page" not in chunk:
            raise ValueError(f"Chunk {index} has no page metadata.")

    return chunks


class BM25:
    """Small in-memory BM25 index for a research paper."""

    def __init__(
        self,
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not documents:
            raise ValueError("Cannot build BM25 over an empty document list.")

        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokens = [tokenize(doc) for doc in documents]
        self.lengths = np.asarray(
            [len(tokens) for tokens in self.tokens],
            dtype=np.float32,
        )
        self.avgdl = float(self.lengths.mean()) or 1.0

        document_frequency: dict[str, int] = {}

        for tokens in self.tokens:
            for term in set(tokens):
                document_frequency[term] = (
                    document_frequency.get(term, 0) + 1
                )

        total_documents = len(documents)

        self.idf = {
            term: np.log(
                1.0
                + (total_documents - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

        self.term_frequencies = []

        for tokens in self.tokens:
            frequencies: dict[str, int] = {}
            for term in tokens:
                frequencies[term] = frequencies.get(term, 0) + 1
            self.term_frequencies.append(frequencies)

    def scores(self, query: str) -> np.ndarray:
        """Calculate the BM25 score for each document."""
        output = np.zeros(len(self.documents), dtype=np.float64)

        for term in tokenize(query):
            inverse_document_frequency = self.idf.get(term, 0.0)

            if not inverse_document_frequency:
                continue

            for index, frequencies in enumerate(self.term_frequencies):
                tf = frequencies.get(term, 0)

                if not tf:
                    continue

                document_length = float(self.lengths[index])

                denominator = tf + self.k1 * (
                    1 - self.b
                    + self.b * document_length / self.avgdl
                )

                output[index] += (
                    inverse_document_frequency
                    * tf
                    * (self.k1 + 1)
                    / denominator
                )

        return output


class ResearchPaperRetriever:
    """Retrieve BM25 candidates and rerank them with a cross-encoder."""

    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        candidate_k: int = 50,
        final_k: int = 5,
        device: str = "cpu",
    ):
        if candidate_k < 1 or final_k < 1:
            raise ValueError("candidate_k and final_k must be positive.")

        self.chunks = load_chunks(chunks_path)
        self.candidate_k = candidate_k
        self.final_k = final_k

        self.bm25 = BM25([chunk["text"] for chunk in self.chunks])

        print(f"Loaded {len(self.chunks)} paper chunks.")
        print("Loading cross-encoder on CPU...")

        self.reranker = CrossEncoder(
            RERANKER_NAME,
            device=device,
        )

        print("Retriever ready.")

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Return the highest-ranked chunks with page and source metadata."""
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        scores = self.bm25.scores(question)
        candidate_indices = np.argsort(
            -scores,
            kind="stable",
        )[:min(self.candidate_k, len(self.chunks))].tolist()

        pairs = [
            (question, self.chunks[index]["text"])
            for index in candidate_indices
        ]

        reranker_scores = self.reranker.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        order = np.argsort(
            -np.asarray(reranker_scores),
            kind="stable",
        )[:min(self.final_k, len(candidate_indices))]

        results = []

        for rank, position in enumerate(order, start=1):
            index = candidate_indices[int(position)]
            chunk = self.chunks[index]

            results.append({
                "rank": rank,
                "reranker_score": float(reranker_scores[int(position)]),
                "bm25_score": float(scores[index]),
                "source": chunk.get("source", "unknown"),
                "page": chunk["page"],
                "chunk_id": (
                    chunk.get("chunk_id")
                    or chunk.get("id", f"chunk-{index}")
                ),
                "text": chunk["text"],
            })

        return results


if __name__ == "__main__":
    retriever = ResearchPaperRetriever()

    while True:
        try:
            question = input("\nAsk about the Restormer paper (or type exit): ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.strip().lower() in {"exit", "quit"}:
            break

        if not question.strip():
            print("Please enter a question.")
            continue

        try:
            results = retriever.retrieve(question)

            for result in results:
                print(
                    f"\nRank {result['rank']} | "
                    f"Page {result['page']} | "
                    f"Reranker score {result['reranker_score']:.4f}"
                )
                print(f"Chunk: {result['chunk_id']}")
                print(result["text"])
                print("-" * 60)

        except Exception as error:
            print(f"Retrieval failed: {error}")
