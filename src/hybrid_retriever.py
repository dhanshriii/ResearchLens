"""
ResearchLens hybrid retrieval.

Pipeline:
1. BM25 keyword retrieval.
2. ChromaDB semantic retrieval.
3. Reciprocal Rank Fusion (RRF).
4. Cross-encoder reranking.
"""

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from src.retriever import (
    DEFAULT_CHUNKS_PATH,
    RERANKER_NAME,
    BM25,
    load_chunks,
)
from src.vector_store import ResearchLensVectorStore


class HybridRetriever:
    """Combine keyword and semantic retrieval before reranking."""

    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        vector_db_path: str | Path | None = None,
        candidate_k: int = 50,
        final_k: int = 5,
        bm25_k: int = 50,
        vector_k: int = 50,
        rrf_k: int = 60,
        device: str = "cpu",
        collection_name: str = "researchlens_hybrid",
    ) -> None:
        if min(candidate_k, final_k, bm25_k, vector_k, rrf_k) < 1:
            raise ValueError("All retrieval parameters must be positive.")

        self.chunks_path = Path(chunks_path)
        self.chunks = load_chunks(self.chunks_path)

        self.candidate_k = candidate_k
        self.final_k = final_k
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.rrf_k = rrf_k

        self.bm25 = BM25([chunk["text"] for chunk in self.chunks])

        if vector_db_path is None:
            vector_db_path = (
                self.chunks_path.parent.parent
                / "chroma_db"
            )

        self.vector_store = ResearchLensVectorStore(
            persist_directory=vector_db_path,
            collection_name=collection_name,
        )

        # Synchronize this collection with the selected chunk file.
        # A stable ID allows repeated indexing without duplicate records.
        self.vector_store.add_chunks(self.chunks)

        print(f"Loaded {len(self.chunks)} document chunks.")
        print("Loading cross-encoder on CPU...")

        self.reranker = CrossEncoder(
            RERANKER_NAME,
            device=device,
        )

        print("Hybrid retriever ready.")

    def _bm25_ranking(self, question: str) -> list[int]:
        """Return chunk indexes ordered by BM25 relevance."""
        scores = self.bm25.scores(question)

        return np.argsort(
            -scores,
            kind="stable",
        )[: min(self.bm25_k, len(self.chunks))].tolist()

    def _vector_ranking(self, question: str) -> list[int]:
        """Map semantic results back to the loaded chunk indexes."""
        results = self.vector_store.search(
            question,
            top_k=self.vector_k,
        )

        index_by_id = {}

        for index, chunk in enumerate(self.chunks):
            chunk_id = (
                chunk.get("chunk_id")
                or chunk.get("id")
            )

            if chunk_id:
                index_by_id[str(chunk_id)] = index

        # The vector store generates a hash when a chunk has no ID.
        # Reuse its ID-generation rule for those chunks.
        from src.vector_store import _make_id

        for index, chunk in enumerate(self.chunks):
            index_by_id.setdefault(_make_id(chunk), index)

        ranking = []

        for result in results:
            index = index_by_id.get(str(result["chunk_id"]))

            if index is not None and index not in ranking:
                ranking.append(index)

        return ranking

    def _fuse_rankings(
        self,
        bm25_ranking: list[int],
        vector_ranking: list[int],
    ) -> list[tuple[int, float]]:
        """Merge rankings using Reciprocal Rank Fusion."""
        fused_scores: dict[int, float] = {}

        for ranking in (bm25_ranking, vector_ranking):
            for rank, index in enumerate(ranking, start=1):
                fused_scores[index] = (
                    fused_scores.get(index, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

        return sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[: min(self.candidate_k, len(fused_scores))]

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        """Return reranked chunks using the existing output interface."""
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        bm25_ranking = self._bm25_ranking(question)
        vector_ranking = self._vector_ranking(question)

        fused_candidates = self._fuse_rankings(
            bm25_ranking,
            vector_ranking,
        )

        if not fused_candidates:
            return []

        candidate_indices = [
            index for index, _ in fused_candidates
        ]

        pairs = [
            (question, self.chunks[index]["text"])
            for index in candidate_indices
        ]

        reranker_scores = np.asarray(
            self.reranker.predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
            )
        ).reshape(-1)

        order = np.argsort(
            -reranker_scores,
            kind="stable",
        )[: min(self.final_k, len(candidate_indices))]

        bm25_scores = self.bm25.scores(question)

        results = []

        for rank, position in enumerate(order, start=1):
            position = int(position)
            index = candidate_indices[position]
            chunk = self.chunks[index]

            results.append(
                {
                    "rank": rank,
                    "reranker_score": float(reranker_scores[position]),
                    "bm25_score": float(bm25_scores[index]),
                    "rrf_score": float(
                        fused_candidates[position][1]
                    ),
                    "source": chunk.get("source", "unknown"),
                    "page": chunk["page"],
                    "chunk_id": (
                        chunk.get("chunk_id")
                        or chunk.get("id", f"chunk-{index}")
                    ),
                    "text": chunk["text"],
                }
            )

        return results
