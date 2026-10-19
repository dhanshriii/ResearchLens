"""Persistent semantic vector search for ResearchLens."""

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_DB_PATH = Path(
    os.getenv(
        "RESEARCHLENS_VECTOR_DB",
        str(PROJECT_ROOT / "data" / "chroma_db"),
    )
)

COLLECTION_NAME = "researchlens_documents"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _make_id(chunk: dict[str, Any]) -> str:
    """Create a stable ID for a chunk."""
    if chunk.get("chunk_id"):
        return str(chunk["chunk_id"])

    source = str(chunk.get("source", "unknown"))
    page = str(chunk.get("page", 0))
    text = str(chunk.get("text", ""))

    fingerprint = hashlib.sha256(
        f"{source}|{page}|{text}".encode("utf-8")
    ).hexdigest()

    return fingerprint


def _metadata_value(value: Any) -> str | int | float | bool:
    """Convert metadata to a ChromaDB-compatible scalar."""
    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


class ResearchLensVectorStore:
    """Embed, persist, and semantically search document chunks."""

    def __init__(
        self,
        persist_directory: str | Path = VECTOR_DB_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> int:
        """Embed and store chunks, preserving their metadata."""
        valid_chunks = []

        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()

            if not text:
                continue

            metadata = {
                key: _metadata_value(value)
                for key, value in chunk.items()
                if key != "text"
            }

            metadata.setdefault("source", "unknown")
            metadata.setdefault("page", 0)

            valid_chunks.append(
                {
                    "id": _make_id(chunk),
                    "text": text,
                    "metadata": metadata,
                }
            )

        if not valid_chunks:
            return 0

        inserted = 0

        for start in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[start : start + batch_size]

            texts = [item["text"] for item in batch]

            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            self.collection.upsert(
                ids=[item["id"] for item in batch],
                documents=texts,
                metadatas=[item["metadata"] for item in batch],
                embeddings=embeddings.tolist(),
            )

            inserted += len(batch)

        return inserted

    def search(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most semantically relevant chunks."""
        query = query.strip()

        if not query:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.collection.count() == 0:
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_filter = None

        if source is not None:
            query_filter = {"source": source}

        result = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, self.collection.count()),
            where=query_filter,
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": metadata.get("source", "unknown"),
                    "page": metadata.get("page", 0),
                    "metadata": metadata,
                    "distance": float(distance),
                    "similarity": 1.0 - float(distance),
                }
            )

        return matches

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()
