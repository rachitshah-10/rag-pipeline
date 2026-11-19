"""ChromaDB-backed vector store for policy chunks and metadata."""

from dataclasses import dataclass
from pathlib import Path

import chromadb

from src.chunking import DocumentChunk
from src.embeddings import Embedder


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved policy chunk paired with its source metadata and similarity distance."""

    chunk_id: str
    text: str
    doc_name: str
    section: str
    distance: float


class PolicyVectorStore:
    """Vector store wrapper supporting in-memory testing and persistent local storage."""

    def __init__(
        self,
        embedder: Embedder,
        collection_name: str = "policy_chunks",
        persist_dir: Path | None = None,
        reset: bool = False,
    ) -> None:
        """Initialize a ChromaDB collection with an injected Embedder."""
        self.embedder = embedder
        self.collection_name = collection_name

        # Use EphemeralClient for isolated tests and PersistentClient for local CLI runs.
        if persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))

        existing = [col.name for col in self._client.list_collections()]
        if reset and collection_name in existing:
            self._client.delete_collection(name=collection_name)

        self.collection = self._client.get_or_create_collection(name=collection_name)

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and upsert document chunks with metadata into ChromaDB."""
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        # Compute explicit embeddings via our Embedder so ChromaDB never downloads default ONNX weights.
        vectors = self.embedder.embed(texts)
        metadatas = [
            {
                "doc_name": chunk.doc_name,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas,
        )
        return len(chunks)

    def count(self) -> int:
        """Return the number of chunks currently stored in the collection."""
        return self.collection.count()

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Embed a query string and retrieve the top_k closest policy chunks."""
        total = self.count()
        if total == 0 or top_k <= 0:
            return []

        query_vector = self.embedder.embed([query])[0]
        raw = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievedChunk] = []
        for chunk_id, doc_text, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc_text,
                    doc_name=str(meta.get("doc_name", "")),
                    section=str(meta.get("section", "")),
                    distance=float(dist),
                )
            )
        return results
