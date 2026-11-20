"""Cross-encoder reranking step to refine initial retrieval candidates."""

from typing import Protocol, runtime_checkable

from src.vector_store import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Interface for reranking retrieved chunks against a query."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 3,
    ) -> list[tuple[RetrievedChunk, float]]:
        """Score and return the top_k most relevant chunks in descending order."""
        ...


class CrossEncoderReranker:
    """Local reranker using a SentenceTransformers CrossEncoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Store the CrossEncoder model name without downloading weights yet."""
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazily load and cache the CrossEncoder model on first use."""
        if self._model is None:
            # Delay import so offline unit tests using FakeReranker avoid torch/model overhead.
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 3,
    ) -> list[tuple[RetrievedChunk, float]]:
        """Jointly score each (query, chunk) pair and return the top_k highest-scoring chunks."""
        if not chunks or top_k <= 0:
            return []

        model = self._get_model()
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = model.predict(pairs)

        ranked = sorted(
            ((chunk, float(score)) for chunk, score in zip(chunks, scores)),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        return ranked[:top_k]
