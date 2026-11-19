"""Embedding interface and SentenceTransformer implementation."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Interface for text embedding providers."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into float embedding vectors."""
        ...


class SentenceTransformerEmbedder:
    """Local embedder backed by a SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Store the model name without downloading weights yet."""
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazily load and cache the SentenceTransformer model on first use."""
        if self._model is None:
            # Delay import so offline tests using FakeEmbedder avoid torch startup overhead.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode input texts into dense float vectors."""
        if not texts:
            return []
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True).tolist()
