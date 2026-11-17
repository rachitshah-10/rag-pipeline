"""Embedding interface and SentenceTransformer implementation."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Protocol defining the text embedding interface used across the pipeline."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of text strings.

        Args:
            texts: Input strings to encode.

        Returns:
            List of float embedding vectors, one per input string.
        """
        ...


class SentenceTransformerEmbedder:
    """Local embedding generator backed by a SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedder configuration without eagerly downloading weights.

        Args:
            model_name: Hugging Face / SentenceTransformers model identifier.
        """
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazily load and cache the SentenceTransformer model on first use.

        Returns:
            Loaded SentenceTransformer model instance.
        """
        if self._model is None:
            # Import lazily so offline unit tests using FakeEmbedder avoid torch/model overhead.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode input texts into dense float vectors.

        Args:
            texts: Input strings to encode.

        Returns:
            List of embedding vectors as standard Python float lists.
        """
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
