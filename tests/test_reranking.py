"""Offline unit tests for cross-encoder reranking and two-stage RAGPipeline retrieval."""

import math

from src.chunking import DocumentChunk
from src.rag_pipeline import RAGPipeline
from src.reranker import CrossEncoderReranker, Reranker
from src.vector_store import PolicyVectorStore, RetrievedChunk


class FakeReranker:
    """Offline reranker scoring chunks by exact phrase/term overlap with the query."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 3,
    ) -> list[tuple[RetrievedChunk, float]]:
        """Score each chunk by shared query terms and return the top_k descending."""
        if not chunks or top_k <= 0:
            return []
        query_words = set(query.lower().split())
        scored = [
            (chunk, float(sum(1 for w in query_words if w in chunk.text.lower())))
            for chunk in chunks
        ]
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return scored[:top_k]


class FakeUniformEmbedder:
    """Offline embedder returning nearly identical vectors so initial retrieval needs reranking."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return unit vectors with minimal separation for testing candidate reranking."""
        vectors: list[list[float]] = []
        for idx, _ in enumerate(texts):
            raw = [1.0, 0.01 * (idx + 1)]
            norm = math.sqrt(sum(v * v for v in raw))
            vectors.append([v / norm for v in raw])
        return vectors


class FakeLLM:
    """Offline LLM stub recording the prompt sent after reranking."""

    def __init__(self) -> None:
        """Initialize captured prompt list."""
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        """Store prompt and return a static answer."""
        self.prompts.append(prompt)
        return "Parental leave is 16 weeks."


def test_reranker_protocol_and_cross_encoder_lazy_init() -> None:
    """Verify FakeReranker and CrossEncoderReranker satisfy Reranker without downloading weights."""
    assert isinstance(FakeReranker(), Reranker)
    reranker = CrossEncoderReranker()
    assert isinstance(reranker, Reranker)
    assert reranker._model is None


def test_reranker_handles_empty_or_zero_top_k() -> None:
    """Confirm reranker returns an empty list for empty candidates or non-positive top_k."""
    reranker = FakeReranker()
    sample = [RetrievedChunk("c1", "Some text", "doc.md", "Sec 1", 0.1)]
    assert reranker.rerank("query", [], top_k=3) == []
    assert reranker.rerank("query", sample, top_k=0) == []


def test_reranker_promotes_best_candidate_to_rank_one() -> None:
    """Confirm reranking reorders candidates and slices to top_k."""
    candidates = [
        RetrievedChunk("c1", "Travel meal per diem is $75 per day.", "exp.md", "Sec 2.2", 0.1),
        RetrievedChunk("c2", "Employees receive 16 weeks of paid parental leave.", "pto.md", "Sec 3.4", 0.4),
        RetrievedChunk("c3", "Carry over up to 5 unused PTO days.", "pto.md", "Sec 3.2", 0.2),
    ]

    ranked = FakeReranker().rerank("paid parental leave weeks", candidates, top_k=2)

    assert len(ranked) == 2
    assert ranked[0][0].chunk_id == "c2"
    assert ranked[0][1] > ranked[1][1]


def test_rag_pipeline_uses_reranker_to_select_final_top_k() -> None:
    """Verify RAGPipeline retrieves candidate_k chunks and uses reranker to pick top_k."""
    store = PolicyVectorStore(FakeUniformEmbedder(), collection_name="test_rerank_e2e", reset=True)
    store.index_chunks(
        [
            DocumentChunk("c1", "Domestic meal per diem is $75.", "exp.md", "Sec 2.2", 0),
            DocumentChunk("c2", "Carry over max 5 PTO days.", "pto.md", "Sec 3.2", 0),
            DocumentChunk("c3", "Paid parental leave is 16 weeks for full-time staff.", "pto.md", "Sec 3.4", 0),
        ]
    )

    llm = FakeLLM()
    pipeline = RAGPipeline(
        vector_store=store,
        llm_client=llm,
        use_hybrid=False,
        reranker=FakeReranker(),
        candidate_k=3,
    )
    result = pipeline.answer("How many weeks of paid parental leave?", top_k=1)

    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0].chunk_id == "c3"
    assert "16 weeks" in llm.prompts[0]
