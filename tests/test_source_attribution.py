"""Offline unit tests for source metadata attribution on generated answers."""

import math

from src.chunking import DocumentChunk
from src.rag_pipeline import RAGPipeline, format_answer_with_sources, format_sources
from src.vector_store import PolicyVectorStore, RetrievedChunk


class FakeEmbedder:
    """Minimal offline embedder grouping meal and PTO keywords."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode each text onto simple topic axes."""
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            meal = 1.0 if "meal" in lowered else 0.0
            pto = 1.0 if "pto" in lowered else 0.0
            raw = [meal, pto, 0.1]
            norm = math.sqrt(sum(v * v for v in raw))
            vectors.append([v / norm for v in raw])
        return vectors


class FakeLLM:
    """Offline LLM stub returning a static answer."""

    def generate(self, prompt: str) -> str:
        """Return a fixed policy answer."""
        return "Domestic meals are reimbursed up to $75/day."


def test_format_sources_deduplicates_preserving_retrieval_order() -> None:
    """Confirm format_sources deduplicates (doc_name, section) pairs while keeping rank order."""
    chunks = [
        RetrievedChunk("c1", "text 1", "expense_policy.md", "Section 2.2: Meal Per Diem", 0.1),
        RetrievedChunk("c2", "text 2", "expense_policy.md", "Section 2.2: Meal Per Diem", 0.2),
        RetrievedChunk("c3", "text 3", "expense_policy.md", "Section 2.5: Receipts", 0.3),
    ]

    sources = format_sources(chunks)

    assert sources == [
        "expense_policy.md | Section 2.2: Meal Per Diem",
        "expense_policy.md | Section 2.5: Receipts",
    ]


def test_format_sources_handles_empty_list() -> None:
    """Confirm format_sources returns an empty list when no chunks were retrieved."""
    assert format_sources([]) == []


def test_rag_pipeline_surfaces_cited_sources_with_answer() -> None:
    """Verify RAGPipeline.answer includes deduplicated sources matching retrieved chunks."""
    store = PolicyVectorStore(FakeEmbedder(), collection_name="test_attribution", reset=True)
    store.index_chunks(
        [
            DocumentChunk("exp::1", "Domestic meal per diem is $75/day.", "expense_policy.md", "Section 2.2: Meal Per Diem", 0),
            DocumentChunk("pto::1", "Max 5 unused PTO days carry over.", "pto_policy.md", "Section 3.2: Carryover", 0),
        ]
    )

    pipeline = RAGPipeline(vector_store=store, llm_client=FakeLLM(), use_hybrid=True)
    result = pipeline.answer("What is the meal per diem?", top_k=1)

    assert result["sources"] == ["expense_policy.md | Section 2.2: Meal Per Diem"]
    rendered = format_answer_with_sources(result["answer"], result["sources"])
    assert "Domestic meals are reimbursed up to $75/day." in rendered
    assert "expense_policy.md | Section 2.2: Meal Per Diem" in rendered
