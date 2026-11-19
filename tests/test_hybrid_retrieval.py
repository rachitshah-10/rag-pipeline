"""Offline unit tests for BM25 + vector hybrid retrieval."""

import math
from pathlib import Path
import pytest

from src.chunking import DocumentChunk, chunk_corpus
from src.hybrid_retrieval import bm25_scores, hybrid_search
from src.rag_pipeline import RAGPipeline
from src.vector_store import PolicyVectorStore


class FakeHeaderBiasedEmbedder:
    """Offline embedder mimicking dense retrieval favoring 'policy' headers over exact codes."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Project texts so generic policy headers beat code sections in vector-only search."""
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            header_score = 1.0 if ("policy id" in lowered or "policy" in lowered) else 0.0
            wifi_score = 2.0 if ("wi-fi" in lowered or "internet" in lowered) else 0.0
            code_query_score = 2.0 if ("exp-402" in lowered and "non-reimbursable" not in lowered) else 0.0
            expense_score = 0.5 if ("non-reimbursable" in lowered or "exp-402" in lowered) else 0.0
            raw = [header_score, wifi_score, code_query_score, expense_score, 0.2]
            norm = math.sqrt(sum(v * v for v in raw))
            vectors.append([v / norm for v in raw])
        return vectors


class FakeEchoLLM:
    """Minimal offline LLM stub for pipeline wiring tests."""

    def generate(self, prompt: str) -> str:
        """Return a fixed confirmation string."""
        return "Stubbed answer"


def test_bm25_scores_ranks_rare_policy_code_above_common_word() -> None:
    """Confirm BM25 naturally weights rare policy codes above common words across a corpus."""
    docs = [
        "# Expense Policy. Policy ID: POL-EXP-2026.",
        "# PTO Policy. Policy ID: POL-PTO-2026.",
        "# Remote Policy. Policy ID: POL-RW-2026.",
        "Under code EXP-402, stand-alone bar tabs and traffic fines are non-reimbursable.",
    ]

    scores = bm25_scores("What does policy EXP-402 say?", docs)
    assert len(scores) == 4
    assert scores[3] == 1.0
    assert scores[3] > scores[0]


def test_hybrid_search_beats_vector_only_on_exact_code_query() -> None:
    """Confirm hybrid_search puts EXP-402 at rank 1 when vector-only top-3 misses it."""
    chunks = chunk_corpus(Path(__file__).resolve().parents[1] / "data" / "policies")
    store = PolicyVectorStore(FakeHeaderBiasedEmbedder(), collection_name="test_hybrid", reset=True)
    store.index_chunks(chunks)

    query = "What does policy EXP-402 say?"
    vector_only = store.search(query, top_k=3)
    assert not any("EXP-402" in c.text for c in vector_only)

    hybrid_results = hybrid_search(store, query, top_k=3)
    assert len(hybrid_results) == 3
    assert "EXP-402" in hybrid_results[0][0].text
    assert hybrid_results[0][1] >= hybrid_results[1][1]


def test_hybrid_search_falls_back_to_vector_on_pure_semantic_query() -> None:
    """Verify queries with zero keyword overlap still rank via vector similarity."""
    store = PolicyVectorStore(FakeHeaderBiasedEmbedder(), collection_name="test_semantic", reset=True)
    store.index_chunks(
        [
            DocumentChunk("net::0", "Residential high-speed internet stipend is $75/month.", "rw.md", "Sec 1.4", 0),
            DocumentChunk("exp::0", "Alcohol-only bar tabs are non-reimbursable.", "exp.md", "Sec 2.6", 0),
        ]
    )

    results = hybrid_search(store, "wi-fi", top_k=2)
    assert len(results) == 2
    assert results[0][0].chunk_id == "net::0"


def test_hybrid_search_rejects_invalid_weights() -> None:
    """Verify hybrid_search raises ValueError on negative or all-zero weights."""
    store = PolicyVectorStore(FakeHeaderBiasedEmbedder(), collection_name="test_weights", reset=True)
    with pytest.raises(ValueError):
        hybrid_search(store, "query", vector_weight=-0.1, keyword_weight=1.0)
    with pytest.raises(ValueError):
        hybrid_search(store, "query", vector_weight=0.0, keyword_weight=0.0)


def test_rag_pipeline_uses_hybrid_search_when_enabled() -> None:
    """Verify RAGPipeline retrieves exact policy code via hybrid_search when use_hybrid=True."""
    chunks = chunk_corpus(Path(__file__).resolve().parents[1] / "data" / "policies")
    store = PolicyVectorStore(FakeHeaderBiasedEmbedder(), collection_name="test_rag_hybrid", reset=True)
    store.index_chunks(chunks)

    pipeline = RAGPipeline(vector_store=store, llm_client=FakeEchoLLM(), use_hybrid=True)
    result = pipeline.answer("What does policy EXP-402 say?", top_k=3)

    assert "EXP-402" in result["retrieved_chunks"][0].text
