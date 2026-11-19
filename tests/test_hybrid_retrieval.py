"""Offline unit tests for keyword search and hybrid vector+keyword retrieval."""

import math
from pathlib import Path
import pytest

from src.chunking import DocumentChunk, chunk_corpus
from src.hybrid_retrieval import HybridRetriever, keyword_search, tokenize
from src.vector_store import PolicyVectorStore, RetrievedChunk


class FakeHeaderBiasedEmbedder:
    """Offline embedder mimicking dense retrieval missing exact policy codes in favor of 'policy'."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Project texts so generic 'policy' headers outscore exact code chunks in vector space."""
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            # Mimic all-MiniLM-L6-v2 prioritizing 'Policy ID' headers over 'EXP-402' section text.
            header_score = 3.0 if ("policy id" in lowered or "policy" in lowered) else 0.0
            wifi_score = 3.0 if ("wi-fi" in lowered or "internet" in lowered or "stipend" in lowered) else 0.0
            expense_score = 1.0 if ("non-reimbursable" in lowered or "alcohol" in lowered) else 0.0
            raw = [header_score, wifi_score, expense_score, 0.2]
            norm = math.sqrt(sum(v * v for v in raw))
            vectors.append([v / norm for v in raw])
        return vectors


def test_tokenize_preserves_hyphenated_codes_and_strips_stopwords() -> None:
    """Verify tokenize keeps hyphenated policy codes intact and removes stopwords."""
    tokens = tokenize("What does policy EXP-402 say about RW-ERG-600?")
    assert "exp-402" in tokens
    assert "rw-erg-600" in tokens
    assert "what" not in tokens
    assert "does" not in tokens


def test_keyword_search_ranks_exact_policy_code_first() -> None:
    """Confirm keyword_search ranks exact rare policy code matches above generic word matches."""
    chunks = [
        RetrievedChunk(
            chunk_id="header::0",
            text="# Expense Policy. Policy ID: POL-EXP-2026.",
            doc_name="expense_reimbursement_policy.md",
            section="Document Header",
            distance=0.0,
        ),
        RetrievedChunk(
            chunk_id="exp::6",
            text="Under code EXP-402, stand-alone bar tabs and traffic fines are non-reimbursable.",
            doc_name="expense_reimbursement_policy.md",
            section="Section 2.6: Non-Reimbursable Expenses (Code: EXP-402)",
            distance=0.0,
        ),
        RetrievedChunk(
            chunk_id="pto::0",
            text="# Paid Time Off Policy. Policy ID: POL-PTO-2026.",
            doc_name="pto_and_leave_policy.md",
            section="Document Header",
            distance=0.0,
        ),
    ]

    matches = keyword_search("What does policy EXP-402 say?", chunks, top_k=3)

    assert len(matches) >= 1
    top_chunk, top_score = matches[0]
    assert top_chunk.chunk_id == "exp::6"
    assert top_score > matches[1][1]


def test_get_all_chunks_returns_stored_chunks_with_metadata() -> None:
    """Verify PolicyVectorStore.get_all_chunks returns indexed chunks and metadata."""
    store = PolicyVectorStore(
        embedder=FakeHeaderBiasedEmbedder(),
        collection_name="test_get_all",
        reset=True,
    )
    chunks = [
        DocumentChunk("c1", "First chunk text.", "doc_a.md", "Section 1", 0),
        DocumentChunk("c2", "Second chunk text.", "doc_b.md", "Section 2", 0),
    ]
    store.index_chunks(chunks)

    stored = store.get_all_chunks()
    assert len(stored) == 2
    by_id = {c.chunk_id: c for c in stored}
    assert by_id["c1"].doc_name == "doc_a.md"
    assert by_id["c2"].section == "Section 2"


def test_hybrid_retriever_beats_vector_only_on_exact_code_query() -> None:
    """Confirm hybrid retrieval surfaces EXP-402 at rank 1 when vector-only top-3 misses it."""
    policies_dir = Path(__file__).resolve().parents[1] / "data" / "policies"
    chunks = chunk_corpus(policies_dir, chunk_size=150, overlap=30)

    store = PolicyVectorStore(
        embedder=FakeHeaderBiasedEmbedder(),
        collection_name="test_hybrid_vs_vector",
        reset=True,
    )
    store.index_chunks(chunks)

    query = "What does policy EXP-402 say?"
    vector_only = store.search(query, top_k=3)
    assert not any("EXP-402" in c.text for c in vector_only)

    hybrid = HybridRetriever(vector_store=store)
    hybrid_results = hybrid.search(query, top_k=3)

    assert len(hybrid_results) == 3
    assert "EXP-402" in hybrid_results[0].chunk.text
    assert hybrid_results[0].keyword_rank == 1
    assert hybrid_results[0].hybrid_score >= hybrid_results[1].hybrid_score


def test_hybrid_retriever_still_works_on_semantic_query_without_keyword_hits() -> None:
    """Verify pure semantic queries with zero keyword overlap still rank via vector similarity."""
    store = PolicyVectorStore(
        embedder=FakeHeaderBiasedEmbedder(),
        collection_name="test_semantic_fallback",
        reset=True,
    )
    store.index_chunks(
        [
            DocumentChunk("net::0", "Residential high-speed internet stipend is $75/month.", "rw.md", "Sec 1.4", 0),
            DocumentChunk("exp::0", "Alcohol-only bar tabs are non-reimbursable.", "exp.md", "Sec 2.6", 0),
        ]
    )

    hybrid = HybridRetriever(vector_store=store)
    # Query uses 'wi-fi' (hits wifi_score in FakeHeaderBiasedEmbedder) with zero exact keyword overlap.
    results = hybrid.search("wi-fi", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "net::0"
    assert results[0].keyword_score == 0.0
    assert results[0].vector_score > 0.0


def test_hybrid_retriever_rejects_invalid_weights() -> None:
    """Verify HybridRetriever raises ValueError on negative or all-zero weights."""
    store = PolicyVectorStore(
        embedder=FakeHeaderBiasedEmbedder(),
        collection_name="test_invalid_weights",
        reset=True,
    )
    with pytest.raises(ValueError):
        HybridRetriever(vector_store=store, vector_weight=-0.1, keyword_weight=1.0)
    with pytest.raises(ValueError):
        HybridRetriever(vector_store=store, vector_weight=0.0, keyword_weight=0.0)
