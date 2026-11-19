"""Offline unit tests for chunking, vector store, LLM interface, and RAG pipeline."""

import math
from pathlib import Path
import pytest

from src.chunking import (
    DocumentChunk,
    chunk_corpus,
    chunk_markdown_document,
    load_policy_documents,
)
from src.llm import AzureFoundryLLMClient, LLMClient
from src.rag_pipeline import RAGPipeline, build_rag_prompt
from src.vector_store import PolicyVectorStore


class FakeTopicEmbedder:
    """Deterministic offline embedder mapping policy keywords onto unit topic axes."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode each text into an L2-normalized 4D vector."""
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            # Separate meal/expense, PTO/leave, and remote-work topics for offline ranking.
            expense_score = sum(
                1.0 for kw in ("meal", "per diem", "receipt", "expense", "reimburse", "exp-301")
                if kw in lowered
            )
            pto_score = sum(
                1.0 for kw in ("pto", "vacation", "carryover", "parental", "sick")
                if kw in lowered
            )
            remote_score = sum(
                1.0 for kw in ("remote", "stipend", "internet", "ergonomic")
                if kw in lowered
            )
            raw = [expense_score, pto_score, remote_score, 0.1]
            norm = math.sqrt(sum(v * v for v in raw))
            vectors.append([v / norm for v in raw])
        return vectors


class FakeLLMClient:
    """Offline fake LLM client that echoes received context and records prompts."""

    def __init__(self, canned_answer: str = "Domestic meals are reimbursed up to $75/day.") -> None:
        """Initialize the fake client with a configurable response."""
        self.canned_answer = canned_answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        """Record the incoming prompt and return the canned answer."""
        self.prompts.append(prompt)
        return self.canned_answer


def test_chunk_markdown_splits_sections_and_applies_overlap() -> None:
    """Verify long sections split into bounded chunks with word overlap and metadata."""
    long_body = " ".join(f"word{i}" for i in range(180))
    doc = f"# Policy Title\n\n## Section 1.1: Overview\nShort intro.\n\n## Section 1.2: Details\n{long_body}\n"

    chunks = chunk_markdown_document(
        doc_name="sample_policy.md",
        content=doc,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) >= 3
    assert chunks[0].doc_name == "sample_policy.md"
    assert "Section 1.1" in chunks[0].section
    assert all(len(c.text.split()) <= 100 for c in chunks)

    # Confirm the second chunk of Section 1.2 overlaps the end of the first chunk.
    sec2_chunks = [c for c in chunks if "Section 1.2" in c.section]
    assert len(sec2_chunks) >= 2
    first_tail = sec2_chunks[0].text.split()[-20:]
    second_head = sec2_chunks[1].text.split()[:20]
    assert first_tail == second_head


def test_chunk_markdown_rejects_invalid_window_params() -> None:
    """Verify invalid chunk_size or overlap raises ValueError."""
    with pytest.raises(ValueError):
        chunk_markdown_document("doc.md", "## Sec\nhello", chunk_size=50, overlap=50)
    with pytest.raises(ValueError):
        chunk_markdown_document("doc.md", "## Sec\nhello", chunk_size=0, overlap=0)


def test_load_and_chunk_real_policy_corpus() -> None:
    """Confirm all 4 generated policy docs load and chunk under the 256-token budget."""
    policies_dir = Path(__file__).resolve().parents[1] / "data" / "policies"
    docs = load_policy_documents(policies_dir)
    assert len(docs) == 4

    chunks = chunk_corpus(policies_dir, chunk_size=150, overlap=30)
    assert len(chunks) > 4
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert all(len(c.text.split()) <= 150 for c in chunks)
    assert all(c.doc_name and c.section for c in chunks)


def test_vector_store_indexes_and_retrieves_relevant_chunk() -> None:
    """Confirm PolicyVectorStore stores chunk metadata and ranks matching policy first."""
    store = PolicyVectorStore(
        embedder=FakeTopicEmbedder(),
        collection_name="test_policy_store",
        reset=True,
    )
    chunks = [
        DocumentChunk(
            chunk_id="exp::0",
            text="Domestic meals are reimbursed up to $75 per day under EXP-301.",
            doc_name="expense_reimbursement_policy.md",
            section="Section 2.2: Meal Per Diem",
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="pto::0",
            text="Employees may carry over a maximum of 5 unused PTO days past December 31.",
            doc_name="pto_and_leave_policy.md",
            section="Section 3.2: Carryover Cap",
            chunk_index=0,
        ),
    ]

    indexed = store.index_chunks(chunks)
    assert indexed == 2
    assert store.count() == 2

    results = store.search("What is the domestic meal per diem?", top_k=1)
    assert len(results) == 1
    assert results[0].doc_name == "expense_reimbursement_policy.md"
    assert "EXP-301" in results[0].text


def test_azure_llm_client_satisfies_protocol_and_requires_api_key(monkeypatch) -> None:
    """Verify FakeLLMClient fits LLMClient protocol and missing API key raises ValueError."""
    assert isinstance(FakeLLMClient(), LLMClient)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError):
        AzureFoundryLLMClient(
            endpoint="https://example.services.ai.azure.com/openai/v1",
            deployment_name="gpt-6-astra",
            api_key="",
        )


def test_rag_pipeline_answers_using_retrieved_context_offline() -> None:
    """Verify end-to-end RAGPipeline retrieves top chunks and passes grounded prompt to LLM."""
    policies_dir = Path(__file__).resolve().parents[1] / "data" / "policies"
    chunks = chunk_corpus(policies_dir, chunk_size=150, overlap=30)

    store = PolicyVectorStore(
        embedder=FakeTopicEmbedder(),
        collection_name="test_rag_e2e",
        reset=True,
    )
    store.index_chunks(chunks)

    fake_llm = FakeLLMClient(canned_answer="Domestic meals are reimbursed up to $75/day.")
    pipeline = RAGPipeline(vector_store=store, llm_client=fake_llm)

    result = pipeline.answer("What is the domestic meal per diem?", top_k=2)

    assert result["answer"] == "Domestic meals are reimbursed up to $75/day."
    assert len(result["retrieved_chunks"]) == 2
    assert len(fake_llm.prompts) == 1
    prompt_sent = fake_llm.prompts[0]
    assert "What is the domestic meal per diem?" in prompt_sent
    assert "expense_reimbursement_policy.md" in prompt_sent
    assert build_rag_prompt("What is the domestic meal per diem?", result["retrieved_chunks"]) == prompt_sent
