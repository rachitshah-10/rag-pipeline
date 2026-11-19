"""Offline unit tests for the minimal embed-store-retrieve loop."""

import math
import pytest

from src.embeddings import Embedder
from src.minimal_loop import (
    create_collection,
    query_texts,
    run_minimal_loop,
    store_texts,
)


class FakeEmbedder:
    """Deterministic offline embedder projecting text onto simple topic axes."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Map each text to an L2-normalized 4D topic vector."""
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            # Project onto distinct topic axes so related texts cluster together.
            remote_score = sum(
                1.0 for kw in ("remote", "remotely", "home", "telecommute", "week") if kw in lowered
            )
            expense_score = sum(
                1.0 for kw in ("expense", "reimbursement", "meal", "receipt", "travel") if kw in lowered
            )
            pto_score = sum(
                1.0 for kw in ("pto", "vacation", "holiday", "leave") if kw in lowered
            )
            raw = [remote_score, expense_score, pto_score, 0.1]
            norm = math.sqrt(sum(v * v for v in raw))
            vectors.append([v / norm for v in raw])
        return vectors


def test_fake_embedder_satisfies_embedder_protocol() -> None:
    """Verify FakeEmbedder satisfies the Embedder protocol."""
    embedder = FakeEmbedder()
    assert isinstance(embedder, Embedder)


def test_store_single_chunk_in_chromadb() -> None:
    """Confirm one short text embeds and stores in ChromaDB."""
    embedder = FakeEmbedder()
    collection = create_collection("test_single_chunk")

    store_texts(
        collection=collection,
        ids=["doc-1"],
        texts=["Employees may work remotely up to three days per week."],
        embedder=embedder,
    )

    assert collection.count() == 1


def test_retrieve_more_relevant_of_two_texts() -> None:
    """Confirm similarity search ranks the relevant document first."""
    embedder = FakeEmbedder()
    collection = create_collection("test_two_chunks")

    store_texts(
        collection=collection,
        ids=["doc-remote", "doc-expense"],
        texts=[
            "Employees may work remotely up to three days per week with manager approval.",
            "Submit travel and meal expense receipts within thirty days for reimbursement.",
        ],
        embedder=embedder,
    )

    results = query_texts(
        collection=collection,
        query="How many days per week can I work remotely?",
        embedder=embedder,
        n_results=2,
    )

    assert len(results) == 2
    top_id, top_text, top_distance = results[0]
    second_id, _, second_distance = results[1]
    assert top_id == "doc-remote"
    assert "remotely" in top_text.lower()
    assert second_id == "doc-expense"
    assert top_distance < second_distance


def test_store_texts_rejects_mismatched_lengths() -> None:
    """Confirm store_texts raises ValueError on mismatched ids and texts."""
    embedder = FakeEmbedder()
    collection = create_collection("test_mismatch")

    with pytest.raises(ValueError):
        store_texts(
            collection=collection,
            ids=["doc-1"],
            texts=["First text", "Second text"],
            embedder=embedder,
        )


def test_run_minimal_loop_end_to_end_offline() -> None:
    """Verify run_minimal_loop returns expected counts and top match offline."""
    summary = run_minimal_loop(embedder=FakeEmbedder())

    assert summary["initial_count"] == 1
    assert summary["final_count"] == 2
    assert isinstance(summary["query"], str) and len(summary["query"]) > 0
    assert len(summary["results"]) == 2
    assert summary["results"][0][0] == "doc-1"
