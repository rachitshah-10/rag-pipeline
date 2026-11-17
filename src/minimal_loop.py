"""Minimal embed-store-retrieve loop to validate SentenceTransformers and ChromaDB."""

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from src.embeddings import Embedder, SentenceTransformerEmbedder


def create_collection(name: str = "minimal_test") -> Collection:
    """Create a fresh in-memory ChromaDB collection for isolated runs.

    Args:
        name: Collection name inside the ephemeral ChromaDB client.

    Returns:
        Empty ChromaDB Collection instance.
    """
    client = chromadb.EphemeralClient()
    # EphemeralClient shares in-memory state within a process, so reset any prior collection
    # with the same name to keep test runs deterministic.
    existing = [col.name for col in client.list_collections()]
    if name in existing:
        client.delete_collection(name=name)
    return client.create_collection(name=name)


def store_texts(
    collection: Collection,
    ids: list[str],
    texts: list[str],
    embedder: Embedder,
) -> None:
    """Embed text chunks and insert them into a ChromaDB collection.

    Args:
        collection: Target ChromaDB collection.
        ids: Unique string identifiers corresponding to each text.
        texts: Text documents or chunks to embed and store.
        embedder: Embedding provider implementing the Embedder protocol.

    Raises:
        ValueError: If ids and texts do not have the same length.
    """
    if len(ids) != len(texts):
        raise ValueError(
            f"ids and texts must have the same length (got {len(ids)} and {len(texts)})"
        )
    if not texts:
        return

    # Pass explicit embeddings so ChromaDB does not attempt to download its default ONNX model.
    vectors = embedder.embed(texts)
    collection.add(ids=ids, documents=texts, embeddings=vectors)


def query_texts(
    collection: Collection,
    query: str,
    embedder: Embedder,
    n_results: int = 2,
) -> list[tuple[str, str, float]]:
    """Embed a query string and retrieve the closest matching stored texts.

    Args:
        collection: ChromaDB collection to search.
        query: Natural language query string.
        embedder: Embedding provider implementing the Embedder protocol.
        n_results: Maximum number of ranked results to return.

    Returns:
        List of (id, document_text, distance) tuples ordered from closest to farthest.
    """
    query_vector = embedder.embed([query])[0]
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "distances"],
    )

    ids = raw["ids"][0]
    documents = raw["documents"][0]
    distances = raw["distances"][0]
    return [
        (doc_id, doc_text, float(dist))
        for doc_id, doc_text, dist in zip(ids, documents, distances)
    ]


def run_minimal_loop(embedder: Embedder | None = None) -> dict[str, Any]:
    """Execute the two-step minimal embed-store-retrieve verification loop.

    Step 1 embeds and stores a single policy snippet and verifies count == 1.
    Step 2 adds a second, unrelated snippet and queries to verify the relevant one ranks first.

    Args:
        embedder: Optional Embedder override (defaults to SentenceTransformerEmbedder).

    Returns:
        Dictionary containing initial_count, final_count, query, and ranked results.
    """
    active_embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
    collection = create_collection("minimal_loop_demo")

    # Step 1: Embed and store one short piece of text before expanding.
    store_texts(
        collection=collection,
        ids=["doc-1"],
        texts=["Employees may work remotely up to three days per week with manager approval."],
        embedder=active_embedder,
    )
    initial_count = collection.count()

    # Step 2: Add a second distinct text and verify similarity search ranks the right one first.
    store_texts(
        collection=collection,
        ids=["doc-2"],
        texts=["Submit travel and meal expense receipts within thirty days for reimbursement."],
        embedder=active_embedder,
    )
    final_count = collection.count()

    query = "How many days per week can employees work remotely?"
    results = query_texts(
        collection=collection,
        query=query,
        embedder=active_embedder,
        n_results=2,
    )

    return {
        "initial_count": initial_count,
        "final_count": final_count,
        "query": query,
        "results": results,
    }


def main() -> None:
    """Run the minimal embed-store-retrieve loop and print formatted results."""
    summary = run_minimal_loop()
    print("=== Minimal Embed-Store-Retrieve Loop ===")
    print(f"Step 1 - Stored single chunk. Collection count: {summary['initial_count']}")
    print(f"Step 2 - Added second chunk.  Collection count: {summary['final_count']}")
    print(f"\nQuery: {summary['query']}\n")
    print("Ranked Results (lowest distance = most similar):")
    for rank, (doc_id, text, distance) in enumerate(summary["results"], start=1):
        print(f"  #{rank} [{doc_id}] (distance={distance:.4f}): {text}")


if __name__ == "__main__":
    main()
