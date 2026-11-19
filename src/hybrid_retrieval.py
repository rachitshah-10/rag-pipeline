"""Hybrid retrieval combining ChromaDB vector search with standard BM25 keyword scoring."""

import argparse
import math
from pathlib import Path
import re

from src.chunking import chunk_corpus
from src.embeddings import SentenceTransformerEmbedder
from src.vector_store import PolicyVectorStore, RetrievedChunk


def tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric and hyphenated tokens from text."""
    return re.findall(r"[a-z0-9-]+", text.lower())


def bm25_scores(
    query: str,
    texts: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Compute normalized [0, 1] Okapi BM25 scores for a query across document texts."""
    if not texts:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return [0.0] * len(texts)

    doc_tokens = [tokenize(text) for text in texts]
    avg_len = sum(len(toks) for toks in doc_tokens) / len(texts)
    if avg_len == 0:
        return [0.0] * len(texts)

    n_docs = len(texts)
    scores = [0.0] * n_docs
    for term in query_terms:
        doc_freq = sum(1 for toks in doc_tokens if term in toks)
        if doc_freq == 0:
            continue
        # Standard non-negative BM25 IDF balances rare terms against frequent corpus words.
        idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        for idx, toks in enumerate(doc_tokens):
            tf = toks.count(term)
            if tf > 0:
                norm = 1.0 - b + b * (len(toks) / avg_len)
                scores[idx] += idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)

    max_score = max(scores)
    return [s / max_score for s in scores] if max_score > 0 else scores


def hybrid_search(
    vector_store: PolicyVectorStore,
    query: str,
    top_k: int = 3,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> list[tuple[RetrievedChunk, float]]:
    """Retrieve chunks by blending ChromaDB vector similarity with normalized BM25 scores."""
    if vector_weight < 0 or keyword_weight < 0 or (vector_weight + keyword_weight) <= 0:
        raise ValueError("Weights must be non-negative and sum to a positive number.")
    if vector_store.count() == 0 or top_k <= 0:
        return []

    all_chunks = vector_store.get_all_chunks()
    kw_scores = bm25_scores(query, [c.text for c in all_chunks])

    # Convert L2 distance into a (0, 1] similarity score so it blends directly with BM25.
    vector_hits = {
        c.chunk_id: (c, 1.0 / (1.0 + c.distance))
        for c in vector_store.search(query=query, top_k=vector_store.count())
    }

    scored: list[tuple[RetrievedChunk, float]] = []
    for chunk, kw_sim in zip(all_chunks, kw_scores):
        vec_chunk, vec_sim = vector_hits.get(chunk.chunk_id, (chunk, 0.0))
        combined = vector_weight * vec_sim + keyword_weight * kw_sim
        scored.append((vec_chunk, combined))

    scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
    return scored[:top_k]


def main() -> None:
    """Compare vector-only vs. hybrid retrieval for any CLI query."""
    parser = argparse.ArgumentParser(description="Compare vector-only vs. hybrid retrieval.")
    parser.add_argument("--question", required=True, help="Query string to evaluate.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to return.")
    args = parser.parse_args()

    chunks = chunk_corpus(Path(__file__).resolve().parents[1] / "data" / "policies")
    store = PolicyVectorStore(SentenceTransformerEmbedder(), collection_name="hybrid_demo", reset=True)
    store.index_chunks(chunks)

    print(f"Query: {args.question}\n")
    print(f"=== Vector-Only Top {args.top_k} ===")
    for rank, chunk in enumerate(store.search(args.question, top_k=args.top_k), start=1):
        print(f"  #{rank} [{chunk.doc_name} | {chunk.section}] (dist={chunk.distance:.4f})")

    print(f"\n=== Hybrid (Vector + BM25) Top {args.top_k} ===")
    for rank, (chunk, score) in enumerate(hybrid_search(store, args.question, top_k=args.top_k), start=1):
        print(f"  #{rank} [{chunk.doc_name} | {chunk.section}] (hybrid_score={score:.4f})")


if __name__ == "__main__":
    main()
