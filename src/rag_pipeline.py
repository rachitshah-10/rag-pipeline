"""End-to-end basic RAG pipeline connecting chunking, ChromaDB, and Azure Foundry LLM."""

import argparse
from pathlib import Path
from typing import Any

from src.chunking import chunk_corpus
from src.embeddings import SentenceTransformerEmbedder
from src.hybrid_retrieval import hybrid_search
from src.llm import AzureFoundryLLMClient, LLMClient
from src.reranker import CrossEncoderReranker, Reranker
from src.vector_store import PolicyVectorStore, RetrievedChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICIES_DIR = PROJECT_ROOT / "data" / "policies"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_db"


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Format retrieved policy excerpts and the user question into a grounded LLM prompt."""
    context_blocks = [
        f"[Excerpt {idx} | {chunk.doc_name} | {chunk.section}]\n{chunk.text}"
        for idx, chunk in enumerate(chunks, start=1)
    ]
    context_text = "\n\n".join(context_blocks)
    return (
        "You are an internal HR and company policy assistant for Northstar Cloud Corp. "
        "Answer the employee's question using only the policy excerpts provided below. "
        "If the excerpts do not contain enough information, say so clearly.\n\n"
        f"Policy Excerpts:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


class RAGPipeline:
    """Retrieve-and-generate pipeline over indexed policy chunks."""

    def __init__(
        self,
        vector_store: PolicyVectorStore,
        llm_client: LLMClient,
        use_hybrid: bool = True,
        reranker: Reranker | None = None,
        candidate_k: int = 10,
    ) -> None:
        """Bind a vector store retriever, LLM generator, and optional reranker."""
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.use_hybrid = use_hybrid
        self.reranker = reranker
        self.candidate_k = candidate_k

    def answer(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Retrieve relevant chunks, optionally rerank, and generate a grounded LLM answer."""
        pool_k = max(self.candidate_k, top_k) if self.reranker is not None else top_k
        if self.use_hybrid:
            candidates = [
                chunk for chunk, _ in hybrid_search(self.vector_store, query=question, top_k=pool_k)
            ]
        else:
            candidates = self.vector_store.search(query=question, top_k=pool_k)

        retrieved = (
            [chunk for chunk, _ in self.reranker.rerank(question, candidates, top_k=top_k)]
            if self.reranker is not None
            else candidates
        )
        prompt = build_rag_prompt(question=question, chunks=retrieved)
        answer_text = self.llm_client.generate(prompt)
        return {
            "question": question,
            "answer": answer_text,
            "retrieved_chunks": retrieved,
            "prompt": prompt,
        }


def run_sample_query(
    question: str,
    policies_dir: Path = DEFAULT_POLICIES_DIR,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    top_k: int = 3,
    use_hybrid: bool = True,
    use_rerank: bool = True,
) -> dict[str, Any]:
    """Index policy documents locally and answer one question using SentenceTransformers and Azure Foundry."""
    chunks = chunk_corpus(policies_dir)
    store = PolicyVectorStore(
        embedder=SentenceTransformerEmbedder(),
        collection_name="northstar_policies",
        persist_dir=persist_dir,
        reset=True,
    )
    store.index_chunks(chunks)

    pipeline = RAGPipeline(
        vector_store=store,
        llm_client=AzureFoundryLLMClient(),
        use_hybrid=use_hybrid,
        reranker=CrossEncoderReranker() if use_rerank else None,
    )
    result = pipeline.answer(question=question, top_k=top_k)
    result["indexed_chunks"] = len(chunks)
    return result


def main() -> None:
    """CLI entry point to run the RAG pipeline on a policy question."""
    parser = argparse.ArgumentParser(description="Query Northstar policy docs via RAG.")
    parser.add_argument(
        "--question",
        required=True,
        help="Natural language policy question to answer.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--mode",
        choices=("hybrid", "vector"),
        default="hybrid",
        help="Retrieval strategy to use (default: hybrid).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable CrossEncoder reranking after initial retrieval.",
    )
    args = parser.parse_args()

    result = run_sample_query(
        question=args.question,
        top_k=args.top_k,
        use_hybrid=(args.mode == "hybrid"),
        use_rerank=not args.no_rerank,
    )
    print(f"Indexed {result['indexed_chunks']} policy chunks into {DEFAULT_PERSIST_DIR}")
    print(f"\nQuestion: {result['question']}\n")
    print("Top Retrieved Chunks:")
    for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(
            f"  #{rank} [{chunk.doc_name} | {chunk.section}] "
            f"(distance={chunk.distance:.4f}): {chunk.text[:120]}..."
        )
    print(f"\nGenerated Answer:\n{result['answer']}")


if __name__ == "__main__":
    main()
