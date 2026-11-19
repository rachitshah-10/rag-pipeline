"""End-to-end basic RAG pipeline connecting chunking, ChromaDB, and Azure Foundry LLM."""

import argparse
from pathlib import Path
from typing import Any

from src.chunking import chunk_corpus
from src.embeddings import SentenceTransformerEmbedder
from src.llm import AzureFoundryLLMClient, LLMClient
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
    """Basic retrieve-and-generate pipeline over indexed policy chunks."""

    def __init__(self, vector_store: PolicyVectorStore, llm_client: LLMClient) -> None:
        """Bind a vector store retriever and an LLM generator."""
        self.vector_store = vector_store
        self.llm_client = llm_client

    def answer(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Retrieve top_k relevant chunks and generate a grounded LLM answer."""
        retrieved = self.vector_store.search(query=question, top_k=top_k)
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

    pipeline = RAGPipeline(vector_store=store, llm_client=AzureFoundryLLMClient())
    result = pipeline.answer(question=question, top_k=top_k)
    result["indexed_chunks"] = len(chunks)
    return result


def main() -> None:
    """CLI entry point to run the basic RAG pipeline on a policy question."""
    parser = argparse.ArgumentParser(description="Query Northstar policy docs via basic RAG.")
    parser.add_argument(
        "--question",
        default="What is the domestic meal per diem and within how many days must receipts be submitted?",
        help="Natural language policy question to answer.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    result = run_sample_query(question=args.question, top_k=args.top_k)
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
