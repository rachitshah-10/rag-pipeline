"""Part 5 evaluation harness measuring real retrieval recall@k and answer accuracy over 10 policy Q&As."""

from dataclasses import dataclass
from pathlib import Path
import pytest

from src.chunking import chunk_corpus
from src.embeddings import SentenceTransformerEmbedder
from src.llm import AzureFoundryLLMClient
from src.rag_pipeline import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.vector_store import PolicyVectorStore, RetrievedChunk


@dataclass(frozen=True)
class EvalCase:
    """Single evaluation question paired with expected source chunk and key answer facts."""

    case_id: str
    question: str
    expected_doc: str
    expected_section: str
    expected_fact_groups: tuple[tuple[str, ...], ...]


EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="exp_meal_per_diem",
        question="What is the domestic meal per diem for overnight business travel?",
        expected_doc="expense_reimbursement_policy.md",
        expected_section="Section 2.2",
        expected_fact_groups=(("75",),),
    ),
    EvalCase(
        case_id="exp_client_entertainment",
        question="What is the per-attendee cap for client entertainment under EXP-302?",
        expected_doc="expense_reimbursement_policy.md",
        expected_section="Section 2.3",
        expected_fact_groups=(("150",),),
    ),
    EvalCase(
        case_id="exp_hotel_metro_target",
        question="What is the target hotel nightly rate in major metropolitan hubs like New York or London?",
        expected_doc="expense_reimbursement_policy.md",
        expected_section="Section 2.4",
        expected_fact_groups=(("275",),),
    ),
    EvalCase(
        case_id="exp_receipt_window",
        question="Within how many calendar days must itemized receipts be submitted in ConcurFlow?",
        expected_doc="expense_reimbursement_policy.md",
        expected_section="Section 2.5",
        expected_fact_groups=(("30",),),
    ),
    EvalCase(
        case_id="exp_non_reimbursable",
        question="Under policy EXP-402, are stand-alone bar tabs or traffic fines reimbursable?",
        expected_doc="expense_reimbursement_policy.md",
        expected_section="Section 2.6",
        expected_fact_groups=(("non-reimbursable", "not reimbursable", "ineligible", "no"),),
    ),
    EvalCase(
        case_id="pto_accrual_3_to_5_years",
        question="How many PTO days per year do employees with 3 to 5 years of service accrue?",
        expected_doc="pto_and_leave_policy.md",
        expected_section="Section 3.1",
        expected_fact_groups=(("20",),),
    ),
    EvalCase(
        case_id="pto_carryover_cap",
        question="What is the maximum number of unused PTO days that can carry over past December 31?",
        expected_doc="pto_and_leave_policy.md",
        expected_section="Section 3.2",
        expected_fact_groups=(("5",),),
    ),
    EvalCase(
        case_id="pto_sick_days",
        question="How many paid sick and wellness days do full-time employees receive each year?",
        expected_doc="pto_and_leave_policy.md",
        expected_section="Section 3.3",
        expected_fact_groups=(("8",),),
    ),
    EvalCase(
        case_id="pto_parental_leave",
        question="How many weeks of paid parental leave does Northstar Cloud Corp. provide?",
        expected_doc="pto_and_leave_policy.md",
        expected_section="Section 3.4",
        expected_fact_groups=(("16",),),
    ),
    EvalCase(
        case_id="pto_bereavement_immediate",
        question="How many paid bereavement days are provided for an immediate family member under PTO-302?",
        expected_doc="pto_and_leave_policy.md",
        expected_section="Section 3.5",
        expected_fact_groups=(("10",),),
    ),
)


def is_expected_chunk_retrieved(case: EvalCase, retrieved: list[RetrievedChunk]) -> bool:
    """Return True if any retrieved chunk matches the expected document and section."""
    return any(
        chunk.doc_name == case.expected_doc and case.expected_section in chunk.section
        for chunk in retrieved
    )


def answer_contains_expected_facts(case: EvalCase, answer: str) -> bool:
    """Return True if every expected fact group has at least one match in the answer."""
    lowered = answer.lower()
    return all(any(option in lowered for option in group) for group in case.expected_fact_groups)


@pytest.fixture(scope="module")
def eval_results() -> list[tuple[EvalCase, bool, bool]]:
    """Index the policy corpus once and evaluate all 10 cases against the real RAG pipeline."""
    policies_dir = Path(__file__).resolve().parents[1] / "data" / "policies"
    chunks = chunk_corpus(policies_dir)

    store = PolicyVectorStore(
        embedder=SentenceTransformerEmbedder(),
        collection_name="eval_harness",
        reset=True,
    )
    store.index_chunks(chunks)

    pipeline = RAGPipeline(
        vector_store=store,
        llm_client=AzureFoundryLLMClient(),
        use_hybrid=True,
        reranker=CrossEncoderReranker(),
    )

    outcomes: list[tuple[EvalCase, bool, bool]] = []
    for case in EVAL_CASES:
        result = pipeline.answer(case.question, top_k=3)
        recalled = is_expected_chunk_retrieved(case, result["retrieved_chunks"])
        accurate = answer_contains_expected_facts(case, result["answer"])
        outcomes.append((case, recalled, accurate))
    return outcomes


@pytest.mark.parametrize("index", range(len(EVAL_CASES)), ids=[c.case_id for c in EVAL_CASES])
def test_eval_case_recall_and_accuracy(
    eval_results: list[tuple[EvalCase, bool, bool]],
    index: int,
) -> None:
    """Assert retrieval recall@3 and expected answer facts for an individual eval case."""
    case, recalled, accurate = eval_results[index]
    assert recalled, f"Retrieval missed {case.expected_doc} ({case.expected_section})"
    assert accurate, f"Answer missing expected facts {case.expected_fact_groups}"


def test_eval_harness_summary_metrics(eval_results: list[tuple[EvalCase, bool, bool]]) -> None:
    """Print aggregate retrieval recall@3 and answer accuracy and enforce >= 80% threshold."""
    total = len(eval_results)
    recall_hits = sum(1 for _, recalled, _ in eval_results if recalled)
    accuracy_hits = sum(1 for _, _, accurate in eval_results if accurate)
    recall_rate = recall_hits / total
    accuracy_rate = accuracy_hits / total

    print(f"\n=== Evaluation Harness Summary ({total} cases) ===")
    for case, recalled, accurate in eval_results:
        print(
            f"  {case.case_id:28s} | recall@3={'PASS' if recalled else 'FAIL'} "
            f"| answer={'PASS' if accurate else 'FAIL'}"
        )
    print(f"Retrieval Recall@3: {recall_hits}/{total} ({recall_rate:.0%})")
    print(f"Answer Accuracy:    {accuracy_hits}/{total} ({accuracy_rate:.0%})")

    assert total >= 8
    assert recall_rate >= 0.8
    assert accuracy_rate >= 0.8
