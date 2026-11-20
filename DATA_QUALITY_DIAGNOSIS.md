# Part 6: Data Quality Diagnosis (Planted Issue Root-Cause Analysis)

## 1. Query That Surfaced the Issue

Command run against the live pipeline (`all-MiniLM-L6-v2` + BM25 hybrid retrieval + `ms-marco-MiniLM-L-6-v2` CrossEncoder + Azure Foundry `gpt-6-astra`):

```bash
python -m src.rag_pipeline --question "How many days per week can employees work remotely and what is the monthly home internet stipend?"
```

## 2. Observed (Flawed) System Output

**Top 3 Retrieved Chunks:**
1. `remote_work_policy_2024_legacy.md | Section 1.4: Home Internet Stipend (Code: RW-NET-50)` ($50/month)
2. `remote_work_policy_2026.md | Section 1.2: Weekly Remote Work Cadence (Code: RW-101)` (3 days/week)
3. `remote_work_policy_2024_legacy.md | Section 1.2: Weekly Remote Work Cadence (Code: RW-101)` (2 days/week)

**Generated Answer:**
- **Remote work:** Under the 2026 policy, eligible employees may work remotely up to **3 days per week**, with Tuesdays and Wednesdays as in-office anchor days. Team leads may approve temporary exceptions for up to 4 remote days in specified circumstances.
- **Home internet stipend:** The provided **2024 legacy policy** specifies **$50 per month** for hybrid employees who work from home at least one scheduled day per week. The excerpts do not confirm whether this stipend remains in effect under the 2026 policy.

**Sources Cited:**
- `remote_work_policy_2024_legacy.md | Section 1.4: Home Internet Stipend (Code: RW-NET-50)`
- `remote_work_policy_2026.md | Section 1.2: Weekly Remote Work Cadence (Code: RW-101)`
- `remote_work_policy_2024_legacy.md | Section 1.2: Weekly Remote Work Cadence (Code: RW-101)`

**Why the answer is wrong/incomplete:** The current active policy (`remote_work_policy_2026.md`, Section 1.4 `RW-NET-75`) provides **$75 per month**, not $50 per month.

---

## 3. Two-Question Diagnosis

### Question 1: Was it retrieval or generation logic?
**No.** Both retrieval and generation worked as designed:
- **Retrieval & Reranking:** Every returned chunk directly matched the two topics in the question (`Section 1.2: Weekly Remote Work Cadence` and `Section 1.4: Home Internet Stipend`). The `top_k=3` context budget simply filled up before `remote_work_policy_2026.md | Section 1.4` could be included because two duplicate chunks from the 2024 file occupied slots `#1` and `#3`.
- **Generation:** `gpt-6-astra` faithfully grounded its answer in the three retrieved chunks and even flagged that the 2024 stipend was not confirmed by the 2026 excerpts.

### Question 2: Was it the source data?
**Yes — the root cause is bad source data and missing document lineage.**
- The ingestion directory (`data/policies/`) contains an outdated duplicate (`remote_work_policy_2024_legacy.md`, `POL-RW-2024`, Effective Jan 15, 2024) alongside the current policy (`remote_work_policy_2026.md`, `POL-RW-2026`, Effective Jan 1, 2026).
- Because both files share identical section structure and nearly identical wording, the retriever cannot distinguish active policy from superseded history without corpus curation or lifecycle metadata (`status: active | superseded`).

## 4. Recommended Fix (Data / Ingestion Layer)
1. Archive or remove `remote_work_policy_2024_legacy.md` from the active `data/policies/` ingestion directory (kept intentionally here for Part 1 & Part 6 demonstration).
2. Add document lifecycle metadata (`effective_date`, `status: active | archived`) at ingestion time and filter out archived documents before indexing into ChromaDB.
