# Policy Corpus & Planted Data Quality Issue

This directory contains 4 generated company policy documents for **Northstar Cloud Corp.** in `policies/`.

## Corpus Inventory
- `policies/remote_work_policy_2026.md` — Active Hybrid & Remote Work Policy (`POL-RW-2026`, v2.1)
- `policies/remote_work_policy_2024_legacy.md` — Legacy Remote Work Guidelines (`POL-RW-2024`, v1.2)
- `policies/expense_reimbursement_policy.md` — Travel & Expense Reimbursement Policy (`POL-EXP-2026`, v3.0)
- `policies/pto_and_leave_policy.md` — Paid Time Off & Leave Policy (`POL-PTO-2026`, v2.4)

## Planted Data Quality Issue (Part 1 & Part 6)
- **Type:** Outdated duplicate document with conflicting details.
- **Location:** `policies/remote_work_policy_2024_legacy.md` (`POL-RW-2024`, Effective January 15, 2024) was left in the ingestion folder alongside the current `policies/remote_work_policy_2026.md` (`POL-RW-2026`, Effective January 1, 2026).
- **Conflicting Details:**
  - Remote days per week: **2 days** (2024) vs. **3 days** (2026)
  - Core collaboration hours: **9:00 AM–4:00 PM EST** (2024) vs. **10:00 AM–3:00 PM local time** (2026)
  - Home internet stipend: **$50/month (`RW-NET-50`)** (2024) vs. **$75/month (`RW-NET-75`)** (2026)
  - Ergonomic allowance: **$400 one-time (`RW-ERG-400`)** (2024) vs. **$600 every 3 years (`RW-ERG-600`)** (2026)
  - Out-of-area work: **5 days out-of-state / no international** (2024) vs. **15 days international** (2026)
