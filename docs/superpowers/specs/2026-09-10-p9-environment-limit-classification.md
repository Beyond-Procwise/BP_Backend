# P9 — the governance limits living in environment variables

Classification for review, 2026-09-10. **Nothing has been migrated.** The prompt
asks for this first, and the naming of what counts as governance is the decision;
the migration after it is mechanical.

Derived by parsing `src/**/*.py` for `os.getenv` / `os.environ.get`
(`scratchpad/p9_scan.py`), not by reading. **165 distinct environment variables**
are read in `src/`; 59 of them are cast to a number at the point of use, which is
the signal that separates a limit from a hostname.

**33 are governance limits (group a). 132 are infrastructure (group b).** The
count matches N3's "33 governance-relevant limits" — arrived at independently,
which is some comfort that we are looking at the same set.

---

## Group (a) — belongs in `bp_policy`

The test applied: *would a customer's auditor expect a change to this to be
recorded?* If yes it is a rule about how the product behaves toward their money,
their suppliers or their data, and it should be versioned, attributable and
visible on a governance screen. If it only changes how fast or how hard the
software runs, it is group (b).

### a1 — what reaches the financial record (6)

The `_trgt` tables are the record of truth. These decide what gets in.

| Variable | Default | Read at |
|---|---|---|
| `PROMOTE_MIN_CONFIDENCE` | 50 | `services/linking_engine.py:53` |
| `PROMOTE_MIN_LINK_SCORE` | 80 | `services/deal_assignment_service.py:38` (+2 more) |
| `PROMOTE_REVIEW_MIN` | 65 | `services/linking_engine.py:55` |
| `PROPOSE_MIN_LINK_SCORE` | 40 | `services/link_proposals.py:105` |
| `PROPOSE_MAX_CANDIDATES` | 5 | `services/link_proposals.py:112` |
| `QUOTE_ANCHOR_MIN_SCORE` | falls back to `PROMOTE_MIN_LINK_SCORE` | `services/deal_assignment_service.py:40` |

### a2 — what counts as a match on money (3)

Reconciliation tolerances. Widen these and discrepancies stop being reported.

| Variable | Default | Read at |
|---|---|---|
| `RECON_AMOUNT_TOLERANCE_PCT` | 0.01 | `services/reconciliation.py:28` |
| `RECON_AMOUNT_TOLERANCE_ABS` | 1.00 | `services/reconciliation.py:29` |
| `RECON_TAX_TOLERANCE_PCT` | 0.1 | `services/reconciliation.py:30` |

### a3 — who a supplier IS (5)

Entity resolution bands. These decide when two names are one company — and
therefore whose bank details an invoice is paid against.

| Variable | Default | Read at |
|---|---|---|
| `SUPPLIER_REVIEW_LOW` | 82 | `services/extraction_v3/supplier_resolver.py:133` |
| `SUPPLIER_REVIEW_HIGH` | 96 | `services/extraction_v3/supplier_resolver.py:134` |
| `SUPPLIER_SWEEP_MIN_SCORE` | 88 | `services/extraction_v3/supplier_resolver.py:868` |
| `SUPPLIER_RESEARCH_NAME_MATCH` | 85 | `services/supplier_enrichment/research.py:80` |
| `SUPPLIER_RESEARCH_PROPOSE_CONF` | 0.75 | `services/supplier_enrichment/research.py:60` |

`SUPPLIER_RESEARCH_APPLY_CONF` is the deprecated spelling of the last one (P5,
`f0fd2f0`); it is still read for one release and warns. It is the same limit and
migrates once, under the new name.

### a4 — what an agent may offer a supplier (8)

Commercial bounds on negotiation. An agent that may concede 40% instead of 20%
is a different agent.

| Variable | Default | Read at |
|---|---|---|
| `NEG_MAX_VOLUME_LIMIT` | 1000 | `agents/negotiation_agent.py:499` |
| `NEG_MAX_TERM_DAYS` | 120 | `agents/negotiation_agent.py:500` |
| `NEG_MAX_SUPPLIER_REPLIES` | 3 | `agents/negotiation_agent.py:62` |
| `NEG_FIRST_COUNTER_AGGR_PCT` | 0.12 | `agents/negotiation_agent.py:102` |
| `NEG_MARKET_REVIEW_PCT` | 0.2 | `agents/negotiation_agent.py:497` |
| `NEG_MARKET_ESCALATION_PCT` | 0.4 | `agents/negotiation_agent.py:498` |
| `NEG_LT_VALUE_PCT_PER_WEEK` | 0.01 | `agents/negotiation_agent.py:74` |
| `NEG_COST_OF_CAPITAL_APR` | 0.12 | `agents/negotiation_agent.py:73` |

### a5 — how far an agent may reach (5)

Delegation bounds. How many agents may be spawned, and how many rounds of
tool-calling one may take before it must stop and answer.

| Variable | Default | Read at |
|---|---|---|
| `MAX_DYNAMIC_AGENTS` | 3 | `orchestration/orchestrator.py:117` |
| `TOOL_RUNTIME_MAX_ROUNDS` | 6 | `services/tool_runtime.py:39` |
| `GOVERNED_REASONING_MAX_ROUNDS` | 5 | `services/governance_tools/governed_reasoning.py:22` |
| `SUPPLIER_RESEARCH_MAX_ROUNDS` | 4 | `services/supplier_enrichment/research.py:41` |
| `NEG_THREAD_TRANSCRIPT_LIMIT` | unset = full history | `agents/negotiation_agent.py:78` |

`NEG_THREAD_TRANSCRIPT_LIMIT` is here rather than in (b) because it decides how
much of a negotiation the agent can see before it counters. That is not a
performance knob; it is how well informed the thing making an offer is.

### a6 — how hard the product tries to be right (2)

| Variable | Default | Read at |
|---|---|---|
| `EXTRACTION_JUDGE_MAX_CALLS` | 12 | `services/extraction/judge_gate.py:71` |
| `EXTRACTION_JUDGE_BUDGET_S` | 25 | `services/extraction/judge_gate.py:72` |

Arguably cost controls, and I nearly filed them under (b). They are here because
extraction accuracy is this product's stated first priority, and these decide
when the L3 judge stops checking. A cost cap that quietly lowers accuracy is a
governance decision whoever set it should have to own.

### a7 — everything else that is a rule (4)

| Variable | Default | Read at | Why |
|---|---|---|---|
| `OPPORTUNITY_MINING_MIN_IMPACT` | 100 | `services/backend_scheduler.py:880` | what the product considers worth telling a buyer about |
| `CAPTURE_RETENTION_DAYS` | code default | `services/capture_retention.py:64` | how long captured data is kept — a data-protection commitment |
| `DUPLICATE_INVOICE_DETECTOR_ENABLED` | 0 | `services/backend_scheduler.py:227` | a fraud control, **off by default**, switchable with no record |
| `SUPPLIER_RESEARCH_ENABLED` | 1 | `api/routers/supplier_research.py:25` | whether queries about suppliers leave the tenant |

The last two are switches rather than numbers, and they are the reason I would
not treat "it is a boolean" as automatically group (b): a switch that turns a
control off is the most consequential setting on the list.

---

## Group (b) — genuine infrastructure, stays in the environment

Not enumerated line by line (132 of them), but by kind, so the rule is checkable
rather than a list to trust:

- **Connection and identity of things** — `DB_*`, `PG*`, `IMAP_*`, `OLLAMA_BASE_URL`,
  `S3_*`, `DDB_TABLE`, model names (`AGENTNICK_MODEL`, `EXTRACTION_V3_JUDGE_MODEL`).
- **Scheduling cadence** — every `*_INTERVAL_MINUTES`, `*_DELAY_SECONDS`,
  `*_POLL_SECONDS`, `KG_SYNC_THROTTLE_SECONDS`. How often a job runs changes
  latency, not what is permitted.
- **Timeouts and concurrency** — `*_TIMEOUT*`, `OLLAMA_MAX_CONCURRENT`,
  `*_WORKERS`, `RERANK_BATCH`, `RAG_EMBED_BATCH`, `OLLAMA_NUM_GPU*`.
- **Retrieval and prompt sizing** — `RAG_TOP_K`, `RAG_PER_DOC_CAP`,
  `RAG_CONTEXT_MAX_CHARS`, `RAG_RERANK_TOP_N`, `COMPRESS_PER_CHUNK_CHARS`,
  `TOOL_RUNTIME_MAX_RESULT_CHARS`. These shape how a question is answered, not
  what may be done — and moving them would make a prompt-size tweak a policy edit.
- **Enablement of features that are not controls** — `ANALYTIC_ANSWER_V2_ENABLED`,
  `EXTRACTION_RENOVATION_ENABLED`, the scheduler's per-job `*_ENABLED` flags.

### One deliberate exception, flagged rather than decided

`ASK_AUTH_MODE` is left in group (b) and I want that argued rather than assumed.
It is the most security-relevant setting in the environment — it decides whether
identity is checked at all. It stays because it is deployment posture rather than
a limit, because it already fails closed when enforcement is configured badly and
warns on every startup when switched off, and because putting it in the database
would mean the database decides whether the product checks who is asking. If you
would rather it were policy, say so and it moves.

---

## What migration would look like (not done)

For each group (a) value, per the prompt:

1. The value moves into a `bp_policy` row and is read through `PolicyEngine`.
2. The environment variable keeps working **for one release** as an override that
   **logs a warning when it differs from policy**, so nothing changes silently
   during rollout.
3. **A missing policy value fails closed** — never a silent fall back to today's
   default. P6 is the precedent: the guard has to be able to tell "unset" from
   "set to the old number", or it is not a guard.

Point 3 is the one with teeth. Fifteen of these thirty-three currently have a
hardcoded fallback in the `os.getenv` call itself, so "the policy is missing" and
"the policy says 50" are the same value today. The migration has to remove those
defaults at the same time it adds the policy read, or the fail-closed behaviour
is untestable — exactly the hole found under P6.

## Open question for you

Group (a) is 33 rows if each value gets its own policy, which would be an
unhelpful governance screen. My proposal is **seven policy rows, one per section
above** (`PromotionThresholdPolicy`, `ReconciliationTolerancePolicy`,
`SupplierIdentityPolicy`, `NegotiationBoundsPolicy`, `AgentReachPolicy`,
`ExtractionEffortPolicy`, and the four in a7 attached to existing rows where one
already fits). Each row holds its values under `rules`, read by name, and none
of them carries `applies_to` — they are configuration, not gate-visible
authority, which is the line P7 drew and P6 followed.
