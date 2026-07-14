# Contract Obligation Intelligence (Hyper-Extract → AgentNick)

**Date:** 2026-07-14
**Status:** Design — approved, pending spec review
**Branch:** Development

---

## 1. Problem

BP_Backend's extraction pipeline reads **tables** and produces **fields** — invoice number,
total, supplier, line items. It is regex-primary, LLM-assisted, and scores near 100% on the
live corpus. It is not in scope for change here.

But a contract is not a table. Its value sits in sentences:

> *"Time of delivery shall be of the essence and if the Contractor fails to deliver the Goods
> within the time promised, the Authority may release itself from any obligation to accept and
> pay for the Goods."*

Nothing in the product reads that. `proc.bp_contracts` has 27 columns of header metadata and
**no column for clause text at all**. So questions that matter commercially cannot be asked:

- Which contracts let us re-open pricing this quarter?
- Which suppliers owe us a late-delivery remedy?
- What are we contractually on the hook for if we cancel?

This feature reads contract prose and turns it into **obligations you can query**.

### Why Hyper-Extract, specifically

An obligation is genuinely **n-ary**. "Contractor must deliver by the Specification date, or
the Authority may refuse payment" involves supplier, buyer, duty, trigger, and remedy *at once*.
A normal graph has only pairwise edges, so it must shred that into fragments and the meaning
is lost. Hyper-Extract's **hypergraph** has edges that connect many nodes at once, holding the
obligation as a single fact. That is the reason to adopt it, and the only reason.

### Honest scoping caveat

**The live corpus contains zero contracts** (235 documents: 182 invoices, 46 POs, 7 quotes;
`bp_contracts` and `bp_contract_raw` are both empty). Contracts are an *intended* document type
— `extraction/dispatch.py` already accepts `contract` and someone built the table for it — but
none have been uploaded. This feature is therefore built **ahead of its data**, on the explicit
decision that contracts are coming and that nothing useful happening to them today is part of
why they aren't uploaded. Validation is against real published contracts (see §7) until real
customer contracts land. **This must not be reported as "validated on the live corpus."**

---

## 2. What we are NOT doing

- **Not touching field extraction.** The `src/services/extraction/` path stays untouched. This
  feature cannot regress the ~100% field accuracy because it never runs in that path.
- **Not building on Neo4j**, despite it running. Every Neo4j write in the codebase today is
  wrapped in a bare `except: log`, so extraction reports success whether or not the graph was
  written. We will not put a product feature on a store with a silent failure mode. Postgres
  first; a Neo4j mirror is a later decision, gated on a health signal.
- **Not merging a corpus-wide graph.** One hypergraph per contract, tied to its `document_id`.
  Cross-contract questions are answered by querying Postgres. This sidesteps the known supplier
  entity-resolution problem.
- **Not using the library's CLI, MCP server, Obsidian export, or its 9 RAG methods.** We use
  `AutoHypergraph` and nothing else.
- **No UI.** Backend + API only. The UI is a separate decision after this is demonstrated.

---

## 3. Architecture

Five units.

### 3.1 `src/services/hyperextract/agentnick.py` — the AgentNick bridge

**This is the "map it to AgentNick" deliverable.**

The mismatch: Hyper-Extract asks for structured output via LangChain's
`with_structured_output(schema, method="function_calling")`. BP_Backend gets structured output
from AgentNick via Ollama's native **`format=` JSON-schema grammar**, through
`services/ollama_client.py` — which also owns a **semaphore capped at 2 concurrent requests**
because GPU contention causes timeouts on the live extraction path.

Pointing LangChain's `ChatOpenAI` at Ollama's OpenAI-compatible endpoint would bypass both:
we would lose grammar-constrained decoding (which is what fixed the `unit_price` bug) *and* we
would hammer the GPU that live extraction depends on, because Hyper-Extract calls
`.batch(max_concurrency=10)`.

So we implement a LangChain `BaseChatModel` that **delegates to `ollama_client.ollama_generate()`**:

```python
class AgentNickChat(BaseChatModel):
    def _generate(self, messages, stop=None, **kw) -> ChatResult:
        text = ollama_generate(
            _messages_to_prompt(messages),
            model=self.model_name,          # BeyondProcwise/AgentNick:unified
            format=self.response_schema.model_json_schema() if self.response_schema else None,
            think=False,                    # hybrid reasoner: else `response` is empty
            temperature=0, stop=stop,
        )
        if text is None:
            raise RuntimeError("AgentNick returned no response")   # never a silent empty success
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def with_structured_output(self, schema, *, method="function_calling", **kw) -> Runnable:
        # `method` accepted and ignored: grammar-constrained decoding is strictly stronger.
        bound = self.model_copy(update={"response_schema": schema})
        return bound | RunnableLambda(lambda m: schema.model_validate_json(m.content))
```

Same module provides `AgentNickEmbeddings`, a LangChain `Embeddings` wrapper over the
GPU-resident `BAAI/bge-large-en-v1.5` already loaded at `base_agent.py:1453`.

Net effect: **no new model, no new API key, no new GPU pressure.** AgentNick is the brain,
reached through the client that already exists, inheriting its semaphore, retries and
`keep_alive` VRAM pinning.

### 3.2 `src/services/hyperextract/schema.py` — the obligation schema

Hyper-Extract's own `legal/contract_obligation.yaml` is bilingual and generic. We define a
procurement-specific schema in Python (grammar-constrained, so the enums are *enforced*, not
merely requested):

```python
class EntityType(str, Enum):
    party = "party"; obligation = "obligation"; trigger = "trigger"
    penalty = "penalty"; deadline = "deadline"; goods = "goods"; document = "document"

class RelType(str, Enum):
    must_perform = "must_perform"; triggered_by = "triggered_by"
    penalised_by = "penalised_by"; entitled_to = "entitled_to"; risk_transfer = "risk_transfer"

class ContractEntity(BaseModel):
    name: str; type: EntityType

class ContractObligation(BaseModel):
    name: str
    type: RelType
    participants: List[str]      # ALL entities in this obligation — the hyperedge
    clause_ref: str              # e.g. "27.4"
    source_quote: str            # COMPLETE sentence, copied word for word
```

`clause_ref` and `source_quote` are **separate fields on purpose**. With a single field the
model fills it with a clause number (`"27.4"`), which then trivially passes a naive
"is this string in the document?" grounding check — a false pass. Splitting them forces a real
span. This was observed, not theorised (§7).

### 3.3 `src/services/hyperextract/obligation_service.py` — extraction + grounding

Takes the contract's parsed text (from the existing L0 parse), runs `AutoHypergraph`, then
passes every hyperedge through a grounding guard. An obligation survives only if its
`source_quote` genuinely appears in the source document. Anything else is **dropped and logged
loudly** — never silently.

**Important correction.** The obvious move — reuse `extraction_v3/grounding.py:is_value_grounded`
— is **wrong here**, and I only found this by reading it. That guard is tuned for *field values*
and grounds a value if **the digit-signature of the value (≥3 digits) appears anywhere in the
document**. For a short field like an invoice total that is a sensible tolerance. For a
*sentence*, it is a hole big enough to drive an invented obligation through.

This is demonstrated, not theorised. Against the real DWP contract:

| Candidate | `is_value_grounded` | `is_quote_grounded` |
|---|---|---|
| *"The Contractor shall indemnify the Authority in full under conditions 27.1 and 27.2 for all consequential loss."* — **entirely fabricated** | **PASS** ❌ | BLOCK ✅ |
| `"27.4"` — a bare clause number | **PASS** ❌ | BLOCK ✅ |
| *"The risk in any over delivered Goods shall remain with the Contractor."* — real, cl. 27.7 | PASS ✅ | PASS ✅ |

The fabricated indemnity clause passes because its digit signature (`271272`) is a substring of
the document's digit stream. A wholly invented liability clause would have been persisted as
grounded fact.

So we reuse its **normalisation** (`_norm` — case, whitespace and punctuation tolerance, which
is what makes it survive OCR/docling formatting drift) but write a quote-specific predicate:

```python
def is_quote_grounded(quote: str, full_text: str) -> bool:
    q = _norm(quote)                      # reuse extraction_v3 normalisation
    if len(q.split()) < 8:      return False   # not a sentence (blocks bare clause numbers)
    if q.replace(" ", "").isdigit(): return False
    return q in _norm(full_text)          # whole-quote containment. No digit fallback.
                                          # No date fallback. No "cannot verify → allow".
```

Strictly containment, with no escape hatches. If we cannot prove the sentence is in the
document, the obligation does not exist.

A knowledge graph is a hallucination surface. An invented obligation carrying a real supplier's
name is exactly the failure mode this codebase keeps producing: *confidently reporting success
it has not earned*. The guard is not optional decoration; it is the feature's safety property.

Required configuration, each one a trap we hit and must encode (§7):

| Setting | Value | Why |
|---|---|---|
| ingest call | `feed_text()` **not** `parse()` | `parse()` returns a NEW instance and leaves the original empty — silent zero results |
| `node_strategy_or_merger` | `MergeStrategy.KEEP_EXISTING` | the default `LLM.BALANCED` has an LLM *rewrite* merged field values, which rewrites `source_quote` and destroys verbatim grounding |
| `edge_strategy_or_merger` | `MergeStrategy.KEEP_EXISTING` | same |
| `max_workers` | `2` | library default is 10; `ollama_client`'s semaphore is 2 |
| `embedder` | real BGE | a stub/zero-vector embedder silently collapses the dedup layer to nothing |
| key extractors | normalised (`"the contractor"` → `"contractor"`) | otherwise the same party duplicates |

### 3.4 Persistence — Postgres

Two new tables, `bp_` prefix per convention:

```sql
proc.bp_contract_obligation (
    obligation_id     bigserial primary key,
    document_id       text not null,
    contract_id       text,
    name              text not null,
    obligation_type   text not null,          -- RelType
    clause_ref        text,
    source_quote      text not null,          -- verbatim, grounded
    grounded          boolean not null,
    confidence        numeric,
    created_date      timestamptz default now()
)
proc.bp_contract_obligation_party (
    obligation_id     bigint references proc.bp_contract_obligation,
    entity_name       text not null,
    entity_type       text not null,          -- EntityType
    primary key (obligation_id, entity_name)
)
```

The party table is what makes the hyperedge queryable: an obligation has *many* parties, and
"which obligations involve supplier X **and** a penalty?" is a join, not a graph traversal.

Extraction status is recorded explicitly. **If AgentNick is unreachable the record is marked
`failed` — never "0 obligations found."**

### 3.5 `src/api/routers/obligations.py` — read API

There is no graph/knowledge/insights router today; this is the first.

- `GET /obligations/contract/{document_id}` — obligations for one contract
- `GET /obligations?party=&type=&expiring_before=` — filtered query
- `POST /obligations/search` — semantic search over the indexed hypergraph

Registered in `src/api/main.py` alongside the existing routers.

### 3.6 Where it hooks in

After a `contract` document promotes in `process_monitor_watcher.py`, enqueue obligation
extraction **asynchronously**. Extraction takes ~55–85 s per 4 KB of contract text, so a full
26 KB contract runs 5–8 minutes. That is fine for background work and **impossible** on a
synchronous request path — the API only ever *reads* what the background job wrote.

---

## 4. Data flow

```
contract PDF
  └─ existing L0 parse (unchanged)            → contract text
       └─ AutoHypergraph.feed_text()          → AgentNick via ollama_client (grammar-constrained)
            ├─ stage 1: extract entities      → ContractEntity[]  (enum-typed)
            ├─ stage 2: extract hyperedges    → ContractObligation[]  (participants = entities)
            └─ prune dangling edges           (library: drops edges naming unknown entities)
                 └─ GROUNDING GUARD           → drop any obligation whose source_quote
                                                 is not verbatim in the document; LOG each drop
                      └─ persist              → bp_contract_obligation (+ _party)
                           └─ GET /obligations
```

Two independent safety nets, in order: the library's structural pruning (an edge cannot name an
entity that does not exist), then our semantic grounding (an obligation cannot cite text that
does not exist).

---

## 5. Failure handling

The governing principle: **the system's failure mode is not crashing, it is confidently
reporting success it has not earned.** Every branch below exists to prevent a green result that
has not been earned.

| Failure | Behaviour |
|---|---|
| AgentNick unreachable / returns nothing | `RuntimeError`; document marked **`failed`**. Never "0 obligations". |
| Model returns an ungrounded obligation | Dropped, logged with the offending quote, counted in the run summary. |
| Model returns an elided/paraphrased quote | Dropped by the guard (proven: §7 dropped exactly such a case). |
| Every obligation dropped as ungrounded | Status **`no_grounded_obligations`** — an explicit, visible state, distinct from "extracted 0". |
| Contract text empty / unparseable | Fail loudly before calling the model. |
| Neo4j | Not used. |

---

## 6. Testing

- **Unit** — `AgentNickChat.with_structured_output` returns validated schema instances; raises
  (never returns empty) when the client returns `None`.
- **Unit** — grounding guard: verbatim quote passes; clause number fails; elided quote fails;
  invented sentence fails.
- **Unit** — key normalisation dedups `"the Contractor"` / `"Contractor"`.
- **Integration** — the real DWP PS1 contract (§7) → assert ≥ 10 grounded obligations, assert
  the known-ungrounded one is dropped, assert every persisted `source_quote` is a substring of
  the source document. This last assertion is the feature's core invariant and is cheap to check.
- **Live** — run through the running local server against `bp_sqldb`, per project practice.

---

## 7. Feasibility: already proven, not assumed

Everything below was executed before this spec was written, against a **real published UK
government procurement contract** (DWP *General Terms and Conditions for the Supply of Goods*,
PS1 — gov.uk), not a fabricated sample.

**Proven:**

1. `pip install hyperextract` adds **15 packages and changes nothing existing** — langchain
   1.2.13, pydantic 2.12.5, openai 2.29.0 all untouched. No dependency conflict. No vendoring
   needed.
2. The `AgentNickChat` bridge **works**: AgentNick, grammar-constrained through the existing
   `ollama_client`, satisfies Hyper-Extract's `with_structured_output(method="function_calling")`.
3. End-to-end on a 4.2 KB slice of the real contract: **31 entities, 13 hyperedges in ~55 s**.
   The obligations are correct — *Risk Transfer at Delivery* (cl. 28.1), *Dispatch Notification
   Requirement* (cl. 29.1), *Over-Delivery Handling* (cl. 27.5) — each holding all its
   participants in one edge.
4. With enum types + strict grounding: **15 obligations kept, 1 dropped.** The dropped one
   ("Non-Delivery Response Protocol") had an elided quote — *"Where the Goods... fail to be
   delivered"* — which does not appear verbatim in the contract. **The guard caught an invented
   span.** That is the safety property working, demonstrated, not hoped for.

**Traps found, each now encoded in §3.3:**

- `parse()` does not mutate; it returns a new instance. Calling `hg.parse(text)` then reading
  `hg.nodes` yields **0 with no error** — a silent-zero failure. Use `feed_text()`.
- The default merge strategy (`MergeStrategy.LLM.BALANCED`) has an **LLM rewrite merged field
  values**, which would rewrite `source_quote` and destroy verbatim grounding.
- A free-text `type` field produces a different vocabulary per chunk (`hyperedge`,
  `contractual obligation`, `Logistics`, `Quantity`…). Enum + grammar constraint fixes it.
- With a single quote field, the model writes a **clause number** into it, which then passes a
  naive grounding check. Hence separate `clause_ref` / `source_quote`.
- A zero-vector stub embedder silently collapses dedup to zero results.

**Known-open, to handle in implementation:**

- Duplicate obligations on the same clause under different `type` values (cl. 27.8 appeared
  twice). Dedup key should include `clause_ref`.
- The 10-word prefix-window fallback in the throwaway spike guard let one elided quote through
  (cl. 30.1). Production uses whole-quote containment with **no** prefix fallback (§3.3).
- `extraction_v3/grounding.py:is_value_grounded` must **not** be reused as-is for quotes — its
  digit-signature escape hatch would pass an invented sentence containing any number found in
  the document. Reuse `_norm` only. See §3.3.

---

## 8. As built (2026-07-14) — where the implementation departs from this spec

Shipped in `e68c824`. Three deliberate departures:

1. **The package is `src/services/obligations/`, not `src/services/hyperextract/`.** The
   latter *shadows the installed library* whenever `src/services` is on `sys.path` — which
   `tests/conftest.py` does. `import hyperextract` then resolves to our package and
   `hyperextract.types` vanishes. Renaming removed the collision.
2. **A third table, `proc.bp_contract_obligation_run`,** was added. Without it, a contract we
   failed to read and a contract with genuinely no obligations both return `[]` from the API,
   and no caller can tell them apart — the green zero this spec exists to prevent. It records
   `extracted | no_grounded_obligations | failed`, and `GET /obligations/contract/{id}` returns
   it alongside the list.
3. **A custom edge-extraction prompt (`EDGE_PROMPT`) was required.** With the library's default
   prompt the model wrote quotes the guard had to reject: it stitched several clauses into one
   quote (`clause_ref: "27.5, 27.6, 27.7"`) or abbreviated with `...`. Those were *true*
   obligations lost to a bad quote — grounded yield was 5–8 of ~14. Instructing the model to
   copy one clause, one sentence, verbatim, no ellipsis, lifted it to **14–19 grounded, 1–3
   dropped**. The fix belonged in the prompt, not in a weaker guard.

Also measured: obligation counts vary run to run (19, then 14, on identical input). Ollama at
`temperature=0` is not bit-deterministic — a known property of this stack. Do not treat a
specific count as a regression signal.

## 9. Success criteria

1. `AgentNickChat` drives Hyper-Extract with no OpenAI/Anthropic key and no non-AgentNick model.
2. Ollama concurrency never exceeds the existing semaphore.
3. Every persisted `source_quote` is verifiably a span of its source document — enforced by test.
4. An unreachable model produces `failed`, never a green zero.
5. A real contract, pushed through the running local server, yields queryable obligations over
   the API.
6. Field-extraction accuracy is **unchanged** — verified by running the existing eval before and
   after.
