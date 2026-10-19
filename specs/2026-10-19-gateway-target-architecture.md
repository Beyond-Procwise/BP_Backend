# Target architecture (verbatim from the user, 2026-10-18)

## Target architecture (do not deviate without asking)

### Principle
Agents never query data stores directly. All reads and writes go through the
data gateway, a service that sits behind the orchestrator. The orchestrator
owns control flow (which agent runs next, with what context) and is the only
component that calls the gateway. Agents receive data from the orchestrator
and request more through it.

### Components
- **Orchestrator**: plans and routes work between agents, assembles each
  agent's context, calls the gateway, enforces step and retry limits.
  Contains no SQL or Cypher.
- **Data gateway**: the only component with credentials to the data stores.
  Exposes a fixed set of typed, parameterized functions. No LLM-generated
  SQL or Cypher, ever.
- **PostgreSQL**: system of record for records and aggregates: suppliers,
  resellers, customers, parts/SKUs, pricing, quotes, orders, POs, inventory,
  performance.
- **Knowledge graph**: entities, relationships and documents: policies,
  contracts, programmes, approval chains, terms. Entity nodes reference
  Postgres records by Postgres ID and do not duplicate their attributes.
- **Embeddings**: stored in [pgvector], used for semantic matching over KG
  documents and entity names. The gateway owns embedding generation,
  re-indexing on document change, and embedding-model versioning.

### Gateway contract
- Each function answers a named business question, e.g.
  `get_price(sku, customer_id, qty, date)`,
  `get_supplier_performance(supplier_id, period)`,
  `find_applicable_contracts(customer_id, sku, date)`,
  `get_approval_chain(doc_type, amount, business_unit)`,
  `search_documents(query, filters, top_k)`.
- Every call carries caller context (user, role, reseller/tenant). The gateway
  applies row-level access; callers cannot widen their own scope.
- Every result includes provenance: source store, record IDs or document ID
  plus clause/section, and an as-of timestamp.
- Questions spanning both stores are answered inside the gateway by joining
  on shared IDs. Agents do not stitch results from separate calls.
- Results are bounded: pagination or top-k, aggregates preferred over raw rows.

### Source of truth
- Postgres is authoritative for all record attributes and numbers (prices,
  quantities, dates, statuses).
- The KG is authoritative for relationships and document content.
- Entity sync runs one way, Postgres → KG, [event-driven on change]. The KG
  never writes back to Postgres.
- Any number shown to a user comes from Postgres, never from document text
  or an embedding match.

### How agents get data
- The orchestrator prefetches the minimum context each step needs.
- If an agent needs more, it returns a structured data request (function name
  and parameters) instead of guessing. The orchestrator checks it against
  that agent's allowlist, calls the gateway, and resumes the agent.
- Each agent has an explicit allowlist of gateway functions.

### Writes
- Quotes, orders and POs are created and changed only through gateway write
  functions, each with an idempotency key.
- Approval rules (thresholds, chains, required sign-offs) are stored as
  structured data and evaluated deterministically in the gateway. An LLM may
  explain a policy; it never decides whether an approval is satisfied.
- Writes [above a set value, or of types X/Y] require human confirmation
  before commit.

### Audit
- Every gateway call is logged: caller, agent, function, parameters, result
  IDs, latency.
- Every write is logged with actor, agent, inputs, and the provenance of the
  data it relied on.
- User-facing outputs carry the provenance of the facts they cite.

### Requires sign-off
- Adding a data store, giving any component other than the gateway database
  credentials, or adding a free-form query function.
