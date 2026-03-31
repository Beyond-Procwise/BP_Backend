# ProcWise Orchestration Rearchitecture & Extraction Pipeline

**Date:** 2026-03-31
**Status:** Draft
**Scope:** Rearchitect agent orchestration layer + extraction pipeline reliability + LLM model strategy

---

## Problem Statement

The current orchestration layer has six compounding issues:

1. **Routing rigidity** — fuzzy matching / token alias routing is brittle and hard to extend
2. **Agent chaining** — difficult to compose multi-agent workflows with proper state passing
3. **Error handling / recovery** — agent failures mid-workflow have no structured recovery path
4. **Observability** — hard to trace what happened across multi-agent executions
5. **Concurrency** — agents that could run in parallel run sequentially (GIL-bound ThreadPoolExecutor)
6. **No distributed execution** — all agents run in a single process

Additionally, the data extraction pipeline — the primary entry point for procurement data — produces inaccurate results due to:

- Field misclassification (data extracted into wrong fields)
- Missing data (fields present in documents not extracted)
- OCR quality variance (scanned documents, photos, mixed digital/paper)
- Inconsistent supplier formats (continuously onboarding new suppliers with unknown layouts)
- No validation gate (bad data enters PostgreSQL silently)

## Constraints

- **Agent internals are untouched** — agents keep their business logic; only the wiring changes
- **Fully local-first LLM** — fine-tuned models hosted in Ollama on NVIDIA A10G (23GB VRAM)
- **Multi-process ready** — agents deployable as independent workers from day one
- **PostgreSQL is the state backbone** — already configured, all procurement data lives there
- **Redis for ephemeral messaging only** — already in stack

---

## Section 1: Component Architecture

The monolithic orchestrator (2,971 lines) decomposes into 5 focused components:

### 1.1 DAG Scheduler

- Accepts workflow requests
- Resolves workflow graph (reuses existing `WorkflowGraph` / `WorkflowNode` / edge conditions)
- Computes topological order
- Determines which nodes are ready (all predecessors completed or skipped)
- Dispatches ready nodes to Task Dispatcher
- Re-evaluates after each node completion

### 1.2 Task Dispatcher

- Publishes agent tasks to Redis Streams
- Each agent type has its own stream and consumer group
- Handles task serialization (AgentContext to message)
- Supports priority lanes (critical vs normal)

### 1.3 Agent Workers

- Independent processes consuming from Redis Streams
- Deserialize message to AgentContext
- Call `agent.execute(context)` (unchanged agent internals)
- Publish AgentOutput to result stream
- ACK message on completion

### 1.4 Result Collector

- Listens on result stream
- Matches results to workflow/node by workflow_id
- Updates WorkflowState (shared_data, node_results) in PostgreSQL
- Notifies DAG Scheduler of completion
- Triggers downstream node evaluation

### 1.5 State Manager

- Owns WorkflowState persistence in PostgreSQL
- Atomic state transitions
- Checkpoint/resume support
- Exposes state queries for API polling

### Communication Pattern

```
API Layer (unchanged routers)
         |
    DAG Scheduler <----> State Manager (PostgreSQL)
         |
    Task Dispatcher
         | (Redis Streams)
    Agent Workers
         | (Redis Streams)
    Result Collector ----> State Manager
         |
    DAG Scheduler (re-evaluate)
```

### What Stays the Same

- `WorkflowGraph`, `WorkflowNode`, edge condition data structures — reused as graph definitions
- `AgentContext`, `AgentOutput` — unchanged dataclasses
- `BaseAgent.execute()` — signature and agent-facing behavior unchanged (see Section 9 for worker bootstrap)
- All 13 agent internals — completely untouched
- `AgentContract` — reused for input validation before dispatch
- `AgentFactory` / `AgentRegistry` — reused inside workers

### What Gets Replaced

- `WorkflowEngine` — replaced by the DAG Scheduler. The engine's sequential topological loop is incompatible with parallel dispatch. The `WorkflowGraph` data structures it consumes are reused; the execution engine is not.
- `Orchestrator` class (2,971 lines) — decomposed into DAG Scheduler + Task Dispatcher + Result Collector + State Manager
- `EventBus` (synchronous pub/sub) — replaced by Redis Streams event protocol
- Model references (Phi4 fallback chains) — updated to Qwen2.5-32B/7B (see Section 8)

### What Gets Modified

- `IMAPSupplierResponseWatcher` — gains Redis Stream publishing capability to emit `reply_received` events (see Section 5). IMAP polling internals unchanged.
- `WorkflowGraph.entry_node` — the DAG Scheduler computes zero-in-degree nodes from graph edges rather than relying on the single `entry_node` field. The field remains for backwards compatibility but is not authoritative.

---

## Section 2: Redis Streams Message Protocol

Three streams handle all communication:

### Stream 1: `agent:tasks:{agent_type}` (Dispatcher to Workers)

One stream per agent type. Message format:

```json
{
    "task_id": "uuid",
    "workflow_id": "wf-123",
    "node_name": "rank_suppliers",
    "agent_type": "supplier_ranking",
    "context": {
        "workflow_id": "wf-123",
        "agent_id": "supplier_ranking",
        "user_id": "user-1",
        "input_data": {},
        "policy_context": [],
        "knowledge_base": {},
        "routing_history": ["data_extraction"],
        "task_profile": {}
    },
    "priority": "normal",
    "dispatched_at": "ISO8601",
    "timeout_seconds": 300,
    "attempt": 1
}
```

### Stream 2: `agent:results` (Workers to Result Collector)

Single stream for all results. Collector routes by workflow_id:

```json
{
    "task_id": "uuid",
    "workflow_id": "wf-123",
    "node_name": "rank_suppliers",
    "agent_type": "supplier_ranking",
    "status": "SUCCESS",
    "data": {},
    "pass_fields": {},
    "next_agents": [],
    "error": null,
    "confidence": 0.92,
    "completed_at": "ISO8601",
    "duration_ms": 4230
}
```

### Stream 3: `workflow:events` (All Components to Observability)

Lifecycle events for tracing and API polling:

```json
{
    "event": "node:started",
    "workflow_id": "wf-123",
    "node_name": "rank_suppliers",
    "timestamp": "ISO8601",
    "metadata": {}
}
```

### Delivery Semantics & Idempotency

Redis Streams consumer groups provide **at-least-once** delivery. If a worker crashes between reading and ACKing a message, the message is re-delivered to another worker via XCLAIM. This means agents must be idempotent.

**Idempotency guard:** Before executing `agent.execute()`, each worker checks `proc.node_execution` for an existing completed result matching the `task_id`. If found, the worker ACKs the message without re-executing. This prevents duplicate email dispatches, duplicate database writes, and duplicate LLM calls on retry.

For agents with external side effects (EmailDispatchAgent, NegotiationAgent), the `task_id` is also written to `proc.dispatch_chain` or `proc.negotiation_sessions` as a deduplication key.

### Why Redis Streams

- Already in the stack
- Consumer groups provide at-least-once delivery with XCLAIM for dead letter recovery
- Persistent (survives restarts, unlike Pub/Sub)
- Lightweight (no broker infrastructure like RabbitMQ)
- XPENDING + XCLAIM provide dead letter handling for free
- Backpressure: streams configured with MAXLEN to cap unbounded growth; dispatcher checks pending message count before publishing

---

## Section 3: DAG Scheduling & Parallel Execution

### Ready-Node Algorithm

```
On workflow start:
    Mark all nodes with zero in-degree as READY
    Dispatch all READY nodes (parallel)

On node completion:
    For each successor of completed node:
        If all predecessors are COMPLETED or SKIPPED:
            Evaluate edge conditions against current WorkflowState
            If conditions pass -> mark READY, dispatch
            If conditions fail -> mark SKIPPED

    If no nodes are READY and no nodes are RUNNING:
        Workflow is complete (or failed if required nodes failed)
```

### Supplier Ranking Workflow (Full)

Two paths based on whether quotes exist:

```
                      data_extraction
                           |
                      supplier_ranking
                           |
                  (has existing quotes?)
                   /                \
                 YES                NO
                  |                  |
          quote_evaluation     email_drafting (RFQ)
                  |                  |
          quote_comparison     email_dispatch
                  |                  |
                  |            email_watcher (await replies)
                  |                  |
                  |            quote_evaluation (new quotes)
                  |                  |
                  |            quote_comparison
                   \                /
                    \              /
                      negotiation
                           |
                    (next_round?)
                    /            \
                  YES             NO -> workflow complete
                   |
              email_drafting (counter-offer)
                   |
              email_dispatch
                   |
              email_watcher
                   |
              negotiation (re-entry)
```

### Multi-Round Negotiation Loop

The DAG is acyclic. Cycles are modeled as workflow re-entry:

1. Negotiation agent output includes `next_round: true`
2. Scheduler increments `current_round` on `proc.workflow_execution`
3. New `proc.node_execution` rows are inserted for the next round's subgraph (email_drafting, dispatch, watcher, negotiation) with `round = N+1`. Previous round's nodes remain as completed historical records.
4. The DAG Scheduler only considers nodes matching the current round when evaluating readiness — this prevents confusion between round N and round N+1 nodes with the same `node_name`.
5. Carries forward cumulative state (all prior rounds in `shared_data`)
6. Stops when `next_round: false` or max rounds reached

### Failure Propagation

| Scenario | Behavior |
|----------|----------|
| Required node fails, retries exhausted | Workflow fails, running nodes cancelled |
| Optional node fails | Mark SKIPPED, evaluate successors normally |
| Node times out | Treat as failure, enter retry logic |
| All successors skipped | Workflow may complete with partial results |

### Cancellation

When a workflow fails or is manually cancelled, the scheduler publishes cancellation messages to active task streams. Workers check cancellation before starting expensive operations.

---

## Section 4: Worker Process Model

### Worker Architecture

Each worker process:

1. Reads from `agent:tasks:{agent_type}` via XREADGROUP
2. Deserializes message to AgentContext
3. Starts timeout guard (threading.Timer)
4. Calls `agent.execute(context)`
5. Publishes AgentOutput to `agent:results` via XADD
6. ACKs the message

### Deployment Modes

| Mode | How | When |
|------|-----|------|
| All-in-one | Single process runs all workers as threads | Development, low traffic |
| Per-type | One process per agent type | Production, moderate traffic |
| Scaled | Multiple processes per agent type | High traffic, heavy agents |

### CLI Entry Point

```bash
# Run all agents in one process
procwise-worker --all

# Run specific agent type(s)
procwise-worker --agents supplier_ranking,negotiation

# Scale a heavy agent
procwise-worker --agents data_extraction --concurrency 4
```

### Worker Behaviors

- **Heartbeat:** Workers send periodic heartbeats to Redis. Scheduler detects dead workers and reclaims pending tasks via XCLAIM.
- **Graceful shutdown:** On SIGTERM, workers finish the current task before exiting.
- **Timeout enforcement:** Per-task timeout_seconds. Guard cancels execution and publishes FAILED result if exceeded.
- **Stateless:** No inter-task state. Agent instances created fresh or pooled per worker.

### Worker Bootstrap & Dependency Injection

The current codebase bundles all dependencies into `agent_nick` — a god-object carrying settings, database connections, all agent instances, policy engine, query engine, and routing engine. Workers cannot import this wholesale.

Each worker process initializes a **lightweight worker context**:

```
WorkerContext
├── settings (Pydantic Settings — same as current)
├── db_pool (psycopg2 connection pool — per-worker, not shared)
├── redis_client (for stream operations)
├── ollama_client (for LLM calls)
├── agent_instances (only the agent types this worker serves)
└── process_routing_service (for audit logging to proc.routing)
```

**Key differences from `agent_nick`:**
- Only instantiates agents this worker is configured to run (not all 13)
- Own connection pool (not shared across process boundary)
- No policy engine or query engine preloaded — these are fetched per-task from the AgentContext which carries policy_context and knowledge_base already populated by the DAG Scheduler before dispatch

The `BaseAgent.execute()` method continues to access services via `self.agent_nick`. In workers, `agent_nick` is replaced by `WorkerContext` which implements the same interface for the subset of services agents actually call during execution.

---

## Section 5: PostgreSQL State Management

### Separation of Concerns

| Layer | Technology | What it stores | Durability |
|-------|-----------|----------------|------------|
| Workflow state | PostgreSQL | DAG progress, node results, shared_data, round counters | Permanent |
| Task dispatch | Redis Streams | In-flight messages between components | Ephemeral |
| Workflow events | Redis Streams -> PostgreSQL | Real-time events for API polling, persisted for audit | Both |

### New Tables (proc schema)

#### proc.workflow_execution

| Column | Type | Purpose |
|--------|------|---------|
| execution_id | PK (SERIAL) | Unique execution identifier |
| workflow_id | TEXT, unique per active | Caller-provided workflow identifier (same convention as existing `proc.routing.workflow_id`). Unique constraint scoped to non-terminal statuses to prevent duplicate active executions. |
| workflow_name | TEXT | Workflow type |
| user_id | TEXT | Triggering user |
| status | TEXT | pending, running, paused, completed, failed, cancelled |
| shared_data | JSONB | Propagated state between nodes |
| current_round | INTEGER | Negotiation round tracking |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |
| completed_at | TIMESTAMP | |

#### proc.node_execution

| Column | Type | Purpose |
|--------|------|---------|
| node_execution_id | PK | Unique node execution identifier |
| execution_id | FK | Parent workflow execution |
| node_name | TEXT | DAG node name |
| agent_type | TEXT | Agent type slug |
| status | TEXT | pending, ready, running, completed, failed, skipped, timed_out |
| attempt | INTEGER | Current retry attempt |
| input_data | JSONB | AgentContext.input_data snapshot |
| output_data | JSONB | AgentOutput.data |
| pass_fields | JSONB | AgentOutput.pass_fields |
| error | TEXT | Error message if failed |
| dispatched_at | TIMESTAMP | |
| completed_at | TIMESTAMP | |
| duration_ms | INTEGER | |
| round | INTEGER | Negotiation round (0 for non-negotiation nodes) |

**Indexes:**
- `(execution_id, status)` — find all running/pending nodes for a workflow
- `(status, dispatched_at)` — find timed-out or stale nodes across all workflows
- `(execution_id, node_name, round)` — unique constraint preventing duplicate node entries per round

### Email Watcher as Event Source

The email watcher is not a blocking DAG node. Instead:

1. Negotiation round dispatches emails
2. `workflow_execution.status` = "paused" (waiting for replies)
3. Email watcher runs independently (existing IMAP polling)
4. Reply arrives -> watcher writes to `proc.supplier_responses` (existing table)
5. Watcher publishes event to Redis Stream: `{workflow_id, supplier_id, "reply_received"}`
6. Result collector picks it up -> updates node_execution -> scheduler resumes workflow

Workflows can sit paused for days with zero resource cost. All state is in PostgreSQL.

### Relationship to Existing proc.routing Table

The existing table continues as the audit/logging layer. New tables handle execution state. No migration of existing data needed — they coexist.

### Crash Recovery

On process restart, the scheduler queries PostgreSQL for workflows with status "running", identifies nodes with status "running" that have no active worker heartbeat, and reclaims them.

---

## Section 6: Extraction Pipeline & Validation Gate

### The Problem

Two input sources feed the system:

1. ERP systems / databases (structured, reliable)
2. PDF / Word documents (scanned POs, invoices, contracts, quotes — unreliable)

Current extraction produces inaccurate results: field misclassification, missing data, OCR quality issues, and inconsistent supplier formats. No validation prevents bad data from entering PostgreSQL. Errors are not reliably caught.

Suppliers are continuously onboarded, so template-based approaches do not work. The pipeline must handle never-before-seen document layouts with unpredictable scan quality.

### Four-Layer Pipeline

#### Layer 1: Adaptive OCR

Input quality detection selects the OCR strategy:

| Input Quality | Strategy |
|--------------|----------|
| Digital PDF | Direct text extraction (no OCR) |
| Clean scan | Standard OCR (EasyOCR / Tesseract) |
| Poor scan | Preprocessing + enhanced OCR |
| Photo / skewed | Deskew + denoise + enhanced OCR |

#### Layer 2: Multi-Strategy Extraction

Three extractors run in parallel, ensemble the results:

- **Strategy A:** Layout-based (PDFPlumber spatial analysis)
- **Strategy B:** LLM-based (Qwen2.5-32B with extraction adapter, structured extraction prompt)
- **Strategy C:** NER-based (entity recognition)

Each strategy produces per-field confidence scores. The ensemble selects the highest-confidence value per field.

**Confidence scoring per strategy:**

| Strategy | Confidence signal |
|----------|-------------------|
| Layout-based (PDFPlumber) | Spatial heuristics: field found in expected region = high; ambiguous position = low. Binary thresholds (0.9 if spatially anchored, 0.4 if not). |
| LLM-based (Qwen2.5-32B) | LLM returns confidence as part of structured output schema. Calibrated via validation set during fine-tuning. |
| NER-based | Entity recognition probability from the NER model (native output). |

**Ensemble algorithm:** Per field, select the value from the strategy with the highest confidence. If the top two strategies agree on a value, boost overall confidence by 0.1 (agreement bonus). If all three disagree, flag the field as low-confidence regardless of individual scores.

**Performance budget:** Strategies A and C are fast (< 1s). Strategy B (LLM) is the bottleneck (~5-15s per document). All three run in parallel via the DAG scheduler's worker model — A and C complete while B is still running. Total extraction latency is bounded by Strategy B.

The existing DataExtractionAgent becomes one strategy within the ensemble. It is wrapped, not rewritten.

#### Layer 3: Validation Gate

Catches errors before data enters PostgreSQL:

**Structural validation:**
- Required fields present?
- Field types correct? (dates, amounts, IDs)
- Cross-field consistency? (line items sum = total)

**Semantic validation:**
- Does supplier name match a known supplier?
- Is PO number format plausible?
- Are amounts within expected ranges?

**Confidence threshold:**
- Per-field confidence from ensemble
- Overall document confidence score
- Below threshold -> route to remediation

#### Layer 4: Remediation Pass

For low-confidence extractions:

1. Re-extract low-confidence fields with an alternative strategy
2. LLM targeted extraction on specific document regions
3. Cross-reference against existing PostgreSQL data (known supplier names, recent PO numbers, price ranges)

If remediation improves confidence above threshold -> commit to PostgreSQL.
If not -> commit with `confidence_score` column and `needs_review` flag. Downstream agents can weight low-confidence data lower.

### DAG Integration

The single `data_extraction` node becomes three nodes:

```
adaptive_ocr -> multi_extraction -> validation_gate
                                        |
                                   (passes?)
                                   /        \
                                 YES         NO
                                  |           |
                            continue      remediation_pass
                            DAG               |
                                        (improved?)
                                        /        \
                                      YES         NO
                                       |           |
                                  continue    commit with
                                  DAG         low_confidence flag
```

### DiscrepancyDetectionAgent Role Shift

The existing DiscrepancyDetectionAgent shifts from catching extraction errors to validating business logic consistency downstream (e.g., invoice amount does not match PO amount). Extraction accuracy is solved before data reaches it.

---

## Section 7: Observability & Tracing

### Event Architecture

Every component publishes structured events:

```
Component -> Redis Stream (real-time) -> Event Persister -> PostgreSQL
                                    |
                              API polling (live status)
```

### proc.workflow_events Table

| Column | Type | Purpose |
|--------|------|---------|
| event_id | PK | |
| workflow_id | FK | |
| node_name | TEXT (nullable) | |
| event_type | TEXT | workflow:started, node:dispatched, node:completed, etc. |
| agent_type | TEXT (nullable) | |
| payload | JSONB | Event-specific data |
| timestamp | TIMESTAMP | |
| round | INTEGER (nullable) | |

### Event Types

| Event | When | Payload |
|-------|------|---------|
| workflow:started | Scheduler accepts request | workflow_name, user_id, node count |
| node:dispatched | Task published to Redis Stream | agent_type, input summary, attempt |
| node:completed | Worker publishes result | duration_ms, confidence, output summary |
| node:failed | Agent returns FAILED | error message, attempt count |
| node:retrying | Scheduler re-dispatches | attempt number, backoff |
| node:timed_out | Timeout guard fires | timeout_seconds, elapsed |
| node:skipped | Edge condition failed | condition name, reason |
| workflow:paused | Waiting for external event | waiting_for, paused_at |
| workflow:resumed | External event received | trigger type |
| workflow:completed | All nodes done | total_duration, node count, success rate |
| extraction:confidence | Validation gate scores | per-field scores, strategy used |

### API Endpoints

```
GET /workflows/{workflow_id}/status     -> current state + node statuses
GET /workflows/{workflow_id}/events     -> event timeline
GET /workflows/{workflow_id}/trace      -> full execution trace
GET /system/workflows/active            -> all running/paused workflows
GET /system/workers/health              -> worker heartbeats, queue depths
```

---

## Section 8: LLM Architecture

### Hardware

- NVIDIA A10G, 23GB VRAM
- CUDA 12.2

### Two-Tier Model Strategy

| Tier | Infrastructure | Models | Use Case |
|------|---------------|--------|----------|
| Local | Ollama on A10G | Qwen2.5-32B (Q4, ~18GB) primary; Qwen2.5-7B (Q4, ~5GB) fallback | Deep extraction, negotiation, complex reasoning |
| Cloud | Ollama hosted API | Lightweight models | Quick classification, summarization, field validation |

### Why Qwen2.5-32B

- Best multilingual understanding at this size class
- Strong structured output capability
- Excels at document comprehension
- Fits A10G with ~5GB headroom at Q4 quantization
- Same architecture as 7B fallback (fine-tuning transfers)
- Outperforms Phi4 (14B) on extraction and structured output
- Denser and better fitting than Mixtral MoE for this VRAM budget

### LLM Router

Routes requests to the appropriate tier and adapter:

| Task | Tier | Adapter |
|------|------|---------|
| Document extraction (ensemble Strategy B) | Local | procurement_extraction |
| Negotiation drafting / counter-offers | Local | procurement_negotiation |
| Quote evaluation / opportunity mining | Local | procurement_general |
| RAG Q&A / policy interpretation | Local | procurement_general |
| Document type classification | Cloud | base |
| Field format validation | Cloud | base |
| Quick summarization | Cloud | base |
| Entity normalization | Cloud | base |

### GPU Concurrency & Request Queuing

**Concurrent capacity:** Ollama natively queues requests to the same model. With `OLLAMA_NUM_PARALLEL=4` (current setting), up to 4 requests share the loaded model's KV cache. At Q4 quantization, Qwen2.5-32B uses ~18GB for weights + ~3-4GB for KV cache at 4K context across 4 parallel slots, totaling ~21-22GB — within the 23GB A10G budget.

**Qwen2.5-7B is not co-resident.** It loads only when the 32B model is explicitly unloaded (e.g., during fine-tuning). In normal operation, only one model is in VRAM at a time.

**When GPU is busy (all 4 slots occupied):**
1. Ollama's internal queue holds the request (default behavior)
2. The LLM Router checks queue depth before routing. If queue depth > 8, non-critical tasks (classification, summarization) are routed to Ollama Cloud instead
3. Critical tasks (extraction, negotiation) always wait for local GPU — they are never degraded to cloud

**During fine-tuning:** Fine-tuning runs during non-business hours via BackendScheduler. The 32B model is unloaded, the 7B fallback model is loaded for any requests that arrive during the training window. Fine-tuning uses the same GPU (~16GB for QLoRA on 32B). After training, the updated adapter is hot-reloaded and the 32B model resumes serving.

### Fine-Tuning Strategy

Three task-specific LoRA adapters via QLoRA (existing Unsloth pipeline):

**Adapter 1: procurement_extraction**
- Training data: corrected extraction outputs, remediated low-confidence fields
- Focus: PO / Invoice / Contract / Quote field extraction

**Adapter 2: procurement_negotiation**
- Training data: completed negotiation rounds with successful outcomes
- Focus: strategy selection, counter-offer drafting

**Adapter 3: procurement_general**
- Training data: policy Q&A pairs, RAG query + validated answers, opportunity analysis
- Focus: domain knowledge, supplier analysis

### Self-Improving Training Loop

Every remediation correction becomes training data:

```
Low-confidence extraction -> remediation pass corrects fields
    -> correction stored in proc.node_execution
    -> BackendScheduler collects corrections during non-business hours
    -> QLoRA fine-tuning updates extraction adapter
    -> New adapter hot-reloaded into Ollama
```

Over time, the extraction adapter improves on the exact document formats suppliers send.

### Ollama Adapter Management

```bash
ollama create procwise-base -f Modelfile.qwen2.5-32b
ollama create procwise-extract -f Modelfile.extraction-adapter
ollama create procwise-negotiate -f Modelfile.negotiation-adapter
ollama create procwise-general -f Modelfile.general-adapter
```

Base weights stay in VRAM. Only small LoRA layers swap per request.

---

---

## Section 9: Migration Strategy

### Phased Rollout

The current `Orchestrator.execute_workflow()` already has a `use_workflow_engine` toggle (line 232) that switches between the new WorkflowEngine and legacy execution paths. This is the natural migration seam.

**Phase 1: Foundation (no behavior change)**
- Create new PostgreSQL tables (`workflow_execution`, `node_execution`, `workflow_events`)
- Build WorkerContext as a lightweight alternative to `agent_nick`
- Build the DAG Scheduler and State Manager as standalone modules
- Wire them behind a new feature flag: `use_dag_scheduler` (extends existing toggle pattern)
- Existing orchestrator continues to serve all traffic

**Phase 2: Shadow mode**
- DAG Scheduler runs in parallel with existing orchestrator on the same workflows
- Results are written to new tables but not used for routing
- Compare outputs between old and new paths to validate correctness
- No user-facing behavior change

**Phase 3: Incremental cutover**
- Switch individual workflow types to the new DAG Scheduler one at a time
- Start with the simplest (document_extraction — linear, no loops)
- Progress to supplier_ranking (parallel branches)
- End with supplier_interaction (negotiation loops, email pausing)
- Rollback per workflow type via the feature flag

**Phase 4: Extraction pipeline**
- Deploy the adaptive OCR + ensemble extraction as new DAG nodes
- Run alongside existing DataExtractionAgent
- Compare extraction accuracy before replacing
- Self-improving loop begins collecting training data

**Phase 5: LLM migration**
- Switch from Phi4 to Qwen2.5-32B
- Update model references in settings and fallback chains
- Deploy LoRA adapters incrementally (extraction first, then negotiation, then general)

**Phase 6: Decommission**
- Remove old `Orchestrator` class, `WorkflowEngine`, synchronous `EventBus`
- Remove `use_workflow_engine` and `use_dag_scheduler` flags
- Clean up Phi4 model references

### Handling In-Flight Workflows During Deployment

Workflows already running when a new version deploys continue on the old path. The feature flag is checked at workflow start time, not per-node. New workflows pick up the new path. This prevents mid-workflow execution model switches.

---

## Section 10: Security & Operational Concerns

### Redis Stream Security

- Redis instance requires AUTH (password authentication)
- TLS enabled for Redis connections (already supported by redis-py)
- AgentContext messages contain `user_id` and `input_data` which may include procurement amounts and supplier names — Redis should not be exposed to public networks
- Stream MAXLEN configured per stream to prevent unbounded memory growth

### Redis Unavailability

If Redis goes down:
- DAG Scheduler pauses workflow dispatch (no new tasks published)
- Workers in-progress finish their current task and write results directly to PostgreSQL (fallback path)
- Result Collector stops consuming but State Manager continues to accept direct writes
- On Redis recovery, scheduler resumes from PostgreSQL state (source of truth)

### Database Connection Management

Each worker process maintains its own connection pool (not shared across process boundaries). Pool size scales with worker concurrency: `min_connections = 2`, `max_connections = concurrency + 2`.

---

## Components Unchanged

- All 13 agent internals (business logic untouched)
- `BaseAgent.execute()` method (signature unchanged; worker bootstrap provides compatible context — see Section 4)
- `AgentContext` and `AgentOutput` dataclasses
- `AgentContract` definitions
- `AgentFactory` and `AgentRegistry`
- `WorkflowGraph`, `WorkflowNode`, edge condition data structures (execution engine replaced — see Section 1)
- Existing `proc.routing` table (continues as audit layer)
- FastAPI API routers (interface unchanged, wiring updated internally)
- IMAP polling internals (watcher gains Redis publishing — see Section 1)
- Existing QLoRA + Unsloth fine-tuning pipeline
