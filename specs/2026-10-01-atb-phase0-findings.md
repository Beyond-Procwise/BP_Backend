# ATB Phase 0 — findings, and the rebase onto the RGA

**Date:** 2026-10-01
**Status:** Phase 0 complete. ATB **parked** behind the GPSS document relationship layer.
**Ruling (Nick, 2026-10-01):** rebase the ATB onto the existing RGA; finish GPSS first.

This records what Phase 0 of the *SpendIQ Agent Template Builder* build brief found, so the
work can resume without repeating the discovery. It is deliberately not a plan — the plan is
written when ATB resumes, against §4 below.

---

## 1. §3.1 branch alignment — already satisfied

The brief's branch table is stale. Every checkout is on the right branch and level with origin:

| Repo | Branch | vs origin | Note |
|---|---|---|---|
| `beyond_procwise_ui` | `spendiq-ui` | 0 / 0 | `engine.js` is **24,505** lines, not 5,668. RB6 landmarks all present. |
| `beyond-procwaise-Api` | `spendiq-ui` | 0 / 0 | `src/modules/spendiq/` present (controller, service, `bp_reports_*.sql`). |
| `BP_Backend` | `Development` | 14 ahead | Local commits unpushed, including the GPSS plan `e07333c`. |

No branch work is needed. All line numbers in the brief should be treated as void; bind to
symbol names.

## 2. §0 root-cause table — verified accurate

Each claim was checked against the current code and holds:

| Claim | Verified at |
|---|---|
| Streamed drafts duplicate text | `index.jsx:676` sends **cumulative** text (`streamed += ev.text; handlers.onDelta(streamed)`); `engine.js:22413` appends it again (`streamed+=(d||'')`). A second caller, `engine.js:15447`, names the arg `soFar` and is correct. The comment at `index.jsx:613` ("a piece of answer prose, append it") is also wrong. |
| `call_ollama` drops `format` on the chat path | `src/agents/base_agent.py:1258` calls `ollama.chat(...)` with no `format=`; `ollama.generate` at :1270 passes it. Signature is `format: Optional[str]` and must also accept a dict. |
| 5 of 12 graphs and all 12 section cards are fixtures | `GRAPHS` 12 keys, `LIVE_GRAPHS` 7 → exactly 5 static. `SECTION_CARDS` 12, all static. |
| Renaming a report forks it | `createReport` sets `report_key = $1 = title` (`spendiq.service.ts:~1988`). |

### §3.2 prerequisite fixes — revised status

1. **UI cumulative streaming** — real, outstanding. Fix `streamed += d` → `streamed = d`.
2. **`format` on the chat path** — real, outstanding. Note: per
   `reference_ollama_schema_no_unions`, never put a union (`oneOf`/discriminator) in a
   `format=` schema; Ollama ignores it.
3. **Gateway stable report key** — real, outstanding (optional `key` on `createReport`).
4. **Docs drift** — real but minor: `endpoints.js:39` lists 3 report routes; 6 exist
   (`reports/index`, `reports/key/:key`, `DELETE reports/key/:key` are missing).
5. **`src/agent_definitions.json`** — **already done.** The file no longer exists; only the
   real 17.7 KB root `agent_definitions.json`.

Libraries the brief asks to add are already installed *and* in `requirements.txt`:
`python-pptx`, `python-docx`, `openpyxl`, `jsonschema`.

---

## 3. The blocking finding: the RGA already exists

The brief's reuse map (§2) does not mention the **Report Generation Agent**,
`src/services/rga/` (~180 KB, built 2026-09-16 → 09-25, reachable via `POST /reports/generate`).
It already implements most of what the brief specifies as new, and in several places to a
higher standard.

| Brief proposes to build | Already exists |
|---|---|
| Fact sheet + fact builder with provenance/confidence (§7) | `factpack.py` — `FactEntry` **cannot construct** without a `provenance_id`; a missing one raises a `FACT_WITHOUT_PROVENANCE` finding rather than killing the build |
| "The model never types a number"; `{{f:id}}` tokens (§1) | `compose.py` — `{{F0042}}` refs; the **type system refuses prose containing a digit**, so it is a shape the output cannot take, not a prompt instruction |
| JSON-schema-constrained writer output (§8.2) | `compose.py` — already uses grammar-guided `format=<json schema>` decoding |
| Verifier, rules RB-001…011 (§9) | `postcheck.py` — **stronger**: scans the *rendered artefact*, not agent text, and fails closed. Its digit scan already strips provenance hashes, fact labels and footnote markers |
| Style pack (§5.1) | `style.py` |
| One renderer input → HTML, print, PPTX (§10.1) | `render/html.py` + `render/pptx.py` from one AST; the PPTX is **byte-deterministic** (zip rewritten with fixed timestamps) |
| `render_pptx.py` (§10.3, Phase 5) | `render/pptx.py` |
| `bp_report_build`, `bp_report_build_page`, `bp_report_fact` (3 new tables) | `proc.bp_report_job`, `bp_report_job_page`, `bp_report_version` (+ heartbeat, dismissal, entitlement) |
| Background runs + SSE + progress (§8.3) | `job_runner.py` / `job_store.py` — one active job per report/scope/day, 30 s heartbeat, stranded jobs healed to failed |
| Human approves before release (§1) | `signoff.py` — policy-driven |
| Light editor + versions | `editing.py`, `proc.bp_report_version` |
| Outline templates per report type (§5.3) | `builders/board_paper.py`, `exec_procurement_summary.py`, `supplier_criticality_review.py` + `sections`/`blocks` in `models.py` |

Building the brief as written would create a second report store, a second verifier, a second
PPTX renderer and a second job runner — which the brief's own §1 forbids
(*"stop — that already exists"*).

### Two open questions the brief asks are already answered

- **Tenant scoping (§3.3).** It does not exist and cannot yet. `style.py` and `src/api/auth.py`
  both record that there is no tenant dimension in the corpus and `x-customer-id` is the
  constant `"001"`; `deploy/sql/2026-08-07_commercial_fact.sql` records that RLS is deliberately
  off because "a policy here would be theatre". Keep recording `customer_id`; enforce nothing.
- **Why the local model (§8.1).** Ruled 2026-09-12 in `compose.py`: grammar-guided decoding
  masks invalid tokens during decoding, which is strictly stronger than asking an API for JSON.

### Related: the RGA was paused for this same task

`feedback_rga_real_task_style_from_existing_reports` records that RGA work was paused because
the real requirement is **learning style from existing/uploaded reports**. The ATB's style
packs and its §11 `POST /style-packs/import` are substantially that task. ATB should resume as
the continuation of that pause, not as a parallel programme.

---

## 4. What ATB should actually be

Keep only what the RGA does not have. The composition layer on top of it:

1. **Layout registry** — ~21 fixed page layouts with typed slots and fixed geometry
   (brief §5.2, §6.2, Appendix A). The RGA has components and renderers but no named,
   reusable page compositions with slot schemas.
2. **Outline templates** — ordered sections with `repeat_over` and `depends_on`
   (brief §5.3). The RGA's three builders are hardcoded per report type; this makes the
   document's shape data. Validate `depends_on` by building a `WorkflowGraph` and calling
   `validate()`.
3. **RB6 `layout` block + HTML layout renderer** — brief §10.2. Genuinely new UI.
4. **Outline editor, style editor, build run screen** — brief §12.
5. **Two new fact adapters** — `workbook` (uploaded `.xlsx`) and `client_snapshot`
   (dashboard parity), feeding `factpack.py`'s existing registry rather than a new builder.

Explicitly **do not** build: a new fact store, a new verifier, a new PPTX renderer, new build
tables, a new job runner, or new sign-off. Extend `factpack.py`, `postcheck.py`,
`render/pptx.py`, `job_store.py` and `signoff.py`.

Phase 5 ("style import from document") is the paused RGA task and should be restated and
approved before it starts.

---

## 5. Notes for whoever resumes

- `docs/` is gitignored in BP_Backend (`.gitignore:12 /docs/*`), so the brief's
  `docs/adr/0001-atb-bindings.md` would be uncommittable. `specs/` is tracked — hence this file.
  The three existing ADRs under `docs/adr/` are local-only for the same reason.
- There is already a `src/api/routers/reports.py`. A new `report_builder.py` router must not
  duplicate its job endpoints.
- `src/services/report_builder/` does not exist; the brief's paths assume it does.
