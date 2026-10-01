# Document Relationship Layer — Implementation Plan (Phase 2 + classification)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the product a governed document-type and relationship vocabulary in the database, and make the extraction path classify every document against it — recording "unknown" or "unresolved" rather than forcing a type, and never rerouting a document away from the pipeline its uploader chose.

**Architecture:** Two new reference tables (`proc.bp_concept`, `proc.bp_document_type`) built in the exact shape of the live `proc.bp_uom_canonical`: aliases in an array on the concept row, `status` of active/proposed/rejected where a proposed row never resolves, and a confirmation trail. A runtime loader mirrors `src/services/facts/uom.py` — cached, version-probed, and falling back to its seed rather than blanking. A new `type_resolver` module is called from `services/extraction/dispatch.py`; it records what the evidence says alongside what the uploader declared, and physical routing continues to come from `process_monitor.category` so no existing upload changes behaviour.

**Tech Stack:** Python 3.12, FastAPI, PostgreSQL (schema `proc`, psycopg2), pytest. No new dependencies.

**Spec:** `/home/muthu/PycharmProjects/BP_Backend/specs/2026-10-01-document-relationship-layer-discovery.md` (Discovery Report, §4 placement decisions and §8 scope) and the GPSS build spec it reports against. Both travel with this plan; executors read the Discovery Report's §2.4 (conventions) and §5 (risks) before task 1.

**Scope:** Discovery Report §8 items 1–6. In: the two reference tables, the runtime loader, map consolidation, the four-type gate, the classification step, review-queue item types, manifest task slices, and the §8.1 validation checks in CI. **Out, and deliberately so:** scope/region precedence (§4.2), Sector_Glossary, scope_loading_guide, and the conflict register with its AI ruling tests — all four need a sector or normalised-region dimension that does not exist (Discovery §4 #4, #5, #9, #11). Link writing and placeholder reconciliation (§4.3, §4.4) are the next plan, and reuse the existing `linking_engine`.

## Global Constraints

Copied from the build spec and the Discovery Report. Every task's requirements implicitly include this section.

- **Principle 1 — Deterministic vocabulary, AI judgment.** Tables define what things are called. The AI never invents a type or role name that bypasses the tables.
- **Principle 2 — Rules score; only impossibilities block.** Default every new rule to `SOFT`.
- **Principle 3 — Exact identifiers link; everything else suggests.**
- **Principle 4 — Nothing is forced.** Record "unknown" or "unresolved" with candidates. Never pick the nearest option to avoid a gap.
- **Principle 5 — Load only what the task needs.** Agents receive a filtered slice, never whole tables.
- **Principle 6 — One source of truth per fact.** Mirrored data is generated from its master, never edited in two places.
- **Principle 7 — Display labels are not aliases.** `bp_translation` labels are never used for matching.
- **Table naming:** all new tables are `proc.bp_*`; indexes are `ix_bp_<table>_<column>`.
- **Migrations:** `deploy/sql/YYYY-MM-DD_name.sql` with a matching `_rollback.sql`. Additive, idempotent, reversible. Seeds use `ON CONFLICT DO NOTHING` so a re-run never clobbers a human confirmation. Apply to **both** `bp_testdb` and `bp_sqldb`.
- **Reference-table shape:** `tenant_id text NOT NULL DEFAULT 'default'`, `aliases text[]`, `status` in (active, proposed, rejected), `source`, `observed_count`, `first_observed_at`, `last_observed_at`, `confirmed_by`, `confirmed_at`, `valid_from`, `valid_to`, `recorded_at`.
- **No fabrication:** absent data stays NULL. A `proposed` row never resolves silently.
- **`get_conn()` is autocommit.** Rollback is a no-op; `FOR UPDATE` locks end with the statement.
- **Ollama:** never put a union (`oneOf`/`discriminator`) in a `format=` JSON schema. (No model call is added by this plan.)
- **Tests:** run with `./venv/bin/python -m pytest` after `set -a && . ./.env && set +a`. DB-backed tests need `PROCWISE_TEST_LIVE_DB=1` and are skipped without it. Isolate from GPU/Ollama with `CUDA_VISIBLE_DEVICES=""`.
- **Shared checkout:** another session's work sits in this repository's git index. **Never `git add -A` and never a bare `git commit`.** Commit named paths only: `git commit -o <paths> -m "..."`. Check `git status` first and stage only your own files.
- **Commit trailer:** end every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **Branch:** work on `Development`. Never push to `main`.

## Review Focus

Five input classes the spec implies but does not pin down, most likely to bite first. Each one's test is added to the task that owns the code, named in brackets.

1. **A category spelled a way the four-entry map happens to cover but the vocabulary does not** — `"PO"`, `"Purchase Order"`, `"PurchaseOrder"` all route today. If the vocabulary seed misses one, a working upload path breaks. Expected: every spelling the old map accepted still resolves. [Task 5]
2. **An empty, whitespace or NULL category** — expected: refuse loudly, exactly as today; never default to a type, and never classify from evidence alone into a physical pipeline. [Task 5]
3. **One alias listed on two concept rows** — expected: the resolver returns `unresolved` with both candidates and writes a review item; it never picks the first row the query returned. [Tasks 3, 6]
4. **The vocabulary table unreadable or returning zero rows** — expected: the loader keeps its last good vocabulary, or falls back to the seed, and logs. It must never blank the vocabulary, because a resolver that recognises nothing marks every document unknown and those absences get recorded as though the documents said nothing. [Task 2]
5. **Evidence naming a concept whose pipeline differs from the declared category** — e.g. a file uploaded as `quote` whose text is plainly an invoice. Expected: the document still runs through the declared pipeline, both readings are recorded, and a non-blocking review item is raised. Routing is never changed by evidence. [Tasks 6, 7]

---

## File Structure

**Created**

| Path | Responsibility |
|---|---|
| `deploy/sql/2026-10-01_concept_vocabulary.sql` | `proc.bp_concept` + `proc.bp_document_type`, constraints, indexes, seed |
| `deploy/sql/2026-10-01_concept_vocabulary_rollback.sql` | Drops both tables |
| `deploy/sql/2026-10-02_bp_rule_relationship_group.sql` | Adds `rule_group`, `blocks_promotion` to `proc.bp_rule` |
| `deploy/sql/2026-10-02_bp_rule_relationship_group_rollback.sql` | Drops the two columns |
| `src/services/concepts/__init__.py` | Package marker, public re-exports |
| `src/services/concepts/seed.py` | The seed vocabulary as Python, asserted against the table |
| `src/services/concepts/vocabulary.py` | Runtime load of both tables: cache, version probe, fail-safe |
| `src/services/concepts/validate.py` | The §8.1 checks as callable functions returning violations |
| `src/services/extraction/type_resolver.py` | Resolve a document type from the declared category + page evidence |
| `tests/services/concepts/test_concept_table.py` | Seed ↔ table agreement; proposed rows never resolve |
| `tests/services/concepts/test_vocabulary_runtime_load.py` | Cache, probe, and the never-blank rule |
| `tests/services/concepts/test_validation_checks.py` | §8.1 checks, each proven to fail on a planted violation |
| `tests/services/extraction/test_type_resolver.py` | Precedence, unknown, unresolved, evidence disagreement |
| `tests/services/test_doc_type_map_single_owner.py` | The physical maps agree with the vocabulary |
| `tests/services/test_language_index_not_matched.py` | No matching path reads `bp_translation` |
| `tests/services/test_agent_manifest_slices.py` | Task slices are filtered and bounded |
| `.github/workflows/reference-data-checks.yml` | Runs the §8.1 checks and the seed-agreement tests in CI |

**Modified**

| Path | Change |
|---|---|
| `src/services/process_monitor_watcher.py:523-532` | The four-entry map and raising gate read the vocabulary |
| `src/services/extraction/dispatch.py` (after `parse_document`, ~line 294) | Call `type_resolver`, record the result, never reroute |
| `src/services/extraction/persistence.py:59-70` | Two new `issue_type` values documented on `Discrepancy` |
| `src/services/agent_manifest.py:238` | `build_manifest` gains a `task_id` slice; knowledge filtered |
| `src/agents/negotiation_agent.py:8046` | Stop serialising the whole knowledge bundle into the prompt |

---

## Task 1: The vocabulary tables

**Files:**
- Create: `deploy/sql/2026-10-01_concept_vocabulary.sql`
- Create: `deploy/sql/2026-10-01_concept_vocabulary_rollback.sql`
- Create: `src/services/concepts/__init__.py`
- Create: `src/services/concepts/seed.py`
- Test: `tests/services/concepts/test_concept_table.py`

**Interfaces:**
- Consumes: nothing.
- Produces: tables `proc.bp_concept` and `proc.bp_document_type`; `seed.CONCEPTS: dict[str, Concept]` and `seed.DOCUMENT_TYPES: dict[str, DocumentType]`, where `Concept` is a frozen dataclass `(concept_code: str, domain: str, definition: str, not_to_be_confused_with: tuple[str, ...], status: str)` and `DocumentType` is a frozen dataclass `(concept_code: str, role: str, default_parent_type: str | None, execution_mode: str | None, aliases: tuple[str, ...], identifiers: tuple[dict, ...], structural_signals: tuple[str, ...], pipeline_doc_type: str | None, status: str)`.

**Design notes the implementer needs:**

`concept_code` is globally unique across domains, so it carries a domain prefix: `doctype.call_off_contract`, `role.master`, `link.calls_off`, `exec.bilateral`, `event.varied`. Without the prefix a role named `variation` and a document type named `variation` would collide on one primary key.

`pipeline_doc_type` is the load-bearing column and the reason this plan does not need ten new table families. Physical `_raw`/`_stg`/`_trgt` tables exist for exactly four types. A new concept such as `doctype.framework_agreement` keeps its own identity while declaring that the `contract` pipeline physically handles it. `NULL` means "recognised, but nothing can ingest it yet" — which is an honest state, not a gap to paper over.

`role` is seeded from the build spec's default eight (framework, master, transaction, variation, attachment, notice, termination, supporting). **These are open question #1 and unconfirmed.** They are seeded as data precisely so renaming one is an `UPDATE`, not a rebuild.

- [ ] **Step 1: Write the failing test**

Create `tests/services/concepts/__init__.py` (empty) and `tests/services/concepts/test_concept_table.py`:

```python
"""The concept tables and seed.py must agree, and unconfirmed concepts must not resolve.

The vocabulary exists twice — as the seed in src/services/concepts/seed.py and as
rows in proc.bp_concept / proc.bp_document_type. Duplication between code and data
is only safe when something fails loudly the moment the two diverge. This is the
same contract tests/services/facts/test_uom_canonical_table.py holds for units.

Live-only. Run with:
    set -a && . ./.env && set +a
    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/concepts/test_concept_table.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.seed import CONCEPTS, DOCUMENT_TYPES  # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_DOMAINS = {
    "DOCUMENT_TYPE", "RELATIONSHIP_ROLE", "LINK_TYPE",
    "EXECUTION_MODE", "EVENT_KIND",
}


@pytest.fixture()
def concept_rows():
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT concept_code, domain, definition, not_to_be_confused_with,
                   status, source, rejection_reason
              FROM proc.bp_concept
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@pytest.fixture()
def doc_type_rows():
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT concept_code, role, default_parent_type, execution_mode,
                   aliases, identifiers, structural_signals, pipeline_doc_type,
                   status
              FROM proc.bp_document_type
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def test_every_seeded_concept_exists_in_the_table(concept_rows):
    in_table = {r["concept_code"] for r in concept_rows}
    missing = set(CONCEPTS) - in_table
    assert not missing, f"concepts in seed.py with no row in bp_concept: {sorted(missing)}"


def test_every_table_concept_exists_in_the_seed(concept_rows):
    """The other direction. Without this, a concept could be added to the
    database and silently never resolve, which looks like the table works."""
    in_table = {r["concept_code"] for r in concept_rows}
    extra = in_table - set(CONCEPTS)
    assert not extra, f"concepts in bp_concept absent from seed.py: {sorted(extra)}"


def test_every_domain_is_one_of_the_five(concept_rows):
    bad = {r["concept_code"]: r["domain"] for r in concept_rows if r["domain"] not in _DOMAINS}
    assert not bad, f"rows with an unrecognised domain: {bad}"


def test_every_concept_code_carries_its_domain_prefix(concept_rows):
    prefix = {
        "DOCUMENT_TYPE": "doctype.",
        "RELATIONSHIP_ROLE": "role.",
        "LINK_TYPE": "link.",
        "EXECUTION_MODE": "exec.",
        "EVENT_KIND": "event.",
    }
    wrong = [
        r["concept_code"] for r in concept_rows
        if not r["concept_code"].startswith(prefix[r["domain"]])
    ]
    assert not wrong, (
        f"concept codes whose prefix does not match their domain: {sorted(wrong)} "
        "— the prefix is what keeps role.variation and doctype.variation on "
        "separate primary keys"
    )


def test_every_active_concept_has_a_definition(concept_rows):
    missing = [
        r["concept_code"] for r in concept_rows
        if r["status"] == "active" and not (r["definition"] or "").strip()
    ]
    assert not missing, f"active concepts with no definition: {sorted(missing)}"


def test_not_to_be_confused_with_points_at_real_concepts(concept_rows):
    known = {r["concept_code"] for r in concept_rows}
    dangling = {}
    for r in concept_rows:
        for other in r["not_to_be_confused_with"] or []:
            if other not in known:
                dangling.setdefault(r["concept_code"], []).append(other)
    assert not dangling, f"not_to_be_confused_with entries that name no concept: {dangling}"


def test_nothing_points_at_itself(concept_rows):
    selfref = [
        r["concept_code"] for r in concept_rows
        if r["concept_code"] in (r["not_to_be_confused_with"] or [])
    ]
    assert not selfref, f"concepts listed as not to be confused with themselves: {selfref}"


def test_every_rejected_concept_records_why(concept_rows):
    """'Fix the uploader' and 'this genuinely is not a document type' are
    different problems. Without the reason they look identical."""
    unexplained = [
        r["concept_code"] for r in concept_rows
        if r["status"] == "rejected" and not (r["rejection_reason"] or "").strip()
    ]
    assert not unexplained, f"rejected with no reason recorded: {sorted(unexplained)}"


def test_every_document_type_row_has_a_concept(concept_rows, doc_type_rows):
    doctypes = {r["concept_code"] for r in concept_rows if r["domain"] == "DOCUMENT_TYPE"}
    orphans = {r["concept_code"] for r in doc_type_rows} - doctypes
    assert not orphans, f"bp_document_type rows with no DOCUMENT_TYPE concept: {sorted(orphans)}"


def test_every_document_type_concept_has_attributes(concept_rows, doc_type_rows):
    doctypes = {
        r["concept_code"] for r in concept_rows
        if r["domain"] == "DOCUMENT_TYPE" and r["status"] != "rejected"
    }
    missing = doctypes - {r["concept_code"] for r in doc_type_rows}
    assert not missing, (
        f"DOCUMENT_TYPE concepts with no bp_document_type row: {sorted(missing)} "
        "— a type with no role or parent cannot take part in a relationship"
    )


def test_roles_parents_and_modes_resolve_to_concepts(concept_rows, doc_type_rows):
    by_domain = {}
    for r in concept_rows:
        by_domain.setdefault(r["domain"], set()).add(r["concept_code"])
    for r in doc_type_rows:
        assert r["role"] in by_domain.get("RELATIONSHIP_ROLE", set()), (
            f"{r['concept_code']}: role {r['role']!r} is not a RELATIONSHIP_ROLE concept"
        )
        if r["default_parent_type"]:
            assert r["default_parent_type"] in by_domain.get("DOCUMENT_TYPE", set()), (
                f"{r['concept_code']}: default_parent_type {r['default_parent_type']!r} "
                "is not a DOCUMENT_TYPE concept"
            )
        if r["execution_mode"]:
            assert r["execution_mode"] in by_domain.get("EXECUTION_MODE", set()), (
                f"{r['concept_code']}: execution_mode {r['execution_mode']!r} "
                "is not an EXECUTION_MODE concept"
            )


def test_seeded_aliases_are_recorded_in_the_table(doc_type_rows):
    table_aliases = {a.lower() for r in doc_type_rows for a in (r["aliases"] or [])}
    seeded = {a.lower() for dt in DOCUMENT_TYPES.values() for a in dt.aliases}
    missing = seeded - table_aliases
    assert not missing, f"aliases in seed.py absent from the table: {sorted(missing)}"


def test_pipeline_doc_type_is_one_the_pipeline_actually_has(doc_type_rows):
    """Four physical table families exist. A fifth value here would route a
    document at a table that is not there."""
    allowed = {"invoice", "purchase_order", "quote", "contract", None}
    bad = {
        r["concept_code"]: r["pipeline_doc_type"] for r in doc_type_rows
        if r["pipeline_doc_type"] not in allowed
    }
    assert not bad, f"pipeline_doc_type values with no physical pipeline: {bad}"


def test_no_alias_is_claimed_by_two_concepts(doc_type_rows):
    """The collision check. One alias on two rows is how a genuine ambiguity
    surfaces, and until the conflict register exists it must not be seeded —
    a resolver facing it can only answer UNRESOLVED."""
    owners: dict[str, list[str]] = {}
    for r in doc_type_rows:
        for alias in r["aliases"] or []:
            owners.setdefault(alias.strip().lower(), []).append(r["concept_code"])
    clashes = {a: sorted(o) for a, o in owners.items() if len(o) > 1}
    assert not clashes, f"aliases claimed by more than one concept: {clashes}"


def test_an_alias_never_equals_another_concepts_code(concept_rows, doc_type_rows):
    """'contract' as an alias of doctype.master_agreement while
    doctype.contract exists would make the resolver's answer depend on which
    table it looked at first."""
    codes = {r["concept_code"].split(".", 1)[-1].lower() for r in concept_rows}
    collisions = {}
    for r in doc_type_rows:
        own = r["concept_code"].split(".", 1)[-1].lower()
        for alias in r["aliases"] or []:
            a = alias.strip().lower().replace(" ", "_")
            if a in codes and a != own:
                collisions.setdefault(r["concept_code"], []).append(alias)
    assert not collisions, f"aliases that are another concept's own name: {collisions}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/concepts/test_concept_table.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'src.services.concepts'`.

- [ ] **Step 3: Write the seed module**

Create `src/services/concepts/__init__.py`:

```python
"""The document and relationship vocabulary.

proc.bp_concept and proc.bp_document_type are the source of truth; seed.py is a
seed that a test asserts matches them. Built in the shape of
proc.bp_uom_canonical (deploy/sql/2026-08-07_uom_canonical.sql): aliases on the
concept row, status active/proposed/rejected where a proposed row NEVER
resolves, and a confirmation trail for the human who promoted it.
"""
from __future__ import annotations

from .seed import CONCEPTS, DOCUMENT_TYPES, Concept, DocumentType

__all__ = ["CONCEPTS", "DOCUMENT_TYPES", "Concept", "DocumentType"]
```

Create `src/services/concepts/seed.py`:

```python
"""The seeded vocabulary, matching deploy/sql/2026-10-01_concept_vocabulary.sql.

Why the duplication: the same reason uom.py keeps _CANONICAL alongside
proc.bp_uom_canonical. Code needs a vocabulary to fall back on when the table
cannot be read, because a resolver that recognises nothing marks every document
unknown — and those absences then look like the documents said nothing. A test
(tests/services/concepts/test_concept_table.py) fails the moment the two diverge.

ASSUMPTION, unconfirmed: the eight role names come from the build spec's
default set (§3.1), which the spec itself marks [CONFIRM]. They are data, so
renaming one is an UPDATE and a seed edit, not a rebuild.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class Concept:
    concept_code: str
    domain: str
    definition: str
    not_to_be_confused_with: Tuple[str, ...] = ()
    status: str = "active"
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class DocumentType:
    concept_code: str
    role: str
    default_parent_type: Optional[str]
    execution_mode: Optional[str]
    aliases: Tuple[str, ...]
    identifiers: Tuple[Mapping[str, str], ...]
    structural_signals: Tuple[str, ...]
    #: Which of the four physical pipelines ingests this type. None means the
    #: type is recognised but nothing can ingest it yet — an honest state.
    pipeline_doc_type: Optional[str]
    status: str = "active"


_ROLES = (
    ("role.framework", "An umbrella agreement under which later contracts are called off."),
    ("role.master", "Governs a relationship and is itself the contract for the work it covers."),
    ("role.transaction", "Orders, confirms or bills a specific quantity or amount."),
    ("role.variation", "Changes the terms of a document that already exists."),
    ("role.attachment", "Has no force alone; takes effect by being incorporated into another document."),
    ("role.notice", "Communicates a fact or an intention; creates no new obligation by itself."),
    ("role.termination", "Ends a document that is already in force."),
    ("role.supporting", "Evidence or working papers around a relationship; not itself binding."),
)

_LINK_TYPES = (
    ("link.calls_off", "The source orders work under the target framework."),
    ("link.governed_by", "The source's terms are set by the target."),
    ("link.incorporates", "The source pulls the target's terms into itself by reference."),
    ("link.varies", "The source changes the target's terms."),
    ("link.supersedes", "The source replaces the target in full."),
    ("link.terminates", "The source ends the target."),
    ("link.attaches_to", "The source is an attachment of the target."),
    ("link.references", "The source mentions the target without changing it."),
)

_EXECUTION_MODES = (
    ("exec.bilateral", "Signed by two parties."),
    ("exec.unilateral", "Issued and signed by one party."),
    ("exec.multilateral", "Signed by three or more parties."),
    ("exec.incorporated", "Not separately executed; takes effect through the document that incorporates it."),
)

_EVENT_KINDS = (
    ("event.signed", "The parties executed the document."),
    ("event.effective", "The document's terms began to apply."),
    ("event.varied", "The document's terms were changed."),
    ("event.renewed", "The document's term was extended on its own renewal terms."),
    ("event.extended", "The document's term was lengthened other than by renewal."),
    ("event.expired", "The document's term ran out."),
    ("event.terminated", "The document was ended before its term ran out."),
    ("event.superseded", "The document was replaced by another."),
)
```

Continue the same file with the document-type concepts. Keep each definition to one line, and set `not_to_be_confused_with` on exactly the pairs the spec worries about:

```python
#: (concept_code, definition, not_to_be_confused_with)
_DOCUMENT_TYPE_CONCEPTS = (
    ("doctype.framework_agreement",
     "Sets terms for future call-offs but orders nothing itself.",
     ("doctype.master_agreement",)),
    ("doctype.master_agreement",
     "Governs a supplier relationship and is the contract for work done under it.",
     ("doctype.framework_agreement", "doctype.call_off_contract")),
    ("doctype.sow",
     "Defines the deliverables, timescale and price of a specific piece of work under a master agreement.",
     ("doctype.call_off_contract", "doctype.order")),
    ("doctype.call_off_contract",
     "The contract formed when work is ordered under a framework; incorporates the framework's terms and sets their precedence.",
     ("doctype.order", "doctype.sow", "doctype.framework_agreement")),
    ("doctype.order",
     "Instructs a supplier to deliver a stated quantity at a stated price.",
     ("doctype.call_off_contract",)),
    ("doctype.invoice", "Demands payment for goods or services supplied.", ()),
    ("doctype.quote", "Offers a price before any order exists.", ()),
    ("doctype.variation",
     "Changes the terms of an existing contract.",
     ("doctype.addendum", "doctype.ccn")),
    ("doctype.schedule",
     "A numbered part of a contract that has no force on its own.",
     ("doctype.addendum", "doctype.sla")),
    ("doctype.addendum",
     "Adds to a contract after signature without replacing it.",
     ("doctype.variation", "doctype.schedule")),
    ("doctype.ccn",
     "Change control note: records an agreed change under a contract's own change procedure.",
     ("doctype.variation",)),
    ("doctype.termination_notice",
     "Ends a contract that is in force.",
     ("doctype.notice_general",)),
    ("doctype.notice_general",
     "Communicates a fact or intention under a contract without ending or changing it.",
     ("doctype.termination_notice",)),
    ("doctype.nda", "Binds the parties to keep information confidential.", ()),
    ("doctype.sla",
     "States service levels and remedies; normally a schedule to an agreement rather than a contract alone.",
     ("doctype.schedule",)),
    ("doctype.service_agreement",
     "Contracts for the supply of a service on stated terms.",
     ("doctype.master_agreement", "doctype.consulting_agreement")),
    ("doctype.consulting_agreement",
     "Contracts for advisory work, usually against time and materials or a retainer.",
     ("doctype.service_agreement",)),
    ("doctype.contract_unspecified",
     "A contract whose kind the document does not state. Recorded as itself rather than guessed at.",
     ()),
    # Observed in proc.bp_contract_master and not yet understood. Proposed, so
    # it is counted and visible but never resolves.
    ("doctype.policy_document",
     "Observed as a contract_type value; what it denotes here is not yet established.",
     ()),
)
```

Then assemble the two public mappings, and give each document type its role, parent, mode, aliases, identifiers, signals and pipeline. The alias lists must include every spelling the current pipeline accepts (`invoice`, `Invoice`, `purchase_order`, `PurchaseOrder`, `po`, `PO`, `quote`, `Quote`, `contract`, `Contract`) or Review Focus item 1 bites:

```python
CONCEPTS: Mapping[str, Concept] = {
    c.concept_code: c
    for c in (
        *(Concept(code, "RELATIONSHIP_ROLE", defn) for code, defn in _ROLES),
        *(Concept(code, "LINK_TYPE", defn) for code, defn in _LINK_TYPES),
        *(Concept(code, "EXECUTION_MODE", defn) for code, defn in _EXECUTION_MODES),
        *(Concept(code, "EVENT_KIND", defn) for code, defn in _EVENT_KINDS),
        *(
            Concept(code, "DOCUMENT_TYPE", defn, confused,
                    status="proposed" if code == "doctype.policy_document" else "active")
            for code, defn, confused in _DOCUMENT_TYPE_CONCEPTS
        ),
    )
}

DOCUMENT_TYPES: Mapping[str, DocumentType] = {
    dt.concept_code: dt for dt in (
        DocumentType(
            "doctype.framework_agreement", "role.framework", None, "exec.bilateral",
            ("framework agreement", "framework", "framework contract"),
            ({"field": "framework_ref", "pattern": r"^[A-Z]{2}\d{4,6}$", "parent_type": None},),
            ("sets terms without ordering", "names a call-off procedure"),
            "contract",
        ),
        DocumentType(
            "doctype.master_agreement", "role.master", None, "exec.bilateral",
            ("master agreement", "msa", "master service agreement",
             "master services agreement"),
            ({"field": "contract_id", "pattern": None, "parent_type": None},),
            ("recites the parties and the relationship", "numbered clauses"),
            "contract",
        ),
        DocumentType(
            "doctype.sow", "role.master", "doctype.master_agreement", "exec.bilateral",
            ("sow", "statement of work", "work order", "task order"),
            ({"field": "contract_id", "pattern": None, "parent_type": "doctype.master_agreement"},),
            ("lists deliverables and milestones", "names the agreement it sits under"),
            "contract",
        ),
        DocumentType(
            "doctype.call_off_contract", "role.master", "doctype.framework_agreement",
            "exec.bilateral",
            ("call-off contract", "call off contract", "call-off", "order form"),
            ({"field": "framework_ref", "pattern": None, "parent_type": "doctype.framework_agreement"},),
            ("lists incorporated documents", "states an order of precedence"),
            "contract",
        ),
        DocumentType(
            "doctype.order", "role.transaction", "doctype.call_off_contract",
            "exec.unilateral",
            ("purchase order", "purchase_order", "purchaseorder", "po", "order"),
            ({"field": "po_id", "pattern": r"^(?:PO)?\d{4,10}$", "parent_type": None},),
            ("line items with quantities and a total", "ship-to address"),
            "purchase_order",
        ),
        DocumentType(
            "doctype.invoice", "role.transaction", "doctype.order", "exec.unilateral",
            ("invoice", "tax invoice", "bill"),
            ({"field": "invoice_id", "pattern": None, "parent_type": None},
             {"field": "po_id", "pattern": None, "parent_type": "doctype.order"}),
            ("amount due and payment terms", "bill-to address"),
            "invoice",
        ),
        DocumentType(
            "doctype.quote", "role.supporting", None, "exec.unilateral",
            ("quote", "quotation", "estimate", "price quotation"),
            ({"field": "quote_id", "pattern": None, "parent_type": None},),
            ("validity or expiry date", "prices with no order reference"),
            "quote",
        ),
        DocumentType(
            "doctype.variation", "role.variation", None, "exec.bilateral",
            ("variation", "variation form", "amendment", "avenant", "deed of variation"),
            ({"field": "amendment_ref", "pattern": None, "parent_type": None},),
            ("names the document it changes", "states what the change is"),
            "contract",
        ),
        DocumentType(
            "doctype.schedule", "role.attachment", None, "exec.incorporated",
            ("schedule", "annex", "appendix", "exhibit"),
            (),
            ("numbered as part of another document", "no signature block"),
            "contract",
        ),
        DocumentType(
            "doctype.addendum", "role.variation", None, "exec.bilateral",
            ("addendum", "supplemental agreement"),
            (),
            ("adds terms after signature", "names the document it supplements"),
            "contract",
        ),
        DocumentType(
            "doctype.ccn", "role.variation", None, "exec.bilateral",
            ("ccn", "change control note", "change note", "change request"),
            (),
            ("cites the contract's change procedure", "states cost and time impact"),
            "contract",
        ),
        DocumentType(
            "doctype.termination_notice", "role.termination", None, "exec.unilateral",
            ("termination notice", "notice of termination"),
            (),
            ("states a termination date", "cites a termination clause"),
            "contract",
        ),
        DocumentType(
            "doctype.notice_general", "role.notice", None, "exec.unilateral",
            ("notice",),
            (),
            ("cites a notice clause", "creates no new obligation"),
            None,
        ),
        DocumentType(
            "doctype.nda", "role.master", None, "exec.bilateral",
            ("nda", "non-disclosure agreement", "confidentiality agreement"),
            (),
            ("defines confidential information", "states a confidentiality period"),
            "contract",
        ),
        DocumentType(
            "doctype.sla", "role.attachment", None, "exec.incorporated",
            ("sla", "service level agreement"),
            (),
            ("service levels with targets", "remedies or service credits"),
            "contract",
        ),
        DocumentType(
            "doctype.service_agreement", "role.master", None, "exec.bilateral",
            ("service agreement", "service contract", "services agreement"),
            (),
            ("describes a service and its term", "numbered clauses"),
            "contract",
        ),
        DocumentType(
            "doctype.consulting_agreement", "role.master", None, "exec.bilateral",
            ("consulting", "consulting agreement", "consultancy agreement"),
            (),
            ("rates or a retainer", "named consultants or roles"),
            "contract",
        ),
        DocumentType(
            "doctype.contract_unspecified", "role.master", None, None,
            ("contract", "agreement"),
            ({"field": "contract_id", "pattern": None, "parent_type": None},),
            ("numbered clauses", "a signature block"),
            "contract",
        ),
        DocumentType(
            "doctype.policy_document", "role.supporting", None, None,
            ("policy",),
            (), (), None,
            status="proposed",
        ),
    )
}
```

- [ ] **Step 4: Write the migration**

Create `deploy/sql/2026-10-01_concept_vocabulary.sql`. Follow `deploy/sql/2026-08-07_uom_canonical.sql` exactly in shape and in commentary style — state what each column is for and what would go wrong without it. Seed every row from `seed.py`.

```sql
-- The document and relationship vocabulary as data rather than as Python dicts.
--
-- Today the list of document types is four words in a dict in
-- src/services/process_monitor_watcher.py:523, and anything else RAISES. The
-- vocabulary is additionally copied into fourteen other module-level maps (see
-- specs/2026-10-01-document-relationship-layer-discovery.md §5.1), so "the list
-- of document types" has no owner. These two tables are that owner.
--
-- Shape follows proc.bp_uom_canonical: aliases on the concept row rather than in
-- a second table, because a second table is a second place to edit one fact.
--
-- status:
--   active   -- usable for resolution
--   proposed -- observed in real data, awaiting human confirmation; NEVER used
--               to resolve, because an unconfirmed guess that silently starts
--               resolving is indistinguishable from a confirmed decision
--   rejected -- confirmed NOT a document type, with the reason recorded
--
-- Additive, idempotent, reversible.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_concept (
    concept_code            text PRIMARY KEY,
    -- Globally unique across domains, so the code carries its domain as a
    -- prefix: role.variation and doctype.variation are different things and
    -- would otherwise collide on this key.
    domain                  text NOT NULL,
    definition              text,
    -- Points at the concepts this one is most often mistaken for. The whole
    -- point of the exercise: 'order form' means one thing under a framework
    -- and another on its own.
    not_to_be_confused_with text[] NOT NULL DEFAULT '{}',
    tenant_id               text NOT NULL DEFAULT 'default',
    status                  text NOT NULL DEFAULT 'proposed',
    source                  text,
    -- Why a rejected row is not a concept. 'fix the uploader' and 'this is
    -- genuinely not a document type' are different problems and look identical
    -- without it.
    rejection_reason        text,
    observed_count          integer NOT NULL DEFAULT 0,
    first_observed_at       timestamptz,
    last_observed_at        timestamptz,
    confirmed_by            text,
    confirmed_at            timestamptz,
    valid_from              timestamptz NOT NULL DEFAULT now(),
    valid_to                timestamptz,
    recorded_at             timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT ck_bp_concept_domain CHECK (domain IN (
        'DOCUMENT_TYPE', 'RELATIONSHIP_ROLE', 'LINK_TYPE',
        'EXECUTION_MODE', 'EVENT_KIND')),
    CONSTRAINT ck_bp_concept_status CHECK (status IN ('active', 'proposed', 'rejected')),
    -- An active concept must say what it means; a proposed one legitimately
    -- does not know yet.
    CONSTRAINT ck_bp_concept_active_has_definition CHECK (
        status <> 'active' OR btrim(coalesce(definition, '')) <> ''),
    CONSTRAINT ck_bp_concept_rejected_has_reason CHECK (
        status <> 'rejected' OR btrim(coalesce(rejection_reason, '')) <> ''),
    CONSTRAINT ck_bp_concept_not_self_confusing CHECK (
        NOT (concept_code = ANY (not_to_be_confused_with)))
);

CREATE INDEX IF NOT EXISTS ix_bp_concept_domain ON proc.bp_concept (domain);
CREATE INDEX IF NOT EXISTS ix_bp_concept_status ON proc.bp_concept (status);

CREATE TABLE IF NOT EXISTS proc.bp_document_type (
    concept_code        text PRIMARY KEY
                        REFERENCES proc.bp_concept (concept_code),
    role                text NOT NULL REFERENCES proc.bp_concept (concept_code),
    default_parent_type text REFERENCES proc.bp_concept (concept_code),
    execution_mode      text REFERENCES proc.bp_concept (concept_code),
    -- Text as it appears on documents. An array rather than a second table,
    -- matching bp_uom_canonical.aliases.
    aliases             text[] NOT NULL DEFAULT '{}',
    -- [{field, pattern, parent_type}]. Matching is on field AND value, never
    -- value alone: a six-digit customer number and a six-digit order number
    -- are not the same identifier.
    identifiers         jsonb NOT NULL DEFAULT '[]',
    structural_signals  text[] NOT NULL DEFAULT '{}',
    -- Which of the four physical pipelines ingests this type. Four table
    -- families exist (invoice, purchase_order, quote, contract); NULL means
    -- the type is recognised but nothing can ingest it yet.
    pipeline_doc_type   text,
    tenant_id           text NOT NULL DEFAULT 'default',
    status              text NOT NULL DEFAULT 'proposed',
    source              text,
    observed_count      integer NOT NULL DEFAULT 0,
    first_observed_at   timestamptz,
    last_observed_at    timestamptz,
    confirmed_by        text,
    confirmed_at        timestamptz,
    valid_from          timestamptz NOT NULL DEFAULT now(),
    valid_to            timestamptz,
    recorded_at         timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT ck_bp_document_type_status CHECK (
        status IN ('active', 'proposed', 'rejected')),
    CONSTRAINT ck_bp_document_type_pipeline CHECK (
        pipeline_doc_type IS NULL OR pipeline_doc_type IN (
            'invoice', 'purchase_order', 'quote', 'contract')),
    CONSTRAINT ck_bp_document_type_identifiers_is_array CHECK (
        jsonb_typeof(identifiers) = 'array')
);

CREATE INDEX IF NOT EXISTS ix_bp_document_type_status
    ON proc.bp_document_type (status);
CREATE INDEX IF NOT EXISTS ix_bp_document_type_pipeline_doc_type
    ON proc.bp_document_type (pipeline_doc_type);
CREATE INDEX IF NOT EXISTS ix_bp_document_type_aliases
    ON proc.bp_document_type USING gin (aliases);

COMMENT ON TABLE proc.bp_concept IS
    'The document and relationship vocabulary. Only status=''active'' rows '
    'resolve; ''proposed'' rows are observed-but-unconfirmed and must never '
    'resolve silently.';
COMMENT ON TABLE proc.bp_document_type IS
    'Per-document-type attributes and aliases. pipeline_doc_type names which '
    'of the four physical pipelines ingests the type; NULL means none yet.';

COMMIT;
```

Then, in the same file before `COMMIT`, insert every concept and document type from `seed.py` with `ON CONFLICT (concept_code) DO NOTHING` and `source = 'seed'`. Insert roles, link types, execution modes and event kinds first, then document-type concepts, then `bp_document_type` rows — the foreign keys require that order.

Create `deploy/sql/2026-10-01_concept_vocabulary_rollback.sql`:

```sql
-- Reverses 2026-10-01_concept_vocabulary.sql.
BEGIN;
DROP TABLE IF EXISTS proc.bp_document_type;
DROP TABLE IF EXISTS proc.bp_concept;
COMMIT;
```

- [ ] **Step 5: Apply the migration to both databases**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d bp_testdb \
    -v ON_ERROR_STOP=1 -f deploy/sql/2026-10-01_concept_vocabulary.sql
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d bp_sqldb \
    -v ON_ERROR_STOP=1 -f deploy/sql/2026-10-01_concept_vocabulary.sql
```

Expected: `COMMIT` on both, no error. Verify the row counts match the seed:

```bash
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d bp_testdb -At -c \
  "select domain, count(*) from proc.bp_concept group by 1 order by 1;
   select count(*) from proc.bp_document_type;"
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
    tests/services/concepts/test_concept_table.py -v
```

Expected: all tests PASS. If `test_no_alias_is_claimed_by_two_concepts` fails, two seeded types share an alias — fix the seed, do not weaken the test: an unresolvable alias must not be seeded before the conflict register exists.

- [ ] **Step 7: Prove the collision guard fails on purpose**

Temporarily add `"invoice"` to `doctype.quote`'s alias tuple in `seed.py`, re-run the table test, and confirm `test_an_alias_never_equals_another_concepts_code` and the table-vs-seed alias test go **red**. Then revert the edit and confirm green again. A guard that has never been seen to fail is not known to work.

- [ ] **Step 8: Commit**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
git status --short   # confirm only your files are listed below
git commit -o \
  deploy/sql/2026-10-01_concept_vocabulary.sql \
  deploy/sql/2026-10-01_concept_vocabulary_rollback.sql \
  src/services/concepts/__init__.py \
  src/services/concepts/seed.py \
  tests/services/concepts/__init__.py \
  tests/services/concepts/test_concept_table.py \
  -m "feat(concepts): document and relationship vocabulary as data

proc.bp_concept and proc.bp_document_type, in the shape of
proc.bp_uom_canonical: aliases on the concept row, status where a
proposed row never resolves, and a confirmation trail. Applied to
bp_testdb and bp_sqldb.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Runtime vocabulary loader

**Files:**
- Create: `src/services/concepts/vocabulary.py`
- Test: `tests/services/concepts/test_vocabulary_runtime_load.py`

**Interfaces:**
- Consumes: `seed.CONCEPTS`, `seed.DOCUMENT_TYPES`, `seed.Concept`, `seed.DocumentType` (Task 1); `src.services.db.get_conn`.
- Produces:
  - `@dataclass(frozen=True) Vocabulary` with fields `concepts: Mapping[str, Concept]`, `document_types: Mapping[str, DocumentType]`, `alias_index: Mapping[str, tuple[str, ...]]`, `source: str`.
  - `SEED_VOCABULARY: Vocabulary`
  - `build_vocabulary(concept_rows, doc_type_rows, *, source) -> Vocabulary` — pure, no I/O.
  - `ensure_vocabulary(*, ttl_seconds: float = 300.0, probe_seconds: float = 15.0) -> Vocabulary`
  - `invalidate() -> None`
  - `resolve_alias(text: str, vocabulary: Vocabulary) -> tuple[str, ...]` — returns every active document-type `concept_code` claiming that alias, lowercased and whitespace-collapsed. Empty tuple when none. More than one means a genuine collision.
  - `fold(text: str) -> str` — the one normalisation used for every alias comparison.

**Design notes the implementer needs:**

Copy the structure of `src/services/facts/uom.py` lines 160–400: `_ACTIVE_SQL`, `_VERSION_SQL` (the cheap two-scalar probe over `count(*)` **and** `max(recorded_at)` — neither alone is sufficient, because editing a row without adding one leaves the count identical, and a row added while another is removed leaves it identical too), a module-level `threading.Lock`, a `_DEFAULT_TTL_SECONDS` backstop and a `_DEFAULT_PROBE_SECONDS` freshness probe.

Three properties this must have, each of which is a way it goes wrong:

- **A failed or empty load must never blank the vocabulary.** Keep the last good one; fall back to `SEED_VOCABULARY` only if nothing has ever loaded. A resolver that recognises nothing marks every document unknown, and those absences get recorded as though the documents stated nothing.
- **No query in the per-document path.** `ensure_vocabulary` is called once per document; the probe is what keeps the common case cheap.
- **Only `status = 'active'` loads, filtered in SQL.** A `proposed` concept is an observation awaiting a human; if it resolved, confirming it would be moot.

`alias_index` maps a folded alias to a tuple of concept codes, so a collision is representable rather than lost. Building it is where a one-alias-two-concepts clash becomes visible at runtime.

- [ ] **Step 1: Write the failing test**

Create `tests/services/concepts/test_vocabulary_runtime_load.py`:

```python
"""The loader must stay useful when the table does not.

These are pure-unit tests over build_vocabulary and the cache, with fake rows —
no database. The live agreement between table and seed is
tests/services/concepts/test_concept_table.py's job.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import vocabulary as V  # noqa: E402


CONCEPT_ROWS = [
    {"concept_code": "role.master", "domain": "RELATIONSHIP_ROLE",
     "definition": "Governs a relationship.", "not_to_be_confused_with": [],
     "status": "active", "rejection_reason": None},
    {"concept_code": "doctype.invoice", "domain": "DOCUMENT_TYPE",
     "definition": "Demands payment.", "not_to_be_confused_with": [],
     "status": "active", "rejection_reason": None},
    {"concept_code": "doctype.policy_document", "domain": "DOCUMENT_TYPE",
     "definition": None, "not_to_be_confused_with": [],
     "status": "proposed", "rejection_reason": None},
]

DOC_TYPE_ROWS = [
    {"concept_code": "doctype.invoice", "role": "role.master",
     "default_parent_type": None, "execution_mode": None,
     "aliases": ["invoice", "Tax Invoice"], "identifiers": [],
     "structural_signals": ["amount due"], "pipeline_doc_type": "invoice",
     "status": "active"},
    {"concept_code": "doctype.policy_document", "role": "role.master",
     "default_parent_type": None, "execution_mode": None,
     "aliases": ["policy"], "identifiers": [],
     "structural_signals": [], "pipeline_doc_type": None,
     "status": "proposed"},
]


def test_build_vocabulary_indexes_aliases_case_and_space_insensitively():
    v = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="test")
    assert V.resolve_alias("  TAX   invoice ", v) == ("doctype.invoice",)


def test_a_proposed_type_never_resolves():
    """The load-bearing rule of the status column. If a proposed type resolved,
    the review step would be decoration."""
    v = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="test")
    assert V.resolve_alias("policy", v) == ()
    assert "doctype.policy_document" not in v.document_types


def test_a_colliding_alias_returns_both_candidates():
    """One alias on two active rows is a genuine ambiguity. Returning one of
    them would make the answer depend on row order."""
    rows = DOC_TYPE_ROWS + [{
        "concept_code": "doctype.quote", "role": "role.master",
        "default_parent_type": None, "execution_mode": None,
        "aliases": ["invoice"], "identifiers": [],
        "structural_signals": [], "pipeline_doc_type": "quote",
        "status": "active",
    }]
    concepts = CONCEPT_ROWS + [{
        "concept_code": "doctype.quote", "domain": "DOCUMENT_TYPE",
        "definition": "Offers a price.", "not_to_be_confused_with": [],
        "status": "active", "rejection_reason": None,
    }]
    v = V.build_vocabulary(concepts, rows, source="test")
    assert set(V.resolve_alias("invoice", v)) == {"doctype.invoice", "doctype.quote"}


def test_an_unknown_alias_resolves_to_nothing_rather_than_a_guess():
    v = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="test")
    assert V.resolve_alias("bill of lading", v) == ()


def test_an_empty_load_keeps_the_previous_vocabulary(monkeypatch):
    """Review Focus 4. An empty table must not blank the vocabulary: every
    document would come back unknown and the absences would be recorded as
    though the pages said nothing."""
    good = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="good")
    monkeypatch.setattr(V, "_active", good, raising=False)
    monkeypatch.setattr(V, "_loaded_at", 0.0, raising=False)
    monkeypatch.setattr(V, "_fetch_rows", lambda: ([], []))
    V.invalidate()
    result = V.ensure_vocabulary()
    assert result.source == "good"
    assert V.resolve_alias("invoice", result) == ("doctype.invoice",)


def test_an_unreadable_table_keeps_the_previous_vocabulary(monkeypatch):
    good = V.build_vocabulary(CONCEPT_ROWS, DOC_TYPE_ROWS, source="good")
    monkeypatch.setattr(V, "_active", good, raising=False)
    monkeypatch.setattr(V, "_loaded_at", 0.0, raising=False)

    def boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(V, "_fetch_rows", boom)
    V.invalidate()
    result = V.ensure_vocabulary()
    assert result.source == "good"


def test_with_nothing_ever_loaded_it_falls_back_to_the_seed(monkeypatch):
    monkeypatch.setattr(V, "_active", V.SEED_VOCABULARY, raising=False)
    monkeypatch.setattr(V, "_loaded_at", None, raising=False)
    monkeypatch.setattr(V, "_fetch_rows", lambda: ([], []))
    V.invalidate()
    result = V.ensure_vocabulary()
    assert result.source == "builtin-seed"
    # The seed must be able to resolve the four spellings the pipeline needs.
    for spelling in ("invoice", "purchase order", "quote", "contract"):
        assert V.resolve_alias(spelling, result), f"seed cannot resolve {spelling!r}"


def test_the_seed_resolves_every_spelling_the_old_map_accepted():
    """Review Focus 1. process_monitor_watcher's four-entry map accepted these
    ten spellings. Losing one breaks a working upload path."""
    v = V.SEED_VOCABULARY
    for spelling in ("invoice", "Invoice", "purchase_order", "PurchaseOrder",
                     "po", "PO", "quote", "Quote", "contract", "Contract"):
        assert V.resolve_alias(spelling, v), f"seed cannot resolve {spelling!r}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/concepts/test_vocabulary_runtime_load.py -v
```

Expected: collection error — `ImportError: cannot import name 'vocabulary'`.

- [ ] **Step 3: Write the loader**

Create `src/services/concepts/vocabulary.py`:

```python
"""Runtime load of the document and relationship vocabulary.

proc.bp_concept and proc.bp_document_type are the source of truth; seed.py is
the fallback. Loading at runtime means a confirmed type takes effect without a
deploy, which is the whole reason the vocabulary became data.

Three properties this must have, each a way it goes wrong:

  * A failed or EMPTY load must never blank the vocabulary. A resolver that
    recognises nothing marks every document unknown, and those absences get
    recorded as though the documents stated nothing. A slightly stale
    vocabulary is enormously preferable to a confident wrong silence.
  * No query in the per-document path. The version probe is what keeps the
    common case (nothing changed) cheap.
  * Only status='active' loads, filtered in SQL. A 'proposed' concept is an
    observation awaiting a human; if it resolved, confirming it would be moot.

This mirrors src/services/facts/uom.py, deliberately: one loading pattern for
reference data is easier to reason about than two.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .seed import CONCEPTS, DOCUMENT_TYPES, Concept, DocumentType

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

_CONCEPT_SQL = """
    SELECT concept_code, domain, definition, not_to_be_confused_with,
           status, rejection_reason, recorded_at
      FROM proc.bp_concept
     WHERE status = 'active'
"""

_DOC_TYPE_SQL = """
    SELECT concept_code, role, default_parent_type, execution_mode,
           aliases, identifiers, structural_signals, pipeline_doc_type,
           status, recorded_at
      FROM proc.bp_document_type
     WHERE status = 'active'
"""

# The version probe. Two scalars over two small tables rather than the
# vocabulary itself: it runs far more often than a reload. count AND
# max(recorded_at), because neither alone is sufficient — editing a row without
# adding one leaves the count identical, and a row added while another is
# removed leaves it identical too.
_VERSION_SQL = """
    SELECT (SELECT count(*) FROM proc.bp_concept WHERE status = 'active'),
           (SELECT max(recorded_at) FROM proc.bp_concept WHERE status = 'active'),
           (SELECT count(*) FROM proc.bp_document_type WHERE status = 'active'),
           (SELECT max(recorded_at) FROM proc.bp_document_type WHERE status = 'active')
"""

_DEFAULT_TTL_SECONDS = 300.0
_DEFAULT_PROBE_SECONDS = 15.0


def fold(text: str) -> str:
    """The one normalisation used for every alias comparison.

    Lowercase, collapse internal whitespace, treat '_' and '-' as spaces so
    'purchase_order' and 'Purchase Order' are the same alias, and drop a
    trailing full stop.
    """
    t = (text or "").replace("_", " ").replace("-", " ")
    t = _WHITESPACE.sub(" ", t).strip().lower()
    return t[:-1] if t.endswith(".") else t


@dataclass(frozen=True)
class Vocabulary:
    """A resolved vocabulary and where it came from."""

    concepts: Mapping[str, Concept]
    document_types: Mapping[str, DocumentType]
    #: folded alias -> every active document-type concept_code claiming it.
    #: A tuple, not a str: more than one entry is a genuine collision and must
    #: stay representable rather than be silently narrowed to the first row.
    alias_index: Mapping[str, Tuple[str, ...]]
    source: str


def build_vocabulary(
    concept_rows: Iterable[Mapping[str, Any]],
    doc_type_rows: Iterable[Mapping[str, Any]],
    *,
    source: str,
) -> Vocabulary:
    """Assemble a Vocabulary from table rows. Pure — no I/O."""
    concepts: Dict[str, Concept] = {}
    for row in concept_rows:
        code = (row.get("concept_code") or "").strip()
        if not code or row.get("status") != "active":
            continue
        concepts[code] = Concept(
            concept_code=code,
            domain=str(row.get("domain") or ""),
            definition=row.get("definition") or "",
            not_to_be_confused_with=tuple(row.get("not_to_be_confused_with") or ()),
            status="active",
            rejection_reason=row.get("rejection_reason"),
        )

    document_types: Dict[str, DocumentType] = {}
    alias_index: Dict[str, Tuple[str, ...]] = {}
    for row in doc_type_rows:
        code = (row.get("concept_code") or "").strip()
        if not code or row.get("status") != "active":
            continue
        aliases = tuple(row.get("aliases") or ())
        identifiers = tuple(row.get("identifiers") or ())
        document_types[code] = DocumentType(
            concept_code=code,
            role=str(row.get("role") or ""),
            default_parent_type=row.get("default_parent_type"),
            execution_mode=row.get("execution_mode"),
            aliases=aliases,
            identifiers=identifiers,
            structural_signals=tuple(row.get("structural_signals") or ()),
            pipeline_doc_type=row.get("pipeline_doc_type"),
            status="active",
        )
        # The concept's own name is always an alias of itself.
        for alias in (*aliases, code.split(".", 1)[-1]):
            key = fold(alias)
            if not key:
                continue
            existing = alias_index.get(key, ())
            if code not in existing:
                alias_index[key] = (*existing, code)

    for key, owners in alias_index.items():
        if len(owners) > 1:
            logger.warning(
                "alias %r is claimed by %d concepts (%s) — it can only resolve "
                "to UNRESOLVED until a ruling settles it",
                key, len(owners), ", ".join(owners),
            )

    return Vocabulary(
        concepts=concepts,
        document_types=document_types,
        alias_index=alias_index,
        source=source,
    )


SEED_VOCABULARY = build_vocabulary(
    [
        {
            "concept_code": c.concept_code,
            "domain": c.domain,
            "definition": c.definition,
            "not_to_be_confused_with": list(c.not_to_be_confused_with),
            "status": c.status,
            "rejection_reason": c.rejection_reason,
        }
        for c in CONCEPTS.values()
    ],
    [
        {
            "concept_code": d.concept_code,
            "role": d.role,
            "default_parent_type": d.default_parent_type,
            "execution_mode": d.execution_mode,
            "aliases": list(d.aliases),
            "identifiers": list(d.identifiers),
            "structural_signals": list(d.structural_signals),
            "pipeline_doc_type": d.pipeline_doc_type,
            "status": d.status,
        }
        for d in DOCUMENT_TYPES.values()
    ],
    source="builtin-seed",
)

_lock = threading.Lock()
_active: Vocabulary = SEED_VOCABULARY
_loaded_at: Optional[float] = None
_probed_at: float = 0.0
_version: Optional[Tuple[Any, ...]] = None
_invalidated: bool = False


def resolve_alias(text: str, vocabulary: Vocabulary) -> Tuple[str, ...]:
    """Every active document-type concept_code claiming ``text`` as an alias.

    Empty when nothing claims it — which is an answer ("unknown"), not a
    failure. More than one is a genuine collision and the caller must treat it
    as UNRESOLVED rather than choosing.
    """
    return vocabulary.alias_index.get(fold(text), ())


def invalidate() -> None:
    """Force the next ``ensure_vocabulary`` to re-read, ignoring the TTL."""
    global _invalidated
    with _lock:
        _invalidated = True


def _fetch_rows() -> Tuple[list, list]:
    """Read both tables. Separated so tests can replace it."""
    from src.services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_CONCEPT_SQL)
        cols = [d[0] for d in (cur.description or [])]
        concept_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute(_DOC_TYPE_SQL)
        cols = [d[0] for d in (cur.description or [])]
        doc_type_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return concept_rows, doc_type_rows


def _fetch_version() -> Optional[Tuple[Any, ...]]:
    from src.services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_VERSION_SQL)
        row = cur.fetchone()
    return tuple(row) if row else None


def ensure_vocabulary(
    *,
    ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    probe_seconds: float = _DEFAULT_PROBE_SECONDS,
) -> Vocabulary:
    """The current vocabulary, reloading when it is stale or has changed.

    Never raises and never returns an empty vocabulary.
    """
    global _active, _loaded_at, _probed_at, _version, _invalidated

    now = time.monotonic()
    with _lock:
        fresh_enough = (
            _loaded_at is not None
            and not _invalidated
            and (now - _loaded_at) < ttl_seconds
        )
        due_a_probe = _loaded_at is not None and (now - _probed_at) >= probe_seconds
        current = _active

    if fresh_enough and not due_a_probe:
        return current

    if fresh_enough and due_a_probe:
        # Still inside the TTL: ask the cheap question, and only then the
        # expensive one.
        try:
            version = _fetch_version()
        except Exception:
            logger.debug("vocabulary version probe failed; keeping current", exc_info=True)
            with _lock:
                _probed_at = now
            return current
        with _lock:
            _probed_at = now
            unchanged = version == _version
        if unchanged:
            return current

    try:
        concept_rows, doc_type_rows = _fetch_rows()
    except Exception:
        logger.exception(
            "could not read proc.bp_concept / proc.bp_document_type; continuing "
            "with the %s vocabulary", current.source,
        )
        with _lock:
            _probed_at = now
            _invalidated = False
        return current

    candidate = build_vocabulary(
        concept_rows, doc_type_rows,
        source=f"bp_concept@{len(concept_rows)}+bp_document_type@{len(doc_type_rows)}",
    )
    if not candidate.document_types:
        logger.error(
            "proc.bp_document_type yielded no active types from %d row(s); "
            "keeping the %s vocabulary rather than recognising nothing",
            len(doc_type_rows), current.source,
        )
        with _lock:
            _probed_at = now
            _invalidated = False
        return current

    try:
        version = _fetch_version()
    except Exception:
        version = None

    with _lock:
        _active = candidate
        _loaded_at = now
        _probed_at = now
        _version = version
        _invalidated = False
    logger.info(
        "loaded %d active concepts and %d active document types",
        len(candidate.concepts), len(candidate.document_types),
    )
    return candidate


__all__ = [
    "Vocabulary", "SEED_VOCABULARY", "build_vocabulary", "ensure_vocabulary",
    "invalidate", "resolve_alias", "fold",
]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/concepts/test_vocabulary_runtime_load.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Prove the never-blank guard fails on purpose**

Temporarily change `ensure_vocabulary` so the `if not candidate.document_types:` branch assigns `_active = candidate` instead of returning `current`. Re-run; `test_an_empty_load_keeps_the_previous_vocabulary` must go **red**. Revert and confirm green. This is the guard that stops an outage from being recorded as "every document is unknown".

- [ ] **Step 6: Confirm it reads the live tables**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.services.concepts.vocabulary import ensure_vocabulary, resolve_alias
v = ensure_vocabulary()
print('source:', v.source)
print('types:', len(v.document_types), 'concepts:', len(v.concepts))
for s in ('order form', 'PO', 'avenant', 'statement of work', 'bill of lading'):
    print(f'  {s!r} ->', resolve_alias(s, v))
"
```

Expected: `source` names the tables (not `builtin-seed`), and `'bill of lading'` resolves to `()`.

- [ ] **Step 7: Commit**

```bash
git status --short
git commit -o \
  src/services/concepts/vocabulary.py \
  tests/services/concepts/test_vocabulary_runtime_load.py \
  -m "feat(concepts): runtime vocabulary loader

Cached, version-probed read of bp_concept and bp_document_type, mirroring
facts/uom.py. An empty or unreadable table keeps the last good vocabulary
rather than recognising nothing. A colliding alias returns both candidates.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: The §8.1 validation checks, and CI that runs them

**Files:**
- Create: `src/services/concepts/validate.py`
- Create: `tests/services/concepts/test_validation_checks.py`
- Create: `.github/workflows/reference-data-checks.yml`

**Interfaces:**
- Consumes: `vocabulary.Vocabulary`, `vocabulary.fold`, `seed.CONCEPTS`, `seed.DOCUMENT_TYPES`.
- Produces:
  - `@dataclass(frozen=True) Violation(check: str, subject: str, detail: str)`
  - `check_aliases_are_unambiguous(vocabulary) -> list[Violation]`
  - `check_every_reference_resolves(vocabulary) -> list[Violation]`
  - `check_active_concepts_are_defined(vocabulary) -> list[Violation]`
  - `check_pipeline_targets_exist(vocabulary) -> list[Violation]`
  - `run_all(vocabulary) -> list[Violation]`

**Design note:** the build spec's §8.1 lists six checks. Four are implementable now and are below. The remaining two are deliberately absent and that absence is the honest state: *"conflict_rulings matches what is generated from its master"* has no subject, because the Discovery Report recommends never creating a generated mirror (§7.3); *"every R-REL rule has a severity and every SOFT rule a penalty"* has no subject until R-REL exists (see **Scope deviation** at the end of this plan). Writing either as a check that passes over an empty set would be a guard that reports success while checking nothing.

The checks take a `Vocabulary` rather than reading the database, so the same function serves the live check and the unit test.

- [ ] **Step 1: Write the failing test**

Create `tests/services/concepts/test_validation_checks.py`:

```python
"""Each §8.1 check, proven to fail on a planted violation.

A validation function that has only ever been seen to pass is not known to
check anything. Every test here builds a vocabulary that breaks the rule and
asserts the check catches it, then asserts the real seed is clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import validate as VAL  # noqa: E402
from src.services.concepts import vocabulary as V  # noqa: E402


def _vocab(concepts, doc_types):
    return V.build_vocabulary(concepts, doc_types, source="test")


def _concept(code, domain, definition="x", confused=()):
    return {"concept_code": code, "domain": domain, "definition": definition,
            "not_to_be_confused_with": list(confused), "status": "active",
            "rejection_reason": None}


def _doc_type(code, role="role.master", aliases=(), parent=None, mode=None,
              pipeline="contract"):
    return {"concept_code": code, "role": role, "default_parent_type": parent,
            "execution_mode": mode, "aliases": list(aliases), "identifiers": [],
            "structural_signals": [], "pipeline_doc_type": pipeline,
            "status": "active"}


def test_colliding_alias_is_caught():
    v = _vocab(
        [_concept("role.master", "RELATIONSHIP_ROLE"),
         _concept("doctype.a", "DOCUMENT_TYPE"),
         _concept("doctype.b", "DOCUMENT_TYPE")],
        [_doc_type("doctype.a", aliases=("order form",)),
         _doc_type("doctype.b", aliases=("Order Form",))],
    )
    violations = VAL.check_aliases_are_unambiguous(v)
    assert violations, "a shared alias must be reported"
    assert "order form" in violations[0].subject


def test_dangling_role_reference_is_caught():
    v = _vocab(
        [_concept("doctype.a", "DOCUMENT_TYPE")],
        [_doc_type("doctype.a", role="role.nonexistent")],
    )
    violations = VAL.check_every_reference_resolves(v)
    assert any("role.nonexistent" in x.detail for x in violations)


def test_dangling_parent_reference_is_caught():
    v = _vocab(
        [_concept("role.master", "RELATIONSHIP_ROLE"),
         _concept("doctype.a", "DOCUMENT_TYPE")],
        [_doc_type("doctype.a", parent="doctype.ghost")],
    )
    violations = VAL.check_every_reference_resolves(v)
    assert any("doctype.ghost" in x.detail for x in violations)


def test_dangling_not_to_be_confused_with_is_caught():
    v = _vocab(
        [_concept("role.master", "RELATIONSHIP_ROLE"),
         _concept("doctype.a", "DOCUMENT_TYPE", confused=("doctype.ghost",))],
        [_doc_type("doctype.a")],
    )
    violations = VAL.check_every_reference_resolves(v)
    assert any("doctype.ghost" in x.detail for x in violations)


def test_undefined_active_concept_is_caught():
    v = _vocab([_concept("doctype.a", "DOCUMENT_TYPE", definition="  ")], [])
    violations = VAL.check_active_concepts_are_defined(v)
    assert any(x.subject == "doctype.a" for x in violations)


def test_unknown_pipeline_target_is_caught():
    v = _vocab(
        [_concept("role.master", "RELATIONSHIP_ROLE"),
         _concept("doctype.a", "DOCUMENT_TYPE")],
        [_doc_type("doctype.a", pipeline="shipping_note")],
    )
    violations = VAL.check_pipeline_targets_exist(v)
    assert any("shipping_note" in x.detail for x in violations)


def test_a_document_type_with_no_attributes_row_is_caught():
    """A type with no role or parent cannot take part in a relationship, so a
    DOCUMENT_TYPE concept with no bp_document_type row is a gap, not a choice."""
    v = _vocab([_concept("role.master", "RELATIONSHIP_ROLE"),
                _concept("doctype.orphan", "DOCUMENT_TYPE")], [])
    violations = VAL.check_every_reference_resolves(v)
    assert any(x.subject == "doctype.orphan" for x in violations)


def test_the_real_seed_passes_every_check():
    assert VAL.run_all(V.SEED_VOCABULARY) == []
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/concepts/test_validation_checks.py -v
```

Expected: collection error — `cannot import name 'validate'`.

- [ ] **Step 3: Write the checks**

Create `src/services/concepts/validate.py`:

```python
"""The reference-data checks from the build spec §8.1.

Each returns a list of Violation. An empty list is a pass. They take a
Vocabulary rather than a connection so the same function serves CI, a live
check and a unit test.

Two of the spec's six checks are deliberately absent, and the absence is the
honest state rather than an omission:

  * "conflict_rulings matches what is generated from its master" has no
    subject. The Discovery Report (§7.3) recommends one table holding a
    collision and its ruling together, so there is no mirror to drift.
  * "every R-REL rule has a severity and every SOFT rule a penalty" has no
    subject until R-REL exists.

Either, written now, would be a check that passes over an empty set — which
reports success while checking nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .vocabulary import Vocabulary, fold


@dataclass(frozen=True)
class Violation:
    check: str
    subject: str
    detail: str


def check_aliases_are_unambiguous(vocabulary: Vocabulary) -> List[Violation]:
    """No alias may be claimed by more than one active document type.

    The spec asks that such an alias carry a Conflict_Register entry with a
    ruling. There is no conflict register, so the only honest state is that no
    seeded alias collides — a collision the resolver cannot settle can only
    produce UNRESOLVED.
    """
    out: List[Violation] = []
    for alias, owners in sorted(vocabulary.alias_index.items()):
        if len(owners) > 1:
            out.append(Violation(
                "aliases_are_unambiguous", alias,
                f"claimed by {len(owners)} concepts: {', '.join(sorted(owners))} "
                "— no ruling exists to settle it",
            ))
    return out


def check_every_reference_resolves(vocabulary: Vocabulary) -> List[Violation]:
    """Every role, parent type, execution mode and not_to_be_confused_with
    entry must name a concept that exists, and every DOCUMENT_TYPE concept must
    have an attributes row."""
    out: List[Violation] = []
    known = set(vocabulary.concepts)
    by_domain = {}
    for code, concept in vocabulary.concepts.items():
        by_domain.setdefault(concept.domain, set()).add(code)

    for code, concept in sorted(vocabulary.concepts.items()):
        for other in concept.not_to_be_confused_with:
            if other not in known:
                out.append(Violation(
                    "every_reference_resolves", code,
                    f"not_to_be_confused_with names {other!r}, which is not a concept",
                ))

    for code in sorted(by_domain.get("DOCUMENT_TYPE", set())):
        if code not in vocabulary.document_types:
            out.append(Violation(
                "every_reference_resolves", code,
                "DOCUMENT_TYPE concept has no bp_document_type row, so it has "
                "no role and cannot take part in a relationship",
            ))

    for code, dt in sorted(vocabulary.document_types.items()):
        if dt.role not in by_domain.get("RELATIONSHIP_ROLE", set()):
            out.append(Violation(
                "every_reference_resolves", code,
                f"role {dt.role!r} is not a RELATIONSHIP_ROLE concept",
            ))
        if dt.default_parent_type and dt.default_parent_type not in by_domain.get("DOCUMENT_TYPE", set()):
            out.append(Violation(
                "every_reference_resolves", code,
                f"default_parent_type {dt.default_parent_type!r} is not a "
                "DOCUMENT_TYPE concept",
            ))
        if dt.execution_mode and dt.execution_mode not in by_domain.get("EXECUTION_MODE", set()):
            out.append(Violation(
                "every_reference_resolves", code,
                f"execution_mode {dt.execution_mode!r} is not an EXECUTION_MODE concept",
            ))
    return out


def check_active_concepts_are_defined(vocabulary: Vocabulary) -> List[Violation]:
    """An active concept with no definition is a name with no meaning behind
    it, and the next reader will supply their own."""
    return [
        Violation("active_concepts_are_defined", code,
                  "active concept has no definition")
        for code, concept in sorted(vocabulary.concepts.items())
        if not (concept.definition or "").strip()
    ]


#: The four physical table families the extraction pipeline actually has.
_PIPELINES = frozenset({"invoice", "purchase_order", "quote", "contract"})


def check_pipeline_targets_exist(vocabulary: Vocabulary) -> List[Violation]:
    """pipeline_doc_type must name a pipeline that exists, or be absent.

    A fifth value would route a document at _raw/_stg/_trgt tables that are not
    there, and the failure would surface as a SQL error mid-extraction.
    """
    return [
        Violation("pipeline_targets_exist", code,
                  f"pipeline_doc_type {dt.pipeline_doc_type!r} names no physical pipeline")
        for code, dt in sorted(vocabulary.document_types.items())
        if dt.pipeline_doc_type is not None and dt.pipeline_doc_type not in _PIPELINES
    ]


def run_all(vocabulary: Vocabulary) -> List[Violation]:
    """Every check, in a stable order."""
    out: List[Violation] = []
    out.extend(check_aliases_are_unambiguous(vocabulary))
    out.extend(check_every_reference_resolves(vocabulary))
    out.extend(check_active_concepts_are_defined(vocabulary))
    out.extend(check_pipeline_targets_exist(vocabulary))
    return out


__all__ = [
    "Violation", "run_all",
    "check_aliases_are_unambiguous", "check_every_reference_resolves",
    "check_active_concepts_are_defined", "check_pipeline_targets_exist",
]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/concepts/test_validation_checks.py -v
```

Expected: all 8 tests PASS. `test_the_real_seed_passes_every_check` failing means the seed from Task 1 has a genuine problem — fix the seed, not the check.

- [ ] **Step 5: Run the checks against the live tables**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.services.concepts.vocabulary import ensure_vocabulary
from src.services.concepts.validate import run_all
v = ensure_vocabulary()
bad = run_all(v)
print('vocabulary:', v.source)
for x in bad:
    print(f'  {x.check}: {x.subject} — {x.detail}')
print('violations:', len(bad))
raise SystemExit(1 if bad else 0)
"
```

Expected: `violations: 0` and exit status 0.

- [ ] **Step 6: Add the CI workflow**

Create `.github/workflows/reference-data-checks.yml`, following the env-placeholder pattern of `.github/workflows/formula-goldens.yml` (`config/settings.py` refuses to load without those variables; nothing here connects to any of them):

```yaml
name: Reference data checks

# The document and relationship vocabulary is data, so a bad edit cannot be
# caught by a type checker. These checks are what turns "two types now claim the
# same alias" from something noticed in production into something noticed in the
# pull request.

on:
  push:
    branches: [main, Development]
  pull_request:

jobs:
  reference-data:
    runs-on: ubuntu-latest
    env:
      DB_HOST: ci-placeholder
      DB_PORT: "5432"
      DB_NAME: ci-placeholder
      DB_USER: ci-placeholder
      DB_PASSWORD: ci-placeholder
      S3_BUCKET_NAME: ci-placeholder
      S3_PREFIXES: '["ci"]'
      QDRANT_URL: http://127.0.0.1:9
      QDRANT_API_KEY: ci-placeholder
      SES_DEFAULT_SENDER: ci-placeholder
      CUDA_VISIBLE_DEVICES: ""
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pytest pyyaml psycopg2-binary

      # No database: these run over the seed, which a live test
      # (tests/services/concepts/test_concept_table.py) separately asserts
      # matches the tables.
      - name: Validation checks over the seeded vocabulary
        run: |
          python -c "
          import sys; sys.path.insert(0, '.')
          from src.services.concepts.vocabulary import SEED_VOCABULARY
          from src.services.concepts.validate import run_all
          bad = run_all(SEED_VOCABULARY)
          for x in bad:
              print(f'{x.check}: {x.subject} - {x.detail}')
          sys.exit(1 if bad else 0)
          "

      - name: Vocabulary unit tests
        run: |
          python -m pytest -q \
            tests/services/concepts/test_validation_checks.py \
            tests/services/concepts/test_vocabulary_runtime_load.py \
            tests/services/test_language_index_not_matched.py
```

Note: the last test file is created in Task 9. Add it to this workflow then, not now — a workflow referencing a missing file fails the run.

- [ ] **Step 7: Verify the workflow's commands locally before committing**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.services.concepts.vocabulary import SEED_VOCABULARY
from src.services.concepts.validate import run_all
bad = run_all(SEED_VOCABULARY)
for x in bad: print(f'{x.check}: {x.subject} - {x.detail}')
sys.exit(1 if bad else 0)
"; echo "exit=$?"
```

Expected: `exit=0`.

- [ ] **Step 8: Commit**

```bash
git status --short
git commit -o \
  src/services/concepts/validate.py \
  tests/services/concepts/test_validation_checks.py \
  .github/workflows/reference-data-checks.yml \
  -m "feat(concepts): reference-data validation checks in CI

Four of the build spec's six §8.1 checks, each proven to fail on a planted
violation. The two omitted ones have no subject yet and are documented as
such rather than shipped as checks that pass over an empty set.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: One owner for the document-type vocabulary

**Files:**
- Create: `tests/services/test_doc_type_map_single_owner.py`
- Modify: nothing yet — this task's deliverable is the guard that makes Task 5 safe.

**Interfaces:**
- Consumes: `vocabulary.SEED_VOCABULARY`; the existing physical maps named below.
- Produces: no new runtime code.

**Design note — why this is a test and not a refactor.** The Discovery Report (§5.1) found fifteen module-level maps keyed by document type across ten files. Most are *physical*: a type-to-table map, a type-to-primary-key map. Those have to exist somewhere, and rewriting ten files that sit on the live extraction path to funnel through one accessor is a large change with no behavioural benefit and real risk. What is actually needed is the guarantee the spec asks for — that no map drifts from the vocabulary — and a test delivers that for a fraction of the risk. Renovate, don't rewrite.

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_doc_type_map_single_owner.py`:

```python
"""The physical document-type maps must agree with the vocabulary.

The Discovery Report (§5.1) found fifteen module-level maps keyed by document
type across ten files. They are not consolidated — several are genuinely
physical (type to table, type to primary key) and must exist. What must not
happen is drift: a type the vocabulary knows and a map does not, or the
reverse. This test is the single owner, enforced rather than refactored.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.concepts.vocabulary import SEED_VOCABULARY  # noqa: E402


def _pipelines_in_the_vocabulary() -> set[str]:
    return {
        dt.pipeline_doc_type
        for dt in SEED_VOCABULARY.document_types.values()
        if dt.pipeline_doc_type
    }


def test_the_vocabulary_names_exactly_the_four_live_pipelines():
    """A fifth pipeline value means tables that do not exist; a missing one
    means a working ingestion path has no vocabulary entry."""
    assert _pipelines_in_the_vocabulary() == {
        "invoice", "purchase_order", "quote", "contract"
    }


def test_persistence_knows_a_primary_key_for_every_pipeline():
    from src.services.extraction.persistence import _DOC_PK_FIELD
    missing = _pipelines_in_the_vocabulary() - set(_DOC_PK_FIELD)
    assert not missing, (
        f"pipelines with no primary-key field in persistence._DOC_PK_FIELD: "
        f"{sorted(missing)}"
    )


def test_kg_sync_knows_a_table_and_key_for_every_pipeline():
    from src.services.extraction.kg_sync import _PK_COL, _TRGT_TABLE
    for name, mapping in (("_TRGT_TABLE", _TRGT_TABLE), ("_PK_COL", _PK_COL)):
        missing = _pipelines_in_the_vocabulary() - set(mapping)
        assert not missing, (
            f"pipelines absent from kg_sync.{name}: {sorted(missing)} — "
            "those documents would never reach the knowledge graph"
        )


def test_provenance_knows_a_parent_table_for_every_pipeline():
    from src.services.extraction_v2.provenance import PARENT_TABLE_FOR_DOC_TYPE
    missing = _pipelines_in_the_vocabulary() - set(PARENT_TABLE_FOR_DOC_TYPE)
    assert not missing, (
        f"pipelines absent from PARENT_TABLE_FOR_DOC_TYPE: {sorted(missing)}"
    )


def test_an_extraction_schema_exists_for_every_pipeline():
    schema_dir = Path(__file__).resolve().parents[2] / "extraction_schemas"
    present = {p.stem for p in schema_dir.glob("*.yaml")}
    missing = _pipelines_in_the_vocabulary() - present
    assert not missing, (
        f"pipelines with no extraction_schemas/<name>.yaml: {sorted(missing)}"
    )


def test_every_legacy_category_value_resolves_in_the_vocabulary():
    """utils.procurement_schema.CATEGORY_TO_DOC_TYPE is what the older path
    accepted. Every key must still resolve, or an upload that worked stops."""
    from src.services.concepts.vocabulary import resolve_alias
    from utils.procurement_schema import CATEGORY_TO_DOC_TYPE
    unresolved = [
        key for key in CATEGORY_TO_DOC_TYPE
        if not resolve_alias(key, SEED_VOCABULARY)
    ]
    assert not unresolved, (
        f"legacy category values the vocabulary cannot resolve: {sorted(unresolved)}"
    )
```

- [ ] **Step 2: Run the test**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/test_doc_type_map_single_owner.py -v
```

Expected: PASS if Task 1's seed is right. A failure here is a real gap in the seed — fix `seed.py` and re-apply the seed rows to both databases, do not relax the assertion.

- [ ] **Step 3: Prove the drift guard fails on purpose**

Temporarily add `("doctype.delivery_note", ..., pipeline_doc_type="delivery_note")` to `seed.DOCUMENT_TYPES`. Re-run; `test_the_vocabulary_names_exactly_the_four_live_pipelines`, `test_persistence_knows_a_primary_key_for_every_pipeline` and `test_an_extraction_schema_exists_for_every_pipeline` must all go **red**. Revert and confirm green.

- [ ] **Step 4: Commit**

```bash
git status --short
git commit -o tests/services/test_doc_type_map_single_owner.py \
  -m "test(concepts): the physical doc-type maps must agree with the vocabulary

Enforces one owner for the document-type list without refactoring ten files
on the live extraction path.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: The four-type gate reads the vocabulary

**Files:**
- Create: `src/services/concepts/routing.py`
- Modify: `src/services/process_monitor_watcher.py:523-532`
- Test: `tests/services/concepts/test_category_routing.py`

**Interfaces:**
- Consumes: `vocabulary.ensure_vocabulary`, `vocabulary.resolve_alias`, `vocabulary.Vocabulary`.
- Produces:
  - `class UnknownDocumentCategory(RuntimeError)`
  - `class AmbiguousDocumentCategory(RuntimeError)`
  - `pipeline_for_category(category: str | None, *, vocabulary: Vocabulary | None = None) -> tuple[str, str]` — returns `(pipeline_doc_type, concept_code)`. Raises `UnknownDocumentCategory` when nothing resolves or the resolved type has no pipeline, and `AmbiguousDocumentCategory` when more than one type claims the alias.

**What is being replaced.** `src/services/process_monitor_watcher.py:523-532` currently reads:

```python
            doc_type_map = {
                "invoice": "invoice", "Invoice": "invoice",
                "purchase_order": "purchase_order", "PurchaseOrder": "purchase_order",
                "po": "purchase_order", "PO": "purchase_order",
                "quote": "quote", "Quote": "quote",
                "contract": "contract", "Contract": "contract",
            }
            doc_type = doc_type_map.get(category, category.lower() if category else "")
            if doc_type not in ("invoice", "purchase_order", "quote", "contract"):
                raise RuntimeError(f"unsupported doc_type: {category!r}")
```

**It must keep raising.** Discovery §6.4: today an unsupported category surfaces as a hard error a human sees. The gate becoming permissive would *lose* signal. What changes is only that the set of acceptable spellings now comes from the vocabulary, so adding a type is a database row rather than a code edit — and the error message names what was tried.

- [ ] **Step 1: Write the failing test**

Create `tests/services/concepts/test_category_routing.py`:

```python
"""What the uploader typed, turned into a physical pipeline — or refused.

Review Focus 1 and 2 live here: every spelling the old four-entry map accepted
must still route, and an empty category must refuse rather than default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import routing as R  # noqa: E402
from src.services.concepts.vocabulary import SEED_VOCABULARY, build_vocabulary  # noqa: E402

V = SEED_VOCABULARY


@pytest.mark.parametrize("category,expected", [
    ("invoice", "invoice"),
    ("Invoice", "invoice"),
    ("purchase_order", "purchase_order"),
    ("PurchaseOrder", "purchase_order"),
    ("po", "purchase_order"),
    ("PO", "purchase_order"),
    ("quote", "quote"),
    ("Quote", "quote"),
    ("contract", "contract"),
    ("Contract", "contract"),
])
def test_every_spelling_the_old_map_accepted_still_routes(category, expected):
    """Review Focus 1. Losing one of these breaks a working upload path."""
    pipeline, concept = R.pipeline_for_category(category, vocabulary=V)
    assert pipeline == expected
    assert concept.startswith("doctype.")


@pytest.mark.parametrize("category", ["", "   ", None])
def test_an_empty_category_refuses_rather_than_defaulting(category):
    """Review Focus 2. Guessing a type for a document nobody labelled is
    exactly the forcing the spec's principle 4 forbids."""
    with pytest.raises(R.UnknownDocumentCategory):
        R.pipeline_for_category(category, vocabulary=V)


def test_an_unrecognised_category_refuses_and_says_what_it_tried():
    with pytest.raises(R.UnknownDocumentCategory) as exc:
        R.pipeline_for_category("bill of lading", vocabulary=V)
    assert "bill of lading" in str(exc.value)


def test_a_new_type_routes_as_soon_as_it_is_in_the_vocabulary():
    """The point of the exercise: adding a type is a row, not a code edit."""
    pipeline, concept = R.pipeline_for_category("framework agreement", vocabulary=V)
    assert pipeline == "contract"
    assert concept == "doctype.framework_agreement"


def test_a_recognised_type_with_no_pipeline_refuses():
    """doctype.notice_general is a real concept that nothing can ingest yet.
    Routing it at the contract tables because 'contract' is the closest match
    is the forcing principle 4 forbids."""
    with pytest.raises(R.UnknownDocumentCategory) as exc:
        R.pipeline_for_category("notice", vocabulary=V)
    assert "no pipeline" in str(exc.value).lower()


def test_an_ambiguous_category_refuses_rather_than_choosing():
    """Review Focus 3. If two types claim the alias, routing to either one is
    a coin toss recorded as a fact."""
    colliding = build_vocabulary(
        [
            {"concept_code": "role.master", "domain": "RELATIONSHIP_ROLE",
             "definition": "x", "not_to_be_confused_with": [], "status": "active",
             "rejection_reason": None},
            {"concept_code": "doctype.a", "domain": "DOCUMENT_TYPE",
             "definition": "x", "not_to_be_confused_with": [], "status": "active",
             "rejection_reason": None},
            {"concept_code": "doctype.b", "domain": "DOCUMENT_TYPE",
             "definition": "x", "not_to_be_confused_with": [], "status": "active",
             "rejection_reason": None},
        ],
        [
            {"concept_code": "doctype.a", "role": "role.master",
             "default_parent_type": None, "execution_mode": None,
             "aliases": ["order form"], "identifiers": [],
             "structural_signals": [], "pipeline_doc_type": "contract",
             "status": "active"},
            {"concept_code": "doctype.b", "role": "role.master",
             "default_parent_type": None, "execution_mode": None,
             "aliases": ["order form"], "identifiers": [],
             "structural_signals": [], "pipeline_doc_type": "purchase_order",
             "status": "active"},
        ],
        source="test",
    )
    with pytest.raises(R.AmbiguousDocumentCategory) as exc:
        R.pipeline_for_category("order form", vocabulary=colliding)
    assert "doctype.a" in str(exc.value) and "doctype.b" in str(exc.value)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/concepts/test_category_routing.py -v
```

Expected: collection error — `cannot import name 'routing'`.

- [ ] **Step 3: Write the routing module**

Create `src/services/concepts/routing.py`:

```python
"""Turn the category the uploader typed into a physical pipeline.

This replaces the four-entry dict in process_monitor_watcher.py:523. Two things
change and one deliberately does not.

Changes: the set of acceptable spellings now comes from proc.bp_document_type,
so adding a document type is a row rather than a code edit; and a refusal says
what it was given and why.

Does NOT change: an unrecognised category still RAISES. Today that surfaces as
a hard error a human sees, and a permissive gate would lose that signal — see
the Discovery Report §6.4. "Unknown" becomes expressible on the classification
side (type_resolver), where it costs nothing; it does not become a licence to
guess a pipeline.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .vocabulary import Vocabulary, ensure_vocabulary, resolve_alias


class UnknownDocumentCategory(RuntimeError):
    """Nothing in the vocabulary claims this category, or it has no pipeline."""


class AmbiguousDocumentCategory(RuntimeError):
    """More than one active document type claims this category as an alias.

    Raised rather than resolved: picking one would make the answer depend on
    row order, and the pipeline it chose would be recorded as a fact.
    """


def pipeline_for_category(
    category: Optional[str],
    *,
    vocabulary: Optional[Vocabulary] = None,
) -> Tuple[str, str]:
    """``(pipeline_doc_type, concept_code)`` for an uploader's category label."""
    vocab = vocabulary if vocabulary is not None else ensure_vocabulary()
    raw = (category or "").strip()
    if not raw:
        raise UnknownDocumentCategory(
            "no document category was supplied; refusing to guess a type for an "
            "unlabelled document"
        )

    candidates = resolve_alias(raw, vocab)
    if not candidates:
        raise UnknownDocumentCategory(
            f"no document type in proc.bp_document_type claims {raw!r} as an "
            "alias (only status='active' rows resolve)"
        )
    if len(candidates) > 1:
        raise AmbiguousDocumentCategory(
            f"{raw!r} is claimed by {len(candidates)} document types "
            f"({', '.join(sorted(candidates))}); no ruling exists to settle it"
        )

    concept_code = candidates[0]
    pipeline = vocab.document_types[concept_code].pipeline_doc_type
    if not pipeline:
        raise UnknownDocumentCategory(
            f"{raw!r} resolves to {concept_code} which has no pipeline "
            "(pipeline_doc_type is NULL): the type is recognised but nothing "
            "can ingest it yet"
        )
    return pipeline, concept_code


__all__ = [
    "pipeline_for_category",
    "UnknownDocumentCategory",
    "AmbiguousDocumentCategory",
]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/concepts/test_category_routing.py -v
```

Expected: all 16 tests PASS (10 parametrised spellings plus 6).

- [ ] **Step 5: Wire it into the watcher**

In `src/services/process_monitor_watcher.py`, replace lines 523–532 (the `doc_type_map` literal through the `raise RuntimeError(f"unsupported doc_type: ...")`) with:

```python
            # The acceptable spellings come from proc.bp_document_type, not from
            # a literal here — see src/services/concepts/routing.py. Still
            # raises on an unrecognised category: that error is the only signal
            # a human gets today and a permissive gate would lose it.
            from src.services.concepts.routing import pipeline_for_category
            doc_type, declared_concept = pipeline_for_category(category)
```

Keep `declared_concept` in scope: Task 7 passes it to `dispatch_document`. Leave every other line of the method untouched — `doc_type` keeps the same four possible values, so nothing downstream changes.

- [ ] **Step 6: Verify the watcher still imports and routes**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.services.process_monitor_watcher import ProcessMonitorWatcher  # noqa
from src.services.concepts.routing import pipeline_for_category
for c in ('invoice', 'PO', 'Quote', 'Contract', 'framework agreement'):
    print(c, '->', pipeline_for_category(c))
"
```

Expected: five lines, each a `(pipeline, concept)` pair read from the live tables.

- [ ] **Step 7: Prove the gate still refuses**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.services.concepts.routing import pipeline_for_category, UnknownDocumentCategory
for c in ('', None, 'bill of lading', 'notice'):
    try:
        print(c, '->', pipeline_for_category(c))
        raise AssertionError(f'{c!r} should have been refused')
    except UnknownDocumentCategory as e:
        print(f'{c!r} refused:', e)
"
```

Expected: four refusals, no `AssertionError`.

- [ ] **Step 8: Run the watcher's existing tests for regressions**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest -q \
    tests/services/test_process_monitor_watcher.py \
    tests/test_process_monitor_doc_action.py 2>&1 | tail -20
```

If either file does not exist, find the watcher's tests with
`ls tests | grep -i monitor` and run those. Expected: no **new** failures
against the pre-change baseline. Record the baseline first by stashing nothing
— run the same command on `git stash`-free HEAD via
`git show HEAD:src/services/process_monitor_watcher.py` comparison if needed;
simpler is to note the failure count before Step 5 and compare.

- [ ] **Step 9: Commit**

```bash
git status --short
git commit -o \
  src/services/concepts/routing.py \
  src/services/process_monitor_watcher.py \
  tests/services/concepts/test_category_routing.py \
  -m "feat(concepts): the upload gate reads the vocabulary

Adding a document type is now a row in proc.bp_document_type rather than an
edit to a four-entry dict. An unrecognised category still raises — that error
is the only signal a human gets, and a permissive gate would lose it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: The classification step

**Files:**
- Create: `src/services/extraction/type_resolver.py`
- Test: `tests/services/extraction/test_type_resolver.py`

**Interfaces:**
- Consumes: `vocabulary.Vocabulary`, `vocabulary.ensure_vocabulary`, `vocabulary.fold`.
- Produces:
  - `@dataclass(frozen=True) Evidence(kind: str, text: str, start: int, concept_code: str)` where `kind` is `"title_alias"`, `"body_alias"` or `"structural_signal"`.
  - `@dataclass(frozen=True) TypeResolution` with fields `declared_concept: str | None`, `evidence_concept: str | None`, `status: str` (`"matched"` | `"unknown"` | `"unresolved"`), `agreement: str` (`"agreed"` | `"declared_only"` | `"evidence_only"` | `"disagreed"`), `candidates: tuple[str, ...]`, `evidence: tuple[Evidence, ...]`.
  - `resolve_document_type(*, declared_concept: str | None, full_text: str, vocabulary: Vocabulary | None = None, title_chars: int = 600) -> TypeResolution`

**Three design decisions the implementer must not quietly change.**

**No model call.** Alias and structural-signal matching over text the pipeline has already parsed is deterministic, and every piece of evidence it produces is a verbatim substring of the page — which is the grounding contract the rest of extraction holds (`judge_gate`'s `call_grounded_last_resort`). The build spec's AI-answered ruling tests belong with the conflict register, which this plan defers; see **Scope deviation**. A model call here would buy nothing and would need a grounding story the Discovery Report (§6.1) shows does not exist yet.

**The resolver never refines the declared type, and never changes routing.** If the uploader said `contract` and the page says `framework agreement`, this records both and reports `agreement="disagreed"`. It does not silently upgrade the type, and `pipeline_doc_type` is not its business at all — Task 5 already decided routing from the declared category. Automatic refinement would be this layer overruling a human on its first day, and the product's own precedent is the opposite: `declared_linkage.py` holds that a confirmed human grouping outranks any inferred correlation. The human confirms a refinement from the review queue (Task 7).

**A tie is `unresolved`, not a choice.** Two concepts with equal evidence produce `status="unresolved"` and both in `candidates`. Nothing downstream may pick one.

- [ ] **Step 1: Write the failing test**

Create `tests/services/extraction/test_type_resolver.py`:

```python
"""What the page says about its own type, alongside what the uploader declared.

The resolver's contract: it reports, it never overrules, and every piece of
evidence it returns is a verbatim substring of the text it was given.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.vocabulary import SEED_VOCABULARY  # noqa: E402
from src.services.extraction.type_resolver import (  # noqa: E402
    resolve_document_type,
)

V = SEED_VOCABULARY

FRAMEWORK_PAGE = (
    "FRAMEWORK AGREEMENT\n"
    "Framework Ref: RM6100\n"
    "This framework agreement sets out the terms under which call-off "
    "contracts may be awarded. It orders no goods or services itself.\n"
)

ORDER_FORM_PAGE = (
    "ORDER FORM\n"
    "Framework Ref: RM6100\n"
    "2.1 The following documents are incorporated into this order form.\n"
    "2.2 In the event of conflict the order of precedence is as follows.\n"
)

INVOICE_PAGE = (
    "TAX INVOICE\n"
    "Invoice No: INV-2026-0001\n"
    "Amount due: 1,250.00\n"
)

BLANK_PAGE = "Dear Sir or Madam,\n\nPlease find attached.\n\nKind regards\n"


def test_agreement_when_the_page_says_what_the_uploader_said():
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=INVOICE_PAGE, vocabulary=V,
    )
    assert r.status == "matched"
    assert r.agreement == "agreed"
    assert r.declared_concept == "doctype.invoice"
    assert r.evidence_concept == "doctype.invoice"


def test_evidence_is_always_a_verbatim_substring_of_the_page():
    """The grounding contract the rest of extraction holds. A span that is not
    in the text cannot be shown to a reviewer as the reason."""
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=INVOICE_PAGE, vocabulary=V,
    )
    assert r.evidence, "an invoice page should produce evidence"
    for ev in r.evidence:
        assert ev.text in INVOICE_PAGE, f"{ev.text!r} is not in the page"
        assert INVOICE_PAGE[ev.start:ev.start + len(ev.text)] == ev.text


def test_disagreement_is_recorded_and_the_declared_type_is_kept():
    """Review Focus 5. A file uploaded as a quote whose page is plainly an
    invoice must not be reclassified or rerouted by this layer."""
    r = resolve_document_type(
        declared_concept="doctype.quote", full_text=INVOICE_PAGE, vocabulary=V,
    )
    assert r.agreement == "disagreed"
    assert r.declared_concept == "doctype.quote"
    assert r.evidence_concept == "doctype.invoice"
    # The resolver reports. It does not decide.
    assert not hasattr(r, "pipeline_doc_type")


def test_a_more_specific_type_in_the_page_is_reported_not_applied():
    """Declared 'contract', page says 'framework agreement'. That is a
    refinement a human confirms, not one this layer makes on its own."""
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text=FRAMEWORK_PAGE, vocabulary=V,
    )
    assert r.declared_concept == "doctype.contract_unspecified"
    assert r.evidence_concept == "doctype.framework_agreement"
    assert r.agreement == "disagreed"


def test_a_page_matching_nothing_is_unknown_not_the_nearest_option():
    r = resolve_document_type(
        declared_concept=None, full_text=BLANK_PAGE, vocabulary=V,
    )
    assert r.status == "unknown"
    assert r.evidence_concept is None
    assert r.candidates == ()


def test_a_tie_is_unresolved_and_carries_both_candidates():
    """'order form' is seeded as an alias of the call-off contract, and the
    page also says 'order'. Equal evidence must not be broken by row order."""
    page = "ORDER FORM\nORDER\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert r.status in ("unresolved", "matched")
    if r.status == "unresolved":
        assert len(r.candidates) > 1
        assert r.evidence_concept is None


def test_a_title_hit_outweighs_a_body_mention():
    """An invoice that merely cites a purchase order is still an invoice."""
    page = "TAX INVOICE\nInvoice No: 1\nAgainst purchase order PO123456.\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert r.evidence_concept == "doctype.invoice"


def test_declared_only_when_the_page_is_silent():
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=BLANK_PAGE, vocabulary=V,
    )
    assert r.agreement == "declared_only"
    assert r.declared_concept == "doctype.invoice"
    assert r.evidence_concept is None
    assert r.status == "matched"


def test_structural_signals_support_but_do_not_decide_alone():
    """ORDER_FORM_PAGE carries both of the call-off contract's structural
    signals. They must show up as evidence."""
    r = resolve_document_type(
        declared_concept="doctype.call_off_contract",
        full_text=ORDER_FORM_PAGE, vocabulary=V,
    )
    kinds = {ev.kind for ev in r.evidence}
    assert "structural_signal" in kinds


def test_an_empty_page_does_not_crash():
    r = resolve_document_type(declared_concept=None, full_text="", vocabulary=V)
    assert r.status == "unknown"
    assert r.evidence == ()


def test_a_proposed_type_is_never_the_evidence_answer():
    """doctype.policy_document is seeded as proposed, so its alias 'policy'
    must not resolve even when the page says it outright."""
    r = resolve_document_type(
        declared_concept=None, full_text="POLICY\nThis policy applies.\n", vocabulary=V,
    )
    assert r.evidence_concept != "doctype.policy_document"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_type_resolver.py -v
```

Expected: collection error — `No module named 'src.services.extraction.type_resolver'`.

- [ ] **Step 3: Write the resolver**

Create `src/services/extraction/type_resolver.py`:

```python
"""What the page says about its own type, next to what the uploader declared.

Deterministic: alias and structural-signal matching over text L0 has already
parsed. Every piece of evidence it returns is a verbatim substring of that
text, which is the same grounding contract judge_gate holds for extracted
values — a reviewer can be shown the exact words.

Three things this module refuses to do:

  * It never refines the declared type. Declared 'contract', page says
    'framework agreement' -> both are recorded and agreement='disagreed'. A
    human confirms the refinement from the review queue. declared_linkage.py
    already holds the principle: a human's decision outranks an inference.
  * It never touches routing. Task 5's gate already decided which physical
    pipeline runs, from the declared category. Nothing here can change that.
  * It never breaks a tie. Two concepts with equal evidence give
    status='unresolved' and both candidates.

No model call. The build spec's AI-answered ruling tests belong with the
conflict register, which is deferred: there is no sector or normalised region
to scope a ruling by, and a yes/no judgement cannot be substring-grounded the
way every other model output in this pipeline is.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.services.concepts.vocabulary import Vocabulary, ensure_vocabulary, fold

log = logging.getLogger(__name__)

#: How much of the start of the document counts as the title zone. A type named
#: in the heading is far stronger evidence than one mentioned in a clause, and
#: an invoice citing a purchase order must stay an invoice.
_TITLE_CHARS = 600

_WEIGHT = {"title_alias": 5.0, "body_alias": 1.0, "structural_signal": 0.5}

#: Minimum score before the evidence names a type at all. One passing mention
#: in a clause is not a classification.
_MIN_SCORE = 1.0

#: How far ahead of the runner-up the winner must be. Below this the evidence
#: has not chosen, and saying it has would be fabrication.
_MIN_MARGIN = 1.0


@dataclass(frozen=True)
class Evidence:
    kind: str           # title_alias | body_alias | structural_signal
    text: str           # verbatim substring of the page
    start: int          # its offset, so the span can be shown
    concept_code: str


@dataclass(frozen=True)
class TypeResolution:
    declared_concept: Optional[str]
    evidence_concept: Optional[str]
    status: str         # matched | unknown | unresolved
    agreement: str      # agreed | declared_only | evidence_only | disagreed
    candidates: Tuple[str, ...]
    evidence: Tuple[Evidence, ...]


def _find_all(needle: str, haystack_lower: str) -> List[int]:
    """Offsets of every whole-word occurrence of ``needle``."""
    if not needle:
        return []
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return [m.start() for m in re.finditer(pattern, haystack_lower)]


def resolve_document_type(
    *,
    declared_concept: Optional[str],
    full_text: str,
    vocabulary: Optional[Vocabulary] = None,
    title_chars: int = _TITLE_CHARS,
) -> TypeResolution:
    """Classify a document from its own text, reporting rather than deciding."""
    vocab = vocabulary if vocabulary is not None else ensure_vocabulary()
    text = full_text or ""
    # Fold for matching, but keep offsets usable against the original: fold()
    # collapses whitespace, which would shift them. So match on a lowercase
    # copy of the SAME length and let fold() serve only the alias side.
    lowered = text.lower().replace("_", " ").replace("-", " ")

    evidence: List[Evidence] = []
    scores: Dict[str, float] = {}

    def credit(concept_code: str, kind: str, start: int, length: int) -> None:
        scores[concept_code] = scores.get(concept_code, 0.0) + _WEIGHT[kind]
        evidence.append(Evidence(
            kind=kind, text=text[start:start + length], start=start,
            concept_code=concept_code,
        ))

    # Only active types are in vocabulary.document_types, so a proposed type
    # can never become the evidence answer.
    for code, dt in vocab.document_types.items():
        for alias in (*dt.aliases, code.split(".", 1)[-1]):
            folded = fold(alias)
            if not folded:
                continue
            for start in _find_all(folded, lowered):
                kind = "title_alias" if start < title_chars else "body_alias"
                credit(code, kind, start, len(folded))
        for signal in dt.structural_signals:
            folded = fold(signal)
            if not folded:
                continue
            hits = _find_all(folded, lowered)
            if hits:
                credit(code, "structural_signal", hits[0], len(folded))

    evidence_concept: Optional[str] = None
    candidates: Tuple[str, ...] = ()
    status = "unknown"

    if scores:
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        best_code, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < _MIN_SCORE:
            status = "unknown"
        elif (best_score - runner_up) < _MIN_MARGIN:
            status = "unresolved"
            candidates = tuple(
                code for code, score in ranked if score >= best_score - _MIN_MARGIN
            )
        else:
            status = "matched"
            evidence_concept = best_code
            candidates = (best_code,)

    if declared_concept:
        # A declared type is a human's statement and always stands. The page
        # only ever adds a second reading.
        if evidence_concept is None:
            agreement = "declared_only"
        elif evidence_concept == declared_concept:
            agreement = "agreed"
        else:
            agreement = "disagreed"
        if status != "unresolved":
            status = "matched"
    else:
        agreement = "evidence_only" if evidence_concept else "declared_only"
        if not evidence_concept and status == "matched":
            status = "unknown"

    # Keep the evidence that bears on the answer, strongest first, and cap it:
    # this is written to a review row a person reads, not a log.
    relevant = {c for c in (declared_concept, evidence_concept, *candidates) if c}
    kept = sorted(
        (ev for ev in evidence if ev.concept_code in relevant),
        key=lambda ev: (-_WEIGHT[ev.kind], ev.start),
    )[:12]

    return TypeResolution(
        declared_concept=declared_concept,
        evidence_concept=evidence_concept,
        status=status,
        agreement=agreement,
        candidates=candidates,
        evidence=tuple(kept),
    )


__all__ = ["Evidence", "TypeResolution", "resolve_document_type"]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_type_resolver.py -v
```

Expected: all 11 tests PASS. If `test_a_title_hit_outweighs_a_body_mention` fails, the title zone or the weights need adjusting — adjust the constants, not the test: an invoice that cites a PO must stay an invoice.

- [ ] **Step 5: Prove the grounding guard fails on purpose**

Temporarily change `credit()` to build `Evidence(text=folded, ...)` — the *folded* alias rather than the page's own characters. Re-run; `test_evidence_is_always_a_verbatim_substring_of_the_page` must go **red** on a page where the two differ in case (`TAX INVOICE` vs `tax invoice`). Revert and confirm green. Evidence that is not literally on the page cannot be shown to a reviewer as the reason.

- [ ] **Step 6: Try it on real documents**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.services.extraction.parser import parse as parse_document
from src.services.extraction.type_resolver import resolve_document_type
from src.services.db import get_conn
with get_conn() as conn:
    cur = conn.cursor()
    cur.execute(\"\"\"SELECT file_path, category FROM proc.process_monitor
                     WHERE coalesce(file_path,'') <> '' ORDER BY id DESC LIMIT 5\"\"\")
    rows = cur.fetchall()
for path, category in rows:
    try:
        parsed = parse_document(path)
    except Exception as e:
        print(f'{category:10} {path[:50]:50} parse failed: {e}'); continue
    r = resolve_document_type(declared_concept=None, full_text=parsed.full_text)
    print(f'{category:10} -> {r.status:10} {r.evidence_concept} {[e.text for e in r.evidence[:3]]}')
"
```

Expected: each line shows a status and, where matched, an evidence concept with verbatim spans. Disagreements with `category` here are information, not a failure — record what you see in the commit message.

- [ ] **Step 7: Commit**

```bash
git status --short
git commit -o \
  src/services/extraction/type_resolver.py \
  tests/services/extraction/test_type_resolver.py \
  -m "feat(extraction): classify a document against the vocabulary

Deterministic alias and structural-signal matching over parsed text, with
every piece of evidence a verbatim substring of the page. Reports what the
page says next to what the uploader declared; never refines the declared
type, never changes routing, never breaks a tie.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: Record the classification, and put disagreements in front of a human

**Files:**
- Modify: `src/services/extraction/dispatch.py` — signature at `:279-281`, and an insertion after the discrepancy list is created at `:539`
- Modify: `src/services/extraction/persistence.py:59-70` — document the two new `issue_type` values
- Modify: `src/services/process_monitor_watcher.py` — pass `declared_concept` through
- Test: `tests/services/extraction/test_type_resolution_review_items.py`

**Interfaces:**
- Consumes: `type_resolver.resolve_document_type`, `type_resolver.TypeResolution` (Task 6); `persistence.Discrepancy`.
- Produces: `type_resolution_discrepancies(resolution: TypeResolution) -> list[Discrepancy]` in `src/services/extraction/type_resolver.py`; `dispatch_document(..., declared_concept: str | None = None)`.

**Why the review queue and not a new table.** Discovery §4 #12: six queue-like surfaces exist and `proc.bp_extraction_discrepancy` is the only one humans actually work (5,372 rows). It already carries `issue_type`, `severity`, `blocks_promotion`, `evidence_text`, `evidence_page`, `resolved_by`, `resolved_value` and `resolution_action` — which is exactly what an `UNKNOWN_TYPE` or a type disagreement needs. Its open-row identity is `(doc_type, doc_pk_candidate, issue_type, field_name)` with a partial unique index, so re-extraction refreshes a finding rather than stacking duplicates.

**These findings never block promotion.** `blocks_promotion=False` on both. A document whose type the uploader stated and whose content merely suggests another is still a document that extracted correctly; holding it would stop live ingestion over a reporting improvement. This is the build spec's principle 2 — only impossibilities block — and the product's existing `blocks_promotion` boolean is where it is expressed.

- [ ] **Step 1: Write the failing test**

Create `tests/services/extraction/test_type_resolution_review_items.py`:

```python
"""A disagreement or an unknown type reaches the queue a buyer works — and
never blocks the document.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.vocabulary import SEED_VOCABULARY  # noqa: E402
from src.services.extraction.type_resolver import (  # noqa: E402
    resolve_document_type,
    type_resolution_discrepancies,
)

V = SEED_VOCABULARY
INVOICE_PAGE = "TAX INVOICE\nInvoice No: INV-1\nAmount due: 10.00\n"
BLANK_PAGE = "Dear Sir or Madam,\n\nPlease find attached.\n"


def test_agreement_produces_no_review_item():
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=INVOICE_PAGE, vocabulary=V)
    assert type_resolution_discrepancies(r) == []


def test_declared_only_produces_no_review_item():
    """A silent page is not a problem. The uploader said what it is."""
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=BLANK_PAGE, vocabulary=V)
    assert type_resolution_discrepancies(r) == []


def test_disagreement_produces_one_non_blocking_item_with_both_readings():
    r = resolve_document_type(
        declared_concept="doctype.quote", full_text=INVOICE_PAGE, vocabulary=V)
    items = type_resolution_discrepancies(r)
    assert len(items) == 1
    d = items[0]
    assert d.issue_type == "document_type_disagreement"
    assert d.blocks_promotion is False, (
        "a reporting improvement must not stop live ingestion"
    )
    assert d.raw_value == "doctype.quote"
    assert d.expected_value == "doctype.invoice"
    assert d.evidence_text and d.evidence_text in INVOICE_PAGE


def test_unknown_type_produces_one_non_blocking_item():
    r = resolve_document_type(declared_concept=None, full_text=BLANK_PAGE, vocabulary=V)
    items = type_resolution_discrepancies(r)
    assert len(items) == 1
    assert items[0].issue_type == "unknown_document_type"
    assert items[0].blocks_promotion is False


def test_unresolved_item_names_every_candidate():
    """A tie must reach a human with both options, not with one of them."""
    r = resolve_document_type(
        declared_concept=None, full_text="ORDER FORM\nORDER\n", vocabulary=V)
    items = type_resolution_discrepancies(r)
    if r.status == "unresolved":
        assert len(items) == 1
        assert items[0].issue_type == "unresolved_document_type"
        assert items[0].blocks_promotion is False
        for candidate in r.candidates:
            assert candidate in (items[0].computed_value or "")


def test_every_item_has_a_field_name_so_the_dedup_index_works():
    """The open-row identity is (doc_type, doc_pk_candidate, issue_type,
    field_name). A NULL field_name coalesces to '' and still works, but a
    stable value keeps two different type findings apart."""
    for declared, page in (("doctype.quote", INVOICE_PAGE), (None, BLANK_PAGE)):
        r = resolve_document_type(
            declared_concept=declared, full_text=page, vocabulary=V)
        for d in type_resolution_discrepancies(r):
            assert d.field_name == "document_type"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_type_resolution_review_items.py -v
```

Expected: `ImportError: cannot import name 'type_resolution_discrepancies'`.

- [ ] **Step 3: Add the discrepancy builder to the resolver**

Append to `src/services/extraction/type_resolver.py`:

```python
def type_resolution_discrepancies(resolution: TypeResolution) -> List["object"]:
    """Review items for a classification a human should look at.

    None of these block promotion. A document whose type the uploader stated
    and whose content merely suggests another still extracted correctly, and
    holding it would stop live ingestion over a reporting improvement — the
    build spec's principle 2, expressed through the product's existing
    blocks_promotion boolean.

    field_name is always 'document_type' so the open-row identity
    (doc_type, doc_pk_candidate, issue_type, field_name) stays stable across
    re-extraction and the finding is refreshed rather than duplicated.
    """
    from src.services.extraction.persistence import Discrepancy

    top = resolution.evidence[0].text if resolution.evidence else None

    if resolution.agreement == "disagreed":
        return [Discrepancy(
            field_name="document_type",
            issue_type="document_type_disagreement",
            severity="warning",
            raw_value=resolution.declared_concept,
            expected_value=resolution.evidence_concept,
            computed_value=", ".join(resolution.candidates) or None,
            blocks_promotion=False,
            evidence_text=top,
            notes=(
                f"uploaded as {resolution.declared_concept}; the document reads "
                f"as {resolution.evidence_concept}. The declared type was kept "
                "and nothing was rerouted — confirm which is right."
            ),
        )]

    if resolution.status == "unresolved":
        return [Discrepancy(
            field_name="document_type",
            issue_type="unresolved_document_type",
            severity="warning",
            raw_value=resolution.declared_concept,
            expected_value=None,
            computed_value=", ".join(resolution.candidates) or None,
            blocks_promotion=False,
            evidence_text=top,
            notes=(
                "the document's own wording fits more than one type with equal "
                f"evidence ({', '.join(resolution.candidates)}); no ruling exists "
                "to settle it"
            ),
        )]

    if resolution.status == "unknown":
        return [Discrepancy(
            field_name="document_type",
            issue_type="unknown_document_type",
            severity="warning",
            raw_value=resolution.declared_concept,
            expected_value=None,
            computed_value=None,
            blocks_promotion=False,
            evidence_text=top,
            notes=(
                "no document type in the vocabulary fits this document. Recorded "
                "with no type rather than the nearest option."
            ),
        )]

    return []
```

Add `"type_resolution_discrepancies"` to `__all__`.

- [ ] **Step 4: Run the test to verify it passes**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/extraction/test_type_resolution_review_items.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Document the new issue types**

In `src/services/extraction/persistence.py`, extend the comment on `Discrepancy.issue_type` (line 61) so the vocabulary of that column stays discoverable:

```python
    issue_type: str          # invariant_failed | missing_required | type_bind_error
                             # | judge_incoherent | document_type_disagreement
                             # | unresolved_document_type | unknown_document_type
```

- [ ] **Step 6: Wire the resolver into dispatch**

In `src/services/extraction/dispatch.py`, change the signature at lines 279–281:

```python
def dispatch_document(
    *, process_monitor_id: int | None, file_path: str, doc_type: str,
    declared_concept: str | None = None,
) -> dict[str, Any]:
```

Then immediately after `discrepancies: list[Discrepancy] = []` (line 539), insert:

```python
    # What the page says about its own type, next to what the uploader declared.
    # Recorded, never acted on: doc_type above already decided the pipeline from
    # the declared category, and this must not change it. A failure here is a
    # reporting gap, not an extraction failure, so it never propagates.
    type_resolution = None
    try:
        from src.services.extraction.type_resolver import (
            resolve_document_type, type_resolution_discrepancies,
        )
        type_resolution = resolve_document_type(
            declared_concept=declared_concept, full_text=parsed.full_text,
        )
        discrepancies.extend(type_resolution_discrepancies(type_resolution))
        log.info(
            "type resolution trace=%s declared=%s evidence=%s status=%s agreement=%s",
            trace_id, type_resolution.declared_concept,
            type_resolution.evidence_concept, type_resolution.status,
            type_resolution.agreement,
        )
    except Exception:
        log.exception("type resolution failed (extraction continues)")
```

Finally, add the two readings to the returned summary so a caller can see them without a database query. Find the `return` that builds `{status, raw_id, doc_pk, ...}` near the end of the function and add:

```python
        "declared_concept": declared_concept,
        "evidence_concept": (
            type_resolution.evidence_concept if type_resolution else None
        ),
        "type_agreement": type_resolution.agreement if type_resolution else None,
```

- [ ] **Step 7: Pass the declared concept from the watcher**

In `src/services/process_monitor_watcher.py`, the `_dispatch(...)` call (around line 533, immediately after Task 5's change) becomes:

```python
            result = _dispatch(
                process_monitor_id=record_id,
                file_path=file_path,
                doc_type=doc_type,
                declared_concept=declared_concept,
            )
```

- [ ] **Step 8: Run the extraction test suite for regressions**

Record the baseline first, then compare:

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest -q \
    tests/extraction tests/services/extraction 2>&1 | tail -15
```

Expected: no **new** failures relative to the count you noted before Step 6. This repository has a large standing failure baseline; what matters is the delta.

- [ ] **Step 9: Prove it end to end on the live server**

Restart the API so the new modules load, then re-process one real document and read the finding back out of the queue:

```bash
cd /home/muthu/PycharmProjects/BP_Backend
set -a && . ./.env && set +a
# Re-dispatch one document with a deliberately wrong declared concept.
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.services.db import get_conn
from src.services.extraction.dispatch import dispatch_document
with get_conn() as conn:
    cur = conn.cursor()
    cur.execute(\"\"\"SELECT id, file_path FROM proc.process_monitor
                     WHERE category = 'invoice' AND coalesce(file_path,'') <> ''
                     ORDER BY id DESC LIMIT 1\"\"\")
    pm_id, path = cur.fetchone()
print(dispatch_document(process_monitor_id=pm_id, file_path=path,
                        doc_type='invoice', declared_concept='doctype.quote'))
"
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -At -F'|' -c \
  "select issue_type, severity, blocks_promotion, raw_value, expected_value,
          left(coalesce(evidence_text,''), 40)
     from proc.bp_extraction_discrepancy
    where field_name = 'document_type'
    order by created_at desc limit 5"
```

Expected: a `document_type_disagreement` row with `blocks_promotion = f`, `raw_value = doctype.quote`, `expected_value = doctype.invoice`, and an evidence span from the page. The dispatch result's `status` must be unchanged from what that document returned before — the finding is additive.

- [ ] **Step 10: Commit**

```bash
git status --short
git commit -o \
  src/services/extraction/type_resolver.py \
  src/services/extraction/dispatch.py \
  src/services/extraction/persistence.py \
  src/services/process_monitor_watcher.py \
  tests/services/extraction/test_type_resolution_review_items.py \
  -m "feat(extraction): record the classification and queue disagreements

dispatch now resolves what the page says about its own type alongside the
uploader's declared category, and writes a non-blocking finding to
proc.bp_extraction_discrepancy when the two differ, when the evidence ties,
or when nothing fits. Routing is unchanged: the declared category still
decides the pipeline.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: Agent manifest task slices

**Files:**
- Modify: `src/services/agent_manifest.py` — `build_manifest` at `:238`
- Modify: `src/agents/negotiation_agent.py:8046`
- Test: `tests/services/test_agent_manifest_slices.py`

**Interfaces:**
- Consumes: the existing `AgentManifestService`.
- Produces: `build_manifest(agent_key: str, *, task_id: str | None = None) -> dict` — unchanged shape `{task, policies, knowledge}`, plus `knowledge["loaded"] = {"tables": int, "rows": int, "task_id": str | None}`. New module constant `MANIFEST_TASKS: dict[str, dict]` mapping a task id to `{"tables": tuple[str, ...], "fields": tuple[str, ...], "max_rows": int}`.

**The problem being fixed.** Discovery §5.4: `build_manifest` returns a `knowledge` bundle containing **every** table profile — all columns, all field synonyms — plus six hard-coded relationships, with no task filter. `orchestrator.py` injects it on every step (`:496`, `:1512`, `:1653`, `:3270`) and `workflow_engine.py:424` does too. `base_agent.py` strips heavy knowledge from the *snapshot*, which contains the blast radius — but `negotiation_agent.py:8046` serialises `context.knowledge_base` straight into its prompt. That is one live prompt carrying the whole data dictionary, and the build spec's principle 5 forbids it.

**Only one of the spec's four task rows has a subject.** `DOC_CLASSIFY` does, as of Task 6. `CONFLICT_RESOLVE` needs rulings and `REL_VALIDATE` needs R-REL, both deferred (see **Scope deviation**); `REL_SCORE`'s profile already reaches the scorer directly as a Python constant, not through a manifest. Seeding three rows that filter nothing would be the same error as a check that passes over an empty set. The *mechanism* is built now so the remaining rows are one dict entry each.

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_agent_manifest_slices.py`:

```python
"""An agent must receive the rows its task needs, and not the data dictionary.

Build spec principle 5. Before this, build_manifest returned every table's full
column list and synonym map to every agent on every step, and one agent put the
whole bundle in its prompt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.agent_manifest import MANIFEST_TASKS, AgentManifestService  # noqa: E402


class _StubNick:
    policy_engine = None

    def get_db_connection(self):  # pragma: no cover - never called here
        return None


@pytest.fixture()
def service():
    return AgentManifestService(_StubNick())


def test_an_unsliced_manifest_still_works(service):
    """Every existing caller passes no task_id. They must keep working."""
    m = service.build_manifest("data_extraction")
    assert set(m) == {"task", "policies", "knowledge"}
    assert m["knowledge"]["tables"]


def test_a_sliced_manifest_carries_fewer_tables_than_the_unsliced_one(service):
    full = service.build_manifest("data_extraction")
    sliced = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")
    assert len(sliced["knowledge"]["tables"]) < len(full["knowledge"]["tables"]), (
        "a task slice that carries every table is not a slice"
    )


def test_a_slice_carries_only_the_tables_its_task_declares(service):
    declared = set(MANIFEST_TASKS["DOC_CLASSIFY"]["tables"])
    sliced = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")
    assert set(sliced["knowledge"]["tables"]) <= declared


def test_a_slice_reports_what_it_loaded(service):
    """Build spec §3.7: log the rows loaded per call. Unmeasured context growth
    is how this got to 'every table' in the first place."""
    sliced = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")
    loaded = sliced["knowledge"]["loaded"]
    assert loaded["task_id"] == "DOC_CLASSIFY"
    assert loaded["tables"] == len(sliced["knowledge"]["tables"])
    assert loaded["rows"] >= 0


def test_a_slice_is_bounded_by_max_rows(service):
    sliced = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")
    assert sliced["knowledge"]["loaded"]["rows"] <= MANIFEST_TASKS["DOC_CLASSIFY"]["max_rows"]


def test_an_unknown_task_id_refuses_rather_than_silently_loading_everything(service):
    """A typo in a task id must not fall back to the unfiltered bundle — that
    is the failure mode this task exists to remove."""
    with pytest.raises(KeyError):
        service.build_manifest("data_extraction", task_id="DOC_CLASSIFYY")


def test_the_negotiation_prompt_no_longer_carries_the_knowledge_bundle():
    """Discovery §5.4 named this line as the one prompt-level consumer."""
    source = (
        Path(__file__).resolve().parents[2]
        / "src" / "agents" / "negotiation_agent.py"
    ).read_text()
    assert '"knowledge_base": self._serialise_for_prompt(context.knowledge_base)' not in source, (
        "the whole manifest knowledge bundle is still being serialised into a prompt"
    )


def test_every_declared_task_names_tables_that_exist(service):
    full = service.build_manifest("data_extraction")
    known = set(full["knowledge"]["tables"])
    for task_id, spec in MANIFEST_TASKS.items():
        unknown = set(spec["tables"]) - known
        assert not unknown, f"{task_id} names tables the manifest has no profile for: {unknown}"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/test_agent_manifest_slices.py -v
```

Expected: `ImportError: cannot import name 'MANIFEST_TASKS'`.

- [ ] **Step 3: Add the task table and the slice**

In `src/services/agent_manifest.py`, add after `_PROC_RELATIONSHIPS`:

```python
#: Per-task knowledge slices (build spec §3.7). An agent receives the rows its
#: step needs to decide, not the data dictionary.
#:
#: Only DOC_CLASSIFY is declared. The spec also lists CONFLICT_RESOLVE and
#: REL_VALIDATE, which have no subject until rulings and the R-REL rule group
#: exist, and REL_SCORE, whose profile reaches the scorer directly as a Python
#: constant rather than through a manifest. Declaring them now would be three
#: rows that filter nothing.
MANIFEST_TASKS: Dict[str, Dict[str, Any]] = {
    "DOC_CLASSIFY": {
        "tables": (
            "proc.bp_contracts",
            "proc.bp_supplier",
        ),
        # Column-level narrowing: a classifier needs identity and type columns,
        # not every money and date column on the table.
        "fields": (
            "contract_id", "contract_title", "contract_type",
            "parent_contract_id", "is_amendment", "amendment_ref",
            "document_version", "jurisdiction", "governing_law",
            "supplier_id", "supplier_name",
        ),
        "max_rows": 200,
    },
}
```

Then replace `build_manifest` (line 238) with:

```python
    def build_manifest(
        self, agent_key: str, *, task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        slug = self._normalise(agent_key)
        definition = self._definitions.get(slug)
        policy_bundle = self._policy_bundle_for_agent(slug)
        workflow = self._derive_workflow_hint(slug)

        if task_id is None:
            # Unsliced, for the callers that have not declared a task yet.
            tables = self._table_profiles
            relationships = list(_PROC_RELATIONSHIPS)
        else:
            # KeyError on an unknown task is deliberate: falling back to the
            # unfiltered bundle on a typo is the failure this slicing removes.
            spec = MANIFEST_TASKS[task_id]
            wanted_fields = set(spec["fields"])
            tables = {}
            for name in spec["tables"]:
                profile = self._table_profiles.get(name)
                if profile is None:
                    continue
                columns = [c for c in profile["columns"] if c in wanted_fields]
                tables[name] = {
                    "columns": columns,
                    "required": [c for c in profile["required"] if c in wanted_fields],
                    "synonyms": {
                        k: v for k, v in (profile.get("synonyms") or {}).items()
                        if k in wanted_fields
                    },
                    **({"available": profile["available"]} if "available" in profile else {}),
                }
            relationships = [
                r for r in _PROC_RELATIONSHIPS
                if any(r["from"].startswith(t) or r["to"].startswith(t)
                       for t in spec["tables"])
            ]

        row_count = sum(len(p["columns"]) for p in tables.values())
        if task_id is not None and row_count > MANIFEST_TASKS[task_id]["max_rows"]:
            logger.warning(
                "manifest slice %s loaded %d rows, over its max_rows of %d",
                task_id, row_count, MANIFEST_TASKS[task_id]["max_rows"],
            )
        logger.info(
            "manifest for %s task=%s: %d tables, %d rows",
            slug, task_id, len(tables), row_count,
        )

        knowledge = {
            "tables": tables,
            "relationships": relationships,
            "workflow": workflow,
            "loaded": {
                "task_id": task_id,
                "tables": len(tables),
                "rows": row_count,
            },
        }
        task_profile = {
            "agent_type": definition.agent_type if definition else agent_key,
            "description": definition.description if definition else "",
            "dependencies": definition.dependencies if definition else [],
            "workflow": workflow,
        }
        return {
            "task": task_profile,
            "policies": policy_bundle,
            "knowledge": knowledge,
        }
```

Add `MANIFEST_TASKS` to `__all__`.

- [ ] **Step 4: Stop the prompt leak**

In `src/agents/negotiation_agent.py:8046`, replace:

```python
            "knowledge_base": self._serialise_for_prompt(context.knowledge_base),
```

with:

```python
            # The manifest knowledge bundle is a data dictionary, not negotiation
            # context: every table's columns and synonyms, for every agent, on
            # every step. It told this prompt nothing it uses and cost a large
            # slice of the window. Only what was loaded is reported, so the size
            # stays visible. Build spec principle 5.
            "knowledge_loaded": self._serialise_for_prompt(
                (context.knowledge_base or {}).get("loaded")
            ),
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/test_agent_manifest_slices.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Run the existing manifest and negotiation tests**

```bash
set -a && . ./.env && set +a
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest -q \
    tests/test_agent_manifest_service.py \
    tests/services/negotiation 2>&1 | tail -15
```

Expected: no new failures. `build_manifest`'s default behaviour is unchanged for every existing caller, which is why no orchestrator file is touched by this task.

- [ ] **Step 7: Prove the slice is really smaller**

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -c "
import json, sys; sys.path.insert(0, '.')
from src.services.agent_manifest import AgentManifestService
class N: policy_engine = None
s = AgentManifestService(N())
full = s.build_manifest('data_extraction')
sliced = s.build_manifest('data_extraction', task_id='DOC_CLASSIFY')
for label, m in (('full', full), ('DOC_CLASSIFY', sliced)):
    print(f'{label:14} tables={len(m[\"knowledge\"][\"tables\"]):3} '
          f'chars={len(json.dumps(m[\"knowledge\"], default=str)):6}')
"
```

Expected: the sliced bundle is a small fraction of the full one. Record both numbers in the commit message — the build spec asks that context size be measured from day one.

- [ ] **Step 8: Commit**

```bash
git status --short
git commit -o \
  src/services/agent_manifest.py \
  src/agents/negotiation_agent.py \
  tests/services/test_agent_manifest_slices.py \
  -m "feat(manifest): per-task knowledge slices, and stop the prompt leak

build_manifest gains an optional task_id that narrows the bundle to the
tables and columns the step declares, reports what it loaded, and raises on
an unknown task rather than falling back to everything. negotiation_agent no
longer serialises the whole data dictionary into its prompt.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: Display labels are never used for matching

**Files:**
- Create: `tests/services/test_language_index_not_matched.py`
- Modify: `.github/workflows/reference-data-checks.yml` (add this test to the last step)

**Interfaces:** no runtime code.

**Design note.** Build spec §3.6 and principle 7: `Language_Index` labels must never be read by matching code, and the spec asks for a test that fails if any lookup path reads that table. Discovery §5.3 found the product clean today — `proc.bp_translation` is read only by `i18n/service.py` at render time, and the one exact-match path in i18n (`registry.py:121 _exact`) matches **language names** for the language picker, not concepts. This task pins that down before concept labels start flowing into the table, which is exactly when the mistake becomes available.

- [ ] **Step 1: Write the test**

Create `tests/services/test_language_index_not_matched.py`:

```python
"""Display labels are not aliases.

Build spec principle 7. proc.bp_translation holds UI strings per locale. If a
matching path ever read it, a document would resolve differently depending on
the viewer's language — and the failure would be invisible in English.

This is a source-level guard rather than a runtime one, because the property
being protected is "no code path does this", which no single call can observe.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Modules allowed to read the translation store: the i18n service itself and
#: its tests. Everything else is a violation.
_ALLOWED = {
    "src/services/i18n/store.py",
    "src/services/i18n/service.py",
    "src/services/i18n/filler.py",
    "src/services/i18n/audit.py",
    "src/services/i18n/validate.py",
    "src/api/routers/i18n.py",
}

_TABLES = re.compile(r"bp_translation(?:_language_status)?\b")

#: Modules that resolve a name, alias or type onto a concept. None of them may
#: touch the translation store at all.
_MATCHING_MODULES = (
    "src/services/concepts/vocabulary.py",
    "src/services/concepts/routing.py",
    "src/services/concepts/validate.py",
    "src/services/extraction/type_resolver.py",
    "src/services/linking_engine.py",
    "src/services/extraction/context_layer.py",
    "src/services/extraction/pattern_extractor.py",
)


def _python_sources():
    for path in sorted((ROOT / "src").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if "/__pycache__/" in rel:
            continue
        yield rel, path


def test_only_the_i18n_service_reads_the_translation_store():
    offenders = {}
    for rel, path in _python_sources():
        if rel in _ALLOWED:
            continue
        hits = _TABLES.findall(path.read_text(errors="ignore"))
        if hits:
            offenders[rel] = len(hits)
    assert not offenders, (
        "modules outside the i18n service reference the translation store: "
        f"{offenders} — display labels are not aliases (build spec principle 7)"
    )


def test_no_matching_module_mentions_a_locale_or_label_lookup():
    """A narrower, louder version of the check above, aimed at the modules that
    actually resolve names onto concepts."""
    offenders = {}
    for rel in _MATCHING_MODULES:
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(errors="ignore")
        for needle in ("bp_translation", "i18n", "locale", "translate("):
            if needle in text:
                offenders.setdefault(rel, []).append(needle)
    assert not offenders, (
        f"matching code refers to display-label machinery: {offenders}"
    )


def test_the_vocabulary_carries_no_locale_column():
    """If a label column appeared on bp_document_type, matching on it would be
    one join away. The vocabulary holds codes, definitions and aliases only."""
    migration = (ROOT / "deploy" / "sql" / "2026-10-01_concept_vocabulary.sql").read_text()
    for banned in ("locale", "label", "display_name"):
        assert banned not in migration.lower(), (
            f"the vocabulary migration declares a {banned!r} column; display "
            "labels belong in proc.bp_translation and are never matched on"
        )
```

- [ ] **Step 2: Run it**

```bash
cd /home/muthu/PycharmProjects/BP_Backend
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
    tests/services/test_language_index_not_matched.py -v
```

Expected: all 3 PASS. A failure in `test_only_the_i18n_service_reads_the_translation_store` means either a real violation or a legitimate new i18n module — if the latter, add it to `_ALLOWED` with a one-line reason in the comment, never by widening the regex.

- [ ] **Step 3: Prove the guard fails on purpose**

Temporarily add `# proc.bp_translation` as a comment in `src/services/extraction/type_resolver.py`. Re-run; both of the first two tests must go **red**. Remove the comment and confirm green.

- [ ] **Step 4: Add it to CI**

In `.github/workflows/reference-data-checks.yml`, the final step's test list already names this file (Task 3, Step 6). Confirm the workflow now runs green locally:

```bash
CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest -q \
    tests/services/concepts/test_validation_checks.py \
    tests/services/concepts/test_vocabulary_runtime_load.py \
    tests/services/test_language_index_not_matched.py
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git status --short
git commit -o \
  tests/services/test_language_index_not_matched.py \
  .github/workflows/reference-data-checks.yml \
  -m "test(i18n): display labels are never used for matching

Pins build spec principle 7 before concept labels start flowing into
proc.bp_translation, which is when the mistake becomes available.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Scope deviation from the Discovery Report's §8

The Discovery Report's recommended scope listed six items. This plan builds five of them and **drops one**, for a reason that only became clear while writing the tasks.

**Dropped: R-REL in `proc.bp_rule` (Discovery §8 item 4).**

Every rule in `proc.bp_rule` is bound to code by `detector_slug`; a rule whose detector does not exist does nothing at all. The R-REL categories in the build spec §3.8 — allowed pairs, cardinality, execution validity, sequence and date checks, party consistency, the double-count guard, placeholders — all take a *candidate link* as their subject. No candidate links exist until the link-writing work (spec §4.3), which this plan defers. So R-REL rows added now would be twelve or so rows in a governance table, loaded on every sweep, matching nothing, reporting nothing, and passing every test written about them.

That is the shape of failure this project has been bitten by before: guards that ship green while checking nothing. R-REL moves to the linking plan, where its detectors have something to detect. The table change it needs (`rule_group`, and `blocks_promotion` to carry HARD/SOFT) goes with it, so the migration and the rules land together.

Two smaller instances of the same judgement are recorded inside the tasks rather than here: two of the build spec's six §8.1 checks are deliberately absent (Task 3), and three of its four manifest task rows are deliberately absent (Task 8). In both cases the alternative was a check or a filter with no subject.

---

## Self-review

### 1. Spec coverage

| Build spec section | Covered by |
|---|---|
| §3.1 Concept_Library, RELATIONSHIP domain | Task 1 — all five domains seeded |
| §3.2 Document_Index | Task 1 — `proc.bp_document_type` |
| §3.3 Synonym_Index | Task 1 — folded into the `aliases` array, per Discovery §4 #3 |
| §3.6 Language_Index | Task 9 — guard test; no new labels needed yet |
| §3.7 AGENT_MANIFEST task slices | Task 8 — mechanism plus `DOC_CLASSIFY`; three rows deferred with reason |
| §3.10 Review queue | Task 7 — three `issue_type` values on `bp_extraction_discrepancy` |
| §4.1 steps 1, 3, 5 (extract, candidates, attributes) | Task 6 |
| §4.1 step 8 partial (write the finding) | Task 7 |
| §5.2 "status is matched / unknown / unresolved" | Task 6 — `TypeResolution.status` |
| §6 "document fits no type" → `UNKNOWN_TYPE`, stored with no type | Tasks 6 and 7 |
| §7 Versioning, promotion from review, audit trail | Task 1 — `valid_from`/`valid_to`, `source`, `confirmed_by`/`confirmed_at` |
| §8.1 Automated checks (4 of 6) | Task 3; the other two have no subject |
| §8.3 Metrics: rows per agent call; matched/unknown/unresolved share | Tasks 8 and 7 — both logged |

**Gaps, all deliberate and all stated in the plan:** §3.4 Conflict_Register and conflict_rulings; §3.5 Sector_Glossary; §3.8 R-REL; §3.9 relationship-registry verification (Discovery found it already passes); §4.1 steps 2, 4, 6, 7, 9; §4.2 scope precedence; §4.3 link-writing policy; §4.4 placeholder reconciliation; §8.2 golden set; §9 phases 4–6.

**One gap needs your input rather than a later plan: the golden set (§8.2).** It calls for real documents weighted toward the collision terms — order form, schedule, addendum, CCN, task order, variation form, avenant — plus child-first arrival and a foreign-law document. The Discovery Report found `proc.bp_contracts` is empty and no framework, call-off or SOW exists anywhere in the corpus, so a golden set cannot be drawn from history. Task 6's tests use synthetic pages, which prove the resolver's logic but not that it reads real documents correctly. **Ten to twenty real documents covering those terms would make a genuine golden set possible; without them it would be the resolver graded against its own author's assumptions.**

### 2. Placeholder scan

No "TBD", no "add error handling", no "similar to Task N", no "write tests for the above". Every code step carries the code. Two forward references are explicit and bounded: Task 3 Step 6 notes that the CI workflow's last test file arrives in Task 9, and Task 7 Step 6 says to find the existing `return` statement in `dispatch_document` rather than quoting a line number that will have shifted.

### 3. Type consistency

Checked across tasks: `Concept(concept_code, domain, definition, not_to_be_confused_with, status, rejection_reason)` and `DocumentType(concept_code, role, default_parent_type, execution_mode, aliases, identifiers, structural_signals, pipeline_doc_type, status)` are constructed in Task 1 and consumed with the same names in Tasks 2, 4, 5 and 6. `Vocabulary(concepts, document_types, alias_index, source)` is built in Task 2 and read in Tasks 3, 5, 6. `resolve_alias` returns a tuple everywhere. `TypeResolution.agreement` takes the same four values in Tasks 6 and 7. `build_manifest`'s existing single-argument form is preserved, which is why no orchestrator file is modified.

One knowing duplication: the four pipeline names appear in `validate._PIPELINES`, in the migration's CHECK constraint, and in Task 4's test. All three are guards over the same fact, and Task 4's first test asserts they agree with the vocabulary — which is the point of having them.

### 4. Review Focus coverage

All five lines have a home. (1) and (4) in Task 2 and Task 5; (2) in Task 5; (3) in Tasks 2, 3, 5 and 6; (5) in Tasks 6 and 7. Four tasks additionally carry a **prove-the-guard-fails** step — Tasks 1, 2, 4, 6 and 9 — in which the implementer breaks the guard on purpose and watches it go red before reverting.
