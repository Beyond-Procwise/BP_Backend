# Graph Entity Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer "same supplier?", "same item?" and "under contract?" with banded, evidence-carrying edges in Neo4j, using the scoring engine that already exists.

**Architecture:** Four relationship profiles registered on `src/services/linking_engine.py` via its existing `register_profile` / `register_signal` seam. Pairwise verdicts pass through the `src/services/resolution` MILP for global coherence, then land as Neo4j edges carrying `F`, `band`, `P_raw`, the per-signal breakdown and an observation digest. A derived edge participates in a later question as a *signal whose match score is that edge's stored `P_raw`* — so corroboration, indifference and contradiction are handled by the engine's existing `(2s−1)` mapping with no new arithmetic.

**Tech Stack:** Python 3, Neo4j (bolt, `neo4j` driver — **installed in `.venv` only, NOT in `venv`**), PostgreSQL (`psycopg2`), `scipy.optimize.milp` (HiGHS), pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-graph-entity-resolution-design.md` — read it before Task 1. The plan argues from the spec; both travel together.

## Global Constraints

- **No new scoring mathematics.** Bands, clustering, dampening and log-odds are inherited from `linking_engine`. A task that introduces a new threshold, confidence formula or band is wrong.
- **Bands are fixed:** `≥92 auto_link`, `≥80 auto_link_with_warning`, `≥65 review`, `≥45 weak_relation`, `<45 block_or_exception`. Read them from `_le._BAND_AUTO` etc. — never re-declare the numbers.
- **One prior per answer.** Never sum `L` across edges. Composition happens inside a single `score_link` call.
- **`contract_coverage` and `contract_succession` edges are capped at `review`** — never `auto_link` — until calibrated against a labelled sample (spec §9).
- **Never rewrite a document's literal value.** Resolution records that records refer to one entity; it does not overwrite what a document said.
- **Absent data stays absent.** A signal that cannot be evaluated returns status `MISSING` (`q=0`), never a default score.
- **Two virtualenvs.** `./.venv/bin/python` is the runtime (has `neo4j`). `./venv/bin/python` runs tests (no `neo4j` — graph tests must mock the driver or be marked live). Never assume one tells you about the other.
- **Table prefix `bp_`**, indexes `ix_bp_<table>_<cols>`.
- **Commit trailers** on every commit:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP
  ```
- **Branch:** `Development`. Never push to `main`.
- **Mutation testing is mandatory** where the plan says so: break the guard on purpose, watch it go red, restore. A guard that stays green when broken is checking nothing.

---

## File Structure

**Create:**
- `src/services/graph_resolution/__init__.py` — public API for the whole subsystem
- `src/services/graph_resolution/observations.py` — observation sets and their digest
- `src/services/graph_resolution/composition.py` — cluster remapping for correlated evidence
- `src/services/graph_resolution/edge_writer.py` — the only place that writes derived edges to Neo4j
- `src/services/graph_resolution/profiles/supplier_identity.py`
- `src/services/graph_resolution/profiles/item_equivalence.py`
- `src/services/graph_resolution/profiles/contract_coverage.py`
- `src/services/graph_resolution/profiles/contract_succession.py`
- `src/services/graph_resolution/pass_runner.py` — the ordered batch pass (DAG)
- `scripts/graph_resolution/calibrate.py` — measures `p0`/`alpha` against ground truth

**Modify:**
- `src/services/linking_engine.py` — add optional `cluster_overrides` param to `score_link`; add a `reads` allow-list check to `register_profile`
- `src/services/extraction/kg_sync.py:36-45` — add the `contract` entry to `_TRGT_TABLE` / `_PK_COL`
- `src/services/procurement_kg_builder.py:61-71` — contract source (spec §6)
- `src/services/backend_scheduler.py` — chain the resolution pass

**Test:** mirrors under `tests/services/graph_resolution/`.

Why a package rather than adding to `linking_engine.py` (1,076 lines) or `opportunity_miner_agent.py` (7,629 lines): each file here has one responsibility and stays small enough to hold in context. `edge_writer.py` being the *only* Neo4j writer is what makes the redaction rule (Task 3) enforceable in one place.

---

## Task 1: Observation sets and their digest

**Files:**
- Create: `src/services/graph_resolution/__init__.py`
- Create: `src/services/graph_resolution/observations.py`
- Test: `tests/services/graph_resolution/test_observations.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Observation = tuple[str, str]` (document_id, field); `observation_digest(obs: Iterable[Observation]) -> str`; `intersects(a: frozenset, b: frozenset) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_observations.py
from src.services.graph_resolution.observations import (
    observation_digest, intersects,
)


def test_digest_is_order_independent():
    a = [("DOC-1", "supplier_id"), ("DOC-2", "vat_number")]
    b = [("DOC-2", "vat_number"), ("DOC-1", "supplier_id")]
    assert observation_digest(a) == observation_digest(b)


def test_digest_distinguishes_different_fields_on_same_doc():
    a = [("DOC-1", "supplier_id")]
    b = [("DOC-1", "vat_number")]
    assert observation_digest(a) != observation_digest(b)


def test_empty_digest_is_stable_and_not_empty_string():
    assert observation_digest([]) == observation_digest(())
    assert observation_digest([]) != ""


def test_intersects_detects_a_shared_observation():
    a = frozenset({("DOC-1", "supplier_id"), ("DOC-1", "vat_number")})
    b = frozenset({("DOC-1", "vat_number")})
    assert intersects(a, b) is True


def test_intersects_is_false_for_disjoint_sets():
    a = frozenset({("DOC-1", "supplier_id")})
    b = frozenset({("DOC-2", "supplier_id")})
    assert intersects(a, b) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_observations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.graph_resolution'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/graph_resolution/__init__.py
"""Entity resolution that lives in the graph rather than in SQL joins."""
```

```python
# src/services/graph_resolution/observations.py
"""What a signal actually looked at.

An observation is one (document_id, field) pair. Two signals that read the same
observation are not independent evidence, however different their arithmetic
looks -- and the whole composition model depends on noticing that. See
docs/superpowers/specs/2026-09-10-graph-entity-resolution-design.md section 4.4.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Tuple

Observation = Tuple[str, str]


def observation_digest(obs: Iterable[Observation]) -> str:
    """Stable digest over an observation set, independent of iteration order."""
    items = sorted(f"{doc}\x1f{field}" for doc, field in obs)
    joined = "\x1e".join(items).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:32]


def intersects(a: frozenset, b: frozenset) -> bool:
    """True when two signals drew on at least one identical observation."""
    return not a.isdisjoint(b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_observations.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/graph_resolution/ tests/services/graph_resolution/
git commit -m "feat(graph-resolution): an observation is what a signal actually read

Two signals reading one field are not independent evidence, however
different their arithmetic looks. Naming the observation is what lets
composition notice.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 2: Cluster remapping — correlated evidence shares a cluster

**Files:**
- Create: `src/services/graph_resolution/composition.py`
- Modify: `src/services/linking_engine.py` (`score_link` gains `cluster_overrides`)
- Test: `tests/services/graph_resolution/test_composition.py`

**Interfaces:**
- Consumes: `observations.intersects`.
- Produces: `remap_clusters(signal_specs: list[dict], observations_by_signal: dict[str, frozenset]) -> dict[str, str]` — maps `signal_id -> cluster_name`; `score_link(..., cluster_overrides: Optional[dict] = None)`.

**Why this shape:** `score_link` currently reads `spec["cluster"]` when grouping. Adding an *optional* override parameter leaves every existing call — and every existing golden vector for `invoice_po`, `quote_po`, `quote_rival`, `invoice_duplicate` — byte-identical, while letting the composition layer route correlated signals together per pair. Observation sets are per-pair, so this cannot be done statically at registration.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_composition.py
from src.services import linking_engine as _le
from src.services.graph_resolution.composition import remap_clusters

SPECS = [
    {"id": "vat", "cluster": "registration"},
    {"id": "reg_no", "cluster": "registration"},
    {"id": "name", "cluster": "identity"},
]


def test_disjoint_signals_keep_their_declared_clusters():
    obs = {
        "vat": frozenset({("D1", "vat_number")}),
        "reg_no": frozenset({("D1", "registration_number")}),
        "name": frozenset({("D1", "supplier_name")}),
    }
    assert remap_clusters(SPECS, obs) == {
        "vat": "registration", "reg_no": "registration", "name": "identity",
    }


def test_signals_sharing_an_observation_are_merged_into_one_cluster():
    # `name` secretly read the same field `vat` did: it must not count twice.
    obs = {
        "vat": frozenset({("D1", "vat_number")}),
        "reg_no": frozenset({("D1", "registration_number")}),
        "name": frozenset({("D1", "vat_number")}),
    }
    out = remap_clusters(SPECS, obs)
    assert out["name"] == out["vat"], "correlated signals must share a cluster"
    assert out["reg_no"] == "registration"


def test_merged_cluster_scores_no_higher_than_the_correlated_truth():
    """The double-counting regression test (spec section 10.3)."""
    src = {"supplier_id": "SUP-A", "vat_number": "GB1", "registration_number": "R1"}
    tgt = {"supplier_id": "SUP-A", "vat_number": "GB1", "registration_number": "R1"}
    honest = _le.score_link(src, tgt, "supplier_identity",
                            cluster_overrides={"vat": "registration",
                                               "reg_no": "registration",
                                               "name": "registration"})
    inflated = _le.score_link(src, tgt, "supplier_identity",
                              cluster_overrides={"vat": "c1", "reg_no": "c2",
                                                 "name": "c3"})
    assert honest["F"] <= inflated["F"], (
        "splitting correlated evidence into separate clusters must not be the "
        "cheaper path to a higher score"
    )


def test_score_link_without_overrides_is_unchanged():
    src = {"po_id": "PO-1", "supplier_id": "S1", "currency": "GBP",
           "converted_amount_usd": 100.0, "invoice_date": "2025-01-05",
           "country": "GB", "region": "London"}
    tgt = {"po_id": "PO-1", "supplier_id": "S1", "currency": "GBP",
           "converted_amount_usd": 100.0, "order_date": "2025-01-01",
           "ship_to_country": "GB", "delivery_region": "London"}
    a = _le.score_link(src, tgt, "invoice_po")
    b = _le.score_link(src, tgt, "invoice_po", cluster_overrides=None)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_composition.py -v`
Expected: FAIL — `ModuleNotFoundError` for `composition`, and `TypeError: score_link() got an unexpected keyword argument 'cluster_overrides'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/graph_resolution/composition.py
"""Route correlated signals into one cluster.

The engine already discounts correlated evidence: `_dampen` is applied per
cluster. So the honest way to stop two signals double-counting one observation
is not new arithmetic -- it is putting them in the same cluster and letting the
existing dampening do its job.
"""
from __future__ import annotations

from typing import Dict, List

from .observations import intersects


def remap_clusters(signal_specs: List[dict],
                   observations_by_signal: Dict[str, frozenset]) -> Dict[str, str]:
    """signal_id -> cluster, merging any signals that share an observation.

    Union-find over "shares at least one observation". A merged group takes the
    declared cluster of its lowest-sorted member, so the result is deterministic
    regardless of the order signals were declared in.
    """
    ids = [s["id"] for s in signal_specs]
    declared = {s["id"]: s["cluster"] for s in signal_specs}
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            oa = observations_by_signal.get(a, frozenset())
            ob = observations_by_signal.get(b, frozenset())
            if oa and ob and intersects(oa, ob):
                union(a, b)

    return {i: declared[find(i)] for i in ids}
```

In `src/services/linking_engine.py`, change the signature at line 330 and the
cluster grouping at line ~373:

```python
def score_link(source_row: dict, target_row: dict, profile_name: str,
               source_lines: Optional[list] = None, target_lines: Optional[list] = None,
               set_amount_usd: Optional[float] = None,
               cluster_overrides: Optional[dict] = None) -> dict:
```

```python
    # Stage 2: cluster dampening
    #
    # `cluster_overrides` lets a composing caller route signals that drew on the
    # same observation into one cluster, so the dampening below discounts them as
    # the correlated evidence they are. Absent (the default), every existing
    # caller and golden vector is byte-identical.
    clusters: dict[str, list[dict]] = {}
    for sig in signals:
        name = (cluster_overrides or {}).get(sig["id"], sig["cluster"])
        clusters.setdefault(name, []).append(sig)
```

**Also return the prior-free evidence term.** `L` includes `log(p0/(1-p0))`; a caller
persisting it as "evidence" would bake in exactly the double-counting the spec forbids.
Add one key to `score_link`'s return dict, beside the existing `"L"`:

```python
        "L": round(L, 6),
        # The evidence term WITHOUT the prior. Persisted on derived edges so a
        # later composition can reuse the evidence without inheriting a prior it
        # did not intend (spec 4.2). L itself keeps the prior, unchanged.
        "L_evidence": round(profile["alpha"] * total_cluster_score, 6),
```

Adding a key is safe: a golden vector compares only the keys present in its `expected`
mapping. Step 5 verifies that.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_composition.py -v`
Expected: PASS (4 passed). `test_merged_cluster_scores_no_higher_than_the_correlated_truth` depends on the `supplier_identity` profile from Task 4 — until then, mark it `@pytest.mark.xfail(reason="profile lands in Task 4", strict=False)` and remove the marker in Task 4 Step 4.

- [ ] **Step 5: Verify no existing golden vector moved**

Run: `./venv/bin/python -m pytest tests/services/formulas/ -v`
Expected: PASS, same count as before this task. If any vector moved, the override default is leaking — fix before committing.

- [ ] **Step 6: Mutation test — prove the guard fails**

Temporarily change `remap_clusters`'s final line to `return {i: declared[i] for i in ids}` (never merging). Run:
`./venv/bin/python -m pytest tests/services/graph_resolution/test_composition.py::test_signals_sharing_an_observation_are_merged_into_one_cluster -v`
Expected: **FAIL**. Restore the line. A guard that stays green when broken is checking nothing.

- [ ] **Step 7: Commit**

```bash
git add src/services/graph_resolution/composition.py \
        tests/services/graph_resolution/test_composition.py \
        src/services/linking_engine.py
git commit -m "feat(graph-resolution): correlated signals share a cluster

The engine already discounts correlated evidence per cluster. So the fix
for two signals double-counting one observation is not new arithmetic --
it is putting them in the same cluster. cluster_overrides defaults to
None, so every existing caller and golden vector is byte-identical.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 3: The edge writer — the only place derived edges reach Neo4j

**Files:**
- Create: `src/services/graph_resolution/edge_writer.py`
- Test: `tests/services/graph_resolution/test_edge_writer.py`

**Interfaces:**
- Consumes: `observations.observation_digest`.
- Produces: `DerivedEdge` dataclass; `write_edges(driver, edges: list[DerivedEdge]) -> int`; `REDACTED_SIGNALS: frozenset`.

**Why one writer:** the bank-account redaction rule (spec §8) is enforceable in exactly one place only if exactly one place writes.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_edge_writer.py
import json
import pytest
from src.services.graph_resolution.edge_writer import (
    DerivedEdge, redact_signals, cypher_for, REDACTED_SIGNALS,
)


def _edge(**kw):
    base = dict(
        rel_type="SAME_ENTITY", from_label="Supplier", from_key="supplier_id",
        from_value="SUP-A", to_label="Supplier", to_key="supplier_id",
        to_value="SUP-B", F=94.2, band="auto_link", P_raw=0.97,
        L_evidence=3.4, profile="supplier_identity", profile_version="1.0.0",
        signals=[{"id": "vat", "s": 1.0, "status": "OK"}],
        observations="abc123", resolution="RESOLVED", margin=0.42,
    )
    base.update(kw)
    return DerivedEdge(**base)


def test_bank_signal_value_is_never_written_in_clear():
    sig = [{"id": "bank_account", "s": 1.0, "status": "OK",
            "value": "GB29NWBK60161331926819"}]
    out = redact_signals(sig)
    assert "GB29NWBK60161331926819" not in json.dumps(out)
    assert out[0]["id"] == "bank_account"
    assert out[0]["s"] == 1.0, "the score survives; only the value is redacted"


def test_non_sensitive_signal_values_survive():
    sig = [{"id": "name", "s": 0.8, "status": "OK", "value": "Acme Ltd"}]
    assert redact_signals(sig)[0]["value"] == "Acme Ltd"


def test_bank_account_is_in_the_redaction_list():
    assert "bank_account" in REDACTED_SIGNALS


def test_cypher_merges_rather_than_creates():
    q, _ = cypher_for(_edge())
    assert "MERGE" in q and "CREATE" not in q


def test_cypher_carries_every_required_property():
    _, params = cypher_for(_edge())
    for key in ("F", "band", "P_raw", "L_evidence", "profile",
                "profile_version", "signals", "observations",
                "resolution", "margin"):
        assert key in params["props"], f"{key} missing from edge properties"


def test_signals_are_serialised_as_json_text():
    _, params = cypher_for(_edge())
    assert isinstance(params["props"]["signals"], str)
    assert json.loads(params["props"]["signals"])[0]["id"] == "vat"


def test_capped_profile_cannot_emit_auto_link():
    e = _edge(profile="contract_coverage", band="auto_link", F=97.0)
    with pytest.raises(ValueError, match="capped at review"):
        cypher_for(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_edge_writer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '...edge_writer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/graph_resolution/edge_writer.py
"""The only place a derived edge reaches Neo4j.

One writer is what makes the redaction rule enforceable: a bank account is
Tier-1 identity evidence and must never sit in an edge property in cleartext.
The score survives; the value does not.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, List, Optional

log = logging.getLogger(__name__)

#: Signals whose raw `value` is never persisted. query_engine already excludes
#: banking detail from the default supplier projection; that exclusion holds here.
REDACTED_SIGNALS = frozenset({"bank_account", "bank_iban", "bank_swift"})

#: Profiles whose parameters are declared unmeasured (spec section 9). Their
#: edges never reach auto_link, whatever F says, until a labelled sample exists.
UNCALIBRATED_PROFILES = frozenset({"contract_coverage", "contract_succession"})


@dataclass(frozen=True)
class DerivedEdge:
    rel_type: str
    from_label: str
    from_key: str
    from_value: str
    to_label: str
    to_key: str
    to_value: str
    F: float
    band: str
    P_raw: float
    L_evidence: float
    profile: str
    profile_version: str
    signals: List[dict]
    observations: str
    resolution: Optional[str] = None
    margin: Optional[float] = None


def redact_signals(signals: List[dict]) -> List[dict]:
    out = []
    for s in signals:
        if s.get("id") in REDACTED_SIGNALS and "value" in s:
            s = {**s, "value": "[redacted]"}
        out.append(s)
    return out


def cypher_for(edge: DerivedEdge) -> tuple[str, dict]:
    if edge.profile in UNCALIBRATED_PROFILES and edge.band == "auto_link":
        raise ValueError(
            f"{edge.profile} is capped at review until calibrated "
            f"(spec section 9); refusing to write band=auto_link"
        )
    props = {
        "F": edge.F, "band": edge.band, "P_raw": edge.P_raw,
        "L_evidence": edge.L_evidence, "profile": edge.profile,
        "profile_version": edge.profile_version,
        "signals": json.dumps(redact_signals(edge.signals)),
        "observations": edge.observations,
        "resolution": edge.resolution, "margin": edge.margin,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    q = (
        f"MATCH (a:{edge.from_label} {{{edge.from_key}: $from_value}}) "
        f"MATCH (b:{edge.to_label} {{{edge.to_key}: $to_value}}) "
        f"MERGE (a)-[r:{edge.rel_type}]->(b) "
        f"SET r += $props "
        f"RETURN count(r) AS cnt"
    )
    return q, {"from_value": edge.from_value, "to_value": edge.to_value,
               "props": props}


def write_edges(driver: Any, edges: List[DerivedEdge]) -> int:
    """Write derived edges. Never raises: the graph is a downstream side effect
    and _trgt remains the source of truth for documents."""
    written = 0
    try:
        with driver.session() as session:
            for edge in edges:
                q, params = cypher_for(edge)
                result = session.run(q, **params)
                written += (result.single() or {}).get("cnt", 0)
    except Exception as exc:  # noqa: BLE001
        log.warning("edge_writer: %d/%d written before failure: %s",
                    written, len(edges), exc)
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_edge_writer.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Mutation test — break both guards**

1. Change `REDACTED_SIGNALS` to `frozenset()`. Run `test_bank_signal_value_is_never_written_in_clear` → expect **FAIL**. Restore.
2. Delete the `UNCALIBRATED_PROFILES` check in `cypher_for`. Run `test_capped_profile_cannot_emit_auto_link` → expect **FAIL**. Restore.

- [ ] **Step 6: Add the Item uniqueness constraint**

```bash
./.venv/bin/python -c "
import os; from dotenv import load_dotenv; load_dotenv('.env')
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
with d.session() as s:
    s.run('CREATE CONSTRAINT uniq_item_item_key IF NOT EXISTS FOR (i:Item) REQUIRE i.item_key IS UNIQUE')
    print([r['name'] for r in s.run('SHOW CONSTRAINTS YIELD name RETURN name')])
d.close()"
```
Expected: `uniq_item_item_key` present in the printed list.

- [ ] **Step 7: Commit**

```bash
git add src/services/graph_resolution/edge_writer.py \
        tests/services/graph_resolution/test_edge_writer.py
git commit -m "feat(graph-resolution): one writer, so redaction is enforceable

Derived edges stop being bare: F, band, P_raw, the prior-free evidence
term, the signal breakdown and an observation digest all travel with the
edge. A bank account keeps its score and loses its value, and an
uncalibrated profile cannot emit auto_link however high F went.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 4: `supplier_identity` profile

**Files:**
- Create: `src/services/graph_resolution/profiles/__init__.py`
- Create: `src/services/graph_resolution/profiles/supplier_identity.py`
- Modify: `tests/services/graph_resolution/test_composition.py` (remove the xfail marker)
- Test: `tests/services/graph_resolution/test_supplier_identity.py`

**Interfaces:**
- Consumes: `linking_engine.register_signal/register_profile`, `observations.Observation`.
- Produces: profile name `"supplier_identity"`; `SIGNALS: list[dict]`; `observations_for(src, tgt) -> dict[str, frozenset]`; `score(src, tgt) -> dict`.

**Signal contract reminder:** `_EXTRA_SIGNALS[kind](src, tgt, src_lines, tgt_lines) -> (score, status)`. Four positional args, no more. Derived data is *stamped onto the row dicts by the caller* before scoring — that is how Task 8 and 10 pass graph edges in without changing this contract.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_supplier_identity.py
from src.services.graph_resolution.profiles import supplier_identity as si

A = {"supplier_id": "SUP-Acme", "supplier_name": "Acme Ltd",
     "vat_number": "GB123", "registration_number": "R1",
     "duns_number": "D1", "postal_code": "SW1A 1AA", "country": "GB"}


def test_identical_records_score_auto_link():
    r = si.score(A, dict(A))
    assert r["decision"] == "auto_link", r["F"]


def test_same_company_across_keyspaces_still_resolves():
    """The whole point: SUP-* and S#### never join on id.

    Asserts SEPARATION, not an absolute band: p0/alpha are uncalibrated until
    Task 5, and asserting a calibrated outcome from uncalibrated parameters
    tests the starting constants rather than the profile.
    """
    same = {**A, "supplier_id": "S9251"}
    other = {"supplier_id": "S9252", "supplier_name": "Globex plc",
             "vat_number": "GB999", "registration_number": "R2",
             "duns_number": "D2", "postal_code": "M1 1AA", "country": "GB"}
    r_same, r_other = si.score(A, same), si.score(A, other)
    assert r_same["F"] > r_other["F"], "agreement must outscore disagreement"
    assert r_same["decision"] != "block_or_exception", (
        f"a company agreeing on VAT, registration and DUNS must at least be "
        f"reported: {r_same['F']}"
    )


def test_different_company_is_not_linked():
    b = {"supplier_id": "SUP-Other", "supplier_name": "Globex plc",
         "vat_number": "GB999", "registration_number": "R2",
         "duns_number": "D2", "postal_code": "M1 1AA", "country": "GB"}
    r = si.score(A, b)
    assert r["decision"] in ("weak_relation", "block_or_exception"), r["F"]


def test_missing_registration_data_contributes_nothing_either_way():
    sparse = {"supplier_id": "SUP-Acme2", "supplier_name": "Acme Ltd"}
    r = si.score(A, sparse)
    statuses = {s["id"]: s["status"] for s in r["signals"]}
    assert statuses["vat"] == "MISSING"
    assert r["F"] > 0.0, "a missing signal must not zero the whole score"


def test_registration_signals_share_one_cluster():
    clusters = {s["id"]: s["cluster"] for s in si.SIGNALS}
    assert clusters["vat"] == clusters["reg_no"] == clusters["duns"]


def test_observations_name_the_fields_each_signal_read():
    obs = si.observations_for(A, dict(A))
    assert ("SUP-Acme", "vat_number") in obs["vat"]
    assert ("SUP-Acme", "supplier_name") in obs["name"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_supplier_identity.py -v`
Expected: FAIL — `ModuleNotFoundError` for `profiles.supplier_identity`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/graph_resolution/profiles/__init__.py
"""Relationship profiles registered on the linking engine."""
```

```python
# src/services/graph_resolution/profiles/supplier_identity.py
"""Are these two supplier records the same company?

Transactions key suppliers as SUP-<Name> (3,510 distinct), contracts as S####
(2,545). Measured overlap: zero. Nothing joins today, which is why every
contract question is unanswerable. Registration identifiers are the bridge:
bp_supplier_master carries VAT, registration number and DUNS on 1,009 of 1,009
rows.
"""
from __future__ import annotations

from typing import Optional

from src.services import linking_engine as _le
from ..composition import remap_clusters
from ..observations import Observation

PROFILE = "supplier_identity"
VERSION = "1.0.0"


def _norm(v) -> Optional[str]:
    if v is None:
        return None
    s = " ".join(str(v).split()).strip().lower()
    return s or None


def _cmp_exact(a, b) -> tuple[float, str]:
    na, nb = _norm(a), _norm(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    return (1.0, "OK") if na == nb else (0.0, "CONFLICT")


def _cmp_name(a, b) -> tuple[float, str]:
    na, nb = _norm(a), _norm(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    if na == nb:
        return 1.0, "OK"
    short, lng = sorted([na, nb], key=len)
    if len(short) >= 8 and lng.startswith(short):
        return 0.7, "WEAK"
    return 0.0, "CONFLICT"


def _cmp_address(src, tgt) -> tuple[float, str]:
    pa, pb = _norm(src.get("postal_code")), _norm(tgt.get("postal_code"))
    ca, cb = _norm(src.get("country")), _norm(tgt.get("country"))
    if pa is None or pb is None:
        return 0.5, "MISSING"
    if pa == pb and ca == cb:
        return 1.0, "OK"
    if ca == cb:
        return 0.6, "WEAK"
    return 0.0, "CONFLICT"


_le.register_signal("si_name", lambda s, t, sl, tl: _cmp_name(
    s.get("supplier_name"), t.get("supplier_name")))
_le.register_signal("si_vat", lambda s, t, sl, tl: _cmp_exact(
    s.get("vat_number"), t.get("vat_number")))
_le.register_signal("si_reg", lambda s, t, sl, tl: _cmp_exact(
    s.get("registration_number"), t.get("registration_number")))
_le.register_signal("si_duns", lambda s, t, sl, tl: _cmp_exact(
    s.get("duns_number"), t.get("duns_number")))
_le.register_signal("si_addr", lambda s, t, sl, tl: _cmp_address(s, t))
_le.register_signal("si_bank", lambda s, t, sl, tl: _cmp_exact(
    s.get("bank_account_number"), t.get("bank_account_number")))

# Registration identifiers share a cluster: a company matching on VAT usually
# matches on registration number too, and dampening stops that reading as three
# independent confirmations.
SIGNALS = [
    {"id": "name",    "cluster": "identity",     "tier": 2, "weight": 4, "appl": 1.0, "cap": 0.65, "kind": "si_name",
     "reads": ["supplier_name"]},
    {"id": "vat",     "cluster": "registration", "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "si_vat",
     "reads": ["vat_number"]},
    {"id": "reg_no",  "cluster": "registration", "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "si_reg",
     "reads": ["registration_number"]},
    {"id": "duns",    "cluster": "registration", "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.65, "kind": "si_duns",
     "reads": ["duns_number"]},
    {"id": "addr",    "cluster": "context",      "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "si_addr",
     "reads": ["postal_code", "country"]},
    {"id": "bank",    "cluster": "financial",    "tier": 1, "weight": 4, "appl": 1.0, "cap": 0.45, "kind": "si_bank",
     "reads": ["bank_account_number"]},
]

# p0/alpha are MEASURED in Task 5 against bp_supplier_master. These are the
# starting values the calibration script refines; they are not a borrowed guess
# left in place.
_le.register_profile(PROFILE, {
    "p0": 0.02, "alpha": 0.30, "floor": 0.55,
    "signals": SIGNALS, "date_field": "created_date",
})


def observations_for(src: dict, tgt: dict) -> dict[str, frozenset]:
    """Which (document, field) pairs each signal actually read."""
    sid, tid = str(src.get("supplier_id")), str(tgt.get("supplier_id"))
    out: dict[str, frozenset] = {}
    for spec in SIGNALS:
        obs: set[Observation] = set()
        for field in spec["reads"]:
            if src.get(field) is not None:
                obs.add((sid, field))
            if tgt.get(field) is not None:
                obs.add((tid, field))
        out[spec["id"]] = frozenset(obs)
    return out


def score(src: dict, tgt: dict) -> dict:
    overrides = remap_clusters(SIGNALS, observations_for(src, tgt))
    return _le.score_link(src, tgt, PROFILE, cluster_overrides=overrides)
```

- [ ] **Step 4: Run tests and remove the xfail from Task 2**

Remove the `@pytest.mark.xfail` marker added in Task 2 Step 4, adding this import at the top of `test_composition.py`:
```python
from src.services.graph_resolution.profiles import supplier_identity  # noqa: F401  registers the profile
```

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/ -v`
Expected: PASS (all tasks 1–4)

- [ ] **Step 5: Commit**

```bash
git add src/services/graph_resolution/profiles/ \
        tests/services/graph_resolution/test_supplier_identity.py \
        tests/services/graph_resolution/test_composition.py
git commit -m "feat(graph-resolution): a supplier is the same company across keyspaces

SUP-<Name> and S#### have zero measured overlap, so nothing joins today.
Registration identifiers bridge them -- VAT, registration number and DUNS
are populated on 1,009 of 1,009 supplier-master rows. They share a cluster,
because a company matching on VAT usually matches on registration number
and that is one fact, not three.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 5: Measure `p0` and `alpha` for `supplier_identity`

**Files:**
- Create: `scripts/graph_resolution/calibrate.py`
- Test: `tests/services/graph_resolution/test_calibrate.py`

**Interfaces:**
- Consumes: `supplier_identity.score`.
- Produces: `build_labelled_pairs(rows) -> list[tuple[dict, dict, bool]]`; `sweep(pairs, grid) -> list[dict]`; `best(results) -> dict`.

**Ground truth by construction:** two `bp_supplier_master` rows are the same company iff they share a `vat_number`; different companies otherwise. Hold VAT *out* of scoring and score on the remaining signals — that is a real labelled sample, not a borrowed floor.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_calibrate.py
from scripts.graph_resolution.calibrate import build_labelled_pairs, best

ROWS = [
    {"supplier_id": "A", "vat_number": "GB1", "supplier_name": "Acme Ltd",
     "registration_number": "R1", "duns_number": "D1", "country": "GB", "postal_code": "P1"},
    {"supplier_id": "B", "vat_number": "GB1", "supplier_name": "Acme Limited",
     "registration_number": "R1", "duns_number": "D1", "country": "GB", "postal_code": "P1"},
    {"supplier_id": "C", "vat_number": "GB2", "supplier_name": "Globex plc",
     "registration_number": "R2", "duns_number": "D2", "country": "GB", "postal_code": "P2"},
]


def test_pairs_are_labelled_by_shared_vat():
    pairs = build_labelled_pairs(ROWS)
    labels = {(a["supplier_id"], b["supplier_id"]): same for a, b, same in pairs}
    assert labels[("A", "B")] is True
    assert labels[("A", "C")] is False


def test_vat_is_withheld_from_the_scored_records():
    pairs = build_labelled_pairs(ROWS)
    for a, b, _ in pairs:
        assert "vat_number" not in a, "VAT is the label; scoring on it is circular"
        assert "vat_number" not in b


def test_best_prefers_the_higher_separation():
    results = [
        {"p0": 0.02, "alpha": 0.30, "separation": 12.0, "false_auto_links": 0},
        {"p0": 0.03, "alpha": 0.55, "separation": 31.5, "false_auto_links": 0},
    ]
    assert best(results)["alpha"] == 0.55


def test_best_refuses_a_setting_that_auto_links_a_false_pair():
    results = [
        {"p0": 0.02, "alpha": 0.30, "separation": 12.0, "false_auto_links": 0},
        {"p0": 0.09, "alpha": 0.90, "separation": 44.0, "false_auto_links": 3},
    ]
    assert best(results)["alpha"] == 0.30, "a false auto_link disqualifies a setting"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_calibrate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.graph_resolution'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/graph_resolution/calibrate.py
"""Measure p0 and alpha for supplier_identity against real ground truth.

Two bp_supplier_master rows are the same company iff they share a VAT number.
Hold VAT out of the scored records and score on what remains: that is a
labelled sample by construction, and the floor it yields is measured rather
than borrowed from another profile that was tuned on a different question.
"""
from __future__ import annotations

import itertools
from typing import Iterable, List

from src.services import linking_engine as _le
from src.services.graph_resolution.profiles import supplier_identity as si


def build_labelled_pairs(rows: Iterable[dict]) -> List[tuple]:
    rows = list(rows)
    out = []
    for a, b in itertools.combinations(rows, 2):
        same = (a.get("vat_number") is not None
                and a.get("vat_number") == b.get("vat_number"))
        sa = {k: v for k, v in a.items() if k != "vat_number"}
        sb = {k: v for k, v in b.items() if k != "vat_number"}
        out.append((sa, sb, same))
    return out


def sweep(pairs: List[tuple], grid: List[tuple]) -> List[dict]:
    """Score every pair under each (p0, alpha) and report separation."""
    results = []
    original = dict(_le.PROFILES[si.PROFILE])
    try:
        for p0, alpha in grid:
            _le.PROFILES[si.PROFILE] = {**original, "p0": p0, "alpha": alpha}
            same_F, diff_F, false_auto = [], [], 0
            for a, b, is_same in pairs:
                r = si.score(a, b)
                (same_F if is_same else diff_F).append(r["F"])
                if not is_same and r["F"] >= _le._BAND_AUTO:
                    false_auto += 1
            results.append({
                "p0": p0, "alpha": alpha,
                "separation": (min(same_F) - max(diff_F)) if same_F and diff_F else 0.0,
                "false_auto_links": false_auto,
                "n_same": len(same_F), "n_diff": len(diff_F),
            })
    finally:
        _le.PROFILES[si.PROFILE] = original
    return results


def best(results: List[dict]) -> dict:
    """Highest separation among settings that auto-link no false pair.

    A false auto_link is disqualifying, not a cost to trade off: it is a wrong
    answer asserted with confidence, which is the failure this whole design
    exists to avoid.
    """
    clean = [r for r in results if r["false_auto_links"] == 0]
    pool = clean or results
    return max(pool, key=lambda r: r["separation"])
```

Add `scripts/graph_resolution/__init__.py` (empty) so the module imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_calibrate.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the real calibration against the live database**

```bash
./.venv/bin/python -c "
import os, itertools; from dotenv import load_dotenv; load_dotenv('.env')
import psycopg2, psycopg2.extras
from scripts.graph_resolution.calibrate import build_labelled_pairs, sweep, best
c = psycopg2.connect(host=os.getenv('DB_HOST'), dbname=os.getenv('DB_NAME'),
                     user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'),
                     port=os.getenv('DB_PORT'))
cur = c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
cur.execute('''SELECT supplier_id, supplier_name, vat_number, registration_number,
                      duns_number, postal_code, country, bank_account_number
               FROM proc.bp_supplier_master LIMIT 300''')
rows = [dict(r) for r in cur.fetchall()]
pairs = build_labelled_pairs(rows)
grid = [(p0, a) for p0 in (0.01, 0.02, 0.03, 0.05) for a in (0.25, 0.30, 0.40, 0.55, 0.70)]
res = sweep(pairs, grid)
b = best(res)
print('pairs:', len(pairs), 'same:', b['n_same'], 'diff:', b['n_diff'])
print('BEST:', b)
"
```

Record the printed `BEST` in the module docstring of `supplier_identity.py` **with the date and sample size**, following the pattern in `requirement_similarity.py:100-113`, and update `p0`/`alpha` in `register_profile` to the measured values.

**If `n_same` is 0** — no two supplier-master rows share a VAT number — stop and report it. The profile is then uncalibrated and must be added to `edge_writer.UNCALIBRATED_PROFILES`, exactly like the contract profiles. Do not ship a borrowed number.

- [ ] **Step 6: Commit**

```bash
git add scripts/graph_resolution/ \
        tests/services/graph_resolution/test_calibrate.py \
        src/services/graph_resolution/profiles/supplier_identity.py
git commit -m "feat(graph-resolution): measure the floor instead of borrowing one

Two supplier-master rows are the same company iff they share a VAT number.
Withhold VAT from the scored record, score on what remains, and the sample
labels itself. A setting that auto-links a false pair is disqualified
rather than traded off -- that is a wrong answer asserted confidently,
which is the failure the whole design exists to avoid.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 6: Global coherence — `SAME_ENTITY` through the MILP

**Files:**
- Create: `src/services/graph_resolution/pass_runner.py`
- Test: `tests/services/graph_resolution/test_pass_runner.py`

**Interfaces:**
- Consumes: `supplier_identity.score`, `edge_writer.DerivedEdge/write_edges`, `src.services.resolution.resolve`.
- Produces: `to_candidate_edges(scored) -> tuple[CandidateEdge, ...]`; `run_supplier_identity(conn, driver, limit=None) -> dict`.

**Why the MILP:** pairwise scoring will assert A≈B, B≈C, A≠C. A supplier cannot be two canonical entities at once — that is an assignment problem, and `DEGENERATE` is how a near-tie says so instead of a coin flip being recorded as a fact.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_pass_runner.py
from src.services.graph_resolution.pass_runner import (
    to_candidate_edges, band_for_resolution,
)
from src.services.resolution import CandidateEdge


def test_candidate_edges_carry_log_odds_not_F():
    scored = [{"source_id": "SUP-A", "target_id": "S1",
               "result": {"L": 3.2, "L_evidence": 7.1, "P_raw": 0.96, "F": 94.0}}]
    edges = to_candidate_edges(scored, profile_id="supplier_identity")
    assert isinstance(edges[0], CandidateEdge)
    assert edges[0].log_odds == 3.2   # L, prior included: the solver wants full log-odds
    assert edges[0].confidence == 0.96


def test_degenerate_resolution_is_capped_below_auto_link():
    assert band_for_resolution("auto_link", "DEGENERATE") == "review"


def test_resolved_status_leaves_the_band_alone():
    assert band_for_resolution("auto_link", "RESOLVED") == "auto_link"


def test_infeasible_never_yields_an_actionable_band():
    assert band_for_resolution("auto_link", "INFEASIBLE") == "weak_relation"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_pass_runner.py -v`
Expected: FAIL — `ModuleNotFoundError` for `pass_runner`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/graph_resolution/pass_runner.py
"""The ordered resolution pass.

Stages run in the dependency order the spec declares (section 4.5): identity
first, then items, then contract succession and coverage. Each stage writes its
edges before the next reads them, because a later profile reads earlier edges as
signals.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.services.resolution import (
    CandidateEdge, CardinalityRule, ResolutionRequest, resolve,
)
from .edge_writer import DerivedEdge, write_edges
from .observations import observation_digest
from .profiles import supplier_identity as si

log = logging.getLogger(__name__)


def to_candidate_edges(scored: List[dict], profile_id: str) -> tuple:
    """Pairwise verdicts as MILP input. log_odds travels; F does not.

    CandidateEdge documents log_odds as "from the existing scorer, pre-sigmoid"
    and confidence as "post-sigmoid, for reporting only". Honour that: the
    solver reasons in log-odds.
    """
    return tuple(
        CandidateEdge(
            source_id=s["source_id"], target_id=s["target_id"],
            log_odds=s["result"]["L"], confidence=s["result"]["P_raw"],
            profile_id=profile_id,
        )
        for s in scored
    )


def band_for_resolution(band: str, status: Optional[str]) -> str:
    """A near-tie is not an auto-link, however high F went.

    DEGENERATE means the solver found the assignment barely forced -- another
    answer was nearly as good. Recording that as a certainty would be the
    precise thing this layer exists to prevent.
    """
    if status == "INFEASIBLE":
        return "weak_relation"
    if status == "DEGENERATE" and band == "auto_link":
        return "review"
    return band


def run_supplier_identity(conn: Any, driver: Any, limit: Optional[int] = None) -> dict:
    """Score supplier pairs, resolve globally, write SAME_ENTITY edges."""
    import psycopg2.extras
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT supplier_id, supplier_name, vat_number, registration_number,
                  duns_number, postal_code, country, bank_account_number
           FROM proc.bp_supplier_master
           ORDER BY supplier_id""" + (f" LIMIT {int(limit)}" if limit else "")
    )
    rows = [dict(r) for r in cur.fetchall()]

    scored = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            result = si.score(a, b)
            if result["F"] < 45.0:      # block_or_exception: nothing is said
                continue
            scored.append({"source_id": a["supplier_id"],
                           "target_id": b["supplier_id"],
                           "result": result, "src": a, "tgt": b})

    if not scored:
        return {"scored": 0, "written": 0, "status": None}

    request = ResolutionRequest(
        request_id="supplier_identity",
        edges=to_candidate_edges(scored, si.PROFILE),
        capacities=(),
        rules=(CardinalityRule(profile_id=si.PROFILE, shape="N:1",
                               max_targets_per_source=1),),
        profile_registry_version=si.VERSION,
    )
    outcome = resolve(request)
    kept = {(l.source_id, l.target_id): l for l in outcome.links}

    edges = []
    for s in scored:
        link = kept.get((s["source_id"], s["target_id"]))
        if link is None:
            continue
        r = s["result"]
        edges.append(DerivedEdge(
            rel_type="SAME_ENTITY",
            from_label="Supplier", from_key="supplier_id", from_value=s["source_id"],
            to_label="Supplier", to_key="supplier_id", to_value=s["target_id"],
            F=r["F"], band=band_for_resolution(r["decision"], outcome.status),
            P_raw=r["P_raw"], L_evidence=r["L_evidence"], profile=si.PROFILE,
            profile_version=si.VERSION, signals=r["signals"],
            observations=observation_digest(
                o for obs in si.observations_for(s["src"], s["tgt"]).values() for o in obs
            ),
            resolution=outcome.status, margin=link.margin_normalised,
        ))

    written = write_edges(driver, edges)
    log.info("supplier_identity: scored=%d kept=%d written=%d status=%s",
             len(scored), len(edges), written, outcome.status)
    return {"scored": len(scored), "written": written, "status": outcome.status}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_pass_runner.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Mutation test — prove the degeneracy cap fails**

Change `band_for_resolution` to `return band` unconditionally. Run
`test_degenerate_resolution_is_capped_below_auto_link` → expect **FAIL**. Restore.

- [ ] **Step 6: Live verification**

```bash
./.venv/bin/python -c "
import os; from dotenv import load_dotenv; load_dotenv('.env')
import psycopg2
from neo4j import GraphDatabase
from src.services.graph_resolution.pass_runner import run_supplier_identity
c = psycopg2.connect(host=os.getenv('DB_HOST'), dbname=os.getenv('DB_NAME'),
                     user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'),
                     port=os.getenv('DB_PORT'))
d = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
print(run_supplier_identity(c, d, limit=200))
with d.session() as s:
    for r in s.run('MATCH ()-[r:SAME_ENTITY]->() RETURN r.band AS band, count(*) AS c ORDER BY c DESC'):
        print(r['band'], r['c'])
d.close(); c.close()"
```

Record the real band distribution in the commit message. **Report it honestly even if
`auto_link` is zero** — that is a finding about the corpus, not a failure to hide.

- [ ] **Step 7: Commit**

```bash
git add src/services/graph_resolution/pass_runner.py \
        tests/services/graph_resolution/test_pass_runner.py
git commit -m "feat(graph-resolution): a supplier cannot be two entities at once

Pairwise scoring will happily assert A is B, B is C and A is not C. The
MILP settles which of those can hold together, and DEGENERATE is how a
near-tie says so -- capped to review however high F went, because
recording a coin flip as a certainty is the failure this layer exists
to prevent.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 7: `item_equivalence` profile and `Item` minting

**Files:**
- Create: `src/services/graph_resolution/profiles/item_equivalence.py`
- Test: `tests/services/graph_resolution/test_item_equivalence.py`

**Interfaces:**
- Consumes: `linking_engine`, `composition.remap_clusters`, `SAME_ENTITY` edges (stamped onto rows by the caller as `_same_entity_p`).
- Produces: `PROFILE = "item_equivalence"`; `score(src, tgt) -> dict`; `item_key(members: list[dict]) -> str`; `observations_for(src, tgt)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_item_equivalence.py
from src.services.graph_resolution.profiles import item_equivalence as ie

L1 = {"invoice_line_id": "L1", "item_id": "ITM001",
      "item_description": "Dell Latitude 5540 laptop", "unit_of_measure": "each",
      "unit_price": 780.0, "_same_entity_p": None}
L2 = {"invoice_line_id": "L2", "item_id": "ITM001",
      "item_description": "Dell Latitude 5540 Laptop", "unit_of_measure": "each",
      "unit_price": 782.0, "_same_entity_p": None}


def test_identical_item_id_auto_links():
    assert ie.score(L1, L2)["decision"] == "auto_link"


def test_descriptive_drift_still_resolves_without_item_id():
    """Separation, not an absolute band: alpha is calibrated in Step 5."""
    a = {**L1, "item_id": None}
    b = {**L2, "item_id": None}
    unrelated = {**L2, "item_id": None,
                 "item_description": "Office chair, mesh back",
                 "unit_price": 120.0}
    assert ie.score(a, b)["F"] > ie.score(a, unrelated)["F"]


def test_different_products_do_not_link():
    b = {**L2, "item_id": "ITM999",
         "item_description": "Office chair, mesh back", "unit_price": 120.0}
    assert ie.score(L1, b)["decision"] in ("weak_relation", "block_or_exception")


def test_item_key_is_deterministic_regardless_of_member_order():
    members = [{"item_id": "ITM009"}, {"item_id": "ITM002"}]
    assert ie.item_key(members) == ie.item_key(list(reversed(members)))


def test_item_key_uses_the_lowest_item_id():
    assert ie.item_key([{"item_id": "ITM009"}, {"item_id": "ITM002"}]) == \
           ie.item_key([{"item_id": "ITM002"}])


def test_item_key_falls_back_to_description_when_no_id_exists():
    k = ie.item_key([{"item_id": None, "item_description": "Widget  A"}])
    assert k and isinstance(k, str)


def test_uom_is_recorded_never_converted():
    a = {**L1, "unit_of_measure": "box of 10"}
    r = ie.score(a, L2)
    uom = [s for s in r["signals"] if s["id"] == "uom"][0]
    assert uom["status"] in ("CONFLICT", "WEAK"), (
        "a pack of 10 and 10 each are related, not equal"
    )


def test_derived_supplier_signal_contributes_nothing_when_absent():
    r = ie.score(L1, L2)
    sup = [s for s in r["signals"] if s["id"] == "supplier_same"][0]
    assert sup["status"] == "MISSING"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_item_equivalence.py -v`
Expected: FAIL — `ModuleNotFoundError` for `item_equivalence`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/graph_resolution/profiles/item_equivalence.py
"""Are these two lines the same product?

item_id is present on 55,421 of 55,483 invoice lines, so the reference signal
carries most cases; the rest is descriptive drift across suppliers. Connecting
193,857 line nodes through a shared Item is what turns price comparison from
string-matching into a graph question.

UoM is RECORDED, never converted: a pack of 10 and 10 each are related, not
equal, and bp_uom_canonical (38 rows) is the only UoM authority.
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

from src.services import linking_engine as _le
from ..composition import remap_clusters
from ..observations import Observation

PROFILE = "item_equivalence"
VERSION = "1.0.0"


def _norm(v) -> Optional[str]:
    if v is None:
        return None
    s = " ".join(str(v).split()).strip().lower()
    return s or None


def _tokens(v) -> set:
    n = _norm(v)
    return set(n.split()) if n else set()


def _cmp_item_id(a, b) -> tuple[float, str]:
    na, nb = _norm(a), _norm(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    return (1.0, "OK") if na == nb else (0.0, "CONFLICT")


def _cmp_desc(a, b) -> tuple[float, str]:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.5, "MISSING"
    j = len(ta & tb) / len(ta | tb)
    if j >= 0.8:
        return 1.0, "OK"
    if j >= 0.5:
        return 0.7, "WEAK"
    return j, "CONFLICT" if j < 0.2 else "WEAK"


def _cmp_uom(a, b) -> tuple[float, str]:
    na, nb = _norm(a), _norm(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    if na == nb:
        return 1.0, "OK"
    return 0.0, "CONFLICT"


def _cmp_price(a, b) -> tuple[float, str]:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return 0.5, "MISSING"
    if fa <= 0 or fb <= 0:
        return 0.5, "MISSING"
    ratio = min(fa, fb) / max(fa, fb)
    if ratio >= 0.95:
        return 1.0, "OK"
    if ratio >= 0.70:
        return 0.6, "WEAK"
    return 0.0, "CONFLICT"


def _cmp_derived_supplier(src, tgt) -> tuple[float, str]:
    """Reads a SAME_ENTITY edge's stored P_raw, stamped on the row by the caller.

    An absent edge is not weak evidence, it is no evidence: q=0, contributing
    nothing in either direction.
    """
    p = src.get("_same_entity_p")
    if p is None:
        p = tgt.get("_same_entity_p")
    if p is None:
        return 0.5, "MISSING"
    return float(p), "OK"


_le.register_signal("ie_item_id", lambda s, t, sl, tl: _cmp_item_id(
    s.get("item_id"), t.get("item_id")))
_le.register_signal("ie_desc", lambda s, t, sl, tl: _cmp_desc(
    s.get("item_description"), t.get("item_description")))
_le.register_signal("ie_uom", lambda s, t, sl, tl: _cmp_uom(
    s.get("unit_of_measure"), t.get("unit_of_measure")))
_le.register_signal("ie_price", lambda s, t, sl, tl: _cmp_price(
    s.get("unit_price"), t.get("unit_price")))
_le.register_signal("ie_supplier", lambda s, t, sl, tl: _cmp_derived_supplier(s, t))

SIGNALS = [
    {"id": "item_id",       "cluster": "reference",   "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "ie_item_id",
     "reads": ["item_id"]},
    {"id": "desc",          "cluster": "description", "tier": 2, "weight": 4, "appl": 1.0, "cap": 0.70, "kind": "ie_desc",
     "reads": ["item_description"]},
    {"id": "uom",           "cluster": "description", "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "ie_uom",
     "reads": ["unit_of_measure"]},
    {"id": "price",         "cluster": "commercial",  "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "ie_price",
     "reads": ["unit_price"]},
    {"id": "supplier_same", "cluster": "identity",    "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.70, "kind": "ie_supplier",
     "reads": ["_same_entity_p"]},
]

_le.register_profile(PROFILE, {
    "p0": 0.02, "alpha": 0.35, "floor": 0.55,
    "signals": SIGNALS, "date_field": "delivery_date",
})


def _line_id(row: dict) -> str:
    return str(row.get("invoice_line_id") or row.get("po_line_id")
               or row.get("quote_line_id") or row.get("item_id") or "?")


def observations_for(src: dict, tgt: dict) -> dict[str, frozenset]:
    sid, tid = _line_id(src), _line_id(tgt)
    out: dict[str, frozenset] = {}
    for spec in SIGNALS:
        obs: set[Observation] = set()
        for field in spec["reads"]:
            if src.get(field) is not None:
                obs.add((sid, field))
            if tgt.get(field) is not None:
                obs.add((tid, field))
        out[spec["id"]] = frozenset(obs)
    return out


def score(src: dict, tgt: dict) -> dict:
    overrides = remap_clusters(SIGNALS, observations_for(src, tgt))
    return _le.score_link(src, tgt, PROFILE, cluster_overrides=overrides)


def item_key(members: List[dict]) -> str:
    """Digest of the class's canonical member: the lowest item_id, else the
    lowest normalised description.

    The choice of canonical member is arbitrary; its DETERMINISM is not. A
    rebuild must reproduce the same item_key or every OF_ITEM edge, and every
    finding citing one, breaks on the next pass.
    """
    ids = sorted(_norm(m.get("item_id")) for m in members if _norm(m.get("item_id")))
    if ids:
        basis = f"id:{ids[0]}"
    else:
        descs = sorted(_norm(m.get("item_description")) for m in members
                       if _norm(m.get("item_description")))
        basis = f"desc:{descs[0]}" if descs else "unknown"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_item_equivalence.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Calibrate against held-out `item_id`**

Reuse `calibrate.sweep`, labelling pairs by shared `item_id` and withholding it from the
scored records — same construction as Task 5, different ground truth. Record the measured
`p0`/`alpha` in the module docstring with date and sample size, and update
`register_profile`.

- [ ] **Step 6: Commit**

```bash
git add src/services/graph_resolution/profiles/item_equivalence.py \
        tests/services/graph_resolution/test_item_equivalence.py
git commit -m "feat(graph-resolution): lines meet at the product they name

193,857 line nodes carry item_id as a property and connect to nothing.
An Item node turns price comparison from string-matching into a graph
question. UoM is recorded, never converted: a pack of 10 and 10 each are
related, not equal.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 8: `Item` nodes and `OF_ITEM` edges

**Files:**
- Modify: `src/services/graph_resolution/pass_runner.py` (add `run_item_equivalence`)
- Test: `tests/services/graph_resolution/test_pass_runner.py` (extend)

**Interfaces:**
- Consumes: `item_equivalence.score/item_key`, `edge_writer`.
- Produces: `run_item_equivalence(conn, driver, limit=None) -> dict`; `mint_items(driver, classes) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/services/graph_resolution/test_pass_runner.py
from src.services.graph_resolution.pass_runner import equivalence_classes


def test_equivalence_classes_are_transitive():
    linked = [("L1", "L2"), ("L2", "L3"), ("L9", "L10")]
    classes = equivalence_classes(["L1", "L2", "L3", "L9", "L10", "L11"], linked)
    as_sets = sorted([sorted(c) for c in classes])
    assert ["L1", "L2", "L3"] in as_sets
    assert ["L9", "L10"] in [sorted(c) for c in classes]
    assert ["L11"] in as_sets, "a line linked to nothing is its own class"


def test_every_line_appears_in_exactly_one_class():
    lines = ["A", "B", "C"]
    classes = equivalence_classes(lines, [("A", "B")])
    flat = [m for c in classes for m in c]
    assert sorted(flat) == sorted(lines)
    assert len(flat) == len(set(flat))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_pass_runner.py -k equivalence -v`
Expected: FAIL — `ImportError: cannot import name 'equivalence_classes'`

- [ ] **Step 3: Write minimal implementation**

Add to `pass_runner.py`:

```python
def equivalence_classes(members: List[str], linked: List[tuple]) -> List[List[str]]:
    """Transitive closure over confirmed links. Every member lands in exactly
    one class; an unlinked member is a class of one."""
    parent = {m: m for m in members}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in linked:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    groups: dict = {}
    for m in members:
        groups.setdefault(find(m), []).append(m)
    return list(groups.values())


def run_item_equivalence(conn: Any, driver: Any, limit: Optional[int] = None) -> dict:
    """Resolve line items into Items and write OF_ITEM edges."""
    import psycopg2.extras
    from .profiles import item_equivalence as ie

    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT invoice_line_id, item_id, item_description, unit_of_measure,
                  unit_price, invoice_id
           FROM proc.bp_invoice_line_items_trgt
           WHERE item_id IS NOT NULL
           ORDER BY invoice_line_id""" + (f" LIMIT {int(limit)}" if limit else "")
    )
    rows = [dict(r) for r in cur.fetchall()]
    by_id = {r["invoice_line_id"]: r for r in rows}

    linked, scored_by_pair = [], {}
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            r = ie.score(a, b)
            if r["F"] >= 80.0:      # auto_link_with_warning and above
                pair = (a["invoice_line_id"], b["invoice_line_id"])
                linked.append(pair)
                scored_by_pair[pair] = r

    classes = equivalence_classes(list(by_id), linked)

    edges = []
    for members in classes:
        key = ie.item_key([by_id[m] for m in members])
        with driver.session() as session:
            session.run("MERGE (i:Item {item_key: $k})", k=key)
        for m in members:
            r = scored_by_pair.get(next(
                (p for p in scored_by_pair if m in p), None), None)
            edges.append(DerivedEdge(
                rel_type="OF_ITEM",
                from_label="InvoiceLine", from_key="invoice_line_id", from_value=m,
                to_label="Item", to_key="item_key", to_value=key,
                F=(r or {}).get("F", 100.0),
                band=(r or {}).get("decision", "auto_link"),
                P_raw=(r or {}).get("P_raw", 1.0),
                L_evidence=(r or {}).get("L", 0.0),
                profile=ie.PROFILE, profile_version=ie.VERSION,
                signals=(r or {}).get("signals", []),
                observations=observation_digest([(m, "item_id")]),
                resolution=None, margin=None,
            ))

    written = write_edges(driver, edges)
    log.info("item_equivalence: lines=%d classes=%d written=%d",
             len(rows), len(classes), written)
    return {"lines": len(rows), "classes": len(classes), "written": written}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_pass_runner.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Live verification — prove the information gain**

Run the pass with `limit=500`, then check the two-hop question is now answerable:

```bash
./.venv/bin/python -c "
import os; from dotenv import load_dotenv; load_dotenv('.env')
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
with d.session() as s:
    q = '''MATCH (l1:InvoiceLine)-[:OF_ITEM]->(i:Item)<-[:OF_ITEM]-(l2:InvoiceLine)
           MATCH (l1)-[:LINE_OF_INVOICE]->(inv1:Invoice)
           MATCH (l2)-[:LINE_OF_INVOICE]->(inv2:Invoice)
           WHERE inv1.supplier_id <> inv2.supplier_id
           RETURN i.item_key AS item, count(DISTINCT inv1.supplier_id) AS suppliers
           ORDER BY suppliers DESC LIMIT 5'''
    for r in s.run(q): print(r['item'], r['suppliers'])
d.close()"
```

Expected: rows returned — the same product bought from more than one supplier, reached by
traversal rather than string comparison. Record the count in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/services/graph_resolution/pass_runner.py \
        tests/services/graph_resolution/test_pass_runner.py
git commit -m "feat(graph-resolution): the same product, reached by traversal

Lines now meet at an Item node, so 'who else sells this' is two hops
instead of a LIKE against item_description. Every line lands in exactly
one class and a line linked to nothing is a class of one.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 9: Contracts reach the graph

**Files:**
- Modify: `src/services/extraction/kg_sync.py:36-45`
- Modify: `src/services/procurement_kg_builder.py:61-71`
- Test: `tests/services/graph_resolution/test_contract_ingest.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Contract` nodes carrying `origin` in `{"extracted", "reference"}`.

**Spec §6 decision:** `proc.bp_contracts` stays the extraction destination (`contract.yaml`
already targets it) and yields `origin='extracted'`. `bp_contract_master`'s 3,051 rows load
as `origin='reference'`. They are different things and a finding must be able to say which
it rests on.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_contract_ingest.py
from src.services.extraction import kg_sync
from src.services import procurement_kg_builder as builder


def test_kg_sync_knows_how_to_sync_a_contract():
    assert "contract" in kg_sync._TRGT_TABLE
    assert "contract" in kg_sync._PK_COL
    assert kg_sync._PK_COL["contract"] == "contract_id"


def test_contract_entity_maps_to_a_table_that_has_rows():
    """bp_contracts is the extraction destination and is empty today; the
    reference set must therefore also be mapped, or Contract stays at 0 nodes."""
    sources = {name: cfg[0] for name, cfg in builder.ENTITY_TABLE_MAP.items()}
    assert "Contract" in sources
    assert "ContractReference" in sources
    assert sources["ContractReference"] == "proc.bp_contract_master"


def test_reference_contracts_are_labelled_as_such():
    assert builder.ORIGIN_BY_ENTITY["ContractReference"] == "reference"
    assert builder.ORIGIN_BY_ENTITY["Contract"] == "extracted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_contract_ingest.py -v`
Expected: FAIL — `KeyError: 'contract'` and `AttributeError: ORIGIN_BY_ENTITY`

- [ ] **Step 3: Write minimal implementation**

In `src/services/extraction/kg_sync.py`:

```python
_TRGT_TABLE = {
    "invoice": "proc.bp_invoice_trgt",
    "purchase_order": "proc.bp_purchase_order_trgt",
    "quote": "proc.bp_quote_trgt",
    # Contracts do not go through _stg/_trgt: contract.yaml writes straight to
    # proc.bp_contracts. Without this entry a contract could be extracted and
    # still never reach the graph.
    "contract": "proc.bp_contracts",
}
_PK_COL = {
    "invoice": "invoice_id",
    "purchase_order": "po_id",
    "quote": "quote_id",
    "contract": "contract_id",
}
```

In `src/services/procurement_kg_builder.py`:

```python
ENTITY_TABLE_MAP = {
    ...
    "Contract": ("proc.bp_contracts", "contract_id", "Contract"),
    # 3,051 seeded reference contracts on a different keyspace (S#### suppliers).
    # Loaded as Contract nodes too, distinguished by `origin` so a finding can
    # say whether it rests on a document or on reference data.
    "ContractReference": ("proc.bp_contract_master", "contract_id", "Contract"),
    ...
}

ORIGIN_BY_ENTITY = {
    "Contract": "extracted",
    "ContractReference": "reference",
}
```

Set `n.origin = $origin` in the node MERGE, reading `ORIGIN_BY_ENTITY.get(entity, "extracted")`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_contract_ingest.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Live verification — Contract nodes finally exist**

```bash
./.venv/bin/python -c "
import os; from dotenv import load_dotenv; load_dotenv('.env')
from neo4j import GraphDatabase
d = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
with d.session() as s:
    for r in s.run('MATCH (c:Contract) RETURN c.origin AS origin, count(*) AS c'):
        print(r['origin'], r['c'])
d.close()"
```

Expected: `reference 3051`. `extracted 0` is correct and expected — `bp_contracts` is empty
until the extraction pipeline produces a contract.

- [ ] **Step 6: Commit**

```bash
git add src/services/extraction/kg_sync.py \
        src/services/procurement_kg_builder.py \
        tests/services/graph_resolution/test_contract_ingest.py
git commit -m "feat(graph-resolution): contracts reach the graph at last

The Contract entity and its uniqueness constraint have existed all along,
producing zero nodes because the builder reads bp_contracts (0 rows) while
3,051 contracts sit in bp_contract_master. Both now load, distinguished by
origin, so a finding can say whether it rests on a document or on reference
data. kg_sync gains a contract entry, without which an extracted contract
would never reach the graph incrementally.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 10: `contract_coverage` — off-contract spend becomes answerable

**Files:**
- Create: `src/services/graph_resolution/profiles/contract_coverage.py`
- Test: `tests/services/graph_resolution/test_contract_coverage.py`

**Interfaces:**
- Consumes: `SAME_ENTITY` and `OF_ITEM` edges (stamped on rows as `_same_entity_p`, `_item_under_contract_p`).
- Produces: `PROFILE = "contract_coverage"`; `score(doc, contract) -> dict`; `uncovered_reason(result) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_contract_coverage.py
from src.services.graph_resolution.profiles import contract_coverage as cc

INV = {"invoice_id": "INV-1", "supplier_id": "SUP-Acme",
       "invoice_date": "2025-06-15", "invoice_total_incl_tax": 5000.0,
       "currency": "GBP", "_same_entity_p": 0.97}
CON = {"contract_id": "C-1", "supplier_id": "S9251",
       "contract_start_date": "2025-01-01", "contract_end_date": "2025-12-31",
       "total_contract_value": 100000.0, "currency": "GBP",
       "spend_category": None}


def test_in_term_with_resolved_supplier_is_covered():
    """Separation, not an absolute band: this profile ships DECLARED
    UNMEASURED (spec section 9), so an absolute threshold here would assert a
    calibration that deliberately does not exist yet."""
    out_of_term = {**INV, "invoice_date": "2026-06-15"}
    assert cc.score(INV, CON)["F"] > cc.score(out_of_term, CON)["F"]


def test_outside_every_term_window_is_not_covered():
    out = {**INV, "invoice_date": "2026-06-15"}
    r = cc.score(out, CON)
    temporal = [s for s in r["signals"] if s["id"] == "date_in_term"][0]
    assert temporal["status"] == "CONFLICT"


def test_missing_category_contributes_nothing_rather_than_penalising():
    r = cc.score(INV, CON)
    cat = [s for s in r["signals"] if s["id"] == "category"][0]
    assert cat["status"] == "MISSING"
    assert cat["c"] == 0.0, "an unevaluable signal must contribute exactly zero"


def test_coverage_never_reaches_auto_link_while_uncalibrated():
    from src.services.graph_resolution.edge_writer import UNCALIBRATED_PROFILES
    assert cc.PROFILE in UNCALIBRATED_PROFILES


def test_uncovered_reason_names_the_missing_evidence():
    out = {**INV, "invoice_date": "2026-06-15", "_same_entity_p": None}
    reason = cc.uncovered_reason(cc.score(out, CON))
    assert "supplier" in reason.lower() or "term" in reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_contract_coverage.py -v`
Expected: FAIL — `ModuleNotFoundError` for `contract_coverage`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/graph_resolution/profiles/contract_coverage.py
"""Is this document covered by a contract?

The contract side is ready: 958 Active contracts, 957 with a full term window,
892 distinct suppliers. What was missing was a way to reach the supplier --
contract_id is populated on 0 of 38,498 transaction rows and the supplier
keyspaces have zero overlap. SAME_ENTITY supplies that reach.

This is a SCOPE statement, never a PRICE statement: contract.yaml declares
db_lines_table: null, so no contracted unit price exists to compare against.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from src.services import linking_engine as _le
from ..composition import remap_clusters
from ..observations import Observation

PROFILE = "contract_coverage"
VERSION = "1.0.0"


def _to_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(v)[:10], fmt).date()
        except ValueError:
            continue
    return None


def _cmp_contract_ref(a, b) -> tuple[float, str]:
    na = str(a).strip().lower() if a else None
    nb = str(b).strip().lower() if b else None
    if not na or not nb:
        return 0.5, "MISSING"
    return (1.0, "OK") if na == nb else (0.0, "CONFLICT")


def _cmp_derived(src, tgt, field) -> tuple[float, str]:
    p = src.get(field)
    if p is None:
        p = tgt.get(field)
    if p is None:
        return 0.5, "MISSING"
    return float(p), "OK"


def _cmp_in_term(src, tgt) -> tuple[float, str]:
    d = _to_date(src.get("invoice_date") or src.get("order_date"))
    start = _to_date(tgt.get("contract_start_date"))
    end = _to_date(tgt.get("contract_end_date"))
    if d is None or start is None or end is None:
        return 0.5, "MISSING"
    if start <= d <= end:
        return 1.0, "OK"
    return 0.0, "CONFLICT"


def _cmp_category(src, tgt) -> tuple[float, str]:
    a = (src.get("spend_category") or "").strip().lower()
    b = (tgt.get("spend_category") or "").strip().lower()
    if not a or not b:
        return 0.5, "MISSING"
    return (1.0, "OK") if a == b else (0.0, "CONFLICT")


def _cmp_within_value(src, tgt) -> tuple[float, str]:
    try:
        amt = float(src.get("invoice_total_incl_tax") or src.get("total_amount"))
        val = float(tgt.get("total_contract_value"))
    except (TypeError, ValueError):
        return 0.5, "MISSING"
    if val <= 0:
        return 0.5, "MISSING"
    if (src.get("currency") or "") != (tgt.get("currency") or ""):
        # Never convert to compare. FX is populated on 10 of 12,408 invoices.
        return 0.5, "MISSING"
    return (1.0, "OK") if amt <= val else (0.3, "WEAK")


_le.register_signal("cc_ref", lambda s, t, sl, tl: _cmp_contract_ref(
    s.get("contract_id"), t.get("contract_id")))
_le.register_signal("cc_supplier", lambda s, t, sl, tl: _cmp_derived(s, t, "_same_entity_p"))
_le.register_signal("cc_term", lambda s, t, sl, tl: _cmp_in_term(s, t))
_le.register_signal("cc_category", lambda s, t, sl, tl: _cmp_category(s, t))
_le.register_signal("cc_value", lambda s, t, sl, tl: _cmp_within_value(s, t))
_le.register_signal("cc_item", lambda s, t, sl, tl: _cmp_derived(s, t, "_item_under_contract_p"))

SIGNALS = [
    {"id": "contract_ref",  "cluster": "reference",  "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "cc_ref",
     "reads": ["contract_id"]},
    {"id": "supplier_same", "cluster": "identity",   "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "cc_supplier",
     "reads": ["_same_entity_p"]},
    {"id": "date_in_term",  "cluster": "temporal",   "tier": 1, "weight": 4, "appl": 1.0, "cap": 0.45, "kind": "cc_term",
     "reads": ["invoice_date", "contract_start_date", "contract_end_date"]},
    {"id": "category",      "cluster": "category",   "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "cc_category",
     "reads": ["spend_category"]},
    {"id": "amount_value",  "cluster": "commercial", "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.70, "kind": "cc_value",
     "reads": ["invoice_total_incl_tax", "total_contract_value", "currency"]},
    {"id": "item_covered",  "cluster": "line",       "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.70, "kind": "cc_item",
     "reads": ["_item_under_contract_p"]},
]

# p0/alpha DECLARED UNMEASURED: zero contract nodes existed to calibrate
# against. edge_writer refuses auto_link for this profile until that changes.
_le.register_profile(PROFILE, {
    "p0": 0.02, "alpha": 0.30, "floor": 0.55,
    "signals": SIGNALS, "date_field": "invoice_date",
})


def observations_for(src: dict, tgt: dict) -> dict[str, frozenset]:
    sid = str(src.get("invoice_id") or src.get("po_id") or "?")
    tid = str(tgt.get("contract_id") or "?")
    out: dict[str, frozenset] = {}
    for spec in SIGNALS:
        obs: set[Observation] = set()
        for field in spec["reads"]:
            if src.get(field) is not None:
                obs.add((sid, field))
            if tgt.get(field) is not None:
                obs.add((tid, field))
        out[spec["id"]] = frozenset(obs)
    return out


def score(src: dict, tgt: dict) -> dict:
    overrides = remap_clusters(SIGNALS, observations_for(src, tgt))
    return _le.score_link(src, tgt, PROFILE, cluster_overrides=overrides)


def uncovered_reason(result: dict) -> str:
    """Why this document is not covered, in words a supplier conversation
    survives. Absence of evidence and conflicting evidence are different
    findings and are reported differently."""
    by_id = {s["id"]: s for s in result["signals"]}
    parts = []
    if by_id["date_in_term"]["status"] == "CONFLICT":
        parts.append("dated outside every active contract term for this supplier")
    elif by_id["date_in_term"]["status"] == "MISSING":
        parts.append("no usable contract term dates")
    if by_id["supplier_same"]["status"] == "MISSING":
        parts.append("supplier could not be resolved to a contracted entity")
    if by_id["contract_ref"]["status"] == "MISSING":
        parts.append("no contract reference on the document")
    return "; ".join(parts) or "no corroborating evidence above the reporting floor"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_contract_coverage.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/graph_resolution/profiles/contract_coverage.py \
        tests/services/graph_resolution/test_contract_coverage.py
git commit -m "feat(graph-resolution): off-contract spend, with the reason attached

The contract side was always ready -- 958 active, 957 with full term
windows. What was missing was reach: contract_id is empty on all 38,498
transaction rows and the supplier keyspaces do not overlap. SAME_ENTITY
supplies it.

A document outside every active term for a resolved supplier is
off-contract, and uncovered_reason says which evidence was absent rather
than asserting a verdict. Scope, never price: contracts have no line
table, so no contracted unit price exists to compare against.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 11: `contract_succession` — renewal uplift

**Files:**
- Create: `src/services/graph_resolution/profiles/contract_succession.py`
- Test: `tests/services/graph_resolution/test_contract_succession.py`

**Interfaces:**
- Consumes: `SAME_ENTITY` edges via `_same_entity_p`.
- Produces: `PROFILE = "contract_succession"`; `score(prev, nxt) -> dict`; `uplift(prev, nxt) -> Optional[dict]`.

**The problem it solves:** `parent_contract_id` is populated on 1,561 contracts and resolves
on **0** — the ids were minted in a different namespace (`C1543` against actual `C00002`).
The chain is reconstructed from evidence rather than repaired by hand.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_contract_succession.py
from src.services.graph_resolution.profiles import contract_succession as cs

PREV = {"contract_id": "C00002", "supplier_id": "S1",
        "contract_start_date": "2024-01-01", "contract_end_date": "2024-12-31",
        "total_contract_value": 100000.0, "currency": "GBP",
        "contract_title": "Service Desk 24x7", "spend_category": "IT Services",
        "_same_entity_p": 0.99}
NEXT = {"contract_id": "C00055", "supplier_id": "S1",
        "contract_start_date": "2025-01-01", "contract_end_date": "2025-12-31",
        "total_contract_value": 112000.0, "currency": "GBP",
        "contract_title": "Service Desk 24x7", "spend_category": "IT Services",
        "_same_entity_p": 0.99}


def test_adjacent_terms_same_supplier_are_a_succession():
    assert cs.score(PREV, NEXT)["F"] >= 65.0


def test_overlapping_unrelated_contract_is_not_a_succession():
    other = {**NEXT, "contract_start_date": "2024-06-01",
             "contract_title": "Office cleaning", "spend_category": "Facilities"}
    assert cs.score(PREV, other)["F"] < 65.0


def test_uplift_is_reported_in_a_single_currency():
    u = cs.uplift(PREV, NEXT)
    assert u["pct"] == 12.0
    assert u["currency"] == "GBP"


def test_uplift_is_unavailable_across_currencies_rather_than_converted():
    eur = {**NEXT, "currency": "EUR"}
    assert cs.uplift(PREV, eur) is None, (
        "converting would fabricate an FX rate; 10 of 12,408 invoices carry one"
    )


def test_succession_never_reaches_auto_link_while_uncalibrated():
    from src.services.graph_resolution.edge_writer import UNCALIBRATED_PROFILES
    assert cs.PROFILE in UNCALIBRATED_PROFILES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_contract_succession.py -v`
Expected: FAIL — `ModuleNotFoundError` for `contract_succession`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/graph_resolution/profiles/contract_succession.py
"""Which contract replaced which?

parent_contract_id is populated on 1,561 contracts and resolves on 0: the
references were minted in a different namespace (C1543 against actual C00002).
Rather than repair ids by hand, reconstruct the chain from evidence -- same
supplier, adjacent terms, same category, comparable value, similar title.

Renewal uplift then follows, in ONE currency. Converting across currencies to
report an uplift would fabricate an FX rate.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from src.services import linking_engine as _le
from ..composition import remap_clusters
from ..observations import Observation

PROFILE = "contract_succession"
VERSION = "1.0.0"

#: A renewal starts near the predecessor's end. Wider than a day to survive
#: signature lag; narrow enough that an unrelated later contract is not a renewal.
ADJACENCY_DAYS = 120


def _to_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _cmp_derived_supplier(src, tgt) -> tuple[float, str]:
    p = src.get("_same_entity_p") or tgt.get("_same_entity_p")
    if p is None:
        return 0.5, "MISSING"
    return float(p), "OK"


def _cmp_adjacency(src, tgt) -> tuple[float, str]:
    prev_end = _to_date(src.get("contract_end_date"))
    next_start = _to_date(tgt.get("contract_start_date"))
    if prev_end is None or next_start is None:
        return 0.5, "MISSING"
    if next_start < prev_end - timedelta(days=ADJACENCY_DAYS):
        return 0.0, "CONFLICT"      # starts well before the predecessor ended
    gap = abs((next_start - prev_end).days)
    if gap <= ADJACENCY_DAYS:
        return 1.0, "OK"
    return 0.0, "CONFLICT"


def _cmp_category(src, tgt) -> tuple[float, str]:
    a = (src.get("spend_category") or "").strip().lower()
    b = (tgt.get("spend_category") or "").strip().lower()
    if not a or not b:
        return 0.5, "MISSING"
    return (1.0, "OK") if a == b else (0.0, "CONFLICT")


def _cmp_value(src, tgt) -> tuple[float, str]:
    try:
        a, b = float(src["total_contract_value"]), float(tgt["total_contract_value"])
    except (TypeError, ValueError, KeyError):
        return 0.5, "MISSING"
    if a <= 0 or b <= 0 or src.get("currency") != tgt.get("currency"):
        return 0.5, "MISSING"
    ratio = min(a, b) / max(a, b)
    if ratio >= 0.75:
        return 1.0, "OK"
    if ratio >= 0.4:
        return 0.6, "WEAK"
    return 0.0, "CONFLICT"


def _cmp_title(src, tgt) -> tuple[float, str]:
    ta = set((src.get("contract_title") or "").lower().split())
    tb = set((tgt.get("contract_title") or "").lower().split())
    if not ta or not tb:
        return 0.5, "MISSING"
    j = len(ta & tb) / len(ta | tb)
    if j >= 0.7:
        return 1.0, "OK"
    if j >= 0.35:
        return 0.6, "WEAK"
    return 0.0, "CONFLICT"


_le.register_signal("csx_supplier", lambda s, t, sl, tl: _cmp_derived_supplier(s, t))
_le.register_signal("csx_adjacent", lambda s, t, sl, tl: _cmp_adjacency(s, t))
_le.register_signal("csx_category", lambda s, t, sl, tl: _cmp_category(s, t))
_le.register_signal("csx_value", lambda s, t, sl, tl: _cmp_value(s, t))
_le.register_signal("csx_title", lambda s, t, sl, tl: _cmp_title(s, t))

SIGNALS = [
    {"id": "supplier_same", "cluster": "identity",    "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "csx_supplier",
     "reads": ["_same_entity_p"]},
    {"id": "adjacency",     "cluster": "temporal",    "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "csx_adjacent",
     "reads": ["contract_end_date", "contract_start_date"]},
    {"id": "category",      "cluster": "category",    "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "csx_category",
     "reads": ["spend_category"]},
    {"id": "value",         "cluster": "commercial",  "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.70, "kind": "csx_value",
     "reads": ["total_contract_value", "currency"]},
    {"id": "title",         "cluster": "description", "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.70, "kind": "csx_title",
     "reads": ["contract_title"]},
]

# DECLARED UNMEASURED, like contract_coverage: no labelled succession sample
# exists while every parent_contract_id dangles.
_le.register_profile(PROFILE, {
    "p0": 0.02, "alpha": 0.35, "floor": 0.55,
    "signals": SIGNALS, "date_field": "contract_start_date",
})


def observations_for(src: dict, tgt: dict) -> dict[str, frozenset]:
    sid, tid = str(src.get("contract_id")), str(tgt.get("contract_id"))
    out: dict[str, frozenset] = {}
    for spec in SIGNALS:
        obs: set[Observation] = set()
        for field in spec["reads"]:
            if src.get(field) is not None:
                obs.add((sid, field))
            if tgt.get(field) is not None:
                obs.add((tid, field))
        out[spec["id"]] = frozenset(obs)
    return out


def score(src: dict, tgt: dict) -> dict:
    overrides = remap_clusters(SIGNALS, observations_for(src, tgt))
    return _le.score_link(src, tgt, PROFILE, cluster_overrides=overrides)


def uplift(prev: dict, nxt: dict) -> Optional[dict]:
    """Renewal uplift, or None when it cannot be stated without inventing a rate."""
    if prev.get("currency") != nxt.get("currency"):
        return None
    try:
        a, b = float(prev["total_contract_value"]), float(nxt["total_contract_value"])
    except (TypeError, ValueError, KeyError):
        return None
    if a <= 0:
        return None
    return {"currency": prev.get("currency"), "previous": a, "current": b,
            "delta": round(b - a, 2), "pct": round((b - a) / a * 100.0, 4)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_contract_succession.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Mutation test — prove the FX guard fails**

Change `uplift`'s first check to `if False:`. Run
`test_uplift_is_unavailable_across_currencies_rather_than_converted` → expect **FAIL**. Restore.

- [ ] **Step 6: Add both contract profiles to `UNCALIBRATED_PROFILES` and run everything**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/ -v`
Expected: PASS, all tasks 1–11.

Run: `./venv/bin/python -m pytest tests/services/formulas/ tests/services/resolution/ -v`
Expected: PASS, unchanged counts — no existing golden vector moved.

- [ ] **Step 7: Commit**

```bash
git add src/services/graph_resolution/profiles/contract_succession.py \
        tests/services/graph_resolution/test_contract_succession.py
git commit -m "feat(graph-resolution): the renewal chain the ids cannot express

parent_contract_id is populated on 1,561 contracts and resolves on 0 --
the references were minted in a different namespace. Reconstruct the chain
from evidence instead of repairing ids by hand.

Uplift is reported in one currency or not at all. Converting to state a
percentage would fabricate an FX rate, and FX is populated on 10 of 12,408
invoices.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Task 12: Wire the pass into the scheduler, and size the review queue

**Files:**
- Modify: `src/services/graph_resolution/pass_runner.py` (add `run_all`)
- Modify: `src/services/backend_scheduler.py`
- Test: `tests/services/graph_resolution/test_pass_order.py`

**Interfaces:**
- Consumes: all four profiles.
- Produces: `run_all(conn, driver) -> dict`; `PASS_ORDER: tuple[str, ...]`.

**Spec §12 — the open question this task must close:** `supplier_identity` will produce a
`review` band, and `bp_supplier_review` already holds **727 rows, all `pending`, none ever
reviewed**. This task does not build a review UI. It *sizes* the population and reports it,
so the decision to leave it unactioned is made deliberately rather than by accident.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_pass_order.py
import pytest
from src.services.graph_resolution.pass_runner import PASS_ORDER, assert_dag_safe


def test_identity_resolves_before_anything_reads_it():
    assert PASS_ORDER.index("supplier_identity") < PASS_ORDER.index("item_equivalence")
    assert PASS_ORDER.index("supplier_identity") < PASS_ORDER.index("contract_coverage")
    assert PASS_ORDER.index("supplier_identity") < PASS_ORDER.index("contract_succession")


def test_items_resolve_before_contract_coverage_reads_them():
    assert PASS_ORDER.index("item_equivalence") < PASS_ORDER.index("contract_coverage")


def test_a_profile_reading_a_downstream_edge_is_refused():
    with pytest.raises(ValueError, match="downstream"):
        assert_dag_safe("supplier_identity", reads=["UNDER_CONTRACT"])


def test_reading_an_upstream_edge_is_allowed():
    assert_dag_safe("contract_coverage", reads=["SAME_ENTITY", "OF_ITEM"]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_pass_order.py -v`
Expected: FAIL — `ImportError: cannot import name 'PASS_ORDER'`

- [ ] **Step 3: Write minimal implementation**

Add to `pass_runner.py`:

```python
#: Dependency order (spec section 4.5). A later profile reads earlier edges as
#: signals, so this order is a correctness requirement, not a preference.
PASS_ORDER = ("supplier_identity", "item_equivalence",
              "contract_succession", "contract_coverage")

#: Which edge each profile produces.
_PRODUCES = {
    "supplier_identity": "SAME_ENTITY",
    "item_equivalence": "OF_ITEM",
    "contract_succession": "SUCCEEDS",
    "contract_coverage": "UNDER_CONTRACT",
}


def assert_dag_safe(profile: str, reads: List[str]) -> None:
    """Refuse a profile that reads an edge produced at or after its own stage.

    A cycle here would be evidence laundering -- a conclusion re-entering as
    its own support -- and it would not be visible in any single score.
    """
    own = PASS_ORDER.index(profile)
    for rel in reads:
        producer = next((p for p, r in _PRODUCES.items() if r == rel), None)
        if producer is None:
            continue
        if PASS_ORDER.index(producer) >= own:
            raise ValueError(
                f"{profile} reads {rel}, which is produced downstream by "
                f"{producer}; that is a cycle, not corroboration"
            )


def run_all(conn: Any, driver: Any, limit: Optional[int] = None) -> dict:
    """Run every stage in dependency order, then size the review band."""
    out = {}
    out["supplier_identity"] = run_supplier_identity(conn, driver, limit)
    out["item_equivalence"] = run_item_equivalence(conn, driver, limit)
    out["review_backlog"] = review_backlog(driver)
    return out


def review_backlog(driver: Any) -> dict:
    """How many edges landed in the review band, and therefore need a person.

    bp_supplier_review already holds 727 pending rows that nobody has reviewed.
    Reporting this number is how the choice to leave it unactioned stays a
    choice rather than an accident.
    """
    counts = {}
    try:
        with driver.session() as session:
            for rel in _PRODUCES.values():
                r = session.run(
                    f"MATCH ()-[e:{rel}]->() WHERE e.band = 'review' "
                    f"RETURN count(e) AS c"
                ).single()
                counts[rel] = (r or {}).get("c", 0)
    except Exception as exc:  # noqa: BLE001
        log.warning("review_backlog unavailable: %s", exc)
    return counts
```

In `backend_scheduler.py`, chain `run_all` after `_chain_opportunity_mining`, following
the existing pattern at line 856 — wrapped in `try/except` and logged, never raised.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/services/graph_resolution/test_pass_order.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Mutation test — prove the cycle guard fails**

Change `assert_dag_safe` to `return None` immediately. Run
`test_a_profile_reading_a_downstream_edge_is_refused` → expect **FAIL**. Restore.

- [ ] **Step 6: Full live run and honest report**

```bash
./.venv/bin/python -c "
import os; from dotenv import load_dotenv; load_dotenv('.env')
import psycopg2
from neo4j import GraphDatabase
from src.services.graph_resolution.pass_runner import run_all
c = psycopg2.connect(host=os.getenv('DB_HOST'), dbname=os.getenv('DB_NAME'),
                     user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'),
                     port=os.getenv('DB_PORT'))
d = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
import json; print(json.dumps(run_all(c, d, limit=300), indent=2))
d.close(); c.close()"
```

Report the real numbers — edges written per band, the review backlog, and whether any
`auto_link` was produced at all. **A run that produces zero `auto_link` edges is a finding
about the corpus and must be reported as one**, not quietly retried with looser parameters.

- [ ] **Step 7: Commit**

```bash
git add src/services/graph_resolution/pass_runner.py \
        src/services/backend_scheduler.py \
        tests/services/graph_resolution/test_pass_order.py
git commit -m "feat(graph-resolution): run the stages in the only order that is correct

A later profile reads earlier edges as signals, so the order is a
correctness requirement. assert_dag_safe refuses a profile that reads an
edge produced downstream -- a cycle there would be a conclusion re-entering
as its own support, invisible in any single score.

The review backlog is counted and reported, because bp_supplier_review
already holds 727 rows nobody has looked at and adding to it silently
would repeat that.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1qA1FfLWtFU23wmWmfrP"
```

---

## Self-review notes

**Spec coverage.** §2 framework → Tasks 2, 4, 6 (inherited, not rebuilt). §3.2 nodes →
Tasks 8, 9. §3.3 edges → Tasks 6, 8, 10, 11. §3.4 edge properties → Task 3. §3.5 constraints
→ Task 3 Step 6. §4.1 composition → Tasks 7, 10, 11 (derived signals). §4.2 single prior →
Task 6 (`to_candidate_edges` passes `L` once, never sums). §4.3 missing evidence → Tasks 4,
10 tests. §4.4 observation discipline → Tasks 1, 2. §4.5 DAG → Task 12. §5 profiles → Tasks
4, 7, 10, 11. §6 contract source → Task 9. §7.1 batch → Task 12. §7.2 MILP → Task 6. §7.3
incremental → Task 9 (`kg_sync`). §8 safety → Task 3 (redaction), Task 6 (never raises). §9
calibration → Tasks 5, 7 Step 5. §10 testing → mutation steps in Tasks 2, 3, 6, 11, 12. §12
review backlog → Task 12.

**Known gap, deliberately left:** §7.2's `consumes` drawdown (a contract cannot cover more
spend than it is worth) is specified but not implemented — `run_contract_coverage` is not
written here because it cannot be verified against a corpus with zero extracted contracts.
Tasks 10 and 11 deliver the profiles and their maths; wiring them into a resolution pass is
a follow-up once `bp_contracts` has rows. This is called out rather than stubbed.

**Type consistency checked:** `score(src, tgt) -> dict` in all four profiles;
`observations_for` returns `dict[str, frozenset]` everywhere; `DerivedEdge` field names match
between `edge_writer` and both `pass_runner` call sites; `PROFILE`/`VERSION` constants exist
on every profile module and are read by `pass_runner`.
