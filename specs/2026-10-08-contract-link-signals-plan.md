# Contract Link Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give contract parent-link scoring per-role profiles (hierarchy, amendment, attachment) and a set of corroborating signals that join a pair's score only when both documents carry the data.

**Architecture:** `linking_engine.py` is not touched. A small generic helper (`graph_resolution/applicability.py`) registers a profile *variant* per pair containing only the optional signals both documents can supply, so absent data cannot lower coverage `C`. Two new profile modules sit beside `contract_hierarchy`; `contract_links.py` picks the profile by the child's vocabulary role and widens the candidate rows so the new signals have data.

**Tech Stack:** Python 3.12, pytest, PostgreSQL (`proc` schema; `bp_testdb` for tests), the existing linking engine and graph_resolution profiles.

**Spec:** `specs/2026-10-08-contract-link-signals-design.md` (read its section 4 "Revision 1" first: it supersedes the draft's signal table).

## Global Constraints

- `src/services/linking_engine.py` is **not modified**. Its golden vectors, and every other profile's, stay byte-identical.
- Signal spec rows use the existing keys: `id, cluster, tier, weight, appl, cap, kind, reads`. Tier 2 = weight 3, cap 0.70; tier 3 = weight 2, cap 0.90 (tier 1 = weight 5, cap 0.45 is used only by the existing signals).
- A pair with **no optional data** must score exactly as today: SOW + resolving reference = **96.9**, SOW + same supplier and no reference = **75.6** (fixtures in Task 3).
- Corroborators add, they do not subtract: a mismatch on payment terms, signatory or cost centre is neutral (`s = 0.5`, status `"WEAK"`). Only currency, governing law, buyer, and value-exceeds-parent return `CONFLICT`.
- Money is compared only inside one currency. No FX rate is ever invented.
- Nothing auto-links. New profile names go into `UNCALIBRATED_PROFILES` (`graph_resolution/edge_writer.py`). Proposals still reach the queue only through `contract_links.propose_parent_links` and are linked only by `confirm()`.
- `default_parent_type` is **not** changed for schedule, SLA or termination notice.
- New SQL is additive, idempotent, with a rollback file, in `deploy/sql/`. The implementer applies migrations to `bp_testdb` only; `bp_sqldb` is applied by the controller after review.
- Test command prefix (call it `RUN`): `set -a; . ./.env 2>/dev/null; set +a; CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest`. Pure unit tests need no flag but are fine with it.
- The working tree is shared with other sessions. **Stage and commit only the files a task names**, using `git add <paths>` then `git commit -o <paths> -m "..."`. Never `git add -A`, never `git stash`. Each commit message ends with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Every task ends by **breaking its own guard on purpose and watching the test go red**, then restoring it.

## Review Focus

1. A contract with no optional fields at all (the corpus norm) must score as it does today; a profile that taxed absent data would silently stop proposing parents (Task 1, Task 3 tests).
2. An amendment whose title shares no words with its parent, but whose reference resolves, must be proposed (Task 4 test, Task 7 matrix).
3. A child whose currency differs from the parent's must not have a value compared (no FX) (Task 2 test).
4. A rows-from-`bp_contract_master` candidate has no `buyer_signatory_name` column; widening the query must not crash on it (Task 6 test).
5. A proposed vocabulary type must not route an upload or resolve a document (Task 8 test).

## File Structure

| File | Responsibility |
|---|---|
| `src/services/graph_resolution/applicability.py` (new) | Generic: pick the optional signals a pair can be evaluated on, register the variant profile, score, report the base profile name |
| `src/services/graph_resolution/profiles/contract_signals.py` (new) | The seven contract comparators and their spec rows; registers the `csh_*` kinds |
| `src/services/graph_resolution/profiles/contract_hierarchy.py` (modify) | `score()` delegates to `applicability.score_pair`; the five base signals are untouched |
| `src/services/graph_resolution/profiles/contract_amendment.py` (new) | Profile for `role.variation` children |
| `src/services/graph_resolution/profiles/contract_attachment.py` (new) | Profile for `role.attachment` children |
| `src/services/graph_resolution/edge_writer.py` (modify) | Two names added to `UNCALIBRATED_PROFILES` |
| `src/services/contract_links.py` (modify) | Profile selection by role, attachment parent types, widened candidate rows, link type |
| `src/services/concepts/seed.py`, `deploy/sql/2026-10-08_contract_link_vocabulary*.sql` (modify/new) | Four proposed document types |
| `tests/services/graph_resolution/test_applicability.py`, `test_contract_signals.py`, `test_contract_amendment.py`, `test_contract_attachment.py` (new); `test_contract_hierarchy.py` (modify); `tests/services/test_contract_link_matrix.py` (new); `tests/services/concepts/test_contract_link_types.py` (new) | Tests |

---

### Task 1: The applicability helper

**Files:**
- Create: `src/services/graph_resolution/applicability.py`
- Test: `tests/services/graph_resolution/test_applicability.py`

**Interfaces:**
- Consumes: `linking_engine.register_profile`, `linking_engine._signal_match`, `linking_engine.score_link`, `composition.remap_clusters`.
- Produces: `applicable(optional, src, tgt, date_field) -> list[dict]`, `variant_name(base, extra) -> str`, `score_pair(base_name, params, base_specs, optional, src, tgt) -> dict` (a `score_link` result plus `"profile": base_name`).

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_applicability.py
"""Optional signals join a pair only when both documents can supply them.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_applicability.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services import linking_engine as le                                  # noqa: E402
from src.services.graph_resolution import applicability as ap                  # noqa: E402

le.register_signal("tap_ref", lambda s, t, a, b: (1.0, "OK") if s.get("ref") == t.get("id") else (0.5, "MISSING"))
le.register_signal("tap_opt", lambda s, t, a, b:
                   (0.5, "MISSING") if not (s.get("x") and t.get("x"))
                   else ((1.0, "OK") if s["x"] == t["x"] else (0.0, "CONFLICT")))

BASE = [{"id": "ref", "cluster": "reference", "tier": 1, "weight": 5, "appl": 1.0,
         "cap": 0.45, "kind": "tap_ref", "reads": ["ref"]}]
OPT = [{"id": "opt", "cluster": "terms", "tier": 3, "weight": 2, "appl": 1.0,
        "cap": 0.90, "kind": "tap_opt", "reads": ["x"]}]
PARAMS = {"p0": 0.02, "alpha": 0.35, "floor": 0.55, "date_field": "d"}


def _direct_base_score(src, tgt):
    le.register_profile("tap_base_only", {**PARAMS, "signals": BASE})
    return le.score_link(src, tgt, "tap_base_only")


def test_absent_optional_data_scores_exactly_as_the_base_profile():
    src, tgt = {"id": "C", "ref": "P"}, {"id": "P"}
    got = ap.score_pair("tap_prof", PARAMS, BASE, OPT, src, tgt)
    assert got["F"] == _direct_base_score(src, tgt)["F"]
    assert got["profile"] == "tap_prof"
    assert "tap_prof" in le.PROFILES and [s["id"] for s in le.PROFILES["tap_prof"]["signals"]] == ["ref"]


def test_present_optional_data_joins_the_pair_and_is_reported_under_the_base_name():
    src, tgt = {"id": "C", "ref": "P", "x": "GBP"}, {"id": "P", "x": "GBP"}
    got = ap.score_pair("tap_prof2", PARAMS, BASE, OPT, src, tgt)
    assert [s["id"] for s in got["signals"]] == ["ref", "opt"]
    assert got["profile"] == "tap_prof2"
    assert "tap_prof2+opt" in le.PROFILES


def test_a_conflicting_optional_signal_lowers_the_score_below_the_base():
    src, tgt = {"id": "C", "ref": "P", "x": "GBP"}, {"id": "P", "x": "USD"}
    got = ap.score_pair("tap_prof3", PARAMS, BASE, OPT, src, tgt)
    assert got["F"] < _direct_base_score(src, tgt)["F"]


def test_the_variant_name_does_not_depend_on_declaration_order():
    a = {"id": "b"}, {"id": "a"}
    assert ap.variant_name("p", list(a)) == ap.variant_name("p", list(reversed(a))) == "p+a+b"
    assert ap.variant_name("p", []) == "p"

- [ ] **Step 2: Run it and see it fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/graph_resolution/test_applicability.py -v`
Expected: collection error, `ImportError: cannot import name 'applicability'`.

- [ ] **Step 3: Write the helper**

```python
# src/services/graph_resolution/applicability.py
"""Score a pair on the evidence BOTH documents can supply.

Why this exists. The engine's coverage term is
    C = floor + (1 - floor) * (sum of weight*r) / (sum of weight)
and the denominator counts every signal in the profile, observed or not. A
profile that carries eight corroborating signals most contracts leave empty
therefore taxes every document for data it was never going to have: measured
2026-10-08, an exact-reference SOW fell 96.9 -> 75.7 and a same-supplier SOW
with no reference fell 75.6 -> 61.4, under the 65.0 proposal floor.

The engine already has the idea this needs -- applicability (`appl`, `q`) -- but
fixes it per profile. Here it is decided per PAIR, outside the engine: an
optional signal joins only when its comparator can actually be evaluated on this
pair (status != MISSING). The pair is then scored under a variant profile named
base + "+" + sorted optional ids, registered on first use.

linking_engine.py is not edited; this uses only register_profile and score_link.
"""
from __future__ import annotations

from typing import Sequence

from src.services import linking_engine as _le
from .composition import remap_clusters


def applicable(optional: Sequence[dict], src: dict, tgt: dict, date_field: str) -> list[dict]:
    """The optional signals this pair can actually be evaluated on."""
    keep = []
    for spec in optional:
        _s, status = _le._signal_match(spec["kind"], src, tgt, [], [], date_field)
        if status != "MISSING":
            keep.append(spec)
    return keep


def variant_name(base: str, extra: Sequence[dict]) -> str:
    """Deterministic: independent of the order the optional signals were declared in."""
    return base if not extra else base + "+" + "+".join(sorted(s["id"] for s in extra))


def _observations(specs: Sequence[dict], src: dict, tgt: dict) -> dict:
    sid, tid = str(src.get("contract_id", id(src))), str(tgt.get("contract_id", id(tgt)))
    out = {}
    for spec in specs:
        obs = set()
        for field in spec["reads"]:
            if src.get(field) is not None:
                obs.add((sid, field))
            if tgt.get(field) is not None:
                obs.add((tid, field))
        out[spec["id"]] = frozenset(obs)
    return out


def score_pair(base_name: str, params: dict, base_specs: Sequence[dict],
               optional: Sequence[dict], src: dict, tgt: dict) -> dict:
    """Score src -> tgt under base_specs plus the optional signals that apply.

    ``params`` is the profile's p0/alpha/floor/date_field. The result is the
    engine's full auditable score_link result with ``profile`` set to the BASE
    name, so callers and the UNCALIBRATED_PROFILES check never see a variant name.
    """
    extra = applicable(optional, src, tgt, params["date_field"])
    name = variant_name(base_name, extra)
    specs = list(base_specs) + extra
    if name not in _le.PROFILES:
        _le.register_profile(name, {**params, "signals": specs})
    overrides = remap_clusters(specs, _observations(specs, src, tgt))
    result = _le.score_link(src, tgt, name, cluster_overrides=overrides)
    result["profile"] = base_name
    return result
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/graph_resolution/test_applicability.py -v`
Expected: 4 passed.

- [ ] **Step 5: Break the guard and watch it go red**

Change `if status != "MISSING":` to `if True:` in `applicable`, run the file. Expected: `test_absent_optional_data_scores_exactly_as_the_base_profile` FAILS (the variant carries the MISSING signal and coverage drops). Restore the line and re-run: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/services/graph_resolution/applicability.py tests/services/graph_resolution/test_applicability.py
git commit -o src/services/graph_resolution/applicability.py tests/services/graph_resolution/test_applicability.py -m "feat(linking): score a pair on the optional signals both documents can supply

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The contract comparators

**Files:**
- Create: `src/services/graph_resolution/profiles/contract_signals.py`
- Test: `tests/services/graph_resolution/test_contract_signals.py`

**Interfaces:**
- Consumes: `contract_hierarchy._norm_ref`, `contract_hierarchy._PLACEHOLDERS`, `linking_engine.register_signal`.
- Produces: comparators `cmp_buyer`, `cmp_value_rollup`, `cmp_currency`, `cmp_payment_terms`, `cmp_governing_law`, `cmp_signatory`, `cmp_cost_centre`, each `(src, tgt) -> (score, status)`; spec rows `BUYER`, `VALUE_ROLLUP`, `CURRENCY`, `PAYMENT_TERMS`, `GOVERNING_LAW`, `SIGNATORY`, `COST_CENTRE`; lists `HIERARCHY_OPTIONAL`, `AMENDMENT_OPTIONAL`, `ATTACHMENT_OPTIONAL`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_contract_signals.py
"""The seven corroborating comparators. Pure, offline.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_contract_signals.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.graph_resolution.profiles import contract_signals as cs  # noqa: E402


def test_buyer():
    assert cs.cmp_buyer({"buyer_org_id": "B-1"}, {"buyer_org_id": " b-1 "}) == (1.0, "OK")
    assert cs.cmp_buyer({"buyer_org_id": "B-1"}, {"buyer_org_id": "B-2"}) == (0.0, "CONFLICT")
    assert cs.cmp_buyer({"buyer_org_id": "B-1"}, {}) == (0.5, "MISSING")
    assert cs.cmp_buyer({"buyer_org_id": "TBC"}, {"buyer_org_id": "B-1"}) == (0.5, "MISSING")


def test_currency():
    assert cs.cmp_currency({"currency": "GBP"}, {"currency": "gbp"}) == (1.0, "OK")
    assert cs.cmp_currency({"currency": "GBP"}, {"currency": "USD"}) == (0.0, "CONFLICT")
    assert cs.cmp_currency({"currency": "GBP"}, {}) == (0.5, "MISSING")


def test_payment_terms_differing_is_neutral_not_a_conflict():
    assert cs.cmp_payment_terms({"payment_terms": "Net 30"}, {"payment_terms": "net-30"}) == (1.0, "OK")
    assert cs.cmp_payment_terms({"payment_terms": "Net 30"}, {"payment_terms": "Net 60"}) == (0.5, "WEAK")
    assert cs.cmp_payment_terms({}, {"payment_terms": "Net 60"}) == (0.5, "MISSING")


def test_governing_law_falls_back_to_jurisdiction():
    assert cs.cmp_governing_law({"governing_law": "England"}, {"governing_law": "england"}) == (1.0, "OK")
    assert cs.cmp_governing_law({"governing_law": "England"}, {"jurisdiction": "England"}) == (1.0, "OK")
    assert cs.cmp_governing_law({"governing_law": "England"}, {"governing_law": "Delaware"}) == (0.0, "CONFLICT")
    assert cs.cmp_governing_law({"governing_law": "England"}, {}) == (0.5, "MISSING")


def test_value_rollup_is_necessary_not_sufficient():
    fits = cs.cmp_value_rollup({"total_contract_value": 50, "currency": "GBP"},
                               {"total_contract_value": 100, "currency": "GBP"})
    assert fits == (0.5, "WEAK"), "a child that fits under its parent proves nothing"
    over = cs.cmp_value_rollup({"total_contract_value": 150, "currency": "GBP"},
                               {"total_contract_value": 100, "currency": "GBP"})
    assert over == (0.0, "CONFLICT")


def test_value_rollup_never_compares_across_currencies_or_without_a_parent_value():
    usd = cs.cmp_value_rollup({"total_contract_value": 150, "currency": "USD"},
                              {"total_contract_value": 100, "currency": "GBP"})
    assert usd == (0.5, "MISSING"), "no FX rate is ever invented"
    assert cs.cmp_value_rollup({"total_contract_value": 5, "currency": "GBP"},
                               {"total_contract_value": None, "currency": "GBP"}) == (0.5, "MISSING")
    assert cs.cmp_value_rollup({"total_contract_value": 5, "currency": "GBP"},
                               {"total_contract_value": 0, "currency": "GBP"}) == (0.5, "MISSING")


def test_signatory_shared_name_is_positive_and_a_different_one_is_neutral():
    a = {"contract_signatory_name": "Jane Doe"}
    assert cs.cmp_signatory(a, {"buyer_signatory_name": "jane  doe"}) == (1.0, "OK")
    assert cs.cmp_signatory(a, {"contract_signatory_name": "Jane Doe"}) == (1.0, "OK")
    assert cs.cmp_signatory(a, {"contract_signatory_name": "Sam Roe"}) == (0.5, "WEAK")
    assert cs.cmp_signatory(a, {}) == (0.5, "MISSING")


def test_cost_centre_any_shared_field_is_positive():
    a = {"cost_centre_id": "CC1", "spend_category": "IT"}
    assert cs.cmp_cost_centre(a, {"cost_centre_id": "CC9", "spend_category": "it"}) == (1.0, "OK")
    assert cs.cmp_cost_centre(a, {"cost_centre_id": "CC9"}) == (0.5, "WEAK")
    assert cs.cmp_cost_centre(a, {"business_unit_id": "BU"}) == (0.5, "MISSING")


def test_every_spec_row_has_the_engine_keys_and_a_registered_kind():
    from src.services import linking_engine as le
    keys = {"id", "cluster", "tier", "weight", "appl", "cap", "kind", "reads"}
    for spec in (cs.BUYER, cs.VALUE_ROLLUP, cs.CURRENCY, cs.PAYMENT_TERMS,
                 cs.GOVERNING_LAW, cs.SIGNATORY, cs.COST_CENTRE):
        assert keys <= set(spec), spec["id"]
        assert spec["kind"] in le._EXTRA_SIGNALS, spec["id"]


def test_the_amendment_list_omits_what_an_amendment_legitimately_changes():
    ids = {s["id"] for s in cs.AMENDMENT_OPTIONAL}
    assert ids == {"buyer", "currency", "governing_law", "signatory"}
    assert "payment_terms" not in ids and "value_rollup" not in ids
```

- [ ] **Step 2: Run it and see it fail**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/graph_resolution/test_contract_signals.py -v`
Expected: `ImportError` on `contract_signals`.

- [ ] **Step 3: Write the module**

```python
# src/services/graph_resolution/profiles/contract_signals.py
"""Corroborating signals for contract parent links.

These are OPTIONAL signals: applicability.score_pair adds one to a pair's profile
only when its comparator can be evaluated on that pair. Two rules shape them.

  * Corroborators add, they do not subtract. A SOW may legitimately name different
    payment terms, signatories or cost centres than its master agreement, so a
    mismatch there is NEUTRAL (0.5, "WEAK": no contribution, full coverage). Only
    currency, governing law, buyer, and a child that EXCEEDS its parent's value
    can CONFLICT -- and at tier 2/3 no conflict caps the score.
  * Necessary is not sufficient. A child's value fitting under the parent's proves
    nothing (a small value fits under any parent), so that case is neutral too.

Money is compared only inside one currency. No FX rate is ever invented.

Weights are DECLARED, not measured: no labelled sample of true parent links
exists (every stored parent_contract_id dangles).
"""
from __future__ import annotations

from typing import Optional

from src.services import linking_engine as _le
from . import contract_hierarchy as _ch


def _has(v) -> bool:
    return _ch._norm_ref(v) not in _ch._PLACEHOLDERS


def _eq(a, b) -> bool:
    return _ch._norm_ref(a) == _ch._norm_ref(b)


def _money(v) -> Optional[float]:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def cmp_buyer(src, tgt) -> tuple[float, str]:
    a, b = src.get("buyer_org_id"), tgt.get("buyer_org_id")
    if not (_has(a) and _has(b)):
        return 0.5, "MISSING"
    return (1.0, "OK") if _eq(a, b) else (0.0, "CONFLICT")


def cmp_currency(src, tgt) -> tuple[float, str]:
    a, b = src.get("currency"), tgt.get("currency")
    if not (_has(a) and _has(b)):
        return 0.5, "MISSING"
    return (1.0, "OK") if _eq(a, b) else (0.0, "CONFLICT")


def cmp_payment_terms(src, tgt) -> tuple[float, str]:
    a, b = src.get("payment_terms"), tgt.get("payment_terms")
    if not (_has(a) and _has(b)):
        return 0.5, "MISSING"
    return (1.0, "OK") if _eq(a, b) else (0.5, "WEAK")


def _law(row) -> Optional[str]:
    for field in ("governing_law", "jurisdiction"):
        if _has(row.get(field)):
            return row[field]
    return None


def cmp_governing_law(src, tgt) -> tuple[float, str]:
    a, b = _law(src), _law(tgt)
    if a is None or b is None:
        return 0.5, "MISSING"
    return (1.0, "OK") if _eq(a, b) else (0.0, "CONFLICT")


def cmp_value_rollup(src, tgt) -> tuple[float, str]:
    c, p = _money(src.get("total_contract_value")), _money(tgt.get("total_contract_value"))
    cs, ct = src.get("currency"), tgt.get("currency")
    if c is None or p is None or not (_has(cs) and _has(ct)) or not _eq(cs, ct):
        return 0.5, "MISSING"
    return (0.5, "WEAK") if c <= p else (0.0, "CONFLICT")


_SIGNATORY_FIELDS = ("contract_signatory_name", "buyer_signatory_name")


def _names(row) -> set:
    return {_ch._norm_ref(row.get(f)) for f in _SIGNATORY_FIELDS if _has(row.get(f))}


def cmp_signatory(src, tgt) -> tuple[float, str]:
    a, b = _names(src), _names(tgt)
    if not a or not b:
        return 0.5, "MISSING"
    return (1.0, "OK") if a & b else (0.5, "WEAK")


_COST_FIELDS = ("cost_centre_id", "business_unit_id", "spend_category")


def cmp_cost_centre(src, tgt) -> tuple[float, str]:
    pairs = [(src.get(f), tgt.get(f)) for f in _COST_FIELDS
             if _has(src.get(f)) and _has(tgt.get(f))]
    if not pairs:
        return 0.5, "MISSING"
    return (1.0, "OK") if any(_eq(a, b) for a, b in pairs) else (0.5, "WEAK")


for _kind, _fn in (("csh_buyer", cmp_buyer), ("csh_value", cmp_value_rollup),
                   ("csh_currency", cmp_currency), ("csh_payment", cmp_payment_terms),
                   ("csh_law", cmp_governing_law), ("csh_signatory", cmp_signatory),
                   ("csh_cost", cmp_cost_centre)):
    _le.register_signal(_kind, (lambda f: lambda s, t, sl, tl: f(s, t))(_fn))


def _spec(id_, cluster, tier, weight, cap, kind, reads):
    return {"id": id_, "cluster": cluster, "tier": tier, "weight": weight,
            "appl": 1.0, "cap": cap, "kind": kind, "reads": reads}


BUYER = _spec("buyer", "identity", 2, 3, 0.70, "csh_buyer", ["buyer_org_id"])
VALUE_ROLLUP = _spec("value_rollup", "commercial", 3, 2, 0.90, "csh_value",
                     ["total_contract_value", "currency"])
CURRENCY = _spec("currency", "terms", 3, 2, 0.90, "csh_currency", ["currency"])
PAYMENT_TERMS = _spec("payment_terms", "terms", 3, 2, 0.90, "csh_payment", ["payment_terms"])
GOVERNING_LAW = _spec("governing_law", "terms", 3, 2, 0.90, "csh_law",
                      ["governing_law", "jurisdiction"])
SIGNATORY = _spec("signatory", "people", 3, 2, 0.90, "csh_signatory", list(_SIGNATORY_FIELDS))
COST_CENTRE = _spec("cost_centre", "category", 3, 2, 0.90, "csh_cost", list(_COST_FIELDS))

HIERARCHY_OPTIONAL = [BUYER, VALUE_ROLLUP, CURRENCY, PAYMENT_TERMS, GOVERNING_LAW,
                      SIGNATORY, COST_CENTRE]
#: An amendment changes terms and value; it is not corroborated by repeating them.
AMENDMENT_OPTIONAL = [BUYER, CURRENCY, GOVERNING_LAW, SIGNATORY]
ATTACHMENT_OPTIONAL = [BUYER, CURRENCY, PAYMENT_TERMS, GOVERNING_LAW]
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/graph_resolution/test_contract_signals.py -v`
Expected: all pass. (`_norm_ref` strips spaces, punctuation and case, so "jane  doe" equals "Jane Doe".) If `test_value_rollup...` fails on the zero-value parent, `_money` rejects `<= 0` as designed and the assertion expects MISSING.

- [ ] **Step 5: Break two guards and watch them go red**

(a) In `cmp_value_rollup` delete `or not _eq(cs, ct)`; expected: `test_value_rollup_never_compares_across_currencies...` FAILS. Restore.
(b) In `cmp_payment_terms` change `(0.5, "WEAK")` to `(0.0, "CONFLICT")`; expected: `test_payment_terms_differing_is_neutral...` FAILS. Restore. Re-run: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/services/graph_resolution/profiles/contract_signals.py tests/services/graph_resolution/test_contract_signals.py
git commit -o src/services/graph_resolution/profiles/contract_signals.py tests/services/graph_resolution/test_contract_signals.py -m "feat(contracts): seven corroborating link signals - buyer, value, currency, payment terms, law, signatory, cost centre

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The hierarchy profile takes the optional signals

**Files:**
- Modify: `src/services/graph_resolution/profiles/contract_hierarchy.py` (the `_le.register_profile(...)` call near the bottom, and `score`)
- Test: `tests/services/graph_resolution/test_contract_hierarchy.py` (append)

**Interfaces:**
- Consumes: `applicability.score_pair`, `contract_signals.HIERARCHY_OPTIONAL`.
- Produces: `contract_hierarchy.score(src, tgt)` unchanged signature; result now also has `"profile": "contract_hierarchy"`. `contract_hierarchy._PARAMS`.

- [ ] **Step 1: Write the failing tests** (append to `test_contract_hierarchy.py`; it already defines `_sow()` and `_msa()`)

```python
def test_no_optional_data_scores_exactly_as_before_the_extension():
    """Review Focus 1: the corpus norm is a row with none of the corroborating fields."""
    with_ref = ch.score(_sow(), _msa())
    assert round(with_ref["F"], 1) == 96.9 and with_ref["decision"] == "auto_link"
    no_ref = ch.score(_sow(parent_agreement_ref=None), _msa())
    assert round(no_ref["F"], 1) == 75.6 and no_ref["decision"] == "review"
    assert with_ref["profile"] == "contract_hierarchy"
    assert [s["id"] for s in with_ref["signals"]] == [
        "declared_reference", "expected_structure", "supplier", "term_containment", "title_overlap"]


def test_a_matching_buyer_joins_the_score_and_a_conflicting_one_lowers_it():
    base = ch.score(_sow(parent_agreement_ref=None), _msa())["F"]
    same = ch.score(_sow(parent_agreement_ref=None, buyer_org_id="B-1"), _msa(buyer_org_id="B-1"))
    diff = ch.score(_sow(parent_agreement_ref=None, buyer_org_id="B-1"), _msa(buyer_org_id="B-2"))
    assert "buyer" in [s["id"] for s in same["signals"]]
    assert same["F"] > base > diff["F"]


def test_absent_corroborators_never_tax_a_pair_with_some_present():
    """Only the fields BOTH sides carry may enter the profile."""
    got = ch.score(_sow(currency="GBP", payment_terms="Net 30"), _msa(currency="GBP"))
    ids = [s["id"] for s in got["signals"]]
    assert "currency" in ids and "payment_terms" not in ids
```

- [ ] **Step 2: Run and see failure**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/graph_resolution/test_contract_hierarchy.py -q`
Expected: the three new tests FAIL (`KeyError: 'profile'`); every older test passes.

- [ ] **Step 3: Implement**

In `contract_hierarchy.py`, replace the existing `_le.register_profile(PROFILE, {...})` call with:

```python
_PARAMS = {"p0": 0.02, "alpha": 0.35, "floor": 0.55, "date_field": "contract_start_date"}
_le.register_profile(PROFILE, {**_PARAMS, "signals": SIGNALS})
```

and replace the whole `score` function with:

```python
def score(src: dict, tgt: dict) -> dict:
    """Score child -> parent, with correlated signals merged into one cluster.

    This is the entry point callers use, NOT score_link directly. The five base
    signals always apply; the corroborating signals in contract_signals join only
    when BOTH documents carry what they read (applicability.score_pair), so a
    contract with none of those fields scores exactly as it did before they
    existed. The result's `profile` is always the base name.
    """
    # Imported here, not at the top: contract_signals imports this module.
    from . import contract_signals as _cs
    from ..applicability import score_pair
    return score_pair(PROFILE, _PARAMS, SIGNALS, _cs.HIERARCHY_OPTIONAL, src, tgt)
```

Leave `observations_for` and `SIGNALS` unchanged (existing tests use them).

- [ ] **Step 4: Run the whole hierarchy and graph_resolution suites**

Run: `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/graph_resolution -q`
Expected: all pass, including the three new tests. If an OLD test now fails, do not edit it: read why. A pinned number that moved means the "no optional data scores as before" rule is broken.

- [ ] **Step 5: Break the guard**

In `applicability.applicable` (Task 1) the guard is already proven. Here, change `score` to pass `[]` instead of `_cs.HIERARCHY_OPTIONAL`; expected: `test_a_matching_buyer_joins...` FAILS. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/services/graph_resolution/profiles/contract_hierarchy.py tests/services/graph_resolution/test_contract_hierarchy.py
git commit -o src/services/graph_resolution/profiles/contract_hierarchy.py tests/services/graph_resolution/test_contract_hierarchy.py -m "feat(contracts): the hierarchy profile takes the corroborating signals when both sides carry them

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The amendment profile

**Files:**
- Create: `src/services/graph_resolution/profiles/contract_amendment.py`
- Modify: `src/services/graph_resolution/edge_writer.py` (the `UNCALIBRATED_PROFILES` frozenset)
- Test: `tests/services/graph_resolution/test_contract_amendment.py`

**Interfaces:**
- Consumes: `contract_hierarchy.SIGNALS`, `contract_signals.AMENDMENT_OPTIONAL`, `applicability.score_pair`.
- Produces: `contract_amendment.PROFILE = "contract_amendment"`, `contract_amendment.score(src, tgt) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_contract_amendment.py
"""An amendment is identified by what it amends, not by repeating its title.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_contract_amendment.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services import linking_engine as le                                   # noqa: E402
from src.services.graph_resolution import edge_writer                           # noqa: E402
from src.services.graph_resolution.profiles import contract_amendment as am     # noqa: E402


def _parent(**o):
    r = {"contract_id": "SOW-1", "contract_title": "Statement of Work Helix Migration",
         "supplier_id": "S-1", "resolved_doc_type": "doctype.sow",
         "contract_start_date": "2026-01-01", "contract_end_date": "2027-12-31"}
    r.update(o)
    return r


def _amend(**o):
    r = {"contract_id": "ADD-1", "contract_title": "Addendum No. 1", "supplier_id": "S-1",
         "resolved_doc_type": "doctype.addendum", "parent_agreement_ref": "SOW-1",
         "parent_contract_id": None, "framework_ref": None, "_ref_resolves": True,
         "contract_start_date": "2026-03-01", "contract_end_date": "2026-09-30"}
    r.update(o)
    return r


def test_the_profile_is_registered_and_never_reaches_auto_link():
    assert am.PROFILE in le.PROFILES
    assert am.PROFILE in edge_writer.UNCALIBRATED_PROFILES


def test_a_resolving_reference_with_a_generic_title_is_proposed():
    """The defect found 2026-10-08: 'Addendum No. 1' scored 36 under the hierarchy profile."""
    got = am.score(_amend(), _parent())
    assert got["F"] >= 65.0, got
    assert got["profile"] == "contract_amendment"
    ids = [s["id"] for s in got["signals"]]
    assert "title_overlap" not in ids and "expected_structure" not in ids


def test_a_matching_buyer_strengthens_it():
    plain = am.score(_amend(), _parent())["F"]
    with_buyer = am.score(_amend(buyer_org_id="B-1"), _parent(buyer_org_id="B-1"))["F"]
    assert with_buyer > plain


def test_no_reference_is_not_proposed():
    """Supplier alone cannot say WHICH contract is being amended."""
    got = am.score(_amend(parent_agreement_ref=None, _ref_resolves=False), _parent())
    assert got["F"] < 65.0


def test_a_reference_to_a_different_real_contract_is_a_conflict():
    got = am.score(_amend(parent_agreement_ref="SOW-9", _ref_resolves=True), _parent())
    assert got["F"] < 50.0


def test_payment_terms_are_never_read_for_an_amendment():
    got = am.score(_amend(payment_terms="Net 60"), _parent(payment_terms="Net 30"))
    assert "payment_terms" not in [s["id"] for s in got["signals"]]
```

- [ ] **Step 2: Run and see failure** — `ImportError` on `contract_amendment`.

- [ ] **Step 3: Implement**

```python
# src/services/graph_resolution/profiles/contract_amendment.py
"""Which contract does this variation, addendum or CCN amend?

The hierarchy profile weighs a title overlap, which is right for a SOW (it repeats
its master agreement's subject) and wrong for an amendment: "Addendum No. 1" shares
no words with the contract it amends, and scored 36.0 (title read as CONFLICT) even
with a resolving reference, the same supplier and a contained term. An amendment is
identified by what it names, so this profile has no title signal.

Also left out, deliberately:
  * expected_structure -- a variation amends ANY structure, so it is always MISSING
    for these types, and a signal that cannot be observed only lowers coverage;
  * value_rollup and payment_terms -- changing value and terms is what an
    amendment does, so repeating them is not corroboration.

Weights are DECLARED, not measured, exactly like contract_hierarchy.
"""
from __future__ import annotations

from src.services import linking_engine as _le
from ..applicability import score_pair
from . import contract_hierarchy as _ch
from . import contract_signals as _cs

PROFILE = "contract_amendment"

_BY_ID = {s["id"]: s for s in _ch.SIGNALS}
SIGNALS = [_BY_ID["declared_reference"], _BY_ID["supplier"], _BY_ID["term_containment"]]
_PARAMS = dict(_ch._PARAMS)

_le.register_profile(PROFILE, {**_PARAMS, "signals": SIGNALS})


def score(src: dict, tgt: dict) -> dict:
    return score_pair(PROFILE, _PARAMS, SIGNALS, _cs.AMENDMENT_OPTIONAL, src, tgt)
```

In `edge_writer.py`, add `"contract_amendment", "contract_attachment"` to `UNCALIBRATED_PROFILES` (both now, so Task 5 needs no second edit):

```python
UNCALIBRATED_PROFILES = frozenset({
    "contract_coverage", "contract_succession", "contract_hierarchy",
    "contract_amendment", "contract_attachment",
    "supplier_identity", "item_equivalence",
})
```

- [ ] **Step 4: Run** — `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/graph_resolution -q`. Expected: all pass. If `test_a_resolving_reference...` reads under 65.0, do not lower the bar: print `got["signals"]` and find which signal is not what the design says (the design measured 65.9 for this case).

- [ ] **Step 5: Break the guard** — in `contract_amendment.py` add `_BY_ID["title_overlap"]` to `SIGNALS`; expected: `test_a_resolving_reference_with_a_generic_title_is_proposed` FAILS. Remove it.

- [ ] **Step 6: Commit** (`git add` then `git commit -o` the three paths: the module, `edge_writer.py`, the test), message `feat(contracts): an amendment is scored on what it names, not on its title`.

---

### Task 5: The attachment profile

**Files:**
- Create: `src/services/graph_resolution/profiles/contract_attachment.py`
- Test: `tests/services/graph_resolution/test_contract_attachment.py`

**Interfaces:**
- Consumes: `contract_hierarchy.SIGNALS`, `contract_signals.ATTACHMENT_OPTIONAL`, `applicability.score_pair`.
- Produces: `contract_attachment.PROFILE = "contract_attachment"`, `contract_attachment.score(src, tgt) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/graph_resolution/test_contract_attachment.py
"""A schedule or SLA attaches to the agreement that cites it.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/graph_resolution/test_contract_attachment.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services import linking_engine as le                                    # noqa: E402
from src.services.graph_resolution import edge_writer                            # noqa: E402
from src.services.graph_resolution.profiles import contract_attachment as at     # noqa: E402


def _parent(**o):
    r = {"contract_id": "MSA-1", "contract_title": "Master Services Agreement Helix",
         "supplier_id": "S-1", "resolved_doc_type": "doctype.master_agreement",
         "contract_start_date": "2026-01-01", "contract_end_date": "2027-12-31"}
    r.update(o)
    return r


def _sla(**o):
    r = {"contract_id": "SLA-1", "contract_title": "Service Level Agreement Helix",
         "supplier_id": "S-1", "resolved_doc_type": "doctype.sla",
         "parent_agreement_ref": "MSA-1", "parent_contract_id": None, "framework_ref": None,
         "_ref_resolves": True,
         "contract_start_date": "2026-03-01", "contract_end_date": "2026-09-30"}
    r.update(o)
    return r


def test_registered_and_uncalibrated():
    assert at.PROFILE in le.PROFILES and at.PROFILE in edge_writer.UNCALIBRATED_PROFILES


def test_an_sla_naming_its_agreement_is_proposed_it():
    got = at.score(_sla(), _parent())
    assert got["F"] >= 80.0 and got["profile"] == "contract_attachment"


def test_an_sla_with_no_reference_and_unrelated_words_is_not():
    got = at.score(_sla(parent_agreement_ref=None, _ref_resolves=False,
                        contract_title="Service Levels"), _parent())
    assert got["F"] < 65.0


def test_the_title_signal_is_kept_for_attachments():
    ids = [s["id"] for s in at.score(_sla(), _parent())["signals"]]
    assert "title_overlap" in ids and "expected_structure" not in ids


def test_payment_terms_corroborate_an_attachment():
    plain = at.score(_sla(), _parent())["F"]
    both = at.score(_sla(payment_terms="Net 30"), _parent(payment_terms="Net 30"))["F"]
    assert both > plain
```

- [ ] **Step 2: Run and see failure** — `ImportError`.

- [ ] **Step 3: Implement**

```python
# src/services/graph_resolution/profiles/contract_attachment.py
"""Which agreement does this schedule or SLA attach to?

A schedule or SLA has no force on its own: it is cited by an agreement. Its title
usually shares the subject of that agreement (an SLA for a named service), so the
title signal is kept here, unlike for an amendment. expected_structure is left
out: a schedule sits under SEVERAL kinds of agreement, the vocabulary's
default_parent_type holds only one value, and a signal that is always MISSING only
lowers coverage. Candidate parent types are chosen in contract_links.

Weights are DECLARED, not measured.
"""
from __future__ import annotations

from src.services import linking_engine as _le
from ..applicability import score_pair
from . import contract_hierarchy as _ch
from . import contract_signals as _cs

PROFILE = "contract_attachment"

_BY_ID = {s["id"]: s for s in _ch.SIGNALS}
SIGNALS = [_BY_ID["declared_reference"], _BY_ID["supplier"],
           _BY_ID["term_containment"], _BY_ID["title_overlap"]]
_PARAMS = dict(_ch._PARAMS)

_le.register_profile(PROFILE, {**_PARAMS, "signals": SIGNALS})


def score(src: dict, tgt: dict) -> dict:
    return score_pair(PROFILE, _PARAMS, SIGNALS, _cs.ATTACHMENT_OPTIONAL, src, tgt)
```

- [ ] **Step 4: Run** — `CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest tests/services/graph_resolution -q`. Expected all pass (design measured 84.7 for the reference case).

- [ ] **Step 5: Break the guard** — add `_BY_ID["expected_structure"]` to `SIGNALS`; expected: `test_the_title_signal_is_kept_for_attachments` FAILS. Remove it.

- [ ] **Step 6: Commit** the module and its test: `feat(contracts): a schedule or SLA is scored for the agreement it attaches to`.

---

### Task 6: contract_links picks the profile and widens the rows

**Files:**
- Modify: `src/services/contract_links.py` (imports ~line 59; `_CHILD_SQL`; `_wanted_parent_types`; `candidate_parents`; the scoring/notes block in `propose_parent_links`)
- Test: `tests/services/test_contract_links.py` (append; live-DB gated)

**Interfaces:**
- Consumes: `contract_amendment.score`, `contract_attachment.score`, `contract_hierarchy.score`, `ensure_vocabulary().document_types[...].role`.
- Produces: `_profile_module(child) -> module`, `_link_type(child) -> str` (`"child_of" | "amends" | "attaches_to"`), `_fetch(cur, table, where, params) -> list[dict]`; `propose_parent_links` details gain `link_type` and `profile`.

- [ ] **Step 1: Write the failing tests** (append to `tests/services/test_contract_links.py`; it already defines `CL`, `fixture_contracts`, `_open_proposals`, the live-DB `pytestmark`)

```python
def test_profile_and_link_type_follow_the_childs_role():
    from src.services.graph_resolution.profiles import (
        contract_amendment, contract_attachment, contract_hierarchy)
    assert CL._profile_module({"resolved_doc_type": "doctype.addendum"}) is contract_amendment
    assert CL._profile_module({"resolved_doc_type": "doctype.ccn"}) is contract_amendment
    assert CL._profile_module({"resolved_doc_type": "doctype.sla"}) is contract_attachment
    assert CL._profile_module({"resolved_doc_type": "doctype.schedule"}) is contract_attachment
    assert CL._profile_module({"resolved_doc_type": "doctype.sow"}) is contract_hierarchy
    assert CL._link_type({"resolved_doc_type": "doctype.addendum"}) == "amends"
    assert CL._link_type({"resolved_doc_type": "doctype.sla"}) == "attaches_to"
    assert CL._link_type({"resolved_doc_type": "doctype.sow"}) == "child_of"


def test_schedules_and_slas_are_children_with_master_and_framework_parents():
    wanted = CL._wanted_parent_types({"resolved_doc_type": "doctype.sla"})
    assert "doctype.master_agreement" in wanted and "doctype.framework_agreement" in wanted
    assert "doctype.sla" not in wanted and "doctype.addendum" not in wanted
    assert CL.is_child({"resolved_doc_type": "doctype.schedule"})
    assert not CL.is_child({"resolved_doc_type": "doctype.termination_notice"})


def test_candidate_rows_carry_the_corroborating_fields_from_both_tables(fixture_contracts):
    """Review Focus 4: bp_contract_master has no buyer_signatory_name; it must not crash."""
    from src.services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        child = {"contract_id": "X-NONE", "resolved_doc_type": "doctype.sow",
                 "supplier_id": fixture_contracts["supplier"]}
        rows = CL.candidate_parents(cur, child)
        master = CL._fetch(cur, "proc.bp_contract_master", "supplier_id = %s", ("nobody",))
    assert any(r["contract_id"] == fixture_contracts["msa"] for r in rows)
    for r in rows:
        assert {"buyer_org_id", "currency", "payment_terms", "governing_law",
                "contract_signatory_name", "buyer_signatory_name", "cost_centre_id"} <= set(r)
    assert master == []


def test_a_proposal_for_an_addendum_says_amend_and_reports_its_link_type():
    from src.services.db import get_conn
    import uuid
    tag = uuid.uuid4().hex[:6].upper()
    sup, sow, add = f"S-{tag}", f"SOW-{tag}", f"ADD-{tag}"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""INSERT INTO proc.bp_contracts (contract_id, contract_title, supplier_id,
                contract_start_date, contract_end_date, resolved_doc_type, resolved_role, type_agreement)
            VALUES (%s,'Statement of Work Helix',%s,'2026-01-01','2027-12-31','doctype.sow','role.master','refined')""",
                    (sow, sup))
        cur.execute("""INSERT INTO proc.bp_contracts (contract_id, contract_title, supplier_id,
                contract_start_date, contract_end_date, resolved_doc_type, resolved_role, type_agreement,
                parent_agreement_ref)
            VALUES (%s,'Addendum No. 1',%s,'2026-03-01','2026-09-30','doctype.addendum','role.variation','refined',%s)""",
                    (add, sup, sow))
    try:
        result = CL.propose_parent_links(contract_id=add)
        assert result["details"][0]["link_type"] == "amends"
        assert result["details"][0]["profile"] == "contract_amendment"
        row = _open_proposals(add)[0]
        assert row["expected_value"] == sow and "appears to amend contract" in row["notes"]
    finally:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate = ANY(%s)", ([add, sow],))
            cur.execute("DELETE FROM proc.bp_contracts WHERE contract_id = ANY(%s)", ([add, sow],))
```

- [ ] **Step 2: Run and see failure**

Run: `RUN tests/services/test_contract_links.py -q -k "profile_and_link or schedules_and_slas or corroborating or says_amend"`
Expected: 4 FAIL (`AttributeError: ... _profile_module`).

- [ ] **Step 3: Implement**

(a) Imports, next to the existing `from src.services.graph_resolution.profiles import contract_hierarchy as _ch`:

```python
from src.services.graph_resolution.profiles import contract_amendment as _am
from src.services.graph_resolution.profiles import contract_attachment as _at
```

(b) Replace the `_CHILD_SQL` constant with:

```python
_CORROBORATING = """c.buyer_org_id, c.currency, c.payment_terms, c.governing_law, c.jurisdiction,
           c.contract_signatory_name, c.buyer_signatory_name, c.cost_centre_id,
           c.business_unit_id, c.spend_category"""
_CHILD_SQL = f"""
    SELECT c.contract_id, c.contract_title, c.supplier_id, c.resolved_doc_type,
           c.resolved_role, c.framework_ref, c.parent_agreement_ref, c.parent_contract_id,
           c.contract_start_date, c.contract_end_date, c.total_contract_value,
           {_CORROBORATING}
      FROM proc.bp_contracts c
     WHERE c.resolved_doc_type IS NOT NULL
"""
```

(`currency` is now selected once, inside `_CORROBORATING`; the old list ended `..., c.total_contract_value, c.currency`.)

(c) After `_is_variation`, add:

```python
def _role_of(child: dict):
    dt = ensure_vocabulary().document_types.get(child.get("resolved_doc_type"))
    return dt.role if dt else child.get("resolved_role")


def _is_attachment(child: dict) -> bool:
    """A schedule or SLA: cited by an agreement, with no force of its own."""
    return _role_of(child) == "role.attachment"


def _profile_module(child: dict):
    """The scoring profile for this child, chosen by its role in the vocabulary."""
    if _is_variation(child):
        return _am
    if _is_attachment(child):
        return _at
    return _ch


def _link_type(child: dict) -> str:
    if _is_variation(child):
        return "amends"
    if _is_attachment(child):
        return "attaches_to"
    return "child_of"
```

(d) In `_wanted_parent_types`, after the `if _is_variation(child): ... return {...}` block and before `want = ...`, add:

```python
    if _is_attachment(child):
        # A schedule sits under SEVERAL kinds of agreement, and default_parent_type
        # holds one value (also read by the upload gate), so the set is chosen here.
        return {code for code, dt in ensure_vocabulary().document_types.items()
                if dt.pipeline_doc_type == "contract"
                and dt.role in ("role.master", "role.framework")}
```

(e) Add the shared row fetcher above `candidate_parents`:

```python
_COMMON_COLS = ("contract_id", "contract_title", "supplier_id", "contract_start_date",
                "contract_end_date", "buyer_org_id", "currency", "total_contract_value",
                "payment_terms", "governing_law", "jurisdiction", "contract_signatory_name",
                "cost_centre_id", "business_unit_id", "spend_category")


def _fetch(cur, table: str, where: str, params: tuple) -> list[dict]:
    """Candidate rows from either table, with the same keys.

    bp_contract_master holds the free-text contract_type (read as a structure, never
    written back) and has no buyer_signatory_name; bp_contracts holds the resolved
    structure and both. The result has resolved_doc_type and buyer_signatory_name
    either way, so a profile never has to know which table a candidate came from.
    """
    master = table.endswith("bp_contract_master")
    cols = list(_COMMON_COLS) + (["contract_type"] if master
                                 else ["resolved_doc_type", "buyer_signatory_name"])
    cur.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE {where}", params)
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    if master:
        for row in rows:
            row["resolved_doc_type"] = structure_for_contract_type(row.pop("contract_type"))
            row["buyer_signatory_name"] = None
    return rows
```

(f) In `candidate_parents`, keep the docstring and the `wanted`/`supplier`/`out`/`seen` setup lines exactly; replace everything from `# 1. The contracts this document names.` to the final `return out` with:

```python
    # 1. The contracts this document names. No supplier needed.
    own = _ch._norm_ref(child.get("contract_id"))
    norm = "upper(regexp_replace(contract_id, '[^A-Za-z0-9]', '', 'g'))"
    for ref in _claimed_references(child):
        if _ch._norm_ref(ref) == own:
            continue                     # itself; _drop_self_parent_reference's case
        where = f"{norm} = upper(regexp_replace(%s, '[^A-Za-z0-9]', '', 'g'))"
        rows = (_fetch(cur, "proc.bp_contracts", where, (ref,))
                or _fetch(cur, "proc.bp_contract_master", where, (ref,)))
        for row in rows:
            key = _ch._norm_ref(row["contract_id"])
            if key in seen or key == own:
                continue
            if row["resolved_doc_type"] not in wanted:
                continue                 # evidence, not an override
            seen.add(key)
            out.append(row)

    # 2. Everything this supplier holds that the child could sit under.
    if not supplier:
        return out
    for row in _fetch(cur, "proc.bp_contracts",
                      "supplier_id = %s AND resolved_doc_type = ANY(%s) AND contract_id <> %s",
                      (supplier, sorted(wanted), child.get("contract_id"))):
        key = _ch._norm_ref(row["contract_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    for row in _fetch(cur, "proc.bp_contract_master",
                      "supplier_id = %s AND contract_id <> %s",
                      (supplier, child.get("contract_id"))):
        key = _ch._norm_ref(row["contract_id"])
        if key in seen:
            continue
        if row["resolved_doc_type"] in wanted:
            seen.add(key)
            out.append(row)
    return out
```

(g) In `propose_parent_links`, replace `((_ch.score(scoring_child, parent), parent)` with `((module.score(scoring_child, parent), parent)`, and add directly after `scoring_child["_ref_resolves"] = ...`:

```python
            module = _profile_module(child)
            link_type = _link_type(child)
            verb = {"child_of": "sit under", "amends": "amend",
                    "attaches_to": "attach to"}[link_type]
```

Change the notes line `f"sit under contract {best_parent['contract_id']} "` to `f"{verb} contract {best_parent['contract_id']} "`, and add `"link_type": link_type, "profile": module.PROFILE,` to the dict appended to `details`. The `scored = sorted(...)` expression uses `module`, so move the two lines defining `module`/`link_type`/`verb` above it.

- [ ] **Step 4: Run the contract suites**

Run: `RUN tests/services/test_contract_links.py tests/services/test_contract_link_wiring.py tests/services/graph_resolution -q`
Expected: all pass. The old note text "appears to sit under" is unchanged for hierarchy children, so older assertions on it still hold.

- [ ] **Step 5: Break two guards**

(a) In `_profile_module` return `_ch` always; expected: `test_profile_and_link_type_follow_the_childs_role` and `test_a_proposal_for_an_addendum_says_amend...` FAIL. Restore.
(b) In `_fetch` delete the `row["buyer_signatory_name"] = None` line; expected: `test_candidate_rows_carry_the_corroborating_fields...` FAILS (KeyError on a master-table candidate, if the supplier has any master rows; otherwise assert it with `bp_contract_master` rows by choosing a supplier from `SELECT supplier_id FROM proc.bp_contract_master LIMIT 1` in the test). Restore.

- [ ] **Step 6: Commit** `src/services/contract_links.py` and `tests/services/test_contract_links.py`: `feat(contracts): pick the scoring profile by the child's role; carry the corroborating fields; say amend or attach`.

---

### Task 7: The matrix, kept

**Files:**
- Create: `tests/services/test_contract_link_matrix.py`

**Interfaces:** consumes `contract_links.propose_parent_links(contract_id=...)` and the live `proc.bp_contracts`.

- [ ] **Step 1: Write the test** (live-DB gated; rows are created per case and removed in `finally`, and each run is scoped to its own child so no real contract is touched)

```python
# tests/services/test_contract_link_matrix.py
"""Every contract child type x six situations, through the real runner.

    PROCWISE_TEST_LIVE_DB=1 CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/test_contract_link_matrix.py -v
"""
from __future__ import annotations

import os
import re
import uuid

import pytest

from src.services import contract_links as CL
from src.services.db import get_conn

pytestmark = pytest.mark.skipif(
    os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() not in ("1", "true", "yes", "on"),
    reason="needs PROCWISE_TEST_LIVE_DB=1")

# child type, parent type, a parent type that must NOT be accepted, child title, role
CASES = {
    "sow":       ("doctype.sow", "doctype.master_agreement", "doctype.framework_agreement",
                  "Statement of Work Helix", "role.master", "Master Services Agreement Helix"),
    "call_off":  ("doctype.call_off_contract", "doctype.framework_agreement", "doctype.master_agreement",
                  "Call-Off Contract Helix", "role.master", "Framework Agreement Helix"),
    "order_form": ("doctype.order_form", "doctype.framework_agreement", "doctype.master_agreement",
                   "Order Form Helix", "role.master", "Framework Agreement Helix"),
    "variation": ("doctype.variation", "doctype.sow", None, "Variation 2", "role.variation",
                  "Statement of Work Helix Migration"),
    "addendum":  ("doctype.addendum", "doctype.master_agreement", None, "Addendum No. 1",
                  "role.variation", "Master Services Agreement Helix"),
    "ccn":       ("doctype.ccn", "doctype.sow", None, "Change Control Note 4", "role.variation",
                  "Statement of Work Helix Migration"),
    "sla":       ("doctype.sla", "doctype.master_agreement", None, "Service Level Agreement Helix",
                  "role.attachment", "Master Services Agreement Helix"),
    "schedule":  ("doctype.schedule", "doctype.master_agreement", None, "Schedule 2 Helix Services",
                  "role.attachment", "Master Services Agreement Helix"),
}
AMENDMENT_TYPES = {"variation", "addendum", "ccn"}


def _insert(made, cid, dtype, title, sup, start, end, role, ref=None):
    made.append(cid)
    with get_conn() as c:
        c.cursor().execute(
            """INSERT INTO proc.bp_contracts (contract_id, contract_title, supplier_id,
                   contract_start_date, contract_end_date, resolved_doc_type, resolved_role,
                   type_agreement, parent_agreement_ref)
               VALUES (%s,%s,%s,%s,%s,%s,%s,'refined',%s)""",
            (cid, title, sup, start, end, dtype, role, ref))


def _proposal(cid):
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("""SELECT expected_value, notes FROM proc.bp_extraction_discrepancy
                        WHERE doc_pk_candidate=%s AND issue_type='contract_parent_proposed'
                          AND status='open'""", (cid,))
        return cur.fetchone()


@pytest.fixture()
def world():
    made = []
    yield made
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate = ANY(%s)", (made,))
        cur.execute("DELETE FROM proc.bp_contracts WHERE contract_id = ANY(%s)", (made,))


def _score(notes):
    return float(re.search(r"score ([\d.]+)", notes).group(1))


@pytest.mark.parametrize("name", sorted(CASES))
def test_declared_reference_and_same_supplier_finds_the_right_parent(world, name):
    ct, pt, _wrong, title, role, ptitle = CASES[name]
    k = uuid.uuid4().hex[:6].upper()
    sup, P, C = f"S-{k}", f"P-{k}", f"C-{k}"
    _insert(world, P, pt, ptitle, sup, "2026-01-01", "2027-12-31", "role.master")
    _insert(world, C, ct, title, sup, "2026-03-01", "2026-09-30", role, ref=P)
    r = CL.propose_parent_links(contract_id=C)
    row = _proposal(C)
    assert row and row[0] == P, (name, r)
    assert _score(row[1]) >= 65.0, row
    if name not in AMENDMENT_TYPES and name not in ("sla", "schedule"):
        assert _score(row[1]) >= 92.0, "an exact reference on a SOW/call-off/order form is auto_link band"


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_reference_is_proposed_only_for_the_types_identified_by_supplier_and_structure(world, name):
    ct, pt, _wrong, title, role, ptitle = CASES[name]
    k = uuid.uuid4().hex[:6].upper()
    sup, P, C = f"S-{k}", f"P-{k}", f"C-{k}"
    _insert(world, P, pt, ptitle, sup, "2026-01-01", "2027-12-31", "role.master")
    _insert(world, C, ct, title, sup, "2026-03-01", "2026-09-30", role)
    CL.propose_parent_links(contract_id=C)
    row = _proposal(C)
    if name in ("sow", "call_off", "order_form"):
        assert row and row[0] == P
    else:
        assert row is None, "an amendment or attachment with no reference is not guessed a parent"


@pytest.mark.parametrize("name", ["sow", "call_off", "order_form"])
def test_only_the_wrong_parent_type_proposes_nothing(world, name):
    ct, _pt, wrong, title, role, _ = CASES[name]
    k = uuid.uuid4().hex[:6].upper()
    sup, P, C = f"S-{k}", f"P-{k}", f"C-{k}"
    _insert(world, P, wrong, "Wrong Type Helix", sup, "2026-01-01", "2027-12-31", "role.master")
    _insert(world, C, ct, title, sup, "2026-03-01", "2026-09-30", role)
    r = CL.propose_parent_links(contract_id=C)
    assert _proposal(C) is None and r["no_candidate"] == 1


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_reference_naming_a_parent_of_another_supplier_is_not_proposed(world, name):
    ct, pt, _wrong, title, role, ptitle = CASES[name]
    k = uuid.uuid4().hex[:6].upper()
    P, C = f"P-{k}", f"C-{k}"
    _insert(world, P, pt, ptitle, f"S-{k}1", "2026-01-01", "2027-12-31", "role.master")
    _insert(world, C, ct, title, f"S-{k}2", "2026-03-01", "2026-09-30", role, ref=P)
    CL.propose_parent_links(contract_id=C)
    row = _proposal(C)
    # The reference resolves a candidate (step 1 of candidate_parents), but the supplier
    # conflict (tier 1) caps the score below the proposal floor.
    assert row is None


@pytest.mark.parametrize("name", ["sow", "call_off", "order_form"])
def test_two_equal_parents_are_contested(world, name):
    ct, pt, _wrong, title, role, ptitle = CASES[name]
    k = uuid.uuid4().hex[:6].upper()
    sup, C = f"S-{k}", f"C-{k}"
    _insert(world, f"P1-{k}", pt, ptitle, sup, "2026-01-01", "2027-12-31", "role.master")
    _insert(world, f"P2-{k}", pt, ptitle, sup, "2026-01-01", "2027-12-31", "role.master")
    _insert(world, C, ct, title, sup, "2026-03-01", "2026-09-30", role)
    r = CL.propose_parent_links(contract_id=C)
    assert r["contested"] == 1 and r["proposed"] == 0
```

- [ ] **Step 2: Run it**

Run: `RUN tests/services/test_contract_link_matrix.py -v`
Expected: all pass. Anything that fails is a finding about the design or the code, never about the test: report the actual score and which signal moved; do NOT change an expectation to match.

- [ ] **Step 3: Break the guard** — in `_profile_module` (contract_links) return `_ch` for variations; expected: the three amendment-type cases in `test_declared_reference_and_same_supplier...` FAIL (title read as CONFLICT again). Restore.

- [ ] **Step 4: Commit** the test file: `test(contracts): every contract child type through the real runner, six situations each`.

---

### Task 8: Four proposed document types

**Files:**
- Modify: `src/services/concepts/seed.py` (`_DOCUMENT_TYPE_CONCEPTS`, the `Concept(...)` comprehension status expression, the `DOCUMENT_TYPES` tuple)
- Create: `deploy/sql/2026-10-08_contract_link_vocabulary.sql`, `deploy/sql/2026-10-08_contract_link_vocabulary_rollback.sql`
- Test: `tests/services/concepts/test_contract_link_types.py`

**Interfaces:** produces four `proposed` document types: `doctype.dpa` (role.attachment), `doctype.side_letter` (role.variation), `doctype.renewal` (role.variation), `doctype.guaranty` (role.supporting); none has a pipeline.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/concepts/test_contract_link_types.py
"""Four types the vocabulary knows about but nothing may resolve to yet.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/concepts/test_contract_link_types.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import seed                                          # noqa: E402

NEW = {"doctype.dpa": "role.attachment", "doctype.side_letter": "role.variation",
       "doctype.renewal": "role.variation", "doctype.guaranty": "role.supporting"}


def test_the_four_types_are_seeded_proposed_with_no_pipeline():
    types = {t.concept_code: t for t in seed.DOCUMENT_TYPES}
    for code, role in NEW.items():
        assert code in types and code in seed.CONCEPTS, code
        assert types[code].status == "proposed" and seed.CONCEPTS[code].status == "proposed"
        assert types[code].role == role and types[code].pipeline_doc_type is None


def test_a_proposed_type_never_routes_an_upload():
    """Review Focus 5: only status='active' rows route, whichever vocabulary is loaded."""
    import pytest
    from src.services.concepts import routing as R
    types = {t.concept_code: t for t in seed.DOCUMENT_TYPES}
    for code in NEW:
        for alias in types[code].aliases:
            with pytest.raises(R.UnknownDocumentCategory):
                R.pipeline_for_category(alias)
```

- [ ] **Step 2: Run and see failure** — the four codes are not in `seed.DOCUMENT_TYPES`.

- [ ] **Step 3: Implement**

`seed.py`: add to `_DOCUMENT_TYPE_CONCEPTS`:

```python
    ("doctype.dpa",
     "Governs how personal data is processed for a parent agreement; forms part of it.",
     ("doctype.addendum",)),
    ("doctype.side_letter",
     "A separate letter that modifies or waives a term of an agreement it names.",
     ("doctype.variation",)),
    ("doctype.renewal",
     "Extends an agreement past its expiry on terms the original already sets.",
     ("doctype.variation",)),
    ("doctype.guaranty",
     "A third party's undertaking to answer for a party's obligations under an agreement.",
     ()),
```

Change the status expression in the `Concept(...)` comprehension from `"proposed" if code == "doctype.policy_document" else "active"` to `"proposed" if code in _PROPOSED_TYPES else "active"`, and define above `CONCEPTS`:

```python
_PROPOSED_TYPES = frozenset({
    "doctype.policy_document", "doctype.dpa", "doctype.side_letter",
    "doctype.renewal", "doctype.guaranty",
})
```

Append to `DOCUMENT_TYPES` (same shape as `doctype.policy_document`, after it):

```python
        DocumentType("doctype.dpa", "role.attachment", None, None,
                     ("dpa", "data processing agreement"), (), (), None, status="proposed"),
        DocumentType("doctype.side_letter", "role.variation", None, None,
                     ("side letter",), (), (), None, status="proposed"),
        DocumentType("doctype.renewal", "role.variation", None, None,
                     ("renewal agreement",), (), (), None, status="proposed"),
        DocumentType("doctype.guaranty", "role.supporting", None, None,
                     ("guaranty", "guarantee", "parent company guarantee"), (), (), None,
                     status="proposed"),
```

Migration `deploy/sql/2026-10-08_contract_link_vocabulary.sql` (concept rows first: `bp_document_type` has foreign keys to `bp_concept`):

```sql
-- Four document types the vocabulary should know about before anything can resolve to them.
-- status='proposed': counted and visible, never resolved or routed (only 'active' rows
-- resolve). Nick confirms each. No pipeline, like doctype.policy_document.
-- Additive (ON CONFLICT DO NOTHING), idempotent, reversible.
BEGIN;

INSERT INTO proc.bp_concept
    (concept_code, domain, definition, not_to_be_confused_with, status, source, rejection_reason)
VALUES
    ('doctype.dpa', 'DOCUMENT_TYPE',
     'Governs how personal data is processed for a parent agreement; forms part of it.',
     ARRAY['doctype.addendum']::text[], 'proposed', 'seed', NULL),
    ('doctype.side_letter', 'DOCUMENT_TYPE',
     'A separate letter that modifies or waives a term of an agreement it names.',
     ARRAY['doctype.variation']::text[], 'proposed', 'seed', NULL),
    ('doctype.renewal', 'DOCUMENT_TYPE',
     'Extends an agreement past its expiry on terms the original already sets.',
     ARRAY['doctype.variation']::text[], 'proposed', 'seed', NULL),
    ('doctype.guaranty', 'DOCUMENT_TYPE',
     'A third party''s undertaking to answer for a party''s obligations under an agreement.',
     ARRAY[]::text[], 'proposed', 'seed', NULL)
ON CONFLICT (concept_code) DO NOTHING;

INSERT INTO proc.bp_document_type
    (concept_code, role, default_parent_type, execution_mode, aliases, identifiers,
     structural_signals, pipeline_doc_type, status, source)
VALUES
    ('doctype.dpa', 'role.attachment', NULL, NULL,
     ARRAY['dpa','data processing agreement']::text[], '[]'::jsonb, '{}'::text[], NULL, 'proposed', 'seed'),
    ('doctype.side_letter', 'role.variation', NULL, NULL,
     ARRAY['side letter']::text[], '[]'::jsonb, '{}'::text[], NULL, 'proposed', 'seed'),
    ('doctype.renewal', 'role.variation', NULL, NULL,
     ARRAY['renewal agreement']::text[], '[]'::jsonb, '{}'::text[], NULL, 'proposed', 'seed'),
    ('doctype.guaranty', 'role.supporting', NULL, NULL,
     ARRAY['guaranty','guarantee','parent company guarantee']::text[], '[]'::jsonb, '{}'::text[],
     NULL, 'proposed', 'seed')
ON CONFLICT (concept_code) DO NOTHING;

COMMIT;
```

Rollback `..._rollback.sql`:

```sql
BEGIN;
DELETE FROM proc.bp_document_type WHERE concept_code IN
    ('doctype.dpa','doctype.side_letter','doctype.renewal','doctype.guaranty');
DELETE FROM proc.bp_concept WHERE concept_code IN
    ('doctype.dpa','doctype.side_letter','doctype.renewal','doctype.guaranty');
COMMIT;
```

- [ ] **Step 4: Apply to `bp_testdb` and run the concept suite**

Apply: `set -a; . ./.env 2>/dev/null; set +a; PYTHONPATH=. ./venv/bin/python -c "from src.services.db import get_conn; sql=open('deploy/sql/2026-10-08_contract_link_vocabulary.sql').read(); c=get_conn().__enter__(); c.cursor().execute(sql)"`
(then run the same snippet a second time: it must be a no-op). Then run:
`RUN tests/services/concepts -q`
Expected: all pass, including `test_document_type_rows_equal_the_seed_column_for_column`, `test_no_alias_is_claimed_by_two_concepts`, and `test_every_document_type_concept_has_attributes`. If an alias is claimed twice, drop that alias from BOTH `seed.py` and the SQL (keep the two in lockstep) and re-apply.

- [ ] **Step 5: Break the guard** — set `status="active"` on `doctype.dpa` in `seed.py` only; expected: `test_the_four_types_are_seeded_proposed...` and the seed-vs-table drift test FAIL. Restore.

- [ ] **Step 6: Commit** `seed.py`, the two SQL files and the test: `feat(vocabulary): dpa, side letter, renewal and guaranty are known, proposed, and resolve to nothing`.

---

### Task 9: Verify against the real database and record it

**Files:**
- Modify: `specs/2026-10-02-contract-structures-verification.md` (append a section)

- [ ] **Step 1: Run every affected suite**

Run each and record counts:
`RUN tests/services/graph_resolution tests/services/test_contract_links.py tests/services/test_contract_link_wiring.py tests/services/test_contract_link_matrix.py tests/services/concepts tests/services/extraction/test_type_resolver.py -q`
Expected: no failures. Also run the engine's own vectors: `RUN tests/test_linking_engine.py -q` (or the file `grep -rl "score_link" tests | head -3` names). Expected: unchanged, identical pass count to before this branch.

- [ ] **Step 2: Re-run the 2026-10-08 matrix as a script on `bp_testdb` and compare**

Use the scenario script from the design session (`scratchpad/cl_matrix.py`) if it is still on disk; otherwise Task 7's tests are the same matrix. Record, for the Variation, Addendum and CCN rows with a declared reference and a generic title, the new score (design measured 65.9; with a buyer 78.4).

- [ ] **Step 3: Measure the corpus effect**

Run `propose_parent_links()` over the whole `bp_testdb` parentless set before and after (`git stash` is forbidden: instead run the "before" by importing `contract_hierarchy.SIGNALS` into a throwaway script that scores the same children with the five-signal profile). Record: proposals before / after, contested before / after, and the highest `decision` band any pair reached. This is the evidence for the spec's section 8 third risk.

- [ ] **Step 4: Append a section "15. Per-role signals, 2026-10-08" to the verification record** with: what was built (commit hashes), the measured scores above, the counts from Step 1, and a plain list of what is still unproven (wording signals; no corpus document has ever produced a proposal; `bp_sqldb` needs the migration; the running server needs a restart to load the new profiles).

- [ ] **Step 5: Commit** the verification file: `docs(contracts): verification record for the per-role link signals`.

---

## Self-review

- **Spec coverage:** section 4 (profiles, signal table, applicability) -> Tasks 1-5; role selection and attachment parent types -> Task 6; section 5 (vocabulary) -> Task 8; section 6 (link type) -> Task 6; section 7 (testing, matrix, defects) -> Task 7 and each task's guards; section 8 risk 3 -> Task 9 step 3. Out-of-scope items (wording signals, amendment sequence, calibration, renewal, start-order) have no task, by design.
- **Placeholders:** none. The loose `or True` assertions drafted for Tasks 2, 3 and 8 were replaced with exact ones in the plan itself.
- **Type consistency:** `score_pair(base_name, params, base_specs, optional, src, tgt)` is used identically in Tasks 3-5; `HIERARCHY_OPTIONAL / AMENDMENT_OPTIONAL / ATTACHMENT_OPTIONAL` defined in Task 2 and consumed under those names; `_profile_module`, `_link_type`, `_fetch` defined in Task 6 and used in its tests; `contract_hierarchy._PARAMS` defined in Task 3 and consumed in Tasks 4-5.

Execution handoff: subagent-driven is recommended: nine tasks, each with its own test cycle, but Tasks 3-6 depend on each other's exact names. Do NOT use Haiku implementers (shared index).
