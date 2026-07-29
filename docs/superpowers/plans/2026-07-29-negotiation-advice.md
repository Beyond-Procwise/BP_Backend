# Negotiation Advice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a buyer ranked, evidence-grounded negotiation advice per deal — which lever, the specific play, why it applies here, its trade-off — with a conversation to challenge the framing and add facts the data lacks.

**Architecture:** The playbook intelligence already exists in `NegotiationAgent._resolve_playbook_context` but returns zero plays because it is gated on `supplier_type` and `negotiation_style`, which no code computes. We extract the ranking into `src/services/negotiation_advice/`, add a Kraljic classifier and signal builders to unlock it, attach a precondition to every play so advice is grounded in the deal's own evidence, and put a thin `negotiation_advisor` agent on top for the conversation. Deterministic spine; AgentNick only phrases the "why".

**Tech Stack:** Python 3.12, PostgreSQL (`proc` schema), FastAPI, pytest, Ollama (`AgentNick:unified` via `src/services/ollama_client.ollama_generate`).

**Spec:** `docs/superpowers/specs/2026-07-29-negotiation-advice-design.md`

## Global Constraints

- **Branch:** `Development`. Never push to `main`.
- **No Claude attribution in commit messages.** No `Co-Authored-By` trailers.
- **Table naming:** new tables `proc.bp_*`, indexes `ix_bp_<table>_<col>`. Migrations additive and idempotent (`CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`).
- **`deal_id` is owned by a database stored procedure.** Read it; never assign it in application code.
- **Model:** `AgentNick:unified` only. Never route to qwen or any non-AgentNick model. Reasoning model — always pass `think=False`, or `response` comes back empty.
- **No fabrication.** A value absent from the data is omitted, never inferred. Nothing is marked usable without its evidence.
- **Behaviour preservation:** `NegotiationAgent`'s existing behaviour must not change. Its suite is the gate.
- **Run tests with `.env` loaded:** `set -a && . ./.env; set +a` first — live tests read `os.environ` directly.
- **Test command prefix:** `.venv/bin/python -m pytest ... -q -p no:cacheprovider`
- **Two files are already named `test_summary_agent.py`** (in `tests/` and `tests/services/`) and collide under pytest. Do not add another duplicate basename.
- **Vocabularies are fixed by the playbook.** `supplier_type` ∈ {`Transactional`, `Leverage`, `Strategic`, `Bottleneck`}. `negotiation_style` ∈ {`Competitive`, `Collaborative`, `Principled`, `Accommodating`, `Compromising`}. Lever categories ∈ {`Commercial`, `Operational`, `Risk`}.
- **`bp_supplier.supplier_type` is NOT Kraljic** — it holds Consulting/Retailer/Manufacturer/Distributor/Wholesaler/Service Provider. Never use it as the quadrant source.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/services/negotiation_advice/__init__.py` | Public surface: `build_advice`, `apply_turn` |
| `src/services/negotiation_advice/signals.py` | Deal facts → spend, alternative-supplier count, risk, variance; the two scorer dicts |
| `src/services/negotiation_advice/classification.py` | Kraljic quadrant + style, each with reasons and confidence |
| `src/services/negotiation_advice/ranking.py` | Playbook load + play scoring (extracted from `NegotiationAgent`) |
| `src/services/negotiation_advice/grounding.py` | Precondition per play → `ready` / `groundwork` / `not_applicable` |
| `src/services/negotiation_advice/store.py` | Read/write `bp_negotiation_advice` and `bp_negotiation_advice_fact` |
| `src/services/negotiation_advice/advisor.py` | Compose an advice payload; apply one conversational turn |
| `src/agents/negotiation_advisor_agent.py` | Thin `BaseAgent` wrapper for the Agent Workspace |
| `deploy/sql/2026-07-29_bp_negotiation_advice.sql` | Both tables + policy seed |
| `src/api/routers/negotiate.py` | 3 new endpoints |
| `src/services/negotiate_dashboard.py` | Replace 2 constant fields with ranked plays |
| `src/agents/negotiation_agent.py` | Delegate scoring to `ranking.py`; behaviour unchanged |
| `agent_definitions.json` | Register `negotiation_advisor` |

---

### Task 1: Signals from deal facts

**Files:**
- Create: `src/services/negotiation_advice/__init__.py` (empty for now)
- Create: `src/services/negotiation_advice/signals.py`
- Test: `tests/services/negotiation_advice/test_signals.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `gather_signals(cur, deal_id: str) -> dict` returning keys `deal_id`, `supplier_id`, `supplier_name`, `currency`, `deal_value`, `quote_supplier_count`, `alternative_supplier_count`, `risk_score`, `is_preferred`, `price_variance_pct`, `invoice_total`, `po_total`. Absent values are `None`, never `0`.
  - `supplier_performance_dict(signals: dict) -> dict` — keys the existing scorer reads.
  - `market_context_dict(signals: dict) -> dict` — keys the existing scorer reads.

**Background the implementer needs:**

`proc.bp_deal_overview` has these columns (no `category`): `deal_id, deal_name, supplier_id, supplier_name, buyer_id, deal_date, first_activity_date, last_activity_date, quote_count, po_count, invoice_count, quote_total, po_total, invoice_total, currency, converted_total_usd, three_way_match, price_variance_pct, cycle_days_quote_to_po, cycle_days_po_to_invoice, has_quote_anchor, orphaned`.

There is no category dimension anywhere at volume (`proc.bp_category` has 0 rows). "Alternative suppliers" therefore means: distinct `bp_quote_trgt.supplier_id` that have quoted any of the same `item_description` values as this deal's quote lines. Measured live: 5,019 distinct item descriptions, suppliers per item min 1 / median 23 / max 41.

`supplier_performance_dict` must emit keys the existing `_score_supplier_performance` reads — it looks for `on_time_delivery`, `on_time`, `delivery_score`, `otif`. `market_context_dict` must emit `supply_risk` (string; the scorer tests for `"high"`/`"elevated"`/`"tight"`) and `demand_trend` (string; tests for `"rising"`/`"high"`). Omit a key entirely when it cannot be computed — the scorers return `(0.0, [])` on an empty dict, which is the correct "no signal" behaviour. Never default a missing signal to a value.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/negotiation_advice/test_signals.py
import pytest
from src.services.negotiation_advice import signals as sg


class _Cur:
    """Matches a SQL substring -> (columns, rows)."""

    def __init__(self, data):
        self._data = data
        self.description = []
        self._rows = []

    def execute(self, sql, params=()):
        for needle, (cols, rows) in self._data.items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._rows = list(rows)
                return
        self.description = []
        self._rows = []

    def fetchall(self):
        return self._rows


_DEAL_COLS = ["deal_id", "supplier_id", "supplier_name", "currency",
              "quote_count", "quote_total", "po_total", "invoice_total",
              "price_variance_pct"]


def _cur(alt_count=23, risk=0.2, preferred=True, variance=8.4):
    return _Cur({
        "from proc.bp_deal_overview": (
            _DEAL_COLS,
            [("D-1", "SUP-1", "Orbis Ltd", "GBP", 1, 100000.0, 98000.0,
              105000.0, variance)],
        ),
        "bp_quote_line_items_trgt": (["n"], [(alt_count,)]),
        "from proc.bp_supplier": (["risk_score", "is_preferred_supplier"],
                                  [(risk, preferred)]),
    })


def test_gather_signals_reads_deal_and_alternatives():
    s = sg.gather_signals(_cur(), "D-1")
    assert s["supplier_id"] == "SUP-1"
    assert s["alternative_supplier_count"] == 23
    assert s["deal_value"] == 105000.0        # invoice preferred over po/quote
    assert s["risk_score"] == 0.2
    assert s["is_preferred"] is True
    assert s["price_variance_pct"] == 8.4


def test_absent_values_are_none_not_zero():
    cur = _Cur({"from proc.bp_deal_overview": (_DEAL_COLS,
               [("D-2", None, None, None, 0, None, None, None, None)])})
    s = sg.gather_signals(cur, "D-2")
    assert s["deal_value"] is None
    assert s["risk_score"] is None
    assert s["alternative_supplier_count"] is None


def test_unknown_deal_returns_none():
    assert sg.gather_signals(_Cur({}), "NOPE") is None


def test_market_context_omits_uncomputable_keys():
    market = sg.market_context_dict({"alternative_supplier_count": None,
                                     "risk_score": None})
    assert "supply_risk" not in market


def test_market_context_flags_high_supply_risk_when_few_alternatives():
    market = sg.market_context_dict({"alternative_supplier_count": 1,
                                     "risk_score": 0.8})
    # the existing scorer tests for these exact strings
    assert market["supply_risk"] in {"high", "elevated", "tight"}


def test_supplier_performance_uses_a_key_the_scorer_reads():
    perf = sg.supplier_performance_dict({"on_time_ratio": 0.72})
    assert set(perf) & {"on_time_delivery", "on_time", "delivery_score", "otif"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_signals.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.negotiation_advice'`

- [ ] **Step 3: Write minimal implementation**

Create `src/services/negotiation_advice/__init__.py` as an empty file, then:

```python
# src/services/negotiation_advice/signals.py
"""Deal facts and the two scorer dicts, computed from grounded data only.

There is no category dimension in this database (proc.bp_category holds 0 rows
and no _trgt table has a category column), so supply-market competitiveness is
measured by item-level supplier overlap instead: how many distinct suppliers have
quoted the same item descriptions as this deal.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

_ALT_SQL = """
select count(distinct q2.supplier_id) as n
from proc.bp_quote_line_items_trgt mine
join proc.bp_quote_line_items_trgt theirs
  on lower(trim(theirs.item_description)) = lower(trim(mine.item_description))
join proc.bp_quote_trgt q2 on q2.quote_id = theirs.quote_id
where mine.deal_id = %s
  and mine.item_description is not null
  and length(trim(mine.item_description)) > 3
  and q2.supplier_id is not null
"""


def _rows(cur, sql: str, params: tuple = ()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _f(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def gather_signals(cur, deal_id: str) -> Optional[dict]:
    """Grounded facts for one deal, or None when the deal is unknown."""
    deal_rows = _rows(
        cur,
        "select deal_id, supplier_id, supplier_name, currency, quote_count, "
        "quote_total, po_total, invoice_total, price_variance_pct "
        "from proc.bp_deal_overview where deal_id=%s",
        (deal_id,),
    )
    if not deal_rows:
        return None
    d = deal_rows[0]

    alt: Optional[int] = None
    try:
        alt_rows = _rows(cur, _ALT_SQL, (deal_id,))
        if alt_rows:
            raw = alt_rows[0].get("n")
            alt = int(raw) if raw not in (None, 0) else None
    except Exception:
        log.debug("alternative-supplier count failed for %s", deal_id,
                  exc_info=True)

    risk: Optional[float] = None
    preferred: Optional[bool] = None
    if d.get("supplier_id"):
        try:
            sup = _rows(cur, "select risk_score, is_preferred_supplier "
                             "from proc.bp_supplier where supplier_id=%s",
                        (d["supplier_id"],))
            if sup:
                risk = _f(sup[0].get("risk_score"))
                raw_pref = sup[0].get("is_preferred_supplier")
                preferred = bool(raw_pref) if raw_pref is not None else None
        except Exception:
            log.debug("supplier lookup failed for %s", d.get("supplier_id"),
                      exc_info=True)

    invoice_total = _f(d.get("invoice_total"))
    po_total = _f(d.get("po_total"))
    quote_total = _f(d.get("quote_total"))
    return {
        "deal_id": d.get("deal_id"),
        "supplier_id": d.get("supplier_id"),
        "supplier_name": d.get("supplier_name"),
        "currency": d.get("currency"),
        "deal_value": invoice_total or po_total or quote_total,
        "invoice_total": invoice_total,
        "po_total": po_total,
        "quote_supplier_count": int(d.get("quote_count") or 0) or None,
        "alternative_supplier_count": alt,
        "risk_score": risk,
        "is_preferred": preferred,
        "price_variance_pct": _f(d.get("price_variance_pct")),
    }


def supplier_performance_dict(signals: dict) -> dict:
    """Only keys _score_supplier_performance actually reads, and only when known."""
    out: dict = {}
    on_time = signals.get("on_time_ratio")
    if on_time is not None:
        out["on_time_delivery"] = on_time
    return out


def market_context_dict(signals: dict) -> dict:
    """Only keys _score_market_context reads, and only when known.

    An uncomputable signal is omitted: the scorer returns (0.0, []) on an empty
    dict, which is the honest "no signal" outcome. Defaulting would invent a nudge.
    """
    out: dict = {}
    alt = signals.get("alternative_supplier_count")
    risk = signals.get("risk_score")
    if alt is not None and alt <= 2:
        out["supply_risk"] = "high"
    elif risk is not None and risk >= 0.6:
        out["supply_risk"] = "elevated"
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_signals.py -q -p no:cacheprovider`
Expected: PASS (6 passed)

- [ ] **Step 5: Verify the alternatives SQL against the live database**

```bash
set -a && . ./.env; set +a
PYTHONPATH=src:. .venv/bin/python - <<'EOF'
from src.services.db import get_conn
from src.services.negotiation_advice.signals import gather_signals
with get_conn() as c:
    cur = c.cursor()
    cur.execute("select deal_id from proc.bp_deal_overview "
                "where invoice_total is not null limit 3")
    for (did,) in cur.fetchall():
        print(gather_signals(cur, did))
EOF
```
Expected: three dicts with a non-null `deal_value` and a plausible
`alternative_supplier_count` (roughly 1–41, median around 23). If every count is
`None`, the join found no item overlap — check `deal_id` is populated on
`bp_quote_line_items_trgt` before proceeding.

- [ ] **Step 6: Commit**

```bash
git add src/services/negotiation_advice/__init__.py \
        src/services/negotiation_advice/signals.py \
        tests/services/negotiation_advice/test_signals.py
git commit -m "feat(negotiation-advice): grounded deal signals and scorer dicts"
```

---

### Task 2: Kraljic classification with reasons

**Files:**
- Create: `src/services/negotiation_advice/classification.py`
- Test: `tests/services/negotiation_advice/test_classification.py`

**Interfaces:**
- Consumes: `signals.gather_signals` output shape from Task 1.
- Produces:
  - `THRESHOLD_POLICY_SLUG = "negotiation_advice_thresholds"`
  - `classify(signals: dict, thresholds: dict | None = None) -> dict` returning `{"quadrant": str|None, "quadrant_reasons": list[str], "quadrant_confidence": float, "style": str|None, "style_reasons": list[str], "indeterminate": bool}`
  - `default_thresholds() -> dict` → `{"high_spend": 98175.0, "many_alternatives": 23}`
  - `_style_for(quadrant: str, signals: dict) -> tuple[str, list[str]]` — Task 6 imports this to recompute the style when a buyer overrides the quadrant, so keep it importable.
  - `QUADRANTS`, `STYLES` tuples for validation.

**Background the implementer needs:**

The quadrant vocabulary must match the playbook file's top-level keys exactly:
`Transactional`, `Leverage`, `Strategic`, `Bottleneck`. Style must match:
`Competitive`, `Collaborative`, `Principled`, `Accommodating`, `Compromising`.

Kraljic mapping: high spend + many alternatives → `Leverage`; high spend + few →
`Strategic`; low spend + many → `Transactional`; low spend + few → `Bottleneck`.

Thresholds come from `proc.bp_policy` at runtime (Task 6 wires that). This module
takes them as a parameter and falls back to `default_thresholds()`, so it stays
pure and unit-testable. The seed values are the live p90 deal value (£98,175;
median is £4,180) and the median suppliers-per-item (23). They are
testdata-derived on purpose — that is why they are data, not constants.

**When spend or alternatives is `None`, do not guess.** Return
`quadrant=None, indeterminate=True` so the caller asks the buyer. A wrong
quadrant misdirects every downstream play.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/negotiation_advice/test_classification.py
from src.services.negotiation_advice import classification as cl

HIGH = 200_000.0
LOW = 1_000.0


def _sig(value, alt, risk=0.2, preferred=False, variance=None):
    return {"deal_value": value, "alternative_supplier_count": alt,
            "risk_score": risk, "is_preferred": preferred,
            "price_variance_pct": variance}


def test_high_spend_many_alternatives_is_leverage():
    out = cl.classify(_sig(HIGH, 30))
    assert out["quadrant"] == "Leverage"
    assert out["indeterminate"] is False


def test_high_spend_few_alternatives_is_strategic():
    assert cl.classify(_sig(HIGH, 2))["quadrant"] == "Strategic"


def test_low_spend_many_alternatives_is_transactional():
    assert cl.classify(_sig(LOW, 30))["quadrant"] == "Transactional"


def test_low_spend_few_alternatives_is_bottleneck():
    assert cl.classify(_sig(LOW, 2))["quadrant"] == "Bottleneck"


def test_quadrant_is_one_of_the_playbook_keys():
    valid = {"Transactional", "Leverage", "Strategic", "Bottleneck"}
    for value, alt in [(HIGH, 30), (HIGH, 2), (LOW, 30), (LOW, 2)]:
        assert cl.classify(_sig(value, alt))["quadrant"] in valid


def test_missing_spend_is_indeterminate_not_guessed():
    out = cl.classify(_sig(None, 30))
    assert out["quadrant"] is None
    assert out["indeterminate"] is True


def test_missing_alternatives_is_indeterminate_not_guessed():
    out = cl.classify(_sig(HIGH, None))
    assert out["quadrant"] is None
    assert out["indeterminate"] is True


def test_reasons_cite_the_actual_numbers():
    reasons = " ".join(cl.classify(_sig(HIGH, 30))["quadrant_reasons"])
    assert "200,000" in reasons or "200000" in reasons
    assert "30" in reasons


def test_thresholds_are_overridable():
    # with a very high spend bar, 200k is now "low"
    out = cl.classify(_sig(HIGH, 30), thresholds={"high_spend": 10_000_000.0,
                                                  "many_alternatives": 23})
    assert out["quadrant"] == "Transactional"


def test_style_follows_quadrant_and_evidence():
    assert cl.classify(_sig(HIGH, 30, variance=8.4))["style"] == "Competitive"
    assert cl.classify(_sig(HIGH, 2, preferred=True))["style"] == "Collaborative"
    assert cl.classify(_sig(LOW, 2))["style"] == "Principled"
    assert cl.classify(_sig(LOW, 30))["style"] == "Competitive"


def test_style_is_one_of_the_playbook_styles():
    valid = {"Competitive", "Collaborative", "Principled", "Accommodating",
             "Compromising"}
    for value, alt in [(HIGH, 30), (HIGH, 2), (LOW, 30), (LOW, 2)]:
        assert cl.classify(_sig(value, alt))["style"] in valid


def test_confidence_is_lower_near_a_threshold():
    near = cl.classify(_sig(98_200.0, 23))["quadrant_confidence"]
    clear = cl.classify(_sig(1_000_000.0, 40))["quadrant_confidence"]
    assert near < clear
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_classification.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.negotiation_advice.classification'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/negotiation_advice/classification.py
"""Kraljic quadrant and negotiation style — suggestions, never impositions.

The quadrant and style vocabularies are fixed by the playbook file's keys; a
value outside them yields zero plays downstream, so they are asserted here.

Note bp_supplier.supplier_type is NOT a Kraljic axis — it holds business types
(Consulting, Retailer, Manufacturer, ...) and is deliberately unused.
"""
from __future__ import annotations

from typing import Optional

THRESHOLD_POLICY_SLUG = "negotiation_advice_thresholds"

QUADRANTS = ("Transactional", "Leverage", "Strategic", "Bottleneck")
STYLES = ("Competitive", "Collaborative", "Principled", "Accommodating",
          "Compromising")


def default_thresholds() -> dict:
    """Seeded from the live distribution: deal-value p90, suppliers-per-item
    median. Testdata-derived, hence governed data rather than constants."""
    return {"high_spend": 98175.0, "many_alternatives": 23}


def _confidence(value: float, bar: float) -> float:
    """1.0 far from the bar, approaching 0.5 at it."""
    if bar <= 0:
        return 0.5
    ratio = value / bar
    distance = abs(ratio - 1.0)
    return round(min(1.0, 0.5 + distance), 3)


def classify(signals: dict, thresholds: Optional[dict] = None) -> dict:
    bars = dict(default_thresholds())
    if thresholds:
        bars.update({k: v for k, v in thresholds.items() if v is not None})

    spend = signals.get("deal_value")
    alternatives = signals.get("alternative_supplier_count")

    if spend is None or alternatives is None:
        missing = []
        if spend is None:
            missing.append("deal value could not be determined")
        if alternatives is None:
            missing.append("no comparable suppliers found for this deal's items")
        return {
            "quadrant": None, "quadrant_reasons": missing,
            "quadrant_confidence": 0.0, "style": None, "style_reasons": [],
            "indeterminate": True,
        }

    high_spend = float(spend) >= float(bars["high_spend"])
    many_alts = int(alternatives) >= int(bars["many_alternatives"])
    quadrant = {
        (True, True): "Leverage",
        (True, False): "Strategic",
        (False, True): "Transactional",
        (False, False): "Bottleneck",
    }[(high_spend, many_alts)]

    reasons = [
        f"Deal value {float(spend):,.0f} is "
        f"{'at or above' if high_spend else 'below'} the "
        f"{float(bars['high_spend']):,.0f} high-spend bar",
        f"{int(alternatives)} supplier(s) quote comparable items — "
        f"{'a contested' if many_alts else 'a thin'} supply market",
    ]
    risk = signals.get("risk_score")
    if risk is not None:
        reasons.append(f"Supplier risk score {risk}")

    confidence = round(
        min(_confidence(float(spend), float(bars["high_spend"])),
            _confidence(float(alternatives), float(bars["many_alternatives"]))),
        3,
    )

    style, style_reasons = _style_for(quadrant, signals)
    return {
        "quadrant": quadrant, "quadrant_reasons": reasons,
        "quadrant_confidence": confidence, "style": style,
        "style_reasons": style_reasons, "indeterminate": False,
    }


def _style_for(quadrant: str, signals: dict) -> tuple[str, list[str]]:
    variance = signals.get("price_variance_pct")
    preferred = signals.get("is_preferred")
    if quadrant == "Strategic" and preferred:
        return "Collaborative", ["Preferred supplier on a strategic spend — "
                                "protect the relationship while negotiating"]
    if quadrant == "Bottleneck":
        return "Principled", ["Thin supply market — continuity is the exposure, "
                              "so argue from objective criteria, not pressure"]
    if quadrant == "Leverage":
        why = ["Contested market on a material spend — competitive tension is "
               "available"]
        if variance is not None:
            why.append(f"Price variance {variance}% across the document chain")
        return "Competitive", why
    if quadrant == "Strategic":
        return "Collaborative", ["Material spend with few alternatives — build "
                                 "value rather than squeeze price"]
    return "Competitive", ["Low-value, contested spend — standardise and "
                           "compete it"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_classification.py -q -p no:cacheprovider`
Expected: PASS (11 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/negotiation_advice/classification.py \
        tests/services/negotiation_advice/test_classification.py
git commit -m "feat(negotiation-advice): Kraljic quadrant and style with reasons"
```

---

### Task 3: Extract play ranking out of NegotiationAgent

**Files:**
- Create: `src/services/negotiation_advice/ranking.py`
- Modify: `src/agents/negotiation_agent.py` (`_resolve_playbook_context`, lines 7328-7428)
- Test: `tests/services/negotiation_advice/test_ranking.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `load_playbook(path=None) -> dict`
  - `rank_plays(supplier_type: str, negotiation_style: str, *, lever_priorities: list[str] | None = None, policy_guidance: dict | None = None, supplier_performance: dict | None = None, market_context: dict | None = None, playbook: dict | None = None, limit: int = 10) -> dict` returning `{"plays": [...], "descriptor": str|None, "examples": list, "style": str|None, "supplier_type": str|None, "lever_priorities": list[str]}` — the same shape `_resolve_playbook_context` returns today.
  - Each play dict keeps today's keys exactly: `supplier_type`, `style`, `lever`, `play`, `score`, `policy_alignment`, `performance_signals`, `market_signals`, `rationale`, `trade_offs`.

**Background the implementer needs:**

`negotiation_agent.py` is ~540KB. The scoring logic to move lives in
`_resolve_playbook_context` (line 7328) and the helpers it calls:
`_load_playbook` (8140), `_normalise_supplier_type`, `_normalise_negotiation_style`,
`_normalise_lever_category` (8207), `_score_policy_alignment`,
`_score_supplier_performance` (8294), `_score_market_context` (8345),
`_compose_play_rationale` (8385), and the module constants `PLAYBOOK_PATH` (483)
and `TRADE_OFF_HINTS` (118).

**Move the pure scoring into `ranking.py` as module functions; leave the
`AgentContext`-reading parts on the agent.** `_resolve_playbook_context` keeps
reading `supplier_type` / `negotiation_style` / `lever_priorities` /
`policy_guidance` / `supplier_performance` / `market_context` off
`context.input_data` exactly as it does now, then delegates to `rank_plays`.
Its return value and behaviour must be unchanged — including returning
`{"plays": [], "lever_priorities": []}` when `supplier_type` is missing or not in
the playbook. That gate stays on the agent path; the new advisor path supplies a
real classification instead of removing the gate.

Scoring rules to preserve exactly: `base_score = 1.0 + (idx * 0.01)` where `idx`
is the play's position within its lever list; total = base + policy + performance
+ market; sort by `(-score, lever, play)`; take the first `limit`.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/negotiation_advice/test_ranking.py
import pytest
from src.services.negotiation_advice import ranking as rk

_PB = {
    "Leverage": {
        "descriptor": "Run competitive events",
        "examples": ["bundle spend"],
        "styles": {
            "Competitive": {
                "Commercial": ["Demand tiered volume discounts", "Benchmark it"],
                "Operational": ["Require priority fulfilment"],
                "Risk": ["Full refund for non-compliant goods"],
            }
        },
    }
}


def test_rank_plays_returns_plays_for_a_valid_pair():
    out = rk.rank_plays("Leverage", "Competitive", playbook=_PB)
    assert out["plays"]
    assert out["supplier_type"] == "Leverage"
    assert out["style"] == "Competitive"
    assert set(out["lever_priorities"]) == {"Commercial", "Operational", "Risk"}


def test_every_play_carries_the_established_keys():
    play = rk.rank_plays("Leverage", "Competitive", playbook=_PB)["plays"][0]
    for key in ("supplier_type", "style", "lever", "play", "score",
                "policy_alignment", "performance_signals", "market_signals",
                "rationale", "trade_offs"):
        assert key in play, key


def test_unknown_supplier_type_yields_no_plays():
    out = rk.rank_plays("Nonsense", "Competitive", playbook=_PB)
    assert out["plays"] == []


def test_unknown_style_yields_no_plays_but_keeps_descriptor():
    out = rk.rank_plays("Leverage", "Nonsense", playbook=_PB)
    assert out["plays"] == []
    assert out["descriptor"] == "Run competitive events"


def test_base_score_follows_position_within_the_lever():
    plays = rk.rank_plays("Leverage", "Competitive", playbook=_PB)["plays"]
    commercial = [p for p in plays if p["lever"] == "Commercial"]
    first = next(p for p in commercial if p["play"] == "Demand tiered volume discounts")
    second = next(p for p in commercial if p["play"] == "Benchmark it")
    assert first["score"] == pytest.approx(1.0)
    assert second["score"] == pytest.approx(1.01)


def test_signals_change_the_ranking():
    plain = rk.rank_plays("Leverage", "Competitive", playbook=_PB)["plays"]
    boosted = rk.rank_plays(
        "Leverage", "Competitive", playbook=_PB,
        supplier_performance={"on_time_delivery": 0.72},
        market_context={"supply_risk": "high"},
    )["plays"]
    assert boosted[0]["score"] > plain[0]["score"]


def test_lever_priorities_restrict_the_levers_considered():
    out = rk.rank_plays("Leverage", "Competitive", playbook=_PB,
                        lever_priorities=["Risk"])
    assert {p["lever"] for p in out["plays"]} == {"Risk"}


def test_limit_caps_the_list():
    out = rk.rank_plays("Leverage", "Competitive", playbook=_PB, limit=2)
    assert len(out["plays"]) == 2


def test_load_playbook_reads_the_shipped_file():
    pb = rk.load_playbook()
    assert set(pb) == {"Transactional", "Leverage", "Strategic", "Bottleneck"}
    for entry in pb.values():
        assert "Competitive" in entry["styles"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_ranking.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.negotiation_advice.ranking'`

- [ ] **Step 3: Record the agent's current behaviour as a baseline**

Before touching `negotiation_agent.py`, capture what it produces so Step 5 can
prove nothing changed:

```bash
set -a && . ./.env; set +a
PYTHONPATH=src:. .venv/bin/python - <<'EOF' > /tmp/playbook_before.json
import json
from agents.negotiation_agent import NegotiationAgent
from agents.base_agent import AgentContext
a = NegotiationAgent.__new__(NegotiationAgent); a._playbook_cache = None
out = []
for st in ("Transactional", "Leverage", "Strategic", "Bottleneck"):
    for sy in ("Competitive", "Collaborative", "Principled", "Accommodating",
               "Compromising"):
        ctx = AgentContext(workflow_id="w", agent_id="negotiation", user_id="u",
                           input_data={"supplier_type": st,
                                       "negotiation_style": sy})
        out.append(a._resolve_playbook_context(ctx, {}))
print(json.dumps(out, sort_keys=True, indent=2))
EOF
sha256sum /tmp/playbook_before.json
```

- [ ] **Step 4: Write `ranking.py` and delegate from the agent**

Move the scoring helpers into `ranking.py` as module-level functions, **copying
each body verbatim** from `negotiation_agent.py` and dropping the `self`
parameter. The bodies are not reproduced here on purpose: retyping ~200 lines
invites transcription error, and Step 5's byte-identical diff is the gate that
proves the copy was faithful. Exact source ranges:

| Symbol | Lines in `negotiation_agent.py` | Becomes |
|---|---|---|
| `TRADE_OFF_HINTS` | 118-124 | module constant, unchanged |
| `PLAYBOOK_PATH` | 483 | module constant, re-rooted (see below) |
| `_load_playbook` | 8140-8168 | `load_playbook(path=None)` |
| `_normalise_lever_category` | 8207-8219 | `normalise_lever_category(value)` |
| `_score_policy_alignment` | 8275-8292 | `_score_policy_alignment(lever, policy_guidance)` |
| `_score_supplier_performance` | 8294-8343 | `_score_supplier_performance(lever, performance)` |
| `_score_market_context` | 8345-8383 | `_score_market_context(lever, market)` |
| `_compose_play_rationale` | 8385-8407 | `_compose_play_rationale(descriptor, style, lever, policy_notes, performance_notes, market_notes)` |
| scoring half of `_resolve_playbook_context` | 7368-7428 | `rank_plays(...)` |

`PLAYBOOK_PATH` is currently `Path(__file__).parent.parent / "resources" /
"reference_data" / "negotiation_playbook.json"` resolved from
`src/agents/`, i.e. `src/resources/reference_data/negotiation_playbook.json`.
From `src/services/negotiation_advice/` that is `parents[2]`, not `parent.parent`
— get this wrong and `load_playbook` returns `{}` and every play disappears.

**Leave on the agent** (they read `AgentContext`, which `ranking.py` must not
know about): `_normalise_supplier_type` (8170-8186),
`_normalise_negotiation_style` (8188-8205), `_extract_policy_guidance`
(8221-8273), `_resolve_lever_priorities` (7430-7460), `_ensure_list`
(10851-10856).

Then:

```python
# src/services/negotiation_advice/ranking.py  (structure — bodies copied verbatim)
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

PLAYBOOK_PATH = (Path(__file__).resolve().parents[2]
                 / "resources" / "reference_data" / "negotiation_playbook.json")

TRADE_OFF_HINTS = {...}          # copied verbatim from negotiation_agent.py:118

_cache: Optional[dict] = None


def load_playbook(path: Optional[Path] = None) -> dict:
    """Body copied from NegotiationAgent._load_playbook, including the
    lever-key normalisation pass."""
    ...


def rank_plays(supplier_type, negotiation_style, *, lever_priorities=None,
               policy_guidance=None, supplier_performance=None,
               market_context=None, playbook=None, limit=10) -> dict:
    """Body copied from the scoring half of _resolve_playbook_context.

    Preserved exactly: base_score = 1.0 + idx * 0.01; total = base + policy +
    performance + market; sort by (-score, lever, play); slice to `limit`.
    """
    ...
```

Then replace the scoring half of `_resolve_playbook_context` in
`negotiation_agent.py` — keep every `context.input_data` read and the
missing-`supplier_type` gate, and delegate:

```python
        return rank_plays(
            supplier_type,
            negotiation_style,
            lever_priorities=lever_priorities,
            policy_guidance=policy_guidance,
            supplier_performance=supplier_performance,
            market_context=market_context,
        )
```

- [ ] **Step 5: Prove the agent's behaviour is unchanged**

```bash
set -a && . ./.env; set +a
# re-run the identical baseline script into a second file
PYTHONPATH=src:. .venv/bin/python - <<'EOF' > /tmp/playbook_after.json
# ... same script as Step 3 ...
EOF
diff /tmp/playbook_before.json /tmp/playbook_after.json \
  && echo "PLAYBOOK OUTPUT IDENTICAL" || echo "*** BEHAVIOUR CHANGED — STOP ***"
.venv/bin/python -m pytest tests/test_negotiation_agent.py -q -p no:cacheprovider
```
Expected: `PLAYBOOK OUTPUT IDENTICAL`, and `tests/test_negotiation_agent.py` with
no new failures versus before the change. If the diff is non-empty, the
extraction was not faithful — revert and redo it rather than adjusting the
baseline.

- [ ] **Step 6: Run the new tests**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/ -q -p no:cacheprovider`
Expected: PASS (all of Tasks 1-3)

- [ ] **Step 7: Commit**

```bash
git add src/services/negotiation_advice/ranking.py \
        src/agents/negotiation_agent.py \
        tests/services/negotiation_advice/test_ranking.py
git commit -m "refactor(negotiation): extract play ranking into negotiation_advice"
```

---

### Task 4: Preconditions — ready, groundwork, not applicable

**Files:**
- Create: `src/services/negotiation_advice/grounding.py`
- Test: `tests/services/negotiation_advice/test_grounding.py`

**Interfaces:**
- Consumes: `signals.gather_signals` (Task 1), play dicts from `ranking.rank_plays` (Task 3).
- Produces:
  - `PLAY_STATES = ("ready", "groundwork", "not_applicable")`
  - `assess(play: dict, signals: dict) -> dict` — returns the play with `state`, `evidence` (list of `{"label","value","source"}`) and, when `groundwork`, `unlocked_by` (str).
  - `apply_states(plays: list[dict], signals: dict) -> list[dict]` — assesses all, drops `not_applicable`, sorts `ready` before `groundwork` then by descending `score`.

**Background the implementer needs:**

A play's family is inferred from keywords in its text; the playbook has no family
field and adding one would mean editing shipped reference data. Match
case-insensitively:

| Family | Keywords in the play text | Precondition |
|---|---|---|
| `competitive_tension` | `competitor`, `e-auction`, `bidders`, `dual-sourc`, `competitive` | `quote_supplier_count >= 2` OR `alternative_supplier_count >= 2` |
| `volume` | `volume`, `tiered`, `bundle`, `rebate` | `deal_value` is not None |
| `price_challenge` | `benchmark`, `cost breakdown`, `should-cost`, `price` | `price_variance_pct` is not None |
| `overbilling` | `refund`, `credit`, `non-compliant` | `invoice_total` and `po_total` both known and `invoice_total > po_total` |
| (no match) | — | always `ready`; a play with no testable precondition is not blocked |

`not_applicable` is reserved for the case where the precondition cannot be met at
all: `competitive_tension` where `alternative_supplier_count` is known and equals
1 — there is no second supplier to find, so telling the buyer to go get one is
noise. Everything else that fails its precondition is `groundwork` with a concrete
`unlocked_by`.

`evidence` entries must carry a `source` string so provenance survives to the UI
(Task 5 adds `"stated by you"` for buyer facts). Use `"deal"` for
`bp_deal_overview` figures and `"item overlap"` for the alternatives count.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/negotiation_advice/test_grounding.py
from src.services.negotiation_advice import grounding as gr


def _play(text, lever="Commercial", score=1.0):
    return {"lever": lever, "play": text, "score": score,
            "trade_offs": "some trade-off", "rationale": "because"}


_RICH = {"deal_value": 100000.0, "quote_supplier_count": 1,
         "alternative_supplier_count": 20, "price_variance_pct": 8.4,
         "invoice_total": 105000.0, "po_total": 98000.0}
_BARE = {"deal_value": None, "quote_supplier_count": 1,
         "alternative_supplier_count": None, "price_variance_pct": None,
         "invoice_total": None, "po_total": None}


def test_competitive_play_is_groundwork_with_one_quote_but_alternatives():
    out = gr.assess(_play("Leverage competitor quotes to pressure pricing"), _RICH)
    assert out["state"] == "groundwork"
    assert "20" in out["unlocked_by"]


def test_competitive_play_not_applicable_when_truly_sole_source():
    sole = dict(_RICH, alternative_supplier_count=1)
    out = gr.assess(_play("Run e-auction with 3 bidders"), sole)
    assert out["state"] == "not_applicable"


def test_overbilling_play_is_ready_when_invoice_exceeds_po():
    out = gr.assess(_play("Full refund/credit for non-compliant goods",
                          lever="Risk"), _RICH)
    assert out["state"] == "ready"
    assert any("105,000" in str(e["value"]) or "105000" in str(e["value"])
               for e in out["evidence"])


def test_overbilling_play_is_groundwork_without_the_figures():
    out = gr.assess(_play("Full refund/credit for non-compliant goods",
                          lever="Risk"), _BARE)
    assert out["state"] == "groundwork"


def test_volume_play_needs_a_known_deal_value():
    assert gr.assess(_play("Demand tiered volume discounts"), _RICH)["state"] == "ready"
    assert gr.assess(_play("Demand tiered volume discounts"), _BARE)["state"] == "groundwork"


def test_price_play_needs_known_variance():
    assert gr.assess(_play("Request a detailed cost breakdown"), _RICH)["state"] == "ready"
    assert gr.assess(_play("Request a detailed cost breakdown"), _BARE)["state"] == "groundwork"


def test_play_with_no_testable_precondition_is_ready():
    out = gr.assess(_play("Agree a quarterly governance cadence",
                          lever="Operational"), _BARE)
    assert out["state"] == "ready"


def test_evidence_carries_a_source():
    out = gr.assess(_play("Demand tiered volume discounts"), _RICH)
    assert out["evidence"]
    assert all(e.get("source") for e in out["evidence"])


def test_no_ready_play_claims_evidence_the_deal_lacks():
    for play in [_play("Leverage competitor quotes"),
                 _play("Demand tiered volume discounts"),
                 _play("Request a detailed cost breakdown")]:
        out = gr.assess(play, _BARE)
        if out["state"] == "ready":
            assert out["evidence"] == []


def test_apply_states_drops_not_applicable_and_puts_ready_first():
    sole = dict(_RICH, alternative_supplier_count=1)
    plays = [_play("Run e-auction with 3 bidders", score=9.0),
             _play("Demand tiered volume discounts", score=1.0),
             _play("Full refund for non-compliant goods", "Risk", 2.0)]
    out = gr.apply_states(plays, sole)
    assert all(p["state"] != "not_applicable" for p in out)
    assert [p["state"] for p in out] == sorted(
        [p["state"] for p in out], key=lambda s: 0 if s == "ready" else 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_grounding.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.negotiation_advice.grounding'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/negotiation_advice/grounding.py
"""Test each play against the deal's own evidence.

A generic play is only advice when its precondition holds. "Leverage competitor
quotes" is strong when a second quote exists and damaging when it does not — a
buyer who bluffs one loses standing with the supplier. So nothing is marked
`ready` without the evidence behind it.
"""
from __future__ import annotations

from typing import Optional

PLAY_STATES = ("ready", "groundwork", "not_applicable")

_FAMILY_KEYWORDS = (
    ("competitive_tension", ("competitor", "e-auction", "bidders",
                             "dual-sourc", "competitive")),
    ("overbilling", ("refund", "credit", "non-compliant")),
    ("volume", ("volume", "tiered", "bundle", "rebate")),
    ("price_challenge", ("benchmark", "cost breakdown", "should-cost", "price")),
)


def _family(play_text: str) -> Optional[str]:
    low = (play_text or "").lower()
    for family, keywords in _FAMILY_KEYWORDS:
        if any(k in low for k in keywords):
            return family
    return None


def _ev(label: str, value, source: str) -> dict:
    return {"label": label, "value": value, "source": source}


def assess(play: dict, signals: dict) -> dict:
    family = _family(play.get("play", ""))
    out = dict(play)
    out["family"] = family
    out["evidence"] = []
    out.pop("unlocked_by", None)

    if family is None:
        out["state"] = "ready"
        return out

    if family == "competitive_tension":
        quotes = signals.get("quote_supplier_count") or 0
        alts = signals.get("alternative_supplier_count")
        if quotes >= 2:
            out["state"] = "ready"
            out["evidence"] = [_ev("Quote suppliers on this deal", quotes, "deal")]
        elif alts is not None and alts >= 2:
            out["state"] = "groundwork"
            out["unlocked_by"] = (
                f"Needs a competing quote — {alts} suppliers quote comparable items"
            )
        elif alts is not None:
            out["state"] = "not_applicable"
        else:
            out["state"] = "groundwork"
            out["unlocked_by"] = ("Needs a competing quote — no comparable "
                                  "suppliers identified yet")
        return out

    if family == "overbilling":
        inv, po = signals.get("invoice_total"), signals.get("po_total")
        if inv is not None and po is not None and inv > po:
            out["state"] = "ready"
            out["evidence"] = [_ev("Invoiced", f"{inv:,.2f}", "deal"),
                               _ev("PO value", f"{po:,.2f}", "deal")]
        else:
            out["state"] = "groundwork"
            out["unlocked_by"] = ("Needs matched invoice and PO totals showing "
                                  "an overcharge")
        return out

    if family == "volume":
        value = signals.get("deal_value")
        if value is not None:
            out["state"] = "ready"
            out["evidence"] = [_ev("Deal value", f"{value:,.2f}", "deal")]
        else:
            out["state"] = "groundwork"
            out["unlocked_by"] = "Needs a known deal value to size a tier"
        return out

    variance = signals.get("price_variance_pct")
    if variance is not None:
        out["state"] = "ready"
        out["evidence"] = [_ev("Price variance", f"{variance}%", "deal")]
    else:
        out["state"] = "groundwork"
        out["unlocked_by"] = "Needs a benchmark or price variance to argue from"
    return out


def apply_states(plays: list[dict], signals: dict) -> list[dict]:
    assessed = [assess(p, signals) for p in plays]
    usable = [p for p in assessed if p["state"] != "not_applicable"]
    usable.sort(key=lambda p: (0 if p["state"] == "ready" else 1,
                               -float(p.get("score") or 0.0)))
    return usable
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_grounding.py -q -p no:cacheprovider`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/negotiation_advice/grounding.py \
        tests/services/negotiation_advice/test_grounding.py
git commit -m "feat(negotiation-advice): precondition each play against deal evidence"
```

---

### Task 5: Tables and the advice store

**Files:**
- Create: `deploy/sql/2026-07-29_bp_negotiation_advice.sql`
- Create: `src/services/negotiation_advice/store.py`
- Test: `tests/services/negotiation_advice/test_store.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `save_advice(conn, *, deal_id, supplier_id, quadrant, quadrant_source, quadrant_confidence, style, style_source, signals, plays, created_by) -> dict` (returns the row including a generated `advice_id`)
  - `load_advice(conn, deal_id) -> dict | None` (most recent for the deal)
  - `state_fact(conn, *, advice_id, fact_key, fact_value, stated_by) -> None`
  - `active_facts(conn, advice_id) -> dict` mapping `fact_key -> fact_value`, excluding withdrawn
  - `withdraw_fact(conn, *, advice_id, fact_key) -> None`

**Background the implementer needs:**

Buyer-stated facts live in their own table so a stated value can never silently
become measured data, and so it can be withdrawn cleanly. `withdrawn_at` is set
rather than the row deleted, keeping the audit trail.

Follow the migration style in `deploy/sql/2026-06-17_bp_requirement.sql`: a
`BEGIN; ... COMMIT;` block, `CREATE TABLE IF NOT EXISTS`, `ix_bp_*` indexes.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/negotiation_advice/test_store.py
from src.services.negotiation_advice import store as st


class _Cur:
    def __init__(self, rec, rows=None):
        self._rec = rec
        self._rows = rows or []
        self.description = []

    def execute(self, sql, params=()):
        self._rec.append((sql, params))
        for needle, (cols, rows) in (self._rows or {}).items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._matched = list(rows)
                return
        self.description = []
        self._matched = []

    def fetchall(self):
        return getattr(self, "_matched", [])

    def fetchone(self):
        m = getattr(self, "_matched", [])
        return m[0] if m else None


class _Conn:
    def __init__(self, rows=None):
        self.rec = []
        self._cur = _Cur(self.rec, rows)
        self.committed = False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True


def test_save_advice_inserts_and_returns_an_id():
    conn = _Conn()
    out = st.save_advice(conn, deal_id="D-1", supplier_id="SUP-1",
                         quadrant="Leverage", quadrant_source="computed",
                         quadrant_confidence=0.8, style="Competitive",
                         style_source="computed", signals={"deal_value": 1},
                         plays=[{"lever": "Commercial"}], created_by="buyer")
    assert out["advice_id"]
    assert out["quadrant"] == "Leverage"
    sql = " ".join(s for s, _ in conn.rec)
    assert "INSERT INTO proc.bp_negotiation_advice" in sql
    assert conn.committed


def test_state_fact_records_provenance():
    conn = _Conn()
    st.state_fact(conn, advice_id="A-1", fact_key="alternative_supplier_count",
                  fact_value="2", stated_by="buyer")
    sql = " ".join(s for s, _ in conn.rec)
    assert "INSERT INTO proc.bp_negotiation_advice_fact" in sql
    assert conn.committed


def test_withdraw_sets_a_timestamp_rather_than_deleting():
    conn = _Conn()
    st.withdraw_fact(conn, advice_id="A-1", fact_key="alternative_supplier_count")
    sql = " ".join(s for s, _ in conn.rec)
    assert "withdrawn_at" in sql
    assert "DELETE" not in sql.upper()


def test_active_facts_excludes_withdrawn():
    conn = _Conn({"FROM proc.bp_negotiation_advice_fact":
                  (["fact_key", "fact_value"],
                   [("alternative_supplier_count", "2")])})
    facts = st.active_facts(conn, "A-1")
    assert facts == {"alternative_supplier_count": "2"}
    sql = " ".join(s for s, _ in conn.rec)
    assert "withdrawn_at IS NULL" in sql
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_store.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.negotiation_advice.store'`

- [ ] **Step 3: Write the migration**

```sql
-- deploy/sql/2026-07-29_bp_negotiation_advice.sql
-- Buyer-facing negotiation advice: one row per advice session, plus the
-- buyer-stated facts that shaped it. Stated facts are kept apart from measured
-- signals by construction so a stated value can never become data.
-- Additive + idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_negotiation_advice (
    advice_id            VARCHAR PRIMARY KEY,
    deal_id              VARCHAR NOT NULL,
    supplier_id          VARCHAR,
    quadrant             VARCHAR,
    quadrant_source      VARCHAR NOT NULL DEFAULT 'computed',
    quadrant_confidence  NUMERIC,
    style                VARCHAR,
    style_source         VARCHAR NOT NULL DEFAULT 'computed',
    signals              JSONB   DEFAULT '{}'::jsonb,
    plays                JSONB   DEFAULT '[]'::jsonb,
    created_by           VARCHAR,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT bp_negotiation_advice_quadrant_source_check
        CHECK (quadrant_source IN ('computed', 'buyer')),
    CONSTRAINT bp_negotiation_advice_style_source_check
        CHECK (style_source IN ('computed', 'buyer'))
);

CREATE INDEX IF NOT EXISTS ix_bp_negotiation_advice_deal_id
    ON proc.bp_negotiation_advice (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_negotiation_advice_created_at
    ON proc.bp_negotiation_advice (created_at DESC);

CREATE TABLE IF NOT EXISTS proc.bp_negotiation_advice_fact (
    advice_id     VARCHAR NOT NULL,
    fact_key      VARCHAR NOT NULL,
    fact_value    TEXT,
    stated_by     VARCHAR,
    stated_at     TIMESTAMPTZ DEFAULT NOW(),
    withdrawn_at  TIMESTAMPTZ,
    PRIMARY KEY (advice_id, fact_key)
);

CREATE INDEX IF NOT EXISTS ix_bp_negotiation_advice_fact_advice_id
    ON proc.bp_negotiation_advice_fact (advice_id);

-- Governed thresholds. Seeded from the live distribution: deal-value p90
-- (98175; median 4180) and median suppliers-per-item (23). Testdata-derived —
-- retune against real spend, which is why these are data and not constants.
INSERT INTO proc.bp_policy (policy_type, policy_name, policy_details, policy_status)
SELECT 'negotiation', 'negotiation_advice_thresholds',
       '{"high_spend": 98175.0, "many_alternatives": 23}'::jsonb, 1
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy
    WHERE policy_name = 'negotiation_advice_thresholds'
);

COMMIT;
```

Before running it, confirm the `bp_policy` column names match this INSERT:

```bash
set -a && . ./.env; set +a
PYTHONPATH=src:. .venv/bin/python -c "
from src.services.db import get_conn
with get_conn() as c:
    cur=c.cursor()
    cur.execute(\"select column_name from information_schema.columns where table_schema='proc' and table_name='bp_policy' order by ordinal_position\")
    print([r[0] for r in cur.fetchall()])
"
```
Adjust the INSERT's column list to whatever that prints — do not guess.

- [ ] **Step 4: Write the store**

```python
# src/services/negotiation_advice/store.py
"""Persistence for advice sessions and buyer-stated facts.

Stated facts live in their own table: a value the buyer asserts must never be
mistaken for measured data, and must be withdrawable so the advice reverts
cleanly. Withdrawal stamps withdrawn_at rather than deleting, keeping the trail.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


def _rows(cur, sql: str, params: tuple = ()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def save_advice(conn, *, deal_id: str, supplier_id: Optional[str],
                quadrant: Optional[str], quadrant_source: str,
                quadrant_confidence: Optional[float], style: Optional[str],
                style_source: str, signals: dict, plays: list,
                created_by: Optional[str]) -> dict:
    advice_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_negotiation_advice "
        "(advice_id, deal_id, supplier_id, quadrant, quadrant_source, "
        " quadrant_confidence, style, style_source, signals, plays, "
        " created_by, created_at, updated_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (advice_id, deal_id, supplier_id, quadrant, quadrant_source,
         quadrant_confidence, style, style_source,
         json.dumps(signals, default=str), json.dumps(plays, default=str),
         created_by, now, now),
    )
    conn.commit()
    return {"advice_id": advice_id, "deal_id": deal_id,
            "supplier_id": supplier_id, "quadrant": quadrant,
            "quadrant_source": quadrant_source,
            "quadrant_confidence": quadrant_confidence, "style": style,
            "style_source": style_source, "signals": signals, "plays": plays,
            "created_by": created_by, "created_at": now.isoformat()}


def load_advice(conn, deal_id: str) -> Optional[dict]:
    rows = _rows(conn.cursor(),
                 "SELECT * FROM proc.bp_negotiation_advice WHERE deal_id=%s "
                 "ORDER BY created_at DESC LIMIT 1", (deal_id,))
    return rows[0] if rows else None


def state_fact(conn, *, advice_id: str, fact_key: str, fact_value: Any,
               stated_by: Optional[str]) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO proc.bp_negotiation_advice_fact "
        "(advice_id, fact_key, fact_value, stated_by, stated_at, withdrawn_at) "
        "VALUES (%s,%s,%s,%s,%s,NULL) "
        "ON CONFLICT (advice_id, fact_key) DO UPDATE SET "
        "fact_value = EXCLUDED.fact_value, stated_by = EXCLUDED.stated_by, "
        "stated_at = EXCLUDED.stated_at, withdrawn_at = NULL",
        (advice_id, fact_key, str(fact_value), stated_by,
         datetime.now(timezone.utc)),
    )
    conn.commit()


def withdraw_fact(conn, *, advice_id: str, fact_key: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE proc.bp_negotiation_advice_fact SET withdrawn_at=%s "
        "WHERE advice_id=%s AND fact_key=%s AND withdrawn_at IS NULL",
        (datetime.now(timezone.utc), advice_id, fact_key),
    )
    conn.commit()


def active_facts(conn, advice_id: str) -> dict:
    rows = _rows(conn.cursor(),
                 "SELECT fact_key, fact_value FROM "
                 "proc.bp_negotiation_advice_fact "
                 "WHERE advice_id=%s AND withdrawn_at IS NULL", (advice_id,))
    return {r["fact_key"]: r["fact_value"] for r in rows}
```

- [ ] **Step 5: Run tests, then apply the migration**

```bash
set -a && . ./.env; set +a
.venv/bin/python -m pytest tests/services/negotiation_advice/test_store.py -q -p no:cacheprovider
psql "$DATABASE_URL" -f deploy/sql/2026-07-29_bp_negotiation_advice.sql
psql "$DATABASE_URL" -c "\d proc.bp_negotiation_advice"
psql "$DATABASE_URL" -c "select policy_name from proc.bp_policy where policy_name='negotiation_advice_thresholds'"
```
Expected: tests PASS (4 passed); both tables listed; one policy row. Re-run the
migration once to confirm it is idempotent — it must succeed with no error and
no duplicate policy row.

If `$DATABASE_URL` is not set in `.env`, get connection details from
`config/settings.py` rather than guessing.

- [ ] **Step 6: Commit**

```bash
git add deploy/sql/2026-07-29_bp_negotiation_advice.sql \
        src/services/negotiation_advice/store.py \
        tests/services/negotiation_advice/test_store.py
git commit -m "feat(negotiation-advice): advice + stated-fact tables and store"
```

---

### Task 6: Compose advice end to end

**Files:**
- Create: `src/services/negotiation_advice/advisor.py`
- Modify: `src/services/negotiation_advice/__init__.py`
- Test: `tests/services/negotiation_advice/test_advisor.py`

**Interfaces:**
- Consumes: `signals` (T1), `classification` (T2), `ranking` (T3), `grounding` (T4), `store` (T5).
- Produces:
  - `build_advice(deal_id: str, *, conn=None, created_by=None, overrides: dict | None = None, stated: dict | None = None, lever: str | None = None, limit: int = 6) -> dict | None`
  - `apply_turn(deal_id: str, message: dict, *, conn=None, created_by=None) -> dict | None`
  - `load_thresholds(conn) -> dict`
  - Payload shape: `{"deal_id", "supplier_id", "supplier_name", "quadrant", "quadrant_source", "quadrant_reasons", "quadrant_confidence", "style", "style_source", "style_reasons", "indeterminate", "signals", "stated_facts", "plays", "advice_id"}`

**Background the implementer needs:**

`overrides` may carry `quadrant` and/or `style`; when present, the corresponding
`*_source` becomes `"buyer"` and the computed value is replaced. `stated` maps
`fact_key -> value` for buyer-supplied signal values; these are merged over the
measured signals **for classification purposes only**, and every stated key must
appear in the payload's `stated_facts` so the UI can label it "stated by you".
The measured value stays in `signals` untouched.

`apply_turn` handles a `message` dict with an `action` of:
`"more_plays"` (optional `lever`), `"set_lever"` (`lever`), `"compare_style"`
(`style` — returns a `comparison` key alongside the primary advice),
`"override"` (`quadrant` and/or `style`), `"state_fact"` (`fact_key`, `fact_value`),
`"withdraw_fact"` (`fact_key`).

`load_thresholds` reads the `negotiation_advice_thresholds` policy via
`PolicyEngine.get_policy(slug)` (`src/engines/policy_engine.py:347`) and falls
back to `classification.default_thresholds()`. Wrap it in try/except: a policy
lookup failure must degrade to defaults, never fail the request.

When `classify` returns `indeterminate=True`, return the payload with
`plays: []` and the reasons populated. Do **not** pick a quadrant to get plays.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/negotiation_advice/test_advisor.py
import pytest
from src.services.negotiation_advice import advisor as ad


_SIGNALS = {"deal_id": "D-1", "supplier_id": "SUP-1", "supplier_name": "Orbis",
            "currency": "GBP", "deal_value": 200000.0, "invoice_total": 205000.0,
            "po_total": 198000.0, "quote_supplier_count": 1,
            "alternative_supplier_count": 30, "risk_score": 0.2,
            "is_preferred": False, "price_variance_pct": 8.4}


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(ad, "gather_signals", lambda cur, deal_id:
                        dict(_SIGNALS) if deal_id == "D-1" else None)
    monkeypatch.setattr(ad, "load_thresholds", lambda conn:
                        {"high_spend": 98175.0, "many_alternatives": 23})
    monkeypatch.setattr(ad, "save_advice", lambda conn, **kw:
                        {"advice_id": "A-1", **kw})
    monkeypatch.setattr(ad, "active_facts", lambda conn, advice_id: {})
    monkeypatch.setattr(ad, "state_fact", lambda conn, **kw: None)
    monkeypatch.setattr(ad, "withdraw_fact", lambda conn, **kw: None)
    monkeypatch.setattr(ad, "load_advice", lambda conn, deal_id:
                        {"advice_id": "A-1"})


class _Conn:
    def cursor(self):
        return object()

    def commit(self):
        pass


def test_build_advice_produces_classified_grounded_plays():
    out = ad.build_advice("D-1", conn=_Conn())
    assert out["quadrant"] == "Leverage"
    assert out["quadrant_source"] == "computed"
    assert out["style"] == "Competitive"
    assert out["plays"]
    assert all(p["state"] in ("ready", "groundwork") for p in out["plays"])
    assert out["quadrant_reasons"]


def test_unknown_deal_returns_none():
    assert ad.build_advice("NOPE", conn=_Conn()) is None


def test_override_marks_the_source_as_buyer():
    out = ad.build_advice("D-1", conn=_Conn(),
                          overrides={"quadrant": "Bottleneck"})
    assert out["quadrant"] == "Bottleneck"
    assert out["quadrant_source"] == "buyer"
    assert out["style_source"] == "computed"


def test_overriding_the_quadrant_recomputes_the_style():
    # Bottleneck must not inherit Leverage's Competitive style — a thin supply
    # market calls for the opposite posture.
    out = ad.build_advice("D-1", conn=_Conn(),
                          overrides={"quadrant": "Bottleneck"})
    assert out["style"] == "Principled"
    assert out["style_source"] == "computed"
    assert out["style_reasons"]


def test_overriding_both_keeps_the_buyers_style():
    out = ad.build_advice("D-1", conn=_Conn(),
                          overrides={"quadrant": "Bottleneck",
                                     "style": "Accommodating"})
    assert out["style"] == "Accommodating"
    assert out["style_source"] == "buyer"


def test_stated_fact_changes_classification_and_is_labelled():
    out = ad.build_advice("D-1", conn=_Conn(),
                          stated={"alternative_supplier_count": 2})
    # high spend + few alternatives -> Strategic, not Leverage
    assert out["quadrant"] == "Strategic"
    assert out["stated_facts"]["alternative_supplier_count"] == 2
    # the measured value is preserved, not overwritten
    assert out["signals"]["alternative_supplier_count"] == 30


def test_indeterminate_returns_no_plays_and_does_not_guess():
    out = ad.build_advice("D-1", conn=_Conn(),
                          stated={"deal_value": None,
                                  "alternative_supplier_count": None})
    assert out["indeterminate"] is True
    assert out["quadrant"] is None
    assert out["plays"] == []
    assert out["quadrant_reasons"]


def test_turn_set_lever_restricts_the_levers():
    out = ad.apply_turn("D-1", {"action": "set_lever", "lever": "Risk"},
                        conn=_Conn())
    assert {p["lever"] for p in out["plays"]} == {"Risk"}


def test_turn_more_plays_returns_more_than_the_default():
    base = ad.build_advice("D-1", conn=_Conn())
    more = ad.apply_turn("D-1", {"action": "more_plays"}, conn=_Conn())
    assert len(more["plays"]) >= len(base["plays"])


def test_turn_compare_style_returns_both():
    out = ad.apply_turn("D-1", {"action": "compare_style",
                                "style": "Collaborative"}, conn=_Conn())
    assert out["style"] == "Competitive"
    assert out["comparison"]["style"] == "Collaborative"
    assert out["comparison"]["plays"]


def test_turn_override_flows_through():
    out = ad.apply_turn("D-1", {"action": "override", "style": "Principled"},
                        conn=_Conn())
    assert out["style"] == "Principled"
    assert out["style_source"] == "buyer"


def test_unknown_action_is_ignored_not_fatal():
    out = ad.apply_turn("D-1", {"action": "nonsense"}, conn=_Conn())
    assert out["quadrant"] == "Leverage"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_advisor.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.services.negotiation_advice.advisor'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/negotiation_advice/advisor.py
"""Compose an advice payload, and apply one conversational turn.

Buyer-stated facts are merged over measured signals for classification only.
The measured value stays in `signals`; the stated value is echoed in
`stated_facts` so the UI can label its provenance and the buyer can withdraw it.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from src.services.db import get_conn
from src.services.negotiation_advice.classification import (
    classify, default_thresholds, _style_for, THRESHOLD_POLICY_SLUG,
)
from src.services.negotiation_advice.grounding import apply_states
from src.services.negotiation_advice.ranking import rank_plays
from src.services.negotiation_advice.signals import (
    gather_signals, market_context_dict, supplier_performance_dict,
)
from src.services.negotiation_advice.store import (
    active_facts, load_advice, save_advice, state_fact, withdraw_fact,
)

log = logging.getLogger(__name__)

_DEFAULT_LIMIT = 6
_MORE_LIMIT = 12


def load_thresholds(conn) -> dict:
    """Governed thresholds, degrading to seeded defaults on any failure."""
    try:
        from src.engines.policy_engine import PolicyEngine
        policy = PolicyEngine().get_policy(THRESHOLD_POLICY_SLUG)
        if policy:
            details = policy.get("policy_details") or policy.get("details")
            if isinstance(details, dict):
                return {**default_thresholds(), **details}
    except Exception:
        log.debug("threshold policy lookup failed; using defaults",
                  exc_info=True)
    return default_thresholds()


def build_advice(deal_id: str, *, conn=None, created_by: Optional[str] = None,
                 overrides: Optional[dict] = None,
                 stated: Optional[dict] = None,
                 lever: Optional[str] = None,
                 limit: int = _DEFAULT_LIMIT) -> Optional[dict]:
    if conn is None:
        with get_conn() as own:
            return build_advice(deal_id, conn=own, created_by=created_by,
                                overrides=overrides, stated=stated,
                                lever=lever, limit=limit)

    measured = gather_signals(conn.cursor(), deal_id)
    if measured is None:
        return None

    stated = dict(stated or {})
    overrides = dict(overrides or {})

    # Stated facts steer classification; measured values are left intact.
    effective = dict(measured)
    effective.update(stated)

    verdict = classify(effective, thresholds=load_thresholds(conn))
    quadrant = overrides.get("quadrant") or verdict["quadrant"]
    quadrant_source = "buyer" if overrides.get("quadrant") else "computed"

    style_reasons = verdict.get("style_reasons") or []
    if overrides.get("quadrant") and not overrides.get("style"):
        # The buyer moved the quadrant; the computed style belonged to the old
        # one. Leaving it would pair e.g. Bottleneck with Competitive — the
        # opposite of what a thin supply market calls for.
        style, style_reasons = _style_for(quadrant, effective)
        style_source = "computed"
    else:
        style = overrides.get("style") or verdict["style"]
        style_source = "buyer" if overrides.get("style") else "computed"

    plays: list = []
    if quadrant and style:
        ranked = rank_plays(
            quadrant, style,
            lever_priorities=[lever] if lever else None,
            supplier_performance=supplier_performance_dict(effective),
            market_context=market_context_dict(effective),
            limit=limit,
        )
        plays = apply_states(ranked.get("plays") or [], effective)

    saved = save_advice(
        conn, deal_id=deal_id, supplier_id=measured.get("supplier_id"),
        quadrant=quadrant, quadrant_source=quadrant_source,
        quadrant_confidence=verdict.get("quadrant_confidence"),
        style=style, style_source=style_source, signals=measured, plays=plays,
        created_by=created_by,
    )

    return {
        "advice_id": saved.get("advice_id"),
        "deal_id": deal_id,
        "supplier_id": measured.get("supplier_id"),
        "supplier_name": measured.get("supplier_name"),
        "quadrant": quadrant,
        "quadrant_source": quadrant_source,
        "quadrant_reasons": verdict.get("quadrant_reasons") or [],
        "quadrant_confidence": verdict.get("quadrant_confidence"),
        "style": style,
        "style_source": style_source,
        "style_reasons": style_reasons,
        "indeterminate": bool(verdict.get("indeterminate")) and not quadrant,
        "signals": measured,
        "stated_facts": stated,
        "plays": plays,
    }


def apply_turn(deal_id: str, message: dict, *, conn=None,
               created_by: Optional[str] = None) -> Optional[dict]:
    if conn is None:
        with get_conn() as own:
            return apply_turn(deal_id, message, conn=own, created_by=created_by)

    action = str((message or {}).get("action") or "").strip()
    existing = load_advice(conn, deal_id)
    advice_id = (existing or {}).get("advice_id")

    if action == "state_fact" and advice_id:
        state_fact(conn, advice_id=advice_id,
                   fact_key=message.get("fact_key"),
                   fact_value=message.get("fact_value"),
                   stated_by=created_by or "buyer")
    elif action == "withdraw_fact" and advice_id:
        withdraw_fact(conn, advice_id=advice_id,
                      fact_key=message.get("fact_key"))

    stated = active_facts(conn, advice_id) if advice_id else {}
    stated = {k: _coerce(v) for k, v in (stated or {}).items()}

    overrides = {}
    if action == "override":
        for key in ("quadrant", "style"):
            if message.get(key):
                overrides[key] = message[key]

    lever = message.get("lever") if action == "set_lever" else None
    limit = _MORE_LIMIT if action == "more_plays" else _DEFAULT_LIMIT

    out = build_advice(deal_id, conn=conn, created_by=created_by,
                       overrides=overrides, stated=stated, lever=lever,
                       limit=limit)
    if out is None:
        return None

    if action == "compare_style" and message.get("style") and out["quadrant"]:
        other = rank_plays(
            out["quadrant"], message["style"],
            supplier_performance=supplier_performance_dict(out["signals"]),
            market_context=market_context_dict(out["signals"]),
            limit=_DEFAULT_LIMIT,
        )
        out["comparison"] = {
            "style": message["style"],
            "plays": apply_states(other.get("plays") or [], out["signals"]),
        }
    return out


def _coerce(value: Any) -> Any:
    """Stated facts arrive as text; numbers must compare as numbers."""
    if value is None or isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text or text.lower() in ("none", "null"):
        return None
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text
```

Then expose the surface:

```python
# src/services/negotiation_advice/__init__.py
"""Buyer-facing negotiation advice."""
from src.services.negotiation_advice.advisor import apply_turn, build_advice

__all__ = ["build_advice", "apply_turn"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/ -q -p no:cacheprovider`
Expected: PASS (all Tasks 1-6)

- [ ] **Step 5: Verify against a live deal**

```bash
set -a && . ./.env; set +a
PYTHONPATH=src:. .venv/bin/python - <<'EOF'
import json
from src.services.db import get_conn
from src.services.negotiation_advice import build_advice
with get_conn() as c:
    cur = c.cursor()
    cur.execute("select deal_id from proc.bp_deal_overview "
                "where invoice_total is not null and po_total is not null "
                "order by invoice_total desc limit 1")
    deal_id = cur.fetchone()[0]
out = build_advice(deal_id)
print("quadrant:", out["quadrant"], out["quadrant_confidence"],
      "| style:", out["style"])
for r in out["quadrant_reasons"]:
    print("  reason:", r)
for p in out["plays"]:
    print(f"  [{p['state']:10s}] {p['lever']:12s} {p['play'][:60]}")
    if p.get("unlocked_by"):
        print(f"               unlocked_by: {p['unlocked_by']}")
EOF
```
Expected: a real quadrant with numeric reasons, and plays each carrying a state.
Given only 1 of 5,038 deals has competing quotes, expect competitive-tension
plays to appear as `groundwork` — that is correct behaviour, not a bug.

- [ ] **Step 6: Commit**

```bash
git add src/services/negotiation_advice/advisor.py \
        src/services/negotiation_advice/__init__.py \
        tests/services/negotiation_advice/test_advisor.py
git commit -m "feat(negotiation-advice): compose advice and apply a conversational turn"
```

---

### Task 7: Endpoints and the dashboard swap

**Files:**
- Modify: `src/api/routers/negotiate.py`
- Modify: `src/services/negotiate_dashboard.py:230-270` (`negotiation_strategy`)
- Test: `tests/api/test_negotiate_advice_endpoints.py`

**Interfaces:**
- Consumes: `build_advice`, `apply_turn` (T6).
- Produces:
  - `GET /negotiate/{deal_id}/advice`
  - `POST /negotiate/{deal_id}/advice/message`
  - `DELETE /negotiate/{deal_id}/advice/fact/{fact_key}`
  - `negotiate_dashboard.negotiation_strategy` gains a `plays` key; `leveragePoints` and `counterStrategy` are removed.

**Background the implementer needs:**

`src/api/routers/negotiate.py:24` already has
`get_negotiate_dashboard(deal_id)` calling `build_negotiate_dashboard`. Follow
its error style: catch, log, `raise HTTPException(status_code=500, ...)`, and
404 when the builder returns `None`.

In `negotiate_dashboard.negotiation_strategy` (line 230) **keep**
`highLevelSummary`, `currentStandpoint`, `preferredOutcome` and
`supplierInsights` — they are computed. **Remove** the two constant fields at
lines 265-267 (`leveragePoints`, `counterStrategy`) and add `plays` from
`build_advice`. Import `build_advice` lazily inside the function to avoid an
import cycle, and wrap it so an advice failure degrades to `plays: []` rather
than breaking the whole dashboard — the rest of that payload is unrelated.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_negotiate_advice_endpoints.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routers.negotiate as rt


_ADVICE = {"advice_id": "A-1", "deal_id": "D-1", "quadrant": "Leverage",
           "quadrant_source": "computed", "quadrant_reasons": ["because"],
           "quadrant_confidence": 0.8, "style": "Competitive",
           "style_source": "computed", "style_reasons": [],
           "indeterminate": False, "signals": {}, "stated_facts": {},
           "plays": [{"lever": "Commercial", "play": "Benchmark it",
                      "state": "ready", "evidence": [], "score": 1.0}]}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(rt.router)
    return TestClient(app)


def test_get_advice_returns_the_payload(client, monkeypatch):
    monkeypatch.setattr(rt, "build_advice", lambda deal_id, **kw: dict(_ADVICE))
    r = client.get("/negotiate/D-1/advice")
    assert r.status_code == 200
    assert r.json()["quadrant"] == "Leverage"
    assert r.json()["plays"][0]["state"] == "ready"


def test_get_advice_404s_for_unknown_deal(client, monkeypatch):
    monkeypatch.setattr(rt, "build_advice", lambda deal_id, **kw: None)
    assert client.get("/negotiate/NOPE/advice").status_code == 404


def test_post_message_applies_a_turn(client, monkeypatch):
    seen = {}

    def _apply(deal_id, message, **kw):
        seen["message"] = message
        return dict(_ADVICE, style="Principled")

    monkeypatch.setattr(rt, "apply_turn", _apply)
    r = client.post("/negotiate/D-1/advice/message",
                    json={"action": "override", "style": "Principled"})
    assert r.status_code == 200
    assert r.json()["style"] == "Principled"
    assert seen["message"]["action"] == "override"


def test_post_message_404s_for_unknown_deal(client, monkeypatch):
    monkeypatch.setattr(rt, "apply_turn", lambda deal_id, message, **kw: None)
    r = client.post("/negotiate/NOPE/advice/message", json={"action": "more_plays"})
    assert r.status_code == 404


def test_delete_fact_withdraws_and_returns_refreshed_advice(client, monkeypatch):
    seen = {}

    def _apply(deal_id, message, **kw):
        seen["message"] = message
        return dict(_ADVICE)

    monkeypatch.setattr(rt, "apply_turn", _apply)
    r = client.delete("/negotiate/D-1/advice/fact/alternative_supplier_count")
    assert r.status_code == 200
    assert seen["message"]["action"] == "withdraw_fact"
    assert seen["message"]["fact_key"] == "alternative_supplier_count"
```

And for the dashboard swap:

```python
# append to tests/api/test_negotiate_advice_endpoints.py
import src.services.negotiate_dashboard as nd


class _Cur:
    description = ()

    def execute(self, *a, **k):
        pass

    def fetchall(self):
        return []


def test_strategy_no_longer_returns_hardcoded_leverage_points(monkeypatch):
    monkeypatch.setattr(nd, "_deal", lambda cur, deal_id: {
        "deal_id": "D-1", "supplier_id": "SUP-1", "currency": "GBP",
        "quote_total": 100.0, "po_total": 90.0, "invoice_total": 95.0,
        "last_activity_date": None})
    monkeypatch.setattr(nd, "_supplier_insights",
                        lambda cur, d: ("p", "k", "r"))
    out = nd.negotiation_strategy(_Cur(), "D-1")[0]
    assert "leveragePoints" not in out
    assert "counterStrategy" not in out
    assert "plays" in out
    # the computed parts survive
    assert "currentStandpoint" in out
    assert "preferredOutcome" in out


def test_strategy_degrades_to_empty_plays_when_advice_fails(monkeypatch):
    monkeypatch.setattr(nd, "_deal", lambda cur, deal_id: {
        "deal_id": "D-1", "supplier_id": "SUP-1", "currency": "GBP",
        "quote_total": 100.0, "po_total": 90.0, "invoice_total": 95.0,
        "last_activity_date": None})
    monkeypatch.setattr(nd, "_supplier_insights",
                        lambda cur, d: ("p", "k", "r"))
    monkeypatch.setattr(nd, "_advice_plays",
                        lambda deal_id: (_ for _ in ()).throw(RuntimeError("x")))
    out = nd.negotiation_strategy(_Cur(), "D-1")[0]
    assert out["plays"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/api/test_negotiate_advice_endpoints.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: module 'src.api.routers.negotiate' has no attribute 'build_advice'`

- [ ] **Step 3: Add the endpoints**

```python
# add to src/api/routers/negotiate.py imports
from src.services.negotiation_advice import apply_turn, build_advice
```

```python
# add to src/api/routers/negotiate.py
@router.get("/{deal_id}/advice", summary="Grounded negotiation advice for a deal")
def get_negotiate_advice(deal_id: str) -> dict[str, Any]:
    try:
        result = build_advice(deal_id)
    except Exception as exc:
        logger.exception("negotiation advice failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No deal {deal_id}")
    return result


@router.post("/{deal_id}/advice/message", summary="One advice conversation turn")
def post_negotiate_advice_message(deal_id: str,
                                  body: dict[str, Any]) -> dict[str, Any]:
    try:
        result = apply_turn(deal_id, body or {})
    except Exception as exc:
        logger.exception("advice turn failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No deal {deal_id}")
    return result


@router.delete("/{deal_id}/advice/fact/{fact_key}",
               summary="Withdraw a buyer-stated fact")
def delete_negotiate_advice_fact(deal_id: str, fact_key: str) -> dict[str, Any]:
    try:
        result = apply_turn(deal_id, {"action": "withdraw_fact",
                                      "fact_key": fact_key})
    except Exception as exc:
        logger.exception("fact withdrawal failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No deal {deal_id}")
    return result
```

- [ ] **Step 4: Swap the dashboard's two constant fields**

In `src/services/negotiate_dashboard.py`, add:

```python
def _advice_plays(deal_id: str) -> list[dict]:
    """Ranked, grounded plays. Imported lazily to avoid an import cycle."""
    from src.services.negotiation_advice import build_advice
    advice = build_advice(deal_id)
    return list((advice or {}).get("plays") or [])
```

Then in `negotiation_strategy`, delete the `leveragePoints` and
`counterStrategy` entries (lines 265-267) and add:

```python
    try:
        plays = _advice_plays(deal_id)
    except Exception:
        log.exception("advice plays unavailable for %s", deal_id)
        plays = []
```

…returning `"plays": plays` in the dict alongside the retained
`highLevelSummary`, `currentStandpoint`, `preferredOutcome` and
`supplierInsights`.

- [ ] **Step 5: Run tests and check for other consumers**

```bash
set -a && . ./.env; set +a
.venv/bin/python -m pytest tests/api/test_negotiate_advice_endpoints.py -q -p no:cacheprovider
# nothing else may depend on the removed keys
grep -rn "leveragePoints\|counterStrategy" --include=*.py --include=*.js \
     --include=*.jsx --include=*.ts src/ tests/ \
     /home/muthu/PycharmProjects/beyond_procwise_ui/src \
     /home/muthu/PycharmProjects/beyond-procwaise-Api 2>/dev/null
.venv/bin/python -m pytest tests/test_negotiate_dashboard.py -q -p no:cacheprovider 2>/dev/null || true
```
Expected: tests PASS; the grep returns nothing (both keys were verified
unreferenced outside `negotiate_dashboard.py` during the investigation — if the
grep now finds a consumer, update it in this task rather than leaving it broken).

- [ ] **Step 6: Commit**

```bash
git add src/api/routers/negotiate.py src/services/negotiate_dashboard.py \
        tests/api/test_negotiate_advice_endpoints.py
git commit -m "feat(negotiation-advice): advice endpoints; dashboard shows ranked plays"
```

---

### Task 8: Register the advisor agent

**Files:**
- Create: `src/agents/negotiation_advisor_agent.py`
- Modify: `agent_definitions.json`
- Test: `tests/agents/test_negotiation_advisor_agent.py`

**Interfaces:**
- Consumes: `build_advice`, `apply_turn` (T6).
- Produces: `NegotiationAdvisorAgent` with `run(context: AgentContext) -> AgentOutput`, registered under slug `negotiation_advisor`.

**Background the implementer needs:**

Every agent in this repo implements `run(self, context: AgentContext) -> AgentOutput`
and is dispatched generically — `src/orchestration/agentnick_control.py:104` calls
`agent.run(ctx)` for any registered slug, and `BaseAgent.execute` does the same.
An agent whose `run` takes different parameters raises `TypeError` through those
paths, so keep the signature exact.

`agent_definitions.json` is `{"agents": [...]}`; each entry needs at minimum
`agentId`, `agentType`, `slug`, `class_path`, `description`, `role`,
`capabilities`, `required_inputs`, `output_fields`, `dependencies`, `inputs`,
`outputs`, `version`. Copy the shape of the `requirements` entry. The highest
existing `agentId` is 14, so use 15. `tests/test_required_agents_are_registerable.py`
and `tests/test_auto_registry.py` assert catalogue consistency — both must pass.

Return `AgentStatus.SUCCESS` with `data={}` and an `error` string when the deal is
unknown rather than raising, matching how other agents handle absent data. Use
`AgentStatus.FAILED` only for genuine failures.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_negotiation_advisor_agent.py
import json

from src.agents.base_agent import AgentContext, AgentStatus


def _ctx(data):
    return AgentContext(workflow_id="W1", agent_id="negotiation_advisor",
                        user_id="buyer", input_data=data)


_ADVICE = {"advice_id": "A-1", "deal_id": "D-1", "quadrant": "Leverage",
           "style": "Competitive", "plays": [{"lever": "Commercial",
                                              "state": "ready"}]}


def _agent(monkeypatch, advice=_ADVICE, turn=None):
    import src.agents.negotiation_advisor_agent as mod
    agent = mod.NegotiationAdvisorAgent.__new__(mod.NegotiationAdvisorAgent)
    agent._with_plan = lambda ctx, out: out
    monkeypatch.setattr(mod, "build_advice", lambda deal_id, **kw: advice)
    monkeypatch.setattr(mod, "apply_turn",
                        lambda deal_id, message, **kw: turn or advice)
    return agent


def test_run_returns_advice_for_a_deal(monkeypatch):
    agent = _agent(monkeypatch)
    out = agent.run(_ctx({"deal_id": "D-1"}))
    assert out.status == AgentStatus.SUCCESS
    assert out.data["quadrant"] == "Leverage"
    assert out.data["plays"]


def test_run_applies_a_turn_when_an_action_is_given(monkeypatch):
    agent = _agent(monkeypatch, turn=dict(_ADVICE, style="Principled"))
    out = agent.run(_ctx({"deal_id": "D-1", "action": "override",
                          "style": "Principled"}))
    assert out.data["style"] == "Principled"


def test_missing_deal_id_is_reported_not_raised(monkeypatch):
    agent = _agent(monkeypatch)
    out = agent.run(_ctx({}))
    assert out.status == AgentStatus.SUCCESS
    assert out.error


def test_unknown_deal_is_reported_not_raised(monkeypatch):
    agent = _agent(monkeypatch, advice=None)
    out = agent.run(_ctx({"deal_id": "NOPE"}))
    assert out.error


def test_run_signature_matches_the_shared_contract():
    import inspect
    from src.agents.negotiation_advisor_agent import NegotiationAdvisorAgent
    params = list(inspect.signature(NegotiationAdvisorAgent.run).parameters)
    assert params == ["self", "context"]


def test_registered_in_the_catalogue():
    with open("agent_definitions.json", encoding="utf-8") as fh:
        agents = json.load(fh)["agents"]
    entry = next(a for a in agents if a["slug"] == "negotiation_advisor")
    assert entry["class_path"] == (
        "agents.negotiation_advisor_agent.NegotiationAdvisorAgent")
    assert len({a["slug"] for a in agents}) == len(agents)
    assert len({a["agentId"] for a in agents}) == len(agents)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/agents/test_negotiation_advisor_agent.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agents.negotiation_advisor_agent'`

- [ ] **Step 3: Write the agent**

```python
# src/agents/negotiation_advisor_agent.py
"""Buyer-facing negotiation advice, as an agent.

Thin by design: classification, ranking, grounding and persistence all live in
src/services/negotiation_advice/. This exists so the advisor appears in the Agent
Workspace alongside the other agents and can be dispatched generically.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from agents.base_agent import AgentContext, AgentOutput, AgentStatus, BaseAgent
from src.services.negotiation_advice import apply_turn, build_advice

logger = logging.getLogger(__name__)


class NegotiationAdvisorAgent(BaseAgent):
    """Ranked, grounded negotiation advice for one deal."""

    AGENTIC_PLAN_STEPS = (
        "Gather the deal's spend, supply-market and variance signals.",
        "Suggest a supplier quadrant and negotiation style, with reasons.",
        "Rank plays and mark each ready or groundwork against the evidence.",
    )

    def run(self, context: AgentContext) -> AgentOutput:
        try:
            data: Dict[str, Any] = context.input_data or {}
            deal_id = str(data.get("deal_id") or "").strip()
            if not deal_id:
                return AgentOutput(
                    status=AgentStatus.SUCCESS, data={},
                    error="deal_id is required for negotiation advice",
                )

            created_by = str(data.get("created_by") or context.user_id or "")
            action = str(data.get("action") or "").strip()
            if action:
                result = apply_turn(deal_id, data, created_by=created_by)
            else:
                result = build_advice(deal_id, created_by=created_by)

            if result is None:
                return AgentOutput(status=AgentStatus.SUCCESS, data={},
                                   error=f"No deal {deal_id}")

            return self._with_plan(context, AgentOutput(
                status=AgentStatus.SUCCESS,
                data=dict(result),
                next_agents=[],
                confidence=float(result.get("quadrant_confidence") or 0.0),
            ))
        except Exception as exc:  # pragma: no cover - top-level guard
            logger.exception("NegotiationAdvisorAgent.run failed")
            return AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))
```

Add to `agent_definitions.json`, inside the `"agents"` array:

```json
    {
      "agentId": 15,
      "agentType": "NegotiationAdvisorAgent",
      "slug": "negotiation_advisor",
      "class_path": "agents.negotiation_advisor_agent.NegotiationAdvisorAgent",
      "description": "Gives a buyer ranked, evidence-grounded negotiation advice for a deal: suggested supplier quadrant and style with reasons, plays marked ready or groundwork, and a conversation to override the framing or add facts the data lacks.",
      "role": "advise",
      "capabilities": ["negotiation_advice"],
      "required_inputs": ["deal_id"],
      "output_fields": ["quadrant", "style", "plays"],
      "dependencies": ["db_client", "ollama_client"],
      "inputs": {
        "required": ["deal_id"],
        "optional": ["action", "lever", "style", "quadrant", "fact_key",
                     "fact_value", "created_by"]
      },
      "outputs": ["quadrant", "style", "plays"],
      "version": "1.0.0",
      "elicit": [
        {
          "any_of": ["deal_id"],
          "type": "text",
          "prompt": "Which deal do you want negotiation advice for?"
        }
      ]
    }
```

- [ ] **Step 4: Run tests including the catalogue guards**

```bash
set -a && . ./.env; set +a
.venv/bin/python -m pytest tests/agents/test_negotiation_advisor_agent.py \
  tests/test_required_agents_are_registerable.py tests/test_auto_registry.py \
  tests/agents/test_agent_elicit_manifest.py -q -p no:cacheprovider
```
Expected: PASS, no new failures.

- [ ] **Step 5: Confirm the agent resolves like the other fifteen**

```bash
set -a && . ./.env; set +a
PYTHONPATH=src:. .venv/bin/python - <<'EOF'
import importlib, inspect, json
agents = json.load(open("agent_definitions.json"))["agents"]
for a in agents:
    mod, _, cls = a["class_path"].rpartition(".")
    C = getattr(importlib.import_module(mod), cls)
    sig = str(inspect.signature(C.run))
    flag = "" if sig == "(self, context: 'AgentContext') -> 'AgentOutput'" \
           or "context" in sig else "  <-- CONTRACT MISMATCH"
    print(f"{a['slug']:22s} {sig}{flag}")
EOF
```
Expected: 15 rows, `negotiation_advisor` among them taking `context`.

- [ ] **Step 6: Commit**

```bash
git add src/agents/negotiation_advisor_agent.py agent_definitions.json \
        tests/agents/test_negotiation_advisor_agent.py
git commit -m "feat(negotiation-advice): register the negotiation_advisor agent"
```

---

### Task 9: The LLM-written "why"

**Files:**
- Modify: `src/services/negotiation_advice/advisor.py`
- Test: `tests/services/negotiation_advice/test_why.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `explain_play(play: dict, signals: dict, *, generate=None) -> str` and each play in `build_advice` gaining a `why` key.

**Background the implementer needs:**

AgentNick phrases *why this play suits this supplier*, from the grounded facts
only. It must not choose levers, reorder plays, or invent preconditions — those
stay deterministic so the advice is auditable.

Use `src/services/ollama_client.ollama_generate(prompt, model=..., temperature=0.0, num_predict=..., timeout=..., retries=..., think=False)`.
`think=False` is mandatory — `AgentNick:unified` is a reasoning model and returns
an empty `response` without it. Read the model name the same way
`deal_summary.py` does (`_SUMMARY_MODEL`) rather than hardcoding a string.

An LLM failure must degrade to the deterministic `rationale` already on the play,
never fail the request — the advice is useful without prose.

Inject `generate` for tests so no unit test calls a live model.

- [ ] **Step 1: Write the failing test**

```python
# tests/services/negotiation_advice/test_why.py
from src.services.negotiation_advice import advisor as ad


_PLAY = {"lever": "Commercial", "play": "Demand tiered volume discounts",
         "rationale": "deterministic fallback", "state": "ready",
         "evidence": [{"label": "Deal value", "value": "200,000",
                       "source": "deal"}]}
_SIG = {"supplier_name": "Orbis", "deal_value": 200000.0, "currency": "GBP"}


def test_why_uses_the_model_output():
    out = ad.explain_play(_PLAY, _SIG,
                          generate=lambda **kw: "Because volume is material.")
    assert out == "Because volume is material."


def test_prompt_carries_only_grounded_facts_and_forbids_invention():
    seen = {}

    def _gen(**kw):
        seen["prompt"] = kw.get("prompt") or ""
        return "ok"

    ad.explain_play(_PLAY, _SIG, generate=_gen)
    prompt = seen["prompt"]
    assert "Orbis" in prompt
    assert "200,000" in prompt or "200000" in prompt
    assert "Demand tiered volume discounts" in prompt
    low = prompt.lower()
    assert "do not" in low and ("invent" in low or "fabricate" in low)


def test_model_is_called_with_think_false():
    seen = {}
    ad.explain_play(_PLAY, _SIG,
                    generate=lambda **kw: seen.update(kw) or "ok")
    assert seen.get("think") is False


def test_llm_failure_falls_back_to_the_deterministic_rationale():
    def _boom(**kw):
        raise RuntimeError("ollama down")

    assert ad.explain_play(_PLAY, _SIG, generate=_boom) == "deterministic fallback"


def test_empty_model_output_falls_back():
    assert ad.explain_play(_PLAY, _SIG, generate=lambda **kw: "   ") == \
        "deterministic fallback"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_why.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: module ... has no attribute 'explain_play'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/services/negotiation_advice/advisor.py`:

```python
_WHY_RULES = (
    "Explain in ONE short sentence why this negotiation play suits this "
    "supplier, using ONLY the facts given. Do not invent figures, terms, "
    "dates or supplier behaviour. Do not suggest a different play."
)


def explain_play(play: dict, signals: dict, *, generate=None) -> str:
    """One grounded sentence on why this play fits, else the fixed rationale.

    The model phrases the reason; it never selects or reorders plays. A failure
    degrades to the deterministic rationale — advice is useful without prose.
    """
    fallback = str(play.get("rationale") or "")
    evidence = "; ".join(
        f"{e.get('label')}: {e.get('value')}" for e in (play.get("evidence") or [])
    )
    prompt = (
        f"{_WHY_RULES}\n\n"
        f"Supplier: {signals.get('supplier_name') or 'unknown'}\n"
        f"Lever: {play.get('lever')}\n"
        f"Play: {play.get('play')}\n"
        f"Evidence: {evidence or 'none'}\n"
        f"Deal value: {signals.get('deal_value')} "
        f"{signals.get('currency') or ''}\n"
    )
    try:
        if generate is None:
            from src.services.deal_summary import _SUMMARY_MODEL
            from src.services.ollama_client import ollama_generate

            def generate(**kw):
                return ollama_generate(kw.pop("prompt"), **kw)

            text = generate(prompt=prompt, model=_SUMMARY_MODEL,
                            temperature=0.0, num_predict=120, timeout=60,
                            retries=1, think=False)
        else:
            text = generate(prompt=prompt, model=None, temperature=0.0,
                            num_predict=120, timeout=60, retries=1,
                            think=False)
    except Exception:
        log.debug("play explanation failed; using deterministic rationale",
                  exc_info=True)
        return fallback
    if not text or not str(text).strip():
        return fallback
    return str(text).strip()
```

Then in `build_advice`, after `plays = apply_states(...)`, attach the prose:

```python
        for p in plays:
            p["why"] = explain_play(p, effective)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `set -a && . ./.env; set +a; .venv/bin/python -m pytest tests/services/negotiation_advice/test_why.py -q -p no:cacheprovider`
Expected: PASS (5 passed)

- [ ] **Step 5: Full regression and live check**

```bash
set -a && . ./.env; set +a
.venv/bin/python -m pytest tests/services/negotiation_advice/ \
  tests/agents/test_negotiation_advisor_agent.py \
  tests/api/test_negotiate_advice_endpoints.py \
  tests/test_negotiation_agent.py tests/test_auto_registry.py \
  tests/test_required_agents_are_registerable.py -q -p no:cacheprovider
```
Expected: PASS with no new failures versus the start of this plan.

Then confirm the whole thing works against a live deal, reusing the Task 6
script — each play should now carry a `why` sentence that mentions only figures
present in its evidence. Read three of them and check no number appears that is
not in the facts. If one does, the prompt is leaking or the model is inventing:
tighten `_WHY_RULES` rather than accepting it.

- [ ] **Step 6: Commit**

```bash
git add src/services/negotiation_advice/advisor.py \
        tests/services/negotiation_advice/test_why.py
git commit -m "feat(negotiation-advice): grounded LLM explanation per play"
```

---

## Deferred, with reasons

Not in this plan, and why:

- **UI wiring.** The frontend is a separate repo (`beyond_procwise_ui`), and the spec covers the backend contract. `plays` is additive on the dashboard payload, so nothing breaks before the UI consumes it.
- **`supplier_performance` beyond `on_time_delivery`.** `bp_supplier` carries no performance history, so a richer performance dict has no source yet. `supplier_performance_dict` is the single place to extend when it does.
- **The three open defects in spec §14** — portfolio summaries (now fixed, commit `d403ec6`), the persona format (same commit), `requirement_service.seed_context()` querying a non-existent `category` column, and the dead `generate_negotiation_strategy` call in `end_to_end_demo.py:42`.
- **`tests/test_deal_summary_api.py`** — 3 pre-existing failures against an obsolete design (the endpoint serves pre-stored summaries; the tests monkeypatch generation onto the router).
- **The `test_summary_agent.py` basename collision** — two files share a module name, so they cannot be collected together. Fix is renaming one.
