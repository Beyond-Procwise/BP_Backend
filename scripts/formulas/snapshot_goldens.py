"""Snapshot the CURRENT output of every formula being migrated.

Run before the migration; the printed values become the golden vectors. Nothing
here computes anything itself --- it calls the live implementations, so the
vectors are a photograph of today's behaviour, not a restatement of what the
code is believed to do.

    ./.venv/bin/python -m scripts.formulas.snapshot_goldens
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

OUT = {}


def snap(key, fn, *a, **kw):
    try:
        OUT[key] = fn(*a, **kw)
    except Exception as exc:  # noqa: BLE001
        OUT[key] = {"__error__": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------- linking
from src.services import linking_engine as le  # noqa: E402

INV = {
    "invoice_id": "INV-9001", "po_id": "PO-4001", "supplier_id": "SUP-100",
    "converted_amount_usd": 12500.00, "currency": "GBP",
    "invoice_date": "2025-04-10", "country": "GB", "region": "London",
}
PO = {
    "po_id": "PO-4001", "supplier_id": "SUP-100",
    "converted_amount_usd": 12500.00, "currency": "GBP",
    "order_date": "2025-03-01", "ship_to_country": "GB", "delivery_region": "London",
}
INV_LINES = [
    {"item_description": "Dell Latitude 5540 laptop", "quantity": 10, "unit_price": 780.0},
    {"item_description": "Docking station USB-C", "quantity": 10, "unit_price": 95.0},
]
PO_LINES = [
    {"item_description": "Dell Latitude 5540 laptop", "quantity": 10, "unit_price": 780.0},
    {"item_description": "Docking station USB-C", "quantity": 10, "unit_price": 95.0},
]

snap("score_link.invoice_po.perfect", le.score_link, INV, PO, "invoice_po", INV_LINES, PO_LINES)

INV_MISMATCH = dict(INV, converted_amount_usd=19000.00, supplier_id="SUP-200")
snap("score_link.invoice_po.conflict", le.score_link, INV_MISMATCH, PO, "invoice_po",
     INV_LINES, PO_LINES)

snap("dampen.1", le._dampen, 1)
snap("dampen.2", le._dampen, 2)
snap("dampen.5", le._dampen, 5)
snap("line_pair.identical", le._line_pair_score, INV_LINES[0], PO_LINES[0])
snap("line_pair.desc_only", le._line_pair_score,
     {"item_description": "blue widget large"}, {"item_description": "blue widget"})
snap("line_composite.identical", le.cmp_line_composite, INV_LINES, PO_LINES)
snap("line_composite.extra_source", le.cmp_line_composite,
     INV_LINES + [{"item_description": "carriage", "quantity": 1, "unit_price": 40.0}], PO_LINES)
snap("line_composite.empty", le.cmp_line_composite, [], PO_LINES)
snap("numeric_tol.equal", le.cmp_numeric_tol, 12500.0, 12500.0)
snap("numeric_tol.drift_3pct", le.cmp_numeric_tol, 12875.0, 12500.0)
snap("numeric_tol.drift_20pct", le.cmp_numeric_tol, 15000.0, 12500.0)
snap("numeric_tol.missing", le.cmp_numeric_tol, None, 12500.0)
snap("temporal.ok", le.cmp_temporal, "2025-04-10", "2025-03-01", None)
snap("temporal.before_parent", le.cmp_temporal, "2025-02-01", "2025-03-01", None)
snap("temporal.old", le.cmp_temporal, "2027-06-01", "2025-03-01", None)
snap("temporal.missing", le.cmp_temporal, None, "2025-03-01", None)
snap("location.match", le.cmp_location, "GB", "London", "GB", "London")
snap("location.mismatch", le.cmp_location, "GB", "London", "DE", "Berlin")
snap("band.95", le._band, 95.0)
snap("band.85", le._band, 85.0)
snap("band.70", le._band, 70.0)
snap("band.50", le._band, 50.0)
snap("band.10", le._band, 10.0)

# --------------------------------------------------------------- rivalry
from src.services import requirement_similarity as rs  # noqa: E402

BID_A = {"quote_id": "Q-1", "supplier_id": "SUP-1", "buyer_id": "BUY-1",
         "currency": "GBP", "converted_amount_usd": 100000.0, "quote_date": "2025-05-01"}
BID_B = {"quote_id": "Q-2", "supplier_id": "SUP-2", "buyer_id": "BUY-1",
         "currency": "GBP", "converted_amount_usd": 107000.0, "quote_date": "2025-05-02"}
LINES_A = [{"item_description": "London Heathrow to Edinburgh full truckload FTL freight",
            "quantity": 120, "unit_price": 833.33}]
LINES_B = [{"item_description": "London Heathrow to Edinburgh full truckload FTL freight",
            "quantity": 120, "unit_price": 891.67}]
snap("rivalry.same_event", rs.rivalry_score, BID_A, BID_B, LINES_A, LINES_B)

LINES_C = [{"item_description": "Managed IT endpoint service desk 24x7 3000 users",
            "quantity": 3000, "unit_price": 210.0}]
BID_C = dict(BID_B, quote_id="Q-3", converted_amount_usd=630000.0)
snap("rivalry.cross_event", rs.rivalry_score, BID_A, BID_C, LINES_A, LINES_C)
snap("desc_overlap.same", rs.cmp_desc_overlap, LINES_A, LINES_B)
snap("desc_overlap.cross", rs.cmp_desc_overlap, LINES_A, LINES_C)
snap("volume.same", rs.cmp_volume, LINES_A, LINES_B)
snap("volume.missing", rs.cmp_volume, [{"item_description": "x"}], LINES_B)
snap("price_prox.close", rs.cmp_price_prox, BID_A, BID_B)
snap("price_prox.far", rs.cmp_price_prox, BID_A, BID_C)

# --------------------------------------------------------------- duplicates
from src.services import duplicate_invoice_detector as did  # noqa: E402

D1 = {"invoice_id": "I1", "invoice_ref": "INV-2025-0455", "supplier_name": "Acme Ltd",
      "supplier_id": "SUP-1", "total_amount": 4500.00, "currency": "GBP",
      "invoice_date": date(2025, 6, 1), "po_id": "PO-9", "line_items": []}
D2 = dict(D1, invoice_id="I2", invoice_ref="INV-2025-0456", invoice_date=date(2025, 6, 3))
snap("dup.near", did.score_pair, D1, D2)
D3 = dict(D1, invoice_id="I3", invoice_ref="XYZ-777", total_amount=99.0,
          invoice_date=date(2024, 1, 1), supplier_name="Other plc", supplier_id="SUP-2")
snap("dup.far", did.score_pair, D1, D3)
snap("ref_prox.same", did.cmp_ref_prox, "INV-2025-0455", "INV-2025-0455")
snap("ref_prox.one_edit", did.cmp_ref_prox, "INV-2025-0455", "INV-2025-0456")
snap("ref_prox.different", did.cmp_ref_prox, "INV-2025-0455", "ZZZ-1")
snap("date_prox.same", did.cmp_date_prox, date(2025, 6, 1), date(2025, 6, 1))
snap("date_prox.2d", did.cmp_date_prox, date(2025, 6, 1), date(2025, 6, 3))
snap("date_prox.far", did.cmp_date_prox, date(2025, 6, 1), date(2024, 1, 1))

# --------------------------------------------------------------- clustering
from src.services import deal_clustering as dc  # noqa: E402

BIDS = [BID_A, BID_B]
LINES = {"Q-1": LINES_A, "Q-2": LINES_B}
_m = dc.pairwise_matrix(BIDS, LINES)
snap("cluster_confidence.pair", dc.cluster_confidence, BIDS, _m)
snap("cluster_confidence.singleton", dc.cluster_confidence, [BID_A], _m)
snap("awarded_po.none", dc.awarded_po_scored, BID_A, [PO], {"PO-4001": PO_LINES}, LINES_A)

# --------------------------------------------------------------- benchmark
from src.services.benchmark.engine import compute_benchmark, excel_round, _confidence  # noqa: E402
from src.services.benchmark.models import BenchmarkPoint, BenchmarkSettings, QuoteLine  # noqa: E402

GOLDEN = json.loads((Path(__file__).resolve().parents[2]
                     / "tests/fixtures/benchmark/golden.json").read_text())
_pts = [BenchmarkPoint(**p) for p in GOLDEN["points"]]
_q = QuoteLine(**GOLDEN["quotes"][0]) if GOLDEN.get("quotes") else None
if _q is not None:
    _r = compute_benchmark(_q, _pts, GOLDEN["location_index_table"], GOLDEN["index_table"],
                           BenchmarkSettings())
    OUT["benchmark.golden0"] = {
        "quote": GOLDEN["quotes"][0],
        "final_benchmark": _r.final_benchmark,
        "selected_benchmark": _r.selected_benchmark,
        "combined_factor": _r.combined_factor,
        "unit_variance_gbp": _r.unit_variance_gbp,
        "unit_variance_pct": _r.unit_variance_pct,
        "total_cost_gap": _r.total_cost_gap,
        "confidence": _r.confidence,
        "gated": _r.gated,
        "n_total": _r.n_total,
    }
snap("excel_round.half_away", excel_round, -172.5, 0)
snap("excel_round.2dp", excel_round, 1.005, 2)
snap("bm_confidence.0", _confidence, 0, 3)
snap("bm_confidence.2of3", _confidence, 2, 3)
snap("bm_confidence.6", _confidence, 6, 3)
snap("bm_confidence.10", _confidence, 10, 3)
snap("bm_confidence.4", _confidence, 4, 3)

# --------------------------------------------------------------- price outlier
from src.services.price_outlier.rule import OutlierSettings, assess  # noqa: E402

_s = OutlierSettings()
PEERS = [100.0, 102.0, 98.0, 101.0, 99.0, 100.5]
snap("outlier.normal", lambda: assess(101.0, PEERS, _s).__dict__)
snap("outlier.extreme", lambda: assess(1200.0, PEERS, _s).__dict__)
snap("outlier.material_only", lambda: assess(310.0, PEERS, _s).__dict__)
snap("outlier.too_few", lambda: assess(1200.0, [100.0, 101.0], _s).__dict__)
snap("outlier.flat_peers", lambda: assess(1200.0, [100.0] * 6, _s).__dict__)

# --------------------------------------------------------------- risk
from src.services.risk_intelligence_service import PredictiveRiskModel  # noqa: E402
from models.risk_intelligence import SupplierRiskSignal  # noqa: E402

_model = PredictiveRiskModel()
snap("risk.no_signals_full_metrics", _model.evaluate,
     {"on_time_delivery_rate": 0.92, "quality_score": 0.88,
      "anomaly_index": 0.1, "resilience_index": 0.7}, [])
snap("risk.no_signals_no_metrics", _model.evaluate, {}, [])

# --------------------------------------------------------------- negotiation
from src.agents.negotiation_agent import (  # noqa: E402
    NegotiationContext, SupplierSignals, compute_decision, plan_counter,
)

for label, payload, prev, msg in [
    ("r1_wide", {"current_offer": 100.0, "target_price": 80.0, "round": 1}, None, ""),
    ("r1_narrow", {"current_offer": 100.0, "target_price": 95.0, "round": 1}, None, ""),
    ("r2_wide", {"current_offer": 100.0, "target_price": 80.0, "round": 2}, 105.0, ""),
    ("r2_narrow", {"current_offer": 100.0, "target_price": 95.0, "round": 2}, 105.0, ""),
    ("r3", {"current_offer": 100.0, "target_price": 80.0, "round": 3}, 102.0, ""),
    ("over_max", {"current_offer": 100.0, "target_price": 80.0, "round": 4}, None, ""),
    ("at_target", {"current_offer": 78.0, "target_price": 80.0, "round": 1}, None, ""),
    ("final_ok", {"current_offer": 79.0, "target_price": 80.0, "round": 2}, None,
     "this is our final offer"),
    ("final_high", {"current_offer": 120.0, "target_price": 80.0, "round": 2}, None,
     "this is our final offer"),
    ("invalid", {"current_offer": 0.0, "target_price": 80.0, "round": 1}, None, ""),
]:
    snap(f"counter.{label}", compute_decision, payload, msg, prev)

from src.services.negotiation_advice import classification as cls  # noqa: E402
from src.services.negotiation_advice import grounding as gnd  # noqa: E402
from src.services.negotiation_advice import ranking as rnk  # noqa: E402

snap("classify.leverage", cls.classify,
     {"deal_value": 250000.0, "alternative_supplier_count": 150, "risk_score": 40.0,
      "is_preferred": False, "price_variance_pct": 8.0})
snap("classify.bottleneck", cls.classify,
     {"deal_value": 4000.0, "alternative_supplier_count": 12, "risk_score": 70.0,
      "is_preferred": False, "price_variance_pct": None})
snap("classify.strategic", cls.classify,
     {"deal_value": 250000.0, "alternative_supplier_count": 12, "is_preferred": True})
snap("classify.transactional", cls.classify,
     {"deal_value": 4000.0, "alternative_supplier_count": 150})
snap("classify.indeterminate", cls.classify, {"deal_value": None,
                                              "alternative_supplier_count": None})
snap("classify_conf.far", cls._confidence, 250000.0, 98175.0)
snap("classify_conf.at_bar", cls._confidence, 98175.0, 98175.0)
snap("classify_conf.zero_bar", cls._confidence, 10.0, 0.0)

snap("policy_align.required", rnk._score_policy_alignment, "Commercial",
     {"required": {"Commercial"}})
snap("policy_align.restricted", rnk._score_policy_alignment, "Commercial",
     {"restricted": {"Commercial"}, "preferred": {"Commercial"}})
snap("policy_align.none", rnk._score_policy_alignment, "Commercial", {})
snap("perf_score.late", rnk._score_supplier_performance, "Operational",
     {"on_time_delivery": 0.8})
snap("perf_score.strong", rnk._score_supplier_performance, "Operational",
     {"on_time_delivery": 0.99})
snap("perf_score.empty", rnk._score_supplier_performance, "Operational", {})
snap("market_score.risk", rnk._score_market_context, "Risk", {"supply_risk": "elevated"})
snap("market_score.empty", rnk._score_market_context, "Commercial", {})

snap("readiness.tension_ready", gnd.assess,
     {"play": "Leverage competitor quotes to pressure pricing", "score": 1.0},
     {"quote_supplier_count": 3, "alternative_supplier_count": 5})
snap("readiness.tension_sole", gnd.assess,
     {"play": "Leverage competitor quotes to pressure pricing", "score": 1.0},
     {"quote_supplier_count": 1, "alternative_supplier_count": 1})
snap("readiness.overbilling", gnd.assess,
     {"play": "Request a refund for non-compliant charges", "score": 1.0},
     {"invoice_total": 1200.0, "po_total": 1000.0})
snap("readiness.no_family", gnd.assess, {"play": "Hold a supplier day", "score": 1.0}, {})

# --------------------------------------------------------------- supplier ranking
from src.agents.supplier_ranking_agent import _normalize_days_to_score  # noqa: E402

for d in (0, 30, 45, 90, 120, None):
    snap(f"payterms.{d}", _normalize_days_to_score, d)

# --------------------------------------------------------------- extraction
from src.services.facts.arithmetic import check_line_arithmetic  # noqa: E402
from src.services.extraction.three_way_match import _agrees  # noqa: E402

snap("agrees.exact", _agrees, 100.00, 100.00)
snap("agrees.penny", _agrees, 100.00, 100.005)
snap("agrees.off", _agrees, 100.00, 105.00)
snap("line_arith.consistent", lambda: str(check_line_arithmetic(10, 5.0, 50.0)))
snap("line_arith.inconsistent", lambda: str(check_line_arithmetic(10, 5.0, 90.0)))
snap("line_arith.qty_one", lambda: str(check_line_arithmetic(1, 5.0, 5.0)))
snap("line_arith.missing", lambda: str(check_line_arithmetic(None, 5.0, 50.0)))

# --------------------------------------------------------------- deal metrics
from src.services.deal_analysis_service import _pct_change, _weighted_unit_price  # noqa: E402

DOCS = [{"line_items": [{"quantity": 10, "unit_price": 100.0},
                        {"quantity": 5, "unit_price": 200.0}]}]
snap("wup.mixed", _weighted_unit_price, DOCS)
snap("wup.empty", _weighted_unit_price, [])
snap("pct.up", _pct_change, 110.0, 100.0)
snap("pct.down", _pct_change, 90.0, 100.0)
snap("pct.zero_base", _pct_change, 90.0, 0.0)

# --------------------------------------------------------------- requirements
from src.services.requirement_service import evaluate_completeness  # noqa: E402

snap("req.partial", evaluate_completeness,
     {"category": "IT", "quantity": None, "budget": "", "need_by": "2025-09-01"},
     ["category", "quantity", "budget", "need_by"])
snap("req.full", evaluate_completeness, {"a": 1, "b": 2}, ["a", "b"])

# --------------------------------------------------------------- opportunity
from src.agents.opportunity_miner_agent import OpportunityMinerAgent  # noqa: E402

_norm = OpportunityMinerAgent._normalise_risk_score
snap("risknorm.0_5", _norm, None, 0.5)
snap("risknorm.75", _norm, None, 75.0)
snap("risknorm.neg", _norm, None, -3.0)
snap("risknorm.bad", _norm, None, "abc")
snap("risknorm.over100", _norm, None, 250.0)


def _default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


print(json.dumps(OUT, indent=1, default=_default, sort_keys=True))
