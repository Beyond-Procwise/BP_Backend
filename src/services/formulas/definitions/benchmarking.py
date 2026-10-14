"""Benchmark pricing, registered.

Delegates to ``src.services.benchmark.engine``, which is already the best
example of this discipline in the codebase: pure, settings-injected, fail-closed
below its evidence threshold, and pinned to the source workbook to the penny by
``tests/fixtures/benchmark/golden.json``. What the registry adds is the contract,
the version and the audit record --- and moving the golden check to import time,
so a mis-edit can no longer load cleanly and fail only under pytest.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from services.benchmark.engine import _confidence as _bm_confidence
from services.benchmark.engine import compute_benchmark as _compute_benchmark
from services.benchmark.engine import excel_round as _excel_round
from services.benchmark.models import BenchmarkPoint, BenchmarkSettings, QuoteLine

from ..contract import COUNT, GBP, LABEL, MONEY, RECORD, ROWS, Output, Term
from ..registry import GoldenVector, formula

_OWNER = "commercial"
_FROM = date(2026, 9, 5)

_FIXTURE = Path(__file__).resolve().parents[4] / "tests/fixtures/benchmark/golden.json"
_G = json.loads(_FIXTURE.read_text()) if _FIXTURE.is_file() else {}
_QUOTE0 = _G.get("quotes", [{}])[0]


@formula(
    "benchmark.adjusted_price",
    version="1.0.0",
    owner=_OWNER,
    purpose="Fair benchmark unit price for one quote line, and the variance against it",
    effective_from=_FROM,
    inputs=[
        Term("quote", RECORD, "the quote line being benchmarked"),
        Term("points", ROWS, "candidate historical price points"),
        Term("location_index_table", RECORD, "location -> cost index", required=False),
        Term("index_table", RECORD, "index_id -> current index value", required=False),
        Term("settings", RECORD, "BenchmarkSettings overrides", required=False),
    ],
    output=Output("BenchmarkResult", GBP,
                  "adjusted benchmark, the five factors, variance and evidence"),
    notes=(
        "Rounding is contractual: every intermediate is rounded exactly where the "
        "prototype rounds, Excel-style (half away from zero). Never substitute "
        "Python's round() here. Below `min_data_points` matching points the engine "
        "returns a gated result with every computed field None, which `evaluate` "
        "reports as a value (not UNASSESSED) because the gate is a MEASURED "
        "outcome -- 'we looked and there is not enough evidence' -- rather than a "
        "contract failure."
    ),
    golden=[
        GoldenVector(
            inputs={
                "quote": _QUOTE0,
                "points": _G.get("points", []),
                "location_index_table": _G.get("location_index_table", {}),
                "index_table": _G.get("index_table", {}),
                "settings": None,
            },
            expected={
                "final_benchmark": 888.98,
                "combined_factor": 1.0159,
                "confidence": "HIGH",
                "gated": False,
                "n_total": 12,
            },
            note="golden.json quote 0 -- penny parity with Benchmark Calculations.xlsx",
        ),
        GoldenVector(
            inputs={"quote": _QUOTE0, "points": [], "location_index_table": {},
                    "index_table": {}, "settings": None},
            expected={"gated": True, "confidence": "No Data", "final_benchmark": None},
            note="fail-closed with no evidence: no benchmark rather than a weak one",
        ),
    ],
)
def adjusted_price(quote, points, location_index_table=None, index_table=None,
                   settings=None):
    q = quote if isinstance(quote, QuoteLine) else QuoteLine(**quote)
    pts = [p if isinstance(p, BenchmarkPoint) else BenchmarkPoint(**p) for p in points]
    cfg = settings
    if cfg is None:
        cfg = BenchmarkSettings()
    elif not isinstance(cfg, BenchmarkSettings):
        cfg = BenchmarkSettings(**cfg)
    return _compute_benchmark(q, pts, location_index_table or {}, index_table or {}, cfg)


@formula(
    "benchmark.evidence_confidence",
    version="1.0.0",
    owner=_OWNER,
    purpose="Confidence label for a benchmark, from how many comparable points backed it",
    effective_from=_FROM,
    inputs=[
        Term("n_total", COUNT, "matching benchmark points", minimum=0),
        Term("min_points", COUNT, "evidence threshold below which nothing is published",
             minimum=0),
    ],
    output=Output("str", LABEL, "No Data | Insufficient | LOW | MEDIUM | HIGH"),
    notes=(
        "The 10 / 6 cuts are the one thing in this module NOT on BenchmarkSettings "
        "(gap report D-14). Registered separately so the inconsistency is visible "
        "and so a later move onto settings is a version bump rather than a silent edit."
    ),
    golden=[
        GoldenVector(inputs={"n_total": 0, "min_points": 3}, expected="No Data"),
        GoldenVector(inputs={"n_total": 2, "min_points": 3}, expected="Insufficient"),
        GoldenVector(inputs={"n_total": 4, "min_points": 3}, expected="LOW"),
        GoldenVector(inputs={"n_total": 6, "min_points": 3}, expected="MEDIUM"),
        GoldenVector(inputs={"n_total": 10, "min_points": 3}, expected="HIGH"),
    ],
)
def evidence_confidence(n_total: int, min_points: int) -> str:
    return _bm_confidence(int(n_total), int(min_points))


@formula(
    "benchmark.excel_round",
    version="1.0.0",
    owner=_OWNER,
    purpose="Excel ROUND semantics: half away from zero",
    effective_from=_FROM,
    inputs=[
        Term("value", MONEY, "the number to round", required=True),
        Term("digits", COUNT, "decimal places", minimum=0, maximum=10),
    ],
    output=Output("float", MONEY, "the rounded value"),
    notes=(
        "Python's round() is half-to-EVEN and drifts from the workbook on ties: "
        "-172.5 rounds to -172 rather than Excel's -173. This is contractual, "
        "not stylistic."
    ),
    golden=[
        GoldenVector(inputs={"value": -172.5, "digits": 0}, expected=-173.0,
                     note="the exact tie Python's round() gets wrong"),
        GoldenVector(inputs={"value": 1.005, "digits": 2}, expected=1.01),
    ],
)
def excel_round(value: float, digits: int) -> float:
    return _excel_round(float(value), int(digits))
