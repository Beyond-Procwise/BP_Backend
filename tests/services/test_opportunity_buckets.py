"""A unit price and a volume are one piece of spend; the savings found on it are
levers on that one thing, not a list of separate opportunities."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services import opportunity_buckets as buckets


def _row(oid, lever, saving, stage="identified", supplier="SUP-1", item="ITM-1", **extra):
    return {"opportunity_id": oid, "detector_type": lever, "supplier_id": supplier,
            "supplier_name": "Copperleaf", "item_id": item, "item_description": "Notebooks",
            "financial_impact_gbp": saving, "stage": stage, "unit_price": 12.0,
            "quantity": 40, "calculation_details": {}, "source_records": ["PO-1"],
            "detected_on": None, "review_requested_at": None, "review_detail": None,
            "rejection_reason": None, **extra}


def test_two_levers_on_one_priced_item_are_one_bucket_not_two_opportunities():
    out = buckets.group([_row("a", "Price Benchmark Variance", 300.0),
                         _row("b", "Volume Consolidation", 120.0)])

    assert out["individual"] == []
    (bucket,) = out["buckets"]
    assert bucket["supplier"] == "Copperleaf" and bucket["item"] == "Notebooks"
    assert (bucket["unitPrice"], bucket["quantity"], bucket["spend"]) == (12.0, 40.0, 480.0)
    assert [l["lever"] for l in bucket["levers"]] == ["Price Benchmark Variance",
                                                      "Volume Consolidation"]


def test_the_bucket_saving_is_the_largest_lever_never_the_sum():
    """Both levers act on the same £480 of spend; adding them counts money twice."""
    (bucket,) = buckets.group([_row("a", "Price Benchmark Variance", 300.0),
                               _row("b", "Volume Consolidation", 120.0)])["buckets"]

    assert bucket["potentialSaving"] == 300.0
    assert "not added together" in bucket["savingBasis"]


def test_a_rejected_lever_stays_visible_but_does_not_count():
    (bucket,) = buckets.group([_row("a", "Price Benchmark Variance", 300.0, stage="rejected",
                                    rejection_reason="benchmark is for a different spec"),
                               _row("b", "Volume Consolidation", 120.0)])["buckets"]

    assert bucket["potentialSaving"] == 120.0
    assert (bucket["leverCount"], bucket["openLeverCount"]) == (2, 1)
    rejected = next(l for l in bucket["levers"] if l["stage"] == "rejected")
    assert rejected["rejectionReason"] == "benchmark is for a different spec"


def test_new_information_on_a_rejected_lever_shows_on_the_bucket():
    asked = datetime(2026, 10, 13, tzinfo=timezone.utc)
    changed = {"new_records": ["INV-2"], "impact_before": 300.0, "impact_after": 410.0}
    (bucket,) = buckets.group([_row("a", "Price Benchmark Variance", 410.0, stage="rejected",
                                    review_requested_at=asked, review_detail=changed),
                               _row("b", "Volume Consolidation", 120.0)])["buckets"]

    assert bucket["reviewRequested"] is True
    lever = next(l for l in bucket["levers"] if l["reviewRequested"])
    assert lever["whatChanged"] == changed


def test_a_settled_review_no_longer_shows_what_changed():
    (bucket,) = buckets.group([_row("a", "Price Benchmark Variance", 410.0, stage="rejected",
                                    review_detail={"decision": "keep_rejected"})])["buckets"]

    assert bucket["reviewRequested"] is False
    assert bucket["levers"][0]["whatChanged"] is None


def test_without_a_unit_price_and_volume_there_is_no_bucket():
    """A duplicate invoice has a supplier and an 'item' (the invoice), but no price
    and volume -- nothing makes it a lever on a piece of spend."""
    out = buckets.group([_row("d", "Duplicate Invoice Recovery", 900.0, item="INV-9-DUP",
                              unit_price=None, quantity=None)])

    assert out["buckets"] == []
    assert [o["opportunityId"] for o in out["individual"]] == ["d"]


def test_the_price_and_volume_can_come_from_the_working_when_the_columns_are_empty():
    """upsert does not fill unit_price/quantity; a backfill does, later."""
    (bucket,) = buckets.group([_row("a", "Price Benchmark Variance", 300.0, unit_price=None,
                                    quantity=None, calculation_details={
                                        "actual_price": 15.0, "quantity": 10})])["buckets"]

    assert (bucket["unitPrice"], bucket["quantity"], bucket["spend"]) == (15.0, 10.0, 150.0)


def test_one_priced_lever_brings_the_item_s_other_levers_into_its_bucket():
    out = buckets.group([_row("a", "Price Benchmark Variance", 300.0),
                         _row("b", "Supplier Risk", 50.0, unit_price=None, quantity=None)])

    assert out["individual"] == []
    assert out["buckets"][0]["leverCount"] == 2


def test_different_items_are_different_buckets_and_no_supplier_means_no_bucket():
    out = buckets.group([_row("a", "Price Benchmark Variance", 300.0, item="ITM-1"),
                         _row("b", "Price Benchmark Variance", 500.0, item="ITM-2"),
                         _row("c", "Invoice Overbilling", 70.0, supplier=None, item=None)])

    assert [b["itemId"] for b in out["buckets"]] == ["ITM-2", "ITM-1"]   # largest first
    assert [o["opportunityId"] for o in out["individual"]] == ["c"]


def test_a_bucket_with_no_open_lever_claims_no_saving():
    (bucket,) = buckets.group([_row("a", "Price Benchmark Variance", 300.0,
                                    stage="rejected")])["buckets"]

    assert bucket["potentialSaving"] is None
