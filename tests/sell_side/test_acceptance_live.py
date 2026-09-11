"""Spec §9 criteria 2-4 through the REAL parser into the REAL database.
The unit suite proves them against a fake connection; this proves them where
they matter."""
import datetime as dt
from decimal import Decimal as D

from src.services import catalog_import
from tests.sell_side.conftest import SENTINEL, live

pytestmark = live
PROFILE = "livetest_v1"
NOCOST = "livetest_nocost_v1"
BASE = [{"target_column": "distributor_sku", "source_header": "SKU"},
        {"target_column": "item_description", "source_header": "Description"},
        {"target_column": "currency", "source_header": "Ccy"}]


def _profiles(conn, dist):
    catalog_import.save_mapping(conn, mapping_profile=PROFILE, distributor_id=dist,
                                entries=BASE + [{"target_column": "cost_price",
                                                 "source_header": "Cost"}])
    catalog_import.save_mapping(conn, mapping_profile=NOCOST, distributor_id=dist, entries=BASE)


def _feed(tmp_path, name, rows, header="SKU,Description,Ccy,Cost"):
    path = tmp_path / name
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def _import(conn, dist, path, profile=PROFILE):
    return catalog_import.import_catalog(
        distributor_id=dist, feed_name=f"{SENTINEL} {path.name}", mapping_profile=profile,
        price_effective=dt.date.today(), imported_by="sub-livetest",
        file_path=str(path), conn=conn)


def _rows(conn, sql, params):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def test_criterion_2_the_same_file_twice_is_one_receipt_and_no_duplicates(live_db, tmp_path):
    conn, dist = live_db
    _profiles(conn, dist)
    feed = _feed(tmp_path, "a.csv", ["LIVETEST-A1,Widget,GBP,10.00",
                                     "LIVETEST-A2,Gadget,GBP,20.00"])
    first, second = _import(conn, dist, feed), _import(conn, dist, feed)
    assert (first.status, first.rows_loaded) == ("imported", 2)
    assert second.status == "duplicate" and second.source_id == first.source_id
    assert _rows(conn, "SELECT count(*) FROM proc.bp_catalog_source WHERE source_id = %s",
                 (first.source_id,)) == [(1,)]
    assert _rows(conn, "SELECT count(*) FROM proc.bp_catalog_item WHERE distributor_sku "
                 "LIKE 'LIVETEST-A%%'", ()) == [(2,)]


def test_criterion_3_a_repriced_feed_keeps_one_current_row_and_the_old_price(live_db, tmp_path):
    conn, dist = live_db
    _profiles(conn, dist)
    _import(conn, dist, _feed(tmp_path, "march.csv", ["LIVETEST-R1,Widget,GBP,10.00"]))
    res = _import(conn, dist, _feed(tmp_path, "april.csv", ["LIVETEST-R1,Widget,GBP,11.00"]))
    assert res.rows_versioned == 1
    rows = _rows(conn, "SELECT cost_price, valid_to IS NULL FROM proc.bp_catalog_item "
                 "WHERE distributor_sku = 'LIVETEST-R1' ORDER BY catalog_item_id", ())
    assert rows == [(D("10.0000"), False), (D("11.0000"), True)]


def test_criterion_4_a_feed_with_no_cost_column_loads_null_cost(live_db, tmp_path):
    conn, dist = live_db
    _profiles(conn, dist)
    res = _import(conn, dist, _feed(tmp_path, "nocost.csv",
                                    ["LIVETEST-N1,Widget,GBP", "LIVETEST-N2,Gadget,GBP"],
                                    header="SKU,Description,Ccy"), profile=NOCOST)
    assert res.rows_loaded == 2
    assert _rows(conn, "SELECT cost_price FROM proc.bp_catalog_item WHERE distributor_sku "
                 "LIKE 'LIVETEST-N%%'", ()) == [(None,), (None,)]


def test_a_bad_row_is_reported_and_the_rest_load(live_db, tmp_path):
    conn, dist = live_db
    _profiles(conn, dist)
    res = _import(conn, dist, _feed(tmp_path, "bad.csv", ["LIVETEST-B1,Widget,GBP,10.00",
                                                          "LIVETEST-B2,Gadget,Pounds,5.00"]))
    assert (res.status, res.rows_loaded, res.rows_rejected) == ("partial", 1, 1)
    assert _rows(conn, "SELECT status, rows_rejected FROM proc.bp_catalog_source "
                 "WHERE source_id = %s", (res.source_id,)) == [("partial", 1)]
