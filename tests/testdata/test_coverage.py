import pytest

from scripts.testdata.coverage import (
    BACKUP,
    OUTPUT,
    REFERENCE,
    SEED,
    UNCLEAR,
    buckets,
    classify,
)


@pytest.mark.parametrize(
    "table",
    [
        "bp_invoice_raw_bkp", "bp_invoice_raw_bkp_june12", "supplier_bkp",
        "invoice_old", "invoice1", "po1", "purchase_order_new1",
        "invoice_agent_260925", "po_line_items_agent_180925",
        "extraction_review_queue_bkp_27_04", "bp_invoice_bkp_may_4th",
        "invoice_line_items_agent_stage", "supplier_test", "category_denorm",
    ],
)
def test_dated_and_duplicated_copies_are_backups(table):
    assert classify(table).bucket == BACKUP


@pytest.mark.parametrize(
    "table",
    [
        "bp_supplier_ranking", "bp_quote_evaluation", "bp_decision", "bp_summary",
        "bp_opportunity", "bp_detection_finding", "bp_deal", "bp_deal_document_map",
        "bp_extraction_telemetry", "process_monitor", "workflow_events",
        "negotiation_sessions", "bp_approval", "action",
    ],
)
def test_product_conclusions_are_outputs(table):
    assert classify(table).bucket == OUTPUT


@pytest.mark.parametrize(
    "table",
    [
        "bp_fx_rates", "bp_policy", "bp_prompt", "bp_category", "category",
        "category_mapping", "procurement_patterns", "bp_style_profile",
        "bp_mailbox_binding", "bp_products",
    ],
)
def test_configuration_is_copied_verbatim(table):
    assert classify(table).bucket == REFERENCE


@pytest.mark.parametrize(
    "table",
    [
        "bp_supplier", "supplier", "business_unit", "cost_centre", "item",
        "bp_requirement", "bp_invoice_trgt", "bp_quote_line_items_stg",
        "bp_purchase_order_raw", "esg_data", "supplier_risk_scores",
        "bp_tprm_supplier", "contact", "sup_mapping",
    ],
)
def test_business_data_is_seeded(table):
    assert classify(table).bucket == SEED


def test_a_backup_of_a_seeded_table_is_still_a_backup():
    """Precedence matters: bp_invoice_trgt is seeded, its June copy is not."""
    assert classify("bp_invoice_trgt").bucket == SEED
    assert classify("bp_invoice_trgt_bkp").bucket == BACKUP
    assert classify("bp_quote_trgt_june12").bucket == BACKUP


def test_every_table_lands_in_exactly_one_bucket():
    tables = ["bp_supplier", "bp_deal", "bp_fx_rates", "supplier_bkp", "wat_is_dit"]
    grouped = buckets(tables)
    assert sum(len(v) for v in grouped.values()) == len(tables)
    assert grouped[UNCLEAR] == ["wat_is_dit"]


def test_classification_reasons_are_populated():
    for table in ("bp_supplier", "bp_deal", "bp_fx_rates", "supplier_bkp"):
        assert classify(table).reason


@pytest.mark.integration
def test_no_live_table_is_left_unclassified():
    """A new table appearing in live must be a decision, not a silent skip."""
    from scripts.testdata.db import connect

    unclear: list[str] = []
    for dbname in ("bp_sqldb", "uicanvas"):
        conn = connect(dbname)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "select table_name from information_schema.tables "
                    "where table_schema = 'proc' and table_type = 'BASE TABLE'"
                )
                for (table,) in cur.fetchall():
                    if classify(table).bucket == UNCLEAR:
                        unclear.append(f"{dbname}.{table}")
        finally:
            conn.close()
    assert not unclear, (
        "these live tables have no classification; decide explicitly in "
        "scripts/testdata/coverage.py:\n  " + "\n  ".join(sorted(unclear))
    )
