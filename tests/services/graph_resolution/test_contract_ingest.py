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
