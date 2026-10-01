"""An agent must receive the rows its task needs, and not the data dictionary.

Build spec principle 5. Before this, build_manifest returned every table's full
column list and synonym map to every agent on every step, and one agent put the
whole bundle in its prompt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from services import agent_manifest  # noqa: E402
from services.agent_manifest import MANIFEST_TASKS, AgentManifestService  # noqa: E402


class _StubNick:
    policy_engine = None

    def get_db_connection(self):  # pragma: no cover - never called here
        return None


@pytest.fixture()
def service():
    return AgentManifestService(_StubNick())


def _chars(manifest) -> int:
    return len(json.dumps(manifest["knowledge"], default=str))


def test_an_unsliced_manifest_still_works(service):
    """Every existing caller passes no task_id. They must keep working."""
    m = service.build_manifest("data_extraction")
    assert set(m) == {"task", "policies", "knowledge"}
    assert m["knowledge"]["tables"]


def test_an_unsliced_manifest_is_the_whole_bundle_exactly_as_before(service):
    """Omitting task_id must change nothing: every profile, untouched, and all
    six relationships."""
    k = service.build_manifest("data_extraction")["knowledge"]
    assert k["tables"] == service._table_profiles
    assert len(k["tables"]) == 10
    assert len(k["relationships"]) == len(agent_manifest._PROC_RELATIONSHIPS) == 6
    assert k["loaded"]["task_id"] is None
    assert k["loaded"]["tables"] == 10


def test_a_sliced_manifest_carries_fewer_tables_than_the_unsliced_one(service):
    full = service.build_manifest("data_extraction")
    sliced = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")
    assert 0 < len(sliced["knowledge"]["tables"]) < len(full["knowledge"]["tables"]), (
        "a task slice that carries every table (or none) is not a slice"
    )


def test_a_slice_is_measurably_smaller_in_bytes(service):
    """Principle 5 with numbers: under a quarter of the whole bundle."""
    full = _chars(service.build_manifest("data_extraction"))
    sliced = _chars(service.build_manifest("data_extraction", task_id="DOC_CLASSIFY"))
    assert sliced * 4 < full, f"slice {sliced} chars vs whole {full} chars"


def test_a_slice_carries_only_the_tables_its_task_declares(service):
    declared = set(MANIFEST_TASKS["DOC_CLASSIFY"]["tables"])
    sliced = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")
    assert set(sliced["knowledge"]["tables"]) == declared


def test_a_slice_carries_only_the_columns_its_task_declares(service):
    wanted = set(MANIFEST_TASKS["DOC_CLASSIFY"]["fields"])
    k = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")["knowledge"]
    carried = {c for p in k["tables"].values() for c in p["columns"]}
    assert carried and carried <= wanted
    # a column the task did not declare is on the real profile and must be gone
    assert "risk_score" in service._table_profiles["proc.bp_supplier"]["columns"]
    assert "risk_score" not in carried


def test_a_slice_drops_relationships_that_do_not_touch_its_tables(service):
    k = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")["knowledge"]
    assert 0 < len(k["relationships"]) < len(agent_manifest._PROC_RELATIONSHIPS)
    for rel in k["relationships"]:
        assert any(
            rel["from"].startswith(t) or rel["to"].startswith(t)
            for t in MANIFEST_TASKS["DOC_CLASSIFY"]["tables"]
        )


def test_a_slice_reports_what_it_loaded(service):
    """Build spec §3.7: log the rows loaded per call."""
    sliced = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")
    loaded = sliced["knowledge"]["loaded"]
    assert loaded["task_id"] == "DOC_CLASSIFY"
    assert loaded["tables"] == len(sliced["knowledge"]["tables"])
    assert loaded["rows"] == sum(len(p["columns"]) for p in sliced["knowledge"]["tables"].values())
    assert loaded["rows"] > 0


def test_a_slice_is_bounded_by_max_rows(service):
    sliced = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")
    assert sliced["knowledge"]["loaded"]["rows"] <= MANIFEST_TASKS["DOC_CLASSIFY"]["max_rows"]


def test_max_rows_is_enforced_when_the_slice_would_exceed_it(service, monkeypatch):
    """The shipped budget is never reached by the real profiles, so a bound
    that is only asserted against it proves nothing. Squeeze the budget."""
    tight = dict(MANIFEST_TASKS["DOC_CLASSIFY"], max_rows=3)
    monkeypatch.setitem(MANIFEST_TASKS, "DOC_CLASSIFY", tight)
    k = service.build_manifest("data_extraction", task_id="DOC_CLASSIFY")["knowledge"]
    assert k["loaded"]["rows"] == 3
    assert sum(len(p["columns"]) for p in k["tables"].values()) == 3


def test_an_unknown_task_id_refuses_rather_than_silently_loading_everything(service):
    with pytest.raises(KeyError):
        service.build_manifest("data_extraction", task_id="DOC_CLASSIFYY")


def test_every_declared_task_names_tables_and_fields_that_exist(service):
    profiles = service._table_profiles
    for task_id, spec in MANIFEST_TASKS.items():
        unknown = set(spec["tables"]) - set(profiles)
        assert not unknown, f"{task_id} names tables the manifest has no profile for: {unknown}"
        available = {c for t in spec["tables"] for c in profiles[t]["columns"]}
        ghosts = set(spec["fields"]) - available
        assert not ghosts, f"{task_id} names fields no declared table carries: {ghosts}"


def test_the_negotiation_prompt_no_longer_carries_the_knowledge_bundle():
    """Behaviour, not source: build the prompt values from a context whose
    knowledge bundle holds a marker, and look at what comes out."""
    from agents.base_agent import AgentContext
    from agents.negotiation_agent import NegotiationAgent

    marker = "ZZ_DATA_DICTIONARY_MARKER"
    knowledge = {
        "tables": {"proc.bp_contracts": {"columns": [marker], "synonyms": {marker: [marker]}}},
        "relationships": [{"from": marker, "to": marker}],
        "loaded": {"task_id": None, "tables": 1, "rows": 1},
    }
    ctx = AgentContext(
        workflow_id="wf", agent_id="negotiation", user_id=None,
        input_data={}, knowledge_base=knowledge,
    )
    agent = object.__new__(NegotiationAgent)
    values = agent._build_prompt_context(
        context=ctx, header="h", lines=["h", "l"], decision={}, price=None,
        target_price=None, currency=None, round_no=1, supplier=None,
        supplier_snippets=[], supplier_message=None, playbook_context=None,
        signals=None, zopa=None, procurement_summary=None, rag_snippets=None,
    )
    blob = json.dumps(values, default=str)
    assert marker not in blob, "the manifest data dictionary is still in the negotiation prompt"
    assert '"tables": 1' in values["knowledge_loaded"]  # what was loaded stays visible
