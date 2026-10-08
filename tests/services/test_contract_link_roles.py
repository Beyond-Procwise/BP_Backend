# tests/services/test_contract_link_roles.py
"""Pure role/profile lookups: they read only the vocabulary, so they need no database
and run in a default test run."""
from __future__ import annotations

from src.services import contract_links as CL


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
