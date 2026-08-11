"""No new module may reach the network directly.

The egress audit found no outbound transport wrapper at all: 35 modules import
a network client and call it, and ``guardrail.authorize`` — the only default-deny
gate in the codebase — is invoked at two of those call sites, both email. A
developer adding an enrichment source could ``import requests`` and reach the
internet without touching any control, and nothing would notice. A control that
must be remembered is a control that will be missed.

This is the part that notices.

It is a RATCHET, not a clean-room rule. Rewriting 35 modules in one change would
be a large, risky diff that touches the email path, the vector store, the model
client and three knowledge-graph writers at once. So the modules that already
import a client are frozen in ``_KNOWN`` below, and the test fails on:

  * a module NOT in the list importing a network client — the boundary is real
    for new code from today;
  * a module IN the list that no longer imports one — the entry is stale and
    must be removed, so the list can only shrink.

The second rule is what makes it a ratchet. Without it the list rots into a
permanent exemption, and someone re-adding an import to a module that was
already cleaned would pass.

To add a legitimate new outbound path: route it through ``services.egress`` and
do not import a client directly. If that is genuinely impossible, adding a line
here is a deliberate, reviewable act with a reason attached — which is the
difference between an exception and an oversight.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"

# Clients that put bytes on a socket. Importing one is not proof of egress, but
# every egress in this codebase goes through one of them, so it is the cheapest
# place to notice a new one.
_NETWORK_MODULES = {
    "requests", "httpx", "urllib3", "aiohttp",      # HTTP
    "boto3", "botocore",                            # AWS
    "smtplib",                                      # SMTP
    "qdrant_client",                                # external vector index
    "neo4j",                                        # graph
    "socket", "ftplib",                             # raw
}

# Matched on the full dotted name rather than the root package, because the root
# is mostly benign and flagging it would make the lint noisy enough to switch
# off. `urllib.parse` is in half the codebase; `urllib.request` opens sockets.
# `http.client` likewise, against a `http` root that is ordinary.
_NETWORK_SUBMODULES = {
    "urllib.request",
    "http.client",
}

# The module that is ALLOWED to import clients, because it is the wrapper.
_THE_CHOKEPOINT = "src/services/egress.py"

# Frozen at the time the boundary was introduced. Every entry is a module that
# already reached the network directly; the comment says which client. This list
# may only shrink.
_KNOWN: dict[str, str] = {
    "src/agents/base_agent.py": "boto3, botocore, neo4j, qdrant_client",
    "src/agents/data_extraction_agent.py": "qdrant_client",
    "src/agents/email_watcher_agent.py": "socket",
    "src/agents/extraction_engine.py": "requests",
    "src/agents/quote_evaluation_agent.py": "qdrant_client",
    "src/api/routers/documents.py": "botocore",
    "src/orchestration/reasoning_engine.py": "urllib.request",
    "src/resources/qdrant_migrations/20241015_add_learning_collection.py": "qdrant_client",
    "src/resources/qdrant_migrations/20241105_add_source_type_indexes.py": "qdrant_client",
    "src/services/conversation_memory.py": "qdrant_client",
    "src/services/data_flow_manager.py": "qdrant_client",
    "src/services/document_embedding_service.py": "qdrant_client",
    "src/services/email_credentials_manager.py": "boto3, botocore",
    "src/services/email_dispatch_service.py": "boto3",
    "src/services/email_ingest_lambda.py": "boto3, botocore",
    "src/services/email_service.py": "boto3, botocore, smtplib",
    "src/services/email_sqs_loader.py": "boto3, botocore",
    "src/services/extraction/parser.py": "boto3",
    "src/services/kg_ingestion_service.py": "neo4j",
    "src/services/learning_repository.py": "qdrant_client",
    "src/services/model_selector.py": "botocore, qdrant_client",
    "src/services/platform_kg.py": "neo4j",
    "src/services/process_routing_service.py": "httpx",
    "src/services/procurement_kg_builder.py": "neo4j",
    "src/services/qdrant_health.py": "qdrant_client, requests",
    "src/services/rag_service.py": "qdrant_client",
    "src/services/static_policy_loader.py": "botocore, qdrant_client",
    "src/services/style/graph_source.py": "boto3, requests",
    "src/services/supplier_relationship_service.py": "qdrant_client",
}


def _imported_network_clients(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            if name.split(".")[0] in _NETWORK_MODULES:
                found.add(name.split(".")[0])
            elif any(name == sub or name.startswith(sub + ".")
                     for sub in _NETWORK_SUBMODULES):
                found.add(name)
    return found


def _scan() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_SRC.parents[0]).as_posix()
        if rel == _THE_CHOKEPOINT:
            continue
        clients = _imported_network_clients(path)
        if clients:
            out[rel] = clients
    return out


def test_the_scan_finds_something():
    """Guards the guard: a path or parser bug that found nothing would make
    both assertions below vacuously true."""
    assert _scan(), (
        "the import scan found no network clients anywhere in src/, which means "
        "it is not looking where it thinks it is"
    )


def test_no_new_module_reaches_the_network_directly():
    new = {k: sorted(v) for k, v in _scan().items() if k not in _KNOWN}
    assert not new, (
        "these modules import a network client and are not on the frozen list:\n  "
        + "\n  ".join(f"{k}  ({', '.join(v)})" for k, v in sorted(new.items()))
        + "\n\nRoute the call through services.egress instead. If that is truly "
          "impossible, add the module to _KNOWN in this test with a reason — "
          "deliberately, in review."
    )


def test_the_frozen_list_has_no_stale_entries():
    """The ratchet. An entry that no longer imports a client must be removed, so
    the list can only shrink and cannot rot into a permanent exemption."""
    scanned = _scan()
    stale = sorted(k for k in _KNOWN if k not in scanned)
    assert not stale, (
        "these modules no longer import a network client, so their exemption is "
        "stale and must be deleted from _KNOWN:\n  " + "\n  ".join(stale)
        + "\n\nLeaving them would let someone re-add a direct client later "
          "without the test noticing."
    )


@pytest.mark.parametrize("module", sorted(_KNOWN))
def test_every_frozen_entry_still_exists(module):
    """A path that no longer exists is an exemption for nothing, and hides a
    typo in the list."""
    assert (_SRC.parents[0] / module).exists(), (
        f"{module} is on the frozen list but does not exist"
    )
