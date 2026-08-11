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

# Submodules of a flagged package that CANNOT open a connection: request/response
# dataclasses, configuration objects, exception types. Importing one is not
# egress and never becomes egress.
#
# This exists because the first version of this lint did not have it, and flagged
# 18 of 25 modules for `from qdrant_client import models` or
# `from botocore.exceptions import ClientError` — roughly 72% false positives,
# against exactly the argument made three lines above about urllib.parse. A lint
# that cries wolf on `models.Filter` is a lint somebody switches off, at which
# point it protects nothing.
#
# The distinction that matters is CONSTRUCTING A CONNECTION. `QdrantClient(...)`
# and `boto3.client(...)` do; `models.PointStruct` and `Config(...)` do not.
_BENIGN_SUBMODULES = {
    "qdrant_client.models",          # Filter, PointStruct, VectorParams, ...
    "qdrant_client.http.exceptions",  # UnexpectedResponse
    "qdrant_client.conversions",
    "botocore.config",               # Config(max_pool_connections=...)
    "botocore.exceptions",           # ClientError, BotoCoreError, ...
    "neo4j.exceptions",
    "requests.exceptions",
}

# The module that is ALLOWED to import clients, because it is the wrapper.
_THE_CHOKEPOINT = "src/services/egress.py"

# Frozen at the time the boundary was introduced. Every entry is a module that
# already reached the network directly; the comment says which client. This list
# may only shrink.
_KNOWN: dict[str, str] = {
    "src/agents/base_agent.py": "neo4j — construction is lazy inside a method; graph_client migration pending",
    "src/agents/email_watcher_agent.py": "socket — used for a timeout constant, not a connection",
    "src/services/email_ingest_lambda.py": "boto3 — separately deployed Lambda; see the note in _KNOWN",
    "src/services/email_service.py": "smtplib — SMTP send, already gated by email_dispatch_guard",
    "src/services/kg_ingestion_service.py": "neo4j",
    "src/services/platform_kg.py": "neo4j",
    "src/services/procurement_kg_builder.py": "neo4j",
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
            # Both the module AND module.name, so `from qdrant_client import
            # models` is seen as `qdrant_client.models` and can be recognised as
            # a dataclass namespace. Checking node.module alone reads it as the
            # root package and flags every dataclass import in the codebase.
            names = [node.module] + [
                f"{node.module}.{a.name}" for a in node.names
            ]
        else:
            continue
        # A `from X import a, b` yields [X, X.a, X.b]. If every specific name is
        # benign the bare X must not flag on its own, or the dataclass import is
        # caught by the root anyway and the benign list does nothing.
        specific = [n for n in names if "." in n]
        if specific and all(
            any(n == b or n.startswith(b + ".") for b in _BENIGN_SUBMODULES)
            for n in specific
        ):
            continue

        for name in names:
            if any(name == b or name.startswith(b + ".")
                   for b in _BENIGN_SUBMODULES):
                continue          # dataclasses / config / exceptions
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
