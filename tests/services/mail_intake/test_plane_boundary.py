"""The plane boundary, enforced as a build failure rather than a convention.

Spec §2: the Interpretation Plane must be structurally incapable of reaching a
mailbox. Both planes currently share one Python process, so this is a STATIC
capability check over the import graph, not a runtime sandbox — see spec §2.2.
It is still the strongest in-process mechanism available, and it fails the
build rather than warning.

If this test ever fails, do not add the offending module to the allowlist.
The import is the bug.
"""
import ast
import pathlib

from src.services.mail_intake.handoff import KNOWN_CROSSINGS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

FORBIDDEN_PREFIXES = (
    "services.mail_intake",
    "src.services.mail_intake",
    "services.tool_runtime",
    "src.services.tool_runtime",
    "services.supplier_enrichment.web_tools",
    "src.services.supplier_enrichment.web_tools",
    "services.email_watcher",
    "src.services.email_watcher",
    "imaplib",
    "smtplib",
    "requests",
    "httpx",
    "urllib.request",
    "boto3",
)

# Absolute, derived from this file. A relative path made rglob yield nothing
# from any other working directory, so three of these tests passed green having
# scanned zero files — a ratchet that silently no-ops is the exact failure it
# exists to prevent, and pytest.ini pins no testpaths.
#
# All THREE roots, not just `extraction`: dispatch runs through extraction_v3
# for parsing, invariants and the judge, and through extraction_feedback. Both
# are clean today, but a new egress import in extraction_v3/parsers/ would not
# have failed the build while the docstring claimed the plane was covered.
EXTRACTION_ROOTS = (
    REPO_ROOT / "src" / "services" / "extraction",
    REPO_ROOT / "src" / "services" / "extraction_v3",
    REPO_ROOT / "src" / "services" / "extraction_feedback",
)


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.module


def test_extraction_gains_no_new_route_out_of_its_plane():
    """A ratchet, not a clean bill of health.

    The boundary is already crossed: parser.py imports boto3 and downloads
    documents by key, so extraction has live egress today and its credentials
    can read any key in the bucket — the quarantine prefix included. Asserting
    zero crossings would simply fail, and asserting nothing would let the next
    one in unnoticed.

    So the existing crossing is recorded explicitly in handoff.KNOWN_CROSSINGS
    and anything else fails the build. Mail-intake's own path is protected
    differently and by construction: handoff.deliver_to_extraction never gives
    extraction a key, so the fallback that does the fetching is never reached.
    """
    offences = []
    scanned = 0
    for root in EXTRACTION_ROOTS:
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            rel = str(path.relative_to(REPO_ROOT))
            allowed = KNOWN_CROSSINGS.get(rel, set())
            for name in _imports(path):
                for forbidden in FORBIDDEN_PREFIXES:
                    if name == forbidden or name.startswith(forbidden + "."):
                        if name in allowed:
                            continue
                        offences.append(f"{rel}: imports {name}")

    # A guard that scanned nothing is not a guard. Without this the tests pass
    # green from any working directory where the globs match no files.
    assert scanned > 0, "the import guard scanned zero files — check the roots"

    assert not offences, (
        "Interpretation Plane gained a NEW route out of its plane:\n"
        + "\n".join(offences)
        + "\n\nDo not add these to KNOWN_CROSSINGS to make this pass. The "
        "import is the bug."
    )


def test_the_recorded_crossings_still_exist():
    """If a recorded crossing is gone, stop recording it.

    An allowlist nobody prunes silently widens over time, and a stale entry
    would hide a genuine regression at that exact location.
    """
    stale = []
    for rel, names in KNOWN_CROSSINGS.items():
        path = REPO_ROOT / rel
        if not path.exists():
            stale.append(f"{rel} no longer exists")
            continue
        found = set(_imports(path))
        for name in names:
            if name not in found:
                stale.append(f"{rel} no longer imports {name}")
    assert not stale, (
        "KNOWN_CROSSINGS is out of date — remove these entries:\n" + "\n".join(stale)
    )


def test_mail_intake_contains_no_llm_calls():
    # The Ingestion Plane is deterministic. Spec §2.
    forbidden = ("ollama", "llm_router", "run_tools", "AgentNick", "openai")
    offences = []
    scanned = 0
    for path in sorted((REPO_ROOT / "src" / "services" / "mail_intake").rglob("*.py")):
        scanned += 1
        body = path.read_text()
        for token in forbidden:
            if token in body:
                offences.append(f"{path.name}: mentions {token}")
    assert scanned > 0, "scanned zero mail_intake files — check the root"
    assert not offences, "Ingestion Plane is not deterministic:\n" + "\n".join(offences)


def test_extraction_is_never_handed_a_binding_or_credential():
    # It receives an artifact_id and bytes. Nothing else.
    offences = []
    scanned = 0
    for root in EXTRACTION_ROOTS:
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            body = path.read_text()
            for token in ("credential_ref", "binding_id", "mailbox_address"):
                if token in body:
                    offences.append(f"{path.relative_to(REPO_ROOT)}: references {token}")
    assert scanned > 0, "scanned zero extraction files — check the roots"
    assert not offences, "\n".join(offences)
