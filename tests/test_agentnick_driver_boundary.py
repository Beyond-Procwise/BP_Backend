"""No new direct calls to the model outside the driver (a ratchet, modelled on test_gateway_boundary).

AgentNick sits alongside the orchestrator. Everything that asks it for something goes through
``src/services/agentnick_driver.py``, so there is one place to see what is asked and one place to
enforce that what comes back is only a proposal. Many files still call the model directly today.
``tests/agentnick_direct_calls_baseline.txt`` lists them. The list may only shrink:
  * a NEW file that posts to the model fails this test (use the driver);
  * a file that no longer calls it directly must be deleted from the list, so the ratchet tightens.
Regenerate the baseline only when moving a caller onto the driver:
    python tests/test_agentnick_driver_boundary.py --write
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "tests", "agentnick_direct_calls_baseline.txt")
# The driver is the boundary. tool_runtime only keeps the names other modules import from it.
EXEMPT = ("src/services/agentnick_driver.py", "src/services/tool_runtime.py")
# Two wires reach the model: raw posts to its HTTP API, and ollama_client.ollama_generate. A caller counts as
# direct when it posts to the API itself or imports/calls the client's generate functions. Importing the driver's
# generate (even under the local name ollama_generate) is the sanctioned route and is not matched.
PATTERN = re.compile(
    r"api/chat|api/generate|api/embed|\b_?OLLAMA_CHAT\b"
    r"|ollama_client\s+import[^\n]*\bollama_(?:cloud_)?generate\b"
    r"|ollama_client\.ollama_(?:cloud_)?generate\b"
    r"|\bdef ollama_(?:cloud_)?generate\b")


def _scan():
    found = []
    for base, _dirs, files in os.walk(os.path.join(ROOT, "src")):
        rel_base = os.path.relpath(base, ROOT).replace(os.sep, "/")
        if "__pycache__" in rel_base:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            rel = f"{rel_base}/{name}"
            if rel in EXEMPT:
                continue
            with open(os.path.join(base, name), encoding="utf-8", errors="ignore") as fh:
                if PATTERN.search(fh.read()):
                    found.append(rel)
    return sorted(found)


def _baseline():
    if not os.path.exists(BASELINE):
        return []
    with open(BASELINE, encoding="utf-8") as fh:
        return sorted(line.strip() for line in fh if line.strip() and not line.startswith("#"))


def test_no_new_file_calls_the_model_directly():
    new = sorted(set(_scan()) - set(_baseline()))
    assert not new, (
        "These files call the model directly. Go through src/services/agentnick_driver.py instead:\n  "
        + "\n  ".join(new))


def test_the_baseline_only_lists_files_that_still_call_the_model_directly():
    gone = sorted(set(_baseline()) - set(_scan()))
    assert not gone, (
        "These files no longer call the model directly. Delete them from "
        "tests/agentnick_direct_calls_baseline.txt so the ratchet tightens:\n  " + "\n  ".join(gone))


if __name__ == "__main__" and "--write" in sys.argv:
    with open(BASELINE, "w", encoding="utf-8") as fh:
        fh.write("# Files that call the model directly. May only shrink. See tests/test_agentnick_driver_boundary.py\n")
        fh.write("\n".join(_scan()) + "\n")
    print(f"wrote {len(_scan())} files to {BASELINE}")


# Callers moved onto the driver. Each must import the driver's generate and must not import the client's.
MOVED_TO_THE_DRIVER = (
    "src/services/sell_side/quote_agent.py", "src/services/sell_side/template_reader.py",
    "src/services/rga/compose.py", "src/services/analytics/insight.py", "src/services/style/compiler.py",
    "src/services/style/drafting.py", "src/services/style/exemplars.py",
    "src/services/negotiation_advice/advisor.py", "src/services/summary_agent.py",
    "src/services/deal_summary.py", "src/services/i18n/adapters.py",
)
# opportunity_critic/llm.py moved too, but through chat() rather than generate().


def test_the_callers_moved_onto_the_driver_use_it_and_not_the_client():
    for rel in MOVED_TO_THE_DRIVER:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        assert "agentnick_driver import generate" in src, f"{rel} should import the driver's generate"
        assert not PATTERN.search(src), f"{rel} calls the model directly again"
        assert rel not in _baseline(), f"{rel} is moved; it should not be in the baseline"


def test_the_critic_goes_through_the_driver_chat():
    src = open(os.path.join(ROOT, "src/services/opportunity_critic/llm.py"), encoding="utf-8").read()
    assert "agentnick_driver" in src and "agentnick_driver.chat(" in src and not PATTERN.search(src)
    assert "src/services/opportunity_critic/llm.py" not in _baseline()
