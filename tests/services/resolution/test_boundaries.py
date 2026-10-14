"""The layer's edges: one entry point, and no reach into extraction."""
import pathlib
import re

import src.services.resolution as resolution

MODULE_DIR = pathlib.Path(resolution.__file__).parent

# Phase 1 is pure optimisation over inputs produced upstream. Any import from
# these would mean a model-derived value had found its way in.
FORBIDDEN = re.compile(
    r"^\s*(?:from|import)\s+.*\b("
    r"extraction|extraction_v2|extraction_v3|llm|ollama|langextract|"
    r"model_selector|llm_router|agents?)\b",
    re.MULTILINE,
)


def test_the_layer_never_imports_the_llm_or_extraction_stack():
    offenders = []
    for path in sorted(MODULE_DIR.glob("*.py")):
        for match in FORBIDDEN.finditer(path.read_text()):
            offenders.append(f"{path.name}: {match.group(0).strip()}")
    assert offenders == []


def test_resolve_is_the_only_callable_the_package_exports():
    exported = [n for n in resolution.__all__ if callable(getattr(resolution, n))
                and not isinstance(getattr(resolution, n), type)]
    assert exported == ["resolve"]
