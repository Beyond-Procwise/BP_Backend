"""The requirements router must be mounted on the app.

Regression: src/api/routers/requirements.py existed, was imported by nothing, and
so POST /requirements/message returned 404 on the live server. The agent behind it
was unreachable — which is why the UI's Requirements screen was still replying from
a canned question list. Importing the module is not enough; it has to be included.

Asserted against the router object rather than a booted app so the check costs
nothing and cannot be defeated by import order.
"""
import importlib
import re
from pathlib import Path

MAIN = Path(__file__).resolve().parents[2] / "src" / "api" / "main.py"


def test_main_imports_and_includes_the_requirements_router():
    source = MAIN.read_text()
    assert re.search(r"import requirements as requirements_router", source), \
        "main.py must import the requirements router"
    assert "app.include_router(requirements_router.router)" in source, \
        "main.py must mount the requirements router"


def test_router_exposes_the_message_turn():
    module = importlib.import_module("src.api.routers.requirements")
    paths = {getattr(r, "path", None) for r in module.router.routes}
    assert "/requirements/message" in paths
    assert "/requirements/run-workflow" in paths
