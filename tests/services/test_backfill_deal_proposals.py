import importlib.util
import os

# Import the backfill script by explicit file path. The repo has TWO importable `scripts`
# packages — ./scripts (this file's target) and ./src/scripts — and both sit on sys.path,
# so a bare `import scripts.backfill_deal_proposals` binds to whichever was imported first.
# In the full suite `src.api.main` binds `scripts` -> src/scripts, which lacks this module.
# A path-based import is order-independent and unambiguous. (CLI use `-m
# scripts.backfill_deal_proposals` resolves to ./scripts correctly on its own.)
_BF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                        "backfill_deal_proposals.py")
_spec = importlib.util.spec_from_file_location("backfill_deal_proposals_undertest", _BF_PATH)
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)


def test_backfill_delegates_to_generate(monkeypatch):
    seen = {}
    monkeypatch.setattr(bf, "_generate",
                        lambda batch, session: seen.update(batch=batch) or
                        {"batch_deal_id": batch, "proposal_ids": [1, 2], "ungrouped": [],
                         "members_with_lines": 0, "members_total": 0})
    out = bf.backfill("ANALYSISSET_19072620260719339")
    assert seen["batch"] == "ANALYSISSET_19072620260719339"
    assert out["proposal_ids"] == [1, 2]
