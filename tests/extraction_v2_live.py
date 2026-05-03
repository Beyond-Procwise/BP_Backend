"""V2 live coverage harness.

Runs the V2 extraction pipeline over every fixture in
tests/structural_extractor/fixtures/docs and reports:

- Per-field commit rate vs abstention rate
- Which locator strategies fired (returned a value) vs returned None
- Per-document residuals with the reason

Run: python tests/extraction_v2_live.py [--doc <substring>]

Exit code 0 always — this is a coverage report, not a pass/fail test.
The accuracy assertions live in tests/extraction_v2_live_assertions.py.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

from src.services.extraction_v2.locator.consensus import run_locators
from src.services.extraction_v2.locator.registry import build_locators
from src.services.extraction_v2.pipeline import ExtractionPipelineV2
from src.services.structural_extractor.parsing import parse


FIXTURES_DIR = project_root / "tests" / "structural_extractor" / "fixtures" / "docs"


# Heuristic doc-type assignment per fixture filename
def _classify_filename(name: str) -> str:
    n = name.upper()
    if "PO" in n and "INV" not in n:
        return "Purchase_Order"
    if "QUT" in n or "QUOTE" in n or "QTE" in n:
        return "Quote"
    return "Invoice"


def _short(value, n: int = 50) -> str:
    s = repr(value)
    return s if len(s) <= n else s[: n - 1] + "…"


def run_one(path: Path, doc_type: Optional[str] = None) -> dict:
    """Run the V2 pipeline against one fixture and return a structured report."""
    file_bytes = path.read_bytes()
    parsed = parse(file_bytes, path.name)
    if doc_type is None:
        doc_type = _classify_filename(path.name)

    locators_per_field = build_locators(doc_type)
    pipeline = ExtractionPipelineV2()
    result = pipeline.extract(parsed, doc_type)

    # Re-run consensus per field to capture per-locator firing stats
    # (run_locators is cheap given structural locator is memoized)
    locator_firing: dict[str, list[tuple[str, bool]]] = {}
    for fname, locators in locators_per_field.items():
        firing = []
        for loc in locators:
            try:
                out = loc.locate(parsed)
            except Exception:
                out = None
            firing.append((loc.name, out is not None))
        locator_firing[fname] = firing

    return {
        "path": str(path),
        "doc_type": doc_type,
        "result": result,
        "locator_firing": locator_firing,
        "n_pages": parsed.pages_or_sheets,
        "text_len": len(parsed.full_text),
    }


def render_report(reports: list[dict]) -> None:
    field_commits: dict[str, int] = defaultdict(int)
    field_total: dict[str, int] = defaultdict(int)
    locator_fires: dict[str, int] = defaultdict(int)
    locator_total: dict[str, int] = defaultdict(int)

    print("\n" + "=" * 78)
    print(" V2 EXTRACTION COVERAGE REPORT")
    print("=" * 78)

    for r in reports:
        path = Path(r["path"])
        result = r["result"]
        n_committed = len(result.committed)
        n_residual = len(result.residuals)
        n_total = n_committed + n_residual

        print(f"\n[{r['doc_type']}] {path.name}")
        print(f"  pages={r['n_pages']} text={r['text_len']}ch  "
              f"committed={n_committed}/{n_total} residuals={n_residual}")

        for fname, cf in sorted(result.committed.items()):
            print(f"    [✓] {fname}: {_short(cf.value)} "
                  f"(conf={cf.confidence:.2f}, locators={cf.locator_count})")
        for rf in sorted(result.residuals, key=lambda x: x.field):
            print(f"    [-] {rf.field}: {rf.why}")

        # Roll up field stats
        for fname in r["locator_firing"]:
            field_total[fname] += 1
            if fname in result.committed:
                field_commits[fname] += 1
            for lname, fired in r["locator_firing"][fname]:
                locator_total[lname] += 1
                if fired:
                    locator_fires[lname] += 1

    print("\n" + "-" * 78)
    print(" FIELD COMMIT RATES (across all docs)")
    print("-" * 78)
    rows = sorted(field_total.keys(),
                  key=lambda f: -field_commits[f] / max(1, field_total[f]))
    for fname in rows:
        commits = field_commits[fname]
        total = field_total[fname]
        rate = commits / total * 100
        bar = "█" * int(rate / 5) + "·" * (20 - int(rate / 5))
        print(f"  {fname:<32} {commits:>3}/{total:<3}  {rate:5.1f}%  {bar}")

    print("\n" + "-" * 78)
    print(" LOCATOR FIRING RATES (returned a value)")
    print("-" * 78)
    for lname in sorted(locator_total.keys()):
        fires = locator_fires[lname]
        total = locator_total[lname]
        rate = fires / total * 100
        bar = "█" * int(rate / 5) + "·" * (20 - int(rate / 5))
        print(f"  {lname:<40} {fires:>3}/{total:<3}  {rate:5.1f}%  {bar}")

    # Overall summary
    print("\n" + "=" * 78)
    total_committed = sum(len(r["result"].committed) for r in reports)
    total_residual = sum(len(r["result"].residuals) for r in reports)
    grand_total = total_committed + total_residual
    rate = total_committed / max(1, grand_total) * 100
    print(f" OVERALL: {total_committed}/{grand_total} committed "
          f"({rate:.1f}%), {total_residual} residuals")
    print("=" * 78)


def main():
    args = sys.argv[1:]
    filter_substring = None
    if "--doc" in args:
        filter_substring = args[args.index("--doc") + 1]

    if not FIXTURES_DIR.exists():
        print(f"ERROR: fixtures dir not found: {FIXTURES_DIR}", file=sys.stderr)
        sys.exit(2)

    paths = sorted(p for p in FIXTURES_DIR.iterdir()
                   if p.suffix.lower() in {".pdf", ".docx", ".xlsx", ".csv"})
    if filter_substring:
        paths = [p for p in paths if filter_substring.lower() in p.name.lower()]

    if not paths:
        print("No fixtures matched.")
        sys.exit(0)

    reports = []
    for p in paths:
        try:
            reports.append(run_one(p))
        except Exception as e:
            print(f"  [ERR] {p.name}: {type(e).__name__}: {e}")

    render_report(reports)


if __name__ == "__main__":
    main()
