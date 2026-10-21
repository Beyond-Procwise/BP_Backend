"""One-time: populate bp_opportunity.deal_id. Run: python scripts/backfill_opportunity_deal_link.py"""
from src.services.opportunity_linkage import link_opportunities_to_deals

if __name__ == "__main__":
    # An operator tool acts for the platform across every company; with row-level security on, a tool
    # that names no company would otherwise see no rows (D-47).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from src.services.tenant_context import enter_system as _enter_system
    _enter_system()
    n = link_opportunities_to_deals()
    print(f"linked {n} opportunities to deals")
