"""One-time: populate bp_opportunity.deal_id. Run: python scripts/backfill_opportunity_deal_link.py"""
from src.services.opportunity_linkage import link_opportunities_to_deals

if __name__ == "__main__":
    n = link_opportunities_to_deals()
    print(f"linked {n} opportunities to deals")
