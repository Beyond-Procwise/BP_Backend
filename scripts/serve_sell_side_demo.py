"""Serve ONLY /catalog and /sales, on the real auth dependency and the real gate,
for a local end-to-end walk.

Not the product server, on purpose: api.main's lifespan starts the full backend
scheduler, and a second one against the same database would run every scheduled
job twice. This mounts the two routers exactly the way main.py does.

Run:
  set -a && . ./.env && set +a
  PYTHONPATH=.:src ./.venv/bin/uvicorn scripts.serve_sell_side_demo:app --port 8765
"""
from fastapi import Depends, FastAPI

from api import auth as _auth
from api.routers import catalog, sales

app = FastAPI(title="Sell-side walk (catalog + sales routers only)")
for _router in (catalog.router, sales.router):
    app.include_router(_router, dependencies=[Depends(_auth.require_user)])
