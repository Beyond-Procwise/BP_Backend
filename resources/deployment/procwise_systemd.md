# procwise systemd service — drop-in config

The procwise app runs as a systemd service. One drop-in is required for the
graceful **REDUCED-mode** startup (so the server still boots when the database is
unreachable instead of hanging the whole startup). The drop-in lives outside the
repo at `/etc/systemd/system/procwise.service.d/` — this note is the source of
truth for recreating it.

## Drop-in: DB timeout + start limit

File: `/etc/systemd/system/procwise.service.d/dbtimeout.conf`

```ini
[Service]
# Fail fast on DB connects when the database is down (psycopg2 floors this at 2s).
# Without it, each startup DB connect hung ~2 min. Read by base_agent.get_db_connection
# and services/db.get_conn (env: DB_CONNECT_TIMEOUT, code default 5).
Environment="DB_CONNECT_TIMEOUT=1"
# Room for the ~14 agents + governance engines to each fast-fail their DB connect
# and fall back during a DB-down (REDUCED-mode) boot. Normal (DB up) boot is fast.
TimeoutStartSec=300
```

Apply:
```bash
sudo mkdir -p /etc/systemd/system/procwise.service.d
sudo tee /etc/systemd/system/procwise.service.d/dbtimeout.conf < the block above
sudo systemctl daemon-reload
sudo systemctl restart procwise
```

## Behaviour

- **DB reachable (normal):** full startup; extraction, scheduling, watchers, and
  persistence all run.
- **DB unreachable (REDUCED mode):** a startup probe logs
  `DB REACHABILITY PROBE FAILED — starting in REDUCED mode` (loud ERROR), the
  DB-coupled subsystems (orchestrator/scheduler, email + process watchers, pattern
  seeding, provenance, schema-DB-verify) are **skipped** (they otherwise
  retry-loop and block startup forever), and AgentNick + the reasoning engine come
  up so `/agents/instruct` and other Ollama-based endpoints work. Measured boot:
  **~56 s** to listening with the DB down. Verified: `/agents/instruct` →
  `planner=llm`, multi-step plan.

The graceful-startup logic itself is in `src/api/main.py` (lifespan) and
`src/agents/base_agent.py` (resilient engine construction). The REDUCED mode is a
**stopgap** — the real fix is restoring the database server `63.35.28.70` (start it
or open its security group to this box's IP `16.61.116.180`); then a
`sudo systemctl restart procwise` returns to full mode.

## Other procwise commands

```bash
sudo systemctl {start|stop|restart|status} procwise
tail -f /home/muthu/PycharmProjects/BP_Backend/src/logs/procwise.log   # full app log
sudo journalctl -u procwise -f                                         # uvicorn/systemd output
```
