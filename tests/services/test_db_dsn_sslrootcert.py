"""N4 / H-7: the database DSN carries the CA bundle when PGSSLROOTCERT is set, so verify-full can verify."""
from __future__ import annotations

from src.services import db


def test_sslrootcert_is_added_to_the_dsn_only_when_set(monkeypatch):
    for k in ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD", "PGPORT", "PGSSLMODE", "PGSSLROOTCERT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PGHOST", "db.example")
    monkeypatch.setenv("PGSSLMODE", "verify-full")
    assert "sslrootcert" not in db._pg_dsn()
    monkeypatch.setenv("PGSSLROOTCERT", "/certs/aws-rds-global-bundle.pem")
    dsn = db._pg_dsn()
    assert "sslmode=verify-full" in dsn and "sslrootcert=/certs/aws-rds-global-bundle.pem" in dsn
