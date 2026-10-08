"""One throwaway Postgres for the whole eval session (skipped where there is neither Docker nor a DSN)."""

import pytest

from evals.email import db


@pytest.fixture(scope="session")
def eval_dsn():
    """The throwaway database's connection string (psycopg2 masks the password in a connection's own .dsn)."""
    try:
        cm = db.database()
        dsn = cm.__enter__()
    except db.NoDatabase as exc:
        pytest.skip(str(exc))
    try:
        yield dsn
    finally:
        cm.__exit__(None, None, None)


@pytest.fixture(scope="session")
def eval_db(eval_dsn):
    import psycopg2
    conn = psycopg2.connect(eval_dsn)
    db.load(conn)
    conn.autocommit = True
    yield conn
