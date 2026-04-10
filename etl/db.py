# db.py

"""
cadsentinel.etl.db
------------------
Database connection management.
Configure via environment variables:
  CADSENTINEL_DATABASE_URL  - full connection string (takes priority)
  PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD  - individual settings
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras


def _get_dsn() -> str:
    url = os.environ.get("CADSENTINEL_DATABASE_URL")
    if url:
        return url

    host     = os.environ.get("PGHOST",     "localhost")
    port     = os.environ.get("PGPORT",     "5432")
    dbname   = os.environ.get("PGDATABASE", "cadsentinel")
    user     = os.environ.get("PGUSER",     "cadsentinel")
    password = os.environ.get("PGPASSWORD", "")

    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


@contextmanager
def get_connection():
    """
    Yields a psycopg2 connection with RealDictCursor as default.
    Commits on clean exit, rolls back on exception.
    """
    conn = psycopg2.connect(
        _get_dsn(),
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()