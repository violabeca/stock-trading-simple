"""Database access: engine, schema, and idempotent UPSERT writers.

The database is the shared TimescaleDB used by the sibling ``stock-trading``
repo — this module connects to it but never creates it. All writes UPSERT on
natural keys (``securities.symbol`` and ``prices_ohlcv (symbol, bar, ts)``), so
backfills can be interrupted and re-run freely without duplicates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError

from trading_simple.config import Settings, load_settings

log = logging.getLogger(__name__)

_SCHEMA_FILE = Path(__file__).with_name("schema.sql")
_UPSERT_CHUNK = 2000
_RETRY_BACKOFF_S = (5, 15, 30, 60)


@dataclass(slots=True)
class Security:
    """One row of the ``securities`` master table."""

    symbol: str
    name: str | None = None
    security_type: str = "equity"
    exchange: str | None = None
    currency: str = "USD"
    sector: str | None = None
    industry: str | None = None


@dataclass(slots=True)
class PriceBar:
    """One OHLCV candle. ``ts`` is the bar OPEN time in UTC."""

    symbol: str
    bar: str
    ts: datetime
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    adj_close: float | None
    volume: float | None
    source: str
    rth: bool = True


def make_engine(settings: Settings | None = None) -> Engine:
    """Create a SQLAlchemy engine for the shared TimescaleDB.

    ``pool_pre_ping`` transparently reconnects if the database dropped a
    connection between uses (it occasionally restarts under heavy load).

    Args:
        settings: Resolved settings; loaded from YAML/env when omitted.

    Returns:
        A ready-to-use SQLAlchemy :class:`~sqlalchemy.Engine`.
    """
    settings = settings or load_settings()
    return create_engine(settings.db.sqlalchemy_url, pool_pre_ping=True, future=True)


def apply_schema(engine: Engine) -> None:
    """Apply ``schema.sql`` idempotently (a no-op against the live shared DB).

    Runs on an AUTOCOMMIT connection because TimescaleDB's hypertable helpers
    refuse to run inside a transaction.

    Args:
        engine: Target database engine.
    """
    sql = _SCHEMA_FILE.read_text()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(sql)


def upsert_securities(engine: Engine, items: list[Security]) -> int:
    """Insert or update securities rows (conflict key: ``symbol``).

    ``COALESCE`` keeps existing non-null values, so re-registering a symbol
    with less metadata never erases richer data already stored.

    Args:
        engine: Target database engine.
        items: Securities to write.

    Returns:
        Number of rows submitted.
    """
    sql = """
        INSERT INTO securities (symbol, name, security_type, exchange, currency, sector, industry)
        VALUES (:symbol, :name, :security_type, :exchange, :currency, :sector, :industry)
        ON CONFLICT (symbol) DO UPDATE SET
            name     = COALESCE(EXCLUDED.name,     securities.name),
            exchange = COALESCE(EXCLUDED.exchange, securities.exchange),
            currency = COALESCE(EXCLUDED.currency, securities.currency),
            sector   = COALESCE(EXCLUDED.sector,   securities.sector),
            industry = COALESCE(EXCLUDED.industry, securities.industry),
            updated_at = now()
    """
    return _executemany(engine, sql, [asdict(s) for s in items])


def upsert_prices(engine: Engine, bars: list[PriceBar]) -> int:
    """Insert or update price bars (conflict key: ``(symbol, bar, ts)``).

    Args:
        engine: Target database engine.
        bars: Price bars to write.

    Returns:
        Number of rows submitted.
    """
    sql = """
        INSERT INTO prices_ohlcv (symbol, bar, ts, open, high, low, close, adj_close,
                                  volume, source, rth)
        VALUES (:symbol, :bar, :ts, :open, :high, :low, :close, :adj_close,
                :volume, :source, :rth)
        ON CONFLICT (symbol, bar, ts) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
            close = EXCLUDED.close, adj_close = EXCLUDED.adj_close,
            volume = EXCLUDED.volume, source = EXCLUDED.source, rth = EXCLUDED.rth
    """
    return _executemany(engine, sql, [asdict(b) for b in bars])


def _executemany(engine: Engine, sql: str, rows: list[dict]) -> int:
    """Execute a statement over rows in chunks, committing per chunk.

    Chunking bounds memory and transaction size; retrying on
    ``OperationalError`` with increasing backoff survives brief database
    restarts mid-backfill.

    Args:
        engine: Target database engine.
        sql: Parameterized statement with named binds.
        rows: One dict of bind values per row.

    Returns:
        Number of rows submitted.
    """
    stmt = text(sql)
    for i in range(0, len(rows), _UPSERT_CHUNK):
        chunk = rows[i : i + _UPSERT_CHUNK]
        for attempt, backoff in enumerate((*_RETRY_BACKOFF_S, None)):
            try:
                with engine.begin() as conn:
                    conn.execute(stmt, chunk)
                break
            except OperationalError:
                if backoff is None:
                    raise
                log.warning("DB write failed (attempt %d), retrying in %ss", attempt + 1, backoff)
                time.sleep(backoff)
    return len(rows)
