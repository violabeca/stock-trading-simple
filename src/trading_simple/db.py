"""Database access: engine, schema, and idempotent UPSERT writers.

The database is the shared TimescaleDB used by the sibling ``stock-trading``
repo — this module connects to it but never drops or migrates it. All writes
UPSERT on natural keys, so backfills can be interrupted and re-run freely
without duplicates: prices on ``(symbol, bar, ts)``, and the alternative-data
tables (:class:`InsiderTrade`, :class:`RatingChange`, :class:`ShortInterest`) on
their NULLS-NOT-DISTINCT natural keys. This repo owns its own alt-data ingestion
(SEC EDGAR / Yahoo / FINRA providers) and defines those tables in ``schema.sql``
with column definitions identical to the sibling's, so the shared database stays
consistent whichever repo populated them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
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


@dataclass(slots=True)
class InsiderTrade:
    """One SEC Form 4 insider transaction.

    ``traded_at`` is the transaction date; ``known_at`` is the SEC filing
    acceptance timestamp (when it became public — the only time a strategy may
    act on it). ``txn_type`` is ``"buy"`` or ``"sell"`` (open-market P/S codes).
    """

    symbol: str
    insider: str | None
    role: str | None
    txn_type: str | None
    shares: float | None
    price: float | None
    traded_at: date
    filed_at: datetime
    known_at: datetime
    accession: str | None
    source: str


@dataclass(slots=True)
class RatingChange:
    """One sell-side analyst rating/grade change.

    ``known_at`` is the announcement timestamp. ``action`` is one of
    ``initiate | upgrade | downgrade | maintain``.
    """

    symbol: str
    firm: str | None
    action: str | None
    grade_from: str | None
    grade_to: str | None
    rated_at: date
    known_at: datetime
    source: str


@dataclass(slots=True)
class ShortInterest:
    """One FINRA bi-monthly consolidated short-interest observation.

    ``settlement_date`` is when the position was measured; ``known_at`` is
    FINRA's dissemination timestamp (~8 business days later). ``days_to_cover``
    is ``short_interest / avg_daily_volume``.
    """

    symbol: str
    settlement_date: date
    short_interest: float | None
    avg_daily_volume: float | None
    days_to_cover: float | None
    disseminated_at: date | None
    known_at: datetime
    source: str


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


#: Alt-data tables with a BIGSERIAL ``id`` this repo inserts into.
_SERIAL_TABLES = ("insider_trades", "analyst_ratings")


def apply_schema(engine: Engine) -> None:
    """Apply ``schema.sql`` idempotently (a no-op against the live shared DB).

    Runs on an AUTOCOMMIT connection because TimescaleDB's hypertable helpers
    refuse to run inside a transaction. Then aligns the BIGSERIAL id sequences
    of the alt-data tables (see :func:`_align_serials`).

    Args:
        engine: Target database engine.
    """
    sql = _SCHEMA_FILE.read_text()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql(sql)
    _align_serials(engine)


def _align_serials(engine: Engine) -> None:
    """Advance each alt-data BIGSERIAL sequence to the table's ``max(id)``.

    A shared table may have been bulk-loaded by the sibling repo with explicit
    ids that never advanced its serial sequence; a fresh ``INSERT`` from here
    would then draw a stale id and collide on the primary key before the
    natural-key ``ON CONFLICT`` clause could apply. Realigning the sequence to
    ``max(id)`` fixes that. Safe and idempotent: it only ever moves the sequence
    up to the existing data (``nextval`` becomes an unused id), never below it,
    and it no-ops on an empty (freshly created) table.

    Args:
        engine: Target database engine.
    """
    for table in _SERIAL_TABLES:
        sql = text(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), max(id)) "
                   f"FROM {table} HAVING max(id) IS NOT NULL")
        try:
            with engine.begin() as conn:
                conn.execute(sql)
        except Exception:  # noqa: BLE001 — a missing table or perms must not break init-db
            log.warning("could not align id sequence for %s", table)


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


def upsert_insider(engine: Engine, trades: list[InsiderTrade]) -> int:
    """Insert insider trades, ignoring duplicates (key: ``insider_trades_uq``).

    Insider rows are immutable once filed, so a conflict means we already have
    the row — ``DO NOTHING`` keeps re-runs idempotent.

    Args:
        engine: Target database engine.
        trades: Insider transactions to write.

    Returns:
        Number of rows submitted.
    """
    sql = """
        INSERT INTO insider_trades (symbol, insider, role, txn_type, shares, price,
                                    traded_at, filed_at, known_at, accession, source)
        VALUES (:symbol, :insider, :role, :txn_type, :shares, :price,
                :traded_at, :filed_at, :known_at, :accession, :source)
        ON CONFLICT ON CONSTRAINT insider_trades_uq DO NOTHING
    """
    return _executemany(engine, sql, [asdict(t) for t in trades])


def upsert_ratings(engine: Engine, ratings: list[RatingChange]) -> int:
    """Insert analyst rating changes, ignoring duplicates (key: ``analyst_ratings_uq``).

    Args:
        engine: Target database engine.
        ratings: Rating/grade changes to write.

    Returns:
        Number of rows submitted.
    """
    sql = """
        INSERT INTO analyst_ratings (symbol, firm, action, grade_from, grade_to,
                                     rated_at, known_at, source)
        VALUES (:symbol, :firm, :action, :grade_from, :grade_to,
                :rated_at, :known_at, :source)
        ON CONFLICT ON CONSTRAINT analyst_ratings_uq DO NOTHING
    """
    return _executemany(engine, sql, [asdict(r) for r in ratings])


def upsert_short_interest(engine: Engine, rows: list[ShortInterest]) -> int:
    """Insert or update short-interest rows (key: ``(symbol, settlement_date)``).

    FINRA occasionally revises a settlement's figures, so a conflict UPDATEs the
    measured columns rather than ignoring them.

    Args:
        engine: Target database engine.
        rows: Short-interest observations to write.

    Returns:
        Number of rows submitted.
    """
    sql = """
        INSERT INTO short_interest (symbol, settlement_date, short_interest,
                                    avg_daily_volume, days_to_cover, disseminated_at,
                                    known_at, source)
        VALUES (:symbol, :settlement_date, :short_interest, :avg_daily_volume,
                :days_to_cover, :disseminated_at, :known_at, :source)
        ON CONFLICT ON CONSTRAINT short_interest_uq DO UPDATE SET
            short_interest   = EXCLUDED.short_interest,
            avg_daily_volume = EXCLUDED.avg_daily_volume,
            days_to_cover    = EXCLUDED.days_to_cover,
            disseminated_at  = EXCLUDED.disseminated_at,
            known_at         = EXCLUDED.known_at,
            source           = EXCLUDED.source
    """
    return _executemany(engine, sql, [asdict(r) for r in rows])


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
