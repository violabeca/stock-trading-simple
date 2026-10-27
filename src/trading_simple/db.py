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
class PriceTarget:
    """One sell-side analyst price-target event.

    ``known_at`` is the announcement timestamp. ``pt_action`` is Benzinga's raw
    label (``Raises | Lowers | Maintains | Announces``). ``pt_prior`` is ``None``
    when the analyst had no previous target — never ``0.0``, which a revision
    ratio would read as a -100% cut.

    Distinct from :class:`RatingChange` on purpose: most target revisions arrive
    with the *rating unchanged*, so the two carry near-orthogonal information.
    """

    symbol: str
    firm: str | None
    pt_action: str | None
    pt_current: float | None
    pt_prior: float | None
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


@dataclass(slots=True)
class Fundamental:
    """One as-reported company fundamental from SEC EDGAR XBRL ``companyfacts``.

    A single logical quantity for one reporting period from one filing. ``concept``
    is normalized (``equity`` | ``assets`` | ``shares`` | ``ni_ttm`` | ``gp_ttm``):
    the balance-sheet concepts are INSTANT levels (``period_start`` is ``None``),
    while ``ni_ttm``/``gp_ttm`` are income-statement FLOWS already summed to
    trailing-twelve-months in the provider parser. ``known_at`` is the filing date
    at 21:00 UTC (public-availability time). ``accn`` (the SEC accession) is part
    of the natural key so a restatement lands as a distinct row.
    """

    symbol: str
    concept: str
    tag: str | None
    unit: str | None
    period_start: date | None
    period_end: date
    val: float | None
    fy: int | None
    fp: str | None
    form: str | None
    filed: date
    known_at: datetime
    accn: str | None
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
_SERIAL_TABLES = ("insider_trades", "analyst_ratings", "analyst_price_targets",
                  "fundamentals")


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


#: A daily refresh whose adj_close / close ratio steps by more than this against the stored
#: bars it meets splices two adjustment bases (engine proposal #52). Below it: rounding, or a
#: dividend going ex between two neighbouring bars that do not overlap.
SPLICE_TOL = 0.005


def splice_step(stored: tuple[float, float] | None, bar: PriceBar) -> float:
    """Relative step of the adjustment ratio ``adj_close / close`` from a stored bar to a new one.

    Args:
        stored: ``(close, adj_close)`` of the stored daily bar the new one meets, or ``None``.
        bar: The new bar.

    Returns:
        ``|ratio_new / ratio_stored − 1|``; 0.0 when either ratio is unknown.
    """
    if stored is None or not all((stored[0], stored[1], bar.close, bar.adj_close)):
        return 0.0
    close, adj = float(stored[0]), float(stored[1])
    return abs((bar.adj_close / bar.close) / (adj / close) - 1.0)


def adjustment_splice(engine: Engine, bars: list[PriceBar]) -> float:
    """The largest step a daily refresh would put into its symbol's stored adjusted history.

    Adjusted closes are a basis, not a fact: yfinance rescales a symbol's whole history after
    every dividend, and IBKR stores ``adj_close = close``. Rows from two downloads (or two
    providers) that meet therefore step at the junction. SPY's stored series jumped +3.43% on
    2023-10-17, where unadjusted IBKR rows met yfinance history adjusted for three later years
    of dividends, and every adjusted backtest read the jump as a return. The new bars are
    compared at both ends with the stored bar they meet: on the same date when the ranges
    overlap (an exact test), else with the neighbouring bar.

    Args:
        engine: Database engine.
        bars: One symbol's new daily bars.

    Returns:
        The larger of the two junction steps (:func:`splice_step`); 0.0 if nothing is stored.
    """
    if not bars:
        return 0.0
    bars = sorted(bars, key=lambda b: b.ts)
    first, last = bars[0], bars[-1]
    sql = """
        SELECT close, adj_close FROM prices_ohlcv
        WHERE symbol = :symbol AND bar = '1d' AND ts {op} :ts ORDER BY ts {order} LIMIT 1
    """
    with engine.connect() as conn:
        before = conn.execute(text(sql.format(op="<=", order="DESC")),
                              {"symbol": first.symbol, "ts": first.ts}).fetchone()
        after = conn.execute(text(sql.format(op=">=", order="ASC")),
                             {"symbol": last.symbol, "ts": last.ts}).fetchone()
    return max(splice_step(tuple(before) if before else None, first),
               splice_step(tuple(after) if after else None, last))


def stored_span(engine: Engine, symbol: str, bar: str = "1d") -> tuple[datetime, datetime] | None:
    """First and last stored bar timestamps of one symbol and bar size, or ``None``."""
    with engine.connect() as conn:
        row = conn.execute(text("SELECT min(ts), max(ts) FROM prices_ohlcv "
                                "WHERE symbol = :symbol AND bar = :bar"),
                           {"symbol": symbol, "bar": bar}).fetchone()
    return (row[0], row[1]) if row and row[0] is not None else None


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


def upsert_price_targets(engine: Engine, targets: list[PriceTarget]) -> int:
    """Insert analyst price targets, ignoring duplicates (key: ``analyst_price_targets_uq``).

    Args:
        engine: Target database engine.
        targets: Price-target events to write.

    Returns:
        Number of rows submitted.
    """
    sql = """
        INSERT INTO analyst_price_targets (symbol, firm, pt_action, pt_current, pt_prior,
                                           rated_at, known_at, source)
        VALUES (:symbol, :firm, :pt_action, :pt_current, :pt_prior,
                :rated_at, :known_at, :source)
        ON CONFLICT ON CONSTRAINT analyst_price_targets_uq DO NOTHING
    """
    return _executemany(engine, sql, [asdict(t) for t in targets])


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


def upsert_fundamentals(engine: Engine, rows: list[Fundamental]) -> int:
    """Insert fundamentals, ignoring duplicates (key: ``fundamentals_uq``).

    SEC facts are immutable per accession, so a conflict means we already have
    the row — ``DO NOTHING`` keeps re-runs idempotent (like :func:`upsert_insider`,
    unlike short interest's in-place revisions). A restatement carries a *new*
    accession and so lands as a distinct row, preserving point-in-time history.

    Args:
        engine: Target database engine.
        rows: Fundamental observations to write.

    Returns:
        Number of rows submitted.
    """
    sql = """
        INSERT INTO fundamentals (symbol, concept, tag, unit, period_start, period_end,
                                  val, fy, fp, form, filed, known_at, accn, source)
        VALUES (:symbol, :concept, :tag, :unit, :period_start, :period_end,
                :val, :fy, :fp, :form, :filed, :known_at, :accn, :source)
        ON CONFLICT ON CONSTRAINT fundamentals_uq DO NOTHING
    """
    return _executemany(engine, sql, [asdict(r) for r in rows])


def all_symbols(engine: Engine) -> list[str]:
    """Every symbol in the ``securities`` registry — the ``ALL`` alt-data target.

    Args:
        engine: Target database engine.

    Returns:
        Sorted list of every registered symbol (all instruments this DB tracks).
    """
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text("SELECT symbol FROM securities ORDER BY symbol"))]


def load_coverage(engine: Engine, dataset: str) -> dict[str, tuple[date, date]]:
    """Return the backfill ledger for one dataset as ``{symbol: (covered_from, covered_to)}``.

    Reads ``altdata_coverage`` — this repo's record of what alt-data has already been
    fetched, so a backfill can skip symbols already covered to the requested depth.

    Args:
        engine: Target database engine.
        dataset: Dataset name (e.g. ``"insider"``, ``"fundamentals"``).

    Returns:
        ``{symbol: (covered_from, covered_to)}`` for every symbol recorded for the
        dataset (empty when nothing has been ingested yet, or the table is absent).
    """
    sql = text("SELECT symbol, covered_from, covered_to FROM altdata_coverage "
               "WHERE dataset = :dataset")
    try:
        with engine.connect() as conn:
            return {r[0]: (r[1], r[2]) for r in conn.execute(sql, {"dataset": dataset})}
    except Exception:  # noqa: BLE001 — a missing ledger table must not block ingestion
        log.warning("altdata_coverage unavailable (run tsim init-db); treating as empty")
        return {}


def upsert_coverage(engine: Engine, dataset: str, rows: list[dict]) -> int:
    """Record/extend backfill coverage for ``(dataset, symbol)`` rows.

    Widens the stored window rather than replacing it (``covered_from`` takes the
    earliest, ``covered_to`` the latest ever seen), so repeated runs accumulate the
    full covered span. Each row is ``{"symbol", "covered_from", "covered_to", "rows"}``.

    Args:
        engine: Target database engine.
        dataset: Dataset the rows belong to.
        rows: One dict per symbol (see above).

    Returns:
        Number of rows submitted.
    """
    if not rows:
        return 0
    sql = """
        INSERT INTO altdata_coverage (dataset, symbol, covered_from, covered_to, rows, updated_at)
        VALUES (:dataset, :symbol, :covered_from, :covered_to, :rows, now())
        ON CONFLICT (dataset, symbol) DO UPDATE SET
            covered_from = LEAST(altdata_coverage.covered_from, EXCLUDED.covered_from),
            covered_to   = GREATEST(altdata_coverage.covered_to, EXCLUDED.covered_to),
            rows         = EXCLUDED.rows,
            updated_at   = now()
    """
    return _executemany(engine, sql, [{"dataset": dataset, **r} for r in rows])


# One row per alt-data event, tagged with its dataset and its own `source` column,
# normalised to (dataset, symbol, source, known_at, period_end) — the provenance
# feed both overview helpers aggregate. `period_end` is only meaningful for
# fundamentals; the others carry NULL.
_EVENT_UNION = """
    SELECT 'insider' AS dataset, symbol, source, known_at, NULL::date AS period_end
      FROM insider_trades         WHERE symbol = ANY(:syms)
    UNION ALL SELECT 'ratings', symbol, source, known_at, NULL::date
      FROM analyst_ratings        WHERE symbol = ANY(:syms)
    UNION ALL SELECT 'price_target', symbol, source, known_at, NULL::date
      FROM analyst_price_targets  WHERE symbol = ANY(:syms)
    UNION ALL SELECT 'short_interest', symbol, source, known_at, NULL::date
      FROM short_interest         WHERE symbol = ANY(:syms)
    UNION ALL SELECT 'fundamentals', symbol, source, known_at, period_end
      FROM fundamentals           WHERE symbol = ANY(:syms)
"""


def coverage_overview(engine: Engine, symbols: list[str], want_from: date) -> list[dict]:
    """Provenance overview per ``(dataset, source)`` for ``symbols`` (one row each).

    Joins the alt-data event tables (breadth, date range, per-``source`` split — so
    insider shows ``sec_edgar`` vs ``sec_edgar_bulk`` etc.) to the ``altdata_coverage``
    ledger. ``will_skip`` mirrors the actual backfill skip: a ``REFRESH=False`` backfill
    at ``want_from`` skips a symbol iff its ``covered_from <= want_from`` (the covered
    tail, not ``covered_to``, is update mode's job), so this is true when EVERY ledgered
    symbol is deep enough (``max(covered_from) <= want_from``). ``recorded_but_empty``
    counts symbols in the ledger with zero rows — the ledger-aware "thin table goes
    quiet" alarm. ``ledger_from``/``ledger_to`` show the deepest/latest covered dates.

    Args:
        engine: Target database engine.
        symbols: Universe to scope the report to.
        want_from: Requested backfill start (drives ``will_skip``).

    Returns:
        One dict per ``(dataset, source)`` (a fully-empty but ledgered dataset yields a
        single ``source=None`` row), ordered by dataset then source.
    """
    sql = text(f"""
        WITH ev AS ({_EVENT_UNION}),
        by_src AS (
            SELECT dataset, source, count(*) AS rows, count(DISTINCT symbol) AS symbols,
                   min(known_at) AS first_known_at, max(known_at) AS last_known_at,
                   max(period_end) AS latest_period_end
            FROM ev GROUP BY dataset, source
        ),
        led AS (
            SELECT dataset, count(*) AS ledger_symbols, min(covered_from) AS ledger_from,
                   max(covered_from) AS shallowest_from, max(covered_to) AS ledger_to
            FROM altdata_coverage WHERE symbol = ANY(:syms) GROUP BY dataset
        ),
        led_empty AS (
            SELECT c.dataset, count(*) AS recorded_but_empty
            FROM altdata_coverage c
            LEFT JOIN (SELECT DISTINCT dataset, symbol FROM ev) e
              ON e.dataset = c.dataset AND e.symbol = c.symbol
            WHERE c.symbol = ANY(:syms) AND e.symbol IS NULL
            GROUP BY c.dataset
        ),
        merged AS (
            SELECT dataset, source, rows, symbols, first_known_at, last_known_at, latest_period_end
              FROM by_src
            UNION ALL
            SELECT dataset, NULL, 0, 0, NULL, NULL, NULL
              FROM led WHERE dataset NOT IN (SELECT dataset FROM by_src)
        )
        SELECT m.dataset, m.source, m.rows, m.symbols, m.first_known_at, m.last_known_at,
               m.latest_period_end, l.ledger_symbols, l.ledger_from, l.ledger_to,
               COALESCE(le.recorded_but_empty, 0) AS recorded_but_empty,
               (l.shallowest_from IS NOT NULL AND l.shallowest_from <= :wf) AS will_skip
        FROM merged m
        LEFT JOIN led l ON l.dataset = m.dataset
        LEFT JOIN led_empty le ON le.dataset = m.dataset
        ORDER BY m.dataset, m.source NULLS FIRST
    """)
    params = {"syms": list(symbols), "wf": want_from}
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(sql, params)]


def coverage_overview_symbols(engine: Engine, symbols: list[str], want_from: date) -> list[dict]:
    """Per-``(dataset, symbol)`` coverage detail for ``symbols``.

    Full-outer-joins the event tables to the ledger so it surfaces both
    unledgered-but-present rows and ledgered-but-empty names. ``will_skip`` mirrors the
    backfill skip exactly (``covered_from <= want_from``); ``recorded_but_empty`` marks
    a symbol the ledger claims but the table has no rows for.

    Args:
        engine: Target database engine.
        symbols: Universe to scope the report to.
        want_from: Requested backfill start.

    Returns:
        One dict per ``(dataset, symbol)`` with data or a ledger entry.
    """
    sql = text(f"""
        WITH ev AS ({_EVENT_UNION}),
        by_sym AS (
            SELECT dataset, symbol, string_agg(DISTINCT source, ',') AS sources,
                   count(*) AS rows, min(known_at) AS first_known_at, max(known_at) AS last_known_at
            FROM ev GROUP BY dataset, symbol
        ),
        led AS (
            SELECT dataset, symbol, covered_from, covered_to
            FROM altdata_coverage WHERE symbol = ANY(:syms)
        )
        SELECT COALESCE(b.dataset, l.dataset) AS dataset,
               COALESCE(b.symbol, l.symbol) AS symbol,
               b.sources, COALESCE(b.rows, 0) AS rows,
               b.first_known_at, b.last_known_at, l.covered_from, l.covered_to,
               (l.covered_from IS NOT NULL AND l.covered_from <= :wf) AS will_skip,
               (l.covered_from IS NOT NULL AND b.symbol IS NULL) AS recorded_but_empty
        FROM by_sym b
        FULL OUTER JOIN led l ON l.dataset = b.dataset AND l.symbol = b.symbol
        ORDER BY dataset, symbol
    """)
    params = {"syms": list(symbols), "wf": want_from}
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(sql, params)]


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
