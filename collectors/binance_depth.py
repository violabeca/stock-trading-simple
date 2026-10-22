#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets>=13", "psycopg[binary]>=3.1"]
# [tool.uv]
# exclude-newer = "2026-10-01T00:00:00Z"
# ///
"""Binance USDⓈ-M futures order books, snapshotted every few seconds into TimescaleDB.

Ported from `trading_strategy/src/data_extraction/stream_order_book.py` (2026-05, the newest of
the old collectors) into one self-contained script: `uv run` installs its two dependencies, so it
runs the same on a laptop and on a VM (`collectors/docker-compose.yml`). It writes the SAME
`market_depth` table (time, symbol, side, price, quantity) the old collector filled, so code that
reads the old data reads this too.

How a book stays correct (Binance's documented futures procedure):
  1. open `<symbol>@depth@100ms` and buffer its diff events;
  2. fetch a REST snapshot (1000 levels a side) while buffering;
  3. drop buffered events with u < lastUpdateId; the first kept one must have U <= lastUpdateId;
  4. apply it and every later event whose `pu` equals the previous event's `u`;
  5. a `pu` break re-runs 2-4 on the open connection.

Differences from the old collector, each a data-quality fix:
  * Continuity is checked on `pu`. On futures `U` is never the previous `u` + 1; the old replay
    step checked `U` and so reconnected whenever more than one buffered event had to be replayed.
  * The book is rebuilt from a fresh snapshot every `--refresh` seconds. The diff stream only
    reports levels that CHANGE, so past the snapshot's 1000 levels the book holds just the levels
    touched since the sync: the old collector grew to 2000 levels a side and stored that partial
    tail. Saved levels are now clipped to the price range the last snapshot covered (for BTC
    that is about ±0.17% around the mid), and at most `--levels` (<= 1000) a side.
  * A book that is out of sync, or has had no update for `STALE_AFTER` seconds, is not saved.
    The old saver kept writing the last book during a reconnect as if it were live.

Usage:
    uv run collectors/binance_depth.py --dsn postgresql://postgres:PW@localhost:5433/market_depth_db
    DB_URL_DEPTH=... uv run collectors/binance_depth.py --symbols BTCUSDT,ETHUSDT --levels 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

import psycopg
import websockets

REST = "https://fapi.binance.com/fapi/v1"
WS = "wss://fstream.binance.com/ws"
SYMBOLS = "BTCUSDT,ETHUSDT,XLMUSDT,ADAUSDT,SOLUSDT,DOGEUSDT,XRPUSDT,AVAXUSDT,DOTUSDT,LTCUSDT"
SNAPSHOT_LEVELS = 1000  # REST depth limit on futures (weight 20 a call)
STREAM_TIMEOUT = 10.0  # seconds without a message before reconnecting
STALE_AFTER = 10.0  # a book with no applied update for this long is not saved
RECONNECT_DELAY = 5.0  # a reconnect costs one snapshot: 10 symbols looping stay < 2400 weight/min
RETRY_DELAY = 10.0  # pause before re-fetching a snapshot that failed or could not be bridged
REST_BUDGET = 1200  # max REST weight/min the refreshes may use: half of Binance's 2400/min an IP
HEARTBEAT = 60.0  # seconds between status log lines

COLUMNS = ("time", "symbol", "side", "price", "quantity")
# Run only when the table does not exist yet: an existing table is validated, never altered.
# One-hour chunks compressed after two keep the uncompressed tail (~83 B/row) to a few GB.
# Ordering by price makes each compressed batch ONE level's quantity over time, which mostly
# repeats: 3.1 B/row measured, against 14.4 for the old collector's time-first order.
CREATE = """
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE TABLE market_depth (
    time     TIMESTAMPTZ      NOT NULL,
    symbol   TEXT             NOT NULL,
    side     TEXT             NOT NULL,
    price    DOUBLE PRECISION NOT NULL,
    quantity DOUBLE PRECISION NOT NULL
);
SELECT create_hypertable('market_depth', by_range('time', INTERVAL '1 hour'));
CREATE INDEX ON market_depth (symbol, time DESC);
ALTER TABLE market_depth SET (timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, side',
    timescaledb.compress_orderby = 'price, time DESC');
SELECT add_compression_policy('market_depth', INTERVAL '2 hours');
"""

log = logging.getLogger("binance_depth")


class Desync(Exception):
    """The diff events do not chain onto the book or cannot bridge the snapshot."""


def apply(book: dict, event: dict) -> None:
    """Apply one depthUpdate's absolute quantities to a book in place.

    Args:
        book: `{"bids": {price: qty}, "asks": {price: qty}, "u": last applied update id, ...}`.
        event: a futures depthUpdate (`b`/`a` lists of [price, qty] strings, `u`); qty 0 removes.
    """
    for side, key in (("bids", "b"), ("asks", "a")):
        levels = book[side]
        for p, q in event[key]:
            p, q = float(p), float(q)
            if q:
                levels[p] = q
            else:
                levels.pop(p, None)  # removing a level the book never had is normal
    book["u"] = event["u"]


def rebuild(snapshot: dict, events: list[dict]) -> dict | None:
    """Build a synced book from a REST snapshot plus the diff events buffered around it.

    Args:
        snapshot: `/fapi/v1/depth` response (`lastUpdateId`, `bids`, `asks`).
        events: depthUpdate events received since the snapshot was requested, in order.

    Returns:
        The book (with `lo`/`hi`, the lowest bid and highest ask the snapshot covered), or None
        when no event reaches the snapshot yet: it is newer than the stream so far, keep
        buffering and call again.

    Raises:
        Desync: the events cannot bridge the snapshot (a gap); fetch a new one.
    """
    last_id = snapshot["lastUpdateId"]
    events = [e for e in events if e["u"] >= last_id]
    if not events:
        return None
    if events[0]["U"] > last_id:
        raise Desync(f"snapshot {last_id} predates the buffered stream (first U {events[0]['U']})")
    book = {side: {float(p): float(q) for p, q in snapshot[side]} for side in ("bids", "asks")}
    book.update(lo=min(book["bids"], default=0.0), hi=max(book["asks"], default=float("inf")))
    book["u"] = events[0]["pu"]  # so the first event chains by construction
    for e in events:
        if e["pu"] != book["u"]:
            raise Desync(f"gap in buffered events (pu {e['pu']} != u {book['u']})")
        apply(book, e)
    return book


def snapshot_rows(books: dict, levels: int, ts: datetime, now: float) -> tuple[list, list]:
    """Rows to save for every live book: the top `levels` a side inside the snapshot's range.

    Args:
        books: symbol -> book from `rebuild` (plus `t`, monotonic time of its last update),
            or None while that symbol is not in sync.
        levels: maximum levels a side.
        ts: the snapshot timestamp written to every row.
        now: current `time.monotonic()`, to skip books that stopped updating.

    Returns:
        `(rows, skipped)`: `(time, symbol, side, price, quantity)` tuples, and the symbols left out.
    """
    rows, skipped = [], []
    for symbol, book in books.items():
        if book is None or now - book["t"] > STALE_AFTER:
            skipped.append(symbol)
            continue
        bids = [(p, q) for p, q in sorted(book["bids"].items(), reverse=True) if p >= book["lo"]]
        asks = [(p, q) for p, q in sorted(book["asks"].items()) if p <= book["hi"]]
        rows += [(ts, symbol, "bid", p, q) for p, q in bids[:levels]]
        rows += [(ts, symbol, "ask", p, q) for p, q in asks[:levels]]
    return rows, skipped


def get_json(url: str) -> dict:
    """GET a Binance REST endpoint (stdlib, blocking: call it through `asyncio.to_thread`)."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


async def fetch_snapshot(symbol: str, delay: float = 0.0) -> dict:
    """Fetch a 1000-level REST snapshot for `symbol`, optionally after `delay` seconds."""
    await asyncio.sleep(delay)
    url = f"{REST}/depth?symbol={symbol}&limit={SNAPSHOT_LEVELS}"
    return await asyncio.to_thread(get_json, url)


async def stream(symbol: str, books: dict, refresh: float) -> None:
    """Keep `books[symbol]` in sync with Binance forever: connect, sync, follow, resync, reconnect.

    `books[symbol]` is None whenever the book is not trustworthy (connecting, or resyncing after
    a gap), so the saver skips it. A periodic refresh swaps in a book rebuilt from a new snapshot
    without dropping the live one: events keep applying to it while the snapshot is in flight.

    Args:
        symbol: uppercase futures symbol, e.g. BTCUSDT.
        books: the shared symbol -> book registry the saver reads.
        refresh: seconds between snapshot rebuilds of a live book.
    """
    url = f"{WS}/{symbol.lower()}@depth@100ms"
    while True:
        books[symbol] = fetch = None
        try:
            async with websockets.connect(url, open_timeout=10) as ws:
                pending, snapshot, next_refresh = [], None, 0.0
                while True:
                    event = json.loads(await asyncio.wait_for(ws.recv(), STREAM_TIMEOUT))
                    if event.get("e") != "depthUpdate":
                        continue
                    book = books[symbol]
                    if book is not None and event["pu"] != book["u"]:
                        log.warning("%s: gap (pu %s != u %s), resyncing", symbol, event["pu"],
                                    book["u"])
                        books[symbol] = book = None
                    elif book is not None:
                        apply(book, event)
                        book["t"] = time.monotonic()
                    if fetch is None and (book is None or time.monotonic() >= next_refresh):
                        fetch = asyncio.create_task(fetch_snapshot(symbol))
                        pending, snapshot, next_refresh = [], None, time.monotonic() + refresh
                    if fetch is None:
                        continue
                    pending.append(event)
                    if snapshot is None and fetch.done():
                        if (err := fetch.exception()) is not None:  # HTTP error or timeout
                            log.warning("%s: snapshot failed: %r", symbol, err)
                            fetch = None if book is not None else asyncio.create_task(
                                fetch_snapshot(symbol, RETRY_DELAY))  # a live book carries on
                            continue
                        snapshot = fetch.result()
                    if snapshot is None:
                        continue
                    try:
                        fresh = rebuild(snapshot, pending)
                    except Desync as e:
                        log.warning("%s: %s, fetching a new snapshot", symbol, e)
                        fetch = asyncio.create_task(fetch_snapshot(symbol, RETRY_DELAY))
                        pending, snapshot = [], None
                        continue
                    if fresh is not None:
                        if book is None:
                            log.info("%s: in sync at u=%s (%d bids, %d asks)", symbol, fresh["u"],
                                     len(fresh["bids"]), len(fresh["asks"]))
                        fresh["t"] = time.monotonic()
                        books[symbol], fetch, pending, snapshot = fresh, None, [], None
        except Exception as e:  # dropped connection, timeout, bad message: start over
            log.warning("%s: %s: %s, reconnecting", symbol, type(e).__name__, e)
        finally:
            books[symbol] = None
            if fetch is not None:
                fetch.cancel()
        await asyncio.sleep(RECONNECT_DELAY)


async def ensure_table(conn: psycopg.AsyncConnection) -> None:
    """Create `market_depth` as a compressed hypertable if missing; else check its columns.

    Raises:
        SystemExit: an existing table's columns differ from `COLUMNS` (no row is written).
    """
    cur = await conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'market_depth'")
    found = {r[0] for r in await cur.fetchall()}
    if not found:
        async with conn.transaction():
            await conn.execute(CREATE)
        log.info("created hypertable market_depth")
    elif not set(COLUMNS) <= found:
        raise SystemExit(f"market_depth lacks columns {set(COLUMNS) - found}; refusing to write")


async def save(dsn: str, books: dict, levels: int, interval: float) -> None:
    """Every `interval` seconds, on the wall-clock boundary, COPY every live book to the DB.

    A failed insert is logged and that snapshot dropped; the next one reconnects.

    Args:
        dsn: PostgreSQL connection string of the database holding `market_depth`.
        books: the shared symbol -> book registry the streams fill.
        levels: maximum levels a side to save.
        interval: seconds between snapshots; timestamps are exact multiples of it.
    """
    conn, saved, failed, rows_saved, beat = None, 0, 0, 0, time.monotonic()
    sql = f"COPY market_depth ({', '.join(COLUMNS)}) FROM STDIN"
    due = (time.time() // interval + 1) * interval
    while True:
        await asyncio.sleep(max(due - time.time(), 0.0))
        ts = datetime.fromtimestamp(due, UTC)
        rows, skipped = snapshot_rows(books, levels, ts, time.monotonic())
        try:
            if conn is None or conn.closed:
                conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
            async with conn.cursor() as cur, cur.copy(sql) as copy:
                for row in rows:
                    await copy.write_row(row)
            saved, rows_saved = saved + 1, rows_saved + len(rows)
        except Exception as e:
            failed += 1
            log.error("insert of %d rows failed: %s: %s", len(rows), type(e).__name__, e)
            if conn is not None:
                await conn.close()
        if time.monotonic() - beat >= HEARTBEAT:
            log.info("saved %d snapshots (%d rows each), %d failed; out of sync: %s", saved,
                     rows_saved // max(saved, 1), failed, ", ".join(skipped) or "none")
            saved, failed, rows_saved, beat = 0, 0, 0, time.monotonic()
        # the next boundary still ahead: a slow insert skips a slot rather than writing it late
        due = max(due + interval, (time.time() // interval + 1) * interval)


def check_symbols(symbols: list[str]) -> None:
    """Fail fast, with a readable reason, when Binance is unreachable or a symbol is not trading.

    Raises:
        SystemExit: HTTP 451 (Binance refuses this IP's location, e.g. any US cloud region),
            another HTTP error, or a symbol that is not a TRADING USDⓈ-M futures contract.
    """
    try:
        info = get_json(f"{REST}/exchangeInfo")
    except urllib.error.HTTPError as e:
        hint = " (restricted location: run from a non-US region)" if e.code == 451 else ""
        raise SystemExit(f"Binance futures REST refused us: HTTP {e.code}{hint}") from e
    status = {s["symbol"]: s["status"] for s in info["symbols"]}
    bad = [s for s in symbols if status.get(s) != "TRADING"]
    if bad:
        raise SystemExit(f"not TRADING on Binance USDⓈ-M futures: {bad}")


async def run(args: argparse.Namespace, symbols: list[str]) -> None:
    """Validate symbols and table, then stream every symbol and save snapshots until cancelled."""
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):  # docker stop / systemd send SIGTERM
        loop.add_signal_handler(sig, task.cancel)
    check_symbols(symbols)
    db = psycopg.conninfo.conninfo_to_dict(args.dsn)  # log the target, never the password
    log.info("symbols %s | %d levels/side every %gs | refresh %gs | db %s@%s:%s/%s",
             ",".join(symbols), args.levels, args.interval, args.refresh, db.get("user"),
             db.get("host"), db.get("port", 5432), db.get("dbname"))
    async with await psycopg.AsyncConnection.connect(args.dsn, autocommit=True) as conn:
        await ensure_table(conn)
    books = dict.fromkeys(symbols)
    await asyncio.gather(save(args.dsn, books, args.levels, args.interval),
                         *(stream(s, books, args.refresh) for s in symbols))


def main() -> None:
    """Parse arguments and run until SIGINT/SIGTERM."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dsn", default=os.environ.get("DB_URL_DEPTH"),
                        help="PostgreSQL DSN (default: $DB_URL_DEPTH)")
    parser.add_argument("--symbols", default=os.environ.get("DEPTH_SYMBOLS") or SYMBOLS,
                        help="comma-separated USDⓈ-M futures symbols (default: $DEPTH_SYMBOLS)")
    parser.add_argument("--levels", type=int, default=SNAPSHOT_LEVELS,
                        help=f"levels a side to save, at most {SNAPSHOT_LEVELS}")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between snapshots")
    parser.add_argument("--refresh", type=float, default=30.0,
                        help="seconds between snapshot rebuilds of a live book")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not args.dsn:
        parser.error("no database: pass --dsn or set DB_URL_DEPTH")
    if not 0 < args.levels <= SNAPSHOT_LEVELS:
        parser.error(f"--levels must be 1..{SNAPSHOT_LEVELS}: past that the book is incomplete")
    if args.interval <= 0 or len(symbols) * 20 * 60 / args.refresh > REST_BUDGET:
        parser.error(f"--interval must be > 0 and {len(symbols)} symbols need --refresh >= "
                     f"{len(symbols) * 20 * 60 / REST_BUDGET:g}s (snapshots cost 20 weight each)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(run(args, symbols))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("stopped")


if __name__ == "__main__":
    main()
