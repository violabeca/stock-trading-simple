#!/usr/bin/env python3
"""Download MarketData.app option chains and news for the names with good 1m coverage in the DB.

A trial (and the free plan after it) reaches back ONE YEAR for every symbol but AAPL, so a day
not downloaded before it slides out of that window is gone. Price bars come from IBKR / Yahoo
into the DB, never from here; per symbol it stores:

- ``chains``     the end-of-day option chain of every session, all expiries: bid/ask + sizes,
                 last, volume, OPEN INTEREST per contract (historical chains carry no IV/Greeks),
                 ~1 credit per 1,000 contracts;
- ``news``       headlines + summaries (beta endpoint), 1 credit per month.

Symbols: every name with at least ``--min-days`` sessions of 1m bars in the shared DB, deepest
first (cached in ``symbols.csv``; ``--refresh-symbols`` re-reads the DB).

Resuming: every chunk lands as ``var/marketdata/<dataset>/<SYMBOL>/<first>_<last>.csv.gz``
(written atomically, an empty file = asked, nothing there), and a run plans only the calendar
days no file name covers. The OLDEST month goes first for every symbol, since it expires first.
When the daily credits run out (429, or ``--reserve`` left) the run stops cleanly; rerun after
the 09:30 New York reset, or pass ``--wait`` to sleep through it and carry on. A chunk that
reaches into the last few days is closed at its last bar, so tomorrow's run fills the rest.

Usage:
    uv run python scripts/marketdata_dump.py --plan        # what is left; no API calls
    uv run python scripts/marketdata_dump.py               # until today's credits run out
    uv run python scripts/marketdata_dump.py --wait        # ... and every following day
    uv run python scripts/marketdata_dump.py --datasets chains --limit-symbols 5
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "var" / "marketdata"
API = "https://api.marketdata.app/v1"
NY = ZoneInfo("America/New_York")
RECENT_DAYS = 4  # a chunk ending this close to today may simply not be published yet

# name -> (path, fixed params, requests: "month" = one per month, "day" = one per session,
# stored per month). Order = priority inside a month.
DATASETS = {
    "chains": ("options/chain/{sym}/", {"expiration": "all"}, "day"),
    "news": ("stocks/news/{sym}/", {}, "month"),
}

STOP = threading.Event()
LOCK = threading.Lock()
STATE: dict = {"credits": 0, "remaining": None, "reset": None, "fatal": None}
_local = threading.local()


class Quota(Exception):
    """The day's credits are used up (or down to ``--reserve``)."""


def token() -> str:
    """The MarketData.app token: env var ``MARKETDATA_TOKEN``, else the repo's ``.env``.

    Returns:
        The token (never printed).
    """
    tok = os.environ.get("MARKETDATA_TOKEN", "")
    env = ROOT / ".env"
    if not tok and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("MARKETDATA_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip("\"'")
    if not tok:
        sys.exit("MARKETDATA_TOKEN not set (env or .env)")
    return tok


def symbols(min_days: int, refresh: bool = False, out: Path = OUT) -> list[str]:
    """Names with at least ``min_days`` sessions of 1m bars in the DB, deepest first.

    The DB scan takes ~40 s, so its result (every 1m symbol with its session count) is cached
    in ``symbols.csv``; resumed runs then work on the same list.

    Args:
        min_days: Minimum number of New York sessions with 1m bars.
        refresh: Re-read the DB even if the cache exists.
        out: Output root.

    Returns:
        Symbols ordered by sessions descending, then name.
    """
    cache = out / "symbols.csv"
    if refresh or not cache.exists():
        from sqlalchemy import text

        from trading_simple.db import make_engine

        sql = text("""SELECT symbol, count(DISTINCT (ts AT TIME ZONE 'America/New_York')::date)
                      AS days FROM prices_ohlcv WHERE bar = '1m' GROUP BY symbol""")
        with make_engine().connect() as conn:
            df = pd.read_sql(sql, conn)
        out.mkdir(parents=True, exist_ok=True)
        df.sort_values(["days", "symbol"], ascending=[False, True]).to_csv(cache, index=False)
    df = pd.read_csv(cache)
    return df.loc[df["days"] >= min_days, "symbol"].tolist()


def _days(a: date, b: date) -> list[date]:
    """Every calendar day from ``a`` to ``b`` inclusive."""
    return [a + timedelta(days=i) for i in range((b - a).days + 1)]


def covered(folder: Path) -> set[date]:
    """Calendar days the stored chunk files of one (dataset, symbol) already cover.

    Args:
        folder: ``<out>/<dataset>/<SYMBOL>``.

    Returns:
        Every day inside some ``<first>_<last>.csv.gz`` file name.
    """
    days: set[date] = set()
    for f in folder.glob("*.csv.gz"):
        first, last = f.name[:21].split("_")
        days.update(_days(date.fromisoformat(first), date.fromisoformat(last)))
    return days


def plan(syms: list[str], datasets: list[str], start: date, end: date,
         out: Path = OUT) -> list[tuple[str, str, date, date]]:
    """The chunks still to fetch, oldest month first, then dataset order, then symbol order.

    Args:
        syms: Symbols in priority order.
        datasets: Dataset names (keys of ``DATASETS``) in priority order.
        start: First day of the window.
        end: Last day of the window.
        out: Output root.

    Returns:
        ``(dataset, symbol, first, last)`` per chunk, one per run of uncovered days.
    """
    months = [(max(m.date(), start), min((m + pd.offsets.MonthEnd(0)).date(), end))
              for m in pd.date_range(start.replace(day=1), end, freq="MS")]
    tasks = []
    for si, sym in enumerate(syms):
        for di, ds in enumerate(datasets):
            have = covered(out / ds / sym)
            for a, b in months:  # one chunk per run of missing days; a covered day ends a run
                run: list[date] = []
                for d in _days(a, b) + [None]:
                    if d is None or d in have:
                        if run:
                            tasks.append((a, di, si, ds, sym, run[0], run[-1]))
                        run = []
                    else:
                        run.append(d)
    return [t[3:] for t in sorted(tasks)]


def call(path: str, params: dict, tok: str, reserve: int) -> dict:
    """One GET with retries; tracks credits and stops everything at the quota.

    Args:
        path: Endpoint path below ``/v1/``.
        params: Query parameters.
        tok: API token.
        reserve: Stop starting requests once this many credits are left.

    Returns:
        The JSON body, ``s`` = ``ok`` or ``no_data``.

    Raises:
        Quota: Credits exhausted (or a fatal account error, recorded in ``STATE``).
        RuntimeError: Any other error after retries.
    """
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    for attempt in range(5):
        if STOP.is_set():
            raise Quota
        try:
            r = _local.session.get(f"{API}/{path}", params=params, timeout=180,
                                   headers={"Authorization": f"Bearer {tok}"})
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        h = r.headers
        with LOCK:
            STATE["credits"] += int(h.get("x-api-ratelimit-consumed") or 0)
            if "x-api-ratelimit-remaining" in h:
                STATE["remaining"] = int(h["x-api-ratelimit-remaining"])
                STATE["reset"] = int(h["x-api-ratelimit-reset"])
        try:
            j = r.json()
        except ValueError:
            j = {}
        msg = str(j.get("errmsg", ""))
        if r.status_code == 429 and "concurrent" in msg.lower() or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            STOP.set()
            raise Quota(msg)
        if r.status_code == 402 and "fully-closed" in msg:  # newest sessions: not served yet
            return {"s": "no_data"}
        if r.status_code in (401, 402, 403):  # bad token, trial over, or outside the window
            STATE["fatal"] = f"{r.status_code} {path}: {msg}"
            STOP.set()
            raise Quota(msg)
        if STATE["remaining"] is not None and STATE["remaining"] <= reserve:
            STOP.set()  # keep this answer, start nothing new
        if j.get("s") in ("ok", "no_data"):
            return j
        raise RuntimeError(f"{r.status_code}: {msg or r.text[:200]}")
    raise RuntimeError(f"gave up after 5 tries: {path}")


def frame(j: dict) -> pd.DataFrame:
    """MarketData's column-array JSON -> a DataFrame (scalars such as ``s`` dropped)."""
    return pd.DataFrame({k: v for k, v in j.items() if isinstance(v, list)})


def last_day(df: pd.DataFrame) -> date | None:
    """The New York date of a chunk's newest row (``None`` when empty)."""
    if df.empty:
        return None
    if "date" in df:
        return date.fromisoformat(df["date"].max())
    return pd.to_datetime(df["publicationDate"].max(), unit="s", utc=True).tz_convert(NY).date()


def write(df: pd.DataFrame, path: Path) -> None:
    """Write a chunk atomically, so an interrupted run never leaves a file that reads as done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)


def fetch(ds: str, sym: str, a: date, b: date, tok: str, reserve: int, today: date,
          out: Path = OUT) -> int:
    """Download one chunk and store it as ``<first>_<last>.csv.gz``.

    Args:
        ds: Dataset name.
        sym: Symbol.
        a: First day.
        b: Last day.
        tok: API token.
        reserve: See :func:`call`.
        today: Today in New York.
        out: Output root.

    Returns:
        Rows stored (0 when a recent chunk had nothing yet and was left for the next run).
    """
    path, fixed, kind = DATASETS[ds]
    path = path.format(sym=sym)
    if kind == "day":
        frames = []
        for d in pd.bdate_range(a, b).date:  # holidays answer no_data for free
            f = frame(call(path, {**fixed, "date": d.isoformat()}, tok, reserve))
            if len(f):
                f.insert(0, "date", d.isoformat())
                frames.append(f)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = frame(call(path, {**fixed, "from": a.isoformat(), "to": b.isoformat()},
                        tok, reserve))
    if b > today - timedelta(days=RECENT_DAYS):  # close a recent chunk at its last row
        if (last := last_day(df)) is None:
            return 0
        b = min(b, last)
    write(df, out / ds / sym / f"{a}_{b}.csv.gz")
    return len(df)


def execute(tasks: list, tok: str, reserve: int, workers: int, today: date,
            out: Path = OUT) -> None:
    """Run the chunks on a thread pool (MarketData allows 50 concurrent requests) until done or
    out of credits, printing progress every minute and logging errors to ``errors.log``.

    Args:
        tasks: Output of :func:`plan`.
        tok: API token.
        reserve: See :func:`call`.
        workers: Threads.
        today: Today in New York.
        out: Output root.
    """
    done = errors = rows = 0
    t0 = shown = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fetch, *t, tok, reserve, today, out): t for t in tasks}
        for fut in as_completed(futs):
            if fut.cancelled():
                continue
            try:
                rows += fut.result()
                done += 1
            except Quota:
                for f in futs:
                    f.cancel()
            except Exception as e:  # noqa: BLE001 - logged; the chunk is retried next run
                errors += 1
                with open(out / "errors.log", "a") as log:
                    log.write(f"{datetime.now():%F %T}\t{futs[fut]}\t{e}\n")
            if time.time() - shown > 60:
                shown = time.time()
                rate = done / (shown - t0) * 3600
                print(f"  {done:,}/{len(tasks):,} chunks ({rate:,.0f}/h), {rows:,} rows, "
                      f"{STATE['credits']:,} credits, {STATE['remaining']} left, {errors} errors",
                      flush=True)
    print(f"run: {done:,} chunks, {rows:,} rows, {STATE['credits']:,} credits, {errors} errors"
          + (f" (see {out / 'errors.log'})" if errors else ""), flush=True)


def load(dataset: str, syms: list[str] | None = None, out: Path = OUT) -> pd.DataFrame:
    """Every stored chunk of a dataset in one frame, with a ``symbol`` column.

    Args:
        dataset: Dataset name.
        syms: Restrict to these symbols (default: all stored).
        out: Output root.

    Returns:
        The rows, duplicates (from overlapping re-fetches) dropped.
    """
    frames = []
    for folder in sorted((out / dataset).glob("*")):
        if syms is not None and folder.name not in syms:
            continue
        for f in sorted(folder.glob("*.csv.gz")):
            with contextlib.suppress(pd.errors.EmptyDataError):  # an asked-but-empty chunk
                frames.append(pd.read_csv(f).assign(symbol=folder.name))
    return pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring.

    Args:
        argv: Arguments (default ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--datasets", default=",".join(DATASETS),
                   help="comma list, priority order (default: %(default)s)")
    p.add_argument("--min-days", type=int, default=100, help="1m sessions in the DB to qualify")
    p.add_argument("--limit-symbols", type=int, help="only the N best-covered names")
    p.add_argument("--workers", type=int, default=8, help="parallel requests (API max 50)")
    p.add_argument("--reserve", type=int, default=500, help="credits to leave unused per day")
    p.add_argument("--wait", action="store_true", help="sleep to the daily reset, continue")
    p.add_argument("--plan", action="store_true", help="print what is left and exit")
    p.add_argument("--refresh-symbols", action="store_true", help="re-read the symbol list")
    args = p.parse_args(argv)
    datasets = args.datasets.split(",")
    if unknown := set(datasets) - set(DATASETS):
        p.error(f"unknown datasets: {sorted(unknown)}")
    tok = "" if args.plan else token()
    syms = symbols(args.min_days, args.refresh_symbols)[:args.limit_symbols]
    while True:
        today = datetime.now(NY).date()
        start, end = today - timedelta(days=364), today - timedelta(days=1)
        tasks = plan(syms, datasets, start, end)
        reqs = {ds: sum(len(pd.bdate_range(a, b)) if DATASETS[ds][2] == "day" else 1
                        for d, _, a, b in tasks if d == ds) for ds in datasets}
        print(f"{datetime.now(NY):%F %H:%M} NY | {len(syms)} symbols | {start} -> {end} | "
              f"{len(tasks):,} chunks left | requests: {reqs}", flush=True)
        if args.plan or not tasks:
            return 0
        STOP.clear()
        execute(tasks, tok, args.reserve, args.workers, today)
        if STATE["fatal"]:
            print(f"stopped: {STATE['fatal']}")
            return 1
        if not STOP.is_set():
            return 0  # finished; anything that errored is retried by the next run
        if not args.wait:
            print("out of credits for today: rerun after 09:30 New York (or use --wait)")
            return 0
        wake = (STATE["reset"] or time.time() + 3600) + 120
        print(f"out of credits: sleeping until {datetime.fromtimestamp(wake, NY):%F %H:%M} NY",
              flush=True)
        time.sleep(max(0.0, wake - time.time()))
        STATE["credits"] = 0


if __name__ == "__main__":
    sys.exit(main())
