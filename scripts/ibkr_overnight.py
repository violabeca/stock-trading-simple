"""Download Interactive Brokers' OVERNIGHT session as 1-minute bars into ``prices_ohlcv``.

US stocks and funds trade on the main exchanges from 04:00 to 20:00 New York (pre-market, regular
session, after-hours); ``tsim backfill --extended-hours`` stores that. Between 20:00 and 04:00 only
a few special venues trade them; IBKR's own overnight venue ("OVERNIGHT") runs Sunday to Thursday
nights, 20:00 to 03:50 New York. Nothing trades on Saturday and Sunday daytime.

    uv run python scripts/ibkr_overnight.py --probe
    uv run python scripts/ibkr_overnight.py --symbols NOW,GEV,IGV,XLI,SPY \\
        --since 2025-01-01 --until 2026-10-16

``--probe`` writes nothing: it asks one recent week and one week starting at ``--since`` for the
first symbol and prints what came back (exit code 0 if both had overnight bars, 1 if none).

Rules: ONE connection, paced like the repo's own IBKR fetches; never run it next to another
IBKR download (the request cap is per account). Only minutes from 20:00 to 03:59 New York are
written, so a run can never overwrite the regular or extended-hours bars of the same symbols.
Bars are stored like every IBKR bar (``source`` "ibkr", ``rth`` False).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from trading_simple.config import load_settings
from trading_simple.db import PriceBar, make_engine, upsert_prices
from trading_simple.exchanges import resolve_exchange
from trading_simple.providers.ibkr import Pacer, fetch_window, frame_to_bars

STEP_DAYS = 7        # one request per week of nights (about 2,400 bars, well under IBKR's cap)
EMPTY_STOP = 3       # this many empty weeks in a row: IBKR's overnight history has ended


def windows(since: date, until: date, step: int = STEP_DAYS) -> list[tuple[str, str]]:
    """``(endDateTime, durationStr)`` requests covering the nights from ``since`` to ``until``
    (a night is named by its evening), newest first, each ending at 04:00 New York the morning
    after its last night."""
    out, last = [], until
    while last >= since:
        span = min(step, (last - since).days + 1)
        out.append((f"{last + timedelta(days=1):%Y%m%d} 04:00:00 US/Eastern", f"{span} D"))
        last -= timedelta(days=step)
    return out


def overnight_only(bars: list[PriceBar]) -> list[PriceBar]:
    """The bars that open between 20:00 and 03:59 New York (the overnight session)."""
    keep = []
    for b in bars:
        hour = pd.Timestamp(b.ts).tz_convert("America/New_York").hour
        if hour >= 20 or hour < 4:
            keep.append(b)
    return keep


def fetch(ib, pacer, symbol: str, plan: list[tuple[str, str]], exchanges) -> list[PriceBar]:
    """Every overnight 1-minute bar of ``symbol`` in the request ``plan``. A window IBKR never
    answers (``fetch_window`` already asked it again) raises, so the symbol is not stored with a
    hole."""
    from ib_async import Stock, util

    spec = resolve_exchange(symbol, exchanges)
    contract = Stock(symbol, "OVERNIGHT", "USD")
    ib.qualifyContracts(contract)
    frames, empty = [], 0
    for end, duration in plan:
        raw, status = fetch_window(ib, contract, end, duration, bar="1m", use_rth=False,
                                   pacer=pacer)
        if status == "timeout":
            raise TimeoutError(f"IBKR never answered for {symbol} (window ending {end})")
        if status == "empty":
            empty += 1
            if empty >= EMPTY_STOP:
                break
            continue
        empty = 0
        frames.append(util.df(raw))
    if not frames:
        return []
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"])
    return overnight_only(frame_to_bars(symbol, "1m", frame, spec))


def describe(symbol: str, bars: list[PriceBar]) -> str:
    """One line: how many bars, from when to when, at which New York hours."""
    if not bars:
        return f"overnight {symbol} 1m: 0 bars"
    t = pd.DatetimeIndex([pd.Timestamp(b.ts) for b in bars]).tz_convert("America/New_York")
    hours = sorted(set(t.hour))
    return (f"overnight {symbol} 1m: {len(bars)} bars, {t.min():%Y-%m-%d %H:%M} -> "
            f"{t.max():%Y-%m-%d %H:%M} New York, hours {hours}, nights {t.normalize().nunique()}")


def main() -> int:
    """Probe or download; see the module docstring."""
    from ib_async import IB

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--symbols", default="SPY")
    p.add_argument("--since", default="2025-01-01")
    p.add_argument("--until", default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--client-id", type=int, default=11)
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    settings = load_settings()
    since, until = date.fromisoformat(args.since), date.fromisoformat(args.until)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    ib = IB()
    ib.connect(settings.ibkr.host, settings.ibkr.port, clientId=args.client_id, timeout=15)
    pacer = Pacer(settings.ibkr.max_requests_per_10min, min_gap_s=settings.ibkr.pace_seconds)
    try:
        if args.probe:
            found = 0
            for label, plan in (("recent week", windows(until - timedelta(days=6), until)),
                                ("first week", windows(since, since + timedelta(days=6)))):
                bars = fetch(ib, pacer, symbols[0], plan, settings.exchanges)
                print(f"PROBE {label}: {describe(symbols[0], bars)}", flush=True)
                found += bool(bars)
            return 0 if found == 2 else 1
        engine = make_engine(settings)
        for symbol in symbols:
            try:
                bars = fetch(ib, pacer, symbol, windows(since, until), settings.exchanges)
            except Exception as exc:  # noqa: BLE001 (one symbol must not stop the others)
                print(f"overnight {symbol}: FAILED {exc!r}", flush=True)
                continue
            if bars:
                upsert_prices(engine, bars)
            print(describe(symbol, bars), flush=True)
        return 0
    finally:
        ib.disconnect()


if __name__ == "__main__":
    sys.exit(main())
