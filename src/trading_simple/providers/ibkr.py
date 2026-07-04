"""Intraday OHLCV bars from Interactive Brokers (via ``ib_async``).

IBKR is the source for deep 1-minute history. Two hard rules, learned live:

1. **One connection, one pacer.** IBKR caps historical requests at ~60 per
   10 minutes *per account*, so fanning out across processes buys nothing and
   trips pacing violations. :func:`fetch_batch` pulls every symbol over a single
   persistent connection, spaced by one shared :class:`Pacer`.
2. **1-minute data is paged in 10-day windows.** IB applies an escalating
   response penalty above ~10k bars per request; a 30-day extended-hours 1m
   request silently returns nothing.

Requires IB Gateway or TWS running and logged in (paper Gateway = port 4002).
The pure planning/parsing helpers (:func:`request_windows`,
:func:`contract_args`, :func:`frame_to_bars`, :class:`Pacer`) have no ``ib_async``
import, so they are unit-testable without a Gateway.
"""

from __future__ import annotations

import logging
import time as _time
from collections import deque
from collections.abc import Callable
from datetime import date, datetime, timedelta

from trading_simple.config import IBKRSettings
from trading_simple.db import PriceBar
from trading_simple.exchanges import INTRADAY_BARS, ExchangeSpec, is_rth, resolve_exchange

log = logging.getLogger(__name__)

_SOURCE = "ibkr"
_BAR_SIZE = {"1m": "1 min", "5m": "5 mins", "1h": "1 hour", "1d": "1 day"}
_STEP_DAYS = {"1m": 10, "5m": 30, "1h": 30}  # request window per bar size (see module doc)


class Pacer:
    """Sliding-window limiter for IBKR historical-data requests.

    Keeps the start times of the last ``max_per_window`` requests and, before
    issuing another, waits until fewer than that many fall within the trailing
    ``window_s`` seconds. Adds zero delay until the cap is approached. A small
    ``min_gap_s`` floor between consecutive requests guards IBKR's
    6-per-2s-per-contract rule. ``clock``/``sleep`` are injectable for tests.
    """

    def __init__(self, max_per_window: int = 55, window_s: float = 600.0,
                 min_gap_s: float = 0.5) -> None:
        """Configure the cap: ``max_per_window`` request starts per ``window_s`` seconds."""
        self.max_per_window = max(max_per_window, 1)
        self.window_s = window_s
        self.min_gap_s = max(min_gap_s, 0.0)
        self._starts: deque[float] = deque()

    def wait(self, *, clock=_time.monotonic, sleep=_time.sleep) -> None:
        """Block (only if needed) until one more request stays within the cap."""
        now = clock()
        self._evict(now)
        if self._starts and self.min_gap_s:
            gap = self.min_gap_s - (now - self._starts[-1])
            if gap > 0:
                sleep(gap)
                now = clock()
                self._evict(now)
        if len(self._starts) >= self.max_per_window:
            sleep(self.window_s - (now - self._starts[0]) + 0.05)
            self._evict(clock())
        self._starts.append(clock())

    def _evict(self, now: float) -> None:
        """Drop request starts that have aged out of the trailing window."""
        while self._starts and now - self._starts[0] >= self.window_s:
            self._starts.popleft()


def request_windows(bar: str, start: date, end: date) -> list[tuple[datetime, str]]:
    """Plan the ``(endDateTime, durationStr)`` requests covering ``[start, end]``.

    ``reqHistoricalData`` takes an end anchor plus a duration (not a start), so
    intraday history is paged **backwards** from ``end``, newest first, in
    per-bar windows (1m = 10 days). Daily bars fit in one request — a duration
    in days, or years past IB's 365-day cap. Pure function.

    Args:
        bar: Bar size, e.g. ``"1m"`` or ``"1d"``.
        start: First day wanted (inclusive).
        end: Last day wanted (inclusive).

    Returns:
        Request plan, newest window first.

    Raises:
        ValueError: If ``start`` is after ``end``.
    """
    if start > end:
        raise ValueError(f"start {start} must be on or before end {end}")
    eod = lambda d: datetime(d.year, d.month, d.day, 23, 59)  # noqa: E731
    if bar not in INTRADAY_BARS:
        days = max((end - start).days + 1, 1)
        return [(eod(end), f"{-(-days // 365)} Y" if days > 365 else f"{days} D")]
    step = _STEP_DAYS.get(bar, 30)
    windows, win_end = [], end
    while win_end >= start:
        span = min(step, (win_end - start).days + 1)
        windows.append((eod(win_end), f"{span} D"))
        win_end -= timedelta(days=step)
    return windows


def contract_args(symbol: str, spec: ExchangeSpec) -> tuple[str, str, str]:
    """``(symbol, exchange, currency)`` for an ``ib_async`` Stock contract.

    US names use SMART routing; non-US names use their explicit primary venue
    (SIX -> ``EBS``/CHF) so history comes from the real listing, not an
    ambiguous cross-listing. Pure function.

    Args:
        symbol: Bare canonical ticker.
        spec: The symbol's resolved venue.

    Returns:
        Positional args for ``ib_async.Stock``.
    """
    return (symbol, spec.ib_exchange, spec.currency)


def frame_to_bars(symbol: str, bar: str, frame, spec: ExchangeSpec) -> list[PriceBar]:
    """Convert an ``ib_async`` historical-data frame into :class:`PriceBar` rows.

    Pure function. IBKR provides no adjusted close, so ``close`` is mirrored
    into ``adj_close``. Volume is already in shares; a negative value is IBKR's
    "not available" sentinel and maps to ``None``. Each bar's ``rth`` flag is
    computed from its local wall-clock time via :func:`~trading_simple.exchanges.is_rth`
    — independent of whether the fetch included extended hours.

    Args:
        symbol: Bare symbol the bars are stored under.
        bar: Bar size.
        frame: DataFrame with date/open/high/low/close/volume columns.
        spec: The symbol's venue (for session tagging).

    Returns:
        One :class:`PriceBar` per row.
    """
    import pandas as pd

    bars = []
    for _, row in frame.iterrows():
        ts = pd.Timestamp(row["date"])
        ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        close, volume = float(row["close"]), float(row["volume"])
        bars.append(PriceBar(
            symbol=symbol, bar=bar, ts=ts_utc.to_pydatetime(),
            open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
            close=close, adj_close=close, volume=volume if volume >= 0 else None,
            source=_SOURCE, rth=is_rth(ts_utc, spec, bar),
        ))
    return bars


def fetch_batch(
    symbols: list[str],
    bar: str,
    start: date,
    end: date,
    *,
    settings: IBKRSettings,
    exchanges: dict[str, str] | None = None,
    on_symbol: Callable[[str, list[PriceBar]], None] | None = None,
) -> dict[str, list[PriceBar]]:
    """Fetch many symbols over ONE persistent IBKR connection, centrally paced.

    A failing symbol (e.g. no market-data permission) is logged and yields
    ``[]`` — it never aborts the batch. ``on_symbol(symbol, bars)`` fires as
    each symbol completes, so the caller can UPSERT incrementally and keep
    memory bounded.

    ``settings.extended_hours`` controls ``useRTH``: when True, pre/post-market
    bars are fetched too (US 04:00–20:00 ET); every bar is session-tagged via
    its ``rth`` flag regardless.

    Args:
        symbols: Bare canonical tickers.
        bar: Bar size, e.g. ``"1m"``.
        start: First day (inclusive).
        end: Last day (inclusive).
        settings: IBKR connection + pacing settings.
        exchanges: Optional per-symbol venue overrides.
        on_symbol: Optional per-symbol completion callback.

    Returns:
        ``{symbol: bars}`` for every requested symbol.
    """
    import pandas as pd
    from ib_async import IB, Stock, util  # lazy: optional dependency

    pacer = Pacer(settings.max_requests_per_10min, min_gap_s=settings.pace_seconds)
    ib = IB()
    ib.connect(settings.host, settings.port, clientId=settings.client_id, timeout=15)
    out: dict[str, list[PriceBar]] = {}
    try:
        for sym in symbols:
            try:
                spec = resolve_exchange(sym, exchanges)
                contract = Stock(*contract_args(sym, spec))
                ib.qualifyContracts(contract)
                frames = []
                for end_dt, duration in request_windows(bar, start, end):
                    pacer.wait(sleep=ib.sleep)
                    raw = ib.reqHistoricalData(
                        contract, endDateTime=end_dt, durationStr=duration,
                        barSizeSetting=_BAR_SIZE[bar], whatToShow="TRADES",
                        useRTH=not settings.extended_hours,
                        formatDate=2,  # tz-aware UTC datetimes
                    )
                    if not raw:  # reached the start of IB's available history
                        break
                    frames.append(util.df(raw))
                bars = []
                if frames:
                    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"])
                    bars = frame_to_bars(sym, bar, frame, spec)
            except Exception as exc:  # noqa: BLE001 — one bad symbol must not abort the batch
                log.warning("IBKR fetch failed for %s: %s", sym, exc)
                bars = []
            out[sym] = bars
            if on_symbol is not None:
                on_symbol(sym, bars)
    finally:
        ib.disconnect()
    return out
