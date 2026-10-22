"""Intraday OHLCV bars from Interactive Brokers (via ``ib_async``).

IBKR is the source for deep 1-minute history. Two hard rules, learned live:

1. **One connection, one pacer.** IBKR caps historical requests at ~60 per
   10 minutes *per account*, so fanning out across processes buys nothing and
   trips pacing violations. :func:`fetch_batch` pulls every symbol over a single
   persistent connection, spaced by one shared :class:`Pacer`.
2. **1-minute data is paged in 10-day windows.** IB applies an escalating
   response penalty above ~10k bars per request; a 30-day extended-hours 1m
   request silently returns nothing.
3. **A slow answer is not the end of history.** IBKR answers one account's
   history requests one at a time (measured 2026-10-17: 4 or 8 concurrent
   requests gave 0.6x the throughput of one at a time), so a request can wait
   minutes behind others. ``ib_async`` then gives up after its timeout and
   returns nothing, exactly like the start of IB's history: that silently cut
   symbols short (XLI 2026-10-16; GEV, IGV, XLI, SPY 2026-10-17).
   :func:`fetch_window` waits up to :data:`REQUEST_TIMEOUT_S` and asks a
   timed-out window again; a symbol that keeps timing out is left out of the
   result (so a sweep keeps it pending), never stored incomplete.

Requires IB Gateway or TWS running and logged in (paper Gateway = port 4002).
The pure planning/parsing helpers (:func:`request_windows`,
:func:`contract_args`, :func:`frame_to_bars`, :class:`Pacer`) have no ``ib_async``
import, and :func:`fetch_window` takes a duck-typed ``ib``, so they are
unit-testable without a Gateway.
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
#: How long one history request may wait for its answer (``ib_async``'s own default is 60 s,
#: too short when the request queues behind others; see rule 3 of the module doc).
REQUEST_TIMEOUT_S = 240.0
#: How often a timed-out window is asked again, after a growing pause, before giving up.
TIMEOUT_RETRIES = 3


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


def _reconnect(ib, settings: IBKRSettings, *, attempts: int = 4, wait_s: float = 45.0) -> bool:
    """Try to restore a dropped IBKR connection (e.g. Gateway's own daily restart).

    ``ib``'s ``connect``/``disconnect``/``sleep`` are duck-typed so this is
    testable with a stand-in object — no ``ib_async`` import needed here.

    Args:
        ib: The (disconnected) ``IB`` client to reconnect in place.
        settings: Host/port/client id to reconnect with.
        attempts: How many tries before giving up.
        wait_s: Delay between attempts (``ib.sleep`` pumps the event loop).

    Returns:
        True once reconnected; False if every attempt failed.
    """
    for attempt in range(1, attempts + 1):
        log.warning("IBKR connection lost; reconnect attempt %d/%d", attempt, attempts)
        ib.sleep(wait_s)
        try:
            ib.disconnect()
            ib.connect(settings.host, settings.port, clientId=settings.client_id, timeout=15)
            return True
        except Exception as exc:  # noqa: BLE001 — report and keep retrying
            log.warning("IBKR reconnect attempt %d/%d failed: %s", attempt, attempts, exc)
    return False


def fetch_window(ib, contract, end_dt, duration: str, *, bar: str, use_rth: bool, pacer: Pacer,
                 timeout: float | None = None, retries: int | None = None,
                 clock=_time.monotonic) -> tuple[list, str]:
    """One window's bars, asked again when IBKR does not answer in time.

    ``ib_async`` returns an empty list both when IBKR has no bars (the start of
    its history) and when it gave up waiting. An empty answer that took (almost)
    the whole ``timeout`` is a timeout: the window is asked again after a pause
    of 30 s, 60 s, ... (paced like every request).

    Args:
        ib: A connected ``ib_async.IB`` (duck-typed: ``reqHistoricalData`` and ``sleep``).
        contract: The qualified contract.
        end_dt: The window's ``endDateTime``.
        duration: Its ``durationStr``.
        bar: Bar size, e.g. ``"1m"``.
        use_rth: Regular trading hours only.
        pacer: The shared request pacer.
        timeout: Seconds to wait for an answer (default :data:`REQUEST_TIMEOUT_S`).
        retries: Extra tries after a timeout (default :data:`TIMEOUT_RETRIES`).
        clock: Monotonic clock (injectable for tests).

    Returns:
        ``(bars, status)``: ``"ok"``; ``"empty"`` (IBKR answered with no bars: the
        start of its history); ``"timeout"`` (still no answer after the retries).
    """
    timeout = REQUEST_TIMEOUT_S if timeout is None else timeout
    retries = TIMEOUT_RETRIES if retries is None else retries
    for attempt in range(retries + 1):
        pacer.wait(sleep=ib.sleep)
        started = clock()
        raw = ib.reqHistoricalData(
            contract, endDateTime=end_dt, durationStr=duration,
            barSizeSetting=_BAR_SIZE[bar], whatToShow="TRADES", useRTH=use_rth,
            formatDate=2,  # tz-aware UTC datetimes
            timeout=timeout,
        )
        if raw:
            return raw, "ok"
        if clock() - started < 0.95 * timeout:
            return [], "empty"
        if attempt < retries:
            log.warning("IBKR gave no answer for %s ending %s within %.0f s; asking again "
                        "(%d of %d)", getattr(contract, "symbol", contract), end_dt, timeout,
                        attempt + 1, retries)
            ib.sleep(30.0 * (attempt + 1))
    return [], "timeout"


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

    A dropped CONNECTION (Gateway's own daily restart, a network blip) is a
    different failure and is handled differently: once ``ib.isConnected()``
    goes false, every subsequent ``reqHistoricalData`` call would otherwise
    fail instantly, racing through the rest of ``symbols`` in seconds and
    mislabeling every one of them ``[]`` — indistinguishable from a real "no
    data" result. Instead, :func:`_reconnect` is given a bounded number of
    paced attempts to restore the connection; if it can't, the batch stops
    there and the untouched remainder of ``symbols`` is simply absent from the
    returned dict (never marked as fetched), so a caller like
    :func:`trading_simple.jobs.run_ibkr_sweep` leaves them ``"pending"`` for
    the next run instead of wrongly recording them ``"empty"``.

    A SLOW request is a third failure: :func:`fetch_window` asks a timed-out
    window again, and a symbol whose window still gets no answer is left out
    of the returned dict (``on_symbol`` is not called for it), like the
    unfetched remainder above, instead of being stored with a hole.

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
            if not ib.isConnected() and not _reconnect(ib, settings):
                log.error("IBKR connection lost and could not be restored; stopping batch "
                         "before %s (%d of %d symbols left unfetched)",
                         sym, len(symbols) - len(out), len(symbols))
                break
            try:
                spec = resolve_exchange(sym, exchanges)
                contract = Stock(*contract_args(sym, spec))
                ib.qualifyContracts(contract)
                frames, unanswered = [], None
                for end_dt, duration in request_windows(bar, start, end):
                    raw, status = fetch_window(ib, contract, end_dt, duration, bar=bar,
                                               use_rth=not settings.extended_hours, pacer=pacer)
                    if status == "empty":  # reached the start of IB's available history
                        break
                    if status == "timeout":
                        unanswered = end_dt
                        break
                    frames.append(util.df(raw))
                if unanswered is not None:
                    log.error("IBKR never answered for %s (window ending %s, %d tries): left "
                              "out, so it stays pending instead of being stored incomplete",
                              sym, unanswered, TIMEOUT_RETRIES + 1)
                    continue
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
