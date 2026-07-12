"""Daily OHLCV bars from Yahoo Finance (free, no API key).

yfinance is the source for **daily** history only: it is deep (decades),
dividend/split-adjusted, and safe to fetch in parallel. Deep 1-minute history
comes from Interactive Brokers instead (:mod:`trading_simple.providers.ibkr`).

Fetching (network) is separated from parsing (:func:`_frame_to_bars`, pure) so
the parser is unit-testable offline.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

import pandas as pd

from trading_simple.db import PriceBar, PriceTarget, RatingChange
from trading_simple.exchanges import resolve_exchange

log = logging.getLogger(__name__)

_SOURCE = "yfinance"
# yfinance Action codes -> our normalized rating action.
_ACTION_MAP = {"up": "upgrade", "down": "downgrade", "main": "maintain",
               "reit": "maintain", "init": "initiate"}


def fetch_daily(
    symbol: str, start: date, end: date, *, exchanges: dict[str, str] | None = None
) -> list[PriceBar]:
    """Fetch daily OHLCV candles for one symbol from Yahoo Finance.

    The bare canonical symbol is translated to Yahoo's namespace for the
    download (Swiss ``NESN`` -> ``NESN.SW``) but bars are returned under the
    bare symbol, keeping daily (yfinance) and intraday (IBKR) data under one
    database key.

    Args:
        symbol: Bare canonical ticker, e.g. ``"AAPL"``.
        start: First calendar day to fetch (inclusive).
        end: Last calendar day to fetch (inclusive).
        exchanges: Optional per-symbol venue overrides (``settings.exchanges``).

    Returns:
        Parsed :class:`~trading_simple.db.PriceBar` rows (``rth=True``: a daily
        bar is the regular-session candle).

    Raises:
        RuntimeError: If Yahoo returned no data — yfinance swallows errors
            (throttle, outage, unknown symbol) into empty frames, so an empty
            result must be surfaced rather than silently stored as "no bars".
    """
    import yfinance as yf  # lazy: keeps import-time light

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # it logs scary noise
    spec = resolve_exchange(symbol, exchanges)
    yf_symbol = f"{symbol}{spec.yf_suffix}"
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    frame = yf.download(
        yf_symbol, start=start.isoformat(), end=end_exclusive,
        interval="1d", auto_adjust=False, progress=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError(
            f"yfinance returned no data for {yf_symbol!r} {start}..{end}; "
            "possible throttle/outage or unknown symbol"
        )
    if isinstance(frame.columns, pd.MultiIndex):  # single-ticker downloads still nest
        frame.columns = frame.columns.get_level_values(0)
    return _frame_to_bars(symbol, frame)


def _frame_to_bars(symbol: str, frame: pd.DataFrame) -> list[PriceBar]:
    """Convert a yfinance OHLCV frame into normalized :class:`PriceBar` rows.

    Pure function: index is a DatetimeIndex, columns are Open/High/Low/Close/
    Volume (+ optional 'Adj Close'). Timestamps are normalized to UTC; NaN
    values become ``None``; ``adj_close`` falls back to the raw close when
    Yahoo omits 'Adj Close'.

    Args:
        symbol: Bare symbol the bars are stored under.
        frame: Raw yfinance download.

    Returns:
        One :class:`PriceBar` per row.
    """
    bars = []
    for ts, row in frame.iterrows():
        ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        bars.append(PriceBar(
            symbol=symbol, bar="1d", ts=ts_utc.to_pydatetime(),
            open=_f(row.get("Open")), high=_f(row.get("High")), low=_f(row.get("Low")),
            close=_f(row.get("Close")),
            adj_close=(_f(row.get("Adj Close")) if "Adj Close" in frame.columns
                       else _f(row.get("Close"))),
            volume=_f(row.get("Volume")), source=_SOURCE, rth=True,
        ))
    return bars


def _f(value) -> float | None:
    """Coerce a pandas/NumPy scalar to a plain float, mapping NaN to None."""
    return None if value is None or pd.isna(value) else float(value)


def fetch_ratings(
    symbol: str, since: date, *, exchanges: dict[str, str] | None = None
) -> list[RatingChange]:
    """Fetch sell-side analyst rating changes for one symbol from Yahoo Finance.

    Rating *changes* (upgrades/downgrades) carry real, dated timestamps in
    yfinance (unlike its consensus-level fields, which are overwritten in place),
    so they are a usable point-in-time signal — the material for rating-change
    momentum (Womack 1996). Only events dated on/after ``since`` are kept.

    NOTE (documented limitation): yfinance returns ratings only for
    currently-covered names, so this feed carries residual survivorship bias even
    though prices and SEC insider data do not. Claims are restricted to the
    liquid universe accordingly.

    Args:
        symbol: Bare canonical ticker, e.g. ``"AAPL"``.
        since: Earliest grade date to include.
        exchanges: Optional per-symbol venue overrides (``settings.exchanges``).

    Returns:
        Parsed :class:`~trading_simple.db.RatingChange` rows (empty when Yahoo
        has no rating history for the symbol).
    """
    import yfinance as yf  # lazy: keeps import-time light

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    spec = resolve_exchange(symbol, exchanges)
    frame = yf.Ticker(f"{symbol}{spec.yf_suffix}").upgrades_downgrades
    if frame is None or frame.empty:
        return []
    return _ratings_frame_to_rows(symbol, frame, since)


def _ratings_frame_to_rows(symbol: str, frame: pd.DataFrame, since: date) -> list[RatingChange]:
    """Convert a yfinance upgrades/downgrades frame into :class:`RatingChange` rows.

    Pure function: ``frame`` is indexed by ``GradeDate`` (a DatetimeIndex) with
    ``Firm``/``ToGrade``/``FromGrade``/``Action`` columns. Each grade date is
    stamped ``known_at`` at end-of-day UTC — a decision at that day's close can
    see it, and execution is next-bar open — matching the insider convention.

    Args:
        symbol: Bare symbol the rows are stored under.
        frame: Raw yfinance ``upgrades_downgrades`` frame.
        since: Earliest grade date to include.

    Returns:
        One :class:`RatingChange` per row on/after ``since``.
    """
    rows: list[RatingChange] = []
    for idx, row in frame.iterrows():
        ts = pd.Timestamp(idx)
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        rated = ts.date()
        if rated < since:
            continue
        known = datetime(rated.year, rated.month, rated.day, 23, 59, 59, tzinfo=UTC)
        rows.append(RatingChange(
            symbol=symbol,
            firm=_str(row.get("Firm")),
            action=_ACTION_MAP.get(str(row.get("Action", "")).lower()),
            grade_from=_str(row.get("FromGrade")),
            grade_to=_str(row.get("ToGrade")),
            rated_at=rated,
            known_at=known,
            source=_SOURCE,
        ))
    return rows


def fetch_price_targets(
    symbol: str, since: date, *, exchanges: dict[str, str] | None = None
) -> list[PriceTarget]:
    """Fetch sell-side analyst price-target events for one symbol from Yahoo Finance.

    Reads the **same** ``upgrades_downgrades`` frame as :func:`fetch_ratings` — no
    extra network call. Yahoo's payload (a Benzinga feed) carries a per-event
    ``currentPriceTarget`` / ``priorPriceTarget`` / ``priceTargetAction`` alongside
    the grade columns, and yfinance passes those keys straight through even though
    it does not document them. They are dated per event, so unlike Yahoo's
    *consensus* target (a snapshot overwritten in place) they are usable
    point-in-time.

    This is a separate signal from the rating stream, not a replacement: the large
    majority of target revisions arrive with the rating unchanged, which
    :func:`fetch_ratings` maps to ``"maintain"`` and the feature layer scores as
    zero.

    NOTE (documented limitations): the price-target fields are **undocumented**, so
    Yahoo may withdraw them without notice (ingested rows remain valid — only future
    ingestion would break). The same survivorship caveat as :func:`fetch_ratings`
    applies: Yahoo covers only currently-followed names.

    Args:
        symbol: Bare canonical ticker, e.g. ``"AAPL"``.
        since: Earliest event date to include.
        exchanges: Optional per-symbol venue overrides (``settings.exchanges``).

    Returns:
        Parsed :class:`~trading_simple.db.PriceTarget` rows (empty when Yahoo has no
        history, or when its payload omits the price-target columns entirely).
    """
    import yfinance as yf  # lazy: keeps import-time light

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    spec = resolve_exchange(symbol, exchanges)
    frame = yf.Ticker(f"{symbol}{spec.yf_suffix}").upgrades_downgrades
    if frame is None or frame.empty:
        return []
    return _targets_frame_to_rows(symbol, frame, since)


def _targets_frame_to_rows(symbol: str, frame: pd.DataFrame, since: date) -> list[PriceTarget]:
    """Convert a yfinance upgrades/downgrades frame into :class:`PriceTarget` rows.

    Pure function. Rows without a usable ``currentPriceTarget`` are skipped — the
    upstream feed omits a target on a meaningful share of rating events, and a row
    with no target carries no revision. A ``priorPriceTarget`` of ``0.0`` means "no
    previous target" (a first-ever initiation), and is stored as ``None`` rather than
    zero so a downstream revision ratio cannot read it as a -100% cut.

    ``known_at`` is stamped at end-of-day UTC of the event date, matching
    :func:`_ratings_frame_to_rows`; the feature layer buckets by UTC date and then
    lags one bar, so the intraday time carries no extra information.

    Args:
        symbol: Bare symbol the rows are stored under.
        frame: Raw yfinance ``upgrades_downgrades`` frame.
        since: Earliest event date to include.

    Returns:
        One :class:`PriceTarget` per row on/after ``since`` that carries a target.
    """
    if "currentPriceTarget" not in frame.columns:  # Yahoo dropped the undocumented fields
        log.warning("yfinance returned no price-target columns for %s", symbol)
        return []
    rows: list[PriceTarget] = []
    for idx, row in frame.iterrows():
        ts = pd.Timestamp(idx)
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        rated = ts.date()
        if rated < since:
            continue
        current = _pos(row.get("currentPriceTarget"))
        if current is None:  # a rating event with no target attached
            continue
        rows.append(PriceTarget(
            symbol=symbol,
            firm=_str(row.get("Firm")),
            pt_action=_str(row.get("priceTargetAction")),
            pt_current=current,
            pt_prior=_pos(row.get("priorPriceTarget")),  # 0.0 -> None: no previous target
            rated_at=rated,
            known_at=datetime(rated.year, rated.month, rated.day, 23, 59, 59, tzinfo=UTC),
            source=_SOURCE,
        ))
    return rows


def _pos(value) -> float | None:
    """Coerce to a positive float; NaN, None and non-positive values become None."""
    number = _f(value)
    return number if number is not None and number > 0.0 else None


def _str(value) -> str | None:
    """Coerce a pandas scalar to a stripped str, mapping NaN/empty to None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None
