"""Read price data from the database into wide, backtest-ready matrices.

The central type is :class:`Panel`: one DataFrame per field (close, open,
high, low, volume), each indexed by UTC timestamp with one column per symbol.
All price fields are **split/dividend-adjusted** (multiplied by
``adj_close / close``) so signals, fills, and stop levels live in one
consistent price space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
from sqlalchemy import Engine, text

from trading_simple.exchanges import resolve_exchange

#: Maps a bar size onto the pandas frequency vectorbt uses to annualize Sharpe.
_FREQ = {"1d": "1D", "1h": "1h", "5m": "5min", "1m": "1min"}
_SESSION_FILTER = {"regular": "AND rth", "extended": "AND NOT rth", "both": ""}


@dataclass(slots=True)
class Panel:
    """Wide price matrices for one bar size, ready for the backtest engine.

    Attributes:
        close: Adjusted close (values the equity curve).
        open_: Adjusted open (the engine's fill price: next-bar open).
        high: Adjusted high (intrabar stop evaluation).
        low: Adjusted low (intrabar stop evaluation).
        volume: Share volume, unadjusted.
        bar: Bar size the panel was loaded at, e.g. ``"1d"``.
        tz: Exchange-local timezone of the panel's (first) venue — used by
            time-of-day indicators such as the volume profile.
        rth: Boolean regular-session mask (only set when ``session="both"``).
        features: Optional alternative-data feature matrices, each a wide
            ``(index x symbol)`` daily frame aligned to this panel and already
            point-in-time lagged, keyed by feature name (e.g. ``"insider"``).
            Populated by :func:`trading_simple.features.attach_features` for
            alt-data strategies; ``None`` for ordinary price-only strategies.
    """

    close: pd.DataFrame
    open_: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    bar: str
    tz: str = "America/New_York"
    rth: pd.DataFrame | None = field(default=None)
    features: dict[str, pd.DataFrame] | None = field(default=None)

    @property
    def freq(self) -> str:
        """Pandas frequency string for this bar size (Sharpe annualization)."""
        return _FREQ[self.bar]

    @property
    def symbols(self) -> list[str]:
        """The panel's symbols (column order of every matrix)."""
        return list(self.close.columns)


def load_panel(
    engine: Engine,
    symbols: list[str],
    bar: str,
    start: date | datetime,
    end: date | datetime | None = None,
    *,
    session: str = "regular",
    exchanges: dict[str, str] | None = None,
) -> Panel:
    """Load a :class:`Panel` from the shared database in one query.

    Args:
        engine: Database engine.
        symbols: Symbols to load (columns of the result).
        bar: Bar size, e.g. ``"1d"`` or ``"1m"``.
        start: First bar timestamp (inclusive).
        end: Last bar timestamp (inclusive); ``None`` means everything stored.
        session: ``"regular"`` (rth only, default), ``"extended"`` (pre/post
            market only), or ``"both"`` (sets ``Panel.rth`` so indicators can
            stay session-aware).
        exchanges: Optional per-symbol venue overrides (for ``Panel.tz``).

    Returns:
        A :class:`Panel` with adjusted OHLC, volume, and a sorted UTC index.

    Raises:
        ValueError: On an unknown ``session`` or when no data matches.
    """
    if session not in _SESSION_FILTER:
        raise ValueError(f"session must be one of {sorted(_SESSION_FILTER)}, got {session!r}")
    sql = text(f"""
        SELECT symbol, ts, open, high, low, close, adj_close, volume, rth
        FROM prices_ohlcv
        WHERE symbol = ANY(:symbols) AND bar = :bar
          AND ts >= :start AND ts <= COALESCE(CAST(:end AS timestamptz), ts)
          {_SESSION_FILTER[session]}
        ORDER BY ts
    """)
    with engine.connect() as conn:
        long = pd.read_sql(sql, conn, params={
            "symbols": symbols, "bar": bar,
            "start": pd.Timestamp(start), "end": pd.Timestamp(end) if end else None,
        })
    if long.empty:
        raise ValueError(f"No {bar} data for {symbols} in [{start}, {end}] (session={session})")
    long["ts"] = pd.to_datetime(long["ts"], utc=True)

    wide = lambda col: long.pivot_table(index="ts", columns="symbol", values=col, aggfunc="first")  # noqa: E731
    close_raw, adj = wide("close"), wide("adj_close")
    ratio = (adj / close_raw).fillna(1.0)  # adjustment factor; 1.0 where close is missing
    return Panel(
        close=adj,
        open_=wide("open") * ratio,
        high=wide("high") * ratio,
        low=wide("low") * ratio,
        volume=wide("volume"),
        bar=bar,
        tz=resolve_exchange(symbols[0], exchanges).tz,
        rth=wide("rth").astype(bool) if session == "both" else None,
    )


def symbols_with_bars(
    engine: Engine,
    symbols: list[str],
    bar: str,
    start: date | datetime,
    end: date | datetime | None = None,
    *,
    limit: int | None = None,
) -> list[str]:
    """Return the subset of ``symbols`` that have ``bar`` data, deepest first.

    Use before :func:`load_panel` to cap a universe at its best-covered names
    (``max_symbols`` in strategy configs).

    Args:
        engine: Database engine.
        symbols: Candidate symbols.
        bar: Bar size to check.
        start: Window start (inclusive).
        end: Window end (inclusive); ``None`` = no upper bound.
        limit: Keep at most this many, by descending bar count.

    Returns:
        Symbols ordered by how many bars they have in the window.
    """
    sql = text("""
        SELECT symbol FROM prices_ohlcv
        WHERE symbol = ANY(:symbols) AND bar = :bar
          AND ts >= :start AND ts <= COALESCE(CAST(:end AS timestamptz), ts)
        GROUP BY symbol ORDER BY count(*) DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"symbols": symbols, "bar": bar,
                                  "start": pd.Timestamp(start),
                                  "end": pd.Timestamp(end) if end else None}).fetchall()
    out = [r[0] for r in rows]
    return out[:limit] if limit else out


def coverage(engine: Engine, symbols: list[str] | None = None,
             bar: str | None = None) -> pd.DataFrame:
    """Summarize what price data is stored.

    Without ``symbols``, returns one row per bar size (a database inventory).
    With ``symbols``, returns one row per (symbol, bar) with the bar count,
    date range, and the regular/extended split.

    Args:
        engine: Database engine.
        symbols: Restrict to these symbols (and switch to per-symbol rows).
        bar: Restrict to one bar size.

    Returns:
        A tidy coverage DataFrame.
    """
    group = "symbol, bar" if symbols else "bar"
    # Explicit casts: psycopg cannot infer the type of a NULL parameter.
    sql = text(f"""
        SELECT {group}, count(*) AS bars,
               min(ts) AS first_ts, max(ts) AS last_ts,
               count(*) FILTER (WHERE rth) AS regular,
               count(*) FILTER (WHERE NOT rth) AS extended
        FROM prices_ohlcv
        WHERE (CAST(:symbols AS text[]) IS NULL OR symbol = ANY(CAST(:symbols AS text[])))
          AND (CAST(:bar AS text) IS NULL OR bar = :bar)
        GROUP BY {group} ORDER BY {group}
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"symbols": symbols, "bar": bar})
