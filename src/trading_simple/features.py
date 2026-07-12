"""Alternative-data features: shared-DB event tables -> panel-aligned matrices.

This is the one engine extension that lets a strategy see beyond OHLCV. It reads
the point-in-time alt-data tables (``insider_trades``, ``analyst_ratings``,
``short_interest`` — ingested by this repo's own providers) and turns each into a
wide ``(panel-index x symbol)`` daily matrix that rides on :attr:`Panel.features`,
so ``signals(panel, **params)`` can consume it with no further plumbing. Adding a
future alt-data strategy needs only a new builder here and a ``data.features:``
line in its config — never a change to the strategy contract.

**Division of labour.** Builders do only DB -> daily aggregate (e.g. net insider
dollars per day). ALL modelling — rolling windows, EMAs, cross-sectional ranks,
weights — stays in the strategy's ``signals()``, where it is tunable.

**Point-in-time discipline.** Every builder buckets events by their ``known_at``
(the public-availability timestamp) in UTC, and :func:`_to_grid` then applies a
one-bar ``shift`` so a feature at bar ``D`` reflects only events knowable
*strictly before* ``D``. Combined with the engine's next-bar-open execution
(``bt.next_bar`` in the strategy), an event known during day ``D`` moves a
position no earlier than the open of ``D+2`` — deliberately conservative, and
immaterial at these signals' weeks-to-months horizons.

Daily bars only: the alignment keys on the UTC calendar date, which for
UTC-midnight daily bars equals the trading session's date.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

# One row per (symbol, UTC event-date): net signed open-market insider dollars.
_INSIDER_SQL = """
    SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d,
           sum(CASE WHEN txn_type = 'buy'  THEN shares * price
                    WHEN txn_type = 'sell' THEN -shares * price ELSE 0 END) AS value
    FROM insider_trades
    WHERE symbol = ANY(:symbols) AND known_at >= :start AND known_at <= :as_of
    GROUP BY symbol, d
"""

# One row per (symbol, UTC event-date): net rating change (+1 upgrade, -1 downgrade).
_RATINGS_SQL = """
    SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d,
           sum(CASE WHEN action = 'upgrade'   THEN 1
                    WHEN action = 'downgrade' THEN -1 ELSE 0 END) AS value
    FROM analyst_ratings
    WHERE symbol = ANY(:symbols) AND known_at >= :start AND known_at <= :as_of
    GROUP BY symbol, d
"""

# One row per (symbol, UTC event-date): summed analyst price-target REVISION.
#
# The revision (new/prior - 1), never the target-to-price LEVEL. Brav & Lehavy (2003)
# locate the drift in the revision — notably on events where the *rating* did not
# change, which `_RATINGS_SQL` scores as zero — while Han & Kang (2022) show price
# drifts *against* the level, most strongly in large, liquid names. Ranking on the
# level would likely invert the sign in exactly this universe.
#
# `pt_prior > 0` drops first-ever targets (stored NULL, no prior to revise from).
_PRICE_TARGET_SQL = """
    SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d,
           sum(pt_current / pt_prior - 1.0) AS value
    FROM analyst_price_targets
    WHERE symbol = ANY(:symbols) AND known_at >= :start AND known_at <= :as_of
      AND pt_prior > 0 AND pt_current > 0
    GROUP BY symbol, d
"""

# One row per (symbol, UTC dissemination-date): days-to-cover level (a stock).
_SHORT_INTEREST_SQL = """
    SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d,
           avg(days_to_cover) AS value
    FROM short_interest
    WHERE symbol = ANY(:symbols) AND known_at >= :start AND known_at <= :as_of
    GROUP BY symbol, d
"""

#: Feature name -> (SQL, fill mode). ``zero`` = event flow (no event = 0);
#: ``ffill`` = a level that persists between observations (short interest).
_BUILDERS = {
    "insider": (_INSIDER_SQL, "zero"),
    "ratings": (_RATINGS_SQL, "zero"),
    "price_target": (_PRICE_TARGET_SQL, "zero"),
    "short_interest": (_SHORT_INTEREST_SQL, "ffill"),
}


def load_features(
    engine: Engine, index: pd.DatetimeIndex, symbols: list[str], names: list[str],
) -> dict[str, pd.DataFrame]:
    """Build point-in-time feature matrices aligned to a panel's index.

    Args:
        engine: Shared-database engine.
        index: The panel's (tz-aware UTC) daily timestamp index.
        symbols: Panel symbols; every returned frame has exactly these columns.
        names: Feature names to build (subset of :data:`_BUILDERS`).

    Returns:
        ``{name: frame}`` where each frame is ``(index x symbols)``, already
        PIT-lagged (see module docstring). Unknown names raise ``KeyError``.

    Raises:
        KeyError: If a requested name has no builder.
    """
    start = pd.Timestamp(index[0]).to_pydatetime()
    as_of = pd.Timestamp(index[-1]).to_pydatetime()
    out: dict[str, pd.DataFrame] = {}
    for name in names:
        sql, fill = _BUILDERS[name]
        with engine.connect() as conn:
            long = pd.read_sql(text(sql), conn,
                               params={"symbols": symbols, "start": start, "as_of": as_of})
        grid = _to_grid(long, index, fill=fill)
        out[name] = grid.reindex(columns=symbols)
    return out


def attach_features(panel, engine: Engine, names: list[str], *, settings=None) -> None:
    """Populate ``panel.features`` in place with the requested alt-data matrices.

    Args:
        panel: A :class:`~trading_simple.data.Panel` (daily bars).
        engine: Shared-database engine.
        names: Feature names to build (e.g. ``["insider", "ratings"]``).
        settings: Accepted for call-site symmetry; unused (kept so callers can
            pass it uniformly alongside other loaders).
    """
    panel.features = load_features(engine, panel.close.index, panel.symbols, names)


def _to_grid(long: pd.DataFrame, index: pd.DatetimeIndex, *, fill: str = "zero") -> pd.DataFrame:
    """Pivot a ``(symbol, d, value)`` long frame onto the panel index (pure).

    Reindexes daily event aggregates onto the panel's calendar (matching on UTC
    date), fills gaps per ``fill`` (``"zero"`` for event flows, ``"ffill"`` for
    persistent levels), then shifts one bar so bar ``D`` sees only events
    knowable strictly before ``D`` — the conservative point-in-time lag.

    Args:
        long: Columns ``symbol``, ``d`` (a date), ``value``; one row per pair.
        index: Target panel index (tz-aware UTC or naive).
        fill: ``"zero"`` (flows) or ``"ffill"`` (levels).

    Returns:
        A ``(index x symbol)`` frame; empty (index-only) when ``long`` is empty.
    """
    if long.empty:
        return pd.DataFrame(index=index, dtype=float)
    aggfunc = "mean" if fill == "ffill" else "sum"
    wide = long.pivot_table(index="d", columns="symbol", values="value", aggfunc=aggfunc)
    wide.index = pd.to_datetime(wide.index)  # date -> naive midnight Timestamp

    keys = pd.DatetimeIndex(index)
    keys = keys.tz_convert("UTC") if keys.tz is not None else keys
    keys = keys.normalize().tz_localize(None) if keys.tz is not None else keys.normalize()

    wide = wide.reindex(keys)
    wide = wide.ffill() if fill == "ffill" else wide.fillna(0.0)
    wide = wide.shift(1)          # PIT: bar D sees only events strictly before D
    wide.index = index            # restore the panel's real (tz-aware) index
    return wide
