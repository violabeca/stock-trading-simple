"""The one backtest engine every strategy runs through (vectorbt under the hood).

A strategy hands over a dict of signal frames (and optional stop parameters);
this module turns it into a vectorbt ``Portfolio``, sweeps parameter grids
efficiently, and reduces portfolios to comparable metrics. Execution rules are
engine-enforced and identical for every strategy:

* **No look-ahead**: strategies time their own signals with :func:`next_bar` /
  :func:`exit_after`, and fills happen at the **next bar's adjusted open**
  (``panel.open_``), never at the signal bar's close.
* **Costs always on**: :func:`leg_costs` converts the configured friction into
  vectorbt ``fees``/``slippage`` fractions.
* **Per-symbol independence**: each symbol trades its own ``init_cash``; the
  combined equity curve is the row-sum.

Speed comes from vectorbt's Rust/numba kernels (``engine: auto`` in
``settings.yaml``) plus chunked column-stacking in :func:`sweep`: many
parameter combos become extra columns of ONE vectorbt call instead of many
calls.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from trading_simple.config import CostSettings, Settings, load_settings
from trading_simple.data import Panel

if TYPE_CHECKING:
    from trading_simple.strategies import LoadedStrategy

#: Boolean signal frames a strategy may return.
SIGNAL_KEYS = ("entries", "exits", "short_entries", "short_exits")
#: Stop parameters a strategy may return (fraction of entry price / trail flag).
STOP_KEYS = ("sl_stop", "sl_trail", "tp_stop")

_configured = False


def configure(settings: Settings | None = None) -> None:
    """Import vectorbt once and select the computation engine.

    ``engine: auto`` dispatches to the Rust kernels (``vectorbt-rust``) where
    supported and falls back to numba elsewhere — always correct, usually
    faster. Called automatically by :func:`run`; safe to call repeatedly.

    Args:
        settings: Resolved settings (loaded when omitted).
    """
    global _configured
    if _configured:
        return
    import vectorbt as vbt

    vbt.settings["engine"] = (settings or load_settings()).backtest.engine
    _configured = True


# ---------------------------------------------------------------------------
# Shared indicator + timing helpers (strategies build signals from these).
# ---------------------------------------------------------------------------
def sma(frame: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
    """Simple moving average with a strict warm-up (NaN until ``window`` bars)."""
    return frame.rolling(window, min_periods=window).mean()


def atr(panel: Panel, window: int = 14) -> pd.DataFrame:
    """Wilder's Average True Range, in (adjusted) price units.

    True range = max(high−low, \\|high−prev close\\|, \\|low−prev close\\|),
    smoothed with Wilder's EWM (alpha = 1/window, strict warm-up). Used to
    scale stop distances to each name's own volatility.

    Args:
        panel: Price panel (uses high/low/close).
        window: Smoothing window in bars.

    Returns:
        ATR frame aligned with ``panel.close``.
    """
    prev_close = panel.close.shift(1)
    tr = pd.concat([
        panel.high - panel.low,
        (panel.high - prev_close).abs(),
        (panel.low - prev_close).abs(),
    ]).groupby(level=0).max()
    return tr.ewm(alpha=1.0 / window, min_periods=window).mean()


def volume_profile(
    volume: pd.DataFrame, tz: str, *, days: int = 10, stat: str = "median",
    min_obs: int = 5, smooth: int = 0
) -> pd.DataFrame:
    """Expected volume per bar from the average *shape* of the trading day.

    For each bar, the baseline is the ``stat`` (median by default) of the
    volume at the **same exchange-local time-of-day** over the previous
    ``days`` sessions — so 09:31 bars compare only to other 09:31 bars. This
    removes the intraday U-shape bias that poisons plain rolling averages, and
    handles mixed regular/extended sessions by construction (a 04:05 pre-market
    bar only ever compares to 04:05 bars). Grouping on *local* time keeps the
    profile stable across DST changes; the median resists one-off blowups; the
    current bar is excluded (shift by one session).

    ``smooth`` widens each time-of-day slot to ``±smooth`` bars *within the
    same session* before profiling — 09:35 then compares to the previous days'
    09:35±2 average rather than the single minute, which steadies the baseline
    on thin one-minute prints. Smoothing never crosses a day boundary and only
    ever feeds *past* sessions' values into the baseline, so no look-ahead.

    For daily bars every timestamp shares one time-of-day, so this degrades to
    a plain shifted rolling ``stat`` — same semantics as the classic SMA baseline.

    Args:
        volume: Wide volume frame (UTC index).
        tz: Exchange-local timezone (``panel.tz``).
        days: Number of past same-time observations in the baseline.
        stat: Aggregation: ``"median"`` (robust, default) or ``"mean"``.
        min_obs: Minimum observations before the baseline is defined
            (clamped to ``days`` so a short lookback stays valid).
        smooth: Half-width (bars) of the centered per-session smoothing window.

    Returns:
        Baseline frame aligned with ``volume`` (NaN during warm-up).
    """
    min_obs = min(min_obs, days)
    local = volume.index.tz_convert(tz)
    if smooth:
        volume = volume.groupby(local.date).transform(
            lambda s: s.rolling(2 * smooth + 1, center=True, min_periods=1).mean())
    return volume.groupby(local.strftime("%H:%M")).transform(
        lambda s: getattr(s.rolling(days, min_periods=min_obs), stat)().shift(1)
    )


def session_mask(index: pd.DatetimeIndex, tz: str, *, skip_open: int = 0,
                 skip_close: int = 0) -> pd.Series:
    """True for bars outside the first/last N minutes of each trading day.

    The entry blocker for intraday strategies: the open auction and the
    closing minutes are structurally chaotic, so signals there are noise. The
    session is defined by the *data* — each exchange-local day's first and
    last stored bar — so half-days and venue quirks need no session table.

    Args:
        index: The panel's UTC bar index.
        tz: Exchange-local timezone (``panel.tz``).
        skip_open: Minutes blocked after the session's first bar.
        skip_close: Minutes blocked before the session's last bar.

    Returns:
        Boolean Series on ``index`` (True = tradable).
    """
    ts = pd.Series(index, index=index)
    day = ts.groupby(index.tz_convert(tz).date)
    return ((ts >= day.transform("min") + pd.Timedelta(minutes=skip_open)) &
            (ts <= day.transform("max") - pd.Timedelta(minutes=skip_close)))


def session_end(index: pd.DatetimeIndex, tz: str) -> pd.Series:
    """True on each exchange-local trading day's final stored bar.

    OR this into an exit frame to force an intraday strategy flat at the end
    of every session (fills at that bar's open, i.e. before the close).

    Args:
        index: The panel's UTC bar index.
        tz: Exchange-local timezone (``panel.tz``).

    Returns:
        Boolean Series on ``index``.
    """
    ts = pd.Series(index, index=index)
    return ts == ts.groupby(index.tz_convert(tz).date).transform("max")


def next_bar(signals: pd.DataFrame) -> pd.DataFrame:
    """Shift raw signals one bar forward: signal at close(t) executes at t+1."""
    return signals.shift(1, fill_value=False)


def exit_after(entries: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Time-based exits ``horizon`` bars after each entry.

    Args:
        entries: Execution-timed entry frame.
        horizon: Holding period in bars (>= 1).

    Returns:
        Exit frame (entries near the end of the data get no exit; vectorbt
        closes those positions at the final bar).

    Raises:
        ValueError: If ``horizon`` is not positive.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return entries.shift(horizon, fill_value=False)


# ---------------------------------------------------------------------------
# Costs and grids.
# ---------------------------------------------------------------------------
def leg_costs(costs: CostSettings, *, price: float = 20.0,
              notional: float = 10_000.0) -> tuple[float, float]:
    """Convert configured friction into per-leg vectorbt fractions.

    ``fees`` = commission (per-share with a minimum) plus half the quoted
    spread; ``slippage`` = the configured impact. Both are charged on entry
    AND exit by vectorbt, so a round trip costs roughly ``2 × (fees + slippage)``.

    Args:
        costs: The ``costs:`` block from settings.
        price: Assumed share price for the per-share commission.
        notional: Assumed position size in dollars.

    Returns:
        ``(fees_fraction, slippage_fraction)`` per leg.
    """
    commission = max(costs.min_commission, notional / price * costs.commission_per_share)
    fees = commission / notional + costs.spread_bps / 2.0 / 1e4
    return fees, costs.slippage_bps / 1e4


def grid_from_config(sweep_cfg: dict[str, list], valid=lambda p: True) -> list[dict]:
    """Expand per-parameter value lists into a filtered cartesian grid.

    Args:
        sweep_cfg: The ``sweep:`` block of a strategy config
            (``{param: [values...]}``).
        valid: The strategy's combo validator; invalid combos are dropped.

    Returns:
        One params dict per surviving combination.
    """
    keys = list(sweep_cfg)
    combos = [dict(zip(keys, vals, strict=True))
              for vals in product(*(sweep_cfg[k] for k in keys))]
    return [c for c in combos if valid(c)]


# ---------------------------------------------------------------------------
# Running portfolios.
# ---------------------------------------------------------------------------
def run(panel: Panel, signals: dict, *, fees: float, slippage: float,
        init_cash: float = 10_000.0):
    """Build one vectorbt Portfolio from a strategy's signal dict.

    The single ``Portfolio.from_signals`` call site: fills at the adjusted
    next-bar open, equity valued on adjusted close, stops (if any) evaluated
    intrabar against high/low.

    Args:
        panel: Price panel.
        signals: Strategy output — any subset of :data:`SIGNAL_KEYS` (boolean
            frames) and :data:`STOP_KEYS` (scalars or frames; stop distances
            are fractions of entry price, NaN = no stop).
        fees: Per-leg fee fraction (see :func:`leg_costs`).
        slippage: Per-leg slippage fraction.
        init_cash: Starting cash per symbol.

    Returns:
        A ``vectorbt.Portfolio``.
    """
    configure()
    import vectorbt as vbt

    kwargs = {k: signals[k] for k in (*SIGNAL_KEYS, *STOP_KEYS) if k in signals}
    if any(k in signals for k in STOP_KEYS):
        kwargs |= {"high": panel.high, "low": panel.low}
    return vbt.Portfolio.from_signals(
        panel.close, price=panel.open_, fees=fees, slippage=slippage,
        init_cash=init_cash, freq=panel.freq, **kwargs,
    )


def run_strategy(strategy: LoadedStrategy, panel: Panel, params: dict | None = None,
                 *, settings: Settings):
    """Run one backtest of a strategy with its config defaults (plus overrides).

    Args:
        strategy: A loaded strategy (see :func:`trading_simple.strategies.load`).
        panel: Price panel matching the strategy's ``data:`` block.
        params: Overrides merged over ``strategy.params``.
        settings: Global settings (costs, init_cash).

    Returns:
        A ``vectorbt.Portfolio``.
    """
    merged = {**strategy.params, **(params or {})}
    fees, slippage = leg_costs(settings.costs)
    return run(panel, strategy.signals(panel, **merged), fees=fees, slippage=slippage,
               init_cash=settings.backtest.init_cash)


def evaluate(strategy: LoadedStrategy, panel: Panel, params: dict, *, settings: Settings) -> dict:
    """Backtest one parameter set and reduce it to plain metric floats.

    The single entry point shared by notebooks, tests, and the Optuna
    objective — whatever runs through here is comparable.

    Args:
        strategy: A loaded strategy.
        panel: Price panel.
        params: Full parameter dict for this run.
        settings: Global settings.

    Returns:
        ``{total_return, sharpe, max_drawdown, win_rate, pct_profitable, n_trades}``.
    """
    return portfolio_stats(run_strategy(strategy, panel, params, settings=settings))


def sweep(strategy: LoadedStrategy, panel: Panel, grid: list[dict], *,
          settings: Settings, chunk_size: int | None = None) -> pd.DataFrame:
    """Backtest every grid combo over every symbol; one tidy row per combo.

    Combos are column-stacked in chunks: each chunk becomes ONE wide
    ``(combo, symbol)`` MultiIndex portfolio call, and per-combo metrics fall
    out of a groupby — amortizing vectorbt's per-call overhead over hundreds
    of columns while ``chunk_size`` bounds peak memory.

    Args:
        strategy: A loaded strategy.
        panel: Price panel.
        grid: Parameter dicts (see :func:`grid_from_config`).
        settings: Global settings.
        chunk_size: Combos per vectorbt call (default from settings).

    Returns:
        DataFrame with the grid's parameter columns plus ``total_return,
        sharpe, max_drawdown, win_rate, pct_profitable, n_trades``.
    """
    configure(settings)
    import vectorbt as vbt

    chunk_size = chunk_size or settings.backtest.chunk_size
    fees, slippage = leg_costs(settings.costs)
    init_cash = settings.backtest.init_cash
    index, columns = panel.close.index, panel.close.columns
    empty_bool = pd.DataFrame(False, index=index, columns=columns)
    no_stop = pd.DataFrame(np.nan, index=index, columns=columns)

    def as_frame(value, fallback: pd.DataFrame) -> pd.DataFrame:
        """Broadcast a scalar signal/stop value to a full (ts × symbol) frame."""
        if isinstance(value, pd.DataFrame):
            return value
        if value is None:
            return fallback
        return pd.DataFrame(value, index=index, columns=columns)

    rows = []
    for start in range(0, len(grid), chunk_size):
        chunk = grid[start:start + chunk_size]
        ids = list(range(start, start + len(chunk)))
        combo_signals = [strategy.signals(panel, **{**strategy.params, **combo}) for combo in chunk]
        used_stops = any(k in sig for sig in combo_signals for k in STOP_KEYS)
        cols = pd.MultiIndex.from_product([ids, columns], names=["combo", "symbol"])

        def stack(frames: list[pd.DataFrame], cols: pd.MultiIndex = cols) -> pd.DataFrame:
            """Concatenate per-combo frames into one (combo, symbol) block."""
            return pd.DataFrame(np.concatenate([f.to_numpy() for f in frames], axis=1),
                                index=index, columns=cols)

        def tile(frame: pd.DataFrame, ids: list[int] = ids,
                 cols: pd.MultiIndex = cols) -> pd.DataFrame:
            """Repeat one panel-wide frame once per combo."""
            return stack([frame] * len(ids), cols)

        kwargs = {
            key: stack([as_frame(sig.get(key), empty_bool) for sig in combo_signals])
            for key in SIGNAL_KEYS
        }
        if used_stops:
            kwargs["sl_stop"] = stack([as_frame(s.get("sl_stop"), no_stop) for s in combo_signals])
            kwargs["tp_stop"] = stack([as_frame(s.get("tp_stop"), no_stop) for s in combo_signals])
            kwargs["sl_trail"] = stack(
                [as_frame(s.get("sl_trail", False), empty_bool) for s in combo_signals])
            kwargs |= {"high": tile(panel.high), "low": tile(panel.low)}

        pf = vbt.Portfolio.from_signals(
            tile(panel.close), price=tile(panel.open_),
            fees=fees, slippage=slippage, init_cash=init_cash, freq=panel.freq, **kwargs,
        )
        per_combo = _metric_frame(pf, cols).groupby(level="combo").agg({
            "total_return": "mean", "sharpe": "mean", "max_drawdown": "mean",
            "win_rate": "mean", "profitable": "mean", "n_trades": "sum",
        }).rename(columns={"profitable": "pct_profitable"})
        rows += [{**grid[cid], **per_combo.loc[cid].to_dict()} for cid in ids]

    stats = pd.DataFrame(rows)
    stats["n_trades"] = stats["n_trades"].astype(int)
    return stats


# ---------------------------------------------------------------------------
# Reducing portfolios to numbers and tables.
# ---------------------------------------------------------------------------
def portfolio_stats(pf) -> dict:
    """Cross-symbol summary of one portfolio as plain floats.

    Means over symbols for return/Sharpe/drawdown/win-rate, plus
    ``pct_profitable`` (share of symbols with a positive total return — the
    "does it generalize across names?" guard) and total ``n_trades``.

    Args:
        pf: A ``vectorbt.Portfolio``.

    Returns:
        Metric dict.
    """
    m = _metric_frame(pf, pf.wrapper.columns)
    return {
        "total_return": float(m["total_return"].mean()),
        "sharpe": float(m["sharpe"].mean()),
        "max_drawdown": float(m["max_drawdown"].mean()),
        "win_rate": float(m["win_rate"].mean()),
        "pct_profitable": float(m["profitable"].mean()),
        "n_trades": int(m["n_trades"].sum()),
    }


def symbol_stats(pf) -> pd.DataFrame:
    """Per-symbol metric table — the cross-section behind :func:`portfolio_stats`.

    The verification view: means can hide an edge carried by one name, so
    judge a parameter set on this breakdown before believing it.

    Args:
        pf: A ``vectorbt.Portfolio`` (one column per symbol).

    Returns:
        One row per symbol (``total_return, sharpe, max_drawdown, win_rate,
        n_trades``), sorted by total return descending.
    """
    m = _metric_frame(pf, pf.wrapper.columns).drop(columns="profitable")
    m["n_trades"] = m["n_trades"].astype(int)
    m.index.name = "symbol"
    return m.sort_values("total_return", ascending=False)


def _metric_frame(pf, cols: pd.Index) -> pd.DataFrame:
    """Per-column metrics with non-finite values (inf Sharpe etc.) nulled out."""
    def safe(fn) -> pd.Series:
        try:
            series = pd.Series(fn(), dtype="float64").reindex(cols)
        except Exception:
            return pd.Series(np.nan, index=cols, dtype="float64")
        return series.where(np.isfinite(series))

    total_return = safe(pf.total_return)
    return pd.DataFrame({
        "total_return": total_return,
        "sharpe": safe(pf.sharpe_ratio),
        "max_drawdown": safe(pf.max_drawdown),
        "win_rate": safe(pf.trades.win_rate),
        "profitable": (total_return > 0).astype(float),
        "n_trades": safe(pf.trades.count).fillna(0),
    })


def best_combo(stats: pd.DataFrame, *, metric: str = "sharpe") -> dict:
    """The parameter dict of a sweep's best row by ``metric`` (NaNs ignored).

    Args:
        stats: Output of :func:`sweep`.
        metric: Column to maximize.

    Returns:
        The winning row's parameters (metric columns stripped).

    Raises:
        ValueError: If every row has a NaN metric.
    """
    valid = stats.dropna(subset=[metric])
    if valid.empty:
        raise ValueError(f"every sweep row has NaN {metric!r} — nothing to pick")
    row = valid.loc[valid[metric].idxmax()]
    metric_cols = {"total_return", "sharpe", "max_drawdown", "win_rate",
                   "pct_profitable", "n_trades"}
    return {k: v for k, v in row.items() if k not in metric_cols}


_TRADE_COLS = {
    "Column": "symbol", "Size": "size", "Entry Timestamp": "entry_ts",
    "Avg Entry Price": "entry_price", "Exit Timestamp": "exit_ts",
    "Avg Exit Price": "exit_price", "PnL": "pnl", "Return": "return",
    "Direction": "direction", "Status": "status",
}


def trade_records(pf) -> pd.DataFrame:
    """Tidy per-trade table sorted by entry time.

    Columns: ``symbol, size, entry_ts, entry_price, exit_ts, exit_price, pnl,
    return, direction, status``. Prices are actual fill prices, so rows draw
    straight onto a price chart.

    Args:
        pf: A ``vectorbt.Portfolio``.

    Returns:
        One row per trade.
    """
    out = pf.trades.records_readable.rename(columns=_TRADE_COLS)
    keep = [c for c in _TRADE_COLS.values() if c in out.columns]
    return out[keep].sort_values("entry_ts").reset_index(drop=True)


def equity_curve(pf, *, init_cash: float | None = None) -> pd.DataFrame:
    """Combined ``value`` ($), cumulative ``return``, ``fees`` and ``realized_pnl``.

    Symbols run as independent portfolios, so combined value is the row-sum of
    per-symbol equity curves. ``fees`` is the cumulative commission+slippage
    actually paid across every fill (so an open position's entry fee counts),
    and ``realized_pnl`` is the cumulative net profit/loss booked as positions
    close (already net of their own fees; open positions contribute nothing).
    Both accumulate across symbols and start at zero.

    Args:
        pf: A ``vectorbt.Portfolio``.
        init_cash: Per-symbol starting cash (else the first row is the base).

    Returns:
        DataFrame with ``value``, ``return``, ``fees`` and ``realized_pnl`` columns.
    """
    value = pf.value()
    if isinstance(value, pd.Series):
        value = value.to_frame()
    total = value.sum(axis=1)
    base = init_cash * value.shape[1] if init_cash is not None else float(total.iloc[0])
    orders, trades = pf.orders.records_readable, pf.trades.records_readable
    fees = orders.groupby("Timestamp")["Fees"].sum()
    closed = trades[trades["Status"] == "Closed"]
    realized = closed.groupby("Exit Timestamp")["PnL"].sum()
    return pd.DataFrame({
        "value": total,
        "return": total / base - 1.0,
        "fees": fees.reindex(total.index, fill_value=0.0).cumsum(),
        "realized_pnl": realized.reindex(total.index, fill_value=0.0).cumsum(),
    })
