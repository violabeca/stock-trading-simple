"""The one backtest engine every strategy runs through (vectorbt under the hood).

A strategy hands over a dict of signal frames (and optional stop parameters);
this module turns it into a vectorbt ``Portfolio``, sweeps parameter grids
efficiently, and reduces portfolios to comparable metrics. Execution rules are
engine-enforced and identical for every strategy:

* **No look-ahead**: strategies time their own signals with :func:`next_bar` /
  :func:`exit_after`, and fills happen on the **next bar** — at its adjusted open
  (``panel.open_``) by default, or at its worst side under a strategy's
  ``backtest.fill: pessimistic`` (:func:`fill_prices`) — never at the signal bar's close.
* **Costs always on**: :func:`leg_costs` converts the configured friction into
  vectorbt ``fees``/``slippage`` fractions.
* **Out-of-sample windows start warm**: :func:`run_strategy` and :func:`evaluate` take
  ``score_from=``. The strategy's ``signals()`` still sees the whole panel, so every rolling
  window, EMA and moving average carries its history into the scored window, while the book
  is traded — and every metric measured — only from that bar on. Backtesting a panel sliced
  to the window instead restarts all of them at its first bar, which live trading never
  does (informed_flow's holdout ranked 2 names on its first day instead of 97).

Two execution models, chosen by what a strategy's ``signals()`` returns:

* **Per-symbol** (``entries``/``exits``) — each symbol trades its own
  ``init_cash`` in its own account; the combined equity curve is the row-sum.
  Cash attached to a symbol the strategy is flat in stays idle. Right for a
  per-symbol signal strategy; fatal for a cross-sectional one (see below).
* **Portfolio** (``weights``) — ONE shared cash account of ``init_cash`` *total*,
  rebalanced to target weights. Buying a name must be funded by selling another.
  This is the only way to express "hold the best K names, fully invested": with
  per-symbol accounts a strategy holding 30 of 100 names is structurally ~30%
  invested and cannot beat a fully-invested benchmark however good its selection.
  Held names drift untouched, but a strategy may add a ``retrim_k`` **concentration
  cap** (:func:`_retrim_loop`) so a runaway winner is trimmed back to its slot rather
  than left to dominate the book. Within a bar, sales and covers run before the new
  shorts and buys they fund (:func:`_call_order`), and an order the cash could not fill
  keeps working on the following bars until it does (:func:`_working_orders`). See
  :func:`_run_portfolio`.
* **Cash-shared signals** (``entries``/``exits`` + ``cash_frac``) — the wallet
  variant of the signal path: ONE shared account of ``init_cash`` *total*, and
  each entry deploys ``cash_frac`` of the currently *free* cash, so the book can
  never be more than 100% invested. This is how a single real account of
  ``init_cash`` would trade a multi-symbol signal scanner. ``cash_frac`` is a
  scalar, or a per-bar frame read at each fill bar (:func:`cash_frac_size`) — how a
  strategy sizes every entry off its own stop distance. Stops work per column
  as usual. Percent sizing cannot *reverse* a position, so an opposite-direction
  entry closes the open position instead (``upon_opposite_entry="close"``); the
  new direction opens on a later bar if its signal persists. Shorts **lock cash**
  (``lock_cash=True``): a short's sale proceeds stay reserved for buying it back, so
  they can never fund a long (the book stays at most 100% invested, shorts included)
  and a stop on a short always covers the whole position.

**Every return metric is per-year** (:func:`_years`). A cumulative return rewards
a long backtest window for being long, so it can compare neither two strategies
with different windows nor two walk-forward folds of different lengths — and
picking on it is exactly that mistake. ``cagr``/``excess_cagr`` are annualized;
``years`` and ``trades_per_year`` travel beside them so the sample size behind a
number is always visible.

Speed comes from vectorbt's Rust/numba kernels (``engine: auto`` in
``settings.yaml``) plus chunked column-stacking in :func:`sweep`: many
parameter combos become extra columns of ONE vectorbt call instead of many
calls.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
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
#: Returning this frame instead of ``entries``/``exits`` opts a strategy into
#: **portfolio mode**: one shared cash account, rebalanced to these target weights.
WEIGHTS_KEY = "weights"
#: Returning this scalar (0 < f <= 1) — or a dense (bar x symbol) frame of such fractions,
#: read at each fill bar, which is how risk-per-trade sizing ``min(slot, risk / stop)`` is
#: expressed (:func:`cash_frac_size`) — alongside ``entries``/``exits`` opts a signal
#: strategy into **one cash-shared wallet** of ``init_cash`` TOTAL: each entry deploys
#: its fraction of the group's currently free cash, so total invested can never
#: exceed the book. Percent sizing cannot reverse a position, so an opposite-direction
#: entry closes the open position instead of flipping it. A short's sale proceeds stay
#: locked for its cover (``lock_cash``), so they never fund a long.
CASH_FRAC_KEY = "cash_frac"
#: Returning this scalar (``k > 1``) alongside ``weights`` opts a portfolio-mode strategy
#: into a **concentration cap**: whenever a held name's realized book weight reaches
#: ``k ×`` its target (e.g. 2x its slot), it is trimmed back to target on the next bar and
#: the freed cash returns to the shared wallet, where the next rotation redeploys it. This
#: is what stops one runaway winner from quietly coming to dominate a book of otherwise
#: drifting positions. ``None`` = off, the **default** — so :func:`benchmark` and any
#: strategy that does not set it keep the pure buy-and-once-hold behavior. See
#: :func:`_run_portfolio` and :func:`_retrim_loop`.
RETRIM_K_KEY = "retrim_k"
#: Returning this frame (bar x symbol) — or a scalar — of YEARLY borrow rates (fractions:
#: 0.005 = 0.5%/yr) alongside ``weights`` charges every short position its borrow fee. The
#: engine models no stock-loan market of its own: the strategy states what each name costs
#: to borrow and :func:`_borrow_fees` charges it, accrued day by day on the calendar, on the
#: order that buys the short back (a short still open on the last bar pays on the order that
#: opened it). Portfolio mode only; without it shorts borrow for free.
BORROW_KEY = "borrow"
#: Fill models a strategy's ``backtest.fill`` may select (see :func:`fill_prices`).
#: ``open`` (the default) fills every order at the bar's adjusted open — symmetric and
#: neutral. ``pessimistic`` fills every buy at the bar's high and every sell at its low: the
#: worst plausible intrabar execution, a bound on how much of an edge lives inside the
#: open-fill assumption.
FILLS = ("open", "pessimistic")
#: Every metric a sweep/trial frame may carry; anything else in such a frame is a
#: strategy parameter. One definition, so a new metric cannot be mistaken for a knob.
#: ``years``/``trades_per_year`` are sample-size columns and belong here for exactly
#: that reason — they describe the run, they are not knobs to turn.
METRIC_COLS = frozenset({
    "total_return", "cagr", "sharpe", "max_drawdown", "win_rate", "pct_profitable",
    "n_trades", "trades_per_year", "years", "excess_cagr", "info_ratio", "tracking_error",
    "volatility", "alpha_cagr", "beta",
})

_configured = False


def _years(index: pd.DatetimeIndex) -> float:
    """Calendar length of a bar index, in years — the clock every metric is put on."""
    return max((index[-1] - index[0]).days, 1) / 365.25


def _cagr(total_return: float, years: float) -> float:
    """Annualize a cumulative return: the constant yearly rate that compounds to it.

    The whole point of the per-year clock. ``total_return`` grows with the length of
    the backtest window, so it cannot rank a 4-year strategy against a 1-month one, nor
    one walk-forward fold against a longer one. This can.

    Args:
        total_return: Cumulative return over the window (0.5 = +50%).
        years: Window length from :func:`_years`.

    Returns:
        The annualized rate, or NaN for a total wipeout (``total_return <= -1``), where
        no rate exists — the Optuna objective prunes NaN trials as degenerate.
    """
    growth = 1.0 + total_return
    if growth <= 0.0 or not np.isfinite(growth):
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


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


def cash_frac_size(frac):
    """A strategy's ``cash_frac`` as vectorbt's ``size``: a float, or a dense per-bar frame.

    A frame lets a strategy size each entry on its own — ``min(slot, risk_per_trade /
    stop_distance)`` is the risk-per-trade rule — and vectorbt reads it at the FILL bar. A
    NaN cell there SKIPS the entry with no error (verified on vectorbt 1.0), which would turn
    a sizing bug into a silent signal gate, so a frame must be dense: fill the bars that
    never fill with the slot fraction.

    Args:
        frac: Scalar fraction of free cash, or a ``(bar x symbol)`` frame of fractions.

    Returns:
        ``float``, or the frame cast to float.

    Raises:
        ValueError: If the frame has any NaN cell.
    """
    if not isinstance(frac, pd.DataFrame):
        return float(frac)
    if frac.isna().to_numpy().any():
        raise ValueError("cash_frac frame has NaN cells — a NaN size silently skips the entry; "
                         "fill the non-entry bars with the slot fraction")
    return frac.astype(float)


# ---------------------------------------------------------------------------
# Costs and grids.
# ---------------------------------------------------------------------------
def leg_costs(costs: CostSettings, *, price: float = 20.0,
              notional: float = 10_000.0) -> tuple[float, float]:
    """Convert configured friction into per-leg vectorbt fractions.

    ``fees`` = commission (per-share with a minimum) plus half the quoted
    spread; ``slippage`` = the configured impact. Both are charged on entry
    AND exit by vectorbt, so a round trip costs roughly ``2 × (fees + slippage)``.
    The fraction assumes ``notional``-sized orders; a cash-shared wallet's child
    orders are smaller (e.g. ~$2.5k at ``cash_frac`` 0.25), where the minimum
    commission is ~4 bp/leg instead of the modeled ~2.5 — a small, constant
    understatement that cannot reorder parameter sets or Pareto fronts. It is NOT small
    for a many-slot portfolio book: 30 slots of a $30k account place $1,000 orders, where
    the $1 minimum is 10 bp/leg, and at ~1 bp a high-turnover parameter set looks almost
    free. Such a strategy declares its real order size as ``backtest.order_notional`` in
    its ``config.yaml`` (:func:`strategy_costs`).

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


def panel_costs(costs: CostSettings, panel: Panel, *,
                notional: float = 10_000.0) -> tuple[float, float]:
    """:func:`leg_costs` priced at the panel's own median close.

    The per-share commission depends on the share price (500 shares of a $20
    name cost $2.50; 50 shares of a $200 name hit the $1 minimum), so a fixed
    ``price=20`` undercharges cheap names ~4x and overcharges expensive ones.
    The median of the per-symbol median closes is one honest scalar per panel —
    per-symbol fees are not expressible here because vectorbt takes one fee
    fraction per portfolio.

    Args:
        costs: The ``costs:`` block from settings.
        panel: The panel the backtest will run on.
        notional: Assumed position size in dollars.

    Returns:
        ``(fees_fraction, slippage_fraction)`` per leg.
    """
    price = float(panel.close.median().median())
    if not np.isfinite(price) or price <= 0:
        price = 20.0
    return leg_costs(costs, price=price, notional=notional)


def strategy_costs(strategy: LoadedStrategy, panel: Panel,
                   settings: Settings) -> tuple[float, float]:
    """:func:`panel_costs` at the strategy's own order size (``backtest.order_notional``).

    The one place the per-strategy order size enters the cost model, so a backtest, a sweep
    and a notebook cell that rebuilds a book by hand all charge the same fee fraction.

    Args:
        strategy: A loaded strategy; its ``order_notional`` is ``None`` for the $10,000 default.
        panel: The panel the backtest will run on (the scored window, when there is one).
        settings: Global settings (the ``costs:`` block).

    Returns:
        ``(fees_fraction, slippage_fraction)`` per leg.
    """
    return panel_costs(settings.costs, panel, notional=strategy.order_notional or 10_000.0)


def fill_prices(panel: Panel, slippage: float, fill: str = "open"):
    """The ``price`` and ``slippage`` vectorbt fills orders at, under a fill model.

    ``open`` fills every order at the bar's adjusted open. ``pessimistic`` fills every buy
    at the bar's high and every sell at its low — whichever way the order turns out to go,
    which a target-percent rebalance, a retrim or a short cover only reveals inside the
    simulation. vectorbt buys at ``price × (1 + slippage)`` and sells at
    ``price × (1 − slippage)``, so pricing at the bar's midpoint with a half-range slippage
    lands exactly on the extremes: ``mid × (1 ± (H − L)/(H + L))`` is ``H`` for a buy and
    ``L`` for a sell. The configured slippage is added on top, so costs stay charged.
    Orders are still *sized* at the midpoint, so a buy spends its slot plus the half-range
    and the last buy of a fully-invested bar is trimmed to the cash left.

    Args:
        panel: Price panel.
        slippage: The configured per-leg slippage fraction.
        fill: One of :data:`FILLS`.

    Returns:
        ``(price, slippage)`` — ``(panel.open_, slippage)`` under ``open``; a midpoint frame
        and a per-bar slippage frame under ``pessimistic``.

    Raises:
        ValueError: On a fill model not in :data:`FILLS`.
    """
    if fill == "open":
        return panel.open_, slippage
    if fill != "pessimistic":
        raise ValueError(f"unknown fill {fill!r}; choose one of {FILLS}")
    span = panel.high + panel.low
    half_range = ((panel.high - panel.low) / span).clip(lower=0.0).fillna(0.0)
    return span / 2.0, half_range + slippage


def grid_from_config(sweep_cfg: dict[str, list], valid=lambda p: True) -> list[dict]:
    """Expand per-parameter value lists into a filtered cartesian grid.

    Args:
        sweep_cfg: A strategy's ``sweep`` view — ``{param: [values...]}``, assembled
            by the loader from every parameter that declares a ``grid:``. Parameters
            without one are absent here and stay pinned at their shipped ``value``,
            which :func:`sweep` supplies from ``strategy.params``.
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
        init_cash: float = 10_000.0, fill: str = "open", whole_shares: bool = False):
    """Build one vectorbt Portfolio from a strategy's signal dict.

    Dispatches on what the strategy returned. A ``weights`` frame selects
    **portfolio mode** (:func:`_run_portfolio`); a :data:`CASH_FRAC_KEY` scalar
    runs the signals against **one cash-shared wallet**; otherwise the classic
    per-symbol ``from_signals`` path runs. Either way fills happen on the next bar
    at the ``fill`` model's price (:func:`fill_prices`) and equity is valued on the
    adjusted close.

    Args:
        panel: Price panel.
        signals: Strategy output — either :data:`WEIGHTS_KEY` (a target-weight
            frame, negative for a short, optionally with a :data:`RETRIM_K_KEY` scalar to
            cap single-name concentration and a :data:`BORROW_KEY` rate for shorts), or any
            subset of :data:`SIGNAL_KEYS` (boolean frames) and
            :data:`STOP_KEYS` (scalars or frames; stop distances are fractions of
            entry price, NaN = no stop), optionally plus :data:`CASH_FRAC_KEY`
            (the fraction of free cash deployed per entry: a scalar, or a dense per-bar
            frame read at the fill bar — see :func:`cash_frac_size`).
        fees: Per-leg fee fraction (see :func:`leg_costs`).
        slippage: Per-leg slippage fraction.
        init_cash: Starting cash — **per symbol** in per-symbol mode, the **whole
            book** in portfolio mode and in cash-shared signal mode.
        fill: Fill model, one of :data:`FILLS` (a strategy's ``backtest.fill``).
        whole_shares: Round every order down to whole shares (a real account's
            constraint; see ``LoadedStrategy.whole_shares``). Off by default: the engine
            trades fractional shares, which makes every book scale-free.

    Returns:
        A ``vectorbt.Portfolio`` (grouped and cash-sharing in portfolio and
        cash-shared signal modes).

    Raises:
        ValueError: If the signals set both ``weights`` and ``cash_frac`` — the
            two capital models are mutually exclusive — or ``borrow`` without ``weights``,
            or on an unknown ``fill``.
    """
    configure()
    import vectorbt as vbt

    if WEIGHTS_KEY in signals and CASH_FRAC_KEY in signals:
        raise ValueError("signals may set 'weights' OR 'cash_frac', not both")
    if BORROW_KEY in signals and WEIGHTS_KEY not in signals:
        raise ValueError("'borrow' is charged in portfolio mode only (return 'weights')")
    price, slippage = fill_prices(panel, slippage, fill)
    whole = {"size_granularity": 1.0} if whole_shares else {}
    if WEIGHTS_KEY in signals:
        return _run_portfolio(panel, signals[WEIGHTS_KEY], price=price, fees=fees,
                              slippage=slippage, init_cash=init_cash,
                              retrim_k=signals.get(RETRIM_K_KEY),
                              borrow=signals.get(BORROW_KEY), **whole)
    kwargs = {k: signals[k] for k in (*SIGNAL_KEYS, *STOP_KEYS) if k in signals} | whole
    if any(k in signals for k in STOP_KEYS):
        # open= lets a bar that GAPS through a stop fill at its (worse) true open
        # instead of at the stop level, and "stopmarket" makes stop legs pay
        # slippage like any market order — without these, vectorbt's defaults
        # ("stoplimit", close substituted for open) are optimistic on exactly the
        # bars where stops matter most. A pessimistic fill prices a stop leg like
        # every other order ("price": the bar's low for a sell, high for a buy).
        kwargs |= {"open": panel.open_, "high": panel.high, "low": panel.low,
                   "stop_exit_price": "stopmarket" if fill == "open" else "price"}
    if CASH_FRAC_KEY in signals:
        # One shared wallet of init_cash TOTAL: each entry deploys `cash_frac` of the
        # free cash, so the book can never be more than 100% invested. call_seq="auto"
        # puts same-bar sells before buys, so an exit frees the cash a new entry spends.
        # upon_opposite_entry="close": percent sizing cannot reverse a position
        # (vectorbt raises), so an opposite entry closes the open position instead.
        # lock_cash=True: a short's sale proceeds stay RESERVED for buying it back.
        # Without it vectorbt counts them as spendable, so a later long entry buys with
        # them (a $10k wallet ran 280% invested in the regression test) and the buy that
        # covers the short is then cash-limited — its stop fills only part of the position
        # and the rest runs on past the stop. burst_scan measured it on a real book: free
        # cash negative on 11% of bars, split covers, 199 skipped entries (2026-09-27).
        kwargs |= dict(size=cash_frac_size(signals[CASH_FRAC_KEY]), size_type="percent",
                       group_by=True, cash_sharing=True, call_seq="auto",
                       upon_opposite_entry="close", lock_cash=True)
    return vbt.Portfolio.from_signals(
        panel.close, price=price, fees=fees, slippage=slippage,
        init_cash=init_cash, freq=panel.freq,
        # The EXIT wins when an entry and an exit signal land on the same bar.
        # vectorbt's default ("ignore") cancels BOTH: a time stop coinciding with a
        # fresh entry signal was swallowed, so the position ran PAST its supposed
        # hard stop and the entry evaporated too (~11% of informed_flow's top-ranked
        # symbol-days back when it was signal-mode). Exit-wins makes exit_after and
        # the EOD flat true hard stops; the same-bar entry is still dropped, which
        # is the one thing from_signals cannot express (exit then re-enter).
        upon_long_conflict="exit", upon_short_conflict="exit", **kwargs,
    )


def rotation_targets(weights: pd.DataFrame) -> pd.DataFrame:
    """Blank out cells where a name's target is unchanged, so no order is placed.

    Target-percent sizing acts on every cell it is given a number for, so a *dense*
    weight frame silently re-trims a position **every single bar** as prices drift —
    hundreds of tiny orders and a fee drag charged straight against the benchmark. NaN
    means "no order", so masking the unchanged cells keeps trading **event-driven and
    per name**: a name is traded only when *its own* target changes (bought when it
    enters, sold whole when it leaves) and is **left untouched — to drift — while other
    names rotate around it**. This is the difference from a whole-*row* mask, which
    re-emits every held name the instant any one name rotates and so re-trims positions
    the strategy only meant to hold.

    Bar 0 always rotates (the whole first row is emitted) so the initial book is placed.

    Args:
        weights: Target fraction of the book per symbol, per bar.

    Returns:
        The same frame with unchanged cells set to NaN (per name, not per row).
    """
    changed = weights.ne(weights.shift())
    changed.iloc[0] = True
    return weights.where(changed, np.nan)


def _book_weights(pf) -> pd.DataFrame:
    """Each name's realized weight in its shared book, per bar: asset value / book value.

    The realized weight only exists *inside* a run — :func:`rotation_targets` sees targets,
    never live position values — so this is how :func:`_retrim_loop` reads back what a held
    name's drift actually did. Works for both a single grouped book (:func:`_run_portfolio`,
    where ``pf.value()`` is one Series and every column divides by it) and a column-stacked
    per-combo sweep (where it is a per-combo frame and each ``(combo, symbol)`` column
    divides by *its own* combo's value).
    """
    asset_value = pf.asset_value(group_by=False)
    value = pf.value()
    if isinstance(value, pd.Series):
        return asset_value.div(value, axis=0)
    return asset_value.div(value, axis=1, level="combo")


#: Backstop on retrim re-simulations. A held name's book weight cannot exceed 1.0 and a
#: slot targets ``1 / max_positions``, so a name can breach a ``k >= 2`` band at most
#: ``log2(max_positions)`` times before it is capped — ~5 for the widest book. The loop
#: converges well inside this; the cap only guards against a pathological non-convergence.
_MAX_RETRIM_ITERS = 8


def _retrim_loop(build, targets: pd.DataFrame, weights: pd.DataFrame,
                 threshold: pd.DataFrame):
    """Iterated concentration cap: re-simulate, trimming any over-band holding, to a fixed point.

    Portfolio mode leaves a held name to drift (:func:`rotation_targets`), so one runaway
    winner can come to dominate the book. When a strategy sets ``retrim_k`` this caps it:
    after a run, any held name whose *realized* book weight has reached ``threshold``
    (``k ×`` its target) is trimmed back to target on the **next** bar — the freed cash
    returns to the shared wallet, where the next rotation redeploys it into whatever the
    strategy selects next (usually a fresh name). The trim is decided from the realized
    weight at a bar's close and executed at the next bar's open, so it carries **no
    look-ahead** — the same discipline as every other order in the engine.

    Realized weight is knowable only inside a simulation, so this is a simulate-then-adjust
    fixed point, not a preprocessing of the weight frame: run, find the first still-untrimmed
    over-band bar per name, un-mask it (re-emit the target there so target-percent sizing
    sells the name back to target), and repeat until no held name is over the band. A trim
    only ever *lowers* a weight, so the un-masked set grows monotonically and the loop
    converges in a handful of passes (:data:`_MAX_RETRIM_ITERS`). A book with no runaway
    costs exactly one extra weight read and no re-run.

    Args:
        build: ``targets -> vectorbt.Portfolio`` — runs one ``from_orders`` pass. Both call
            sites pass their own (single-book vs per-combo grouped) builder.
        targets: The masked target frame (:func:`rotation_targets` output) to start from.
        weights: The dense (un-masked) target frame — the level a trim rebalances back to.
        threshold: ``k × weights`` — the realized book weight at which a held name is trimmed.

    Returns:
        The final ``vectorbt.Portfolio`` with the cap applied.
    """
    pf = build(targets)
    for _ in range(_MAX_RETRIM_ITERS):
        # Over-band at a bar's CLOSE; act on the NEXT bar (shift) so the trim never uses
        # information from the bar it fills on, and only where the name is still drifting
        # (target NaN) so a real rotation is never fought.
        over = ((_book_weights(pf) >= threshold) & (weights > 0)).shift(1, fill_value=False)
        cand = over & targets.isna()
        # The FIRST still-untrimmed breach per name this pass. Trimming resets the name to
        # target, so a later re-doubling is found and capped on a subsequent pass.
        edge = cand & (cand.cumsum() == 1)
        if not edge.to_numpy().any():
            break
        targets = targets.where(~edge, weights)
        pf = build(targets)
    return pf


def _borrow_fees(targets: pd.DataFrame, borrow, fees: float) -> pd.DataFrame:
    """The fee frame that charges every short its borrow: ``fees`` plus a surcharge.

    A short accrues ``rate × days / 365`` for every calendar day it is held (the rate read
    at each bar; weekends and holidays count, as a stock loan does). The accrued fee is
    charged as a fraction of the order that buys the short back — scaled by the short's
    size over the order's size, since that order may also buy into a long. A short still
    open on the last bar has no such order, so its accrual is charged on the order that
    opened it instead: the cost lands earlier than in reality, but it is never skipped,
    which a backtest window cut short would otherwise do to every position it ends on.

    Sizes are read from the TARGETS (what each order asks for), so a short re-sized while
    open is charged at its size at the buy-back — exact for books that open a short once and
    buy it back whole (event-driven rotation), approximate otherwise. A re-trim that buys back
    less than a quarter of the short does NOT trigger the charge (engine proposal #48): the
    surcharge is a fee FRACTION ``accrued × |w| / |x − w|`` of the order, which a hair-sized
    re-trim turns into most of the order's value (etf_trend's review measured borrow charged
    4-5× over what was owed); its accrual carries on to the next substantial buy-back or the
    end of the window instead.

    Args:
        targets: Masked target weights (:func:`rotation_targets`): NaN = no order.
        borrow: Yearly borrow rate per name and bar (frame on ``targets``' index), or one
            scalar for every name; NaN reads as 0.
        fees: The per-leg fee fraction every order pays.

    Returns:
        A frame shaped like ``targets``: ``fees`` everywhere, plus the borrow surcharge on
        the orders that pay it.
    """
    index = targets.index
    if isinstance(borrow, pd.DataFrame):
        rate = borrow.reindex(index=index, columns=targets.columns).fillna(0.0).to_numpy()
    else:
        rate = np.full(targets.shape, float(borrow))
    years = np.r_[0.0, np.diff(index.asi8) / 86_400e9] / 365.0
    accrued = np.cumsum(rate * years[:, None], axis=0)      # accrued[i]: bars 1..i
    t = targets.to_numpy(dtype=float)
    out = np.zeros_like(t)
    for j in range(t.shape[1]):
        orders = np.flatnonzero(~np.isnan(t[:, j]))
        w, since, opened = 0.0, 0, None                     # in-force weight, last charge
        for k in orders:
            x = t[k, j]
            small = x < 0 and (x - w) < 0.25 * (-w)         # a hair re-trim of the short
            if w < 0 and x > w and not small:               # buys (part of) the short back
                out[k, j] += (accrued[k, j] - accrued[since, j]) * (-w) / (x - w)
                since = k
            if x < 0 and w >= 0:                            # a new short starts here
                opened, since = (k, w), k
            w = x
        if w < 0 and opened is not None:                    # still short on the last bar
            k, before = opened
            out[k, j] += (accrued[-1, j] - accrued[since, j]) * (-w) / (before - t[k, j])
    return pd.DataFrame(fees + out, index=index, columns=targets.columns)


def _call_order(targets: pd.DataFrame, cats: np.ndarray | None = None,
                groups: int = 1) -> np.ndarray:
    """Per-bar execution order (a vectorbt ``call_seq`` array) for a portfolio-mode book.

    Each order is classed against the name's IN-FORCE target before the bar (its last emitted
    target): 0 a long cut or exit, 1 a short cover or cut, 2 a new or larger short (a long
    flipped short included), 3 a long bought or added (a short flipped long included); names
    with no order run last. Sales and covers free the cash that new shorts need as collateral
    and longs spend (engine proposal #46). vectorbt's ``call_seq="auto"`` sorts by order value
    and runs a new short (a sale) BEFORE the cover (a buy) that would release locked cash, so on
    a full book it refused or part-filled the new short.

    Args:
        targets: Masked target weights (NaN = no order); with ``groups > 1``, ``groups`` equal
            consecutive column blocks (a sweep's per-combo books).
        cats: Optional class per cell (-1 = derive it), for a re-sent order that repeats an
            earlier order and so keeps that order's class (:func:`_working_orders`).
        groups: Number of equal consecutive column groups, each its own book.

    Returns:
        ``call_seq[row, k]`` = the position, inside its group, of the column called k-th
        (vectorbt's convention).
    """
    t = targets.to_numpy(dtype=float)
    prev = pd.DataFrame(t).ffill().shift(1).fillna(0.0).to_numpy()
    has = ~np.isnan(t)
    x = np.where(has, t, prev)
    cat = np.full(t.shape, 4, dtype=np.int64)
    cat[has & (x > 0) & (x > prev)] = 3
    cat[has & (x < 0) & (x < prev)] = 2
    cat[has & (prev < 0) & (x >= prev) & (x <= 0)] = 1
    cat[has & (prev > 0) & (x <= prev) & (x >= 0)] = 0
    if cats is not None:
        cat = np.where(cats >= 0, cats, cat)
    width = t.shape[1] // groups
    return np.concatenate([np.argsort(cat[:, g * width:(g + 1) * width], axis=1, kind="stable")
                           for g in range(groups)], axis=1)


#: An order counts as filled when it executed at least this share of the shares it asked
#: for (the rest is fees, slippage and the valuation gap between the open and the order).
_FILL_SHARE = 0.98
#: Backstop on :func:`_working_orders` passes. Each pass resolves every unfilled order in one
#: re-simulation (an order keeps working on every later priced bar until it fills), so a book
#: converges in two or three; the cap only guards a cascade that never settles.
_MAX_WORKING_ITERS = 12


def _unfilled(pf, targets: np.ndarray, price: np.ndarray, value_px: np.ndarray, *,
              init_cash: float, groups: int, granularity: float | None):
    """The emitted orders a run refused or part-filled, read from its positions (no order log).

    vectorbt's order log holds a record for EVERY cell of the frame (600k records for a
    1,200-bar, 500-name book), so the requested shares are rebuilt instead: target weight ×
    the group's value at the bar's open ÷ the fill price, against the shares the bar actually
    added. An order on an unpriced bar (vectorbt drops it) is not counted.

    Returns:
        ``(rows, cols, cats)`` of every order that executed less than :data:`_FILL_SHARE` of
        its shares (and more than one ``granularity`` short of them), with each order's class
        for :func:`_call_order`: 0 long cut, 1 short cover, 2 new or larger short, 3 long buy.
    """
    assets = pf.assets().to_numpy(dtype=float)
    cash = np.asarray(pf.cash(), dtype=float)
    cash = cash[:, None] if cash.ndim == 1 else cash
    n, m = assets.shape
    prev = np.vstack([np.zeros((1, m)), assets[:-1]])
    prev_cash = np.vstack([np.full((1, cash.shape[1]), float(init_cash)), cash[:-1]])
    held = np.nan_to_num(prev * value_px).reshape(n, groups, m // groups).sum(axis=2)
    book = prev_cash + held                                    # each group's value at the open
    rows, cols = np.nonzero(~np.isnan(targets) & np.isfinite(price))
    group = cols // (m // groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        want = targets[rows, cols] * book[rows, group] / price[rows, cols]
    need = want - prev[rows, cols]
    got = assets[rows, cols] - prev[rows, cols]
    big = np.abs(need) * price[rows, cols] > 1e-4 * np.abs(book[rows, group])
    with np.errstate(invalid="ignore", divide="ignore"):
        short = (got / need < _FILL_SHARE) & (np.abs(need - got) > (granularity or 0.0))
    bad = big & short & np.isfinite(need)
    rows, cols, need, p0 = rows[bad], cols[bad], need[bad], prev[rows[bad], cols[bad]]
    cats = np.where(need > 0, np.where(p0 < 0, 1, 3), np.where(p0 > 0, 0, 2))
    return rows, cols, cats


def _working_orders(build, targets: pd.DataFrame, price: pd.DataFrame, *, init_cash: float,
                    groups: int = 1, granularity: float | None = None,
                    report: dict | None = None):
    """Run a book with every refused or part-filled order kept working until it fills.

    :func:`rotation_targets` emits a target only when it changes, so an order vectorbt
    refuses or part-fills (no free cash under ``lock_cash``: a cover of a short that more than
    doubled, a buy on a fully invested book) used to be lost: the book held what the strategy
    had exited, or missed what it wanted, until the name's next signal change (Wayfair 2020:
    46 bars; GME 2024: 32 bars; engine proposal #47). Now each such order is re-sent on every
    later priced bar up to the name's next own order, the way a working order stays on a
    broker's book, and dropped after the bar it fills on. A re-sent order keeps the class of
    the order it repeats in :func:`_call_order`, and ``build`` receives the frame its borrow is
    charged on: each chain's LAST attempt, the one that completes it.

    Args:
        build: ``(targets, cats, fee_targets) -> vectorbt.Portfolio``: one ``from_orders`` pass.
        targets: The masked target frame (:func:`rotation_targets`, perhaps with trims).
        price: The fill price frame (``price`` of ``from_orders``), aligned to ``targets``.
        init_cash: Each group's starting cash.
        groups: Equal consecutive column groups (a sweep's per-combo books).
        granularity: ``size_granularity`` of the run (1.0 = whole shares), so a remainder
            smaller than one share is not chased.
        report: Given, the number of orders still unfilled is stored as ``report["stuck"]``
            and nothing is warned: the caller warns once about its FINAL run (the retrim loop
            simulates several times, and only its last run is the book).

    Returns:
        The final ``vectorbt.Portfolio``. Without ``report``, a run that still holds unfilled
        orders when the loop stops warns (:func:`_warn_unfilled`, engine proposal #49).
    """
    cur = targets.to_numpy(dtype=float).copy()
    own = ~np.isnan(cur)                                       # the strategy's own orders
    px = price.to_numpy(dtype=float)
    value_px = pd.DataFrame(px).ffill().to_numpy()
    cats = np.full(cur.shape, -1, dtype=np.int64)
    # (row, col) of a failed order -> [every priced row up to the name's next own order,
    # the number of those rows currently re-sent]
    chains: dict[tuple[int, int], list] = {}

    def frames():
        fee = cur.copy()
        for (r, c), (rows, used) in chains.items():
            fee[r, c] = np.nan
            fee[rows[:used - 1], c] = np.nan
        wrap = lambda a: pd.DataFrame(a, index=targets.index, columns=targets.columns)  # noqa: E731
        return wrap(cur), cats, wrap(fee)

    def failures(pf):
        rows, cols, kinds = _unfilled(pf, cur, px, value_px, init_cash=init_cash,
                                      groups=groups, granularity=granularity)
        return set(zip(rows.tolist(), cols.tolist(), strict=True)), rows, cols, kinds

    pf = build(*frames())
    for _ in range(_MAX_WORKING_ITERS):
        failed, rows, cols, kinds = failures(pf)
        changed = False
        for (r, c), entry in chains.items():        # end each chain on the bar it fills
            full, used = entry
            filled = [k for k in range(used) if (int(full[k]), c) not in failed]
            want = filled[0] + 1 if filled else len(full)
            if want != used:
                cur[full[want:], c], cats[full[want:], c] = np.nan, -1
                cur[full[:want], c], cats[full[:want], c] = cur[r, c], cats[full[0], c]
                entry[1] = want
                changed = True
        working = {(int(row), c) for (_, c), (full, used) in chains.items()
                   for row in full[:used]}
        for r, c, kind in zip(rows.tolist(), cols.tolist(), kinds.tolist(), strict=True):
            if (r, c) in chains or (r, c) in working or r + 1 >= len(cur):
                continue
            nxt = np.flatnonzero(own[r + 1:, c])
            end = r + 1 + (int(nxt[0]) if len(nxt) else len(cur) - r - 1)
            full = r + 1 + np.flatnonzero(np.isfinite(px[r + 1:end, c]))
            if not len(full):
                continue
            cur[full, c], cats[full, c] = cur[r, c], kind
            chains[(r, c)] = [full, len(full)]
            changed = True
        if not changed:
            break
        pf = build(*frames())
    failed, _, _, _ = failures(pf)
    working = {(int(row), c) for (_, c), (full, used) in chains.items() for row in full[:used]}
    stuck = sum(all((int(row), c) in failed for row in full[:used])
                for (_, c), (full, used) in chains.items())
    stuck += sum(cell not in chains and cell not in working for cell in failed)
    if report is None:
        _warn_unfilled(stuck)
    else:
        report["stuck"] = stuck
    return pf


def _warn_unfilled(stuck: int) -> None:
    """Warn that a book ran below its targets (engine proposal #49) if an order stayed unfilled."""
    if stuck:
        warnings.warn(f"{stuck} portfolio orders stayed unfilled (lock_cash left no free "
                      "cash): the book ran below its targets. A full long/short book needs "
                      "gross <= ~0.90 (engine proposal #49).", RuntimeWarning, stacklevel=4)


def _run_portfolio(panel: Panel, weights: pd.DataFrame, *, price: pd.DataFrame, fees: float,
                   slippage, init_cash: float, retrim_k: float | None = None,
                   borrow=None, size_granularity: float | None = None, working: bool = True):
    """One shared cash account rebalanced to target weights — buying requires selling.

    The ``from_orders`` call site. ``from_signals`` cannot do this: it rejects
    target-percent sizing outright (*"Only SizeType.Amount, Value, and Percent are
    supported"*), because boolean entries carry no notion of "rebalance to this book".

    A negative weight is a short. Its sale proceeds stay locked for the buy-back
    (``lock_cash``), exactly as in the cash-shared wallet: without it vectorbt counted them
    as spendable, so a long/short book silently ran above 100% gross (``trend_following``
    measured free cash below zero on 94% of bars, gross up to 114%). Under the lock a short
    reserves its whole notional, so gross exposure — longs plus shorts — tops out near 100%
    of the book when orders are placed: a dollar-neutral book is about 50/50.

    Args:
        panel: Price panel.
        weights: Target fraction of the book per symbol, per bar; negative = short (gross
            ``|weights|`` summing to <= 1; the remainder is held as cash). Already
            execution-timed by the strategy.
        price: Fill price frame (from :func:`fill_prices`).
        fees: Per-leg fee fraction.
        slippage: Per-leg slippage fraction, or per-bar frame (from :func:`fill_prices`).
        init_cash: The whole book's starting cash.
        retrim_k: Concentration cap. ``None`` (default) leaves held names to drift
            untouched — one ``from_orders`` pass, identical to the original behavior and
            zero extra cost. A number ``> 1`` trims a held name back to its slot once its
            realized book weight reaches ``retrim_k ×`` its target, via :func:`_retrim_loop`.
            Longs only: a short is never trimmed.
        borrow: Yearly borrow rate of each short (:data:`BORROW_KEY`; frame or scalar), or
            ``None`` (default): shorts borrow for free.
        size_granularity: ``1.0`` rounds every order down to whole shares, so a name priced
            above its slot is never bought and each slot's remainder waits in cash;
            ``None`` (default) trades fractional shares.
        working: Keep a refused or part-filled order working on the following bars until it
            fills (:func:`_working_orders`, the default). ``False`` sends every order once:
            :func:`benchmark`, whose late listings buy with the cash they find.

    Returns:
        A grouped, cash-sharing ``vectorbt.Portfolio``.
    """
    import vectorbt as vbt

    whole = {} if size_granularity is None else {"size_granularity": size_granularity}
    targets = rotation_targets(weights)

    def build(targets, cats=None, fee_targets=None):
        return vbt.Portfolio.from_orders(
            panel.close, size=targets, size_type="targetpercent", price=price,
            fees=fees if borrow is None else _borrow_fees(
                targets if fee_targets is None else fee_targets, borrow, fees),
            slippage=slippage, init_cash=init_cash, **whole,
            group_by=True, cash_sharing=True, lock_cash=True,
            # MANDATORY, not cosmetic: within a bar, sales and covers run before the new
            # shorts and buys they fund (_call_order, engine proposal #46). Without an order
            # a buy is rejected because the sale funding it has not executed yet, and under
            # lock_cash a new short is starved by the cover that would release its collateral.
            call_seq=_call_order(targets, cats),
            freq=panel.freq,
        )

    report: dict = {}

    def simulate(targets):
        if not working:
            return build(targets)
        return _working_orders(build, targets, price.reindex_like(targets),
                               init_cash=init_cash, granularity=size_granularity, report=report)

    pf = (simulate(targets) if retrim_k is None
          else _retrim_loop(simulate, targets, weights, retrim_k * weights))
    _warn_unfilled(report.get("stuck", 0))           # the final run's, not a retrim pass's
    return pf


def _score_window(panel: Panel, signals: dict, start) -> tuple[Panel, dict]:
    """Cut a panel, and the signals already computed on it, to the bars from ``start`` on.

    The warm-start half of :func:`run_strategy`: the signals were built on the whole panel,
    so their rolling windows are already warm at ``start``; only the book and the metrics are
    restricted. A portfolio-mode book buys its targets on the window's first bar. A signal
    strategy starts flat and waits for its next entry, because an entry fired before
    ``start`` is outside the window.

    Args:
        panel: The full panel the signals were computed on.
        signals: The strategy's signal dict; every frame or series on the panel's index is
            cut, scalars (``cash_frac``, ``retrim_k``, stop distances) pass through.
        start: First scored bar (date string or timestamp; naive means the panel's tz).

    Returns:
        ``(window_panel, window_signals)``.

    Raises:
        ValueError: If fewer than two bars remain from ``start`` on.
    """
    index = panel.close.index
    ts = pd.Timestamp(start)
    if ts.tzinfo is None and index.tz is not None:
        ts = ts.tz_localize(index.tz)
    if int((index >= ts).sum()) < 2:
        raise ValueError(f"score_from={start!r} leaves fewer than 2 bars of the panel")

    def cut(value):
        if (isinstance(value, pd.DataFrame | pd.Series)
                and isinstance(value.index, pd.DatetimeIndex)):
            return value.loc[value.index >= ts]
        return value

    window = replace(panel, close=cut(panel.close), open_=cut(panel.open_), high=cut(panel.high),
                     low=cut(panel.low), volume=cut(panel.volume), rth=cut(panel.rth),
                     features=None if panel.features is None
                     else {k: cut(v) for k, v in panel.features.items()})
    return window, {k: cut(v) for k, v in signals.items()}


def _run_scored(strategy: LoadedStrategy, panel: Panel, params: dict | None,
                settings: Settings, score_from):
    """Signals on the whole panel, book on ``[score_from, end]`` — see :func:`run_strategy`."""
    signals = strategy.signals(panel, **{**strategy.params, **(params or {})})
    if score_from is not None:
        panel, signals = _score_window(panel, signals, score_from)
    fees, slippage = strategy_costs(strategy, panel, settings)
    pf = run(panel, signals, fees=fees, slippage=slippage,
             init_cash=settings.backtest.init_cash, fill=strategy.fill,
             whole_shares=strategy.whole_shares)
    return pf, panel, fees, slippage


def run_strategy(strategy: LoadedStrategy, panel: Panel, params: dict | None = None,
                 *, settings: Settings, score_from=None):
    """Run one backtest of a strategy with its config defaults (plus overrides).

    Args:
        strategy: A loaded strategy (see :func:`trading_simple.strategies.load`).
        panel: Price panel matching the strategy's ``data:`` block.
        params: Overrides merged over ``strategy.params``.
        settings: Global settings (costs, init_cash).
        score_from: **Warm start** for an out-of-sample window. ``None`` backtests the whole
            panel. A date runs ``signals()`` on the whole panel — so rolling windows, EMAs
            and moving averages enter the window with their history, as they would live —
            and trades the book only from that bar on (:func:`_score_window`). Pass the
            long panel plus the window's first date; never a panel pre-sliced to the window,
            which restarts every indicator on the window's first bar.

    Returns:
        A ``vectorbt.Portfolio`` on the scored bars, filled under the strategy's
        ``backtest.fill`` model.
    """
    return _run_scored(strategy, panel, params, settings, score_from)[0]


#: Cached ``(total_return, per-bar returns)`` of the equal-weight benchmark, keyed by
#: panel identity + costs. Keyed by content (not ``id()``) because walk-forward slices
#: create and discard panels, and a recycled id would silently return a stale benchmark.
_BENCHMARK: dict[tuple, tuple[float, pd.Series]] = {}


def benchmark(panel: Panel, *, fees: float, slippage: float,
              fill: str = "open") -> tuple[float, pd.Series]:
    """Equal-weight buy-and-hold of every symbol in the panel — the hurdle to clear.

    A long-only strategy in a rising market earns a positive Sharpe simply by being
    invested, so an *absolute* Sharpe cannot tell skill from market exposure. This is
    the reference that can: own everything, from the first bar to the last, paying the
    same costs. A strategy that does not beat it is not adding value, however good its
    headline number looks.

    Built in **portfolio mode** — one shared book, equal weights, no rotation after bar
    zero — so it is the same construction a portfolio-mode strategy runs: both books
    start from the same ``init_cash`` and both are fully invested, which is what makes
    ``excess_cagr`` an apples-to-apples number rather than a comparison between a
    partly-idle book and a fully-deployed one. (For a per-symbol strategy the figure is
    unchanged: an equal-weight fixed-pot buy-and-hold return is *identically* the mean
    of the per-symbol returns, which is what :func:`portfolio_stats` reports there.)

    It fills under the **same fill model** as the strategy it is the hurdle for, exactly as
    it pays the same costs: under ``pessimistic`` its one buy per name prints at that bar's
    high, so the comparison stays about selection, not about who got the kinder fills.

    Cached per (panel, costs, fill) — the benchmark is identical across optimizer trials,
    and recomputing it every trial would roughly double the optimization's runtime.

    Args:
        panel: The price panel to benchmark over.
        fees: Per-leg fee fraction (see :func:`leg_costs`).
        slippage: Per-leg slippage fraction.
        fill: Fill model, one of :data:`FILLS` — must match the strategy's.

    Returns:
        ``(total_return, per_bar_returns)``.
    """
    key = (panel.close.index[0], panel.close.index[-1], panel.close.shape,
           tuple(panel.symbols), fees, slippage, fill,
           # Listing pattern: the weights below depend on WHERE each name's prices
           # start, so two panels sharing window/shape/symbols but not listing dates
           # (e.g. one refreshed with a newly backfilled IPO) must not share a cache row.
           tuple(panel.close.notna().idxmax()))
    if key not in _BENCHMARK:
        weights = pd.DataFrame(1.0 / len(panel.close.columns), index=panel.close.index,
                               columns=panel.close.columns)   # own everything, forever:
        # Masked to bars where the name actually trades: a late-listed symbol has NaN
        # prices at bar 0, so its one dense-frame order would be silently skipped and
        # its 1/N would sit in cash for the whole backtest — an understated hurdle.
        # With the mask, rotation_targets sees NaN -> 1/N on the first valid bar and
        # buys the name at its listing, which is what buy-and-hold actually means.
        weights = weights.where(panel.close.notna())
        configure()
        price, slip = fill_prices(panel, slippage, fill)
        # Constant weights -> a name's listing bar is its only rotation: buy + hold. Sent
        # once (working=False): a late listing buys with the cash it finds, never by
        # waiting for cash a buy-and-hold book does not free.
        pf = _run_portfolio(panel, weights, price=price, fees=fees, slippage=slip,
                            init_cash=10_000.0, working=False)
        curve = equity_curve(pf)["value"].pct_change().fillna(0.0)
        if len(_BENCHMARK) > 16:                    # bounded: walk-forward makes many panels
            _BENCHMARK.clear()
        _BENCHMARK[key] = (float(portfolio_stats(pf)["total_return"]), curve)
    return _BENCHMARK[key]


def active_stats(pf, panel: Panel, *, fees: float, slippage: float,
                 total_return: float, fill: str = "open") -> dict:
    """Benchmark-relative metrics: does this beat simply owning the universe?

    Both are **per year**, so they are comparable across strategies with different
    backtest windows and across walk-forward folds of different lengths. A cumulative
    excess return is not: it rewards a long window for being long.

    Args:
        pf: The strategy's ``vectorbt.Portfolio``.
        panel: The panel it ran on.
        fees: Per-leg fee fraction (must match the run).
        slippage: Per-leg slippage fraction (must match the run).
        total_return: The strategy's ``portfolio_stats`` total return.
        fill: Fill model (must match the run — pass ``strategy.fill``).

    Returns:
        ``{"excess_cagr", "alpha_cagr", "beta", "info_ratio", "tracking_error"}``.
        ``excess_cagr`` is the strategy's annualized return minus the benchmark's — what
        it earned per year over simply owning the universe. ``beta`` is the book's
        per-bar return sensitivity to the benchmark's, and ``alpha_cagr`` is the return
        left after the market exposure the book actually ran: ``cagr − beta × benchmark
        cagr``. A fully invested long book has beta ≈ 1 and alpha ≈ excess, so nothing is
        given away; a book that is flat most of the time has beta ≈ 0 and still pays for
        the exposure it did take — the honest hurdle for a near-zero-beta long/short
        scanner, which no cash-heavy strategy can clear by sitting out (its alpha is then
        just its own small return). ``tracking_error`` is the annualized standard
        deviation of the *active* per-bar return (strategy minus benchmark): how far the
        strategy's path strays from simply owning the universe (0 = it IS the benchmark).
        ``info_ratio`` is the annualized mean active return over that same deviation —
        excess per unit of straying, the Sharpe ratio of the value the strategy adds.
    """
    bench_return, bench_curve = benchmark(panel, fees=fees, slippage=slippage, fill=fill)
    own = equity_curve(pf)["value"].pct_change().fillna(0.0)
    active = own - bench_curve
    sd = float(active.std())
    index = panel.close.index
    years = _years(index)
    ann = np.sqrt(len(index) / years)               # works for daily and intraday
    var = float(bench_curve.var())
    beta = float(own.cov(bench_curve) / var) if var else 0.0
    cagr, bench_cagr = _cagr(total_return, years), _cagr(bench_return, years)
    return {
        "excess_cagr": cagr - bench_cagr,
        "alpha_cagr": cagr - beta * bench_cagr,
        "beta": beta,
        "info_ratio": float(active.mean() / sd * ann) if sd else 0.0,
        "tracking_error": sd * ann,
    }


def evaluate(strategy: LoadedStrategy, panel: Panel, params: dict, *, settings: Settings,
             score_from=None) -> dict:
    """Backtest one parameter set and reduce it to plain metric floats.

    The single entry point shared by notebooks, tests, and the Optuna
    objective — whatever runs through here is comparable.

    Args:
        strategy: A loaded strategy.
        panel: Price panel.
        params: Full parameter dict for this run.
        settings: Global settings.
        score_from: Warm start (see :func:`run_strategy`): signals on the whole panel,
            book, benchmark and every metric on the bars from this date on. Use it for every
            holdout and walk-forward test window.

    Returns:
        ``{total_return, cagr, sharpe, volatility, max_drawdown, win_rate, pct_profitable,
        n_trades, trades_per_year, years, excess_cagr, alpha_cagr, beta, info_ratio,
        tracking_error}`` — the last five are measured against an equal-weight buy-and-hold
        of the same (scored) panel (see :func:`benchmark`, :func:`active_stats`).
        :mod:`trading_simple.optimize` scores ``excess_cagr`` against ``volatility`` by
        default: the return must beat the market, the risk is the book's own.
    """
    pf, window, fees, slippage = _run_scored(strategy, panel, params, settings, score_from)
    stats = portfolio_stats(pf)
    stats.update(active_stats(pf, window, fees=fees, slippage=slippage,
                              total_return=stats["total_return"], fill=strategy.fill))
    return stats


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
        DataFrame with the grid's parameter columns plus every metric in
        :data:`METRIC_COLS` except ``info_ratio``, ``tracking_error``, ``alpha_cagr`` and
        ``beta`` (which need a per-combo equity curve the column-stacked sweep does not
        keep — use :func:`evaluate` for those).
    """
    configure(settings)
    import vectorbt as vbt

    chunk_size = chunk_size or settings.backtest.chunk_size
    if strategy.whole_shares:
        raise NotImplementedError(
            "sweep() trades fractional shares; run a whole_shares strategy through run()")
    fees, slippage = strategy_costs(strategy, panel, settings)
    price, fill_slip = fill_prices(panel, slippage, strategy.fill)   # same fills as run()
    init_cash = settings.backtest.init_cash
    index, columns = panel.close.index, panel.close.columns
    years = _years(index)
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

        # A pessimistic fill's slippage is a per-bar frame; the default stays a scalar, so
        # an open-fill sweep allocates nothing extra.
        slip = tile(fill_slip) if isinstance(fill_slip, pd.DataFrame) else fill_slip

        if WEIGHTS_KEY in combo_signals[0]:
            if any(BORROW_KEY in s for s in combo_signals):
                raise NotImplementedError("sweep() does not charge 'borrow'; run the combos "
                                          "with evaluate() / run_strategy()")
            # Portfolio mode: group by the combo level, so each combo is its own shared
            # cash account of `init_cash` — not one giant book pooling every combo.
            weights_stacked = stack([s[WEIGHTS_KEY] for s in combo_signals])
            targets = stack([rotation_targets(s[WEIGHTS_KEY]) for s in combo_signals])

            books = len(ids)

            def build(t, cats=None, fee_targets=None, slip=slip, n=books):
                return vbt.Portfolio.from_orders(
                    tile(panel.close), size=t, size_type="targetpercent",
                    price=tile(price), fees=fees, slippage=slip,
                    init_cash=init_cash, group_by="combo", cash_sharing=True,
                    lock_cash=True,               # short proceeds never fund a long
                    call_seq=_call_order(t, cats, groups=n),   # as in _run_portfolio
                    freq=panel.freq,
                )

            report: dict = {}

            def simulate(t, build=build, n=books, report=report):
                return _working_orders(build, t, tile(price), init_cash=init_cash, groups=n,
                                       report=report)

            retrim_k = combo_signals[0].get(RETRIM_K_KEY)
            if retrim_k is None:
                pf = simulate(targets)
            else:
                # Same concentration cap as _run_portfolio, per combo. retrim_k is a fixed
                # policy (never swept), so every combo shares it — but a per-column threshold
                # tolerates a hand-swept k all the same. _book_weights divides each
                # (combo, symbol) column by its own combo's book value (group_by="combo").
                k_per_col = np.repeat(
                    [float(s[RETRIM_K_KEY]) if s.get(RETRIM_K_KEY) is not None else np.nan
                     for s in combo_signals], len(columns))
                pf = _retrim_loop(simulate, targets, weights_stacked,
                                  weights_stacked.mul(k_per_col, axis=1))
            _warn_unfilled(report.get("stuck", 0))
            # A one-combo chunk has one group and cannot be indexed into further.
            rows += [{**grid[cid], **_grouped_stats(pf[cid] if len(ids) > 1 else pf)}
                     for cid in ids]
            continue

        kwargs = {
            key: stack([as_frame(sig.get(key), empty_bool) for sig in combo_signals])
            for key in SIGNAL_KEYS
        }
        if used_stops:
            kwargs["sl_stop"] = stack([as_frame(s.get("sl_stop"), no_stop) for s in combo_signals])
            kwargs["tp_stop"] = stack([as_frame(s.get("tp_stop"), no_stop) for s in combo_signals])
            kwargs["sl_trail"] = stack(
                [as_frame(s.get("sl_trail", False), empty_bool) for s in combo_signals])
            # Same pessimistic stop-fill semantics as run() — sweep numbers must
            # match what evaluate() reports for the same combo.
            kwargs |= {"open": tile(panel.open_), "high": tile(panel.high),
                       "low": tile(panel.low),
                       "stop_exit_price": "stopmarket" if strategy.fill == "open" else "price"}

        grouped = CASH_FRAC_KEY in combo_signals[0]
        if grouped:
            # Cash-shared signal mode: one wallet PER COMBO — group on the combo level,
            # each entry deploying that combo's fraction of its own free cash. Reduced
            # via _grouped_stats like the weights branch, so sweep numbers match what
            # evaluate() reports for the same combo. Scalars stay a one-row broadcast; a
            # per-entry frame (risk-per-trade sizing) stacks like a stop frame.
            fracs = [s[CASH_FRAC_KEY] for s in combo_signals]
            if any(isinstance(f, pd.DataFrame) for f in fracs):
                size = stack([as_frame(cash_frac_size(f), no_stop) for f in fracs])
            else:
                size = np.repeat([float(f) for f in fracs], len(columns))[None, :]
            kwargs |= dict(
                size=size, size_type="percent", group_by="combo", cash_sharing=True,
                call_seq="auto", upon_opposite_entry="close",
                lock_cash=True,   # short proceeds stay reserved for the cover, as in run()
            )

        pf = vbt.Portfolio.from_signals(
            tile(panel.close), price=tile(price),
            fees=fees, slippage=slip, init_cash=init_cash, freq=panel.freq,
            # Exit-wins conflict resolution, same as run() — sweep numbers must
            # match what evaluate() reports for the same combo.
            upon_long_conflict="exit", upon_short_conflict="exit", **kwargs,
        )
        if grouped:
            # A one-combo chunk has one group and cannot be indexed into further.
            rows += [{**grid[cid], **_grouped_stats(pf[cid] if len(ids) > 1 else pf)}
                     for cid in ids]
            continue
        per_combo = _metric_frame(pf, cols).groupby(level="combo").agg({
            "total_return": "mean", "sharpe": "mean", "volatility": "mean",
            "max_drawdown": "mean", "win_rate": "mean", "profitable": "mean",
            "n_trades": "sum",
        }).rename(columns={"profitable": "pct_profitable"})
        rows += [{**grid[cid], **per_combo.loc[cid].to_dict()} for cid in ids]

    stats = pd.DataFrame(rows)
    stats["n_trades"] = stats["n_trades"].astype(int)
    # Annualized return over an equal-weight buy-and-hold of the same panel. A long-only
    # sweep in a rising market posts a positive return everywhere, so the absolute number
    # cannot rank combos on skill; this can. Per-year, so folds of unequal length stay
    # comparable. (info_ratio needs per-combo equity curves the sweep does not keep.)
    bench_return, _ = benchmark(panel, fees=fees, slippage=slippage, fill=strategy.fill)
    bench_cagr = _cagr(bench_return, years)
    stats["cagr"] = [_cagr(r, years) for r in stats["total_return"]]
    stats["excess_cagr"] = stats["cagr"] - bench_cagr
    stats["years"] = years
    stats["trades_per_year"] = stats["n_trades"] / years
    return stats


# ---------------------------------------------------------------------------
# Reducing portfolios to numbers and tables.
# ---------------------------------------------------------------------------
def is_grouped(pf) -> bool:
    """True for a portfolio-mode (shared cash account) portfolio.

    Worth its own function because the two modes report metrics in **different
    shapes**: ungrouped, ``pf.total_return()`` is a per-symbol Series; grouped, it is
    the one book's scalar. Reading the wrong one is silent, not loud (see
    :func:`_metric_frame`).
    """
    return bool(pf.wrapper.grouper.is_grouped())


def portfolio_stats(pf) -> dict:
    """Summary of one portfolio as plain floats — the same keys in both modes.

    Per-symbol mode: means over symbols for return/Sharpe/drawdown/win-rate.
    Portfolio mode: the shared book's own return/Sharpe/drawdown/win-rate, which are
    the real numbers rather than an average of fictions.

    ``pct_profitable`` is the "does it generalize across names, or is one lucky stock
    carrying it?" guard, and it survives the mode change: per-symbol it is the share of
    symbols with a positive total return; in portfolio mode a symbol has no standalone
    return (it shares the book's cash), so it becomes the share of **traded** symbols
    whose realized profit is positive. Same question, same spirit.

    ``volatility`` is the annual risk rate: the standard deviation of the book's per-bar
    returns, annualized on the same bars-per-calendar-year clock as ``sharpe`` (the
    per-symbol mean in per-symbol mode). It is one of the default study objectives
    (:data:`trading_simple.optimize.OBJECTIVES`).

    Args:
        pf: A ``vectorbt.Portfolio``.

    Returns:
        Metric dict (see :func:`evaluate`).
    """
    index = pf.wrapper.index
    years = _years(index)
    if is_grouped(pf):
        stats = _grouped_stats(pf)
    else:
        m = _metric_frame(pf, pf.wrapper.columns)
        stats = {
            "total_return": float(m["total_return"].mean()),
            "sharpe": float(m["sharpe"].mean()),
            "volatility": float(m["volatility"].mean()),
            "max_drawdown": float(m["max_drawdown"].mean()),
            "win_rate": float(m["win_rate"].mean()),
            "pct_profitable": float(m["profitable"].mean()),
            "n_trades": int(m["n_trades"].sum()),
        }
    stats["cagr"] = _cagr(stats["total_return"], years)
    stats["years"] = years
    stats["trades_per_year"] = stats["n_trades"] / years
    return stats


def _bar_sharpe(returns, index):
    """Sharpe of per-bar returns, annualized by ACTUAL bars per calendar year.

    vectorbt's own ``sharpe_ratio`` annualizes with its ``year_freq`` default of 365
    calendar days over the bar frequency: √365 on daily bars that number ~252 a year
    (~20% high), and ~2.3x high on 1m bars. ``info_ratio`` already annualizes with
    ``sqrt(len(index)/years)``; the diagnostic Sharpe must sit on the same clock or
    the two risk-adjusted numbers on one trial disagree by construction. Diagnostic
    only — the objectives are benchmark-relative and never read this.

    Args:
        returns: Per-bar returns — a Series (one book) or DataFrame (per column).
        index: The panel/backtest index the returns live on.

    Returns:
        Float for a Series, per-column Series for a DataFrame; non-finite where
        the standard deviation is zero (callers null those out).
    """
    ann = np.sqrt(len(index) / _years(index))
    with np.errstate(invalid="ignore", divide="ignore"):   # flat returns -> NaN, nulled by callers
        return returns.mean() / returns.std() * ann


def _bar_volatility(returns, index):
    """Annualized volatility of per-bar returns, on :func:`_bar_sharpe`'s clock.

    Args:
        returns: Per-bar returns — a Series (one book) or DataFrame (per column).
        index: The panel/backtest index the returns live on.

    Returns:
        Float for a Series, per-column Series for a DataFrame.
    """
    return returns.std() * np.sqrt(len(index) / _years(index))


def _grouped_stats(pf) -> dict:
    """The shared book's metrics, read as the scalars vectorbt returns when grouped."""
    def scalar(fn) -> float:
        try:
            value = float(fn())
        except Exception:
            return float("nan")
        return value if np.isfinite(value) else float("nan")

    trades = trade_records(pf)
    by_symbol = trades.groupby("symbol")["pnl"].sum() if len(trades) else pd.Series(dtype=float)
    return {
        "total_return": scalar(pf.total_return),
        "sharpe": scalar(lambda: _bar_sharpe(pf.returns(), pf.wrapper.index)),
        "volatility": scalar(lambda: _bar_volatility(pf.returns(), pf.wrapper.index)),
        "max_drawdown": scalar(pf.max_drawdown),
        "win_rate": scalar(pf.trades.win_rate),
        # No symbol has a standalone return here — they share one pot — so ask the
        # question of realized profit instead: how many of the names it actually traded
        # made money? A single name carrying the whole book still shows up.
        "pct_profitable": float((by_symbol > 0).mean()) if len(by_symbol) else 0.0,
        "n_trades": int(len(trades)),
    }


def symbol_stats(pf) -> pd.DataFrame:
    """Per-symbol metric table — the cross-section behind :func:`portfolio_stats`.

    The verification view: an aggregate can hide an edge carried by one name, so judge a
    parameter set on this breakdown before believing it.

    **In portfolio mode, read ``total_return`` as a contribution, not a return.** The
    symbols share one cash account, so a name's figure is its profit divided by the
    *whole book's* starting cash — how much it added to the book — not what it earned on
    its own. The column still ranks the names correctly; it just does not mean "this
    stock returned 8%".

    Args:
        pf: A ``vectorbt.Portfolio``.

    Returns:
        One row per symbol (``total_return, sharpe, volatility, max_drawdown, win_rate,
        n_trades``), sorted by total return descending.
    """
    m = _metric_frame(pf, pf.wrapper.get_columns(group_by=False)).drop(columns="profitable")
    m["n_trades"] = m["n_trades"].astype(int)
    m.index.name = "symbol"
    return m.sort_values("total_return", ascending=False)


def _metric_frame(pf, cols: pd.Index) -> pd.DataFrame:
    """Per-column metrics with non-finite values (inf Sharpe etc.) nulled out.

    ``group_by=False`` is passed explicitly so this keeps returning per-symbol numbers
    on a cash-sharing portfolio, where the bare calls would hand back the *group's*
    scalar. That failure is silent — ``safe()`` swallows the shape mismatch and yields
    an all-NaN frame — so it must not be left to chance.
    """
    def safe(fn) -> pd.Series:
        try:
            series = pd.Series(fn(group_by=False), dtype="float64").reindex(cols)
        except Exception:
            return pd.Series(np.nan, index=cols, dtype="float64")
        return series.where(np.isfinite(series))

    total_return = safe(pf.total_return)
    return pd.DataFrame({
        "total_return": total_return,
        "sharpe": safe(lambda group_by: _bar_sharpe(pf.returns(group_by=group_by),
                                                    pf.wrapper.index)),
        "volatility": safe(lambda group_by: _bar_volatility(pf.returns(group_by=group_by),
                                                            pf.wrapper.index)),
        "max_drawdown": safe(pf.max_drawdown),
        "win_rate": safe(pf.trades.win_rate),
        "profitable": (total_return > 0).astype(float),
        "n_trades": safe(pf.trades.count).fillna(0),
    })


def best_combo(stats: pd.DataFrame, *, metric: str = "excess_cagr") -> dict:
    """The parameter dict of a sweep's best row by ``metric`` (NaNs ignored).

    Defaults to ``excess_cagr`` — the *annualized* return over an equal-weight
    buy-and-hold of the same panel — and both halves of that matter. Not raw Sharpe: a
    long-only strategy earns a positive Sharpe just by being invested in a rising
    market, so picking on the absolute number selects for market exposure rather than
    skill. And not a *cumulative* excess return: that rewards a long window for being
    long, so it cannot rank walk-forward folds of unequal length against each other.

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
    return {k: v for k, v in row.items() if k not in METRIC_COLS}


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

    In per-symbol mode the symbols run as independent portfolios, so combined value is
    the row-sum of their equity curves; in portfolio mode there is one book already.
    ``fees`` is the cumulative commission+slippage actually paid across every fill (so
    an open position's entry fee counts), and ``realized_pnl`` is the cumulative net
    profit/loss booked as positions close (already net of their own fees; open positions
    contribute nothing). Both accumulate across symbols and start at zero.

    Args:
        pf: A ``vectorbt.Portfolio``.
        init_cash: Starting cash — per symbol in per-symbol mode, the whole book in
            portfolio mode. Omit to use the first bar's value as the base.

    Returns:
        DataFrame with ``value``, ``return``, ``fees`` and ``realized_pnl`` columns.
    """
    value = pf.value()
    if isinstance(value, pd.Series):
        value = value.to_frame()
    total = value.sum(axis=1)
    if init_cash is None:
        base = float(total.iloc[0])
    else:
        # One book in portfolio mode; one book per symbol otherwise.
        base = init_cash if is_grouped(pf) else init_cash * value.shape[1]
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
