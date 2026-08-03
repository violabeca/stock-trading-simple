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
  than left to dominate the book. See :func:`_run_portfolio`.
* **Cash-shared signals** (``entries``/``exits`` + ``cash_frac``) — the wallet
  variant of the signal path: ONE shared account of ``init_cash`` *total*, and
  each entry deploys ``cash_frac`` of the currently *free* cash, so the book can
  never be more than 100% invested. This is how a single real account of
  ``init_cash`` would trade a multi-symbol signal scanner. Stops work per column
  as usual. Percent sizing cannot *reverse* a position, so an opposite-direction
  entry closes the open position instead (``upon_opposite_entry="close"``); the
  new direction opens on a later bar if its signal persists.

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
#: Returning this scalar (0 < f <= 1) alongside ``entries``/``exits`` opts a signal
#: strategy into **one cash-shared wallet** of ``init_cash`` TOTAL: each entry deploys
#: this fraction of the group's currently free cash, so total invested can never
#: exceed the book. Percent sizing cannot reverse a position, so an opposite-direction
#: entry closes the open position instead of flipping it.
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
#: Every metric a sweep/trial frame may carry; anything else in such a frame is a
#: strategy parameter. One definition, so a new metric cannot be mistaken for a knob.
#: ``years``/``trades_per_year`` are sample-size columns and belong here for exactly
#: that reason — they describe the run, they are not knobs to turn.
METRIC_COLS = frozenset({
    "total_return", "cagr", "sharpe", "max_drawdown", "win_rate", "pct_profitable",
    "n_trades", "trades_per_year", "years", "excess_cagr", "info_ratio",
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
    understatement that cannot reorder parameter sets or Pareto fronts.

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
        init_cash: float = 10_000.0):
    """Build one vectorbt Portfolio from a strategy's signal dict.

    Dispatches on what the strategy returned. A ``weights`` frame selects
    **portfolio mode** (:func:`_run_portfolio`); a :data:`CASH_FRAC_KEY` scalar
    runs the signals against **one cash-shared wallet**; otherwise the classic
    per-symbol ``from_signals`` path runs. Either way fills happen at the adjusted
    next-bar open and equity is valued on the adjusted close.

    Args:
        panel: Price panel.
        signals: Strategy output — either :data:`WEIGHTS_KEY` (a target-weight
            frame, optionally with a :data:`RETRIM_K_KEY` scalar to cap single-name
            concentration), or any subset of :data:`SIGNAL_KEYS` (boolean frames) and
            :data:`STOP_KEYS` (scalars or frames; stop distances are fractions of
            entry price, NaN = no stop), optionally plus :data:`CASH_FRAC_KEY`
            (a scalar fraction of free cash deployed per entry).
        fees: Per-leg fee fraction (see :func:`leg_costs`).
        slippage: Per-leg slippage fraction.
        init_cash: Starting cash — **per symbol** in per-symbol mode, the **whole
            book** in portfolio mode and in cash-shared signal mode.

    Returns:
        A ``vectorbt.Portfolio`` (grouped and cash-sharing in portfolio and
        cash-shared signal modes).

    Raises:
        ValueError: If the signals set both ``weights`` and ``cash_frac`` — the
            two capital models are mutually exclusive.
    """
    configure()
    import vectorbt as vbt

    if WEIGHTS_KEY in signals and CASH_FRAC_KEY in signals:
        raise ValueError("signals may set 'weights' OR 'cash_frac', not both")
    if WEIGHTS_KEY in signals:
        return _run_portfolio(panel, signals[WEIGHTS_KEY], fees=fees, slippage=slippage,
                              init_cash=init_cash, retrim_k=signals.get(RETRIM_K_KEY))
    kwargs = {k: signals[k] for k in (*SIGNAL_KEYS, *STOP_KEYS) if k in signals}
    if any(k in signals for k in STOP_KEYS):
        # open= lets a bar that GAPS through a stop fill at its (worse) true open
        # instead of at the stop level, and "stopmarket" makes stop legs pay
        # slippage like any market order — without these, vectorbt's defaults
        # ("stoplimit", close substituted for open) are optimistic on exactly the
        # bars where stops matter most.
        kwargs |= {"open": panel.open_, "high": panel.high, "low": panel.low,
                   "stop_exit_price": "stopmarket"}
    if CASH_FRAC_KEY in signals:
        # One shared wallet of init_cash TOTAL: each entry deploys `cash_frac` of the
        # free cash, so the book can never be more than 100% invested. call_seq="auto"
        # puts same-bar sells before buys, so an exit frees the cash a new entry spends.
        # upon_opposite_entry="close": percent sizing cannot reverse a position
        # (vectorbt raises), so an opposite entry closes the open position instead.
        kwargs |= dict(size=float(signals[CASH_FRAC_KEY]), size_type="percent",
                       group_by=True, cash_sharing=True, call_seq="auto",
                       upon_opposite_entry="close")
    return vbt.Portfolio.from_signals(
        panel.close, price=panel.open_, fees=fees, slippage=slippage,
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


def _run_portfolio(panel: Panel, weights: pd.DataFrame, *, fees: float, slippage: float,
                   init_cash: float, retrim_k: float | None = None):
    """One shared cash account rebalanced to target weights — buying requires selling.

    The ``from_orders`` call site. ``from_signals`` cannot do this: it rejects
    target-percent sizing outright (*"Only SizeType.Amount, Value, and Percent are
    supported"*), because boolean entries carry no notion of "rebalance to this book".

    Args:
        panel: Price panel.
        weights: Target fraction of the book per symbol, per bar (rows summing to <= 1;
            the remainder is held as cash). Already execution-timed by the strategy.
        fees: Per-leg fee fraction.
        slippage: Per-leg slippage fraction.
        init_cash: The whole book's starting cash.
        retrim_k: Concentration cap. ``None`` (default) leaves held names to drift
            untouched — one ``from_orders`` pass, identical to the original behavior and
            zero extra cost. A number ``> 1`` trims a held name back to its slot once its
            realized book weight reaches ``retrim_k ×`` its target, via :func:`_retrim_loop`.

    Returns:
        A grouped, cash-sharing ``vectorbt.Portfolio``.
    """
    import vectorbt as vbt

    def build(targets):
        return vbt.Portfolio.from_orders(
            panel.close, size=targets, size_type="targetpercent",
            price=panel.open_, fees=fees, slippage=slippage, init_cash=init_cash,
            group_by=True, cash_sharing=True,
            # MANDATORY, not cosmetic: sorts sells before buys within a bar. Without it a
            # buy gets partially rejected because the sell funding it has not executed yet,
            # silently leaving cash idle — the exact bug this whole mode exists to remove.
            call_seq="auto",
            freq=panel.freq,
        )

    targets = rotation_targets(weights)
    if retrim_k is None:
        return build(targets)
    return _retrim_loop(build, targets, weights, retrim_k * weights)


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
    fees, slippage = panel_costs(settings.costs, panel)
    return run(panel, strategy.signals(panel, **merged), fees=fees, slippage=slippage,
               init_cash=settings.backtest.init_cash)


#: Cached ``(total_return, per-bar returns)`` of the equal-weight benchmark, keyed by
#: panel identity + costs. Keyed by content (not ``id()``) because walk-forward slices
#: create and discard panels, and a recycled id would silently return a stale benchmark.
_BENCHMARK: dict[tuple, tuple[float, pd.Series]] = {}


def benchmark(panel: Panel, *, fees: float, slippage: float) -> tuple[float, pd.Series]:
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

    Cached per (panel, costs) — the benchmark is identical across optimizer trials, and
    recomputing it every trial would roughly double the optimization's runtime.

    Args:
        panel: The price panel to benchmark over.
        fees: Per-leg fee fraction (see :func:`leg_costs`).
        slippage: Per-leg slippage fraction.

    Returns:
        ``(total_return, per_bar_returns)``.
    """
    key = (panel.close.index[0], panel.close.index[-1], panel.close.shape,
           tuple(panel.symbols), fees, slippage,
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
        pf = run(panel, {WEIGHTS_KEY: weights},   # constant weights -> a name's listing
                 fees=fees, slippage=slippage)    # bar is its only rotation: buy + hold
        curve = equity_curve(pf)["value"].pct_change().fillna(0.0)
        if len(_BENCHMARK) > 16:                    # bounded: walk-forward makes many panels
            _BENCHMARK.clear()
        _BENCHMARK[key] = (float(portfolio_stats(pf)["total_return"]), curve)
    return _BENCHMARK[key]


def active_stats(pf, panel: Panel, *, fees: float, slippage: float,
                 total_return: float) -> dict:
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

    Returns:
        ``{"excess_cagr", "info_ratio"}``. ``excess_cagr`` is the strategy's annualized
        return minus the benchmark's — the alpha you would have earned per year.
        ``info_ratio`` is the annualized mean of the *active* per-bar return (strategy
        minus benchmark) over its standard deviation: the Sharpe ratio of the value the
        strategy adds, with the market removed.
    """
    bench_return, bench_curve = benchmark(panel, fees=fees, slippage=slippage)
    active = equity_curve(pf)["value"].pct_change().fillna(0.0) - bench_curve
    sd = float(active.std())
    index = panel.close.index
    years = _years(index)
    ann = np.sqrt(len(index) / years)               # works for daily and intraday
    return {
        "excess_cagr": _cagr(total_return, years) - _cagr(bench_return, years),
        "info_ratio": float(active.mean() / sd * ann) if sd else 0.0,
    }


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
        ``{total_return, cagr, sharpe, max_drawdown, win_rate, pct_profitable, n_trades,
        trades_per_year, years, excess_cagr, info_ratio}`` — ``excess_cagr`` and
        ``info_ratio`` are measured against an equal-weight buy-and-hold of the same
        panel (see :func:`benchmark`), and are what :mod:`trading_simple.optimize`
        maximizes: an absolute Sharpe rewards market exposure, not stock selection.
    """
    pf = run_strategy(strategy, panel, params, settings=settings)
    stats = portfolio_stats(pf)
    fees, slippage = panel_costs(settings.costs, panel)
    stats.update(active_stats(pf, panel, fees=fees, slippage=slippage,
                              total_return=stats["total_return"]))
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
        :data:`METRIC_COLS` except ``info_ratio`` (which needs a per-combo equity
        curve the column-stacked sweep does not keep — use :func:`evaluate` for that).
    """
    configure(settings)
    import vectorbt as vbt

    chunk_size = chunk_size or settings.backtest.chunk_size
    fees, slippage = panel_costs(settings.costs, panel)
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

        if WEIGHTS_KEY in combo_signals[0]:
            # Portfolio mode: group by the combo level, so each combo is its own shared
            # cash account of `init_cash` — not one giant book pooling every combo.
            weights_stacked = stack([s[WEIGHTS_KEY] for s in combo_signals])
            targets = stack([rotation_targets(s[WEIGHTS_KEY]) for s in combo_signals])

            def build(t):
                return vbt.Portfolio.from_orders(
                    tile(panel.close), size=t, size_type="targetpercent",
                    price=tile(panel.open_), fees=fees, slippage=slippage,
                    init_cash=init_cash, group_by="combo", cash_sharing=True,
                    call_seq="auto", freq=panel.freq,
                )

            retrim_k = combo_signals[0].get(RETRIM_K_KEY)
            if retrim_k is None:
                pf = build(targets)
            else:
                # Same concentration cap as _run_portfolio, per combo. retrim_k is a fixed
                # policy (never swept), so every combo shares it — but a per-column threshold
                # tolerates a hand-swept k all the same. _book_weights divides each
                # (combo, symbol) column by its own combo's book value (group_by="combo").
                k_per_col = np.repeat(
                    [float(s[RETRIM_K_KEY]) if s.get(RETRIM_K_KEY) is not None else np.nan
                     for s in combo_signals], len(columns))
                pf = _retrim_loop(build, targets, weights_stacked,
                                  weights_stacked.mul(k_per_col, axis=1))
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
                       "low": tile(panel.low), "stop_exit_price": "stopmarket"}

        grouped = CASH_FRAC_KEY in combo_signals[0]
        if grouped:
            # Cash-shared signal mode: one wallet PER COMBO — group on the combo level,
            # each entry deploying that combo's fraction of its own free cash. Reduced
            # via _grouped_stats like the weights branch, so sweep numbers match what
            # evaluate() reports for the same combo.
            kwargs |= dict(
                size=np.repeat([float(s[CASH_FRAC_KEY]) for s in combo_signals],
                               len(columns))[None, :],
                size_type="percent", group_by="combo", cash_sharing=True,
                call_seq="auto", upon_opposite_entry="close",
            )

        pf = vbt.Portfolio.from_signals(
            tile(panel.close), price=tile(panel.open_),
            fees=fees, slippage=slippage, init_cash=init_cash, freq=panel.freq,
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
            "total_return": "mean", "sharpe": "mean", "max_drawdown": "mean",
            "win_rate": "mean", "profitable": "mean", "n_trades": "sum",
        }).rename(columns={"profitable": "pct_profitable"})
        rows += [{**grid[cid], **per_combo.loc[cid].to_dict()} for cid in ids]

    stats = pd.DataFrame(rows)
    stats["n_trades"] = stats["n_trades"].astype(int)
    # Annualized return over an equal-weight buy-and-hold of the same panel. A long-only
    # sweep in a rising market posts a positive return everywhere, so the absolute number
    # cannot rank combos on skill; this can. Per-year, so folds of unequal length stay
    # comparable. (info_ratio needs per-combo equity curves the sweep does not keep.)
    bench_return, _ = benchmark(panel, fees=fees, slippage=slippage)
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
        One row per symbol (``total_return, sharpe, max_drawdown, win_rate,
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
