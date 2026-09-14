"""Short-term momentum, only where the tape is deep: buy last month's winners among the
names that trade the most.

Thesis. Sorting stocks on their previous month's return and on share turnover at the
same time splits the classic one-month effect in two: **low-turnover** names *reverse*
(the Jegadeesh/Lehmann short-term reversal), **high-turnover** names *continue*. Medhat
& Schmeling (RFS 2022) document this on US and international stocks and find the
continuation leg "as profitable and as persistent as conventional price momentum",
surviving transaction costs and "strongest among the largest, most liquid, and most
extensively covered stocks" — the names a $10k book can actually trade at the modeled
costs. Blitz, Hanauer, Honarvar, Huisman & van Vliet (FAJ 2023) show a composite of such
short-term signals earns significant *net* alpha in a liquid global universe, works
long-only, and is robust to implementation lags of several days — the property this
engine's next-bar-open fill needs. The interpretation both offer: some traders
under-appreciate the information in prices, and heavy trading is where that shows.

This strategy trades only the continuation leg, long only, and treats the turnover
conditioning as *the* thesis. ``turnover_q: 0.0`` (everyone eligible) is plain one-month
momentum — the control the paper says should be weaker or reversed — and
``polarity: reversal`` is the other half of the falsification: on the high-turnover slice
it must lose.

Construction (pure pandas; the feature layer only supplies the share count):

1. ``ret`` — the trailing ``formation``-bar return, ending ``skip`` bars ago.
2. ``turnover`` — the trailing ``turnover_window``-bar mean volume divided by shares
   outstanding (SEC XBRL ``fund_shares``, point-in-time), after re-expressing the share
   count in the units of the split-adjusted volume series (see :func:`_in_volume_units`).
   A name whose cross-sectional turnover rank is below ``turnover_q`` is **dropped from
   the cross-section**, not handed a neutral score, so it neither ranks nor pushes anyone
   else's percentile around. If ``fund_shares`` is missing or too thin to rank
   (``min_coverage``), turnover falls back to dollar volume — a size × turnover mix, not
   turnover, so the notebook reports which one ran.
3. Every ``rebalance_days`` bars, hold the top ``max_positions`` eligible names by ``ret``
   (the bottom, under ``polarity: reversal``), one fixed slot of ``1 / max_positions``
   each; carry the book between rebalances (a monthly cadence *is* the holding period the
   literature tests, and it needs no stateful hysteresis loop).

**One shared book, portfolio mode.** ``signals()`` returns a ``weights`` frame, so the
engine runs one cash account of ``init_cash`` *total*: a new name is funded by selling an
old one, a thin selection holds cash rather than concentrating, and the ``retrim_k``
concentration cap trims a runaway winner back to its slot (fixed policy, never searched —
see :func:`trading_simple.backtest._retrim_loop`). Every position is next-bar-open
executed (``bt.next_bar``); ``fund_shares`` is already point-in-time lagged in
:mod:`trading_simple.features`; the price inputs are lagged by construction of ``ret``.
There is no look-ahead — with one documented residual in :func:`_in_volume_units`.

:func:`scores` exposes the intermediates so the notebook can test the thesis *without*
the trading engine: rank IC of ``ret`` against cross-sectionally demeaned forward returns,
by turnover tercile. That test is pre-registered and runs before any optimization.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel

#: Consecutive as-reported share counts whose ratio crosses this (either way) are read as a
#: split or reverse split — a change of UNITS, not of the float. Buybacks and issuance move a
#: share count by single-digit percent a quarter; nothing but a corporate action doubles it.
_SPLIT_JUMP = 1.9

_POLARITIES = ("momentum", "reversal")


def signals(
    panel: Panel,
    *,
    formation: int = 21,
    skip: int = 0,
    turnover_window: int = 63,
    turnover_q: float = 0.5,
    polarity: str = "momentum",
    rebalance_days: int = 21,
    max_positions: int = 10,
    min_coverage: float = 0.5,
    retrim_k: float | None = 2.0,
) -> dict:
    """Build long-only target weights: top short-term movers among high-turnover names.

    Args:
        panel: Price panel; ``panel.features`` should carry ``fund_shares`` (built by
            :func:`trading_simple.features.attach_features`). Without it, turnover falls
            back to dollar volume.
        formation: Trailing bars the momentum return is measured over (~1 month at 21).
        skip: Bars skipped between the end of the formation window and today. ``0`` is the
            paper's plain previous-month return; a few bars skips the very short-term
            reversal that the low-turnover leg is made of.
        turnover_window: Trailing bars the turnover level is averaged over.
        turnover_q: Cross-sectional turnover quantile a name must reach to be ranked at
            all (``0.5`` = the more-traded half). ``0.0`` disables the conditioning — the
            plain-momentum control.
        polarity: ``"momentum"`` holds the top movers, ``"reversal"`` the bottom ones —
            the falsification control (the thesis says reversal loses where turnover is
            high).
        rebalance_days: Bars between re-selections; the book is carried in between.
        max_positions: Slots in the book, each a fixed ``1 / max_positions`` of the capital
            at entry. A risk decision, not a searched knob: fewer slots is a purer signal
            and a lumpier book.
        min_coverage: ``fund_shares`` must cover at least this share of the universe on its
            best-covered bar to define turnover; otherwise dollar volume stands in.
        retrim_k: Concentration cap — a held name that grows to ``retrim_k ×`` its slot is
            trimmed back to it by the engine; ``None`` disables.

    Returns:
        ``{"weights", "retrim_k"}`` — a target-weight frame, next-bar-open executed, which
        puts the run in the engine's **portfolio mode** (one shared cash account), plus the
        concentration-cap scalar. Long only; the rebalance cadence is the exit rule.
    """
    parts = _compose(panel, formation=formation, skip=skip, turnover_window=turnover_window,
                     turnover_q=turnover_q, polarity=polarity, min_coverage=min_coverage)
    held = _select(parts["score"], max_positions=max_positions, rebalance_days=rebalance_days)
    return {"weights": bt.next_bar(held).astype(float) / max_positions, "retrim_k": retrim_k}


def _compose(
    panel: Panel,
    *,
    formation: int,
    skip: int,
    turnover_window: int,
    turnover_q: float,
    polarity: str,
    min_coverage: float,
    **_,
) -> dict:
    """Build every intermediate the selection rests on (shared by signals + scores).

    Args:
        panel: Price panel, optionally carrying ``fund_shares``.
        formation: Momentum lookback in bars.
        skip: Bars skipped before today.
        turnover_window: Turnover averaging window in bars.
        turnover_q: Turnover-rank floor for eligibility.
        polarity: ``"momentum"`` or ``"reversal"``.
        min_coverage: Breadth floor for trusting ``fund_shares``.
        **_: Ignored, so callers can splat a full parameter dict.

    Returns:
        ``{"ret", "turnover", "turnover_source", "turnover_rank", "eligible", "score",
        "score_pct"}``. ``score`` is the signed formation return, NaN for ineligible
        names; ``score_pct`` is its cross-sectional percentile rank; ``turnover_source``
        is ``"shares"`` or ``"dollar_volume"``.
    """
    close = panel.close
    ret = close.shift(skip) / close.shift(skip + formation) - 1.0
    turnover, source = _turnover(panel, turnover_window, min_coverage)
    turnover_rank = turnover.rank(axis=1, pct=True)
    # NaN turnover -> NaN rank -> False: a name without a turnover reading is not ranked.
    eligible = ret.notna() & (turnover_rank >= turnover_q)
    signed = ret if polarity == "momentum" else -ret
    score = signed.where(eligible)
    return {"ret": ret, "turnover": turnover, "turnover_source": source,
            "turnover_rank": turnover_rank, "eligible": eligible, "score": score,
            "score_pct": score.rank(axis=1, pct=True)}


def _turnover(panel: Panel, window: int, min_coverage: float) -> tuple[pd.DataFrame, str]:
    """Trailing share turnover per name, or dollar volume when shares are unavailable.

    Args:
        panel: Price panel; ``panel.features["fund_shares"]`` is used when present and
            broad enough.
        window: Trailing bars to average over (``min_periods`` is half of it).
        min_coverage: Breadth floor for ``fund_shares``.

    Returns:
        ``(turnover, source)`` — the ``(bars x symbols)`` level and ``"shares"`` or
        ``"dollar_volume"``.
    """
    min_periods = max(1, window // 2)
    shares = (panel.features or {}).get("fund_shares")
    if shares is not None and _coverage(shares) >= min_coverage:
        volume = panel.volume.rolling(window, min_periods=min_periods).mean()
        return volume / _in_volume_units(shares).replace(0.0, np.nan), "shares"
    dollar = (panel.close * panel.volume).rolling(window, min_periods=min_periods).mean()
    return dollar, "dollar_volume"


def _in_volume_units(shares: pd.DataFrame) -> pd.DataFrame:
    """Re-express as-reported share counts in the units of the split-adjusted volume series.

    The daily bars store *split-adjusted* volume (Yahoo restates the whole history into the
    latest share units), while ``fund_shares`` is *as reported* at each filing. Dividing one
    by the other therefore overstates pre-split turnover by the full future split factor —
    and since splits follow run-ups, that quietly pushes past winners into the
    "high-turnover" bucket. The fix reads each ≥ :data:`_SPLIT_JUMP` jump between consecutive
    filings as a unit change and multiplies every earlier bar by the product of all later
    jumps, which is exactly the restatement the volume series already carries.

    This is a *units* correction, not information: a split ratio is a deterministic
    corporate action. The one residual it cannot close: a split that happens **after the
    panel ends** but before the data was backfilled leaves the whole window's volume in
    newer units than the last in-window filing — that name's turnover is then overstated
    by that factor throughout, uniformly, so its cross-sectional rank is biased for the
    whole window rather than shifted at a point in time.

    Args:
        shares: The forward-filled, point-in-time ``fund_shares`` frame.

    Returns:
        A frame of the same shape, in end-of-window share units.
    """
    ratio = shares / shares.shift(1)
    jump = ratio.where((ratio >= _SPLIT_JUMP) | (ratio <= 1.0 / _SPLIT_JUMP), 1.0)
    # Product of every jump strictly AFTER each bar: converts that bar's units into the
    # units of the last filing in the window.
    future = jump.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    return shares * future


def _select(score: pd.DataFrame, *, max_positions: int, rebalance_days: int) -> pd.DataFrame:
    """Hold the top ``max_positions`` names by score, re-picked every ``rebalance_days`` bars.

    On a rebalance bar the best-scored eligible names fill the slots; between rebalances
    the holding set is carried forward, so turnover is paced by the cadence rather than by
    every wiggle in the ranking. Ineligible names (NaN score) can never be selected. The
    rebalance grid is anchored at bar 0; the first bars fall inside the ``formation`` /
    ``turnover_window`` warm-up, so the first pick with any names happens on the first
    rebalance bar after both are available.

    Args:
        score: The ``(bars x symbols)`` signed score; NaN where ineligible.
        max_positions: Number of slots.
        rebalance_days: Bars between re-selections.

    Returns:
        A boolean ``(bars x symbols)`` mask of names the book wants to own, on the *signal*
        clock (not yet shifted for execution — :func:`signals` does that).
    """
    ranked = score.rank(axis=1, ascending=False, method="first")   # 1 = best
    target = ranked.le(max_positions).astype(float)                # NaN rank -> False -> 0
    is_rebalance = pd.Series(False, index=score.index)
    is_rebalance.iloc[::rebalance_days] = True
    return target.where(is_rebalance).ffill().fillna(0.0) > 0.5


def _coverage(frame: pd.DataFrame) -> float:
    """Fraction of the universe a frame covers on its best-covered bar.

    Args:
        frame: A ``(bars x symbols)`` level frame.

    Returns:
        The max over bars of the per-bar share of non-NaN cells; 0.0 when empty.
    """
    if frame.empty or not frame.shape[1]:
        return 0.0
    return float(frame.notna().mean(axis=1).max())


def valid(params: dict) -> bool:
    """Reject incoherent combos (tolerates partial dicts from sweep grids).

    Args:
        params: Candidate parameter combo (may hold only the swept keys).

    Returns:
        True when the windows are positive, ``skip`` is non-negative, the turnover floor is
        a quantile below 1 (``1.0`` would make every name ineligible), ``polarity`` is one
        of the two legs, and the book has a slot and a cadence.
    """
    return (params.get("formation", 21) >= 1
            and params.get("skip", 0) >= 0
            and params.get("turnover_window", 63) >= 1
            and 0.0 <= params.get("turnover_q", 0.5) < 1.0
            and params.get("polarity", "momentum") in _POLARITIES
            and params.get("max_positions", 10) >= 1
            and params.get("rebalance_days", 21) >= 1)


_DEFAULTS = {name: p.default for name, p in inspect.signature(signals).parameters.items()
             if p.default is not inspect.Parameter.empty}


def scores(panel: Panel, **params) -> dict:
    """Expose the intermediates for the notebook's signal check (no engine involved).

    Shares its whole computation with :func:`signals` (via :func:`_compose`), so the
    rank-IC / decile-spread test sees exactly the ranking the backtest trades. Missing
    parameters default to :func:`signals`' own signature.

    Args:
        panel: The panel the backtest ran on, with ``panel.features`` attached.
        **params: Any :func:`signals` parameters to override.

    Returns:
        The :func:`_compose` result — ``ret``, ``turnover``, ``turnover_source``,
        ``turnover_rank``, ``eligible``, ``score``, ``score_pct``.
    """
    p = {**_DEFAULTS, **params}
    return _compose(panel, **p)
