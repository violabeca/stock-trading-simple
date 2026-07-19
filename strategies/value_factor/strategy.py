"""Value + quality cross-sectional ranker, daily: buy cheap, profitable names.

Thesis: two of the most-replicated equity factors are slow, fundamentals-driven,
and near-orthogonal to the flow signals in ``informed_flow`` —

1. **Value** — cheap stocks outperform expensive ones over the long run
   (Fama-French HML). We express it two ways that a single price ratio cannot:
   **book-to-price** (equity / market cap) and **earnings yield** (trailing-
   twelve-month net income / market cap).
2. **Quality** — profitable firms outperform unprofitable ones at the same price
   (Novy-Marx 2013). We use **gross profitability** (TTM gross profit / total
   assets), the cleanest measured-at-the-top-of-the-income-statement definition,
   which Novy-Marx shows has as much predictive power as book-to-price and is
   strongest *among* value names.

All three come from the ``fundamentals`` sleeve (SEC EDGAR XBRL, point-in-time by
filing date; see :mod:`trading_simple.features`). Fundamentals are levels, so the
feature layer forward-fills each from its filing; this module forms the
price-relative ratios here (market cap needs price, which lives on the Panel, not
the feature table), ranks each factor cross-sectionally, blends them into one
composite rank, and holds the top names.

**Construction.** Percentile-rank each present factor across the *eligible*
cross-section (a name missing any used factor is dropped from the ranking
outright — not handed a neutral score — so it neither ranks nor shifts anyone
else's percentile), blend with tunable weights (an equal-weighted rank composite
resists overfitting far better than a fitted blend at this scale), then hold the
top ``max_positions`` names, refreshed on a ``rebalance_days`` cadence. Value and
quality turn over on the scale of quarters, so a monthly rebalance captures the
signal while keeping trading costs low — and it needs no stateful hysteresis loop,
just a periodic re-selection.

**One shared book, portfolio mode.** Like every shipped strategy this returns a
``weights`` frame, so the engine runs one shared cash account of ``init_cash``
*total*, split into ``max_positions`` fixed slots of ``1 / max_positions`` each —
a new name is funded by selling an old one, and a thin selection holds cash rather
than concentrating. A per-symbol cash account would be fatal for a ranker: cash
attached to a name it is flat in sits idle, so the book would be structurally
under-invested against a fully-invested benchmark however well it ranks.

A factor that is missing, or that covers too little of the universe to rank
meaningfully (see ``min_coverage``), is dropped from the composite and its weight
renormalized away — so the strategy degrades gracefully and gains a factor without
an engine change. :func:`scores` exposes the intermediate factor frames and the
composite rank for the notebook's rank-IC / decile-spread signal check.

Every position is next-bar-open executed (``bt.next_bar``), and every feature is
already point-in-time lagged in :mod:`trading_simple.features`, so there is no
look-ahead.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel

#: Factor key -> the composite weight parameter that scales it.
_WEIGHT_KEY = {"bp": "w_bp", "ey": "w_ey", "gp": "w_gp"}


def signals(
    panel: Panel,
    *,
    w_bp: float = 1.0,
    w_ey: float = 1.0,
    w_gp: float = 1.0,
    rebalance_days: int = 21,
    max_positions: int = 15,
    min_coverage: float = 0.5,
) -> dict:
    """Build long-only target weights from the ranked value+quality composite.

    Args:
        panel: Price panel; ``panel.features`` must carry the fundamentals
            matrices (``fund_equity``, ``fund_shares``, ``fund_ni_ttm``,
            ``fund_gp_ttm``, ``fund_assets``) built by
            :func:`trading_simple.features.attach_features`.
        w_bp: Composite weight on book-to-price (value).
        w_ey: Composite weight on earnings yield (value).
        w_gp: Composite weight on gross profitability (quality).
        rebalance_days: Bars between re-selections; the book is held fixed between
            them (value/quality are slow — ~1 month is the sensible default).
        max_positions: Number of slots in the book. Each holds a fixed
            ``1 / max_positions`` of the capital, so the book is fully invested
            when all slots are filled and holds cash when fewer names qualify.
        min_coverage: A factor covering less than this fraction of the universe on
            its best-covered bar is dropped and its weight renormalized away.

    Returns:
        ``{"weights"}`` — a target-weight frame, next-bar-open executed, which puts
        the run in the engine's **portfolio mode**: one shared cash account, so
        buying is funded by selling. Long only.
    """
    parts = _compose(panel, w_bp=w_bp, w_ey=w_ey, w_gp=w_gp, min_coverage=min_coverage)
    if parts["composite"] is None:
        return {"weights": pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)}
    held = _select(parts["composite"], max_positions=max_positions, rebalance_days=rebalance_days)
    return {"weights": bt.next_bar(held).astype(float) / max_positions}


def _factors(panel: Panel) -> dict[str, pd.DataFrame]:
    """Form the raw (pre-rank) factor frames from fundamentals + price (pure).

    Market cap is ``price * shares_outstanding``; the ratios are NaN wherever an
    input is missing (an early-history name without a filing yet), which the
    ranking step reads as "not in the cross-section".

    Args:
        panel: Price panel carrying ``panel.features``.

    Returns:
        ``{factor_key: frame}`` for whichever factors have their inputs present.
    """
    feats = panel.features or {}
    raw: dict[str, pd.DataFrame] = {}
    shares = feats.get("fund_shares")
    if shares is not None:
        mktcap = (panel.close * shares).replace(0.0, np.nan)
        if "fund_equity" in feats:
            raw["bp"] = feats["fund_equity"] / mktcap            # book-to-price (value)
        if "fund_ni_ttm" in feats:
            raw["ey"] = feats["fund_ni_ttm"] / mktcap            # earnings yield (value)
    if "fund_gp_ttm" in feats and "fund_assets" in feats:
        raw["gp"] = feats["fund_gp_ttm"] / feats["fund_assets"].replace(0.0, np.nan)  # quality
    return raw


def _compose(panel: Panel, *, w_bp: float, w_ey: float, w_gp: float,
             min_coverage: float) -> dict:
    """Rank each present factor cross-sectionally and blend into one composite (pure).

    A name is *eligible* only if it carries every present factor; ineligible names
    are dropped from the cross-section (their composite is NaN), so they neither
    receive a rank nor push anyone else's percentile around. Present factors are
    those with a positive weight and at least ``min_coverage`` breadth.

    Args:
        panel: Price panel carrying ``panel.features``.
        w_bp: Weight on book-to-price.
        w_ey: Weight on earnings yield.
        w_gp: Weight on gross profitability.
        min_coverage: Coverage floor below which a factor is dropped.

    Returns:
        ``{"raw", "present", "ranks", "composite", "score_pct", "eligible"}``.
        ``composite`` is the weighted mean of the present factors' cross-sectional
        ranks (NaN for ineligible names), or ``None`` when no factor is usable;
        ``score_pct`` is the cross-sectional rank *of* the composite.
    """
    raw = _factors(panel)
    weights = {"bp": w_bp, "ey": w_ey, "gp": w_gp}
    present = [k for k in raw if weights[k] > 0 and _coverage(raw[k]) >= min_coverage]
    base = {"raw": raw, "present": present, "ranks": {}, "composite": None,
            "score_pct": None, "eligible": None}
    if not present:
        return base

    eligible = None
    for k in present:
        ok = raw[k].notna()
        eligible = ok if eligible is None else (eligible & ok)

    wsum = sum(weights[k] for k in present)
    ranks = {k: raw[k].where(eligible).rank(axis=1, pct=True) for k in present}
    composite = sum(weights[k] * ranks[k] for k in present).where(eligible) / wsum
    base.update(ranks=ranks, composite=composite,
                score_pct=composite.rank(axis=1, pct=True), eligible=eligible)
    return base


def _select(composite: pd.DataFrame, *, max_positions: int, rebalance_days: int) -> pd.DataFrame:
    """Hold the top ``max_positions`` names, re-picked every ``rebalance_days`` bars.

    On a rebalance bar the best-ranked eligible names fill the slots; between
    rebalances the holding set is carried forward, so turnover is paced by the
    cadence rather than by every wiggle in the ranking. Ineligible names (NaN
    composite) can never be selected.

    Args:
        composite: The ``(bars x symbols)`` composite rank; NaN where ineligible.
        max_positions: Number of slots.
        rebalance_days: Bars between re-selections.

    Returns:
        A boolean ``(bars x symbols)`` mask of names the book wants to own, on the
        *signal* clock (not yet shifted for execution — :func:`signals` does that).
    """
    ranked = composite.rank(axis=1, ascending=False, method="first")  # 1 = cheapest/best
    target = ranked.le(max_positions).astype(float)                   # top-K -> 1, else 0/NaN->0
    is_rebalance = pd.Series(False, index=composite.index)
    is_rebalance.iloc[::rebalance_days] = True
    # keep the selection only on rebalance bars, then carry it forward between them
    held = target.where(is_rebalance).ffill().fillna(0.0) > 0.5
    return held


def _coverage(frame: pd.DataFrame) -> float:
    """Fraction of the universe this factor covers on its best-covered bar.

    Args:
        frame: A raw (pre-rank) factor frame.

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
        True when at least one composite weight is positive, the book has at least
        one slot, and the rebalance cadence is at least one bar.
    """
    weights = sum(params.get(key, 0.0) for key in _WEIGHT_KEY.values())
    return (weights > 0 and params.get("max_positions", 15) >= 1
            and params.get("rebalance_days", 21) >= 1)


_DEFAULTS = {name: p.default for name, p in inspect.signature(signals).parameters.items()
             if p.default is not inspect.Parameter.empty}


def scores(panel: Panel, **params) -> dict:
    """Expose the factor frames and composite rank for signal diagnostics.

    Shares its whole computation with :func:`signals` (via :func:`_compose`), so
    the notebook's rank-IC / decile-spread checks see exactly the ranking the
    backtest trades. Missing parameters default to :func:`signals`' own signature.

    Args:
        panel: The panel the backtest ran on, with ``panel.features`` attached.
        **params: Any :func:`signals` parameters to override.

    Returns:
        The :func:`_compose` result: ``raw`` (per-factor pre-rank frames),
        ``present``, ``ranks``, ``composite``, ``score_pct``, ``eligible``.
    """
    p = {**_DEFAULTS, **params}
    return _compose(panel, w_bp=p["w_bp"], w_ey=p["w_ey"], w_gp=p["w_gp"],
                    min_coverage=p["min_coverage"])
