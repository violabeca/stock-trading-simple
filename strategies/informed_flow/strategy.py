"""Informed-flow composite, daily: rank stocks by what informed participants do.

Thesis: several groups leave a public footprint that predicts drift over weeks to
months, and combining their independent signals is more robust than any one.

1. **Corporate insiders** — open-market Form 4 *buys* are the strongest
   replicated alt-data signal (Cohen-Malloy-Pomorski 2012; Lakonishok-Lee 2001:
   buys informative, sells not). We use net signed buy dollars, **scaled by
   dollar volume** so a mega-cap's routine print does not dwarf a small-cap's
   conviction buy (the sibling strategy's un-normalized bug).
2. **Sell-side analysts, ratings** — rating *changes* drift for months (Womack
   1996: downgrades −9%/6mo, upgrades +2%); consensus *levels* do not pay net of
   costs (Barber 2001), so we use only the up/down change stream, EMA-smoothed.
3. **Sell-side analysts, price targets** — a *separate* sleeve, not a substitute
   for (2): the large majority of target revisions arrive with the rating
   unchanged, which the rating stream scores as exactly zero. Brav-Lehavy (2003)
   find the drift precisely there (6-month CAR +6.2% top vs −1.9% bottom tercile
   of ΔTP, *conditional on a reiteration*); Asquith-Mikhail-Au (2005) find targets
   subsume earnings forecasts, not the reverse. We rank the **revision**
   (ΔTP/prior TP), **never the target-to-price level** — Han-Kang (2022) show
   price drifts *against* the level, most strongly in large liquid names, i.e.
   exactly this universe, so ranking on the level would likely invert the sign.
4. **Short sellers** — high days-to-cover is bearish, but the short leg's alpha
   mostly accrues to the lender as a borrow fee (Drechsler-Drechsler) and the
   value-weighted effect is weak (Asquith-Pathak-Ritter). So we run **long-only**
   and treat *low* days-to-cover as a favourable tilt, never shorting crowded
   names.

Construction (all in this pure module; the feature layer only supplies daily
aggregates): percentile-rank each present signal cross-sectionally, combine with
tunable weights (equal-ish default — an equal-weighted rank composite resists
overfitting far better than ML stacking at this scale; Blitz; Israel-Kelly-
Moskowitz), re-rank into a composite score, and go long the top names with
**TopK-Drop hysteresis**: enter above ``q_in``, hold until the rank falls below a
wider ``q_out`` band, with a ``max_hold`` time-stop backstop. All entries are
next-bar-open executed (``bt.next_bar``), and every feature is already
point-in-time lagged in :mod:`trading_simple.features`, so there is no
look-ahead.

**This is a capital-constrained portfolio, and that is the whole design.** The
strategy runs in the engine's portfolio mode: it returns a ``weights`` frame, not
entries/exits, so all names share **one** cash account of ``init_cash`` *total*.
There are exactly ``max_positions`` slots, each ``1 / max_positions`` of the book
**at entry** (then left to drift — a held name is bought once and sold whole), and
**a new name can only be bought by selling an existing one**. That constraint is not
incidental — it is what the earlier design got wrong. With a cash
account *per symbol*, a strategy holding 40 of 100 names sat ~30% invested against a
100%-invested benchmark, forfeiting most of the market's return to buy a selection
edge, and the ranking was diluted across four times as many names as the edge could
survive (measured: top decile +5.8%/yr active, top 40% +0.6%/yr). Slots fix both.
``displace`` decides what happens when a better name arrives and the book is full:
wait for a vacancy (low turnover), or evict the weakest holding (chases the top of
the ranking, pays for it in trading costs). ``displace_margin`` is the brake on that:
a challenger must out-rank the weakest holding **by a clear gap**, not by an epsilon.
It exists because the bare comparison turned out to drive essentially all the turnover
and to carry no information — see :func:`_select`.

The one exception to "left to drift" is the ``retrim_k`` concentration cap: a held name
whose book weight reaches ``retrim_k ×`` its slot (double it, by default) is trimmed back
to the slot on the next bar, and the freed cash rotates into the next pick — so a single
lucky pop cannot come to dominate the book. It is risk control, not signal, and a fixed
policy rather than a searched one (see :func:`trading_simple.backtest._retrim_loop`).

**The ranked universe is only the eligible names.** A name with fewer than
``min_events`` insider/rating events in the trailing window is dropped from the
cross-section entirely — not merely blocked from entering. Ranking it and then
refusing to buy it would be the worst of both: it could never be held, yet it
would still occupy rank space and push every other name's percentile around. So
``score_pct`` is NaN for such names, and a *held* name that falls out of
eligibility is exited rather than carried on a score the model no longer computes.

A signal that is missing, or that covers too few names to rank meaningfully (see
:data:`_MIN_COVERAGE`), is dropped from the composite and its weight renormalized
away — so the strategy degrades gracefully and gains a new sleeve without an
engine change.

:func:`explain` exposes every intermediate frame for one symbol, so you can see
*why* a name sits inside or outside the bucket on any given bar. It shares its
whole computation with :func:`signals` (via :func:`_compose` / :func:`_decide`),
so the diagnostic cannot drift from what the backtest actually traded.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel

#: Feature name -> the weight parameter that scales it. The last two are the
#: price-derived market-condition sleeves (trend + RSI); every other one is alt-data.
_WEIGHT_KEY = {"insider": "w_insider", "ratings": "w_rating",
               "price_target": "w_ptrev", "short_interest": "w_dtc",
               "trend": "w_trend", "rsi": "w_rsi"}

#: A sleeve must carry data for at least this fraction of the universe on its
#: best-covered bar to count as live. Without this floor, a barely-ingested table
#: still passes a bare "any data?" test: every uncovered name lands on the neutral
#: ``.fillna(0.5)`` rank while the covered handful gets real percentiles, so the
#: sleeve silently becomes a large, arbitrary tilt on those few names and pure
#: noise for everyone else. (This is not hypothetical — ``short_interest`` shipped
#: covering 3 of 100 names.) Ranking a cross-section needs breadth; below this, the
#: honest move is to drop the sleeve and renormalize.
_MIN_COVERAGE = 0.5


def signals(
    panel: Panel,
    *,
    insider_window: int = 63,
    rating_halflife: int = 21,
    ptrev_halflife: int = 21,
    trend_fast: int = 21,
    trend_slow: int = 126,
    rsi_period: int = 14,
    w_insider: float = 0.4,
    w_rating: float = 0.3,
    w_ptrev: float = 0.3,
    w_dtc: float = 0.3,
    w_trend: float = 0.0,
    w_rsi: float = 0.0,
    q_in: float = 0.8,
    q_out: float = 0.6,
    max_hold: int = 126,
    min_events: int = 1,
    max_positions: int = 10,
    displace: bool = False,
    displace_margin: float = 0.0,
    retrim_k: float | None = 2.0,
) -> dict:
    """Build long-only target weights from the ranked informed-flow composite.

    Args:
        panel: Price panel; ``panel.features`` must carry the alt-data matrices
            (``insider``, ``ratings``, and optionally ``price_target`` and
            ``short_interest``) built by
            :func:`trading_simple.features.attach_features`.
        insider_window: Trailing bars over which insider dollars (and the
            dollar-volume normalizer, and the eligibility event count) accumulate.
        rating_halflife: EMA half-life in bars for the rating-change stream.
        ptrev_halflife: EMA half-life in bars for the price-target revision stream.
        trend_fast: Fast-MA lookback (bars) for the price-derived MA-trend sleeve
            (``SMA(trend_fast) / SMA(trend_slow) - 1``) — a fast-over-slow crossover, so it
            reads the trend's direction/slope rather than a single bar's extension. A
            market-condition tilt scaled by ``w_trend``.
        trend_slow: Slow-MA lookback (bars) for the MA-trend sleeve; must exceed
            ``trend_fast`` (enforced by :func:`valid`).
        rsi_period: Lookback (bars) for the price-derived Wilder-RSI sleeve — persistent
            up- vs down-close pressure. A market-condition tilt scaled by ``w_rsi``.
        w_insider: Composite weight on the insider signal.
        w_rating: Composite weight on the rating-change signal.
        w_ptrev: Composite weight on the price-target revision signal.
        w_dtc: Composite weight on the (negated) days-to-cover signal.
        w_trend: Composite weight on the MA-trend sleeve; ``0`` ablates it. Config ships
            ``1.0`` (on by default); the signature default is ``0.0``.
        w_rsi: Composite weight on the RSI sleeve; ``0`` ablates it. Config ships ``1.0``.
        q_in: A name may be bought when its composite cross-sectional rank is at/above
            this quantile (e.g. 0.8 = top 20%). Qualifying is necessary, not sufficient
            — it still needs a free slot.
        q_out: Sell when the rank falls below this (wider) quantile; the gap to ``q_in``
            is the hysteresis band that damps turnover.
        max_hold: Time-stop backstop — see :func:`_select` for the one case it binds.
        min_events: Minimum insider+rating events in the trailing window for a
            name to be eligible (drops never-covered names from the ranking).
        max_positions: Number of slots in the book. Each holds a fixed
            ``1 / max_positions`` of the capital, so the book is fully invested when all
            slots are filled and holds cash when fewer names qualify.
        displace: What happens when a qualifying name arrives and the book is full.
            ``False`` — it waits for a vacancy. ``True`` — it evicts the weakest current
            holding if it outranks it by ``displace_margin``.
        displace_margin: How much a challenger must out-rank the weakest holding by, in
            composite-rank units, before it may evict it. ``0.0`` reproduces the bare
            "any epsilon wins" comparison; one rank place in a 90-name cross-section is
            ~0.011, so e.g. ``0.05`` means "about five places clear". Ignored when
            ``displace`` is off. See :func:`_select` for why it exists.
        retrim_k: Single-name concentration cap. A held name drifts untouched until its
            book weight reaches ``retrim_k ×`` its ``1 / max_positions`` slot (``2.0`` =
            double), at which point the engine trims it back to the slot and the freed cash
            rotates into the next name the selection picks — so a lucky pop cannot come to
            dominate the book (and flatter a parameter set). ``None`` disables the cap.
            Enforced in :func:`trading_simple.backtest._retrim_loop`; a fixed policy, not a
            search dimension (kept out of the Optuna space on purpose).

    Returns:
        ``{"weights", "retrim_k"}`` — a target-weight frame plus the concentration-cap
        scalar, next-bar-open executed, which puts the run in the engine's **portfolio
        mode**: one shared cash account, so buying is funded by selling. Long only; no
        stops (the rank-exit band is the exit thesis).
    """
    parts = _compose(panel, insider_window=insider_window, rating_halflife=rating_halflife,
                     ptrev_halflife=ptrev_halflife, w_insider=w_insider, w_rating=w_rating,
                     w_ptrev=w_ptrev, w_dtc=w_dtc, min_events=min_events,
                     trend_fast=trend_fast, trend_slow=trend_slow, rsi_period=rsi_period,
                     w_trend=w_trend, w_rsi=w_rsi)
    if not parts["present"]:
        flat = pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
        return {"weights": flat, "retrim_k": retrim_k}

    held = _select(parts, q_in=q_in, q_out=q_out, max_hold=max_hold,
                   max_positions=max_positions, displace=displace,
                   displace_margin=displace_margin)["held"]
    return {"weights": _weights(held, max_positions), "retrim_k": retrim_k}


def _weights(held: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    """Turn the held mask into execution-timed target weights: one fixed slot each.

    Each slot is a **fixed** ``1 / max_positions`` of the book rather than
    ``1 / len(held)``, so a thin selection leaves the remaining capital in cash instead
    of silently concentrating the whole book into the few names that happened to
    qualify. The ``1 / max_positions`` is the target **at entry**: the engine
    (:func:`trading_simple.backtest.rotation_targets`) masks per name, so a held name is
    bought once, **left to drift** while other names rotate around it, and sold whole in
    one order when it leaves the mask — the only mid-hold trim is the optional ``retrim_k``
    concentration cap (:func:`trading_simple.backtest._retrim_loop`), which sells a runaway
    winner back to its slot. ``bt.next_bar`` keeps the execution discipline identical to the
    rest of the repo: decided on bar t from data already lagged a bar, filled at the open of
    t+1.

    Args:
        held: Boolean (bars x symbols) mask of names the book wants to own.
        max_positions: Number of slots.

    Returns:
        Target fraction of the book per symbol, per bar. Rows sum to at most 1.0.
    """
    return bt.next_bar(held).astype(float) / max_positions


def _rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    """Wilder's RSI in [0, 100] with a strict warm-up (NaN until ``period`` bars).

    Momentum as *persistent buying pressure*: the ratio of the average up-move to the
    average down-move over the trailing ``period``, smoothed the same way as
    :func:`trading_simple.backtest.atr` (``ewm(alpha=1/period)``) and mapped onto 0-100.
    Higher = buyers have been in control, so it ranks like every other sleeve (higher is
    favourable) and the 0-100 scale is immaterial once cross-sectionally ranked. A flat
    stretch leaves up = down = 0 -> RS = 0/0 = NaN; that bar lands on the neutral 0.5 fill
    via :func:`_compose`'s ``ranks.fillna(0.5)``, exactly like a missing alt-data cell. An
    only-up stretch gives RS = inf -> RSI 100; only-down gives RS = 0 -> RSI 0.

    Args:
        close: Wide ``(bars x symbols)`` adjusted-close frame.
        period: Trailing bars for the Wilder average (the classic default is 14).

    Returns:
        A ``(bars x symbols)`` RSI frame in [0, 100]; NaN during warm-up and on flat names.
    """
    delta = close.diff()
    avg_up = delta.clip(lower=0.0).ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_down = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_up / avg_down
    return 100.0 - 100.0 / (1.0 + rs)


def _compose(
    panel: Panel,
    *,
    insider_window: int,
    rating_halflife: int,
    ptrev_halflife: int,
    w_insider: float,
    w_rating: float,
    w_ptrev: float,
    w_dtc: float,
    min_events: int,
    trend_fast: int = 21,
    trend_slow: int = 126,
    rsi_period: int = 14,
    w_trend: float = 0.0,
    w_rsi: float = 0.0,
    **_,
) -> dict:
    """Build every intermediate the composite rests on (shared by signals + explain).

    Kept as one function so the diagnostic and the backtest can never disagree:
    :func:`signals` and :func:`explain` both read their numbers from here.

    Args:
        panel: Price panel carrying ``panel.features``.
        insider_window: Trailing bars for insider dollars and the normalizer.
        rating_halflife: EMA half-life for the rating-change stream.
        ptrev_halflife: EMA half-life for the price-target revision stream.
        w_insider: Weight on the insider signal.
        w_rating: Weight on the rating-change signal.
        w_ptrev: Weight on the price-target revision signal.
        w_dtc: Weight on the (negated) days-to-cover signal.
        min_events: Eligibility floor on trailing insider+rating events.
        trend_fast: Fast-MA lookback (bars) for the price-derived MA-trend sleeve.
        trend_slow: Slow-MA lookback (bars) for the MA-trend sleeve (> trend_fast).
        rsi_period: Lookback (bars) for the price-derived Wilder-RSI sleeve.
        w_trend: Weight on the MA-trend sleeve (``0`` ablates it — the default).
        w_rsi: Weight on the RSI sleeve (``0`` ablates it — the default).
        **_: Ignored, so callers can splat a full parameter dict.

    Returns:
        ``{"raw", "ranks", "filled", "contrib", "composite", "score_pct",
        "eligible", "n_events", "n_ranked", "present"}``. ``raw``/``ranks``/
        ``filled``/``contrib`` are ``{signal: frame}`` over the *present* signals
        only; ``filled[k]`` marks cells where an **eligible** name had no data for
        sleeve ``k`` and so received the neutral 0.5 rank;
        ``sum(contrib.values()) == composite``; ``score_pct`` is the
        cross-sectional rank *of* ``composite``, and is **NaN for ineligible
        names** — they are removed from the cross-section, not merely blocked from
        entering. ``present`` is empty when no signal is usable.
    """
    feats = panel.features or {}

    raw: dict[str, pd.DataFrame] = {}
    if "insider" in feats:
        dollar_vol = (panel.close * panel.volume).rolling(insider_window, min_periods=1).sum()
        net = feats["insider"].rolling(insider_window, min_periods=1).sum()
        raw["insider"] = net / dollar_vol.replace(0.0, np.nan)
    if "ratings" in feats:
        raw["ratings"] = feats["ratings"].ewm(halflife=rating_halflife, min_periods=1).mean()
    if "price_target" in feats:
        raw["price_target"] = feats["price_target"].ewm(
            halflife=ptrev_halflife, min_periods=1).mean()
    if "short_interest" in feats:
        raw["short_interest"] = -feats["short_interest"]  # low days-to-cover is favourable

    # Price-derived "market-condition" sleeves. Unlike the alt-data sleeves above they need
    # no feature table — they read panel.close directly — so they are always available and
    # are ablated purely by a 0 weight in the `present` filter below. Higher = more
    # favourable for both (fast MA above slow = a confirmed uptrend / stronger up-close
    # pressure), so, unlike short_interest, neither is negated. They tilt the ranking toward
    # names in a favourable price regime and away from ones drifting sideways or down.
    raw["trend"] = bt.sma(panel.close, trend_fast) / bt.sma(panel.close, trend_slow) - 1.0
    raw["rsi"] = _rsi(panel.close, rsi_period)

    weights = {"insider": w_insider, "ratings": w_rating,
               "price_target": w_ptrev, "short_interest": w_dtc,
               "trend": w_trend, "rsi": w_rsi}
    present = [k for k in raw if weights[k] > 0 and _coverage(raw[k]) >= _MIN_COVERAGE]
    n_events = _eligibility(feats, insider_window)
    eligible = n_events >= min_events
    base = {"raw": raw, "ranks": {}, "filled": {}, "contrib": {}, "n_events": n_events,
            "eligible": eligible, "n_ranked": eligible.sum(axis=1), "present": present}
    if not present:
        return base

    wsum = sum(weights[k] for k in present)
    for k in present:
        # .where(eligible): a name with too little alt-data is dropped from the
        # cross-section outright, so it neither receives a rank nor displaces anyone.
        ranks = raw[k].where(eligible).rank(axis=1, pct=True)
        base["ranks"][k] = ranks
        base["filled"][k] = ranks.isna() & eligible   # ranked, but no data -> neutral 0.5
        base["contrib"][k] = (weights[k] * ranks.fillna(0.5) / wsum).where(eligible)
    base["weights"], base["wsum"] = weights, wsum
    base["composite"] = sum(base["contrib"].values())
    base["score_pct"] = base["composite"].rank(axis=1, pct=True)
    return base


def _score_from_raw(raw: dict, weights: dict, present: list, wsum: float,
                    eligible: pd.DataFrame) -> pd.DataFrame:
    """Rank each raw signal across the eligible cross-section, blend, and re-rank.

    The composite pipeline in one expression, so :func:`_own_vs_peer` can re-run it
    on a counterfactual cross-section and get a number directly comparable to the
    real ``score_pct``. Must apply the same ``eligible`` mask as :func:`_compose`,
    or the counterfactual would be ranked against a different universe.

    Args:
        raw: ``{signal: frame}`` of pre-rank signal values.
        weights: ``{signal: weight}``.
        present: Signals to include.
        wsum: Sum of the present weights (the renormalizer).
        eligible: Boolean mask of names that are in the ranked universe.

    Returns:
        The ``(bars x symbols)`` composite rank in [0, 1]; NaN where ineligible.
    """
    composite = sum(weights[k] * raw[k].where(eligible).rank(axis=1, pct=True).fillna(0.5)
                    / wsum for k in present)
    return composite.where(eligible).rank(axis=1, pct=True)


def _select(parts: dict, *, q_in: float, q_out: float, max_hold: int,
            max_positions: int, displace: bool, displace_margin: float = 0.0) -> dict:
    """Walk the bars and decide which names the book owns, subject to a slot limit.

    The single place the ``q_in``/``q_out`` bands, the time-stop and the capital
    constraint are applied, so :func:`explain` reports exactly the decisions
    :func:`signals` hands the engine. It is a sequential loop and not a vectorized
    expression because it genuinely is stateful: whether a qualifying name can be bought
    today depends on which names are held today, which depends on every bar before it.

    Per bar, in order:

    1. **Reset the clock** on any holding currently ranked at/above ``q_in``. A name back
       in the top quantile has a *fresh* thesis, so an old timer should not throw it out.
       This is what makes ``max_hold`` bind exactly where it is useful — on a name
       **parked in the hysteresis band**, too weak to re-qualify but not weak enough to
       trip ``q_out``, quietly aging while the evidence that bought it decays out of the
       trailing windows. It also prevents a **wash trade**: without the reset, a
       top-ranked name would be evicted by the timer and re-bought on the same bar, at
       the cost of two fees and no change in the book.
    2. **Sell** anything that fell below ``q_out``, lost eligibility (its score is NaN —
       the model has stopped evaluating it, so holding it is holding an unscored
       position), or ran out its ``max_hold`` clock. This is what frees a slot.
    3. **Optionally displace.** With ``displace``, a qualifying challenger that outranks
       the weakest holding **by ``displace_margin``** takes its place. Without it, the
       challenger waits.
    4. **Fill free slots** from the qualifying names, best-ranked first.

    **Why the margin exists.** Steps 1-2 gate a name's rank *level*; step 3 is an
    *ordering* comparison, and nothing in the ``q_in``/``q_out`` band or the ``max_hold``
    clock can see it. Measured on the v16 pick's holdout book, that asymmetry was the
    whole story: **122 of 127 exits were displacements** (5 were ``q_out``/eligibility,
    zero were ``max_hold``), the median challenger won by ~2 rank places out of ~91
    eligible names, and 16% won by a single place. Because ``score_pct`` is a
    *percentile*, a name can lose its slot with its composite bit-identical to the
    previous bar — MRVL was sold whole on 2025-12-15 on a one-place slip with an
    unchanged composite, and re-bought seven bars later. An event study over all 121
    holdout evictions found the challenger's forward return minus the evicted name's to
    be indistinguishable from zero at every horizon (-0.28% at 5 bars, -0.45% at 10,
    +0.04% at 21, +1.84% at 63; win rate ~50%, p >= 0.67) — the swaps were a coin flip
    paying two legs of costs each. The margin is the missing hysteresis on the ordering.

    Args:
        parts: A :func:`_compose` result with a non-empty ``present``.
        q_in: Rank a name must reach to be *eligible to buy* (not to be bought — that
            also needs a slot).
        q_out: Rank a holding must stay at or above to be kept.
        max_hold: Time-stop horizon in bars.
        max_positions: Number of slots.
        displace: Whether a better name may evict the weakest holding.
        displace_margin: Rank gap a challenger must clear the weakest holding by to evict
            it. ``0.0`` is the bare comparison; ignored when ``displace`` is off.

    Returns:
        ``{"held", "qualified", "slot_blocked"}`` boolean frames on the *signal* clock
        (not yet shifted for execution — :func:`_weights` does that). ``slot_blocked``
        marks a name that qualified but got no slot: the direct, visible consequence of
        the capital constraint, and a state that simply did not exist before it.
    """
    score = parts["score_pct"]          # NaN for ineligible names (dropped from the ranking)
    values = score.to_numpy(dtype=float)
    n_bars, n_symbols = values.shape

    held = np.zeros((n_bars, n_symbols), dtype=bool)
    blocked = np.zeros((n_bars, n_symbols), dtype=bool)
    entered = np.full(n_symbols, -1, dtype=int)     # bar the current holding was opened
    current: list[int] = []

    for t in range(n_bars):
        row = values[t]
        # NaN fails every comparison below, which is exactly right: an ineligible name is
        # neither kept nor bought.
        for j in current:
            if row[j] >= q_in:
                entered[j] = t                      # fresh thesis -> the timer restarts
        current = [j for j in current
                   if row[j] >= q_out and t - entered[j] < max_hold]

        # Challengers: qualified, not already held, best-ranked first. NaN sorts last and
        # fails `>= q_in`, so ineligible names never appear here.
        challengers = [j for j in np.argsort(-row, kind="stable")
                       if row[j] >= q_in and j not in current]
        while displace and len(current) >= max_positions and challengers:
            weakest = min(current, key=lambda j: row[j])
            # The margin is what makes this a real preference rather than a rank-noise
            # coin flip: a challenger inside the band is simply not better enough to be
            # worth two legs of costs. At 0.0 this is the bare comparison.
            if row[challengers[0]] <= row[weakest] + displace_margin:
                break                               # no challenger beats the worst holding
            current.remove(weakest)
            entered[weakest] = -1
            taken = challengers.pop(0)
            current.append(taken)
            entered[taken] = t
        while challengers and len(current) < max_positions:
            taken = challengers.pop(0)              # buy into a free slot, best first
            current.append(taken)
            entered[taken] = t

        held[t, current] = True
        blocked[t, challengers] = True              # qualified, but the book was full

    def frame(data: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(data, index=score.index, columns=score.columns)

    return {"held": frame(held), "slot_blocked": frame(blocked),
            "qualified": score >= q_in}


def _coverage(frame: pd.DataFrame) -> float:
    """Fraction of the universe this signal covers on its best-covered bar.

    Args:
        frame: A raw (pre-rank) signal frame.

    Returns:
        The max over bars of the per-bar share of non-NaN cells; 0.0 when empty.
    """
    if frame.empty or not frame.shape[1]:
        return 0.0
    return float(frame.notna().mean(axis=1).max())


def _eligibility(feats: dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
    """Count trailing insider+rating events per name (for the coverage mask).

    A day counts as an event when the underlying flow is non-zero; NaN warm-up
    cells are treated as no-event. Both present flows contribute; an absent one
    adds nothing. Price targets are deliberately *not* counted: they ride the same
    Yahoo feed as ratings, so they would double-count the same coverage.

    Args:
        feats: The panel's feature matrices.
        window: Trailing bars to count over.

    Returns:
        A ``(index x symbol)`` event-count frame (0 where no coverage).
    """
    counts: pd.DataFrame | None = None
    for name in ("insider", "ratings"):
        if name in feats:
            events = (feats[name].abs() > 0).rolling(window, min_periods=1).sum()
            counts = events if counts is None else counts.add(events, fill_value=0.0)
    if counts is None:  # neither present — nothing is eligible
        any_frame = next(iter(feats.values()))
        return pd.DataFrame(0.0, index=any_frame.index, columns=any_frame.columns)
    return counts


def valid(params: dict) -> bool:
    """Reject incoherent combos (tolerates partial dicts from sweep grids).

    Args:
        params: Candidate parameter combo (may hold only the swept keys).

    Returns:
        True when the exit band sits below the entry band (``q_out < q_in``, the
        hysteresis requirement), at least one composite weight is positive, the MA-trend
        crossover is ordered (``trend_fast < trend_slow``) whenever both are given, and the
        book has at least one slot.
    """
    q_in = params.get("q_in", 0.8)
    q_out = params.get("q_out", 0.6)
    weights = sum(params.get(key, 0.0) for key in _WEIGHT_KEY.values())
    fast, slow = params.get("trend_fast"), params.get("trend_slow")
    trend_ok = fast is None or slow is None or fast < slow
    return (q_out < q_in and weights > 0 and trend_ok
            and params.get("max_positions", 10) >= 1)


# ---------------------------------------------------------------------------
# Diagnostics — why is this name in (or out of) the bucket?
# ---------------------------------------------------------------------------
#: Feature name -> the column :func:`explain` reports its raw daily event under.
_EVENT_COL = {"insider": "insider_usd", "ratings": "rating_change",
              "price_target": "ptrev", "short_interest": "days_to_cover"}

_DEFAULTS = {name: p.default for name, p in inspect.signature(signals).parameters.items()
             if p.default is not inspect.Parameter.empty}


def explain(panel: Panel, symbol: str, **params) -> pd.DataFrame:
    """Per-bar audit trail for one symbol: why it sits where it sits in the bucket.

    Answers the question the bare weight frame cannot: *why* did this name cross ``q_in``
    (or fall through ``q_out``), and why was it — or was it not — actually bought? Four
    traps this makes visible, all of which look like bugs until you see them:

    * **A name can cross with no event of its own.** The score is a *cross-sectional*
      rank, so a name rises when its peers fall. The ``own``/``peer`` columns split
      each bar's rank change into exactly those two parts (they sum to ``d_score``
      with no residual), and ``own_event`` says whether anything actually happened
      to this name.
    * **A name's own value moves with no new event.** ``raw["insider"]`` is a
      trailing sum over a rolling window divided by trailing dollar volume, so an
      old buy rolling off the back of the window moves the score by itself. Hence
      ``own`` and ``own_event`` are reported *separately* — "own-driven" and "an
      event happened" are not the same claim.
    * **A missing signal is not a zero, it is a neutral 0.5.** ``filled_<k>`` marks
      bars where this name had no data for sleeve ``k`` and so was handed the median
      rank — it looks scored, but carries no information.
    * **Qualifying is not being bought.** The book has only ``max_positions`` slots, so a
      top-ranked name can sit unbought for weeks because the money is already spent.
      ``slot_blocked`` marks exactly that, and it is the reason "its rank is high, why do
      I not own it?" has an answer other than "a bug".

    Timing, honestly: features are already lagged one bar in
    :mod:`trading_simple.features`, and orders are next-bar-open executed. So an event
    known on day ``D`` appears in ``event_date == D``, lands on the feature bar ``D+1``
    (this frame's index), and can fill no earlier than the open of ``D+2``. ``held`` and
    ``weight`` are on the *signal* clock — what the book wants to own, decided from bar
    ``t``'s data — while ``target_weight`` is the execution-timed frame the engine
    actually receives. For the positions that truly resulted, read
    ``backtest.trade_records``.

    Args:
        panel: The same panel the backtest ran on, with ``panel.features`` attached.
        symbol: The name to explain (a column of ``panel.close``).
        **params: Any :func:`signals` parameters to override; the rest default to
            that function's own signature, so this cannot silently diverge from it.

    Returns:
        A frame indexed by the panel's bars with, per bar: ``composite``,
        ``score_pct``, ``zone`` (``in``/``band``/``out``), ``state`` (``held``,
        ``slot_blocked``, ``ineligible``, or the ``zone``), ``eligible``, ``n_events``,
        ``n_ranked`` (how many names it was ranked against that day), ``n_held`` (how
        many slots the whole book had filled); per present sleeve ``rank_<k>``,
        ``contrib_<k>`` (summing to ``composite``), ``d_contrib_<k>`` (summing to
        ``composite.diff()``) and ``filled_<k>``; the rank-change split ``d_score``,
        ``own``, ``peer``, ``driver`` and ``own_event``; the raw point-in-time events
        (``insider_usd``, ``rating_change``, ``ptrev``, ``days_to_cover``) and their
        ``event_date``; and ``qualified`` / ``held`` / ``slot_blocked`` /
        ``target_weight``.

    Raises:
        KeyError: If ``symbol`` is not in the panel.
        ValueError: If no signal is usable, so there is nothing to explain.
    """
    if symbol not in panel.close.columns:
        raise KeyError(f"{symbol!r} not in panel ({len(panel.symbols)} symbols)")
    p = {**_DEFAULTS, **params}
    parts = _compose(panel, **p)
    if not parts["present"]:
        raise ValueError(
            "no usable signal: every sleeve is absent, zero-weighted, or below the "
            f"{_MIN_COVERAGE:.0%} coverage floor — nothing to explain"
        )

    composite, score = parts["composite"], parts["score_pct"]
    selected = _select(parts, q_in=p["q_in"], q_out=p["q_out"], max_hold=p["max_hold"],
                       max_positions=p["max_positions"], displace=p["displace"],
                       displace_margin=p["displace_margin"])
    target = _weights(selected["held"], p["max_positions"])

    out = pd.DataFrame(index=panel.close.index)
    out["composite"] = composite[symbol]
    out["score_pct"] = score[symbol]
    out["eligible"] = parts["eligible"][symbol]
    out["n_events"] = parts["n_events"][symbol]
    out["n_ranked"] = parts["n_ranked"]        # how many names it is ranked against
    out["n_held"] = selected["held"].sum(axis=1)   # how many of the book's slots are full

    for key in parts["present"]:
        out[f"rank_{key}"] = parts["ranks"][key][symbol]
        out[f"contrib_{key}"] = parts["contrib"][key][symbol]
        out[f"d_contrib_{key}"] = parts["contrib"][key][symbol].diff()
        out[f"filled_{key}"] = parts["filled"][key][symbol]

    own, peer = _own_vs_peer(parts, symbol)
    out["d_score"] = score[symbol].diff()
    out["own"], out["peer"] = own, peer
    out["driver"] = np.where(own.abs() >= peer.abs(), "own", "peer")

    feats = panel.features or {}
    event = pd.Series(False, index=out.index)
    for key, col in _EVENT_COL.items():
        if key in feats:
            out[col] = feats[key][symbol]
            if key != "short_interest":  # a level, not an event flow
                event |= feats[key][symbol].abs() > 0
    out["own_event"] = event
    out["event_date"] = out.index.to_series().shift(1)  # features are lagged one bar

    out["zone"] = np.select(
        [out["score_pct"] >= p["q_in"], out["score_pct"] >= p["q_out"]],
        ["in", "band"], default="out")
    out["qualified"] = selected["qualified"][symbol]
    out["held"] = selected["held"][symbol]                  # what the book wants to own
    out["slot_blocked"] = selected["slot_blocked"][symbol]  # qualified, but no slot free
    out["target_weight"] = target[symbol]                   # what the engine receives
    # Most specific first: "held" and "slot_blocked" outrank the raw zone, because the
    # whole point is that a top-quantile rank does not by itself get you into the book.
    out["state"] = np.select(
        [~out["eligible"], out["held"], out["slot_blocked"]],
        ["ineligible", "held", "slot_blocked"], default=out["zone"])
    return out


def _own_vs_peer(parts: dict, symbol: str) -> tuple:
    """Split each bar's rank change into the part this name caused and the part its peers did.

    The score is a cross-sectional rank, so it moves for two quite different
    reasons: this name's own inputs changed, or everyone else's did. To separate
    them, rebuild the score on a counterfactual cross-section in which **this name's
    raw signals are held at yesterday's values** while every peer keeps today's::

        own  = score[t]    - score_cf     # own inputs moved, peers held at today's
        peer = score_cf    - score[t-1]   # peers moved, own inputs held at yesterday's

    which telescopes to ``score[t] - score[t-1]`` with **zero residual**.

    The swap must happen on the **raw** frames, not on the composite: the composite
    is built from cross-sectional *ranks*, which are already relative, so a peer's
    move changes this name's rank — and hence its composite — all by itself.
    Decomposing at the composite level would charge that to ``own`` and understate
    peer drift by roughly half.

    Args:
        parts: A :func:`_compose` result with a non-empty ``present``.
        symbol: The name being explained.

    Returns:
        ``(own, peer)`` Series; ``own + peer == score_pct[symbol].diff()`` exactly.
    """
    raw, present = parts["raw"], parts["present"]
    frozen = {k: raw[k].assign(**{symbol: raw[k][symbol].shift(1)}) for k in present}
    score_cf = _score_from_raw(frozen, parts["weights"], present, parts["wsum"],
                               parts["eligible"])[symbol]
    score = parts["score_pct"][symbol]
    return score - score_cf, score_cf - score.shift(1)
