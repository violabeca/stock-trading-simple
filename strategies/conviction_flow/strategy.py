"""Conviction-flow: informed_flow, made selective — and gated on fundamentals.

A copy of :mod:`strategies.informed_flow` with the *same* alt-data composite (insider
buying + rating changes + price-target revisions + short-interest avoidance) and the
*same* TopK-Drop hysteresis selector, but built to answer a specific problem the parent
strategy exposes: on a liquid universe **many more names qualify (rank ≥ ``q_in``) than
the book has slots**, so most are turned away (``slot_blocked``). Simply raising ``q_in``
narrows on the *same* axis (composite rank). This strategy narrows on **new, orthogonal
axes**, and rotates the book faster into genuinely better names.

Five levers, layered on top of informed_flow's ``q_in`` / ``q_out`` / ``max_hold``.
**Every one defaults to OFF**, so at default parameters this strategy reproduces
informed_flow exactly (a pinned test asserts it); the optimizer then discovers which
levers earn their keep rather than being forced to use them.

* **Fundamentals quality gate** (``min_quality_pct``). A name may open only if it also
  clears a cross-sectional **fundamentals-quality** floor — a veto *on top of* the flow
  rank, not another blended score. Quality is an equal-weight rank composite of three
  point-in-time SEC-XBRL factors, ported from :mod:`strategies.value_factor`:
  **book-to-price** (``fund_equity / mktcap``), **earnings yield**
  (``fund_ni_ttm / mktcap``) and **gross profitability** (``fund_gp_ttm / fund_assets``),
  with ``mktcap = close * fund_shares``. So a hot-flow name that is fundamentally weak is
  never bought, however loud the flow. If the fundamentals table is too thin to rank
  (below :data:`_MIN_COVERAGE`), the gate self-disables and the strategy degrades to
  informed_flow, the same way a missing alt-data sleeve does.
* **Signal-agreement gate** (``min_agree``). Require at least this many *individual* flow
  sleeves to each rank the name in their own top (``q_in``) quantile — a real rank, not a
  neutral-filled 0.5. A high *composite* rank can come from being mildly good everywhere;
  agreement demands independent conviction, which replicates better (Blitz;
  Israel-Kelly-Moskowitz). ``0`` is the true no-op (a composite rank ≥ ``q_in`` needs no
  single sleeve to be top-quantile).
* **Higher conviction bar** (``min_score``). Require the composite score *level* (the
  blended rank, in [0, 1]) — not merely its cross-sectional rank — to clear a floor. In a
  weak field where even the top-ranked names are mediocre, the book holds cash instead.
* **Margin displacement** (``displace`` + ``displace_margin``). When the book is full and
  ``displace`` is on, evict the weakest holding for a challenger only if the challenger
  outranks it by at least ``displace_margin``. This is the "close faster when another name
  looks *much* more attractive" lever: bigger margin ⇒ rotate only on a decisive edge
  (less churn), zero margin ⇒ informed_flow's strict any-epsilon beat.
* **Rank-decay exit** (``exit_decay``). Sell a holding whose ``score_pct`` has fallen by
  at least this much from its **peak while held** (the running max of its rank since it
  was bought), before it drifts all the way down to ``q_out`` — catching a fading thesis
  early. ``1.0`` is unreachable, so the default is off.

Everything else — the composite math, per-sleeve eligibility, ``_MIN_COVERAGE`` graceful
degradation, portfolio-mode ``weights`` output (one shared cash account, ``max_positions``
slots of ``1/max_positions`` each, bought once and sold whole, never re-trimmed), and
next-bar-open execution — is inherited unchanged from informed_flow. :func:`explain`
exposes the per-bar audit trail, including the new gate columns, and shares its whole
computation with :func:`signals` so it cannot drift from what the backtest traded.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel

#: Feature name -> the weight parameter that scales it (the flow sleeves).
_WEIGHT_KEY = {"insider": "w_insider", "ratings": "w_rating",
               "price_target": "w_ptrev", "short_interest": "w_dtc"}

#: A sleeve (flow or fundamentals) must carry data for at least this fraction of the
#: universe on its best-covered bar to count as live. Below it the honest move is to drop
#: the sleeve and renormalize rather than let a handful of covered names become an
#: arbitrary tilt while everyone else lands on the neutral 0.5. (``short_interest`` once
#: shipped covering 3 of 100 names.) Shared by the flow composite and the quality gate.
_MIN_COVERAGE = 0.5


def signals(
    panel: Panel,
    *,
    insider_window: int = 63,
    rating_halflife: int = 21,
    ptrev_halflife: int = 21,
    w_insider: float = 0.4,
    w_rating: float = 0.3,
    w_ptrev: float = 0.3,
    w_dtc: float = 0.3,
    q_in: float = 0.8,
    q_out: float = 0.6,
    max_hold: int = 126,
    min_events: int = 1,
    max_positions: int = 10,
    displace: bool = False,
    min_quality_pct: float = 0.0,
    min_agree: int = 0,
    min_score: float = 0.0,
    displace_margin: float = 0.0,
    exit_decay: float = 1.0,
) -> dict:
    """Build long-only target weights from the ranked, gated informed-flow composite.

    Args:
        panel: Price panel; ``panel.features`` must carry the alt-data matrices
            (``insider``, ``ratings``, and optionally ``price_target`` and
            ``short_interest``); for the quality gate it should also carry the
            fundamentals matrices (``fund_shares``, ``fund_equity``, ``fund_ni_ttm``,
            ``fund_gp_ttm``, ``fund_assets``). All are built by
            :func:`trading_simple.features.attach_features`.
        insider_window: Trailing bars over which insider dollars (and the
            dollar-volume normalizer, and the eligibility event count) accumulate.
        rating_halflife: EMA half-life in bars for the rating-change stream.
        ptrev_halflife: EMA half-life in bars for the price-target revision stream.
        w_insider: Composite weight on the insider signal.
        w_rating: Composite weight on the rating-change signal.
        w_ptrev: Composite weight on the price-target revision signal.
        w_dtc: Composite weight on the (negated) days-to-cover signal.
        q_in: A name may be bought when its composite cross-sectional rank is at/above
            this quantile. Necessary, not sufficient — it also needs a free slot and must
            clear every enabled open-gate below.
        q_out: Sell when the rank falls below this (wider) quantile; the gap to ``q_in``
            is the hysteresis band that damps turnover.
        max_hold: Time-stop backstop — see :func:`_select` for the one case it binds.
        min_events: Minimum insider+rating events in the trailing window for a
            name to be eligible (drops never-covered names from the ranking).
        max_positions: Number of slots in the book, each a fixed ``1/max_positions`` of
            the capital at entry.
        displace: Whether a qualifying challenger may evict the weakest holding when the
            book is full (subject to ``displace_margin``); ``False`` makes it wait.
        min_quality_pct: **Fundamentals quality gate.** A challenger opens only if its
            cross-sectional fundamentals-quality rank is at/above this (``0`` = off; a
            name with no fundamentals data fails a positive floor). Self-disables if the
            fundamentals table is below :data:`_MIN_COVERAGE`.
        min_agree: **Signal-agreement gate.** A challenger opens only if at least this
            many individual flow sleeves each rank it (with a real, non-filled rank)
            at/above ``q_in`` (``0`` = off).
        min_score: **Conviction bar.** A challenger opens only if its composite score
            *level* (in [0, 1]) is at/above this (``0`` = off).
        displace_margin: **Displacement margin.** With ``displace``, evict the weakest
            holding only if the challenger's rank exceeds it by at least this (``0`` =
            informed_flow's strict any-epsilon beat).
        exit_decay: **Rank-decay exit.** Sell a holding whose ``score_pct`` has fallen at
            least this far from its peak-while-held (``1.0`` = off, unreachable).

    Returns:
        ``{"weights"}`` — a target-weight frame, next-bar-open executed, which puts the
        run in the engine's **portfolio mode**: one shared cash account, so buying is
        funded by selling. Long only; no stops (the rank/decay exits are the exit thesis).
    """
    parts = _compose(panel, insider_window=insider_window, rating_halflife=rating_halflife,
                     ptrev_halflife=ptrev_halflife, w_insider=w_insider, w_rating=w_rating,
                     w_ptrev=w_ptrev, w_dtc=w_dtc, min_events=min_events)
    if not parts["present"]:
        flat = pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
        return {"weights": flat}

    held = _select(parts, q_in=q_in, q_out=q_out, max_hold=max_hold,
                   max_positions=max_positions, displace=displace,
                   min_quality_pct=min_quality_pct, min_agree=min_agree,
                   min_score=min_score, displace_margin=displace_margin,
                   exit_decay=exit_decay)["held"]
    return {"weights": _weights(held, max_positions)}


def _weights(held: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    """Turn the held mask into execution-timed target weights: one fixed slot each.

    Each slot is a **fixed** ``1 / max_positions`` of the book rather than
    ``1 / len(held)``, so a thin selection leaves the remaining capital in cash instead
    of silently concentrating the whole book into the few names that happened to qualify.
    The ``1 / max_positions`` is the target **at entry**: the engine
    (:func:`trading_simple.backtest.rotation_targets`) masks per name, so a held name is
    bought once, **left to drift**, and sold whole in one order when it leaves the mask —
    never re-trimmed mid-hold. ``bt.next_bar`` keeps execution decided on bar t from data
    already lagged a bar, filled at the open of t+1.

    Args:
        held: Boolean (bars x symbols) mask of names the book wants to own.
        max_positions: Number of slots.

    Returns:
        Target fraction of the book per symbol, per bar. Rows sum to at most 1.0.
    """
    return bt.next_bar(held).astype(float) / max_positions


def _quality(panel: Panel) -> tuple[pd.DataFrame | None, list[str]]:
    """Cross-sectional fundamentals-quality rank per name (the gate's raw material).

    Ported from :mod:`strategies.value_factor`: three point-in-time SEC-XBRL factors —
    book-to-price and earnings yield (value) and gross profitability (quality) — each
    percentile-ranked across the names that carry it, blended equal-weight, and re-ranked.
    A name missing any present factor is dropped from the quality cross-section (its rank
    is NaN), exactly as it is dropped from the flow cross-section when it lacks alt-data.

    Args:
        panel: Price panel carrying ``panel.features`` (needs ``fund_*`` matrices).

    Returns:
        ``(quality_pct, present)`` — the ``(bars x symbols)`` quality rank in (0, 1]
        (NaN where a name lacks the factors), or ``(None, [])`` when no factor clears the
        :data:`_MIN_COVERAGE` breadth floor and the gate must self-disable.
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
    present = [k for k in raw if _coverage(raw[k]) >= _MIN_COVERAGE]
    if not present:
        return None, present

    eligible = None
    for k in present:
        ok = raw[k].notna()
        eligible = ok if eligible is None else (eligible & ok)
    ranks = {k: raw[k].where(eligible).rank(axis=1, pct=True) for k in present}
    composite = sum(ranks[k] for k in present).where(eligible) / len(present)
    return composite.rank(axis=1, pct=True), present


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
    **_,
) -> dict:
    """Build every intermediate the composite + gates rest on (shared by signals + explain).

    Kept as one function so the diagnostic and the backtest can never disagree:
    :func:`signals` and :func:`explain` both read their numbers from here. Identical to
    informed_flow's ``_compose`` plus the fundamentals-quality frame for the gate.

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
        **_: Ignored, so callers can splat a full parameter dict.

    Returns:
        informed_flow's ``{"raw", "ranks", "filled", "contrib", "composite", "score_pct",
        "eligible", "n_events", "n_ranked", "present"}`` plus ``"quality_pct"`` (the
        fundamentals-quality rank, or ``None`` when the gate self-disables) and
        ``"quality_present"`` (the fundamentals factors that were live).
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

    weights = {"insider": w_insider, "ratings": w_rating,
               "price_target": w_ptrev, "short_interest": w_dtc}
    present = [k for k in raw if weights[k] > 0 and _coverage(raw[k]) >= _MIN_COVERAGE]
    n_events = _eligibility(feats, insider_window)
    eligible = n_events >= min_events
    quality_pct, quality_present = _quality(panel)
    base = {"raw": raw, "ranks": {}, "filled": {}, "contrib": {}, "n_events": n_events,
            "eligible": eligible, "n_ranked": eligible.sum(axis=1), "present": present,
            "quality_pct": quality_pct, "quality_present": quality_present}
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

    The composite pipeline in one expression, so :func:`_own_vs_peer` can re-run it on a
    counterfactual cross-section and get a number directly comparable to the real
    ``score_pct``. Must apply the same ``eligible`` mask as :func:`_compose`.

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


def _agreement(parts: dict, q_in: float) -> np.ndarray:
    """Per-bar count of flow sleeves that independently rank a name at/above ``q_in``.

    Only a **real** top-quantile rank counts — a neutral-filled 0.5 (``filled[k]``) does
    not, since it carries no information. This is the raw material of the signal-agreement
    gate: a name with a high blended composite but agreement from only one sleeve is
    riding a single signal, not a consensus.

    Args:
        parts: A :func:`_compose` result with a non-empty ``present``.
        q_in: The per-sleeve top-quantile threshold (reuses the entry quantile).

    Returns:
        A ``(bars x symbols)`` integer array; 0 everywhere when no sleeve is present.
    """
    agree = None
    for k in parts["present"]:
        hit = ((parts["ranks"][k] >= q_in) & ~parts["filled"][k]).astype(int)
        agree = hit if agree is None else agree + hit
    score = parts["score_pct"]
    if agree is None:
        return np.zeros(score.shape, dtype=int)
    return agree.to_numpy()


def _select(parts: dict, *, q_in: float, q_out: float, max_hold: int, max_positions: int,
            displace: bool, min_quality_pct: float = 0.0, min_agree: int = 0,
            min_score: float = 0.0, displace_margin: float = 0.0,
            exit_decay: float = 1.0) -> dict:
    """Walk the bars and decide which names the book owns, subject to slots + open-gates.

    informed_flow's stateful slot loop, extended with the five levers. Per bar, in order:

    1. **Track the peak; restart the clock on re-qualification.** For each holding, keep
       the running-max ``score_pct`` seen since it was bought; a name back at/above
       ``q_in`` has a fresh thesis, so its ``max_hold`` clock restarts (the peak is a
       running max and is never lowered).
    2. **Sell** anything below ``q_out``, out of eligibility (NaN score), out of
       ``max_hold``, **or** faded ``exit_decay`` from its peak. This frees a slot.
    3. **Qualify challengers** — rank ≥ ``q_in``, not held, and clearing every enabled
       open-gate: the fundamentals-quality floor (``min_quality_pct``), the
       signal-agreement floor (``min_agree``), and the composite-level floor
       (``min_score``). Best-ranked first.
    4. **Optionally displace** the weakest holding, but only by a challenger that beats it
       by at least ``displace_margin``.
    5. **Fill free slots** from the surviving challengers, best-ranked first.

    Args:
        parts: A :func:`_compose` result with a non-empty ``present``.
        q_in: Rank a name must reach to be *eligible to buy*.
        q_out: Rank a holding must stay at or above to be kept.
        max_hold: Time-stop horizon in bars.
        max_positions: Number of slots.
        displace: Whether a better name may evict the weakest holding.
        min_quality_pct: Fundamentals-quality floor for opening (0 = off; self-disabled
            when ``parts["quality_pct"]`` is None).
        min_agree: Minimum agreeing flow sleeves for opening (0 = off).
        min_score: Minimum composite score *level* for opening (0 = off).
        displace_margin: Rank margin a challenger must beat the weakest holding by.
        exit_decay: Sell a holding faded at least this far below its peak-while-held
            (1.0 = off).

    Returns:
        ``{"held", "qualified", "slot_blocked"}`` boolean frames on the *signal* clock.
        ``qualified`` is rank-only (``score_pct >= q_in``); ``slot_blocked`` marks names
        that cleared every open-gate but got no slot.
    """
    score = parts["score_pct"]          # NaN for ineligible names (dropped from the ranking)
    values = score.to_numpy(dtype=float)
    composite = parts["composite"].to_numpy(dtype=float)
    n_bars, n_symbols = values.shape

    nagree = _agreement(parts, q_in)
    # Quality gate is live only when fundamentals are broad enough AND a floor is asked
    # for; otherwise it self-disables (a NaN quality must not silently block every name).
    quality_on = parts["quality_pct"] is not None and min_quality_pct > 0.0
    quality = (parts["quality_pct"].to_numpy(dtype=float) if quality_on
               else np.ones((n_bars, n_symbols)))

    def opens(t: int, j: int) -> bool:
        """Does name j clear every enabled open-gate on bar t? (rank ≥ q_in checked upstream)."""
        return (nagree[t, j] >= min_agree
                and composite[t, j] >= min_score
                and (not quality_on or quality[t, j] >= min_quality_pct))

    held = np.zeros((n_bars, n_symbols), dtype=bool)
    blocked = np.zeros((n_bars, n_symbols), dtype=bool)
    entered = np.full(n_symbols, -1, dtype=int)     # bar the current holding was opened
    peak = np.full(n_symbols, -np.inf)              # best score_pct seen while held
    current: list[int] = []

    for t in range(n_bars):
        row = values[t]
        for j in current:
            peak[j] = max(peak[j], row[j])          # running max of the rank while held
            if row[j] >= q_in:
                entered[j] = t                      # fresh thesis -> restart the clock
        # Keep a holding only if it holds its rank band, is inside max_hold, and has not
        # faded exit_decay from its peak. NaN fails every comparison -> ineligible sold.
        current = [j for j in current
                   if row[j] >= q_out and t - entered[j] < max_hold
                   and peak[j] - row[j] < exit_decay]

        # Challengers: qualified (rank + every open-gate), not held, best-ranked first.
        challengers = [j for j in np.argsort(-row, kind="stable")
                       if row[j] >= q_in and j not in current and opens(t, j)]
        while displace and len(current) >= max_positions and challengers:
            weakest = min(current, key=lambda j: row[j])
            if row[challengers[0]] <= row[weakest] + displace_margin:
                break                               # not decisively better -> no eviction
            current.remove(weakest)
            entered[weakest] = -1
            taken = challengers.pop(0)
            current.append(taken)
            entered[taken], peak[taken] = t, row[taken]
        while challengers and len(current) < max_positions:
            taken = challengers.pop(0)              # buy into a free slot, best first
            current.append(taken)
            entered[taken], peak[taken] = t, row[taken]

        held[t, current] = True
        blocked[t, challengers] = True              # cleared the gates, but the book was full

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
    """Count trailing insider+rating events per name (for the flow coverage mask).

    A day counts as an event when the underlying flow is non-zero; NaN warm-up cells are
    treated as no-event. Price targets are deliberately not counted (they ride the same
    Yahoo feed as ratings, so they would double-count the same coverage).

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
        True when informed_flow's core holds (``q_out < q_in``, at least one positive
        composite weight, ``max_positions >= 1``) and every new lever is in range:
        ``0 <= min_quality_pct < 1``, ``min_agree >= 0``, ``0 <= min_score < 1``,
        ``displace_margin >= 0``, ``0 < exit_decay <= 1``.
    """
    q_in = params.get("q_in", 0.8)
    q_out = params.get("q_out", 0.6)
    weights = sum(params.get(key, 0.0) for key in _WEIGHT_KEY.values())
    mq = params.get("min_quality_pct", 0.0)
    ms = params.get("min_score", 0.0)
    ed = params.get("exit_decay", 1.0)
    return (q_out < q_in and weights > 0 and params.get("max_positions", 10) >= 1
            and 0.0 <= mq < 1.0 and params.get("min_agree", 0) >= 0 and 0.0 <= ms < 1.0
            and params.get("displace_margin", 0.0) >= 0.0 and 0.0 < ed <= 1.0)


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

    Identical to informed_flow's ``explain`` plus the new open-gate columns, so "its rank
    is high, why is it not held?" now also has the answers "its fundamentals quality is
    too low", "too few sleeves agree", or "the field is too weak". Shares its whole
    computation with :func:`signals` (via :func:`_compose` / :func:`_select`), so it
    cannot drift from what the backtest traded.

    Args:
        panel: The same panel the backtest ran on, with ``panel.features`` attached.
        symbol: The name to explain (a column of ``panel.close``).
        **params: Any :func:`signals` parameters to override; the rest default to that
            function's own signature, so this cannot silently diverge from it.

    Returns:
        A frame indexed by the panel's bars carrying, per bar: ``composite``,
        ``score_pct``, ``quality_pct``, ``n_agree``, ``zone``, ``state``, ``eligible``,
        ``n_events``, ``n_ranked``, ``n_held``; per present sleeve ``rank_<k>``,
        ``contrib_<k>``, ``d_contrib_<k>``, ``filled_<k>``; the rank-change split
        ``d_score``/``own``/``peer``/``driver``/``own_event``; the raw point-in-time
        events and ``event_date``; and ``qualified``/``held``/``slot_blocked``/
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
                       min_quality_pct=p["min_quality_pct"], min_agree=p["min_agree"],
                       min_score=p["min_score"], displace_margin=p["displace_margin"],
                       exit_decay=p["exit_decay"])
    target = _weights(selected["held"], p["max_positions"])

    out = pd.DataFrame(index=panel.close.index)
    out["composite"] = composite[symbol]
    out["score_pct"] = score[symbol]
    out["quality_pct"] = (parts["quality_pct"][symbol] if parts["quality_pct"] is not None
                          else np.nan)
    out["n_agree"] = _agreement(parts, p["q_in"])[:, panel.close.columns.get_loc(symbol)]
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
    out["slot_blocked"] = selected["slot_blocked"][symbol]  # cleared gates, but no slot free
    out["target_weight"] = target[symbol]                   # what the engine receives
    out["state"] = np.select(
        [~out["eligible"], out["held"], out["slot_blocked"]],
        ["ineligible", "held", "slot_blocked"], default=out["zone"])
    return out


def _own_vs_peer(parts: dict, symbol: str) -> tuple:
    """Split each bar's rank change into the part this name caused and the part its peers did.

    The score is a cross-sectional rank, so it moves for two reasons: this name's own
    inputs changed, or everyone else's did. Rebuild the score on a counterfactual
    cross-section in which this name's raw signals are held at yesterday's values while
    every peer keeps today's::

        own  = score[t]    - score_cf     # own inputs moved, peers held at today's
        peer = score_cf    - score[t-1]   # peers moved, own inputs held at yesterday's

    which telescopes to ``score[t] - score[t-1]`` with zero residual. The swap must happen
    on the raw frames, not the composite (built from already-relative ranks).

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
