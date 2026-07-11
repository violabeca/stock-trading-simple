"""Informed-flow composite, daily: rank stocks by what informed participants do.

Thesis: three groups leave a public footprint that predicts drift over weeks to
months, and combining their independent signals is more robust than any one.

1. **Corporate insiders** — open-market Form 4 *buys* are the strongest
   replicated alt-data signal (Cohen-Malloy-Pomorski 2012; Lakonishok-Lee 2001:
   buys informative, sells not). We use net signed buy dollars, **scaled by
   dollar volume** so a mega-cap's routine print does not dwarf a small-cap's
   conviction buy (the sibling strategy's un-normalized bug).
2. **Sell-side analysts** — rating *changes* drift for months (Womack 1996:
   downgrades −9%/6mo, upgrades +2%); consensus *levels* do not pay net of costs
   (Barber 2001), so we use only the up/down change stream, EMA-smoothed.
3. **Short sellers** — high days-to-cover is bearish, but the short leg's alpha
   mostly accrues to the lender as a borrow fee (Drechsler-Drechsler) and the
   value-weighted effect is weak (Asquith-Pathak-Ritter). So we run **long-only**
   and treat *low* days-to-cover as a favourable tilt, never shorting crowded
   names.

Construction (all in this pure function; the feature layer only supplies daily
aggregates): percentile-rank each present signal cross-sectionally, combine with
tunable weights (equal-ish default — an equal-weighted rank composite resists
overfitting far better than ML stacking at this scale; Blitz; Israel-Kelly-
Moskowitz), re-rank into a composite score, and go long the top quantile with
**TopK-Drop hysteresis**: enter above ``q_in``, hold until the rank falls below a
wider ``q_out`` band, with a ``max_hold`` time-stop backstop. All entries are
next-bar-open executed (``bt.next_bar``), and every feature is already
point-in-time lagged in :mod:`trading_simple.features`, so there is no
look-ahead.

Missing or empty feature (e.g. short interest not yet ingested) is dropped from
the composite and its weight renormalized away — the strategy runs on whatever
signals are present, so it degrades gracefully and needs no engine change to
gain a fourth signal later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel

_WEIGHT_KEY = {"insider": "w_insider", "ratings": "w_rating", "short_interest": "w_dtc"}


def signals(
    panel: Panel,
    *,
    insider_window: int = 63,
    rating_halflife: int = 21,
    w_insider: float = 0.4,
    w_rating: float = 0.3,
    w_dtc: float = 0.3,
    q_in: float = 0.8,
    q_out: float = 0.6,
    max_hold: int = 126,
    min_events: int = 1,
) -> dict:
    """Build long-only entries/exits from the ranked informed-flow composite.

    Args:
        panel: Price panel; ``panel.features`` must carry the alt-data matrices
            (``insider``, ``ratings``, and optionally ``short_interest``) built by
            :func:`trading_simple.features.attach_features`.
        insider_window: Trailing bars over which insider dollars (and the
            dollar-volume normalizer, and the eligibility event count) accumulate.
        rating_halflife: EMA half-life in bars for the rating-change stream.
        w_insider: Composite weight on the insider signal.
        w_rating: Composite weight on the rating-change signal.
        w_dtc: Composite weight on the (negated) days-to-cover signal; ignored
            when short interest is absent.
        q_in: Enter when the composite cross-sectional rank is at/above this
            quantile (e.g. 0.8 = top 20%).
        q_out: Exit when the rank falls below this (wider) quantile; the gap to
            ``q_in`` is the hysteresis band that damps turnover.
        max_hold: Time-stop backstop — exit this many bars after entry regardless.
        min_events: Minimum insider+rating events in the trailing window for a
            name to be eligible (drops never-covered names from the ranking).

    Returns:
        ``{"entries", "exits"}`` boolean frames, next-bar-open executed. Long
        only; no stops (the rank-exit band is the exit thesis).
    """
    feats = panel.features or {}
    none = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)

    raw: dict[str, pd.DataFrame] = {}
    if "insider" in feats:
        dollar_vol = (panel.close * panel.volume).rolling(insider_window, min_periods=1).sum()
        net = feats["insider"].rolling(insider_window, min_periods=1).sum()
        raw["insider"] = net / dollar_vol.replace(0.0, np.nan)
    if "ratings" in feats:
        raw["ratings"] = feats["ratings"].ewm(halflife=rating_halflife, min_periods=1).mean()
    if "short_interest" in feats:
        raw["short_interest"] = -feats["short_interest"]  # low days-to-cover is favourable

    weights = {"insider": w_insider, "ratings": w_rating, "short_interest": w_dtc}
    present = [k for k in raw if weights[k] > 0 and bool(raw[k].notna().any().any())]
    if not present:
        return {"entries": none, "exits": none}

    wsum = sum(weights[k] for k in present)
    composite = sum(weights[k] * raw[k].rank(axis=1, pct=True).fillna(0.5)
                    for k in present) / wsum
    score_pct = composite.rank(axis=1, pct=True)

    eligible = _eligibility(feats, insider_window) >= min_events
    entries = bt.next_bar((score_pct >= q_in) & eligible)
    exits = bt.next_bar(score_pct < q_out) | bt.exit_after(entries, max_hold)
    return {"entries": entries, "exits": exits}


def _eligibility(feats: dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
    """Count trailing insider+rating events per name (for the coverage mask).

    A day counts as an event when the underlying flow is non-zero; NaN warm-up
    cells are treated as no-event. Both present flows contribute; an absent one
    adds nothing.

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
        hysteresis requirement) and at least one composite weight is positive.
    """
    q_in = params.get("q_in", 0.8)
    q_out = params.get("q_out", 0.6)
    weights = params.get("w_insider", 0.4) + params.get("w_rating", 0.3) + params.get("w_dtc", 0.3)
    return q_out < q_in and weights > 0
