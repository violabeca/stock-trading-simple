"""informed_flow_ls — informed_flow as a long/short book: short sellers and option traders.

The long-only ``informed_flow`` (v41) owned the names short sellers avoid. This version also
SELLS SHORT the names they crowd into, so the book earns on the spread between the two ends of
the ranking and makes money in a falling market when its shorts fall faster than its longs.
Two signals, each a cross-sectional rank:

1. **Short interest** (FINRA days-to-cover, averaged over ``dtc_window`` bars): low is good
   (long), high is bad (short) — Hong, Li, Wang & Zhao; Boehmer, Jones & Zhang.
2. **The change in put skew** over ``skew_window`` bars (the IV of the 95% put minus the ATM
   call's, ``opt_skew``, first averaged over ``skew_smooth`` bars): puts getting dearer is bad
   (short), cheaper is good (long) — informed traders buy downside protection before bad news
   (An, Ang, Bali & Cakici 2014). Option chains exist only from October 2025: before that this
   sleeve is silent (every name neutral) and the book is short interest alone.

The composite (the weighted average of the two ranks) is averaged over ``score_smooth`` bars
before it is ranked. Both averages exist because the raw daily skew change made the book cross
its bands day after day: in 2026 fees took 10% of the book (RESEARCH_LOG.md §1).

**Why the old version did not short, and what changed.** The short leg's alpha largely goes to
the stock lender as a borrow fee (Drechsler & Drechsler), and crowded names squeeze. So:

* every short pays a borrow fee (the engine's ``borrow`` key, charged per day held), set from
  how crowded the name is: short interest as a share of its shares outstanding (``si_mid`` /
  ``si_high`` tiers: ``borrow_gc``, ``borrow_mid``, ``borrow_high`` a year);
* a name with more than ``short_max_si`` of its shares sold short is never shorted (hard to
  borrow, squeeze risk), and a held short is bought back when it gets there;
* a short that rises ``short_stop`` above its entry close is bought back (the squeeze stop);
* the book is dollar-neutral (``long_share`` 0.5 of ``gross`` long, the rest short; the engine
  locks short proceeds, so a long/short book can never exceed 100% gross).

**Eligibility.** A name is ranked only where it has a price of at least ``min_price``, a days-to-
cover, and a share count (``fund_shares``) — SEC filers: an interlisted foreign name's FINRA
short interest covers only its US line and looks falsely low (v37) — and, when the panel
carries a ``member`` feature (the research notebooks' point-in-time universe), only on the dates
it was a member. Ranks are taken inside ``neutral_groups`` dollar-volume groups (days-to-cover
has volume in its denominator: v31's liquidity lean).

**The book** (pure, stateful, bar by bar in :func:`_select`): a name enters the long side when
its composite rank reaches ``q_in`` and leaves when it falls below ``q_out``; the short side is
the mirror image (enter at ``1 - q_in``, leave above ``1 - q_out``). ``long_slots`` longs and
``short_slots`` shorts, each ``gross × side share / its side's slots`` of the book, held at most
``max_hold`` bars unless it still ranks in the entry zone. Shorts pay the borrow tiers below, or
the broker's own rate where the panel carries one (a ``borrow_fee`` feature, yearly). Signals
use bar t's close; trades fill at t+1's open (``bt.next_bar``).
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd

from trading_simple import backtest as bt

#: A sleeve must rank at least this share of the eligible names on a bar, or it is silent
#: there (every name 0.5): a thin table must not tilt the few names it still covers.
_MIN_COVERAGE = 0.5
#: Sleeve name -> its weight parameter.
WEIGHT_OF = {"si": "w_si", "skew": "w_skew"}


def signals(panel, *, dtc_window: int = 42, skew_window: int = 20, w_si: float = 1.0,
            w_skew: float = 1.0, skew_smooth: int = 1, score_smooth: int = 1,
            neutral_groups: int = 5, q_in: float = 0.9, q_out: float = 0.8, long_slots: int = 15,
            short_slots: int = 15, max_hold: int = 126, gross: float = 0.98,
            long_share: float = 0.5,
            sides: str = "both", short_stop: float = 1.0, short_max_si: float = 0.3,
            si_mid: float = 0.1, si_high: float = 0.2, borrow_gc: float = 0.0075,
            borrow_mid: float = 0.01, borrow_high: float = 0.03, min_price: float = 5.0,
            retrim_k: float = 2.0, si_plausible_max: float = 1.0, **_) -> dict:
    """Target weights (long > 0, short < 0), the borrow rate of every name, and the long cap.

    Args:
        panel: Daily panel with ``features`` ``short_interest`` (days-to-cover) and
            ``fund_shares``; ``opt_skew`` optional (no options: short interest alone);
            ``member`` optional (1 where a name belongs to the universe on that date).
        dtc_window: Bars days-to-cover is averaged over.
        skew_window: Bars over which the put skew's change is measured.
        w_si, w_skew: Sleeve weights in the composite (relative).
        skew_smooth: Bars the put skew is averaged over before its change is taken (1 = the
            raw daily level).
        score_smooth: Bars the composite is averaged over before it is ranked (1 = off).
        neutral_groups: Dollar-volume groups the sleeves are ranked within (< 2: one group).
        q_in, q_out: Long entry / exit ranks; the short side mirrors them (``1 - q``).
        long_slots, short_slots: Positions on the long and on the short side.
        max_hold: Bars a position is held unless it still ranks in its entry zone.
        gross: Long plus short exposure as a share of the book (<= 1: shorts lock cash).
        long_share: The long side's share of ``gross`` (0.5 = dollar-neutral).
        sides: ``"both"``, ``"long"`` (the long side alone) or ``"short"``.
        short_stop: A short whose close rises this far above its entry close is bought back.
        short_max_si: Short interest / shares outstanding above which a name is not shorted.
        si_mid, si_high: Crowding tiers for the borrow rate.
        borrow_gc, borrow_mid, borrow_high: Yearly borrow rates below ``si_mid``, up to
            ``si_high``, and above it (a ``borrow_fee`` feature overrides them where present).
        min_price: Minimum close for a name to be ranked.
        retrim_k: Long positions are trimmed back to their slot at ``k ×`` it (engine cap).
        si_plausible_max: A short-interest share above this is read as unknown: not shortable,
            the cheapest tier. After a k:1 split FINRA's shares short is post-split while the
            SEC count stays pre-split until the next filing, so the ratio reads k times too high;
            the cap catches that only when k x the true ratio exceeds it (a causal error, not a
            look-ahead; RESEARCH_LOG §44).

    Returns:
        ``{"weights", "borrow", "retrim_k"}`` for the engine's portfolio mode.
    """
    parts = compose(panel, dtc_window=dtc_window, skew_window=skew_window, w_si=w_si,
                    w_skew=w_skew, skew_smooth=skew_smooth, score_smooth=score_smooth,
                    neutral_groups=neutral_groups, short_max_si=short_max_si, si_mid=si_mid,
                    si_high=si_high, borrow_gc=borrow_gc, borrow_mid=borrow_mid,
                    borrow_high=borrow_high, min_price=min_price,
                    si_plausible_max=si_plausible_max)
    longs, shorts, _ = _select(parts["score"], panel.close, parts["shortable"], q_in=q_in,
                               q_out=q_out, slots=(long_slots, short_slots), max_hold=max_hold,
                               short_stop=short_stop, sides=sides)
    weights = (bt.next_bar(longs).astype(float) * (gross * long_share / long_slots)
               - bt.next_bar(shorts).astype(float) * (gross * (1 - long_share) / short_slots))
    return {"weights": weights, "borrow": parts["borrow"], "retrim_k": retrim_k}


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


def valid(params: dict) -> bool:
    """Reject parameter sets the book cannot express (missing keys read as ``signals()``'s
    defaults, so a partial dict — a search point alone — is judged too)."""
    p = {**_DEFAULTS, **params}
    return (0.5 < p["q_out"] < p["q_in"] < 1.0 and p["long_slots"] >= 1
            and p["short_slots"] >= 1
            and 0.0 < p["gross"] <= 1.0 and 0.0 < p["long_share"] < 1.0
            and p["sides"] in ("both", "long", "short")
            and 0.0 < p["si_mid"] < p["si_high"] <= p["short_max_si"]
            and p["dtc_window"] >= 1 and p["skew_window"] >= 1
            and p["skew_smooth"] >= 1 and p["score_smooth"] >= 1
            and p["w_si"] >= 0 and p["w_skew"] >= 0 and p["w_si"] + p["w_skew"] > 0
            and p["si_plausible_max"] > 0)


def compose(panel, *, dtc_window: int, skew_window: int, w_si: float, w_skew: float,
            neutral_groups: int, short_max_si: float, si_mid: float, si_high: float,
            borrow_gc: float, borrow_mid: float, borrow_high: float, min_price: float,
            skew_smooth: int = 1, score_smooth: int = 1, si_plausible_max: float = 1.0) -> dict:
    """Everything the book reads, bar by bar (pure; also what the notebooks inspect).

    Returns:
        ``raw`` (sleeve -> values, high = good), ``ranks`` (sleeve -> percentile ranks inside
        liquidity groups, NaN where silent), ``score`` (the composite's percentile rank among
        eligible names, NaN where ineligible), ``eligible``, ``si_share`` (short interest /
        shares outstanding), ``shortable`` and ``borrow`` (yearly rate per name and bar: the
        ``borrow_fee`` feature where it has a value, the tiers elsewhere).
    """
    feats = panel.features or {}
    close = panel.close
    nan = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    dtc = feats.get("short_interest", nan).reindex_like(close)
    shares = feats.get("fund_shares", nan).reindex_like(close)
    eligible = (close >= min_price) & dtc.notna() & shares.gt(0)
    if "member" in feats:
        eligible &= feats["member"].reindex_like(close).fillna(0).astype(bool)

    raw = {"si": -dtc.rolling(dtc_window, min_periods=1).mean().where(dtc.notna())}
    skew = feats.get("opt_skew")
    if skew is not None:
        skew = skew.reindex_like(close).rolling(skew_smooth, min_periods=1).mean()
    raw["skew"] = nan if skew is None else -(skew - skew.shift(skew_window))
    groups = _liquidity_groups(panel, neutral_groups)
    weights = {"si": w_si, "skew": w_skew}
    ranks = {k: _sleeve_rank(v, eligible, groups) for k, v in raw.items()}
    total = sum(weights[k] for k in ranks if weights[k] > 0)
    composite = sum(weights[k] * ranks[k].fillna(0.5) for k in ranks if weights[k] > 0) / total
    if score_smooth > 1:
        composite = composite.where(eligible).rolling(score_smooth, min_periods=1).mean()
    score = composite.where(eligible).rank(axis=1, pct=True)

    if "si_shares" in feats:    # FINRA's shares short / SEC's count, both as of their own date
        si_share = (feats["si_shares"].reindex_like(close) / shares).where(eligible)
        si_share = si_share.where(si_share <= si_plausible_max)
    else:                       # fallback: the stored volume is back-adjusted for every LATER
        if dtc.notna().any().any():                                  # split (look-ahead)
            warnings.warn("informed_flow_ls: no si_shares feature on this panel (only "
                          "research.load attaches it) — the crowding guard and borrow tiers "
                          "fall back to days-to-cover x back-adjusted volume, which reads "
                          "future splits", stacklevel=3)
        adv = panel.volume.rolling(20, min_periods=10).median()
        si_share = (dtc * adv / shares).where(eligible)
    shortable = si_share.le(short_max_si)
    borrow = pd.DataFrame(np.where(si_share > si_high, borrow_high,
                                   np.where(si_share > si_mid, borrow_mid, borrow_gc)),
                          index=close.index, columns=close.columns)
    if "borrow_fee" in feats:                   # the broker's own rate wins where it is known
        fee = feats["borrow_fee"].reindex_like(close)
        borrow = borrow.where(fee.isna(), fee)
    return {"raw": raw, "ranks": ranks, "score": score, "eligible": eligible,
            "si_share": si_share, "shortable": shortable, "borrow": borrow}


def _liquidity_groups(panel, n: int) -> pd.DataFrame | None:
    """Each name's dollar-volume group per bar (trailing 63-bar median, ``n`` equal groups,
    1 = least traded), or ``None`` when ``n < 2``. Known at the bar: no look-ahead."""
    if n < 2:
        return None
    dollars = (panel.close * panel.volume).rolling(63, min_periods=21).median()
    return np.ceil(dollars.rank(axis=1, pct=True) * n).clip(1, n)


def _rank(frame: pd.DataFrame, groups: pd.DataFrame | None) -> pd.DataFrame:
    """Percentile rank per bar, inside each liquidity group when ``groups`` is given."""
    if groups is None:
        return frame.rank(axis=1, pct=True)
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    labels = groups.to_numpy()
    for g in np.unique(labels[~np.isnan(labels)]):
        inside = groups == g
        out = out.where(~inside, frame.where(inside).rank(axis=1, pct=True))
    return out


def _sleeve_rank(raw: pd.DataFrame, eligible: pd.DataFrame,
                 groups: pd.DataFrame | None) -> pd.DataFrame:
    """One sleeve's ranks over the eligible names — none on a bar where it covers fewer than
    :data:`_MIN_COVERAGE` of them (every name then sits at the neutral 0.5)."""
    values = raw.where(eligible)
    ranks = _rank(values, groups)
    covered = values.notna().sum(axis=1) / eligible.sum(axis=1).replace(0, np.nan)
    ranks.loc[~(covered >= _MIN_COVERAGE)] = np.nan
    return ranks


def _select(score: pd.DataFrame, close: pd.DataFrame, shortable: pd.DataFrame, *,
            q_in: float, q_out: float, slots: tuple[int, int], max_hold: int, short_stop: float,
            sides: str = "both"):
    """The long and the short book, bar by bar (decided at the close, traded next bar).

    Exits come first, so a slot freed today can be refilled today: a NaN score (ineligible),
    a rank out of the band, ``max_hold`` reached (the clock restarts while the name still
    ranks in its entry zone), and for a short also the squeeze stop and becoming too crowded
    to borrow. A short stopped out is not shorted again until its rank has left the short
    entry zone once (else the stop would just be a round trip). Then each side fills its free
    slots, best candidates first.

    Returns:
        ``(longs, shorts, exits)``: boolean held frames (at the deciding bar, before
        ``next_bar``) and a list of ``(side, symbol, opened, closed, cause)``.
    """
    s = score.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    ok_short = shortable.reindex_like(score).fillna(False).to_numpy(dtype=bool)
    n, m = s.shape
    held = {+1: np.zeros((n, m), bool), -1: np.zeros((n, m), bool)}
    book = {+1: {}, -1: {}}                         # column -> [opened, clock, entry close]
    stopped = set()             # shorts bought back by the stop: no re-entry until they leave
    exits = []
    use = {"both": (+1, -1), "long": (+1,), "short": (-1,)}[sides]
    eps = 1e-9                  # rank thresholds are exact fractions (0.2 vs 1 - 0.8)
    for t in range(n):
        row = s[t]
        for side in use:
            enter = row >= q_in - eps if side > 0 else row <= 1 - q_in + eps
            for j in list(book[side]):
                opened, clock, entry = book[side][j]
                if enter[j]:
                    book[side][j][1] = t                # still in the entry zone: clock restarts
                cause = None
                if np.isnan(row[j]):
                    cause = "ineligible"
                elif (row[j] < q_out - eps) if side > 0 else (row[j] > 1 - q_out + eps):
                    cause = "band"
                elif t - book[side][j][1] >= max_hold:
                    cause = "max_hold"
                elif side < 0 and c[t, j] >= entry * (1 + short_stop):
                    cause = "short_stop"
                elif side < 0 and not ok_short[t, j]:
                    cause = "crowded"
                if cause:
                    exits.append((side, score.columns[j], score.index[opened],
                                  score.index[t], cause))
                    del book[side][j]
                    if cause == "short_stop":
                        stopped.add(j)
            if side < 0:                                 # out of the short zone: stop forgotten
                stopped -= {j for j in stopped if not enter[j]}
            free = slots[0 if side > 0 else 1] - len(book[side])
            if free > 0:
                cand = enter & ~np.isnan(row) & ~np.isin(np.arange(m), list(book[side]))
                cand &= ~np.isin(np.arange(m), list(book[-side]))
                if side < 0:
                    cand &= ok_short[t] & ~np.isin(np.arange(m), list(stopped))
                idx = np.flatnonzero(cand)
                order = idx[np.argsort(-row[idx] if side > 0 else row[idx], kind="stable")]
                for j in order[:free]:
                    book[side][j] = [t, t, c[t, j]]
            held[side][t, list(book[side])] = True
    frame = lambda a: pd.DataFrame(a, index=score.index, columns=score.columns)  # noqa: E731
    return frame(held[+1]), frame(held[-1]), exits
