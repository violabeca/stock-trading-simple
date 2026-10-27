"""quality_junk — quality minus junk, long/short and beta-neutral (STRATEGY_OVERVIEW.md §6, #4).

Long profitable, low-volatility companies; short unprofitable, volatile "junk". Investors overpay
for lottery-like stocks and leverage-constrained investors bid up risky ones (Frazzini & Pedersen
2014, betting against beta; Asness, Frazzini & Pedersen 2019, quality minus junk), gross
profitability is underpriced (Novy-Marx 2013), and in a crisis money flees junk first, so the
factor has earned most in falling markets. Two sleeves, each a percentile rank (high = quality):

1. **Profitability** (``w_prof``): gross profit TTM ÷ total assets; where gross profit is missing
   (banks, insurers, and a third of all filers, who never tag it) net income TTM ÷ equity
   (positive equity only). Ranked WITHIN sector (the ``sector`` feature; a name without one is
   its own group; no feature at all: one group), so the sleeve is not a sector bet, and against
   the sector's names measured the same way (:func:`_prof_rank`: the two ratios live on different
   scales). A ratio of two fundamentals needs no price or share count, which sidesteps the
   share-count unit slips and ADR ratios of ``value_factor``.
2. **Safety** (``w_safe``): minus the ``vol_window``-bar volatility of daily returns, ranked across
   all eligible names (``safe_by_sector``: within sector too); ``safety: beta`` ranks minus the
   ``vol_window``-bar beta to the equal-weight eligible names instead (betting against beta).

The composite (the weighted mean of the two ranks) is re-ranked across the eligible names. A
name is eligible on a bar where its close is at least ``min_price``, one of the two profitability
ratios is defined and its volatility exists (half the window of returns) — never an AND across
the profitability measures, which collapsed ``value_factor``'s cross-section.

**The book** (pure and stateful, :func:`_select`): the composite is consulted only on decision
bars (``rebalance``: the last session of every month or quarter, decided at its close, filled at
the next open). There a long leaves when its rank falls below ``q_out`` and a short when it rises
above ``1 - q_out``; then each side fills its free slots, best first, from the names ranking at
least ``q_in`` (long) or at most ``1 - q_in`` (short). Between decision bars the book is held;
only two risk exits act daily, both copied from ``informed_flow_ls_p1`` v6: a short whose close
rises ``short_stop`` above its entry close is bought back (and not shorted again until its rank has
left the short zone), and a short whose short interest exceeds ``short_max_si`` of its shares
outstanding is bought back (and never entered).

**Sizing** (:func:`_weights`, ``informed_flow_ls_p1``'s): ``"beta"`` scales a slot AT ENTRY by the
bar's median beta ÷ the name's trailing ``beta_window``-bar beta (between ``1 / max_size`` and
``max_size`` of the equal slot ``gross × side share / slots``), frozen while held — the engine
trades a name only when its own target changes. The low-beta longs are bigger and the high-beta
shorts smaller, so the two sides carry about the same market risk. New entries are scaled down
together when they would lift nominal gross above ``gross`` (shorts lock their proceeds, so no
order can take a long/short book above ~100% gross). That bound holds at entry only: held
positions drift with their prices and are never re-sized (realized gross averaged 0.99 in the lab
and reached 1.31).

**Borrow** (``informed_flow_ls_p1``'s tiers): short interest ÷ shares outstanding above ``si_high``
pays ``borrow_high`` a year, above ``si_mid`` ``borrow_mid``, otherwise — and wherever there is no
short-interest data, e.g. before 2018 — ``borrow_gc``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from pandas.tseries import holiday as hol

from trading_simple import backtest as bt

#: Profitability measures in the order they are tried: gross profit / assets, else ROE.
PROFITABILITY = ("gpa", "roe")


class _NYSE(hol.AbstractHolidayCalendar):
    """The NYSE's full-day holidays (New Year falling on a Saturday is not moved to Friday)."""

    rules = [hol.Holiday("New Year", month=1, day=1, observance=hol.sunday_to_monday),
             hol.USMartinLutherKingJr, hol.USPresidentsDay, hol.GoodFriday, hol.USMemorialDay,
             hol.Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                         observance=hol.nearest_workday),
             hol.Holiday("Independence Day", month=7, day=4, observance=hol.nearest_workday),
             hol.USLaborDay, hol.USThanksgivingDay,
             hol.Holiday("Christmas", month=12, day=25, observance=hol.nearest_workday)]


#: One-off NYSE closures since the panels start (national days of mourning).
CLOSED = ("2018-12-05", "2025-01-09")


def next_session(day) -> pd.Timestamp:
    """The NYSE session after ``day`` (a date; weekends, holidays and :data:`CLOSED` skipped)."""
    d = pd.Timestamp(day)
    d = (d.tz_convert(None) if d.tzinfo is not None else d).normalize()
    shut = _NYSE().holidays(d, d + pd.Timedelta(days=14)).union(pd.DatetimeIndex(CLOSED))
    return d + pd.offsets.CustomBusinessDay(1, holidays=shut)


def signals(panel, *, w_prof: float = 1.0, w_safe: float = 1.0, vol_window: int = 252,
            safety: str = "vol", safe_by_sector: bool = False, w_growth: float = 0.0,
            growth_window: int = 252, q_in: float = 0.9,
            q_out: float = 0.8, slots: int = 20, rebalance: str = "month", gross: float = 0.98,
            long_share: float = 0.5, sides: str = "both", sizing: str = "beta",
            beta_window: int = 126, max_size: float = 1.5, short_stop: float = 1.0,
            short_max_si: float = 0.3, si_mid: float = 0.1, si_high: float = 0.2,
            borrow_gc: float = 0.005, borrow_mid: float = 0.03, borrow_high: float = 0.1,
            min_price: float = 5.0, retrim_k: float = 2.0, **_) -> dict:
    """Target weights (long > 0, short < 0), every name's borrow rate, and the long cap.

    Args:
        panel: Daily panel. ``features``: ``fund_gp_ttm``, ``fund_assets``, ``fund_ni_ttm``,
            ``fund_equity`` (profitability), ``short_interest`` and ``fund_shares`` (borrow
            tiers; optional), ``sector`` (numeric codes; optional).
        w_prof, w_safe: Weights of the profitability and safety ranks in the composite.
        vol_window: Bars of daily returns behind the safety measure (half of them required).
        safety: ``"vol"`` (minus volatility) or ``"beta"`` (minus beta to the equal-weight
            eligible names, over the same window).
        safe_by_sector: Rank the safety measure within sector as well (False: across all).
        w_growth: Weight of a third sleeve, profitability GROWTH: the change in the name's own
            profitability ratio over ``growth_window`` bars, ranked like profitability (QMJ's
            growth dimension; a name without a year of history sits at the neutral 0.5; 0 = off).
        growth_window: Bars over which profitability growth is measured (252 = four quarters).
        q_in, q_out: Long entry / exit ranks; the short side mirrors them (``1 - q``).
        slots: Positions per side.
        rebalance: ``"month"`` or ``"quarter"``: the decision bars (last session of each).
        gross: Long plus short exposure as a share of the book at entry (<= 1).
        long_share: The long side's share of ``gross`` before beta sizing (0.5 = equal slots).
        sides: ``"both"``, ``"long"`` or ``"short"``.
        sizing: ``"beta"`` (slot × median beta ÷ beta at entry) or ``"equal"``.
        beta_window: Bars of the trailing beta that sizing reads.
        max_size: Bounds of beta sizing: between ``1 / max_size`` and ``max_size`` of a slot.
        short_stop: A short whose close rises this far above its entry close is bought back.
        short_max_si: Short interest / shares outstanding above which a name is not shorted.
        si_mid, si_high: Crowding tiers of the borrow rate.
        borrow_gc, borrow_mid, borrow_high: Yearly borrow rates below ``si_mid``, up to
            ``si_high``, above it (``borrow_gc`` also where there is no short-interest data).
        min_price: Minimum close for a name to be ranked.
        retrim_k: A long grown to ``k ×`` its slot is trimmed back (the engine's cap).

    Returns:
        ``{"weights", "borrow", "retrim_k"}`` for the engine's portfolio mode.
    """
    parts = compose(panel, w_prof=w_prof, w_safe=w_safe, vol_window=vol_window, safety=safety,
                    safe_by_sector=safe_by_sector, w_growth=w_growth,
                    growth_window=growth_window, min_price=min_price, beta_window=beta_window,
                    short_max_si=short_max_si, si_mid=si_mid, si_high=si_high,
                    borrow_gc=borrow_gc, borrow_mid=borrow_mid, borrow_high=borrow_high)
    decide = decision_bars(panel.close.index, rebalance)
    longs, shorts, _ = _select(parts["score"], panel.close, parts["shortable"], decide, q_in=q_in,
                               q_out=q_out, slots=slots, short_stop=short_stop, sides=sides)
    if sizing == "legs":
        weights = _leg_weights(longs, shorts, parts["beta"], decide, gross=gross, slots=slots)
    else:
        weights = _weights(longs, shorts, parts["beta"], gross=gross, long_share=long_share,
                           slots=slots, sizing=sizing, max_size=max_size)
    return {"weights": weights, "borrow": parts["borrow"], "retrim_k": retrim_k}


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


#: ``q_out`` must sit below ``q_in`` by more than float noise: an optimizer's float steps give
#: 0.8500000000000001, which a strict ``q_out < q_in`` let through as a hysteresis band of zero
#: width (round 1's tuned pick, RESEARCH_LOG.md §4).
BAND_TOL = 1e-9


def valid(params: dict) -> bool:
    """Reject parameter sets the book cannot express (missing keys read as ``signals()``'s
    defaults, so a partial dict — a search point alone — is judged too)."""
    p = {**_DEFAULTS, **params}
    return (0.5 < p["q_out"] < p["q_in"] - BAND_TOL and p["q_in"] < 1.0 and p["slots"] >= 1
            and 0.0 < p["gross"] <= 1.0 and 0.0 < p["long_share"] < 1.0
            and p["sides"] in ("both", "long", "short")
            and p["rebalance"] in ("month", "quarter")
            and p["safety"] in ("vol", "beta")
            and p["sizing"] in ("equal", "beta", "legs") and p["beta_window"] >= 2
            and p["vol_window"] >= 20 and p["max_size"] >= 1.0
            and p["w_prof"] >= 0 and p["w_safe"] >= 0 and p["w_growth"] >= 0
            and p["w_prof"] + p["w_safe"] + p["w_growth"] > 0 and p["growth_window"] >= 21
            and 0.0 < p["si_mid"] < p["si_high"] <= p["short_max_si"]
            and p["short_stop"] > 0)


def compose(panel, *, w_prof: float = 1.0, w_safe: float = 1.0, vol_window: int = 252,
            safety: str = "vol", safe_by_sector: bool = False, w_growth: float = 0.0,
            growth_window: int = 252, min_price: float = 5.0,
            beta_window: int = 126, short_max_si: float = 0.3, si_mid: float = 0.1,
            si_high: float = 0.2, borrow_gc: float = 0.005, borrow_mid: float = 0.03,
            borrow_high: float = 0.1) -> dict:
    """Everything the book reads, bar by bar (pure; also what the research scripts inspect).

    Returns:
        ``raw`` (``gpa``, ``roe``, ``prof`` = the hybrid, ``vol``, ``growth``), ``measure``
        (which ratio each name's profitability uses: 0 = gpa, 1 = roe, NaN = none), ``ranks``
        (``prof``, ``safe``, ``growth``: percentile ranks over the eligible names, high =
        quality; growth NaN where it has no history), ``score`` (the
        composite's percentile rank, NaN where ineligible), ``eligible``, ``beta`` (trailing
        ``beta_window``-bar beta to the equal-weight eligible names, known at the bar's close),
        ``si_share``, ``shortable`` and ``borrow`` (yearly rate per name and bar).
    """
    feats = panel.features or {}
    close = panel.close
    nan = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    def get(name):
        return feats.get(name, nan).reindex(index=close.index, columns=close.columns)

    assets, equity = get("fund_assets"), get("fund_equity")
    gpa = get("fund_gp_ttm") / assets.where(assets > 0)
    roe = get("fund_ni_ttm") / equity.where(equity > 0)
    prof = gpa.where(gpa.notna(), roe)
    measure = pd.DataFrame(np.where(gpa.notna(), 0.0, np.where(roe.notna(), 1.0, np.nan)),
                           index=close.index, columns=close.columns)

    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(vol_window, min_periods=vol_window // 2).std()
    base = (close >= min_price) & prof.notna()          # the market: no volatility needed yet
    market = ret.where(base).mean(axis=1)
    eligible = base & vol.notna()
    beta = _beta(ret, market, beta_window)
    risk = vol if safety == "vol" else _beta(ret, market, vol_window, vol_window // 2)
    eligible &= risk.notna()

    sector = feats.get("sector")
    groups = None if sector is None else sector.reindex(index=close.index,
                                                         columns=close.columns).fillna(-1.0)
    ranks = {"prof": _prof_rank(prof.where(eligible), gpa.notna(), groups),
             "safe": _rank((-risk).where(eligible), groups if safe_by_sector else None)}
    # growth: the change in the ratio the name is measured on TODAY (NaN without a year of it)
    growth = (gpa - gpa.shift(growth_window)).where(gpa.notna(), roe - roe.shift(growth_window))
    composite = w_prof * ranks["prof"] + w_safe * ranks["safe"]
    if w_growth > 0:
        ranks["growth"] = _prof_rank(growth.where(eligible), gpa.notna(), groups)
        composite = composite + w_growth * ranks["growth"].fillna(0.5)
    composite = composite / (w_prof + w_safe + w_growth)
    score = composite.where(eligible).rank(axis=1, pct=True)

    dtc, shares = get("short_interest"), get("fund_shares")
    adv = panel.volume.reindex_like(close).rolling(20, min_periods=10).median()
    si_share = dtc * adv / shares.where(shares > 0)
    shortable = ~(si_share > short_max_si)              # no short-interest data: shortable
    borrow = pd.DataFrame(np.where(si_share > si_high, borrow_high,
                                   np.where(si_share > si_mid, borrow_mid, borrow_gc)),
                          index=close.index, columns=close.columns)
    return {"raw": {"gpa": gpa, "roe": roe, "prof": prof, "vol": vol, "risk": risk,
                    "growth": growth},
            "measure": measure, "ranks": ranks, "score": score, "eligible": eligible,
            "beta": beta.where(eligible), "si_share": si_share, "shortable": shortable,
            "borrow": borrow}


def _beta(ret: pd.DataFrame, market: pd.Series, window: int,
          min_periods: int | None = None) -> pd.DataFrame:
    """Trailing ``window``-bar beta of every column to ``market`` (known at the bar's close)."""
    lag = max(window // 2, 2) if min_periods is None else min_periods
    return ret.rolling(window, min_periods=lag).cov(market).div(
        market.rolling(window, min_periods=lag).var(), axis=0)


def decision_bars(index: pd.DatetimeIndex, rebalance: str = "month") -> np.ndarray:
    """The bars the book is re-ranked on: the last session of every month (``"month"``) or of
    March, June, September and December (``"quarter"``).

    A bar is a month's last session when the next bar is in another month. The panel's last
    bar has no next bar: it counts when the next NYSE session (:func:`next_session`) falls in
    another month — a calendar fact known in advance, no price is read. A plain business-day
    step would miss the session before Good Friday at the end of March (2018-03-29,
    2024-03-28) or before a Memorial Day on May 31 (2021-05-28), so a live run on such a day
    would skip the month-end decision the backtest makes.
    """
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    months = np.asarray(index.year * 12 + index.month)
    after = next_session(index[-1])
    nxt = np.r_[months[1:], after.year * 12 + after.month]
    last = months != nxt
    if rebalance == "quarter":
        last &= np.isin(np.asarray(index.month), (3, 6, 9, 12))
    return last


def _rank(frame: pd.DataFrame, groups: pd.DataFrame | None,
          with_counts: bool = False):
    """Percentile rank per bar, inside each group of ``groups`` (same shape) when given; with
    ``with_counts`` also the number of ranked (non-NaN) names in each cell's group."""
    if groups is None:
        ranks = frame.rank(axis=1, pct=True)
        if not with_counts:
            return ranks
        n = np.repeat(frame.notna().sum(axis=1).to_numpy()[:, None], frame.shape[1], axis=1)
        return ranks, n
    out, count = np.full(frame.shape, np.nan), np.zeros(frame.shape)
    labels = groups.to_numpy(dtype=float)
    for g in np.unique(labels[~np.isnan(labels)]):
        inside = labels == g
        cols = np.flatnonzero(inside.any(axis=0))
        sub = frame.iloc[:, cols].where(inside[:, cols])
        out[:, cols] = np.where(inside[:, cols], sub.rank(axis=1, pct=True).to_numpy(),
                                out[:, cols])
        count[:, cols] = np.where(inside[:, cols], sub.notna().sum(axis=1).to_numpy()[:, None],
                                  count[:, cols])
    ranks = pd.DataFrame(out, index=frame.index, columns=frame.columns)
    return (ranks, count) if with_counts else ranks


#: A (sector, measure) group needs this many eligible names on a bar to be ranked on its own.
MIN_GROUP = 5


def _prof_rank(prof: pd.DataFrame, uses_gpa: pd.DataFrame,
               groups: pd.DataFrame | None) -> pd.DataFrame:
    """The profitability percentile (NaN where ``prof`` is NaN): each name against the names of
    its sector measured the same way (gross profit / assets, or ROE where gross profit is
    missing) — the two scales differ (per-period medians 0.26-0.32 vs 0.10-0.15 in the lab's
    universes), so ranking them together would put most ROE-measured names below most
    gross-profit names. A group with fewer than
    :data:`MIN_GROUP` names on a bar ranks against every name on its measure instead."""
    measure = uses_gpa.astype(float)
    wide = _rank(prof, measure)                          # each measure's own cross-section
    if groups is None:
        return wide
    within, n = _rank(prof, groups * 2 + measure, with_counts=True)
    return within.where(n >= MIN_GROUP, wide)


def _select(score: pd.DataFrame, close: pd.DataFrame, shortable: pd.DataFrame,
            decide: np.ndarray, *, q_in: float, q_out: float, slots: int, short_stop: float,
            sides: str = "both"):
    """The long and the short book, bar by bar (decided at the close, traded next bar).

    On a decision bar, exits come first (a NaN score = ineligible, a rank out of the band), so a
    freed slot is refilled the same day; then each side fills its free slots, best candidates
    first. Every bar, before that, a short may be bought back by the squeeze stop or for being
    too crowded to borrow; a short stopped out is not shorted again until its rank has been seen
    outside the short entry zone on a decision bar.

    Returns:
        ``(longs, shorts, exits)``: boolean held frames (at the deciding bar, before the one-bar
        execution shift) and a list of ``(side, symbol, opened, closed, cause)``.
    """
    s = score.to_numpy(dtype=float)
    c = close.reindex_like(score).to_numpy(dtype=float)
    ok_short = shortable.reindex_like(score).fillna(True).to_numpy(dtype=bool)
    decide = np.asarray(decide, dtype=bool)
    n, m = s.shape
    held = {+1: np.zeros((n, m), bool), -1: np.zeros((n, m), bool)}
    book = {+1: {}, -1: {}}                         # column -> [opened bar, entry close]
    stopped: set[int] = set()
    exits = []
    use = {"both": (+1, -1), "long": (+1,), "short": (-1,)}[sides]
    eps = 1e-9                  # rank thresholds are exact fractions (0.1 vs 1 - 0.9)

    def close_out(side, j, t, cause):
        exits.append((side, score.columns[j], score.index[book[side][j][0]], score.index[t],
                      cause))
        del book[side][j]

    for t in range(n):
        for j in list(book[-1]):                    # daily risk exits of the short side
            if c[t, j] >= book[-1][j][1] * (1 + short_stop):
                close_out(-1, j, t, "short_stop")
                stopped.add(j)
            elif not ok_short[t, j]:
                close_out(-1, j, t, "crowded")
        if decide[t]:
            row = s[t]
            for side in use:
                enter = row >= q_in - eps if side > 0 else row <= 1 - q_in + eps
                for j in list(book[side]):
                    if np.isnan(row[j]):
                        close_out(side, j, t, "ineligible")
                    elif (row[j] < q_out - eps) if side > 0 else (row[j] > 1 - q_out + eps):
                        close_out(side, j, t, "band")
                if side < 0:                         # seen out of the short zone: forgotten
                    stopped -= {j for j in stopped if not enter[j]}
                free = slots - len(book[side])
                if free <= 0:
                    continue
                cand = enter & ~np.isnan(row)
                cand[list(book[+1]) + list(book[-1])] = False
                if side < 0:
                    cand &= ok_short[t]
                    cand[list(stopped)] = False
                idx = np.flatnonzero(cand)
                order = idx[np.argsort(-row[idx] if side > 0 else row[idx], kind="stable")]
                for j in order[:free]:
                    book[side][j] = [t, c[t, j]]
        for side in use:
            held[side][t, list(book[side])] = True
    frame = lambda a: pd.DataFrame(a, index=score.index, columns=score.columns)  # noqa: E731
    return frame(held[+1]), frame(held[-1]), exits


def _weights(longs: pd.DataFrame, shorts: pd.DataFrame, beta: pd.DataFrame, *, gross: float,
             long_share: float, slots: int, sizing: str = "beta",
             max_size: float = 1.5) -> pd.DataFrame:
    """Target weights at the fill bar from the held frames (copied from informed_flow_ls_p1).

    ``"equal"``: every long ``gross × long_share / slots``, every short the rest over ``slots``.
    ``"beta"``: a slot is scaled AT ENTRY by the bar's median beta ÷ the name's beta (clipped to
    ``1 / max_size`` .. ``max_size``; no beta: 1; a zero or negative beta: ``1 / max_size``), so
    each position carries about the same market risk. The size is frozen while the name is held
    (re-sizing would re-trade every holding), and a bar's new entries are scaled down together
    whenever they would lift the book's nominal gross above ``gross``.
    """
    size_l, size_s = gross * long_share / slots, gross * (1 - long_share) / slots
    if sizing == "equal":
        return (bt.next_bar(longs).astype(float) * size_l
                - bt.next_bar(shorts).astype(float) * size_s)
    b = beta.reindex_like(longs).to_numpy(dtype=float)
    median = pd.DataFrame(np.where(b > 0, b, np.nan)).median(axis=1).to_numpy()[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = median / b
    ratio = np.clip(np.where(np.isfinite(ratio), ratio, 1.0), 1.0 / max_size, max_size)
    ratio[b <= 0] = 1.0 / max_size
    held_l, held_s = longs.to_numpy(dtype=bool), shorts.to_numpy(dtype=bool)
    out = np.zeros(held_l.shape)
    prev = np.zeros(held_l.shape[1])
    for t in range(len(out)):
        side = held_l[t].astype(float) - held_s[t].astype(float)
        keep = (side != 0) & (np.sign(prev) == side)
        size = np.where(keep, prev, 0.0)
        new = np.where((side != 0) & ~keep, side * ratio[t] * np.where(side > 0, size_l, size_s),
                       0.0)
        need, room = np.abs(new).sum(), gross - np.abs(size).sum()
        if need > 0:
            size = size + new * min(1.0, max(room, 0.0) / need)
        out[t] = prev = size
    frame = pd.DataFrame(out, index=longs.index, columns=longs.columns)
    return frame.shift(1, fill_value=0.0)                  # == bt.next_bar for sizes


#: ``"legs"`` sizing re-sizes a held name only when its target moved more than this share of its
#: current size: a monthly micro-trim of every holding would pay a $1 minimum commission each
#: time in a real account (the cost model charges it only in proportion).
RESIZE_BAND = 0.25
#: Name betas are clipped to this range before a side's average is taken.
BETA_CLIP = (0.1, 3.0)


def _leg_weights(longs: pd.DataFrame, shorts: pd.DataFrame, beta: pd.DataFrame,
                 decide: np.ndarray, *, gross: float, slots: int) -> pd.DataFrame:
    """BAB-style sizing: the two SIDES carry the same beta (Frazzini & Pedersen 2014).

    On a decision bar the long side gets ``D_L = gross × β̄_S ÷ (β̄_L + β̄_S)`` dollars and the
    short side ``D_S = gross − D_L``, where ``β̄`` is the mean trailing beta (clipped to
    :data:`BETA_CLIP`) of the side's held names, so ``D_L β̄_L = D_S β̄_S``; every slot of a side
    is ``D / slots`` (an empty slot is cash), so the balance holds for the dollars actually
    invested only when both sides are full: a side holding ``n < slots`` names invests
    ``n / slots`` of its ``D``. A new name takes its side's slot size; a held name
    keeps its size unless its slot size moved by more than :data:`RESIZE_BAND`, or the book's
    nominal gross would exceed ``gross``, when every holding is re-sized. Between decision bars
    nothing is re-sized (a short bought back mid-month leaves its dollars in cash). An empty side
    falls back to ``gross / 2`` a side.
    """
    b = np.clip(beta.reindex_like(longs).to_numpy(dtype=float), *BETA_CLIP)
    held_l, held_s = longs.to_numpy(dtype=bool), shorts.to_numpy(dtype=bool)
    out = np.zeros(held_l.shape)
    prev = np.zeros(held_l.shape[1])
    for t in range(len(out)):
        side = held_l[t].astype(float) - held_s[t].astype(float)
        size = np.where((side != 0) & (np.sign(prev) == side), prev, 0.0)
        if decide[t] and (side != 0).any():
            with np.errstate(invalid="ignore"):
                bl, bs = np.nanmean(b[t][held_l[t]]), np.nanmean(b[t][held_s[t]])
            d_l = gross / 2 if not (np.isfinite(bl) and np.isfinite(bs)) else gross * bs / (bl + bs)
            target = np.where(side > 0, d_l / slots, -(gross - d_l) / slots) * (side != 0)
            move = (size == 0) | (np.abs(target - size) > RESIZE_BAND * np.abs(size))
            size = np.where((side != 0) & move, target, size)
            if np.abs(size).sum() > gross + 1e-9:
                size = target
        out[t] = prev = size
    frame = pd.DataFrame(out, index=longs.index, columns=longs.columns)
    return frame.shift(1, fill_value=0.0)                  # == bt.next_bar for sizes
