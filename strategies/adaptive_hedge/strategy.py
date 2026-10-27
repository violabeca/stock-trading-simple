"""adaptive_hedge — a long short-interest book plus breakdown shorts that appear when stocks break
down (STRATEGY_OVERVIEW.md §6, proposal 3; evaluated in ``strategies/proposals_lab``).

A long-only book cannot beat buy-and-hold in a bear market, and a permanently hedged one gives up
the bull market. Here the hedge comes from the cross-section, not from a market-timing rule:

* **Longs** are informed_flow v41's rule: own the names short sellers avoid. Rank minus the
  days-to-cover averaged over ``dtc_window`` bars inside ``neutral_groups`` dollar-volume groups,
  enter at rank ``q_in``, leave below ``q_out``; eligible = a close of at least ``min_price``, a
  days-to-cover and a share count (``fund_shares``: SEC filers only, so an interlisted name whose
  FINRA short interest covers only its US line cannot look falsely unshorted).
* **Shorts** are trend_following's short starts: a close below the prior ``entry_window``-bar low
  with ADX(``adx_window``) >= ``adx_min`` and -DI > +DI. A short leaves on a close through the
  ratcheting chandelier (the lowest low since entry plus ``atr_mult`` x ATR(``atr_window``)), on
  an opposite (long) start, or when the name becomes too crowded to borrow.
* **One book of ``slots`` equal slots** of ``gross / slots``. A short start takes a free slot, and
  when none is free it displaces the weakest-ranked long; at most ``max_shorts`` shorts. In a
  rising market few names break down and the book stays nearly fully long; in a falling one many
  do, and each short moves net exposure by two slots (a long out, a short in).

Every short pays a borrow fee (the engine's ``borrow`` key) set by how crowded the name is: short
interest / shares outstanding up to ``si_mid`` costs ``borrow_gc`` a year, up to ``si_high``
``borrow_mid``, above it ``borrow_high``; no short-interest data reads as ``borrow_gc``. A name
above ``short_max_si`` is never shorted. ``hedge`` (an index fund in the panel, e.g. ``"SPY"``)
holds every short slot as a short of the index instead of the single name: the same breadth
signal, without the single-name squeeze risk.

Everything is decided at bar t's close and filled at t+1's open (``bt.next_bar``). A held name's
weight never changes while held, so the engine trades a name only when it enters or leaves.

**Missing prices** (review fix, 2026-10-23). The book decides only on trading days: a row on
which almost no name is priced (a US holiday that a foreign listing in the panel traded on) is
skipped, so it neither empties the book nor breaks the rolling windows, and the decision of the
day before it fills at the next trading day's open. A held name without a price on a bar is
carried, not sold (trend_following's ``_ride`` does the same), and a target change due on a bar
where the name has no open waits for its next priced open: the engine drops an order at a NaN
price and, trading a name only when its target changes, never sends it again. A name whose data
ENDS is carried to the end of the panel, so its slot stays taken (CWAN in the lab's T2 2026
periods; freeing it after N unpriced bars is parked in IDEAS.md).
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pandas as pd

from trading_simple import backtest as bt

#: The long sleeve ranks only on bars where at least this share of the priced names (close >=
#: ``min_price``) carry a live days-to-cover; below it every name sits at the neutral 0.5, so a
#: table that went stale turns the long side into cash, never into a tilt on the few names left.
_MIN_COVERAGE = 0.5
#: A row is a trading day when at least this share of the median count of priced names over the
#: :data:`_LIVE_LOOKBACK` rows before it carries a close (causal; no real session halves it).
_LIVE_FRACTION = 0.5
_LIVE_LOOKBACK = 21
#: Entry filters for the short side: none; only names in the crowded half by days-to-cover (long
#: rank <= 0.5, variant B); only names with a negative 12-1 month return.
SHORT_FILTERS = ("none", "crowded", "momentum")


def signals(panel, *, dtc_window: int = 42, neutral_groups: int = 5, q_in: float = 0.9,
            q_out: float = 0.8, slots: int = 30, max_hold: int = 126, gross: float = 0.98,
            retrim_k: float = 2.0, min_price: float = 5.0, entry_window: int = 9,
            adx_window: int = 14, adx_min: float = 34.0, atr_window: int = 37,
            atr_mult: float = 14.0, rs_window: int = 252, rs_skip: int = 21,
            max_shorts: int = 20, short_hold: int = 0, exit_window: int = 0,
            short_stop: float = 0.0, short_filter: str = "none", short_max_si: float = 0.3,
            si_mid: float = 0.1, si_high: float = 0.2, borrow_gc: float = 0.005,
            borrow_mid: float = 0.03, borrow_high: float = 0.1, hedge: str = "",
            **_) -> dict:
    """Target weights (long > 0, short < 0), every name's borrow rate and the long cap.

    Args:
        panel: Daily panel with ``features`` ``short_interest`` (days-to-cover) and
            ``fund_shares``. Without them the long side is empty and only shorts trade.
        dtc_window: Bars days-to-cover is averaged over before ranking.
        neutral_groups: Dollar-volume groups the long sleeve is ranked within (< 2: one group).
        q_in, q_out: Long entry / exit ranks.
        slots: Equal slots in the book, shared by both sides.
        max_hold: Bars a long is held unless it still ranks >= ``q_in`` (the clock restarts).
        gross: Long plus short exposure as a share of the book (<= 1: shorts lock cash).
        retrim_k: A long that grows to ``k ×`` its slot is trimmed back (engine cap, longs only).
        min_price: Minimum close to be ranked long or shorted.
        entry_window: Donchian lookback of a start (prior ``entry_window``-bar low / high).
        adx_window: Wilder smoothing window of the ADX and the directional indicators.
        adx_min: ADX floor a breakdown needs (0 = no confirmation).
        atr_window: ATR window of the chandelier.
        atr_mult: Chandelier distance in ATRs above the lowest low since entry.
        rs_window, rs_skip: 12-1 month relative strength that orders competing short starts
            (most negative first).
        max_shorts: Most shorts held at once (0 = the long side alone).
        short_hold: A short is bought back once this many bars pass without a fresh start on
            the name (the clock restarts on every new start while held; 0 = off).
        exit_window: A short is also bought back on a close above the highest high of the
            prior ``exit_window`` bars, with no ADX confirmation (the Turtle exit; 0 = off).
        short_stop: A short whose close rises this far above its entry close is bought back
            (the squeeze stop; 0 = off).
        short_filter: One of :data:`SHORT_FILTERS`, applied at entry.
        short_max_si: Short interest / shares outstanding above which a name is not shorted
            (and a held short is bought back).
        si_mid, si_high: Crowding tiers of the borrow fee.
        borrow_gc, borrow_mid, borrow_high: Yearly borrow fees of the three tiers.
        hedge: Symbol of an index fund in the panel ("" = off): every short slot is held as a
            short of it instead of the single name (it is never ranked or shorted itself).

    Returns:
        ``{"weights", "borrow", "retrim_k"}`` for the engine's portfolio mode.
    """
    if hedge and hedge not in panel.close.columns:
        raise ValueError(f"hedge {hedge!r} is not a column of the panel (lab: --hedge)")
    core = _without(panel, hedge) if hedge else panel
    days = _rows(core, _live_rows(core.close))      # decisions on trading days only
    parts = compose(days, dtc_window=dtc_window, neutral_groups=neutral_groups,
                    min_price=min_price, entry_window=entry_window, adx_window=adx_window,
                    adx_min=adx_min, atr_window=atr_window, exit_window=exit_window,
                    rs_window=rs_window,
                    rs_skip=rs_skip, short_filter=short_filter, short_max_si=short_max_si,
                    si_mid=si_mid, si_high=si_high, borrow_gc=borrow_gc,
                    borrow_mid=borrow_mid, borrow_high=borrow_high)
    longs, shorts, _ = _book(parts, q_in=q_in, q_out=q_out, slots=slots, max_hold=max_hold,
                             max_shorts=max_shorts, atr_mult=atr_mult, short_hold=short_hold,
                             short_stop=short_stop)
    size = gross / slots
    weights = (bt.next_bar(longs).astype(float) - bt.next_bar(shorts).astype(float)) * size
    borrow = parts["borrow"]
    if hedge:
        weights = weights.clip(lower=0.0).reindex(columns=panel.close.columns, fill_value=0.0)
        weights[hedge] = -(shorts.sum(axis=1).astype(float) * size).shift(1, fill_value=0.0)
        borrow = borrow.reindex(columns=panel.close.columns, fill_value=borrow_gc)
    index = panel.close.index                       # a skipped row holds the book unchanged
    weights = _on_priced_opens(weights.reindex(index).ffill(), panel.open_)
    borrow = borrow.reindex(index).ffill().fillna(borrow_gc)
    return {"weights": weights, "borrow": borrow, "retrim_k": retrim_k}


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


def valid(params: dict) -> bool:
    """Reject parameter sets the book cannot express (a partial dict is read over the
    ``signals()`` defaults, so a search point alone is judged too)."""
    p = {**_DEFAULTS, **params}
    return (0.5 < p["q_out"] < p["q_in"] < 1.0 and p["slots"] >= 1
            and 0 <= p["max_shorts"] <= p["slots"] and 0.0 < p["gross"] <= 1.0
            and p["short_hold"] >= 0 and p["exit_window"] >= 0 and p["short_stop"] >= 0
            and p["dtc_window"] >= 1 and p["neutral_groups"] >= 0 and p["max_hold"] >= 1
            and p["entry_window"] >= 2 and p["adx_window"] >= 2 and p["atr_window"] >= 2
            and 0.0 <= p["adx_min"] < 100.0 and p["atr_mult"] > 0.0
            and 0 <= p["rs_skip"] < p["rs_window"]
            and p["short_filter"] in SHORT_FILTERS
            and 0.0 < p["si_mid"] < p["si_high"] <= p["short_max_si"]
            and min(p["borrow_gc"], p["borrow_mid"], p["borrow_high"]) >= 0.0
            and isinstance(p["hedge"], str))


def compose(panel, *, dtc_window: int, neutral_groups: int, min_price: float,
            entry_window: int, adx_window: int, adx_min: float, atr_window: int,
            rs_window: int, rs_skip: int, short_filter: str, short_max_si: float,
            si_mid: float, si_high: float, borrow_gc: float, borrow_mid: float,
            borrow_high: float, exit_window: int = 0, **_) -> dict:
    """Everything the book reads, bar by bar (pure; also what the research scripts inspect).

    Returns:
        ``score`` (long rank among eligible names, NaN where ineligible), ``eligible``,
        ``short_start`` / ``long_start`` (trend starts at the deciding bar), ``rel`` (12-1 month
        relative strength), ``low`` / ``close`` / ``atr`` (the chandelier's inputs),
        ``cover`` (a close above the prior ``exit_window``-bar high; all False when off),
        ``short_ok`` (a start may be shorted), ``crowded`` (too crowded to stay short),
        ``si_share`` (short interest / shares outstanding) and ``borrow`` (yearly rate).
    """
    feats = panel.features or {}
    close = panel.close
    nan = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    dtc = feats.get("short_interest", nan).reindex_like(close)
    shares = feats.get("fund_shares", nan).reindex_like(close)
    priced = close >= min_price
    eligible = priced & dtc.notna() & shares.gt(0)

    raw = -dtc.rolling(dtc_window, min_periods=1).mean().where(dtc.notna())
    ranks = _sleeve_rank(raw, eligible, priced, _liquidity_groups(panel, neutral_groups))
    score = ranks.fillna(0.5).where(eligible).rank(axis=1, pct=True)

    upper = panel.high.rolling(entry_window, min_periods=entry_window).max().shift(1)
    lower = panel.low.rolling(entry_window, min_periods=entry_window).min().shift(1)
    strength, plus_di, minus_di = adx(panel, adx_window)
    atr = bt.atr(panel, atr_window)
    ready = atr.notna() & strength.notna()
    strong = strength >= adx_min
    long_start = ready & (close > upper) & strong & (plus_di > minus_di)
    short_start = ready & (close < lower) & strong & (minus_di > plus_di)
    if exit_window > 0:                     # the Turtle exit: the name's own trend reversed
        cover = close > panel.high.rolling(exit_window, min_periods=exit_window).max().shift(1)
    else:
        cover = pd.DataFrame(False, index=close.index, columns=close.columns)
    momentum = close.shift(rs_skip) / close.shift(rs_window) - 1.0
    rel = momentum.sub(momentum.mean(axis=1), axis=0)

    adv = panel.volume.rolling(20, min_periods=10).median()
    si_share = (dtc * adv / shares).where(shares.gt(0))
    crowded = si_share > short_max_si
    short_ok = priced & ~crowded
    if short_filter == "crowded":           # variant B: only the crowded half by days-to-cover
        short_ok &= score <= 0.5
    elif short_filter == "momentum":        # only names already losing over 12-1 months
        short_ok &= momentum < 0
    borrow = pd.DataFrame(np.where(si_share > si_high, borrow_high,
                                   np.where(si_share > si_mid, borrow_mid, borrow_gc)),
                          index=close.index, columns=close.columns)
    return {"score": score, "eligible": eligible, "short_start": short_start,
            "long_start": long_start, "rel": rel, "low": panel.low, "close": close, "atr": atr,
            "cover": cover,
            "short_ok": short_ok, "crowded": crowded, "si_share": si_share, "borrow": borrow}


def adx(panel, window: int = 14) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wilder's ADX with its two directional indicators (copied from trend_following).

    Returns:
        ``(adx, plus_di, minus_di)``, each in 0-100, NaN during the warm-up.
    """
    up, down = panel.high.diff(), -panel.low.diff()
    known = up.notna() & down.notna()
    plus = up.where((up > down) & (up > 0), 0.0).where(known)
    minus = down.where((down > up) & (down > 0), 0.0).where(known)

    def smooth(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.ewm(alpha=1.0 / window, min_periods=window).mean()

    atr = bt.atr(panel, window)
    plus_di, minus_di = 100.0 * smooth(plus) / atr, 100.0 * smooth(minus) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return smooth(dx), plus_di, minus_di


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


def _sleeve_rank(raw: pd.DataFrame, eligible: pd.DataFrame, priced: pd.DataFrame,
                 groups: pd.DataFrame | None) -> pd.DataFrame:
    """The sleeve's ranks over the eligible names — none on a bar where fewer than
    :data:`_MIN_COVERAGE` of the priced names carry a value (every name then sits at 0.5)."""
    values = raw.where(eligible)
    ranks = _rank(values, groups)
    covered = values.notna().sum(axis=1) / priced.sum(axis=1).replace(0, np.nan)
    ranks.loc[~(covered >= _MIN_COVERAGE)] = np.nan
    return ranks


def _without(panel, symbol: str):
    """The panel without one column: the hedge is traded, never ranked or shorted itself."""
    def drop(frame):
        return frame.drop(columns=[symbol], errors="ignore") if frame is not None else None

    return dataclasses.replace(
        panel, close=drop(panel.close), open_=drop(panel.open_), high=drop(panel.high),
        low=drop(panel.low), volume=drop(panel.volume), rth=drop(panel.rth),
        features={k: drop(v) for k, v in (panel.features or {}).items()})


def _live_rows(close: pd.DataFrame) -> pd.Series:
    """True on a trading day: some name is priced, and at least :data:`_LIVE_FRACTION` of the
    median count of the :data:`_LIVE_LOOKBACK` rows before it (known at the row: causal).

    The lab's v1 panels carried three 2026 US-holiday rows on which only a Swiss listing
    traded. Read as a session, such a row is a day on which every long turns ineligible and
    every short loses its price, and it blanks every rolling window that requires full
    history. The lab v2 panels hold NYSE sessions only, so there this rule skips nothing; it
    stays as a safeguard for any panel built from a union of calendars.
    """
    count = close.notna().sum(axis=1)
    typical = count.shift(1).rolling(_LIVE_LOOKBACK, min_periods=1).median()
    return (count > 0) & ~(count < _LIVE_FRACTION * typical)


def _rows(panel, keep: pd.Series):
    """The panel on the rows where ``keep`` is True (itself when every row is kept)."""
    if bool(keep.all()):
        return panel
    mask = keep.to_numpy()

    def cut(frame):
        return frame.loc[mask] if frame is not None else None

    return dataclasses.replace(
        panel, close=cut(panel.close), open_=cut(panel.open_), high=cut(panel.high),
        low=cut(panel.low), volume=cut(panel.volume), rth=cut(panel.rth),
        features={k: cut(v.reindex(panel.close.index)) for k, v in
                  (panel.features or {}).items()})


def _on_priced_opens(weights: pd.DataFrame, open_: pd.DataFrame) -> pd.DataFrame:
    """Each target change moved to the name's next bar with an open price.

    The engine drops an order whose fill price is NaN and, trading a name only when its
    target changes, never sends it again: the position (or its absence) would stick.
    """
    priced = open_.reindex(index=weights.index, columns=weights.columns).notna()
    return weights.where(priced).ffill().fillna(0.0)


def _book(parts: dict, *, q_in: float, q_out: float, slots: int, max_hold: int,
          max_shorts: int, atr_mult: float, short_hold: int = 0, short_stop: float = 0.0):
    """The long and the short side of one slot book, bar by bar (decided at the close).

    Per bar: (1) long exits — no score (ineligible), a rank below ``q_out``, ``max_hold``
    reached (the clock restarts while the name ranks >= ``q_in``); (2) each short's chandelier
    is trailed on the bar (lowest low, current ATR, never loosened) and the short exits on a
    close above it, an opposite start, a close above the prior ``exit_window``-bar high (when
    on), a close ``short_stop`` above the entry close (when on), becoming too crowded, or
    (with ``short_hold``) ``short_hold`` bars without a fresh start on the name; (3) short
    starts take slots, most negative relative strength first, up to ``max_shorts`` — a free
    slot if there is one, else the weakest-ranked priced long's; (4) longs fill what is left,
    best rank first. A name held on one side is never a candidate for the other. A held name
    without a close on the bar is carried: no exit is decided for it until it is priced again,
    and it cannot be displaced (a name whose data ends is carried to the end).

    Returns:
        ``(longs, shorts, exits)``: boolean held frames at the deciding bar (before
        ``next_bar``) and a list of ``(side, symbol, opened, closed, cause)``.
    """
    score = parts["score"]
    s = score.to_numpy(dtype=float)
    c = parts["close"].to_numpy(dtype=float)
    lo = parts["low"].to_numpy(dtype=float)
    a = parts["atr"].to_numpy(dtype=float)
    rel = parts["rel"].to_numpy(dtype=float)
    ss = parts["short_start"].to_numpy(dtype=bool)
    ls = parts["long_start"].to_numpy(dtype=bool)
    cover = parts["cover"].to_numpy(dtype=bool)
    ok = parts["short_ok"].to_numpy(dtype=bool)
    crowded = parts["crowded"].to_numpy(dtype=bool)
    n, m = s.shape
    held_l, held_s = np.zeros((n, m), bool), np.zeros((n, m), bool)
    longs: dict[int, list] = {}             # column -> [opened, clock]
    shorts: dict[int, list] = {}            # column -> [opened, entry close, best low, stop,
    #                                         clock: the bar of the name's last start]
    exits = []
    eps = 1e-9                              # rank thresholds are exact fractions
    names, dates = score.columns, score.index
    for t in range(n):
        row = s[t]
        unpriced = np.isnan(c[t])                   # carried today, decided when priced
        for j in list(longs):
            if unpriced[j]:
                continue
            if row[j] >= q_in - eps:
                longs[j][1] = t
            cause = ("ineligible" if np.isnan(row[j]) else "band" if row[j] < q_out - eps
                     else "max_hold" if t - longs[j][1] >= max_hold else None)
            if cause:
                exits.append(("long", names[j], dates[longs[j][0]], dates[t], cause))
                del longs[j]
        for j in list(shorts):
            pos = shorts[j]
            pos[2] = np.fmin(pos[2], lo[t, j])
            pos[3] = np.fmin(pos[3], pos[2] + atr_mult * a[t, j])
            if ss[t, j]:
                pos[4] = t                  # a fresh breakdown: the short clock restarts
            if unpriced[j]:
                continue
            cause = ("stop" if c[t, j] > pos[3]
                     else "opposite" if ls[t, j] else "reversal" if cover[t, j]
                     else "squeeze" if short_stop and c[t, j] >= pos[1] * (1 + short_stop)
                     else "crowded" if crowded[t, j]
                     else "short_hold" if short_hold and t - pos[4] >= short_hold else None)
            if cause:
                exits.append(("short", names[j], dates[pos[0]], dates[t], cause))
                del shorts[j]
        room = max_shorts - len(shorts)
        if room > 0:
            cand = ss[t] & ok[t]
            cand[list(longs)] = False
            cand[list(shorts)] = False
            idx = np.flatnonzero(cand)
            for j in idx[np.argsort(rel[t, idx], kind="stable")][:room]:
                if len(longs) + len(shorts) >= slots:
                    pool = [k for k in longs if not unpriced[k]]
                    if not pool:
                        break
                    weakest = min(pool, key=lambda k: row[k])
                    exits.append(("long", names[weakest], dates[longs[weakest][0]], dates[t],
                                  "displaced"))
                    del longs[weakest]
                shorts[j] = [t, c[t, j], c[t, j], c[t, j] + atr_mult * a[t, j], t]
        free = slots - len(longs) - len(shorts)
        if free > 0:
            cand = row >= q_in - eps
            cand[list(longs)] = False
            cand[list(shorts)] = False
            idx = np.flatnonzero(cand)
            for j in idx[np.argsort(-row[idx], kind="stable")][:free]:
                longs[j] = [t, t]
        held_l[t, list(longs)] = True
        held_s[t, list(shorts)] = True

    def frame(arr):
        return pd.DataFrame(arr, index=dates, columns=names)

    return frame(held_l), frame(held_s), exits


def book(panel, **params) -> dict:
    """The held frames, the exits and the slot counts of one run (missing params default to
    ``signals()``'s), for the research scripts and the tests.

    The held frames are on the panel's index (a skipped row repeats the trading day before
    it, as ``signals()`` holds the book there); ``parts`` are on the trading days only.
    """
    p = {**_DEFAULTS, **params}
    core = _without(panel, p["hedge"]) if p["hedge"] else panel
    parts = compose(_rows(core, _live_rows(core.close)), **p)
    longs, shorts, exits = _book(parts, **{k: p[k] for k in (
        "q_in", "q_out", "slots", "max_hold", "max_shorts", "atr_mult", "short_hold",
        "short_stop")})

    def carry(held):
        return held.astype(float).reindex(core.close.index).ffill().fillna(0.0).astype(bool)

    longs, shorts = carry(longs), carry(shorts)
    return {"longs": longs, "shorts": shorts, "exits": exits, "parts": parts,
            "n_long": longs.sum(axis=1), "n_short": shorts.sum(axis=1)}
