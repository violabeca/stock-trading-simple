"""earnings_drift — post-earnings-announcement drift as a dollar-neutral long/short book.

Stocks keep drifting in the direction of their earnings news for weeks after the report
(Ball & Brown 1968; Bernard & Thomas 1989), and the market's own reaction measures that news
more completely than the EPS surprise does (Brandt, Kishore, Santa-Clara & Venkatachalam 2008).
Every usable earnings event gets a score from two ranks:

1. **SUE**, the price-scaled surprise: (EPS actual − estimate) / the close before the report
   session (``surprise: price``; ``pct`` reads yfinance's % surprise instead);
2. **EAR**, the earnings-announcement return: the stock's return from the close before the
   report session to the close ``ear_end`` sessions after it, minus the equal-weight market's
   (the panel's names) over the same days.

Each component is ranked against the events of the trailing ``trail`` bars, the two ranks are
averaged (``w_ear`` = EAR's share), and that score is ranked against the trailing scores again,
so every threshold is point in time.

**Coarse surprises are not used** (``cent_max``). yfinance stores EPS on TODAY's split basis,
rounded to the cent, so a name that split later has tiny, coarse EPS before the split (NVDA 2016:
actual = estimate = 0.01). The precision of its surprise then depends on its future splits. An
event whose cent quantum (0.01 / close before the report, or 0.01 / |estimate| for ``pct``)
exceeds ``cent_max`` × the trailing P20-P80 spread of the surprise ranks on its reaction alone
(RESEARCH_LOG "Review fixes"), and so does one whose trailing window is still too small to judge
(round 2, V0a).

The top ``q`` of the events go long, the bottom ``q``
short, each held ``hold`` sessions in one of ``slots`` equal slots a side: the most extreme
events take the free slots first, and an event that finds none is dropped (it never waits).
A held name's next scored event re-decides it (the same side restarts its clock, anything else
closes it). Shorts pay a yearly borrow fee tiered by days-to-cover and are never held above
``short_max_dtc`` days (hard to borrow). Days-to-cover, not short interest ÷ shares: the lab's
volumes are split-adjusted backward while SEC share counts are as reported, so that ratio is
inflated before every forward split — knowledge of future splits (RESEARCH_LOG §0).

**Timing** (the lab's ``earn_*`` frames): every value sits on its USABLE bar ``u``, the second
session after the report session ``p = u − 2``. The event is decided at u's close and traded at
u+1's open (the weights are shifted one bar); the EAR window ends at ``p + ear_end <= u``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

#: Sessions between the report session (``earn_day``) and the usable bar (``earn_flag``).
REPORT_LAG = 2


def signals(panel, *, surprise: str = "price", w_ear: float = 0.5, ear_end: int = 1,
            trail: int = 252, min_events: int = 100, q: float = 0.2, hold: int = 60,
            slots: int = 30, gross: float = 0.98, long_share: float = 0.5,
            sides: str = "both", dtc_mid: float = 7.0, dtc_high: float = 10.0,
            short_max_dtc: float = 15.0, borrow_gc: float = 0.005, borrow_mid: float = 0.03,
            borrow_high: float = 0.1, min_price: float = 5.0, retrim_k: float = 2.0,
            crowd: str = "all", cent_max: float = 0.25) -> dict:
    """Target weights (long > 0, short < 0), every name's borrow rate, and the long cap.

    Args:
        panel: Daily panel with ``features`` ``earn_flag``, ``earn_actual``, ``earn_estimate``
            (``earn_surprise`` for ``surprise="pct"``) on each event's usable bar, and
            ``short_interest`` (days-to-cover) for the borrow tiers (optional: none = every
            short at ``borrow_gc``). No earnings frames: an empty book.
        surprise: ``"price"`` ((actual − estimate) / close before the report) or ``"pct"``.
        w_ear: EAR's share of the score (0 = the surprise alone, 1 = the reaction alone).
        ear_end: The reaction window ends at the close this many sessions after the report
            session (0, 1 or 2; it starts at the close before it).
        trail: Bars of past events each rank is taken against.
        min_events: Scored events the trailing window needs before it ranks anything.
        q: The top / bottom share of the trailing scores that goes long / short.
        hold: Sessions a position is held.
        slots: Positions per side.
        gross: Long plus short exposure at full slots (<= 1: shorts lock their proceeds).
        long_share: The long side's share of ``gross`` (0.5 = dollar-neutral).
        sides: ``"both"``, ``"long"`` or ``"short"``.
        dtc_mid, dtc_high: Days-to-cover above which a short pays ``borrow_mid`` /
            ``borrow_high`` a year (``borrow_gc`` below, and where there is no data).
        short_max_dtc: Days-to-cover above which a name is not shorted (a held short is
            bought back).
        borrow_gc, borrow_mid, borrow_high: Yearly borrow rates of the three tiers.
        min_price: Minimum close at the decision bar.
        retrim_k: A long that grows to ``k ×`` its slot is trimmed back (engine cap).
        crowd: ``"all"``; ``"crowded"`` / ``"uncrowded"`` open a position (either side) only
            for a name whose days-to-cover at the decision bar is above / at most the median of
            the panel's names with data (no data = no evidence: allowed). Re-decisions of held
            names are unchanged.
        cent_max: The surprise is used only where one cent of EPS is at most this share of the
            trailing P20-P80 spread of the surprise measure (:func:`events`); inf = no rule.

    Returns:
        ``{"weights", "borrow", "retrim_k"}`` for the engine's portfolio mode.
    """
    close = panel.close
    n, m = close.shape
    borrow, ok_short = crowding(panel, dtc_mid=dtc_mid, dtc_high=dtc_high,
                                short_max_dtc=short_max_dtc, borrow_gc=borrow_gc,
                                borrow_mid=borrow_mid, borrow_high=borrow_high)
    ev = events(panel, surprise=surprise, w_ear=w_ear, ear_end=ear_end, trail=trail,
                min_events=min_events, min_price=min_price, cent_max=cent_max)
    longs, shorts = _select(ev["bar"].to_numpy(), ev["col"].to_numpy(), ev["pct"].to_numpy(),
                            n, m, q=q, hold=hold, slots=slots, sides=sides,
                            ok_short=ok_short.to_numpy(), allow=_crowd_filter(panel, ev, crowd))
    size_l, size_s = gross * long_share / slots, gross * (1 - long_share) / slots
    held = longs.astype(float) * size_l - shorts.astype(float) * size_s
    weights = pd.DataFrame(held, index=close.index, columns=close.columns)
    return {"weights": weights.shift(1, fill_value=0.0), "borrow": borrow,
            "retrim_k": retrim_k}


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


def valid(params: dict) -> bool:
    """Reject parameter sets the book cannot express (missing keys read as ``signals()``'s
    defaults, so a partial dict — a search point alone — is judged too)."""
    p = {**_DEFAULTS, **params}
    return (p["surprise"] in ("price", "pct") and 0.0 <= p["w_ear"] <= 1.0
            and p["ear_end"] in (0, 1, 2) and p["trail"] >= 21 and p["min_events"] >= 1
            and 0.0 < p["q"] < 0.5 and p["hold"] >= 1 and p["slots"] >= 1
            and 0.0 < p["gross"] <= 1.0 and 0.0 < p["long_share"] < 1.0
            and p["sides"] in ("both", "long", "short")
            and 0.0 < p["dtc_mid"] < p["dtc_high"] <= p["short_max_dtc"]
            and min(p["borrow_gc"], p["borrow_mid"], p["borrow_high"]) >= 0.0
            and p["min_price"] >= 0.0 and p["retrim_k"] > 1.0
            and p["crowd"] in ("all", "crowded", "uncrowded") and p["cent_max"] > 0.0)


def _crowd_filter(panel, ev: pd.DataFrame, crowd: str) -> np.ndarray:
    """Which events may open a position under ``crowd`` (see :func:`signals`).

    Args:
        panel: Daily panel; ``features["short_interest"]`` optional (none: all allowed).
        ev: The scored events (:func:`events`).
        crowd: ``"all"``, ``"crowded"`` or ``"uncrowded"``.

    Returns:
        A boolean array aligned with ``ev``.
    """
    dtc = (panel.features or {}).get("short_interest")
    if crowd == "all" or dtc is None or ev.empty:
        return np.ones(len(ev), bool)
    dtc = dtc.reindex_like(panel.close)
    median = dtc.median(axis=1).to_numpy(dtype=float)
    bar, col = ev["bar"].to_numpy(), ev["col"].to_numpy()
    d, med = dtc.to_numpy(dtype=float)[bar, col], median[bar]
    known = np.isfinite(d) & np.isfinite(med)
    above = d > med
    return ~known | (above if crowd == "crowded" else ~above)


def crowding(panel, *, dtc_mid: float, dtc_high: float, short_max_dtc: float,
             borrow_gc: float, borrow_mid: float, borrow_high: float):
    """Each name's yearly borrow rate and whether it may be shorted, per bar.

    Args:
        panel: Daily panel; ``features["short_interest"]`` (days-to-cover) optional.
        dtc_mid, dtc_high, short_max_dtc: Days-to-cover tiers and the no-short cap.
        borrow_gc, borrow_mid, borrow_high: Yearly rates below ``dtc_mid``, above it, and
            above ``dtc_high``.

    Returns:
        ``(borrow, shortable)``: a rate frame (``borrow_gc`` where there is no data) and a
        boolean frame (True where there is no data: no evidence of crowding).
    """
    close = panel.close
    dtc = (panel.features or {}).get("short_interest")
    if dtc is None:
        dtc = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    d = dtc.reindex_like(close).to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        rate = np.where(d > dtc_high, borrow_high, np.where(d > dtc_mid, borrow_mid, borrow_gc))
        shortable = ~(d > short_max_dtc)
    frame = lambda a: pd.DataFrame(a, index=close.index, columns=close.columns)  # noqa: E731
    return frame(rate), frame(shortable)


def events(panel, *, surprise: str = "price", w_ear: float = 0.5, ear_end: int = 1,
           trail: int = 252, min_events: int = 100, min_price: float = 5.0,
           cent_max: float = 0.25) -> pd.DataFrame:
    """Every tradable earnings event of the panel with its components, ranks and score.

    Pure and point in time: each rank is taken among the events whose usable bar lies in the
    ``trail`` bars up to and including the event's own (:func:`_trailing_pct`).

    **Coarse surprises.** The EPS are whole cents on today's split basis, so one cent can be a
    large share of a surprise (a name that split later has tiny early EPS). The cent quantum of
    an event's surprise, 0.01 / close(p − 1) under ``price`` and 0.01 / |estimate| under
    ``pct``, is compared with the P20-P80 spread of that measure among the trailing window's
    events (:func:`_trailing_spread`). Where it exceeds ``cent_max`` × that spread, the surprise
    is set to NaN: the event leaves the surprise's reference distribution and ranks on its
    reaction alone (``coarse`` = True). A window too small to judge (a panel's first
    ``min_events`` surprises) counts as coarse too, so no unjudged surprise enters a later
    event's reference distribution; ``cent_max`` = inf switches the rule off.

    Args:
        panel: Daily panel with the ``earn_*`` features (none: an empty table).
        surprise, w_ear, ear_end, trail, min_events, min_price, cent_max: As :func:`signals`.

    Returns:
        One row per event, sorted by bar: ``bar`` (usable-bar position), ``col`` (column
        position), ``date``, ``symbol``, ``sue`` (the raw surprise measure), ``coarse``,
        ``ear``, ``r_sue``, ``r_ear`` (component ranks), ``score`` (their weighted mean) and
        ``pct`` (the score's trailing rank; NaN while the window holds fewer than
        ``min_events`` scored events).
    """
    feats = panel.features or {}
    close = panel.close
    cols = ["bar", "col", "date", "symbol", "sue", "coarse", "ear", "r_sue", "r_ear", "score",
            "pct"]
    flag = feats.get("earn_flag")
    if flag is None:
        return pd.DataFrame(columns=cols)
    c = close.to_numpy(dtype=float)
    u, j = np.nonzero(flag.reindex_like(close).notna().to_numpy())   # sorted by bar
    p = u - REPORT_LAG
    keep = p >= 1
    u, j, p = u[keep], j[keep], p[keep]
    before, now = c[p - 1, j], c[u, j]
    with np.errstate(invalid="ignore"):
        keep = (before > 0) & (now >= min_price)
    u, j, p, before = u[keep], j[keep], p[keep], before[keep]

    ret = close.pct_change(fill_method=None)
    market = (1.0 + ret.mean(axis=1).fillna(0.0)).cumprod().to_numpy()
    end = p + ear_end
    ear = c[end, j] / before - market[end] / market[p - 1]
    nan = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    estimate = feats.get("earn_estimate", nan).reindex_like(close).to_numpy(dtype=float)[u, j]
    with np.errstate(divide="ignore", invalid="ignore"):
        if surprise == "price":
            actual = feats.get("earn_actual", nan).reindex_like(close).to_numpy(dtype=float)
            sue = (actual[u, j] - estimate) / before
            quantum = 0.01 / before
        else:
            sue = feats["earn_surprise"].reindex_like(close).to_numpy(dtype=float)[u, j]
            quantum = 0.01 / np.abs(estimate)
        spread = _trailing_spread(u, sue, trail, min_events)      # NaN: too few to judge
        judged = np.isfinite(spread)          # unjudged: coarse, unless the rule is off (inf)
        coarse = np.isfinite(sue) & np.where(judged, quantum > cent_max * spread,
                                             np.isfinite(cent_max))
    r_sue = _trailing_pct(u, np.where(coarse, np.nan, sue), trail, min_events)
    r_ear = _trailing_pct(u, ear, trail, min_events)
    w = np.where(np.isnan(r_sue), 0.0, 1.0 - w_ear), np.where(np.isnan(r_ear), 0.0, w_ear)
    total = w[0] + w[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        score = (w[0] * np.nan_to_num(r_sue) + w[1] * np.nan_to_num(r_ear)) / total
    score[total <= 0] = np.nan
    pct = _trailing_pct(u, score, trail, min_events)
    return pd.DataFrame({"bar": u, "col": j, "date": close.index[u],
                         "symbol": close.columns[j], "sue": sue, "coarse": coarse, "ear": ear,
                         "r_sue": r_sue, "r_ear": r_ear, "score": score, "pct": pct})


def _trailing_spread(bar: np.ndarray, x: np.ndarray, trail: int, min_n: int) -> np.ndarray:
    """Each event's P20-P80 spread of the finite values whose bar lies in ``(bar - trail,
    bar]`` (the window of :func:`_trailing_pct`); NaN for a window under ``min_n``.

    Args:
        bar: Each event's bar position, sorted ascending.
        x: Each event's value.
        trail: Window length in bars.
        min_n: Fewest finite values a window must hold.

    Returns:
        The spreads, aligned with ``x``.
    """
    out = np.full(len(x), np.nan)
    for t in np.unique(bar):
        lo = np.searchsorted(bar, t - trail, "right")
        first, hi = np.searchsorted(bar, t, "left"), np.searchsorted(bar, t, "right")
        window = x[lo:hi]
        window = window[np.isfinite(window)]
        if len(window) < min_n:
            continue
        p20, p80 = np.quantile(window, [0.2, 0.8])
        out[first:hi] = p80 - p20
    return out


def _trailing_pct(bar: np.ndarray, x: np.ndarray, trail: int, min_n: int) -> np.ndarray:
    """Each value's percentile rank (average ties, 1/n .. 1) among the non-NaN values whose
    bar lies in ``(bar - trail, bar]``; NaN for a NaN value or a window under ``min_n``.

    Args:
        bar: Each event's bar position, sorted ascending.
        x: Each event's value.
        trail: Window length in bars.
        min_n: Fewest non-NaN values a window must hold.

    Returns:
        The ranks, aligned with ``x``.
    """
    out = np.full(len(x), np.nan)
    for t in np.unique(bar):
        lo = np.searchsorted(bar, t - trail, "right")
        first, hi = np.searchsorted(bar, t, "left"), np.searchsorted(bar, t, "right")
        window = x[lo:hi]
        window = np.sort(window[~np.isnan(window)])
        if len(window) < min_n:
            continue
        today = x[first:hi]
        less = np.searchsorted(window, today, "left")
        upto = np.searchsorted(window, today, "right")
        rank = (less + 1 + upto) / 2.0 / len(window)
        out[first:hi] = np.where(np.isnan(today), np.nan, rank)
    return out


def _select(bar: np.ndarray, col: np.ndarray, pct: np.ndarray, n: int, m: int, *, q: float,
            hold: int, slots: int, sides: str, ok_short: np.ndarray,
            allow: np.ndarray | None = None):
    """The long and the short book, bar by bar (decided at the close, traded next bar).

    On each bar: a held name with a new scored event is re-decided (the same side restarts its
    clock, anything else closes it); positions held ``hold`` sessions close, and so does a
    short that became too crowded to borrow; then each side fills its free slots from the
    bar's qualifying events, the most extreme ranks first.

    Args:
        bar, col, pct: The scored events (:func:`events`), sorted by bar.
        n, m: Panel shape.
        q, hold, slots, sides: As :func:`signals`.
        ok_short: ``(n, m)`` boolean: may the name be shorted on that bar.
        allow: Per event, may it open a position (default: all may).

    Returns:
        ``(longs, shorts)``: boolean ``(n, m)`` arrays of the names held at each bar's close.
    """
    held = {+1: {}, -1: {}}                         # column -> bar its clock started
    out = {+1: np.zeros((n, m), bool), -1: np.zeros((n, m), bool)}
    use = {"both": (+1, -1), "long": (+1,), "short": (-1,)}[sides]
    eps = 1e-9

    def qualifies(side, x):
        return x >= 1.0 - q - eps if side > 0 else x <= q + eps

    allow = np.ones(len(bar), bool) if allow is None else np.asarray(allow, bool)
    starts = np.searchsorted(bar, np.arange(n + 1), "left")
    for t in range(n):
        a, b = starts[t], starts[t + 1]
        for k in range(a, b):                       # a held name reports again
            j, x = col[k], pct[k]
            if np.isnan(x):
                continue
            for side in use:
                if j in held[side]:
                    if qualifies(side, x):
                        held[side][j] = t
                    else:
                        del held[side][j]
        for side in use:                            # time and crowding exits
            for j, opened in list(held[side].items()):
                if t - opened >= hold or (side < 0 and not ok_short[t, j]):
                    del held[side][j]
        for side in use:                            # entries, most extreme first
            free = slots - len(held[side])
            if free <= 0 or a == b:
                continue
            x, c = pct[a:b], col[a:b]
            with np.errstate(invalid="ignore"):
                cand = qualifies(side, x) & ~np.isnan(x) & allow[a:b]
            cand &= np.array([j not in held[+1] and j not in held[-1] for j in c])
            if side < 0:
                cand &= ok_short[t, c]
            idx = np.flatnonzero(cand)
            order = idx[np.argsort(-x[idx] if side > 0 else x[idx], kind="stable")]
            for k in order[:free]:
                held[side][c[k]] = t
        for side in use:
            out[side][t, list(held[side])] = True
    return out[+1], out[-1]
