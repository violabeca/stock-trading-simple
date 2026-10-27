"""resid_momentum — crash-protected residual momentum, long/short (STRATEGY_OVERVIEW.md §6, #2).

Rank every name on its 12-1 month return AFTER removing its market exposure, divided by its
residual volatility (Blitz, Huij & Martens 2011; Gutierrez & Prinsky 2007); long the top of the
ranking, short the bottom; shrink the whole book when its own trailing volatility jumps
(Barroso & Santa-Clara 2015).

* **Residual return** ``e = r − β·m``: ``m`` is the equal-weight daily return of the panel's
  names that closed at or above ``min_price`` the bar before; ``β`` is each name's OLS slope on
  ``m`` over the trailing ``beta_window`` bars, known at the bar's close.
* **Score** ``"residual"``: ``Σe / sd(e)`` over the formation bars ``t − lookback + 1 ..
  t − skip``, all with the bar-t beta. ``"raw"`` (the control arm): the total return
  ``P(t − skip) / P(t − lookback) − 1``. ``factors="market+sector"`` adds a second regressor,
  the equal-weight return of the name's OTHER sector members, so the residual also drops the
  sector's own trend (industry momentum, Moskowitz & Grinblatt 1999): ``e = r − β_m·m − β_s·s``.
* **Decisions** only on the last trading bar of each month; the rank is held in between, so a
  name trades only when it enters or leaves a band, or when the book scale changes. Long enters
  at rank ``>= q_in`` and leaves below ``q_out``; the short side mirrors it (enter ``<= 1 −
  q_in``, leave above ``1 − q_out``); ``slots`` names a side, best ranks first.
* **Execution**: every order at the first open after the decision (``phased=False``, the
  default). ``phased=True`` trades what closes or shrinks at the first open and what opens or
  grows at the second: a workaround for the SHARED engine only, whose ``call_seq="auto"``
  places a bar's new shorts before the covers that fund them while short proceeds are locked,
  so a new short is rejected or part-filled and never re-sent (the engine re-sends a target
  only when it changes). The proposals lab's runner (v2) orders covers first, so the lab runs
  unphased (RESEARCH_LOG.md "Review fixes" R0, "Round 2" R2.1). Gross is 0.90, not 0.98:
  vectorbt's free cash leaves out a winning short's gain twice, so at 0.98 a full book's last
  long buys still run out of cash.
* **Sizing**: equal slots, ``gross × long_share / slots`` long and ``gross × (1 − long_share) /
  slots`` short (dollar-neutral at 0.5), all times the **book scale** ``min(1, vol_target / σ)``,
  ``σ`` the trailing ``vol_window``-bar realized volatility of the UNSCALED book (simulated from
  its own target weights and close-to-close returns), decided at month ends only, rounded to
  ``vol_step`` and floored at ``min_scale``. ``vol_target`` 0 turns the scaling off.
  ``sizing="beta"`` (informed_flow_ls_p1's rule) scales a slot AT ENTRY by the median beta ÷
  the name's beta (clipped to ``1 / max_size .. max_size``) and freezes it while held, so the
  long and the short side carry about the same market risk.
* **Borrow**: informed_flow_ls_p1's tiers by short interest / shares outstanding (``borrow_gc``
  where there is no short-interest data); a name above ``short_max_si`` is never shorted, and a
  held short above it is covered at the next month end.

Holiday stub rows (a bar where fewer than half the usual names print) are skipped: no return,
no decision, no order — a decision due on such a row moves to the next full row.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

#: Ranking choices: the residual score (the design) and raw 12-1 momentum (the control).
SCORES = ("residual", "raw")
#: Regressors the residual is taken against.
FACTORS = ("market", "market+sector")
#: Slot sizing rules.
SIZINGS = ("equal", "beta")


def signals(panel, *, score: str = "residual", lookback: int = 252, skip: int = 21,
            beta_window: int = 252, factors: str = "market", min_sector: int = 5,
            min_obs: float = 0.8, q_in: float = 0.9,
            q_out: float = 0.8, slots: int = 20, gross: float = 0.9, long_share: float = 0.5,
            vol_target: float = 0.10, vol_window: int = 126, vol_step: float = 0.25,
            min_scale: float = 0.25, min_price: float = 5.0, min_history: int = 252,
            short_max_si: float = 0.3, si_mid: float = 0.1, si_high: float = 0.2,
            borrow_gc: float = 0.005, borrow_mid: float = 0.03, borrow_high: float = 0.1,
            retrim_k: float = 2.0, sizing: str = "equal", max_size: float = 1.5,
            phased: bool = False, **_) -> dict:
    """Target weights (long > 0, short < 0), every name's borrow rate, and the long cap.

    Args:
        panel: Daily panel; ``features`` ``short_interest`` (days-to-cover) and
            ``fund_shares`` are optional (absent: every short borrows at ``borrow_gc``).
        score: ``"residual"`` or ``"raw"`` (see the module docstring).
        lookback, skip: The formation window is bars ``t − lookback + 1 .. t − skip``.
        beta_window: Bars of the trailing beta to the equal-weight market.
        factors: ``"market"`` or ``"market+sector"`` (needs the ``sector`` feature).
        min_sector: A name's sector regressor needs at least this many other priced members
            on the bar (else the market stands in for it there).
        min_obs: Share of the formation and beta windows that must be observed.
        q_in, q_out: Long entry / exit ranks; the short side mirrors them (``1 − q``).
        slots: Positions per side.
        gross: Long plus short exposure at full scale (<= 1: the engine locks short proceeds;
            0.90 leaves the cash a winning short's locked gain takes, see the module docstring).
        long_share: The long side's share of ``gross`` (0.5 = dollar-neutral).
        vol_target: Yearly volatility the book is scaled down to (0 = no scaling).
        vol_window: Bars of the unscaled book's realized volatility.
        vol_step: The scale is rounded to this step, so it changes (and re-trades) rarely.
        min_scale: The smallest scale (the book is never switched off entirely).
        min_price: Minimum close to be ranked (and to count in the market return).
        min_history: Minimum bars of history to be ranked.
        short_max_si: Short interest / shares outstanding above which a name is not shorted.
        si_mid, si_high: Crowding tiers of the borrow rate.
        borrow_gc, borrow_mid, borrow_high: Yearly borrow rates below ``si_mid``, up to
            ``si_high``, and above it.
        retrim_k: A long is trimmed back to its slot at ``k ×`` it (engine cap; 0 = off).
        sizing: ``"equal"`` slots or ``"beta"`` (median beta ÷ beta at entry, frozen).
        max_size: ``"beta"`` sizing's bounds: a slot is scaled between ``1 / max_size`` and
            ``max_size`` of its equal size.
        phased: Close and shrink at the first open after a decision, open and grow at the
            second (for the shared engine only); ``False``: everything at the first open (see
            :func:`_execute`).

    Returns:
        ``{"weights", "borrow"}`` (+ ``"retrim_k"`` when on) for the engine's portfolio mode.
    """
    parts = compose(panel, score=score, lookback=lookback, skip=skip, beta_window=beta_window,
                    factors=factors, min_sector=min_sector, min_obs=min_obs,
                    min_price=min_price, min_history=min_history,
                    short_max_si=short_max_si, si_mid=si_mid, si_high=si_high,
                    borrow_gc=borrow_gc, borrow_mid=borrow_mid, borrow_high=borrow_high)
    longs, shorts = _select(parts["rank"], parts["shortable"], parts["rebalance"],
                            panel.close.notna(), q_in=q_in, q_out=q_out, slots=slots)
    size_l, size_s = gross * long_share / slots, gross * (1 - long_share) / slots
    if sizing == "beta":
        decided = _beta_sizes(longs, shorts, parts["beta"], parts["eligible"],
                              parts["rebalance"], size_l=size_l, size_s=size_s, gross=gross,
                              max_size=max_size)
    else:
        decided = longs.astype(float) * size_l - shorts.astype(float) * size_s
    scale = book_scale(_execute(decided, parts["rows"], phased), parts["returns"], parts["rows"],
                       parts["rebalance"], vol_target=vol_target, vol_window=vol_window,
                       vol_step=vol_step, min_scale=min_scale)
    out = {"weights": _execute(decided.mul(scale, axis=0), parts["rows"], phased),
           "borrow": parts["borrow"]}
    if retrim_k > 0:
        out["retrim_k"] = retrim_k
    return out


_DEFAULTS = {k: v.default for k, v in inspect.signature(signals).parameters.items()
             if v.default is not inspect.Parameter.empty}


def valid(params: dict) -> bool:
    """Reject parameter sets the book cannot express (missing keys read as ``signals()``'s
    defaults, so a partial dict — a search point alone — is judged too)."""
    p = {**_DEFAULTS, **params}
    return (p["score"] in SCORES and p["factors"] in FACTORS and p["min_sector"] >= 1
            and 0.5 < p["q_out"] < p["q_in"] < 1.0 and p["slots"] >= 1
            and 0 <= p["skip"] < p["lookback"] and p["lookback"] - p["skip"] >= 21
            and p["beta_window"] >= 21 and 0.0 < p["min_obs"] <= 1.0
            and 0.0 < p["gross"] <= 1.0 and 0.0 < p["long_share"] < 1.0
            and p["vol_target"] >= 0 and p["vol_window"] >= 10 and 0.0 < p["vol_step"] <= 1.0
            and 0.0 < p["min_scale"] <= 1.0 and p["min_history"] >= 1
            and 0.0 < p["si_mid"] < p["si_high"] <= p["short_max_si"]
            and (p["retrim_k"] == 0 or p["retrim_k"] > 1)
            and p["sizing"] in SIZINGS and p["max_size"] >= 1.0
            and p["phased"] in (True, False))


def compose(panel, *, score: str = "residual", lookback: int = 252, skip: int = 21,
            beta_window: int = 252, factors: str = "market", min_sector: int = 5,
            min_obs: float = 0.8, min_price: float = 5.0,
            min_history: int = 252, short_max_si: float = 0.3, si_mid: float = 0.1,
            si_high: float = 0.2, borrow_gc: float = 0.005, borrow_mid: float = 0.03,
            borrow_high: float = 0.1) -> dict:
    """Everything the book reads, bar by bar (pure; also what the research script inspects).

    Returns:
        ``rows`` (full rows: not a holiday stub), ``returns`` (close-to-close, from the last
        full row), ``market`` (equal-weight return), ``beta``, ``value`` (the score, high =
        long), ``beta_sector`` (the sector regressor's slope; ``None`` for ``"market"``),
        ``eligible``, ``rebalance`` (month-end decision rows), ``rank`` (percentile
        rank among the eligible names on rebalance rows, NaN elsewhere), ``si_share``,
        ``shortable`` and ``borrow`` (yearly rate per name and bar).
    """
    close, feats = panel.close, panel.features or {}
    index, columns = close.index, close.columns
    rows = _full_rows(close)
    c = close.where(rows, axis=0)
    prev = c.ffill().shift(1)
    ret = (c / prev - 1.0).to_numpy(dtype=float)
    priced = (prev >= min_price).to_numpy() & np.isfinite(ret)
    count = priced.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        market = np.where(count >= 5, np.where(priced, ret, 0.0).sum(axis=1)
                          / np.maximum(count, 1), np.nan)
    obs = np.isfinite(ret) & np.isfinite(market)[:, None]
    r0 = np.where(obs, ret, 0.0)
    m0 = np.where(obs, market[:, None], 0.0)

    n_b, s_r, s_m = _wsum(obs, beta_window), _wsum(r0, beta_window), _wsum(m0, beta_window)
    s_rm, s_mm = _wsum(r0 * m0, beta_window), _wsum(m0 * m0, beta_window)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = (s_rm - s_r * s_m / n_b) / (n_b - 1)
        var = (s_mm - s_m * s_m / n_b) / (n_b - 1)
        beta = np.where((n_b >= min_obs * beta_window) & (var > 0), cov / var, np.nan)

    b_m, b_s = beta, None                   # the residual's slopes (market only by default)
    if factors == "market+sector":
        sec = feats.get("sector")
        codes = (np.full(len(columns), np.nan) if sec is None
                 else sec.reindex(columns=columns).iloc[-1].to_numpy(dtype=float))
        sector = _sector_returns(ret, priced, codes, min_sector)
        s0 = np.where(obs, np.where(np.isfinite(sector), sector, market[:, None]), 0.0)
        s_s, s_rs = _wsum(s0, beta_window), _wsum(r0 * s0, beta_window)
        s_ss, s_ms = _wsum(s0 * s0, beta_window), _wsum(m0 * s0, beta_window)
        with np.errstate(invalid="ignore", divide="ignore"):
            c_mm, c_ss = s_mm - s_m * s_m / n_b, s_ss - s_s * s_s / n_b
            c_ms, c_rm = s_ms - s_m * s_s / n_b, s_rm - s_r * s_m / n_b
            c_rs = s_rs - s_r * s_s / n_b
            det = c_mm * c_ss - c_ms * c_ms
            two = np.isfinite(beta) & (det > 1e-6 * c_mm * c_ss)   # else: market only
            b_m = np.where(two, (c_ss * c_rm - c_ms * c_rs) / det, beta)
            b_s = np.where(two, (c_mm * c_rs - c_ms * c_rm) / det, 0.0)

    span = lookback - skip
    n_f = _lag(_wsum(obs, span), skip)
    if score == "residual":
        f_r, f_m = _lag(_wsum(r0, span), skip), _lag(_wsum(m0, span), skip)
        f_rr, f_rm = _lag(_wsum(r0 * r0, span), skip), _lag(_wsum(r0 * m0, span), skip)
        f_mm = _lag(_wsum(m0 * m0, span), skip)
        with np.errstate(invalid="ignore", divide="ignore"):
            sum_e = f_r - b_m * f_m
            sum_e2 = f_rr - 2.0 * b_m * f_rm + b_m * b_m * f_mm
            if b_s is not None:
                f_s, f_rs = _lag(_wsum(s0, span), skip), _lag(_wsum(r0 * s0, span), skip)
                f_ss, f_ms = _lag(_wsum(s0 * s0, span), skip), _lag(_wsum(m0 * s0, span), skip)
                sum_e = sum_e - b_s * f_s
                sum_e2 = (sum_e2 + b_s * b_s * f_ss - 2.0 * b_s * f_rs
                          + 2.0 * b_m * b_s * f_ms)
            var_e = (sum_e2 - sum_e * sum_e / n_f) / (n_f - 1)
            value = np.where(var_e > 0, sum_e / np.sqrt(var_e), np.nan)
    else:
        filled = c.ffill().to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = _lag(filled, skip) / _lag(filled, lookback) - 1.0
    history = np.cumsum(np.isfinite(c.to_numpy(dtype=float)), axis=0)
    eligible = (rows.to_numpy()[:, None] & (c.to_numpy(dtype=float) >= min_price)
                & (history >= min_history) & (n_f >= min_obs * span)
                & np.isfinite(beta) & np.isfinite(value))
    value = np.where(eligible, value, np.nan)

    rebalance = _month_ends(index, rows)
    frame = lambda a: pd.DataFrame(a, index=index, columns=columns)  # noqa: E731
    rank = frame(np.nan)
    at = rebalance.to_numpy()
    rank.loc[at] = frame(value).loc[at].rank(axis=1, pct=True)

    nan = frame(np.nan)
    dtc = feats.get("short_interest", nan).reindex_like(close)
    shares = feats.get("fund_shares", nan).reindex_like(close)
    adv = panel.volume.rolling(20, min_periods=10).median()
    si_share = (dtc * adv / shares.where(shares > 0)).where(rows, axis=0)
    borrow = frame(np.where(si_share > si_high, borrow_high,
                            np.where(si_share > si_mid, borrow_mid, borrow_gc)))
    return {"rows": rows, "returns": frame(ret), "market": pd.Series(market, index=index),
            "beta": frame(beta), "value": frame(value),
            "beta_sector": None if b_s is None else frame(b_s), "eligible": frame(eligible),
            "rebalance": rebalance, "rank": rank, "si_share": si_share,
            "shortable": ~(si_share > short_max_si), "borrow": borrow}


def book_scale(executed: pd.DataFrame, returns: pd.DataFrame, rows: pd.Series,
               rebalance: pd.Series, *, vol_target: float, vol_window: int, vol_step: float,
               min_scale: float) -> pd.Series:
    """The book scale DECIDED at each bar (held from one month end to the next).

    ``executed`` is the unscaled book's weights at the fill bar; its daily return is
    ``Σ weight × close-to-close return`` (drift and costs ignored: an estimate of risk, not a
    P&L). The scale is ``min(1, vol_target / σ)`` with ``σ`` that return's trailing
    ``vol_window``-row standard deviation × √252, rounded to ``vol_step`` and floored at
    ``min_scale``; 1 while fewer than half the window has passed, and everywhere when
    ``vol_target`` is 0.
    """
    one = pd.Series(1.0, index=executed.index)
    if vol_target <= 0:
        return one
    book = (executed * returns.fillna(0.0)).sum(axis=1)[rows]
    sigma = book.rolling(vol_window, min_periods=max(vol_window // 2, 2)).std() * np.sqrt(252)
    with np.errstate(divide="ignore"):
        raw = (vol_target / sigma).clip(upper=1.0)
    scale = ((raw / vol_step).round() * vol_step).clip(lower=min_scale, upper=1.0)
    scale = scale.where(sigma > 0, 1.0).fillna(1.0)
    decided = scale.where(rebalance[rows]).ffill().fillna(1.0)
    return decided.reindex(executed.index).ffill().fillna(1.0)


def _select(rank: pd.DataFrame, shortable: pd.DataFrame, rebalance: pd.Series,
            priced: pd.DataFrame, *, q_in: float, q_out: float,
            slots: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The long and the short book, decided on rebalance rows and held until the next one.

    On a rebalance row, exits come first (a NaN rank = ineligible, a rank out of the band, and
    for a short also becoming too crowded to borrow), then each side fills its free slots with
    the best-ranked candidates not held on either side. A held name with no price on the row
    stays held: the engine cannot sell it (an order at a NaN price is dropped and the position
    frozen at its last value), so its slot must not be handed to another name.

    Returns:
        ``(longs, shorts)``: boolean held frames at the deciding bar (before execution).
    """
    r = rank.to_numpy(dtype=float)
    ok = shortable.reindex_like(rank).fillna(True).to_numpy(dtype=bool)
    stuck = ~priced.reindex_like(rank).fillna(False).to_numpy(dtype=bool)
    n, m = r.shape
    at = np.flatnonzero(rebalance.to_numpy())
    eps = 1e-9                                   # rank thresholds are exact fractions
    long_, short = np.zeros(m, bool), np.zeros(m, bool)
    states = np.zeros((len(at) + 1, 2, m), bool)  # state 0: nothing held yet
    for k, t in enumerate(at, start=1):
        row = r[t]
        long_ &= (row >= q_out - eps) | stuck[t]  # NaN compares False: ineligible -> out
        short &= ((row <= 1 - q_out + eps) & ok[t]) | stuck[t]
        for side, held in ((+1, long_), (-1, short)):
            free = slots - int(held.sum())
            if free <= 0:
                continue
            enter = (row >= q_in - eps) if side > 0 else ((row <= 1 - q_in + eps) & ok[t])
            idx = np.flatnonzero(enter & ~long_ & ~short)
            order = idx[np.argsort(-row[idx] if side > 0 else row[idx], kind="stable")]
            held[order[:free]] = True
        states[k, 0], states[k, 1] = long_, short
    which = np.searchsorted(at, np.arange(n), side="right")   # 0 = before the first decision
    frame = lambda a: pd.DataFrame(a, index=rank.index, columns=rank.columns)  # noqa: E731
    return frame(states[which, 0]), frame(states[which, 1])


def _sector_returns(ret: np.ndarray, priced: np.ndarray, codes: np.ndarray,
                    min_names: int) -> np.ndarray:
    """Each name's leave-one-out sector return: the equal-weight return of the OTHER priced
    names with its sector code on the bar (NaN with no code or fewer than ``min_names``).
    The lab's sector codes are constant per name, so the last row's code stands for every bar."""
    r = np.where(priced, ret, 0.0)
    out = np.full(ret.shape, np.nan)
    for k in np.unique(codes[np.isfinite(codes)]):
        cols = np.flatnonzero(codes == k)
        others = priced[:, cols].sum(axis=1, keepdims=True) - priced[:, cols]
        total = r[:, cols].sum(axis=1, keepdims=True) - r[:, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, cols] = np.where(others >= min_names, total / others, np.nan)
    return out


def _beta_sizes(longs: pd.DataFrame, shorts: pd.DataFrame, beta: pd.DataFrame,
                eligible: pd.DataFrame, rebalance: pd.Series, *, size_l: float, size_s: float,
                gross: float, max_size: float) -> pd.DataFrame:
    """Signed slot sizes decided on rebalance rows (informed_flow_ls_p1's ``"beta"`` sizing).

    A new position's slot is scaled by the bar's median beta (eligible names, beta > 0) ÷ its
    own beta, clipped to ``1 / max_size .. max_size`` (no beta: 1; beta <= 0: the least risk,
    ``1 / max_size``). A held position keeps its size (the engine trades a name only when its
    target changes), and a bar's new entries are scaled down together when they would lift the
    nominal gross above ``gross``. Sizes are held from one rebalance row to the next.
    """
    at = np.flatnonzero(rebalance.to_numpy())
    held_l, held_s = longs.to_numpy(dtype=bool), shorts.to_numpy(dtype=bool)
    b, ok = beta.to_numpy(dtype=float), eligible.to_numpy(dtype=bool)
    n, m = held_l.shape
    states = np.zeros((len(at) + 1, m))
    prev = np.zeros(m)
    for k, t in enumerate(at, start=1):
        side = held_l[t].astype(float) - held_s[t].astype(float)
        pos = ok[t] & (b[t] > 0)
        median = np.median(b[t][pos]) if pos.any() else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = median / b[t]
        ratio = np.clip(np.where(np.isfinite(ratio), ratio, 1.0), 1.0 / max_size, max_size)
        ratio[b[t] <= 0] = 1.0 / max_size
        keep = (side != 0) & (np.sign(prev) == side)
        size = np.where(keep, prev, 0.0)
        new = np.where((side != 0) & ~keep,
                       side * ratio * np.where(side > 0, size_l, size_s), 0.0)
        need, room = np.abs(new).sum(), gross - np.abs(size).sum()
        if need > 0:
            size = size + new * min(1.0, max(room, 0.0) / need)
        states[k] = prev = size
    which = np.searchsorted(at, np.arange(n), side="right")
    return pd.DataFrame(states[which], index=longs.index, columns=longs.columns)


def _execute(decided: pd.DataFrame, rows: pd.Series, phased: bool = False) -> pd.DataFrame:
    """Weights decided at a full row's close, traded at the next full rows' opens (holiday
    stubs carry the previous weight, so no order is placed on them).

    ``phased=False``: every change trades at the first full row after the decision.
    ``phased=True`` splits a rebalance over the first TWO full rows. On the first, a position
    that closes or shrinks trades, and the old side of a flip closes; a position that opens or
    grows keeps its old weight. On the second, everything reaches its new target. Under the
    SHARED engine's locked short proceeds and ``call_seq="auto"``, the cash a cover or a sale
    releases is then free before any new position asks for it.
    """
    ok = rows.to_numpy()
    done = decided[ok].shift(1, fill_value=0.0)
    if phased:
        new = done.to_numpy()
        old = done.shift(1, fill_value=0.0).to_numpy()
        opens = (old == 0) & (new != 0)
        grows = (np.sign(old) == np.sign(new)) & (np.abs(new) > np.abs(old))
        first = np.where(opens | grows, old, np.where(old * new < 0, 0.0, new))
        done = pd.DataFrame(np.where(new != old, first, new), index=done.index,
                            columns=done.columns)
    return done.reindex(decided.index).ffill().fillna(0.0)


def _full_rows(close: pd.DataFrame) -> pd.Series:
    """True on a real session: at least half the names of the trailing 21-row median print
    (a holiday stub — a bar a few names carry on a market holiday — is False)."""
    count = close.notna().sum(axis=1)
    return (count > 0) & (count >= 0.5 * count.rolling(21, min_periods=1).median())


def _month_ends(index: pd.DatetimeIndex, rows: pd.Series) -> pd.Series:
    """True on the last full row of each calendar month (the panel's last row never: what it
    decides would fill after the panel ends).

    A row is known to be a month's last only once the next row exists, which is exact in a
    backtest (the decision fills at the next open). A LIVE run deciding at the month's last
    close needs an exchange calendar instead: without one it would decide a session late.
    """
    ok = rows.to_numpy()
    days = index[ok]
    month = days.year * 12 + days.month
    last = np.r_[month[1:] != month[:-1], False]
    out = np.zeros(len(index), bool)
    out[np.flatnonzero(ok)[last]] = True
    return pd.Series(out, index=index)


def _wsum(a: np.ndarray, window: int) -> np.ndarray:
    """Trailing sums over ``window`` rows (the row itself included) of a 2-d array."""
    c = np.cumsum(np.asarray(a, dtype=float), axis=0)
    out = c.copy()
    out[window:] -= c[:-window]
    return out


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    """``a`` moved ``k`` rows down (NaN above)."""
    if k <= 0:
        return a
    out = np.full(a.shape, np.nan)
    out[k:] = a[:-k]
    return out
