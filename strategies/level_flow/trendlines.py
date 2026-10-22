"""Trendlines: sloped support and resistance drawn from a chart's own swing points.

A horizontal level is a line with slope zero. This module finds lines of ANY slope the way a
chart reader draws them, and states the rule exactly:

1. **Swing points.** A candle whose high (low) is the highest (lowest) of the ``pivot_k``
   candles on each side is a swing high (low). It is only known ``pivot_k`` candles later, so
   lines drawn "as of" candle ``t`` use only swings confirmed by ``t``.
2. **Candidate lines.** Every pair of same-side swings at least ``min_gap`` candles apart
   defines a line in (candle, log price). The pair is kept only if no candle between the two
   swings traded through the line by more than the touch tolerance: the line was the upper
   (lower) boundary of price the whole way. The pairs that pass are exactly the edges of the
   upper (lower) convex hulls of every sub-window, so this is a hull search made exhaustive;
   ``max_pierce`` lets a few candles poke through.
3. **Extension and scoring.** Each line is carried forward to ``asof``. A CLOSE beyond it by
   more than the tolerance breaks it. Swings within the tolerance are touches, but a touch only
   counts as a new TEST if some candle CLOSED ``leave_atr`` away from the line since the previous
   one, so a consolidation hugging the line is one test, not five; the two anchors must be
   separate tests too. A line needs ``min_touches`` tests; its score is its tests plus bonuses
   for the share of the lookback they span and for how recently the last one came.
   A trendline is a boundary price TRAVELED ALONG, so a line whose price stood on average more
   than ``max_gap_atr`` from it between its first and last test is dropped. On an accelerating
   move EVERY chord between two swings clears the path, so without this the hull test has no
   power there; the price it also drops, an old extreme retested after a long excursion, is a
   horizontal level's job (the HSAR table in ``strategy.py``), not a trendline's. And a line is
   only carried as far forward as it was shown to hold: once more than ``max_extend`` times its
   tested span has passed since its last test, it is stale.
4. **One line per idea.** Lines are taken best-first, skipping any that is the same idea as an
   accepted one: it shares two tests (two points fix a line), or it ends on the same latest test
   (a fan of lines converging into one swing says one thing about the present), or it stays
   within ``dup_atr`` of it over the stretch both are alive. Lines that merely START at the same
   swing survive: a flat level and a trendline off one top point to different futures.

Time is the candle POSITION (trading time: no overnight or weekend gaps), prices are logs, and
every distance is a multiple of the candles' average true range, so one set of knobs reads the
same on any timeframe and at any price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

SIDES = ("resistance", "support")
_SIGN = {"resistance": 1.0, "support": -1.0}
_CHUNK = 2048            # candidate pairs evaluated per block (bounds memory, not results)


def log_atr(candles: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range of the candles in log price, i.e. as a fraction of price.

    Args:
        candles: ``open/high/low/close`` frame of one symbol.
        window: Candles averaged.

    Returns:
        One value per candle (a trailing mean; early candles average what exists).
    """
    h, lo, c = (np.log(candles[k].astype(float)) for k in ("high", "low", "close"))
    prev = c.shift(1)
    tr = pd.concat([h - lo, (h - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


def swings(candles: pd.DataFrame, pivot_k: int) -> pd.DataFrame:
    """Swing highs and lows of the candles.

    Args:
        candles: ``high/low`` frame of one symbol.
        pivot_k: Candles on each side a swing must dominate.

    Returns:
        ``pos`` (candle position), ``y`` (log price), ``side`` (``resistance`` for a swing
        high, ``support`` for a swing low) and ``known`` (the position it is confirmed at,
        ``pos + pivot_k``), sorted by position.
    """
    w = 2 * pivot_k + 1
    rows = []
    for side, col, pick in (("resistance", "high", np.max), ("support", "low", np.min)):
        y = np.log(candles[col].to_numpy(float))
        if len(y) < w:
            continue
        core = y[pivot_k:len(y) - pivot_k]
        hit = np.flatnonzero(core == pick(sliding_window_view(y, w), axis=1)) + pivot_k
        rows.append(pd.DataFrame({"pos": hit, "y": y[hit], "side": side, "known": hit + pivot_k}))
    if not rows:
        return pd.DataFrame(columns=["pos", "y", "side", "known"])
    return pd.concat(rows).sort_values("pos", kind="stable").reset_index(drop=True)


def asof_position(candles: pd.DataFrame, asof) -> int:
    """The position of the last candle at or before ``asof``.

    Args:
        candles: The candle frame.
        asof: ``None`` (the last candle), an integer position (negative counts from the end)
            or anything ``pd.Timestamp`` reads, compared with the candles' own index.

    Returns:
        A position in ``[0, len(candles))``.

    Raises:
        ValueError: If ``asof`` precedes the first candle.
    """
    if asof is None:
        return len(candles) - 1
    if isinstance(asof, (int, np.integer)):
        return int(asof) % len(candles)
    pos = int(candles.index.searchsorted(pd.Timestamp(asof), side="right")) - 1
    if pos < 0:
        raise ValueError(f"asof {asof} precedes the first candle {candles.index[0]}")
    return pos


def find_lines(
    candles: pd.DataFrame,
    asof=None,
    *,
    lookback: int = 300,
    pivot_k: int = 3,
    min_gap: int = 6,
    tol_atr: float = 0.25,
    max_pierce: int = 0,
    max_slope_atr: float = 1.0,
    min_touches: int = 3,
    max_gap_atr: float | None = 5.0,
    max_extend: float | None = 1.0,
    leave_atr: float = 1.0,
    span_weight: float = 2.0,
    recency_weight: float = 1.0,
    dup_atr: float = 1.0,
    max_lines: int = 4,
    reach_atr: float | None = 6.0,
    keep_broken: bool = False,
    atr_window: int = 14,
    candidates: bool = False,
) -> pd.DataFrame:
    """The trendlines a chart reader could have drawn at ``asof`` (see the module docstring).

    Args:
        candles: ``open/high/low/close`` frame of one symbol, in time order.
        asof: The last candle the lines may use (see :func:`asof_position`).
        lookback: Candles back from ``asof`` whose swings may anchor a line.
        pivot_k: Candles on each side a swing must dominate.
        min_gap: Fewest candles between a line's two anchors.
        tol_atr: Touch tolerance, in candle ATRs (also the margin a break must clear).
        max_pierce: Candles allowed to trade through the line between its anchors.
        max_slope_atr: Steepest line kept, in candle ATRs per candle.
        min_touches: Separate tests a line needs, its two anchors included.
        max_gap_atr: Largest mean distance (candle ATRs) between the line and the highs (lows)
            it bounds, from its first test to its last; ``None`` keeps chords too.
        max_extend: Candles since the last test a live line may run, as a multiple of the
            candles between its first and last test; ``None`` never lets a line go stale.
        leave_atr: How far (candle ATRs) a close must get from the line between two touches
            for the second to count as a new test.
        span_weight: Score bonus for tests spanning the whole lookback.
        recency_weight: Score bonus for a last test on the ``asof`` candle, fading linearly
            to nothing a whole lookback back.
        dup_atr: Two lines closer than this (candle ATRs) wherever both are alive are one.
        max_lines: Lines kept per side (and, with ``keep_broken``, broken lines per side).
        reach_atr: Keep only lines within this many candle ATRs of the ``asof`` close
            (``None`` keeps every line, however far away).
        keep_broken: Also return lines broken before ``asof`` (with ``break_pos``).
        atr_window: Candles in the ATR (the ATR is averaged over the lookback).
        candidates: Return every hull-valid pair before scoring and deduplication instead.

    Returns:
        One row per line: ``side``, anchors ``x0/x1`` and ``y0`` (log price at ``x0``),
        ``slope`` (log price per candle), ``touches`` and ``touch_pos`` (the positions of its
        separate tests), ``gap_atr`` (the mean distance above),
        ``last_touch``, ``break_pos`` (``NaN`` while unbroken), ``status``, ``score``,
        ``level`` (price at ``asof``), ``dist_atr`` (the level's distance above the ``asof``
        close) and ``slope_atr`` (candle ATRs per candle); ``attrs`` carries ``asof``,
        ``atr`` and ``tol``.
    """
    a = asof_position(candles, asof)
    lo_pos = max(0, a - lookback + 1)
    x = np.arange(lo_pos, a + 1)
    logs = {k: np.log(candles[k].to_numpy(float)) for k in ("high", "low", "close")}
    atr = float(log_atr(candles, atr_window).iloc[lo_pos:a + 1].mean())
    tol, leave = tol_atr * atr, leave_atr * atr
    sw = swings(candles.iloc[:a + 1], pivot_k)
    sw = sw[(sw["pos"] >= lo_pos) & (sw["known"] <= a)]
    found = []
    for side in SIDES:
        s = _SIGN[side]
        piv = sw[sw["side"] == side]
        P, Y = piv["pos"].to_numpy(), piv["y"].to_numpy()
        ii, jj = np.triu_indices(len(P), 1)
        dx = P[jj] - P[ii]
        slope = (Y[jj] - Y[ii]) / np.maximum(dx, 1)
        keep = (dx >= min_gap) & (np.abs(slope) <= max_slope_atr * atr)
        ii, jj, slope = ii[keep], jj[keep], slope[keep]
        ext = logs["high" if side == "resistance" else "low"][x]
        close = logs["close"][x]
        for c0 in range(0, len(ii), _CHUNK):
            ci, cj, cs = ii[c0:c0 + _CHUNK], jj[c0:c0 + _CHUNK], slope[c0:c0 + _CHUNK]
            x0, x1 = P[ci][:, None], P[cj][:, None]
            line = Y[ci][:, None] + cs[:, None] * (x[None, :] - x0)
            between = (x[None, :] > x0) & (x[None, :] < x1)
            pierce = (between & (s * (ext[None, :] - line) > tol)).sum(axis=1)
            ok = pierce <= max_pierce
            if not ok.any():
                continue
            ci, cj, cs, line, pierce = ci[ok], cj[ok], cs[ok], line[ok], pierce[ok]
            x0, x1 = P[ci][:, None], P[cj][:, None]
            broke = (x[None, :] > x1) & (s * (close[None, :] - line) > tol)
            brk = np.where(broke.any(axis=1), x[broke.argmax(axis=1)], a + 1)
            at_piv = Y[ci][:, None] + cs[:, None] * (P[None, :] - x0)
            near = ((np.abs(Y[None, :] - at_piv) <= tol) & (P[None, :] >= x0)
                    & (P[None, :] < brk[:, None]))
            # Candles that CLOSED clear of the line on the side price came from: a candle's own
            # range is about one ATR, so its far extreme would clear the line trivially.
            away = s * (line - close[None, :]) >= leave
            for r in range(len(ci)):
                tp = _tests(P[near[r]], away[r], lo_pos)
                if P[cj[r]] not in tp:
                    continue                      # the anchors are one test, not two
                span = slice(tp[0] - lo_pos, tp[-1] - lo_pos + 1)
                gap = float(np.mean(s * (line[r, span] - ext[span]))) / atr
                found.append((side, int(P[ci[r]]), int(P[cj[r]]), float(Y[ci[r]]),
                              float(cs[r]), len(tp), tp, gap, int(pierce[r]), int(brk[r])))
    lines = pd.DataFrame(found, columns=["side", "x0", "x1", "y0", "slope", "touches",
                                         "touch_pos", "gap_atr", "pierces", "break_pos"])
    lines.attrs.update(asof=a, atr=atr, tol=tol)
    if lines.empty:
        return _finish(lines, logs["close"][a], a, atr)
    lines["last_touch"] = lines["touch_pos"].map(max)
    lines["status"] = np.where(lines["break_pos"] <= a, "broken", "active")
    lines["break_pos"] = lines["break_pos"].where(lines["break_pos"] <= a)
    span = (lines["last_touch"] - lines["x0"]) / max(lookback - 1, 1)
    fresh = 1.0 - (a - lines["last_touch"]) / max(lookback - 1, 1)
    lines["score"] = lines["touches"] + span_weight * span + recency_weight * fresh
    lines = _finish(lines, logs["close"][a], a, atr)
    if candidates:
        return lines
    lines = lines[lines["touches"] >= min_touches]
    if max_gap_atr is not None:
        lines = lines[lines["gap_atr"] <= max_gap_atr]
    if max_extend is not None:
        stale = a - lines["last_touch"] > max_extend * (lines["last_touch"] - lines["x0"])
        lines = lines[(lines["status"] == "broken") | ~stale]
    if not keep_broken:
        lines = lines[lines["status"] == "active"]
    if reach_atr is not None:
        lines = lines[(lines["status"] == "broken") | (lines["dist_atr"].abs() <= reach_atr)]
    # Live and broken lines are chosen separately: a broken line never takes a live one's seat.
    kept = [_dedup(g, a, dup_atr * atr, max_lines)
            for _, g in lines.groupby(["side", "status"], sort=False)]
    out = pd.concat(kept) if kept else lines.iloc[:0]
    out = out.sort_values(["side", "score"], ascending=[True, False]).reset_index(drop=True)
    out.attrs.update(asof=a, atr=atr, tol=tol)
    return out


def line_values(lines: pd.DataFrame, positions) -> np.ndarray:
    """Prices of each line at the given candle positions.

    Args:
        lines: Output of :func:`find_lines`.
        positions: Candle positions (scalar or array).

    Returns:
        ``(len(lines), len(positions))`` array of prices.
    """
    pos = np.atleast_1d(np.asarray(positions, float))
    y = (lines["y0"].to_numpy()[:, None]
         + lines["slope"].to_numpy()[:, None] * (pos[None, :] - lines["x0"].to_numpy()[:, None]))
    return np.exp(y)


def forward_tests(lines: pd.DataFrame, candles: pd.DataFrame, horizon: int, *,
                  hold_atr: float = 1.0) -> pd.DataFrame:
    """What price did at each line after the lines were drawn.

    Starting the candle after ``lines.attrs["asof"]``, the FIRST candle whose range reaches the
    line's touch band is its test. From that candle on, the line BROKE if a candle closes beyond
    it by more than the tolerance before price retreats ``hold_atr`` candle ATRs back from it,
    and HELD if the retreat comes first; a close beyond on the retreat candle counts as a
    break. Symmetric, so a line at an arbitrary price has no built-in answer.

    Args:
        lines: Output of :func:`find_lines` (its ``attrs`` carry ``asof``, ``atr``, ``tol``).
        candles: The same candle frame.
        horizon: Candles after ``asof`` to watch.
        hold_atr: Retreat from the line, in candle ATRs, that counts as a hold.

    Returns:
        ``lines`` plus ``test_pos`` (``NaN`` if never tested), ``outcome`` (``held``,
        ``broke``, ``open`` or ``untested``) and ``outcome_pos``.
    """
    a, atr, tol = (lines.attrs[k] for k in ("asof", "atr", "tol"))
    x = np.arange(a + 1, min(a + horizon, len(candles) - 1) + 1)
    hi, lo, cl = (np.log(candles[k].to_numpy(float))[x] for k in ("high", "low", "close"))
    out = lines.copy()
    test, outcome, when = [], [], []
    for r, yl in zip(lines.itertuples(), np.log(line_values(lines, x)), strict=True):
        s = _SIGN[r.side]
        reach, far = (hi, lo) if s > 0 else (lo, hi)
        tested = np.flatnonzero(s * (reach - yl) >= -tol)
        if r.status == "broken" or not len(tested):
            test.append(np.nan)
            outcome.append("untested")
            when.append(np.nan)
            continue
        t = tested[0]
        brk = np.flatnonzero(s * (cl[t:] - yl[t:]) > tol)
        away = np.flatnonzero(s * (yl[t:] - far[t:]) >= hold_atr * atr)
        b, h = (brk[0] if len(brk) else np.inf), (away[0] if len(away) else np.inf)
        test.append(x[t])
        if np.isinf(b) and np.isinf(h):
            outcome.append("open")
            when.append(np.nan)
        else:
            outcome.append("broke" if b <= h else "held")
            when.append(x[t + int(min(b, h))])
    out["test_pos"], out["outcome"], out["outcome_pos"] = test, outcome, when
    return out


def shifted(lines: pd.DataFrame, shift_atr) -> pd.DataFrame:
    """Parallel copies of the lines, moved by ``shift_atr`` candle ATRs: the placebo.

    Same slope and same anchors in time, so the geometry is kept and only the anchoring to the
    swings that built the line is lost. A positive shift moves a resistance up and a support
    down, i.e. away from price; a negative one moves it toward (and possibly across) price.

    Args:
        lines: Output of :func:`find_lines` (``attrs`` must carry ``atr``).
        shift_atr: Shift per line, in candle ATRs (scalar or one value per row).

    Returns:
        The shifted lines, same columns and ``attrs``; ``level`` and ``dist_atr`` updated.
    """
    out = lines.copy()
    move = out["side"].map(_SIGN).to_numpy() * np.asarray(shift_atr, float)
    out["y0"] = out["y0"] + move * lines.attrs["atr"]
    out["level"] = out["level"] * np.exp(move * lines.attrs["atr"])
    out["dist_atr"] = out["dist_atr"] + move
    out.attrs = dict(lines.attrs)
    return out


def _tests(pos: np.ndarray, away: np.ndarray, lo_pos: int) -> list[int]:
    """The touches that are separate tests: price left the line between each and the last.

    Args:
        pos: Sorted positions of the swings within the touch tolerance.
        away: Per window candle, whether it closed ``leave_atr`` clear of the line.
        lo_pos: Position of the window's first candle (``away[0]``).

    Returns:
        The first touch, then every touch preceded by a candle clear of the line.
    """
    kept: list[int] = []
    for p in pos:
        if not kept or away[kept[-1] - lo_pos + 1:p - lo_pos].any():
            kept.append(int(p))
    return kept


def _finish(lines: pd.DataFrame, close_a: float, a: int, atr: float) -> pd.DataFrame:
    """Add the ``asof`` reading of each line: its price, distance and slope in ATRs."""
    y_a = lines["y0"] + lines["slope"] * (a - lines["x0"])
    lines["level"] = np.exp(y_a)
    lines["dist_atr"] = (y_a - close_a) / atr
    lines["slope_atr"] = lines["slope"] / atr
    return lines


def _dedup(lines: pd.DataFrame, a: int, tol: float, max_lines: int) -> pd.DataFrame:
    """Best-first lines, skipping any that is the same idea as one already kept.

    Two lines are the same if they share two tests, end on the same latest test, or run within
    ``tol`` (log) of each other at both ends of the stretch where both are alive (lines are
    straight, so the ends bound the gap).
    """
    order = lines.sort_values(["score", "touches"], ascending=False)
    end = order["break_pos"].fillna(a).to_numpy()
    x0, y0, sl = (order[k].to_numpy() for k in ("x0", "y0", "slope"))
    tests = [set(t) for t in order["touch_pos"]]
    last = order["last_touch"].to_numpy()

    def same(r: int, q: int) -> bool:
        if len(tests[r] & tests[q]) >= 2 or last[r] == last[q]:
            return True
        lo_, hi_ = max(x0[r], x0[q]), min(end[r], end[q])
        return lo_ <= hi_ and all(
            abs(y0[r] + sl[r] * (t - x0[r]) - y0[q] - sl[q] * (t - x0[q])) <= tol
            for t in (lo_, hi_))

    kept: list[int] = []
    for r in range(len(order)):
        if not any(same(r, q) for q in kept):
            kept.append(r)
            if len(kept) == max_lines:
                break
    return order.iloc[kept]
