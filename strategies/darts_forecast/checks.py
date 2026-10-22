"""darts_forecast's integrity and leakage checks: the helpers behind the notebooks' CHECKS rows.

Each function re-derives or perturbs something the pipeline claims and reports whether the claim
holds: a panel with fresh random bars after a cut (nothing before the cut may move), the options
files against the stock data of the day they are named after, the rule's decisions against the
engine's position path. Loaded by path from the notebooks, like ``research.py``; ``strategy.py``
never imports it, and it never loads ``research.py`` (a notebook passes in what a check needs).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

#: The tolerance of every "unchanged" comparison of two ``build_inputs`` results.
TOL = {"rtol": 1e-6, "atol": 1e-6, "equal_nan": True}


def scrambled(p, rows: np.ndarray, seed: int):
    """A copy of panel ``p`` whose 1-minute bars in ``rows`` are fresh random bars: every open,
    high, low, close and volume there is new (a missing bar stays missing).

    Args:
        p: The panel.
        rows: Boolean mask over the panel's 1-minute bars.
        seed: The random seed.

    Returns:
        The scrambled copy (``p`` is not modified).
    """
    r = np.random.default_rng(seed)
    idx = np.flatnonzero(rows)
    shape = (len(idx), p.close.shape[1])
    gap = r.normal(0.0, 0.0005, shape)  # previous close -> open
    move = r.normal(0.0, 0.002, shape)  # open -> close
    start = p.close.iloc[: max(int(idx[0]), 1)].ffill().to_numpy(float)[-1]
    log_c = np.log(start) + np.cumsum(gap + move, axis=0)
    o, c = np.exp(log_c - move), np.exp(log_c)
    new = {
        "open_": o,
        "close": c,
        "high": np.maximum(o, c) * np.exp(np.abs(r.normal(0.0, 0.001, shape))),
        "low": np.minimum(o, c) * np.exp(-np.abs(r.normal(0.0, 0.001, shape))),
        "volume": p.volume.to_numpy(float)[idx] * r.uniform(0.2, 5.0, shape),
    }
    gone = p.close.isna().to_numpy()[idx]
    out = {}
    for field, x in new.items():
        frame = getattr(p, field)
        arr = frame.to_numpy(float).copy()
        arr[idx] = np.where(gone, np.nan, x)
        out[field] = pd.DataFrame(arr, index=frame.index, columns=frame.columns)
    return replace(p, **out)


def same_inputs(a: dict, b: dict, upto: int | None = None) -> bool:
    """Do two ``build_inputs`` results agree on every input, y, scale, target and valid, on the
    grid bars before ``upto`` (all of them for None), within :data:`TOL`?

    Args:
        a: The reference result (its ``names`` are compared).
        b: The other result.
        upto: One past the last grid bar compared.

    Returns:
        True when they agree.
    """
    sl = slice(None, upto)
    ok = all(np.allclose(a["past"][s][sl], b["past"][s][sl], **TOL) for s in a["names"])
    ok &= all(np.allclose(a[x][sl], b[x][sl], **TOL) for x in ("y", "scale", "target"))
    return ok and bool(np.array_equal(a["valid"][sl], b["valid"][sl]))


def per_day(values, dates, names, sessions: pd.DatetimeIndex, symbols: list[str]) -> tuple:
    """The rows of an options file labelled with a session of ``sessions`` but the last (whose
    next session is known) and a name of ``symbols``, with a finite value.

    Args:
        values: One value per row.
        dates: Each row's labelled day.
        names: Each row's underlying.
        sessions: The panel's sessions (naive dates, in order).
        symbols: The panel's names.

    Returns:
        ``(value, session number, name number)`` arrays of the kept rows.
    """
    v = np.asarray(values, float)
    i = sessions.get_indexer(pd.to_datetime(pd.Series(np.asarray(dates))))
    j = np.array([symbols.index(s) if s in symbols else -1 for s in names])
    ok = (i >= 0) & (i + 1 < len(sessions)) & (j >= 0) & np.isfinite(v)
    return v[ok], i[ok], j[ok]


def vs_close(values, dates, names, close: np.ndarray, what: str, *, sessions: pd.DatetimeIndex,
             symbols: list[str]) -> dict:
    """A price in an options file against the stock's close: on the labelled day and on the next
    session. The labelled day must match clearly better (median gap under a third of the next
    session's, and closer on more than 75% of the rows).

    Args:
        values: The file's price per row.
        dates: Each row's labelled day.
        names: Each row's underlying.
        close: The stock's session close (sessions x names).
        what: The test's description.
        sessions: The panel's sessions.
        symbols: The panel's names.

    Returns:
        One row of the check's table (``ok`` decides).
    """
    v, i, j = per_day(values, dates, names, sessions, symbols)
    same, nxt = np.abs(v / close[i, j] - 1), np.abs(v / close[i + 1, j] - 1)
    return {
        "test": what,
        "rows": len(v),
        "labelled day": f"median |gap| {np.nanmedian(same):.2%}",
        "next session": f"median |gap| {np.nanmedian(nxt):.2%}",
        "labelled day closer": f"{np.nanmean(same < nxt):.0%} of rows",
        "ok": bool(np.nanmedian(same) < np.nanmedian(nxt) / 3 and np.nanmean(same < nxt) > 0.75),
    }


def vs_series(values, dates, names, ref: np.ndarray, what: str, *, sessions: pd.DatetimeIndex,
              symbols: list[str]) -> dict:
    """The rank correlation of an options file's daily value with a reference series: on the
    labelled day and on the next session. The labelled day must correlate better by more than 0.1.

    Args:
        values: The file's value per row.
        dates: Each row's labelled day.
        names: Each row's underlying.
        ref: The reference (sessions x names).
        what: The test's description.
        sessions: The panel's sessions.
        symbols: The panel's names.

    Returns:
        One row of the check's table (``ok`` decides).
    """
    v, i, j = per_day(values, dates, names, sessions, symbols)
    same = pd.Series(v).corr(pd.Series(ref[i, j]), method="spearman")
    nxt = pd.Series(v).corr(pd.Series(ref[i + 1, j]), method="spearman")
    return {
        "test": what,
        "rows": len(v),
        "labelled day": f"rank corr {same:+.2f}",
        "next session": f"rank corr {nxt:+.2f}",
        "labelled day closer": "",
        "ok": bool(same > nxt + 0.1),
    }


def path_misses(d: dict, sig: dict, end: np.ndarray, held) -> int:
    """Disagreements between a rule's decisions and ``strategy.trade_frames``' position path:
    entries not on the fill bar of a decision of their side (one bar later after a switch), and
    decisions whose side is not held from their fill bar (or the bar after).

    Args:
        d: The rule's decisions on the grid (``research.decisions``: ``long`` / ``short``).
        sig: ``strategy.trade_frames`` output for the same rule (1-minute bars x names).
        end: Each grid bar's last 1-minute row (``g["end"]``): a decision fills one row later.
        held: The holding-per-bar function (``research.held``).

    Returns:
        The number of disagreements (0 = the decisions reproduce the path).
    """
    n1, m = sig["entries"].shape
    misses = 0
    for side, ent, ext in (("long", "entries", "exits"), ("short", "short_entries", "short_exits")):
        rr, cc = np.nonzero(d[side])
        fill = np.zeros((n1, m), bool)
        fill[np.minimum(end[rr] + 1, n1 - 1), cc] = True
        e, x = sig[ent].to_numpy(), sig[ext].to_numpy()
        near = fill | np.vstack([np.zeros((1, m), bool), fill[:-1]])
        h = held(e, x)
        held_soon = h | np.vstack([h[1:], np.zeros((1, m), bool)])
        misses += int((e & ~near).sum()) + int((fill & ~held_soon).sum())
    return misses


# ---------------------------------------------------------------------------------------------
# The walkthrough's checks (darts_forecast_walkthrough.ipynb): each returns [(check, ok, detail)].
# ---------------------------------------------------------------------------------------------
def data_checks(panel, inp: dict, *, factors: pd.DataFrame, last_allowed: str,
                coverage: pd.Series | None = None) -> list[tuple]:
    """The raw 1-minute data's health, before anything is built from it.

    Args:
        panel: The 1-minute panel.
        inp: Its ``strategy.build_inputs`` output.
        factors: The stored DAILY bars' dividend factors over the same window
            (``research.adjustment_factors``): every daily-bar session must be a 1-minute session
            (the first and the last included), and every 1-minute session needs a factor for
            every name (``adjust_dividends`` would carry the previous one: a missed ex-date).
        last_allowed: The last session the notebook may read ("YYYY-MM-DD").
        coverage: ``research.coverage`` of the option inputs over the graded window.

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    day = pd.Series(np.asarray(panel.close.index.tz_convert(panel.tz).strftime("%Y-%m-%d")))
    per = panel.close.notna().groupby(day.to_numpy()).sum()      # 1-minute closes: session x name
    sess = list(per.index)
    fac = factors.reindex(columns=panel.symbols)
    want = {f"{pd.Timestamp(d):%Y-%m-%d}" for d in fac.index}
    lost = sorted(want - set(sess))
    worst = per.stack().idxmin()                                  # (session, name)
    full = {f"{pd.Timestamp(d):%Y-%m-%d}" for d in fac.index[fac.notna().all(axis=1)]}
    bare = sorted(set(sess) - full)
    g = inp["g"]
    bars = pd.Series(g["code"]).value_counts()
    empty = ~g["regular"]
    zero = (panel.volume == 0).mean()
    first = np.r_[True, day.to_numpy()[1:] != day.to_numpy()[:-1]]
    prev = panel.close.ffill().shift(1).to_numpy()[first]
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.nanmax(np.abs(np.log(panel.open_.to_numpy()[first] / prev)))
    rows = [
        ("every daily-bar session has its 1-minute bars", not lost,
         f"{len(sess)} sessions {sess[0]} → {sess[-1]}; missing: {', '.join(lost[:3]) or 'none'}"),
        ("every 1-minute session has its dividend factor for every name", not bare,
         f"without one: {', '.join(bare[:3]) or 'none'}"),
        ("every session holds all 390 one-minute bars for every name",
         bool((per == 390).all().all()),
         f"{int(per.min().min())}–{int(per.max().max())} bars per session and name"
         + (f"; fewest: {worst[1]} on {worst[0]}" if per.min().min() < 390 else "")),
        ("every session has the same number of hourly bars", bool((bars == g["n_pos"]).all()),
         f"{g['n_pos']} bars in each of {len(bars)} sessions"),
        ("half-day bars after the 13:00 close are empty", bool(np.isnan(inp["y"][empty]).all()),
         f"{int(empty.sum())} empty bars"),
        ("volume-0 placeholder minutes are rare (< 2% per name)", bool((zero < 0.02).all()),
         ", ".join(f"{n} {v:.1%}" for n, v in zero.items())),
        ("no overnight gap above 25% (a split would show as one)", bool(gap < math.log(1.25)),
         f"largest |gap| {math.expm1(gap):.1%}"),
        ("the panel ends inside the allowed window (TEST and HOLDOUT unread)",
         sess[-1] <= last_allowed, f"last session {sess[-1]}, allowed through {last_allowed}"),
    ]
    if coverage is not None:
        rows.append(("the option inputs cover the graded window (>= 90%)",
                     bool(coverage.min() >= 0.9), f"lowest coverage {coverage.min():.0%}"))
    return rows


def _last_real(close: np.ndarray, regular: np.ndarray, row: int) -> int:
    """The last 1-minute row at or before ``row`` inside the real session with a close."""
    while not (regular[row] and np.isfinite(close[row])):
        row -= 1
    return row


def target_by_hand(panel, inp: dict, n: int = 5, *, seed: int = 0, include=()) -> tuple:
    """The 5-day change recomputed straight from the 1-minute bars for a few decisions, against
    the pipeline's target.

    For a decision at grid bar s the price "now" is the last real 1-minute close of its bar, the
    price "then" the last real close of bar s + horizon (the same clock time 5 sessions later);
    the change is log(then / now). The pipeline stores it divided by ``tscale`` at s.

    Args:
        panel: The 1-minute panel.
        inp: Its ``strategy.build_inputs`` output.
        n: Random decisions with a usable label.
        seed: Random seed.
        include: ``(grid bar, name column)`` decisions to show first (e.g. the worked example).

    Returns:
        ``(table, check row)``: one row per decision (times, prices, the change by hand and the
        pipeline's), and the row "by hand = pipeline" (every difference below 1e-9).
    """
    g, H = inp["g"], inp["horizon"]
    cand = np.argwhere(inp["valid"][H:])                    # (decision bar, name) with a label
    pick = cand[np.random.default_rng(seed).choice(len(cand), n, replace=False)]
    close, reg = panel.close.to_numpy(float), g["regular_1m"]
    shut = (panel.close.index + pd.Timedelta(minutes=1)).tz_convert(panel.tz)  # a bar's close
    rows = []
    for s, j in [*include, *map(tuple, pick)]:
        a = _last_real(close[:, j], reg, int(g["end"][s]))
        b = _last_real(close[:, j], reg, int(g["end"][s + H]))
        hand = math.log(close[b, j] / close[a, j])
        pipe = inp["target"][s + H, j] * inp["tscale"][s, j]
        rows.append({"name": inp["names"][j], "decision": f"{shut[a]:%Y-%m-%d %H:%M}",
                     "price now": close[a, j], "target": f"{shut[b]:%Y-%m-%d %H:%M}",
                     "price then": close[b, j], "change %": 100 * (close[b, j] / close[a, j] - 1),
                     "log change by hand": hand, "pipeline": pipe, "difference": hand - pipe,
                     "typical 5-day move %": 100 * inp["tscale"][s, j],
                     "target (typical moves)": inp["target"][s + H, j]})
    tab = pd.DataFrame(rows)
    worst = float(tab["difference"].abs().max())
    return tab, ("the target recomputed from the 1-minute bars = the pipeline's", worst < 1e-9,
                 f"{len(tab)} decisions, largest difference {worst:.1e}")


def walk_forward_checks(fc: dict, params: dict) -> list[tuple]:
    """The walk-forward did what it claims, bar by bar.

    Args:
        fc: The forecast.
        params: Its parameters (``retrain_minutes``, ``expanding``, ``vol_days``, ``val_days``).

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    inp, log = fc["inputs"], fc["log"]
    g = inp["g"]
    pts = log["retrain_bucket"].to_numpy(int)
    bars, cols = np.nonzero(np.isfinite(fc["edge"]))
    k = np.searchsorted(pts, bars, side="right") - 1          # the newest fit at or before the bar
    newest = (k >= 0) & (log["fit"].to_numpy()[np.maximum(k, 0)] == fc["fit"][bars, cols])
    warm = str(g["date"][g["sess_first"][max(21, params["vol_days"] + 1)]])
    step, want = np.diff(g["code"][pts]), params["retrain_minutes"] // 390
    apart = ("" if not len(step) else f", {step[0]} session{'' if step[0] == 1 else 's'} apart"
             if step.min() == step.max() else f", {step.min()}–{step.max()} sessions apart")
    return [
        ("every forecast comes from the newest fit made at or before its bar", bool(newest.all()),
         f"{len(bars):,} forecasts, {len(log)} fits"),
        ("every fit learns only from outcomes known before it",
         bool((log["train_last"] < log["session"]).all()),
         "each fit's last training label closes before the session it forecasts"),
        ("every fit starts right after the warm-up (expanding)",
         bool((log["train_from"] == warm).all()) or not params["expanding"],
         f"first training session {warm}"),
        ("a fresh fit before every session" if want == 1 else "fits on the planned schedule",
         bool((step == want).all()) if want else len(pts) == 1,
         f"{len(pts)} fit{'s' * (len(pts) != 1)}{apart}"),
        ("no forecast before the first fit", bool(bars.min() >= pts[0]),
         f"first forecast {g['date'][bars.min()]}, first fit {log['session'].iloc[0]}"),
        ("no validation slice (the trees have a fixed size)",
         bool((log["val_days"] == params["val_days"]).all()), f"val_days {params['val_days']}"),
    ]


def truncation_check(panel, params: dict, fc: dict, run, *, cut_sessions: int = 10,
                     seed: int = 0) -> list[tuple]:
    """Nothing reads the future: fresh random bars from a cut on (:func:`scrambled`) and the whole
    walk-forward re-run; every input and every forecast made before the cut must be identical,
    and the later forecasts must change (or the check could not fail).

    The cut sits ONE bar after a fit (the last at or before the session ``cut_sessions`` before
    the end): that fit's first forecast is made before the cut, so a fit that also learned from
    any later label (scrambled in the re-run) changes it. A cut right on a fit would let a leak of
    up to a session through: every label such a fit could peek at would still be a real bar.

    Args:
        panel: The panel.
        params: The parameters.
        fc: Their forecast on ``panel``.
        run: ``research.cached_forecast`` (panel, params, extra) -> forecast.
        cut_sessions: The cut is in the session this many sessions before the panel's end.
        seed: Random seed of the fresh bars.

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    g = fc["inputs"]["g"]
    pts = np.asarray(fc["points"])
    fit = int(pts[pts <= g["sess_first"][-cut_sessions]][-1])
    cut = fit + 1
    alt = run(scrambled(panel, np.arange(len(panel.close)) >= g["start"][cut], seed), params,
              f"-scrambled{seed}")
    before = int(np.isfinite(fc["edge"][:cut]).sum())
    return [
        ("the inputs before the cut ignore every later bar",
         same_inputs(fc["inputs"], alt["inputs"], cut),
         f"cut after the first bar of {g['date'][fit]} (one bar after a fit)"),
        ("forecasts made before the cut are unchanged",
         bool(np.array_equal(fc["edge"][:cut], alt["edge"][:cut], equal_nan=True)),
         f"{before:,} forecasts compared"),
        ("forecasts after the cut did change (the check can fail)",
         not np.array_equal(fc["edge"][cut:], alt["edge"][cut:], equal_nan=True), ""),
    ]


def refit_check(fc: dict, first: int, edge: np.ndarray, day: str) -> list[tuple]:
    """The direct leak test: a model fit on the data through ``day`` alone (it physically lacks
    every later bar) must give exactly the walk-forward's forecasts for the next session. It
    catches a training leak of any length, and it checks that ``research.latest_forecast``'s fit
    (made the same way) is the walk-forward's own model.

    Args:
        fc: The forecast.
        first: The next session's first grid bar (``research.refit_forecast``).
        edge: The refit model's forecasts of that session (its bars x names).
        day: The last session the refit saw.

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    want = fc["edge"][first:first + len(edge)]
    ok = bool(np.isfinite(want).all() and np.allclose(edge, want, rtol=0, atol=1e-9))
    diff = float(np.max(np.abs(edge - want))) if np.isfinite(want).all() else np.nan
    nxt = fc["inputs"]["g"]["date"][first]
    return [(f"a model fit on the data through {day} alone gives the walk-forward's forecasts "
             f"for {nxt}", ok, f"{edge.size} forecasts, largest difference {diff:.1e}")]


def window_checks(t: pd.DataFrame, window: tuple, expected: pd.DataFrame | None = None
                  ) -> list[tuple]:
    """The graded rows stay inside their window, and every expected decision is graded.

    Args:
        t: A ``research.horizon_table``.
        window: ``(first, last)`` session of the window ("YYYY-MM-DD").
        expected: The decisions that should be graded (``research.outcome_table``).

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    a, b = pd.Timestamp(window[0]), pd.Timestamp(window[1]) + pd.Timedelta(days=1)
    per = t.groupby("name").size()
    rows = []
    if expected is not None:
        key = ["bucket", "name"]
        rows.append(("every decision with a known outcome got a forecast",
                     t[key].reset_index(drop=True).equals(expected[key].reset_index(drop=True)),
                     f"{len(t):,} graded of {len(expected):,} expected"))
    return rows + [
        ("every graded decision lies inside the window",
         bool((t["ts"] >= a).all() and (t["ts"] < b).all()),
         f"{t['ts'].min():%Y-%m-%d} → {t['ts'].max():%Y-%m-%d}"),
        ("every graded outcome lies inside the window (no later price is read)",
         bool((t["target_ts"] < b).all()), f"last outcome {t['target_ts'].max():%Y-%m-%d %H:%M}"),
        ("every name has the same number of graded forecasts", bool(per.nunique() == 1),
         ", ".join(f"{n} {v}" for n, v in per.items())),
    ]
