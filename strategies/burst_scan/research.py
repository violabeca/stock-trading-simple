"""burst_scan research helpers: the computations the five notebooks share, no figures.

Loaded by path from the notebooks, like level_flow's ``trendlines.py``; ``strategy.py`` never
imports it, so nothing here can leak into a backtest. Every function is pure and seeded.

* **Statistics** — :func:`day_t` (mean with a t clustered by trading day: events on one day
  share its news and are not independent), :func:`paired_day_t`, :func:`auc`.
* **Controls** — :func:`placebo_pool` / :func:`draw_placebo_a` (same name, same month, same
  half hour, no burst nearby) and :func:`draw_placebo_b` (same-size price move, normal volume).
* **Outcomes and tables** — :func:`side_outcomes`, :func:`summarize`, :func:`detector_grid`,
  :func:`dose_response`, :func:`calibration`, :func:`split_overnight`.
* **Audits** — :func:`reconcile` (every engine trade against its label), :func:`purge_audit`,
  :func:`cost_anatomy`, and the :func:`checks_table` / :func:`scorecard` every notebook ends on.
* **Move study / explorer / extended hours** — :func:`move_table`, :func:`swing_runs`,
  :func:`confirmations`, and for ``move_extended.ipynb`` :func:`mask_placeholders`,
  :func:`mask_late_reports`, :func:`split_sessions`, :func:`open_table`, :func:`rank_ic`,
  :func:`monthly_walk`, :func:`match_runs`.
* **Move detectors** (``move_detectors.ipynb``) — :func:`detector_features`,
  :func:`sign_surrogate` (the coin-flip market), :func:`detector_rows`, :func:`atr_move`,
  :func:`move_events`, :func:`detector_table`, :func:`walk_detector`, :func:`causal_alerts`.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import load_panel, resample_panel, symbols_with_bars

STRAT = strategies.load("burst_scan")
S = STRAT.signals.__globals__            # the strategy module's own functions
#: Pre-registered horizon grid (minutes of trading time; None = to the close) per bar size.
CELLS = {"1m": (30, 60, 120, None, 390),
         "5m": (30, 60, 120, None, 390, 780, 1950),
         "10m": (30, 60, 120, None, 390, 780, 1950)}
#: The pre-registered primary cell: 5-minute bars held one session.
PRIMARY = ("5m", 390)
#: Pre-registered detector grid for G0 (deduplicated: a volume trigger ignores range_mult...).
GRID = {"trigger": ("volume", "range", "both"), "impulse": ("any", "up", "down"),
        "vol_mult": (2.0, 3.0, 5.0), "range_mult": (1.5, 2.0, 3.0)}


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------
def load(settings, engine, *, subset: int | None = None, end=None, resample: bool = True,
         symbols: list[str] | None = None):
    """The strategy's panel as its config describes it, optionally cut to ``subset`` names.

    Args:
        settings: Global settings.
        engine: Database engine.
        subset: Keep only the ``subset`` best-covered names (a quick first run); None = all.
        end: Override ``data.end`` (``"none"`` = everything stored — the holdout reload).
        resample: Apply ``data.resample``; False returns the stored 1-minute bars.
        symbols: Explicit symbols (skips the universe lookup).

    Returns:
        The :class:`~trading_simple.data.Panel`.
    """
    d = STRAT.data
    stop = d.get("end") if end is None else (None if end == "none" else end)
    syms = symbols or symbols_with_bars(engine, settings.universe_symbols(d["universe"]),
                                        d["bar"], d["start"], stop, limit=d.get("max_symbols"))
    panel = load_panel(engine, syms[:subset] if subset else syms, d["bar"], d["start"], stop,
                       session=d.get("session", "regular"), exchanges=settings.exchanges)
    return resample_panel(panel, d["resample"]) if resample and d.get("resample") else panel


# ---------------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------------
def day_t(values, days) -> dict:
    """Mean of ``values`` with a t-statistic clustered by day.

    Cluster-robust (CR1) standard error of a mean: events on one day share that day's news and
    move together, so treating them as independent overstates t by roughly √(events per day).

    Args:
        values: One value per event (NaN dropped).
        days: The event's trading day (any hashable per event).

    Returns:
        ``{"mean", "t", "n", "days"}``.
    """
    v = pd.Series(np.asarray(values, dtype=float))
    d = pd.Series(np.asarray(days))
    ok = v.notna().to_numpy()
    v, d = v[ok], d[ok]
    n, g = len(v), d.nunique()
    if n < 3 or g < 3:
        return {"mean": float(v.mean()) if n else np.nan, "t": np.nan, "n": n, "days": g}
    m = float(v.mean())
    sums = (v - m).groupby(d.to_numpy()).sum().to_numpy()
    se = math.sqrt((sums ** 2).sum() * g / (g - 1)) / n
    return {"mean": m, "t": m / se if se > 0 else np.nan, "n": n, "days": g}


def paired_day_t(a, a_days, b, b_days) -> dict:
    """Day-by-day difference of two event sets' daily means, over the days both traded.

    Returns:
        ``{"diff", "t", "days"}`` — the mean daily difference and its t across days.
    """
    ma = pd.Series(np.asarray(a, float)).groupby(np.asarray(a_days)).mean()
    mb = pd.Series(np.asarray(b, float)).groupby(np.asarray(b_days)).mean()
    diff = (ma - mb).dropna()
    if len(diff) < 3 or diff.std() == 0:
        return {"diff": float(diff.mean()) if len(diff) else np.nan, "t": np.nan,
                "days": len(diff)}
    return {"diff": float(diff.mean()), "t": float(diff.mean() / diff.std() * math.sqrt(len(diff))),
            "days": len(diff)}


def auc(p, y) -> float:
    """Area under the ROC curve (NaN when either class is missing)."""
    from sklearn.metrics import roc_auc_score
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    return float(roc_auc_score(y[ok], p[ok])) if len(np.unique(y[ok])) == 2 else np.nan


def monthly_auc(p, y, fold) -> float:
    """Mean of the per-month AUCs: discrimination WITHIN months, free of base-rate drift.

    A pooled AUC also rewards predicting which months have more winners; this does not.
    """
    frame = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float),
                          "f": np.asarray(fold)}).dropna()
    per = [auc(g["p"], g["y"]) for _, g in frame.groupby("f")]
    per = [a for a in per if np.isfinite(a)]
    return float(np.mean(per)) if per else np.nan


def auc_null_se(y) -> float:
    """Standard error of the AUC under 'no skill' (Hanley-McNeil), for the shuffled control."""
    y = np.asarray(y, float)
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    return math.sqrt((n1 + n0 + 1) / (12 * n1 * n0)) if n1 and n0 else np.nan


# ---------------------------------------------------------------------------
# Events, controls and their outcomes.
# ---------------------------------------------------------------------------
def events(ctx: dict, p: dict, side: str, **detector) -> tuple[np.ndarray, np.ndarray]:
    """A sub-strategy's candidate events under ``p`` (detector knobs overridable)."""
    q = {**S["_side_params"](p, side), **detector}
    ev = S["detect"](ctx, trigger=q["trigger"], impulse=q["impulse"], vol_mult=q["vol_mult"],
                     range_mult=q["range_mult"], cooldown=S["_bars"](p["cooldown"], ctx["bar"]))
    if side == "short" and p["ssr_guard"]:
        ev &= ~ctx["ssr"]
    return np.nonzero(ev)


def _near(mask: np.ndarray, n: int) -> np.ndarray:
    """True where ``mask`` fired within ``n`` bars either side (same column)."""
    cs = np.vstack([np.zeros((1, mask.shape[1]), np.int32), np.cumsum(mask, 0, dtype=np.int32)])
    t = np.arange(len(mask))
    hi = np.minimum(t + n + 1, len(mask))
    lo = np.maximum(t - n, 0)
    return (cs[hi] - cs[lo]) > 0


def placebo_pool(ctx: dict, *, cooldown_bars: int, bucket_minutes: int = 30) -> dict:
    """Bars a placebo A may sit on, keyed by (symbol, month, half hour).

    Eligible = inside the tradable window and no side-agnostic burst (relative volume
    ≥ the strategy's context level) within ``cooldown_bars`` either side.

    Returns:
        The pool (sorted keys with their bar positions) for :func:`draw_placebo_a`.
    """
    busy = (ctx["vol_ratio"] >= S["_CONTEXT_MULT"]) & ctx["inside"][:, None]
    rows, cols = np.nonzero(ctx["window"] & ~_near(busy, cooldown_bars))
    key = _key_a(ctx, rows, cols, bucket_minutes)
    order = np.argsort(key, kind="stable")
    keys, start, count = np.unique(key[order], return_index=True, return_counts=True)
    return {"keys": keys, "start": start, "count": count, "rows": rows[order],
            "cols": cols[order], "bucket": bucket_minutes}


def _key_a(ctx, rows, cols, bucket):
    month, _ = S["month_starts"](ctx["index"], ctx["tz"])
    tod = ctx["feat"]["tod"][rows].astype(np.int64) // bucket
    return (cols.astype(np.int64) * 10_000 + month[rows]) * 100 + tod


def draw_placebo_a(pool: dict, ctx: dict, rows, cols, *, seed: int = 0):
    """One placebo A per event: a random eligible bar of the same (symbol, month, half hour).

    Returns:
        ``(rows, cols, matched)`` — ``matched`` flags events that found a placebo.
    """
    rng = np.random.default_rng(seed)
    key = _key_a(ctx, np.asarray(rows), np.asarray(cols), pool["bucket"])
    if not len(pool["keys"]):
        empty = np.zeros(0, np.int64)
        return empty, empty, np.zeros(len(key), bool)
    pos = np.minimum(np.searchsorted(pool["keys"], key), len(pool["keys"]) - 1)
    matched = pool["keys"][pos] == key
    pick = pool["start"][pos] + (rng.random(len(key)) * pool["count"][pos]).astype(np.int64)
    return pool["rows"][pick][matched], pool["cols"][pick][matched], matched


def draw_placebo_b(ctx: dict, rows, cols, *, max_rvol: float = 1.5, seed: int = 0,
                   sample: float = 0.25):
    """One placebo B per event: a bar with the same-size price move but normal volume.

    Matched on month, the move's sign and its size decile (|burst move| in session ATRs, deciles
    of the events), drawn from a ``sample`` of the bars whose relative volume is below
    ``max_rvol`` (pooled across symbols). It asks whether the VOLUME adds anything to the move.

    Returns:
        ``(rows, cols, matched)``.
    """
    rng = np.random.default_rng(seed)
    size = np.abs(ctx["feat"]["burst_atr"])
    with np.errstate(invalid="ignore"):
        cand = ctx["window"] & (ctx["vol_ratio"] < max_rvol) & np.isfinite(size) & (size > 0)
    cr, cc = np.nonzero(cand)
    keep = rng.random(len(cr)) < sample
    cr, cc = cr[keep], cc[keep]
    rows, cols = np.asarray(rows), np.asarray(cols)
    edges = np.nanquantile(size[rows, cols], np.linspace(0.1, 0.9, 9)) if len(rows) else []
    month, _ = S["month_starts"](ctx["index"], ctx["tz"])

    def key(r, c):
        sign = (ctx["move"][r, c] > 0).astype(np.int64)
        return (month[r].astype(np.int64) * 2 + sign) * 10 + np.searchsorted(edges, size[r, c])

    ck = key(cr, cc)
    order = np.argsort(ck, kind="stable")
    keys, start, count = np.unique(ck[order], return_index=True, return_counts=True)
    ek = key(rows, cols)
    if not len(keys):
        empty = np.zeros(0, np.int64)
        return empty, empty, np.zeros(len(ek), bool)
    pos = np.minimum(np.searchsorted(keys, ek), len(keys) - 1)
    matched = keys[pos] == ek
    pick = start[pos] + (rng.random(len(ek)) * count[pos]).astype(np.int64)
    return cr[order][pick][matched], cc[order][pick][matched], matched


def side_outcomes(panel, ctx: dict, rows, cols, *, side: str, horizon: int | None,
                  move_atr: float) -> pd.DataFrame:
    """:func:`strategy.outcomes` for a horizon in minutes (``None`` = flat at the close).

    The barrier is the strategy's: ``move_atr`` session ATRs × √(hold / 390), where the hold of
    a to-the-close trade is each event's own minutes to the close.

    Returns:
        The outcomes frame plus ``day`` (session id) and ``frac``.
    """
    rows, cols = np.asarray(rows), np.asarray(cols)
    atr_pct = ctx["feat"]["atr_pct"][rows, cols].astype(float)
    if horizon is None:
        minutes = np.maximum(ctx["feat"]["to_close"][rows].astype(float), 1.0)
        bars, eod = int(ctx["n_left"].max()) + 1, True
    else:
        minutes, bars, eod = float(horizon), S["_bars"](horizon, ctx["bar"]), False
    frac = move_atr * atr_pct * np.sqrt(minutes / S["_SESSION_MINUTES"])
    oc = S["outcomes"](panel, ctx, rows, cols, side=1 if side == "long" else -1, horizon=bars,
                       frac=frac, eod=eod)
    return oc.assign(day=ctx["codes"][rows], frac=frac)


def summarize(oc: pd.DataFrame) -> dict:
    """One row of an outcome table: rates, demeaned return with its day-t, excursions, holds."""
    oc = oc[oc["complete"]]
    t = day_t(oc["dret"], oc["day"])
    return {"n": len(oc), "win": oc["reason"].eq(1).mean(), "loss": oc["reason"].eq(-1).mean(),
            "timeout": oc["reason"].eq(0).mean(), "dret_bp": 1e4 * t["mean"], "t": t["t"],
            "mfe_bp": 1e4 * oc["mfe"].mean(), "mae_bp": 1e4 * oc["mae"].mean(),
            "bars": oc["bars"].mean(), "overnight": oc["nights"].gt(0).mean(), "days": t["days"]}


def lift(ev: pd.DataFrame, pl: pd.DataFrame) -> dict:
    """Event vs its matched placebo, event by event: win-rate ratio and the day-t of the gap."""
    ok = (ev["complete"] & pl["complete"]).to_numpy()
    e, p = ev[ok], pl[ok]
    diff = day_t(e["win"].astype(float).to_numpy() - p["win"].astype(float).to_numpy(), e["day"])
    rate_p = p["win"].mean()
    return {"win": e["win"].mean(), "placebo_win": rate_p,
            "lift": e["win"].mean() / rate_p if rate_p else np.nan, "t": diff["t"],
            "n": int(ok.sum())}


def matched_outcomes(panel, ctx, rows, cols, draw, *, side, horizon, move_atr):
    """Outcomes of the events that found a placebo, and of their placebos, aligned by event."""
    pr, pc, matched = draw(rows, cols)
    ev = side_outcomes(panel, ctx, rows[matched], cols[matched], side=side, horizon=horizon,
                       move_atr=move_atr).reset_index(drop=True)
    pl = side_outcomes(panel, ctx, pr, pc, side=side, horizon=horizon,
                       move_atr=move_atr).reset_index(drop=True)
    return ev, pl


def detector_grid(panel, ctx: dict, p: dict, side: str, pool: dict, *, horizon: int,
                  move_atr: float, seed: int = 0) -> pd.DataFrame:
    """Every pre-registered detector setting (:data:`GRID`) against placebo A, one side.

    Returns:
        One row per setting: events, events per name per session, win rate, the placebo's, the
        lift and its day-t, the loss rate and the demeaned return — plus ``eligible`` (at least
        one event per name per five sessions, the pre-registered rate floor).
    """
    out = []
    n_names = len(ctx["symbols"])
    n_sess = len(np.unique(ctx["codes"][ctx["window"].any(axis=1)]))
    for trig, imp, vm, rm in product(*GRID.values()):
        if (trig == "volume" and rm != GRID["range_mult"][0]) or (
                trig == "range" and vm != GRID["vol_mult"][0]):
            continue                                    # the knob that trigger never reads
        rows, cols = events(ctx, p, side, trigger=trig, impulse=imp, vol_mult=vm, range_mult=rm)
        row = {"trigger": trig, "impulse": imp, "vol_mult": vm if trig != "range" else np.nan,
               "range_mult": rm if trig != "volume" else np.nan, "events": len(rows),
               "per_name_session": len(rows) / max(n_names * n_sess, 1)}
        if len(rows) >= 30:
            ev, pl = matched_outcomes(
                panel, ctx, rows, cols,
                lambda r, c: draw_placebo_a(pool, ctx, r, c, seed=seed),
                side=side, horizon=horizon, move_atr=move_atr)
            row |= lift(ev, pl) | {"loss": ev.loc[ev["complete"], "reason"].eq(-1).mean(),
                                   "dret_bp": 1e4 * ev.loc[ev["complete"], "dret"].mean()}
        out.append(row)
    grid = pd.DataFrame(out)
    grid["eligible"] = grid["per_name_session"] >= 0.2
    return grid


def pick_detector(grid: pd.DataFrame) -> pd.Series | None:
    """The pre-registered pick: the highest lift over placebo A among eligible settings."""
    ok = grid[grid["eligible"] & grid["lift"].notna()] if "lift" in grid else grid.iloc[:0]
    return None if ok.empty else ok.sort_values("lift", ascending=False).iloc[0]


def dose_response(x, win, dret, days, *, bins: int = 10) -> pd.DataFrame:
    """Win rate and demeaned return by quantile bin of ``x`` (e.g. relative volume)."""
    frame = pd.DataFrame({"x": np.asarray(x, float), "win": np.asarray(win, float),
                          "dret": np.asarray(dret, float), "day": np.asarray(days)}).dropna()
    frame["bin"] = pd.qcut(frame["x"], bins, labels=False, duplicates="drop")
    rows = []
    for b, g in frame.groupby("bin"):
        t = day_t(g["dret"], g["day"])
        rows.append({"bin": int(b), "x_mid": g["x"].median(), "n": len(g), "win": g["win"].mean(),
                     "dret_bp": 1e4 * t["mean"], "t": t["t"]})
    return pd.DataFrame(rows)


def calibration(p, win, *, bins: int = 10) -> pd.DataFrame:
    """Predicted probability against the realized win rate, by decile of the prediction."""
    frame = pd.DataFrame({"p": np.asarray(p, float), "win": np.asarray(win, float)}).dropna()
    frame["bin"] = pd.qcut(frame["p"], bins, labels=False, duplicates="drop")
    return frame.groupby("bin").agg(p=("p", "mean"), realized=("win", "mean"),
                                    n=("win", "size")).reset_index()


def split_overnight(panel, ctx: dict, table: pd.DataFrame, side: str) -> pd.DataFrame:
    """Each trade's signed log return split into the overnight gaps it held and the rest.

    Returns:
        ``total``, ``overnight`` and ``intraday`` signed log returns per row of ``table``.
    """
    sign = 1.0 if side == "long" else -1.0
    first = pd.Series(np.arange(len(ctx["codes"]))).groupby(ctx["codes"]).first().to_numpy()
    last = pd.Series(np.arange(len(ctx["codes"]))).groupby(ctx["codes"]).last().to_numpy()
    o, c = panel.open_.to_numpy(float), panel.close.to_numpy(float)
    gap = np.zeros((len(first), o.shape[1]))
    gap[1:] = np.log(o[first[1:]] / c[last[:-1]])
    cum = np.nancumsum(np.nan_to_num(gap), axis=0)
    rows, cols = table["row"].to_numpy(), table["col"].to_numpy()
    f = rows + 1
    ex = table["exit_row"].to_numpy()
    total = sign * np.log(table["exit_px"].to_numpy() / table["entry_px"].to_numpy())
    night = sign * (cum[ctx["codes"][ex], cols] - cum[ctx["codes"][f], cols])
    return pd.DataFrame({"total": total, "overnight": night, "intraday": total - night},
                        index=table.index)


# ---------------------------------------------------------------------------
# The classifier's controls.
# ---------------------------------------------------------------------------
def why(table: pd.DataFrame) -> pd.Series:
    """Why each event did or did not trade, in words (for tables and hovers)."""
    return pd.Series(np.select(
        [table["p"].isna(), ~table["picked"], table["clash"], ~table["traded"]],
        ["no model yet (training months)", "below the fold's threshold",
         "clash: the other side picked it too", "spaced out: own position still open"],
        "traded"), index=table.index)


def wf_kwargs(p: dict, side: str) -> dict:
    """The walk-forward settings :func:`strategy.signals` uses for ``side``."""
    return {"model": p["model"], "trade_q": p[f"{side}_trade_q"],
            "warmup_sessions": p["warmup_sessions"], "min_train_sessions": p["min_train_sessions"]}


def _xy(table: pd.DataFrame):
    return (table[list(S["FEATURES"])].to_numpy(float), table["win"].to_numpy(float),
            table["row"].to_numpy(), table["exit_row"].to_numpy(), table["complete"].to_numpy())


def refit(table: pd.DataFrame, ctx: dict, p: dict, side: str, *, X=None, y=None,
          **overrides) -> pd.DataFrame:
    """Re-run a side's walk-forward with other features, labels or settings."""
    X0, y0, rows, ex, done = _xy(table)
    return S["walk_forward"](X0 if X is None else X, y0 if y is None else y, rows, ex, done, ctx,
                             **{**wf_kwargs(p, side), **overrides})


def fold_fits(table: pd.DataFrame, ctx: dict, p: dict, side: str, *, model: str | None = None,
              only: list[int] | None = None):
    """Each walk-forward fold's fitted model, with its training and test masks.

    The same rule as :func:`strategy.walk_forward` (purged, expanding), exposed so a notebook
    can read coefficients or importances. Yields ``{"fold", "month", "est", "train", "test"}``
    for every predicted fold, or only the fold ids in ``only``.
    """
    X, y, rows, ex, done = _xy(table)
    kw = wf_kwargs(p, side)
    _, first = S["month_starts"](ctx["index"], ctx["tz"])
    for fm in sorted(f for f in table["fold"].unique() if f >= 0 and (only is None or f in only)):
        train = done & np.isfinite(y) & (ex < first[fm])
        test = (table["fold"] == fm).to_numpy()
        est = S["make_model"](model or kw["model"]).fit(X[train], y[train].astype(int))
        month = pd.Period(ctx["index"][first[fm]].tz_convert(ctx["tz"]).tz_localize(None), "M")
        yield {"fold": fm, "month": str(month), "est": est, "train": train, "test": test}


def shuffled(table: pd.DataFrame, ctx: dict, p: dict, side: str, *, seed: int = 0):
    """The walk-forward trained on OUTCOMES permuted across the whole sample (the leak control).

    Score it against the SAME permuted outcomes (``perm``): the only way a model trained on
    shuffled labels can predict shuffled test labels is contamination between training and test
    (a failed purge, duplicated rows). Scoring it against the TRUE labels tests nothing useful — a
    model fit to noise still inherits a random share of the real (volatility) structure through
    the features and lands off 0.5 with no leak at all. Nor within each month: that keeps every
    month's win rate, which a model can learn legitimately.

    Returns:
        ``(walk-forward frame, perm)`` — row ``i``'s shuffled outcome is ``table.iloc[perm[i]]``.
    """
    perm = np.random.default_rng(seed).permutation(len(table))
    return refit(table, ctx, p, side, y=table["win"].to_numpy(float)[perm]), perm


def canary(table: pd.DataFrame, ctx: dict, p: dict, side: str) -> pd.DataFrame:
    """The walk-forward with the LABEL appended as a feature: it must find it (AUC ≈ 1)."""
    X, y, *_ = _xy(table)
    return refit(table, ctx, p, side, X=np.column_stack([X, y]))


def placebo_trained(panel, ctx: dict, table: pd.DataFrame, p: dict, side: str, pool: dict, *,
                    seed: int = 0) -> pd.DataFrame:
    """Train on placebo-A events (random bars, same name/month/half hour), score the REAL ones.

    If the model's skill were generic to any bar of these names, a model that never saw a burst
    would pick as well. Returns the walk-forward frame for the real events.
    """
    rows, cols = table["row"].to_numpy(), table["col"].to_numpy()
    pr, pc, _ = draw_placebo_a(pool, ctx, rows, cols, seed=seed)
    q = S["_side_params"](p, side)
    pl = side_outcomes(panel, ctx, pr, pc, side=side, horizon=q["horizon"], move_atr=q["move_atr"])
    Xp = S["features"](ctx, pr, pc).to_numpy(float)
    X, _, _, ex, done = _xy(table)
    both = S["walk_forward"](
        np.vstack([Xp, X]), np.r_[pl["win"].to_numpy(float), np.full(len(rows), np.nan)],
        np.r_[pr, rows], np.r_[pl["exit_row"].to_numpy(), ex],
        np.r_[pl["complete"].to_numpy(), done], ctx, **wf_kwargs(p, side))
    return both.iloc[len(pr):].reset_index(drop=True)


# ---------------------------------------------------------------------------
# The move study: events defined by their OUTCOME (a big move within 4 hours).
# ---------------------------------------------------------------------------
#: Holding window of the move study, minutes of trading time (may run past the close).
MOVE_MINUTES = 240
#: Share of training decision points that should reach either barrier (the "unusual" rule).
MOVE_TARGET_RATE = 0.20
#: Features the move study adds to the strategy's twenty.
EXTRA_FEATURES = ("rng60", "cum_rvol")


def decision_points(ctx: dict, every_minutes: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Every ``every_minutes`` of every name, inside the strategy's tradable window.

    Returns:
        ``(rows, cols)`` — bar numbers within the session that are multiples of the step, where
        ``ctx["window"]`` allows an event (open/close blackout, warm-up, price and volume floors).
    """
    step = max(1, every_minutes // S["_minutes_per_bar"](ctx["bar"]))
    return np.nonzero(ctx["window"] & (ctx["bar_no"] % step == 0)[:, None])


def scaled_barrier(ctx: dict, rows, cols, k: float, minutes: int = MOVE_MINUTES) -> np.ndarray:
    """``k`` × the name's session ATR (as a fraction of price) × √(minutes / 390)."""
    atr = ctx["feat"]["atr_pct"][np.asarray(rows), np.asarray(cols)].astype(float)
    return k * atr * math.sqrt(minutes / S["_SESSION_MINUTES"])


def move_outcomes(panel, ctx: dict, rows, cols, frac,
                  minutes: int = MOVE_MINUTES) -> pd.DataFrame:
    """Symmetric-barrier outcome of each point, as a LONG trade would book it (reason 1 = up first,
    −1 = down first, 0 = neither within ``minutes``), plus ``day``."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    oc = S["outcomes"](panel, ctx, rows, cols, side=1, horizon=S["_bars"](minutes, ctx["bar"]),
                       frac=np.asarray(frac, dtype=float), eod=False)
    return oc.assign(day=ctx["codes"][rows])


#: The pre-registered grid of barrier multiples ``calibrate_k`` chooses from.
K_GRID = tuple(np.arange(0.5, 4.01, 0.25).round(2))


def calibrate_k(panel, ctx: dict, rows, cols, *, target: float = MOVE_TARGET_RATE,
                ks=K_GRID, sample: int = 60_000, seed: int = 0) -> tuple:
    """The pre-registered threshold rule: the smallest ``k`` whose barrier is reached by at most
    ``target`` of the given (training) points, measured on a seeded sample.

    Returns:
        ``(k, table)`` — the chosen ``k`` and the reach rate of every candidate.
    """
    rows, cols = np.asarray(rows), np.asarray(cols)
    if len(rows) > sample:
        pick = np.sort(np.random.default_rng(seed).choice(len(rows), sample, replace=False))
        rows, cols = rows[pick], cols[pick]
    out = []
    for k in ks:
        oc = move_outcomes(panel, ctx, rows, cols, scaled_barrier(ctx, rows, cols, k))
        done = oc["complete"]
        out.append({"k": float(k), "reach": float(oc.loc[done, "reason"].ne(0).mean())})
    table = pd.DataFrame(out)
    ok = table[table["reach"] <= target]
    return (float(ok["k"].iloc[0]) if len(ok) else float(table["k"].iloc[-1])), table


def extra_features(panel, ctx: dict, rows, cols, *, profile_days: int = 10) -> pd.DataFrame:
    """Two occurrence metrics, known at the point's close.

    ``rng60``: the last 60 minutes' high-low range over close, against its own same-clock median
    over the previous ``profile_days`` sessions (log). ``cum_rvol``: the session's volume so far
    against the same clock's expected cumulative volume (log).
    """
    rows, cols = np.asarray(rows), np.asarray(cols)
    n = S["_bars"](60, ctx["bar"])
    codes = ctx["codes"]
    rng = (panel.high.rolling(n, min_periods=n).max() - panel.low.rolling(n, min_periods=n).min()
           ) / panel.close
    rng_ratio = (rng / bt.volume_profile(rng, ctx["tz"], days=profile_days)).to_numpy()
    expected = bt.volume_profile(panel.volume, ctx["tz"], days=profile_days).groupby(codes).cumsum()
    cum = panel.volume.groupby(codes).cumsum()
    rvol = (cum / expected).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"rng60": np.log(rng_ratio[rows, cols]), "cum_rvol": np.log(rvol[rows, cols])}
    return pd.DataFrame({k: np.where(np.isfinite(v), v, np.nan).astype(np.float32)
                         for k, v in out.items()})


def move_table(panel, ctx: dict, rows, cols, frac) -> pd.DataFrame:
    """One row per decision point: position, month, every feature and the move outcome."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    month, _ = S["month_starts"](ctx["index"], ctx["tz"])
    head = pd.DataFrame({"row": rows, "col": cols,
                         "symbol": np.asarray(ctx["symbols"], dtype=object)[cols],
                         "session": ctx["codes"][rows], "month": month[rows],
                         "frac": np.asarray(frac, dtype=float)})
    return pd.concat([head, S["features"](ctx, rows, cols), extra_features(panel, ctx, rows, cols),
                      move_outcomes(panel, ctx, rows, cols, frac)], axis=1)


def metric_auc(table: pd.DataFrame, y, features) -> pd.DataFrame:
    """Within-month AUC of each feature, taken alone, against ``y`` (NaN rows dropped).

    Returns:
        One row per feature: ``auc`` and ``edge`` (``|auc − 0.5|``, the strength whichever way the
        feature points), sorted strongest first.
    """
    y = np.asarray(y, dtype=float)
    rows = []
    for f in features:
        a = monthly_auc(table[f].to_numpy(float), y, table["month"].to_numpy())
        rows.append({"feature": f, "auc": a, "edge": abs(a - 0.5)})
    return pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Move explorer: a few names, every move of MOVE_PCT within WINDOW, and what came before.
# ---------------------------------------------------------------------------
def move_episodes(panel, *, move_pct: float, window_min: int,
                  allow_overnight: bool = False) -> pd.DataFrame:
    """Every non-overlapping move of at least ``move_pct`` in one direction within ``window_min``.

    For every bar ``t`` the anchor is its close; the move is the first bar within the window whose
    high reaches ``anchor × (1 + move_pct)`` (up) or whose low reaches ``anchor × (1 − move_pct)``
    (down). A bar that reaches both is ambiguous and dropped. Many anchors share one hit; the event
    keeps the LATEST of them — the last bar from which the full move still lay ahead, i.e. where the
    move took off. Events are then kept in time order, never overlapping a kept one.

    Args:
        panel: Price panel (any intraday bar).
        move_pct: The move, as a fraction (0.02 = 2%).
        window_min: The most the move may take, in minutes of trading time.
        allow_overnight: Let a window run past the session's close (overnight gaps then count).

    Returns:
        One row per event: ``symbol, col, start, hit`` (bar positions), ``direction`` (+1 / −1),
        ``anchor`` (the start bar's close), ``bars`` and ``minutes`` to the hit, ``start_ts``.
    """
    codes, _ = S["_sessions"](panel.close.index, panel.tz)
    h, lo, c = (f.to_numpy(float) for f in (panel.high, panel.low, panel.close))
    n, m = c.shape
    w = S["_bars"](window_min, panel.bar)
    last = pd.Series(np.arange(n)).groupby(codes).transform("max").to_numpy()
    cap = np.full(n, n - 1) if allow_overnight else last
    hit = np.full((n, m), -1)
    sign = np.zeros((n, m), np.int8)
    up_b, dn_b = c * (1 + move_pct), c * (1 - move_pct)
    open_ = np.isfinite(c)
    for k in range(1, w + 1):
        j = np.arange(n) + k
        valid = (j <= np.minimum(cap, n - 1))[:, None] & open_ & (hit < 0)
        if not valid.any():
            break
        jj = np.minimum(j, n - 1)
        u = valid & (h[jj] >= up_b)
        d = valid & (lo[jj] <= dn_b)
        both = u & d
        hit[valid & (u | d)] = np.broadcast_to(jj[:, None], (n, m))[valid & (u | d)]
        sign[u & ~both], sign[d & ~both] = 1, -1
        open_ &= ~(u | d)                               # resolved (ambiguous ones stay sign 0)
    t0, col = np.nonzero(sign != 0)
    ev = pd.DataFrame({"col": col, "start": t0, "hit": hit[t0, col], "direction": sign[t0, col]})
    ev = ev.sort_values("start").groupby(["col", "hit", "direction"], as_index=False).last()
    keep, last_hit = [], {}
    for r in ev.sort_values("start").itertuples():
        if r.start > last_hit.get(r.col, -1):
            keep.append(r.Index)
            last_hit[r.col] = r.hit
    ev = ev.loc[keep].sort_values(["col", "start"]).reset_index(drop=True)
    ev["symbol"] = np.asarray(panel.symbols, dtype=object)[ev["col"]]
    ev["anchor"] = c[ev["start"], ev["col"]]
    ev["bars"] = ev["hit"] - ev["start"]
    ev["minutes"] = ev["bars"] * S["_minutes_per_bar"](panel.bar)
    ev["start_ts"] = panel.close.index[ev["start"]]
    ev["overnight"] = codes[ev["hit"]] != codes[ev["start"]]
    return ev[["symbol", "col", "start", "hit", "direction", "anchor", "bars", "minutes",
               "start_ts", "overnight"]]


def quiet_controls(panel, events: pd.DataFrame, *, move_pct: float, window_min: int,
                   per_event: int = 2, bucket_minutes: int = 30, seed: int = 0) -> pd.DataFrame:
    """Bars of the same name and half hour from which NO ``move_pct`` move came within the window.

    The comparison group for the features: "an ordinary moment of this stock at this time of day".
    """
    rng = np.random.default_rng(seed)
    codes, bar_no = S["_sessions"](panel.close.index, panel.tz)
    n = len(codes)
    last = pd.Series(np.arange(n)).groupby(codes).transform("max").to_numpy()
    h, lo, c = (f.to_numpy(float) for f in (panel.high, panel.low, panel.close))
    w = S["_bars"](window_min, panel.bar)
    fwd_hi = pd.DataFrame(h[::-1]).rolling(w, min_periods=1).max().to_numpy()[::-1]
    fwd_lo = pd.DataFrame(lo[::-1]).rolling(w, min_periods=1).min().to_numpy()[::-1]
    nxt_hi = np.vstack([fwd_hi[1:], np.full((1, h.shape[1]), np.nan)])
    nxt_lo = np.vstack([fwd_lo[1:], np.full((1, h.shape[1]), np.nan)])
    room = (last - np.arange(n)) >= w                          # a full window inside the session
    quiet = room[:, None] & (nxt_hi < c * (1 + move_pct)) & (nxt_lo > c * (1 - move_pct))
    mpb = S["_minutes_per_bar"](panel.bar)
    bucket = (bar_no * mpb) // bucket_minutes
    out = []
    for r in events.itertuples():
        pool = np.flatnonzero(quiet[:, r.col] & (bucket == bucket[r.start]))
        if len(pool):
            for t in rng.choice(pool, size=min(per_event, len(pool)), replace=False):
                out.append({"symbol": r.symbol, "col": r.col, "start": int(t), "direction": 0})
    return pd.DataFrame(out)


def explore_features(panel, *, profile_days: int = 10, stat: str = "mean", atr_days: int = 14,
                     short_minutes: int = 15, pre_minutes: int = 60) -> dict:
    """Candidate identifier features, each known at the bar's close (dense ``bar × symbol`` frames).

    The "normal for this time of day" baselines are the ``stat`` over the previous ``profile_days``
    sessions at the same clock time (``bt.volume_profile``; the current session excluded).

    Returns:
        ``{name: DataFrame}`` — see the notebook's feature table for what each one means.
    """
    tz, bar = panel.tz, panel.bar
    c, h, lo, v, o = panel.close, panel.high, panel.low, panel.volume, panel.open_
    codes, bar_no = S["_sessions"](c.index, tz)
    ks = S["_bars"](short_minutes, bar)
    kp = S["_bars"](pre_minutes, bar)
    bar_rng = (h - lo) / c

    def tod(frame):
        return bt.volume_profile(frame, tz, days=profile_days, stat=stat)

    def rows_ok(cond):                         # a per-bar condition, full frame shape
        return np.broadcast_to(np.asarray(cond)[:, None], c.shape)

    def within(frame):                         # blank bars whose window crosses the open
        return frame.where(rows_ok(bar_no >= ks - 1))

    vol_k = v.rolling(ks, min_periods=ks).sum()
    rng_k = (h.rolling(ks, min_periods=ks).max() - lo.rolling(ks, min_periods=ks).min()) / c
    rng_pre = (h.rolling(kp, min_periods=kp).max() - lo.rolling(kp, min_periods=kp).min()) / c
    atr_d = pd.DataFrame(S["_session_atr"](panel, codes, atr_days).to_numpy()[codes],
                         index=c.index, columns=c.columns)
    prev_close = pd.DataFrame(c.groupby(codes).last().shift(1).to_numpy()[codes], index=c.index,
                              columns=c.columns)
    sess_open = o.groupby(codes).transform("first")
    s_hi, s_lo = h.groupby(codes).cummax(), lo.groupby(codes).cummin()
    vwap = (c * v).groupby(codes).cumsum() / v.groupby(codes).cumsum()
    prev_hi = pd.DataFrame(h.groupby(codes).max().shift(1).to_numpy()[codes], index=c.index,
                           columns=c.columns)
    prev_lo = pd.DataFrame(lo.groupby(codes).min().shift(1).to_numpy()[codes], index=c.index,
                           columns=c.columns)
    daily_close = c.groupby(codes).last()
    daily_vol = v.groupby(codes).sum()

    def daily(frame):                          # a session-grid frame (known at the open)
        return pd.DataFrame(frame.to_numpy()[codes], index=c.index, columns=c.columns)

    ret5 = daily(daily_close.shift(1) / daily_close.shift(6) - 1)
    mpb = S["_minutes_per_bar"](bar)
    n_left = pd.Series(codes).groupby(codes).transform("size").to_numpy() - 1 - bar_no
    clock = pd.DataFrame(np.broadcast_to((bar_no * mpb)[:, None], c.shape), index=c.index,
                         columns=c.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = {
            "vol_vs_tod": within(vol_k / tod(v).rolling(ks, min_periods=ks).sum()),
            "range_vs_tod": within(rng_k / tod(within(rng_k))),
            "bar_vol_vs_tod": v / tod(v),
            "bar_range_vs_tod": bar_rng / tod(bar_rng),
            "cum_vol_vs_tod": v.groupby(codes).cumsum() / tod(v).groupby(codes).cumsum(),
            "pre_range_vs_tod": rng_pre.where(rows_ok(bar_no >= kp - 1))
            / tod(rng_pre.where(rows_ok(bar_no >= kp - 1))),
            "atr_daily_pct": atr_d / c,
            "atr_intraday_pct": bt.atr(panel, 14) / c,
            "ret_pre": (c / c.shift(kp) - 1).where(rows_ok(bar_no >= kp)),
            "ret_session": c / sess_open - 1,
            "gap_pct": sess_open / prev_close - 1,
            "vwap_dist_atr": (c - vwap) / atr_d,
            "day_pos": (c - s_lo) / (s_hi - s_lo),
            "to_prev_high_atr": (c - prev_hi) / atr_d,
            "to_prev_low_atr": (c - prev_lo) / atr_d,
            "ret_5d": ret5,
            "ret_20d": daily(daily_close.shift(1) / daily_close.shift(21) - 1),
            "prev_day_ret": daily(daily_close.shift(1) / daily_close.shift(2) - 1),
            "prev_day_vol_vs_avg": daily(daily_vol.shift(1)
                                         / daily_vol.shift(2).rolling(20, min_periods=5).mean()),
            "prev_day_range_atr": (prev_hi - prev_lo) / atr_d,
            "minutes_since_open": clock,
            "minutes_to_close": pd.DataFrame(np.broadcast_to((n_left * mpb)[:, None], c.shape),
                                             index=c.index, columns=c.columns),
        }
    return {k: f.replace([np.inf, -np.inf], np.nan) for k, f in feats.items()}


def at_points(feats: dict, rows, cols) -> pd.DataFrame:
    """The feature values at ``(rows, cols)``, one row per point."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    return pd.DataFrame({k: f.to_numpy(float)[rows, cols] for k, f in feats.items()})


def _segments(panel, allow_overnight: bool) -> list[tuple[int, int]]:
    """``[start, end)`` bar ranges a run may live in: each session, or the whole panel."""
    codes, _ = S["_sessions"](panel.close.index, panel.tz)
    if allow_overnight:
        return [(0, len(codes))]
    edges = np.r_[0, np.flatnonzero(np.diff(codes)) + 1, len(codes)]
    return list(zip(edges[:-1], edges[1:], strict=True))


def zigzag(hh: np.ndarray, ll: np.ndarray, reversal_pct: float) -> list[tuple[int, str, bool]]:
    """Alternating swing points of one price path; a swing ends on a ``reversal_pct`` pullback.

    Walks the bars once. Until the first reversal the direction is open; after it, an up swing
    extends while highs make new highs and ends (its high confirmed as a swing ``"H"``) on the
    first bar whose low is ``reversal_pct`` below that high — and the mirror for a down swing.
    The last extreme, never reversed before the path ends, is returned flagged provisional.

    Args:
        hh: Highs of one path.
        ll: Lows of one path (NaN bars are skipped).
        reversal_pct: The pullback that ends a swing (0.01 = 1%).

    Returns:
        ``[(bar, "L" | "H", provisional), ...]`` in time order, alternating.
    """
    idx = np.flatnonzero(np.isfinite(hh) & np.isfinite(ll))
    if len(idx) < 2:
        return []
    piv, d = [], 0
    hi = lo = ext = idx[0]
    for i in idx[1:]:
        if d == 0:
            hi = i if hh[i] > hh[hi] else hi
            lo = i if ll[i] < ll[lo] else lo
            if hh[hi] >= ll[lo] * (1 + reversal_pct):
                if lo < hi:
                    piv.append((lo, "L", False))
                    d, ext = 1, hi
                else:
                    piv.append((hi, "H", False))
                    d, ext = -1, lo
        elif d == 1:
            if hh[i] > hh[ext]:                    # a tie keeps the EARLIER extreme (double top)
                ext = i
            elif ll[i] <= hh[ext] * (1 - reversal_pct):
                piv.append((ext, "H", False))
                d, ext = -1, i
        else:
            if ll[i] < ll[ext]:
                ext = i
            elif hh[i] >= ll[ext] * (1 + reversal_pct):
                piv.append((ext, "L", False))
                d, ext = 1, i
    if d != 0:
        piv.append((ext, "H" if d == 1 else "L", True))
    return piv


def swing_runs(panel, *, reversal_pct: float, min_run_pct: float, max_run_min: int,
               confirm_pct: float, allow_overnight: bool = False) -> pd.DataFrame:
    """Whole runs: swing low → swing high (up) or high → low (down) of at least ``min_run_pct``.

    Runs are the legs of :func:`zigzag` — a run ends only when price pulls back ``reversal_pct``,
    so one rally is one run, never a chain of ``min_run_pct`` slices. A run is kept when its first
    ``min_run_pct`` is reached within ``max_run_min`` of its onset (the speed rule), whatever
    it does after. Per run: the **onset** (the swing extreme it starts from — known only with
    hindsight), the **confirmation** (the first bar whose CLOSE is ``confirm_pct`` past the
    onset — knowable live), the bar that reached ``min_run_pct`` and the **peak** (the run's end).

    Args:
        panel: Price panel.
        reversal_pct: The pullback that ends a run.
        min_run_pct: The smallest run kept (the move, e.g. 0.02).
        max_run_min: The first ``min_run_pct`` must come within this many minutes of the onset.
        confirm_pct: Distance from the onset that confirms the run (e.g. 0.005).
        allow_overnight: Let runs cross the close (default: each session on its own).

    Returns:
        One row per run: ``symbol, col, direction, onset, confirm, hit, peak`` (bar positions),
        ``base`` (the onset's low / high), ``size`` (the whole run), ``to_confirm_min``,
        ``to_hit_min``, ``duration_min``, ``confirm_after_hit``, ``ended_by_close``.
    """
    h, lo, c = (f.to_numpy(float) for f in (panel.high, panel.low, panel.close))
    mpb = S["_minutes_per_bar"](panel.bar)
    max_b = S["_bars"](max_run_min, panel.bar)
    out = []
    for col in range(h.shape[1]):
        for a, b in _segments(panel, allow_overnight):
            piv = zigzag(h[a:b, col], lo[a:b, col], reversal_pct)
            for (i0, k0, _), (i1, _, prov) in zip(piv, piv[1:], strict=False):
                up = k0 == "L"
                onset, peak = a + i0, a + i1
                base = lo[onset, col] if up else h[onset, col]
                ext = h[peak, col] if up else lo[peak, col]
                size = ext / base - 1 if up else 1 - ext / base
                if size < min_run_pct:
                    continue
                path = slice(onset + 1, peak + 1)
                target = base * (1 + min_run_pct) if up else base * (1 - min_run_pct)
                reach = (np.flatnonzero(h[path, col] >= target) if up
                         else np.flatnonzero(lo[path, col] <= target))
                if not len(reach) or reach[0] + 1 > max_b:
                    continue
                hit = onset + 1 + int(reach[0])
                level = base * (1 + confirm_pct) if up else base * (1 - confirm_pct)
                closes = c[onset:peak + 1, col]
                cf = np.flatnonzero(closes >= level) if up else np.flatnonzero(closes <= level)
                confirm = onset + int(cf[0]) if len(cf) else peak
                out.append({"col": col, "direction": 1 if up else -1, "onset": onset,
                            "confirm": confirm, "hit": hit, "peak": peak, "base": base,
                            "size": size, "ended_by_close": bool(prov)})
    runs = pd.DataFrame(out, columns=["col", "direction", "onset", "confirm", "hit", "peak", "base",
                                      "size", "ended_by_close"])
    runs["symbol"] = np.asarray(panel.symbols, dtype=object)[runs["col"].to_numpy(int)]
    for name, frm, to in (("to_confirm_min", "onset", "confirm"), ("to_hit_min", "onset", "hit"),
                          ("duration_min", "onset", "peak")):
        runs[name] = (runs[to] - runs[frm]) * mpb
    runs["confirm_after_hit"] = runs["confirm"] >= runs["hit"]
    return runs.sort_values(["col", "onset"]).reset_index(drop=True)


def confirmations(panel, *, confirm_pct: float, lookback_min: int, min_run_pct: float,
                  max_run_min: int, allow_overnight: bool = False) -> pd.DataFrame:
    """Every moment a move got ``confirm_pct`` off a recent extreme — the ones that ran AND failed.

    Causal: at bar ``t`` the reference is the lowest low (up) / highest high (down) of the last
    ``lookback_min`` minutes up to ``t``, same session; a confirmation is the bar whose CLOSE
    CROSSES ``confirm_pct`` past it — the previous close was still short of that level, or the
    extreme is this very bar (one per reference extreme and side). Without the crossing rule an
    older extreme leaving the window hands price a new, lower reference it is already far past,
    and "confirms" a move that never happened (65% of 5m, 78% of 1m candidates on three
    names). A confirming bar whose close is already AT or PAST the target is dropped too: the run
    happened inside that one bar, there is nothing left to confirm, and booking it as a "run"
    exits at a target below the entry (9% of 5-minute confirmations in the first half hour, 15%
    at 8-9 am, news bars; < 1% later in the day). Bars without a print (NaN, e.g. masked
    placeholder minutes) are skipped: the previous close is the last one that printed. Its
    outcome, scanned from the
    next bar: **run** (price reached ``min_run_pct`` past the reference within ``max_run_min``),
    **failed** (price came back through the reference first) or **stalled** (neither). This is the
    event a live identifier sees, with its natural control group: the confirmations that failed.

    Args:
        panel: Price panel.
        confirm_pct: Distance from the reference that counts as a confirmation.
        lookback_min: The window the reference extreme is taken over.
        min_run_pct: The run to reach, measured from the reference.
        max_run_min: Minutes from the reference extreme in which the run must be reached.
        allow_overnight: Let the lookback and the outcome cross the close.

    Returns:
        One row per confirmation: ``symbol, col, direction, ref`` (the extreme's bar), ``t`` (the
        confirmation bar), ``base``, ``entry`` (the confirmation close), ``outcome`` (1 run, −1
        failed, 0 stalled), ``end`` (bar it resolved), ``r_multiple`` (reward ÷ risk from the
        confirmation close: the run's target over the distance back to the reference) and ``ret``
        — the trade from the confirmation CLOSE (target, the reference as the stop, or the close
        at ``end``), signed as positioned, before costs. Exploratory: a live order would fill at
        the next open and a gap can fill beyond the stop.
    """
    h, lo, c = (f.to_numpy(float) for f in (panel.high, panel.low, panel.close))
    lb = S["_bars"](lookback_min, panel.bar)
    max_b = S["_bars"](max_run_min, panel.bar)
    out = []
    for col in range(h.shape[1]):
        for a, b in _segments(panel, allow_overnight):
            # the last close that printed up to each bar (NaN bars carry the previous one)
            last_c = pd.Series(c[a:b, col]).ffill().to_numpy()
            prev_c = np.r_[np.nan, last_c[:-1]]
            for up in (True, False):
                src = lo[a:b, col] if up else h[a:b, col]
                seen = set()
                for t in range(b - a):
                    w0 = max(0, t - lb + 1)
                    window = src[w0:t + 1]
                    if not np.isfinite(window).any() or not np.isfinite(c[a + t, col]):
                        continue
                    k = w0 + int(np.nanargmin(window) if up else np.nanargmax(window))
                    if k in seen:
                        continue
                    base = src[k]
                    level = base * (1 + confirm_pct) if up else base * (1 - confirm_pct)
                    if (c[a + t, col] >= level) if up else (c[a + t, col] <= level):
                        seen.add(k)
                        # A real CROSSING only: the previous close was still short of the level
                        # (or the low / high is this very bar). Otherwise the reference just
                        # changed because an older extreme left the window while price was
                        # already beyond the level — no move off a low happened here.
                        prev = prev_c[t]
                        if k != t and not ((prev < level) if up else (prev > level)):
                            continue
                        target = base * (1 + min_run_pct) if up else base * (1 - min_run_pct)
                        entry = c[a + t, col]
                        if (entry >= target) if up else (entry <= target):
                            continue                       # the whole run happened in this bar
                        last = min(b - 1 - a, k + max_b)
                        fh, fl = h[a + t + 1:a + last + 1, col], lo[a + t + 1:a + last + 1, col]
                        win = np.flatnonzero(fh >= target) if up else np.flatnonzero(fl <= target)
                        lose = np.flatnonzero(fl < base) if up else np.flatnonzero(fh > base)
                        w_ = win[0] if len(win) else np.inf
                        l_ = lose[0] if len(lose) else np.inf
                        if w_ == l_ == np.inf:
                            outcome, end = 0, a + last                   # stalled
                        else:                                            # a tie counts as failed
                            outcome = 1 if w_ < l_ else -1
                            end = a + t + 1 + int(min(w_, l_))
                        risk = abs(entry - base)
                        exit_px = {1: target, -1: base, 0: last_c[end - a]}[outcome]
                        sign = 1 if up else -1
                        out.append({"col": col, "direction": sign, "ref": a + k,
                                    "t": a + t, "base": base, "entry": entry,
                                    "outcome": outcome, "end": end,
                                    "r_multiple": abs(target - entry) / risk if risk else np.nan,
                                    "ret": sign * (exit_px / entry - 1)})
    conf = pd.DataFrame(out, columns=["col", "direction", "ref", "t", "base", "entry", "outcome",
                                      "end", "r_multiple", "ret"])
    conf["symbol"] = np.asarray(panel.symbols, dtype=object)[conf["col"].to_numpy(int)]
    return conf.sort_values(["col", "t"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Extended hours: the same prints with and without the pre- and post-market.
# ---------------------------------------------------------------------------
_OHLC = ("open_", "high", "low", "close")


def mask_placeholders(panel):
    """IBKR 1-minute placeholder bars (volume 0, flat at the previous close) as no print.

    A minute with fewer than 100 shares is stored as a copy of the previous close; before 7:00 ET
    that is about half of all minutes even for the busiest names. Left in, the flat copies drag
    every range down and hand a resampled candle a stale open nobody traded. Volume stays 0.

    Args:
        panel: A 1-minute panel.

    Returns:
        The panel with open / high / low / close NaN wherever volume is 0.
    """
    none = panel.volume == 0
    return replace(panel, **{k: getattr(panel, k).mask(none) for k in _OHLC})


def mask_late_reports(panel, *, start: str = "08:00", minutes: int = 10):
    """The minutes from 08:00 New York as no print: overnight trades reported late, at stale prices.

    08:00 is when FINRA's trade reporting facilities open, and trades done while they were closed
    (the overnight sessions) are reported then — most likely why IBKR's bars carry them, stamped at
    the REPORT time with the overnight price. Measured on eight pre-market names: 82% of 08:00
    minutes have a wick > 25 bp beyond the prints either side (0.2% at 07:59), 44% at 08:01, 27%
    at 08:02, under 2% from 08:07. Left in, they are the pre-market's high or low, the onset of a
    run and the stop of a trade far more often than chance: they manufactured a −33 bp "pre-market
    reversal" (t −13.5) that is gone once they are masked. Volume stays.

    Args:
        panel: A 1-minute panel.
        start: First masked minute, exchange-local ``HH:MM``.
        minutes: How many minutes from ``start``.

    Returns:
        The panel with open / high / low / close NaN in those minutes.
    """
    loc = panel.close.index.tz_convert(panel.tz)
    first = int(start[:2]) * 60 + int(start[3:])
    clock = np.asarray(loc.hour * 60 + loc.minute)
    rows = (clock >= first) & (clock < first + minutes)
    hit = np.repeat(rows[:, None], panel.close.shape[1], axis=1)
    return replace(panel, **{k: getattr(panel, k).mask(hit) for k in _OHLC})


def session_parts(panel) -> np.ndarray:
    """Each row's part of its exchange-local day: ``"pre"``, ``"rth"`` or ``"post"``.

    A day's regular session runs from its first to its last row with a regular-hours PRINT (the
    loader reads a missing bar's ``rth`` as True, so an unprinted row cannot decide it). The stored
    flag is a clock rule: on a half day (Jul 3, Nov 28, Dec 24) 13:00-16:00 is after-hours trading
    at a tenth of the volume, yet reads as regular hours — three days a year, left as stored.

    Args:
        panel: A ``session="both"`` panel (``panel.rth`` set).

    Returns:
        One label per row.

    Raises:
        ValueError: If the panel has no session flags.
    """
    if panel.rth is None:
        raise ValueError("session_parts needs a session='both' panel")
    printed = (panel.rth & panel.close.notna()).any(axis=1).to_numpy()
    pos = np.arange(len(printed))
    day = panel.close.index.tz_convert(panel.tz).date
    first = pd.Series(np.where(printed, pos, len(pos))).groupby(day).transform("min").to_numpy()
    last = pd.Series(np.where(printed, pos, -1)).groupby(day).transform("max").to_numpy()
    return np.where(pos < first, "pre", np.where(pos > last, "post", "rth"))


def split_sessions(panel):
    """The extended panel with an exact ``rth`` flag, and its regular-hours rows as a panel.

    Both views hold the SAME prints, so any difference between them is the extended hours alone.

    Args:
        panel: A ``session="both"`` panel.

    Returns:
        ``(extended, regular)``.
    """
    rth = session_parts(panel) == "rth"
    ext = replace(panel, rth=pd.DataFrame(np.repeat(rth[:, None], panel.close.shape[1], axis=1),
                                          index=panel.close.index, columns=panel.close.columns))
    return ext, replace(ext, **{k: getattr(ext, k).iloc[rth]
                                for k in (*_OHLC, "volume", "rth")})


#: Knowable at the 9:30 open WITHOUT extended data (the open print, the regular sessions).
OPEN_BASE = ("gap_atr", "prev_day_ret", "ret_5d", "atr_pct", "prev_vol_rel")
#: Knowable only from the extended hours (the open print aside), by 9:29.
OPEN_EXT = ("ah_ret", "ah_vol_rel", "pm_ret", "pm_trend", "open_vs_pm", "pm_vol_rel",
            "pm_vol_share", "pm_range_atr", "pm_pos")


def open_table(panel, *, move_pct: float, window_min: int, trend_min: int = 30,
               atr_days: int = 14, profile_days: int = 10) -> pd.DataFrame:
    """One row per name and day: what was known at the open, and what the first minutes did.

    The trade starts at the day's first regular-hours OPEN print. Its outcome is a race over the
    first ``window_min`` minutes of regular hours: ``+1`` if the high reaches ``open × (1 +
    move_pct)`` first, ``−1`` if the low reaches ``open × (1 − move_pct)`` first, ``0`` if neither,
    NaN when one bar reaches both. ``ret`` is the last printed close of the window over the open.

    Features: :data:`OPEN_BASE` from regular hours (``gap_atr`` = gap ÷ ATR, ATR = Wilder over
    ``atr_days`` regular sessions, known at the open; ``prev_vol_rel`` = yesterday's volume ÷ its
    20-day mean) and :data:`OPEN_EXT` from the extended hours: yesterday's after-hours return and
    volume (``ah_*``, volume ÷ its own ``profile_days`` mean), the pre-market's return from the last
    close (``pm_ret`` — the gap seen early), its last ``trend_min`` minutes (``pm_trend``), the open
    print against the last pre-market print (``open_vs_pm``), its volume against its own
    ``profile_days`` mean (``pm_vol_rel``) and against a regular session's (``pm_vol_share``), its
    range in ATRs and where the last print sits in it (``pm_pos``, 0 = low, 1 = high).

    Args:
        panel: The extended panel of :func:`split_sessions` (any intraday bar).
        move_pct: The race's barrier.
        window_min: The race's length, minutes of regular hours.
        trend_min: The pre-market's closing stretch.
        atr_days: ATR smoothing in sessions.
        profile_days: Sessions in the extended-volume baselines.

    Returns:
        ``day, month, symbol``, the features, ``gap``, ``race`` and ``ret``.
    """
    loc = panel.close.index.tz_convert(panel.tz)
    day = np.asarray(loc.date)
    rth = panel.rth.iloc[:, 0].to_numpy()
    parts = np.where(rth, "rth", np.where(~pd.Series(rth).groupby(day).cummax().to_numpy(),
                                          "pre", "post"))
    open_ts = pd.Series(np.where(rth, loc.asi8, np.iinfo(np.int64).max)).groupby(day).transform(
        "min").to_numpy()
    late_pre = (parts == "pre") & (loc.asi8 >= open_ts - trend_min * 60_000_000_000)

    def agg(frame, mask, how):
        return frame[mask].groupby(day[mask]).agg(how)

    c, h, lo, v = panel.close, panel.high, panel.low, panel.volume
    r = parts == "rth"
    opn, cls = agg(panel.open_, r, "first"), agg(c, r, "last")
    days = opn.index
    hi, low, vol = agg(h, r, "max"), agg(lo, r, "min"), agg(v, r, "sum")
    pre, post = parts == "pre", parts == "post"
    pm_last = agg(c, pre, "last").reindex(days)
    pm_mid = agg(c, pre & ~late_pre, "last").reindex(days)
    pm_hi, pm_lo = agg(h, pre, "max").reindex(days), agg(lo, pre, "min").reindex(days)
    pm_vol = agg(v, pre, "sum").reindex(days)
    ah_last, ah_vol = agg(c, post, "last").reindex(days), agg(v, post, "sum").reindex(days)
    prev = cls.shift(1)
    tr = np.fmax(np.fmax(hi - low, (hi - prev).abs()), (low - prev).abs())
    atr = tr.ewm(alpha=1.0 / atr_days, min_periods=atr_days).mean().shift(1)

    def mean_of(frame, n):                     # the mean of the n days BEFORE each day
        return frame.shift(1).rolling(n, min_periods=min(5, n)).mean()

    gap = opn / prev - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = {
            "gap_atr": gap / (atr / prev),
            "prev_day_ret": prev / cls.shift(2) - 1,
            "ret_5d": prev / cls.shift(6) - 1,
            "atr_pct": atr / prev,
            "prev_vol_rel": vol.shift(1) / mean_of(vol.shift(1), 20),
            "ah_ret": ah_last.shift(1) / prev - 1,
            "ah_vol_rel": ah_vol.shift(1) / mean_of(ah_vol.shift(1), profile_days),
            "pm_ret": pm_last / prev - 1,
            "pm_trend": pm_last / pm_mid - 1,
            "open_vs_pm": opn / pm_last - 1,
            "pm_vol_rel": pm_vol / mean_of(pm_vol, profile_days),
            "pm_vol_share": pm_vol / mean_of(vol, 20),
            "pm_range_atr": (pm_hi - pm_lo) / atr,
            "pm_pos": (pm_last - pm_lo) / (pm_hi - pm_lo),
        }
    w = S["_bars"](window_min, panel.bar)
    first_row = pd.Series(np.flatnonzero(r)).groupby(day[r]).first()
    H, L, C = h.to_numpy(float), lo.to_numpy(float), c.to_numpy(float)
    race = np.full(opn.shape, np.nan)
    ret = np.full(opn.shape, np.nan)
    for i, d in enumerate(days):
        a = first_row[d]
        span = slice(a, min(a + w, len(r)))
        o = opn.loc[d].to_numpy(float)
        up, dn = H[span] >= o * (1 + move_pct), L[span] <= o * (1 - move_pct)
        tu = np.where(up.any(0), up.argmax(0), np.iinfo(np.int32).max)
        td = np.where(dn.any(0), dn.argmax(0), np.iinfo(np.int32).max)
        race[i] = np.where(tu < td, 1.0, np.where(td < tu, -1.0,
                                                  np.where(up.any(0), np.nan, 0.0)))
        ret[i] = pd.DataFrame(C[span]).ffill().to_numpy()[-1] / o - 1
    frames = {**feats, "gap": gap,
              "race": pd.DataFrame(race, index=days, columns=opn.columns),
              "ret": pd.DataFrame(ret, index=days, columns=opn.columns)}
    out = pd.concat({k: f.stack(future_stack=True) for k, f in frames.items()}, axis=1)
    out = out.rename_axis(["day", "symbol"]).reset_index()
    out = out[np.isfinite(out["atr_pct"]) & np.isfinite(out["ret"])]
    out.insert(1, "month", pd.to_datetime(out["day"]).dt.to_period("M").astype(str))
    return out.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def rank_ic(x, y, days) -> dict:
    """Spearman rank correlation of ``x`` with ``y``, with a t clustered by day.

    The t is :func:`day_t` of the slope of rank(y) on rank(x) (both centred) — the rank
    correlation up to scale — because rows on one day share its news and are not independent.

    Returns:
        ``{"ic", "t", "n"}``.
    """
    x, y, days = np.asarray(x, float), np.asarray(y, float), np.asarray(days)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"ic": np.nan, "t": np.nan, "n": int(ok.sum())}
    rx = pd.Series(x[ok]).rank(pct=True).to_numpy() - 0.5
    ry = pd.Series(y[ok]).rank(pct=True).to_numpy() - 0.5
    return {"ic": float(np.corrcoef(rx, ry)[0, 1]),
            "t": day_t(rx * ry / np.mean(rx * rx), days[ok])["t"], "n": int(ok.sum())}


def monthly_walk(table: pd.DataFrame, cols, y: str, *, warm_months: int = 2) -> pd.Series:
    """Out-of-sample P(y) for every row: a logistic model fit on all EARLIER months only.

    Args:
        table: One row per case, with a ``month`` column.
        cols: The features.
        y: The 0/1 target column.
        warm_months: Months used only for training before the first prediction.

    Returns:
        P(y = 1), NaN through the warm-up months.
    """
    p = pd.Series(np.nan, index=table.index)
    months = sorted(table["month"].unique())
    for m in months[warm_months:]:
        fit = table[(table["month"] < m) & table[y].notna()]
        test = table["month"] == m
        model = S["make_model"]("lr").fit(fit[list(cols)], fit[y].astype(int))
        p[test] = model.predict_proba(table.loc[test, list(cols)])[:, 1]
    return p


def match_runs(inner: pd.DataFrame, inner_index, outer: pd.DataFrame,
               outer_index) -> pd.DataFrame:
    """For each ``inner`` run, the ``outer`` run of the same name and direction that contains it.

    Contains = the outer run's onset is at or before the inner run's onset timestamp and its peak
    at or after it. Used to follow a regular-hours run into the extended view of the same prints.

    Args:
        inner: Runs of :func:`swing_runs` (e.g. regular hours).
        inner_index: The panel index their bar positions refer to.
        outer: Runs of :func:`swing_runs` (e.g. extended hours).
        outer_index: Their panel index.

    Returns:
        ``inner`` plus ``onset_ts``, ``peak_ts`` and the containing run's ``x_onset_ts``,
        ``x_peak_ts``, ``x_base``, ``x_size`` (NaN / NaT where none contains it).
    """
    a = inner.assign(onset_ts=inner_index[inner["onset"]], peak_ts=inner_index[inner["peak"]])
    b = outer.assign(x_onset_ts=outer_index[outer["onset"]], x_peak_ts=outer_index[outer["peak"]])
    b = b.rename(columns={"base": "x_base", "size": "x_size"})
    m = a.reset_index().merge(b[["col", "direction", "x_onset_ts", "x_peak_ts", "x_base",
                                 "x_size"]], on=["col", "direction"])
    m = m[(m["x_onset_ts"] <= m["onset_ts"]) & (m["x_peak_ts"] >= m["onset_ts"])]
    m = m.drop_duplicates("index").set_index("index")
    return a.join(m[["x_onset_ts", "x_peak_ts", "x_base", "x_size"]])


# ---------------------------------------------------------------------------
# Move detectors: will price rise (LONG) / fall (SHORT) by a set move within a window?
# ---------------------------------------------------------------------------
#: explore_features that divide by the CURRENT close — at a low they read high, which leaks
#: location into a volatility measure; detector_features rescales them to the previous close.
PRICE_DENOMINATOR = ("atr_daily_pct", "atr_intraday_pct", "range_vs_tod", "pre_range_vs_tod",
                     "bar_range_vs_tod")


def _sess_roll(frame, codes, fn: str, k: int):
    """A rolling ``fn`` over ``k`` bars that never reaches into the previous session."""
    r = getattr(frame.groupby(codes).rolling(k, min_periods=max(2, k // 2)), fn)()
    return r.reset_index(level=0, drop=True).sort_index()


def detector_features(panel) -> dict:
    """The move detectors' inputs: every value known at the bar's close, float32 (bars, names).

    :func:`explore_features` (the five of :data:`PRICE_DENOMINATOR` rescaled to the previous
    session's close) plus multi-scale extras: returns over 1-120 minutes and realized volatility
    over 15/60/240 minutes (both in daily-ATR units), their ratio, volume over 5/60 minutes
    against its clock-time normal, the position in the 5- and 20-day range and the distance to
    those highs, 30/120-minute trend slopes, the distance to 20/120-bar EMAs, signed flow (close
    location x volume) over 15/60 minutes, the up-bar volume share, the previous day's close
    position and last-hour return, the 5- vs 20-day volatility ratio, the other names' 15/60-minute
    return (leave-one-out) and the weekday.

    Args:
        panel: A 1-minute regular-hours panel.

    Returns:
        ``{name: array}``.
    """
    c, h, lo, v = panel.close, panel.high, panel.low, panel.volume
    codes, _ = S["_sessions"](c.index, panel.tz)
    F = explore_features(panel)
    prev_close = c.groupby(codes).last().shift(1).to_numpy()[codes]
    for f in PRICE_DENOMINATOR:
        F[f] = F[f] * (c.to_numpy() / prev_close)
    out = {name: f.to_numpy(np.float32) for name, f in F.items()}
    atr_d = F["atr_daily_pct"]
    lc = np.log(c)
    first = np.r_[True, codes[1:] != codes[:-1]]
    r1 = lc.diff()
    r1.iloc[first] = np.log(c / panel.open_).iloc[first]      # no overnight gap in 1-min returns
    for k in (1, 5, 15, 30, 60, 120):
        out[f"ret_{k}m"] = ((lc - lc.groupby(codes).shift(k)) / atr_d).to_numpy(np.float32)
    sq = r1 ** 2
    for k in (15, 60):
        out[f"rv_{k}m"] = (np.sqrt(_sess_roll(sq, codes, "sum", k)) / atr_d).to_numpy(np.float32)
    out["rv_240m"] = (np.sqrt(sq.rolling(240, min_periods=120).sum()) / atr_d).to_numpy(np.float32)
    out["rv_ratio"] = out["rv_15m"] / out["rv_240m"]
    profile = bt.volume_profile(v, panel.tz, days=10, stat="mean")
    for k in (5, 60):
        out[f"vol_{k}m_vs_tod"] = (_sess_roll(v, codes, "sum", k)
                                   / _sess_roll(profile, codes, "sum", k)).to_numpy(np.float32)
    day_hi, day_lo = h.groupby(codes).max(), lo.groupby(codes).min()
    run_hi, run_lo = h.groupby(codes).cummax().to_numpy(), lo.groupby(codes).cummin().to_numpy()
    for nd in (5, 20):
        past_hi = day_hi.shift(1).rolling(nd - 1, min_periods=1).max().to_numpy()[codes]
        past_lo = day_lo.shift(1).rolling(nd - 1, min_periods=1).min().to_numpy()[codes]
        hi, low = np.fmax(past_hi, run_hi), np.fmin(past_lo, run_lo)
        out[f"pos_{nd}d"] = ((c.to_numpy() - low) / (hi - low)).astype(np.float32)
        out[f"to_{nd}d_high_atr"] = ((c.to_numpy() / hi - 1) / atr_d.to_numpy()).astype(np.float32)
    t = pd.DataFrame(np.broadcast_to(np.arange(len(c), dtype=float)[:, None], c.shape),
                     index=c.index, columns=c.columns)
    for k in (30, 120):                  # least-squares slope of log price, as a move in daily ATRs
        mt, my = _sess_roll(t, codes, "mean", k), _sess_roll(lc, codes, "mean", k)
        slope = ((_sess_roll(t * lc, codes, "mean", k) - mt * my)
                 / (_sess_roll(t * t, codes, "mean", k) - mt * mt))
        out[f"slope_{k}m"] = (slope * k / atr_d).to_numpy(np.float32)
    atr_i = bt.atr(panel, 14)
    for span in (20, 120):
        ema = c.groupby(codes).transform(lambda s, n=span: s.ewm(span=n, min_periods=n // 2).mean())
        out[f"ema{span}_dist"] = ((c - ema) / atr_i).to_numpy(np.float32)
    clv = (((c - lo) - (h - c)) / (h - lo).replace(0, np.nan)).fillna(0)
    for k in (15, 60):
        out[f"flow_{k}m"] = (_sess_roll(clv * v, codes, "sum", k)
                             / _sess_roll(v, codes, "sum", k)).to_numpy(np.float32)
    up_v = (c > panel.open_).astype(float) * v
    out["upvol_30m"] = (_sess_roll(up_v, codes, "sum", 30)
                        / _sess_roll(v, codes, "sum", 30)).to_numpy(np.float32)
    day_c = c.groupby(codes).last()
    out["prev_close_pos"] = ((day_c - day_lo) / (day_hi - day_lo)).shift(1).to_numpy()[codes]
    ends = np.r_[np.flatnonzero(np.diff(codes)), len(codes) - 1]
    starts = np.r_[0, ends[:-1] + 1]
    cn = c.to_numpy()
    last_hour = cn[ends] / cn[np.maximum(starts, ends - 60)] - 1          # per session
    out["prev_last_hour_ret"] = np.vstack([np.full((1, c.shape[1]), np.nan),
                                           last_hour[:-1]])[codes]
    d_ret = np.log(day_c).diff()
    vol_ratio = (d_ret.rolling(5).std() / d_ret.rolling(20).std()).shift(1)
    out["daily_vol_ratio"] = vol_ratio.to_numpy()[codes]
    for k in (15, 60):                   # the other names over the same minutes (leave-one-out)
        rk = (lc - lc.groupby(codes).shift(k)).to_numpy()
        own = np.isfinite(rk)
        total, count = np.nansum(rk, axis=1, keepdims=True), own.sum(1, keepdims=True)
        others = count - own
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"mkt_{k}m"] = np.where(others > 0, (total - np.where(own, rk, 0)) / others, np.nan)
    weekday = np.asarray(c.index.tz_convert(panel.tz).weekday, dtype=float)
    out["weekday"] = np.broadcast_to(weekday[:, None], c.shape)
    return {k: np.where(np.isfinite(x), x, np.nan).astype(np.float32) for k, x in out.items()}


def sign_surrogate(panel, seed: int = 0):
    """A direction-free copy of the market: every bar's move from the previous close (the
    overnight gap included) is mirrored with probability 1/2 and the path is re-chained from the
    first open. Keeps each bar's size, range, volume and time of day; removes all direction —
    anything a detector still "finds" on it is volatility, not foresight.

    Args:
        panel: A price panel.
        seed: Random seed of the signs.

    Returns:
        The surrogate panel (same index, columns and volume).
    """
    o, h, lo, c = (x.to_numpy(float) for x in (panel.open_, panel.high, panel.low, panel.close))
    prev = np.vstack([o[:1], c[:-1]])
    a, hh, ll, cc = (np.log(x / prev) for x in (o, h, lo, c))
    sign = np.where(np.random.default_rng(seed).random(c.shape) < 0.5, -1.0, 1.0)
    level = np.log(o[0]) + np.vstack([np.zeros((1, c.shape[1])),
                                      np.nancumsum(np.nan_to_num(sign * cc), axis=0)[:-1]])
    moves = {"open_": sign * a, "high": np.where(sign > 0, hh, -ll),
             "low": np.where(sign > 0, ll, -hh), "close": sign * cc}
    return replace(panel, **{k: pd.DataFrame(np.exp(level + x), index=panel.close.index,
                                             columns=panel.close.columns)
                             for k, x in moves.items()})


def detector_rows(panel, *, every: int = 5, first_min: int = 15, last_min: int = 30,
                  warm: int = 21) -> np.ndarray:
    """Decision bars: every ``every`` minutes from ``first_min`` after the open to ``last_min``
    before the close, after ``warm`` warm-up sessions (all daily features defined)."""
    codes, bar_no = S["_sessions"](panel.close.index, panel.tz)
    n_left = pd.Series(codes).groupby(codes).transform("size").to_numpy() - 1 - bar_no
    ok = (bar_no >= first_min) & (n_left >= last_min) & (codes >= warm) & (bar_no % every == 0)
    return np.flatnonzero(ok)


def atr_move(panel, k: float) -> np.ndarray:
    """A per-name, per-day move: ``k`` x the daily ATR% known at the open — the same event is
    then equally rare in a calm name and a wild one. Shape (bars, names)."""
    codes, _ = S["_sessions"](panel.close.index, panel.tz)
    atr = S["_session_atr"](panel, codes, 14).to_numpy()[codes]
    return k * atr / panel.close.groupby(codes).last().shift(1).to_numpy()[codes]


def move_events(panel, rows, *, move, window: int) -> dict:
    """LONG / SHORT events at decision ``rows`` (every name); entry = the next bar's open.

    LONG: the high reaches ``entry x (1 + move)`` within ``window`` minutes, never past the
    session's close; SHORT: the low reaches ``entry x (1 - move)``. ``move`` is a fraction or a
    (bars, names) array (:func:`atr_move`).

    Returns:
        (rows, names) arrays: ``long``, ``short``; ``t_long`` / ``t_short`` (bars to the first
        touch, inf if none); ``first`` (+1 up first, -1 down first, 0 neither, NaN both in one
        bar); ``ret_long`` / ``ret_short`` — the bracket trade: target at the move, stop at the
        opposite move (a bar touching both is a stop), else the window's last close; ``ok``.
    """
    o, h, lo, c = (x.to_numpy(float) for x in (panel.open_, panel.high, panel.low, panel.close))
    n, m = c.shape
    codes, _ = S["_sessions"](panel.close.index, panel.tz)
    last = pd.Series(np.arange(n)).groupby(codes).transform("max").to_numpy()
    rows = np.asarray(rows)
    entry = o[np.minimum(rows + 1, n - 1)]
    mv = (np.asarray(move, float)[rows] if np.ndim(move) == 2
          else np.full((len(rows), m), float(move)))
    up, dn = entry * (1 + mv), entry * (1 - mv)
    end = np.minimum(rows + window, last[rows])
    t_up, t_dn = np.full((len(rows), m), np.inf), np.full((len(rows), m), np.inf)
    for k in range(1, window + 1):
        live = (rows + k <= end)[:, None]
        if not live.any():
            break
        j = np.minimum(rows + k, n - 1)
        t_up[live & (h[j] >= up) & np.isinf(t_up)] = k
        t_dn[live & (lo[j] <= dn) & np.isinf(t_dn)] = k
    first = np.where(t_up < t_dn, 1.0, np.where(t_dn < t_up, -1.0,
                                                  np.where(np.isinf(t_up), 0.0, np.nan)))
    drift = c[end] / entry - 1
    stopped = np.isnan(first)
    ret_long = np.where(first == 1, mv, np.where((first == -1) | stopped, -mv, drift))
    ret_short = np.where(first == -1, mv, np.where((first == 1) | stopped, -mv, -drift))
    ok = np.isfinite(entry) & np.isfinite(mv) & (rows + 1 <= last[rows])[:, None]
    return {"long": np.isfinite(t_up) & ok, "short": np.isfinite(t_dn) & ok, "t_long": t_up,
            "t_short": t_dn, "first": np.where(ok, first, np.nan),
            "ret_long": np.where(ok, ret_long, np.nan),
            "ret_short": np.where(ok, ret_short, np.nan), "ok": ok}


def detector_table(panel, rows, feats: dict, ev: dict) -> pd.DataFrame:
    """One row per (decision bar, name): calendar, the features and the events."""
    m = panel.close.shape[1]
    rr, cc = np.repeat(rows, m), np.tile(np.arange(m), len(rows))
    loc = panel.close.index.tz_convert(panel.tz)[rr]
    d = {"row": rr, "col": cc, "symbol": np.asarray(panel.symbols, dtype=object)[cc],
         "day": np.asarray(loc.date), "month": np.asarray(loc.strftime("%Y-%m")),
         "minute": np.asarray(loc.hour * 60 + loc.minute)}
    d.update({k: x[rr, cc] for k, x in feats.items()})
    d.update({k: np.asarray(x).reshape(-1) for k, x in ev.items()})
    table = pd.DataFrame(d)
    return table[table["ok"]].reset_index(drop=True)


def walk_detector(table: pd.DataFrame, cols, y: str, *, kind: str = "gbm",
                  warm_months: int = 3) -> pd.Series:
    """Out-of-sample P(y) for every row: month m is scored by a model fitted on months < m only.

    The events end at the session's close, so no label reaches into the scored month.

    Args:
        table: :func:`detector_table` rows.
        cols: Feature columns.
        y: The 0/1 target column.
        kind: ``gbm`` or ``lr`` (the strategy's :func:`make_model`).
        warm_months: Months used only for training before the first scored month.

    Returns:
        P(y = 1), NaN in the warm-up months.
    """
    score = pd.Series(np.nan, index=table.index)
    month = table["month"].to_numpy()
    x, target = table[list(cols)].to_numpy(np.float32), table[y].to_numpy(float)
    for mth in sorted(np.unique(month))[warm_months:]:
        fit = (month < mth) & np.isfinite(target)
        model = S["make_model"](kind).fit(x[fit], target[fit].astype(int))
        score[month == mth] = model.predict_proba(x[month == mth])[:, 1]
    return score


def causal_alerts(table: pd.DataFrame, score, q: float) -> np.ndarray:
    """Alert when the score is in the top ``q`` of the PREVIOUS month's scores — a threshold set
    from the scored month itself would peek at its future (it inflated every trade lead by ~40%
    in the 2026-09-30 onset study). The first scored month raises no alerts."""
    s = np.asarray(score, float)
    month = table["month"].to_numpy()
    ok = np.isfinite(s)
    out = np.zeros(len(s), bool)
    months = sorted(np.unique(month[ok]))
    for prev, mth in zip(months, months[1:], strict=False):
        cur = ok & (month == mth)
        out[cur] = s[cur] >= np.quantile(s[ok & (month == prev)], 1 - q)
    return out


# ---------------------------------------------------------------------------
# Audits.
# ---------------------------------------------------------------------------
#: Statuses :func:`reconcile` explains by an engine rule; anything else is a defect.
#: ``cover limited by free cash`` is NOT one of them since the engine locks short proceeds
#: (``lock_cash=True``, 2026-09-27): a split cover is named, but counted as a defect.
ENGINE_RULES = ("opposite close", "closed an opposite position", "wallet had no free cash",
                "open at data end")


def reconcile(panel, c: dict) -> pd.DataFrame:
    """Every traded event against the engine's booked trade (no costs, exact fill prices).

    Returns:
        One row per event or trade with a ``status``: ``ok`` (same entry bar, exit bar and exit
        price), or one of the engine's wallet rules the labels do not model (:data:`ENGINE_RULES`):
        an opposite-side entry on a held name CLOSES that position instead of opening its own
        (``upon_opposite_entry="close"``) — the held trade reads ``opposite close``, the new event
        ``closed an opposite position``; an entry on a bar where the shared wallet's FREE cash
        was gone (≤ $1: every slot is a share of free cash, and open shorts tie cash up) never
        fills — ``wallet had no free cash``; a label that ran past the last bar is ``open at
        data end``. Anything else is a defect: ``exit differs``, ``label only``, ``engine
        only``, and ``cover limited by free cash`` — a short covered in pieces because the
        cash to buy it back was gone, which the engine's ``lock_cash=True`` (2026-09-27)
        rules out; before it, a long could spend a short's proceeds.
    """
    pf = bt.run(panel, c["signals"], fees=0.0, slippage=0.0)
    trades = bt.trade_records(pf)
    free = pf.cash(free=True)
    free = (free if isinstance(free, pd.Series) else free.sum(axis=1)).to_numpy()
    ix = panel.close.index
    out = []
    opposite = {s: set(zip(t.loc[t["traded"], "symbol"], t.loc[t["traded"], "row"] + 1,
                           strict=True)) for s, t in c["events"].items()}
    for side, t in c["events"].items():
        mine = t[t["traded"]].assign(entry_ts=lambda x: ix[x["row"].to_numpy() + 1])
        theirs = trades[trades["direction"] == side.capitalize()]
        j = theirs.merge(mine, on=["symbol", "entry_ts"], how="outer", indicator=True)
        j["side"] = side
        bar_ok = ix.get_indexer(j["exit_ts"]) == j["exit_row"]
        px_ok = np.isclose(j["exit_price"], j["exit_px"], rtol=1e-9)
        other_side = "Short" if side == "long" else "Long"
        other = opposite.get(other_side.lower(), set())
        opp = [(s, e) in other for s, e in zip(j["symbol"], ix.get_indexer(j["exit_ts"]),
                                               strict=True)]
        held = trades[trades["direction"] == other_side]          # the other side's positions
        blocked = [bool(((held["symbol"] == s) & (held["entry_ts"] < e)
                         & (held["exit_ts"] >= e)).any()) if m == "right_only" else False
                   for s, e, m in zip(j["symbol"], j["entry_ts"], j["_merge"], strict=True)]
        fill = ix.get_indexer(j["entry_ts"])
        broke = (fill > 0) & (np.fmin(free[np.maximum(fill - 1, 0)], free[np.maximum(fill, 0)])
                              <= 1.0)
        at_end = (j["status"] == "Open").to_numpy() & ~j["complete"].fillna(True).astype(bool)
        pieces = j.duplicated(["symbol", "entry_ts"], keep=False).to_numpy()   # one position, split
        label_exit = np.nan_to_num(j["exit_row"].to_numpy(dtype=float), nan=0).astype(int)
        short_of_cash = pieces & (free[np.clip(label_exit, 0, len(free) - 1)] <= 1.0)
        j["status"] = np.select(
            [j["_merge"] == "left_only", np.asarray(blocked),
             (j["_merge"] == "right_only").to_numpy() & broke, j["_merge"] == "right_only",
             bar_ok & px_ok, np.asarray(opp), at_end, short_of_cash],
            ["engine only", "closed an opposite position", "wallet had no free cash",
             "label only", "ok", "opposite close", "open at data end",
             "cover limited by free cash"], "exit differs")
        out.append(j)
    return pd.concat(out, ignore_index=True)


def purge_audit(table: pd.DataFrame, ctx: dict, p: dict, *, folds: int = 3) -> pd.DataFrame:
    """Re-fit chosen folds with every label still running at the fold's start FLIPPED.

    A purged walk-forward cannot notice: the fold's predictions must come out identical.

    Returns:
        One row per audited fold: its month, events, labels flipped, and ``unchanged``.
    """
    X = table[list(S["FEATURES"])].to_numpy(float)
    y = table["win"].to_numpy(float)
    rows, ex = table["row"].to_numpy(), table["exit_row"].to_numpy()
    done = table["complete"].to_numpy()
    kw = {"model": p["model"], "trade_q": 0.33, "warmup_sessions": p["warmup_sessions"],
          "min_train_sessions": p["min_train_sessions"]}
    _, first = S["month_starts"](ctx["index"], ctx["tz"])
    fold_ids = [f for f in np.unique(table["fold"]) if f >= 0]
    picks = sorted({fold_ids[i] for i in np.linspace(0, len(fold_ids) - 1, folds).astype(int)}
                   ) if fold_ids else []
    out = []
    for fm in picks:
        keep = table["fold"].le(fm).to_numpy() | (ex < first[fm])      # nothing later matters
        running = ex >= first[fm]
        a = S["walk_forward"](X[keep], y[keep], rows[keep], ex[keep], done[keep], ctx, **kw)
        b = S["walk_forward"](X[keep], np.where(running, 1 - y, y)[keep], rows[keep], ex[keep],
                              done[keep], ctx, **kw)
        this = (a["fold"] == fm).to_numpy()
        month = pd.Period(ctx["index"][first[fm]].tz_convert(ctx["tz"]).tz_localize(None), "M")
        out.append({"fold": str(month), "events": int(this.sum()),
                    "labels flipped": int((running & keep).sum()),
                    "unchanged": bool(np.array_equal(a["p"][this], b["p"][this]))})
    return pd.DataFrame(out)


def cost_anatomy(trades: pd.DataFrame, costs, *, notional: float, fees: float,
                 slippage: float) -> pd.DataFrame:
    """Per-trade round-trip cost at the real order size, against the engine's one fee fraction.

    The engine charges one fee fraction per book (priced at the panel's median close); a real
    order of ``notional`` dollars pays ``max($1, shares × $0.005)``, half the spread and the
    slippage on each leg. The gap is the cost model's own error on this book.

    Returns:
        One row per trade: ``commission_bp`` (per leg), ``min_binds``, ``real_rt_bp`` and
        ``engine_rt_bp`` (round trip, both legs).
    """
    px = trades["entry_price"].to_numpy(float)
    shares = notional / px
    commission = np.maximum(costs.min_commission, shares * costs.commission_per_share)
    per_leg = commission / notional + costs.spread_bps / 2e4 + costs.slippage_bps / 1e4
    return pd.DataFrame({"symbol": trades["symbol"], "price": px,
                         "commission_bp": 1e4 * commission / notional,
                         "min_binds": commission <= costs.min_commission,
                         "real_rt_bp": 2e4 * per_leg,
                         "engine_rt_bp": 2e4 * (fees + slippage)})


def checks_table(checks: list[tuple]) -> pd.DataFrame:
    """The table every notebook ends on: ``(check, value, expected, ok)`` rows, printed."""
    frame = pd.DataFrame(checks, columns=["check", "value", "expected", "ok"])
    frame["ok"] = np.where(frame["ok"].astype(bool), "OK", "FAIL")
    n_fail = int((frame["ok"] == "FAIL").sum())
    print(f"CHECKS: {len(frame) - n_fail} OK, {n_fail} FAIL")
    return frame


def _rounded(x):
    if isinstance(x, dict):
        return {str(k): _rounded(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_rounded(v) for v in x]
    if isinstance(x, np.bool_ | bool):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        return None if not np.isfinite(x) else round(float(x), 4)
    return x


def scorecard(payload: dict) -> None:
    """Print the one-line ``SCORECARD {json}`` a headless run is read by."""
    print("SCORECARD " + json.dumps(_rounded(payload)))
