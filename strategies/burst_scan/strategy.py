"""Burst scan: activity anomalies on minute bars, a long and a short sub-strategy, each learned.

A **burst** is a sudden change in how a name trades: over the last ``burst_minutes`` its volume
and/or its high-low range far exceed what that name normally does at that time of day (the
same-clock baseline of ``bt.volume_profile``, so the routinely busy open never fires by itself).
A burst says *that* something is happening, not *which way* it will go. So the strategy runs two
independent **sub-strategies**:

* **long** — its own detector (``long_trigger``, ``long_impulse``, ``long_vol_mult``,
  ``long_range_mult``) flags candidate events; its own classifier, trained walk-forward on the
  outcomes of past events, predicts which of them will reach a target in the LONG direction
  before a stop of the same size; only its most confident picks are bought.
* **short** — the mirror, with its own ``short_*`` knobs, its own labels and its own model. The
  two never see each other's labels, so one can work while the other does not.

**The label is the trade.** An event at bar ``t`` fills at bar ``t+1``'s open. Its target and
stop sit ``move_atr`` session-ATRs × √(horizon / 390 minutes) away — the random-walk scaling, so
one knob means the same thing for a 30-minute and a 5-day hold — measured from the FILL bar's
close, and are checked from the bar after the fill: exactly how vectorbt books ``sl_stop`` /
``tp_stop`` (``stop_entry_price="close"``; a bar touching both is a stop; a gap through a
barrier fills at the open). An event is a **win** when the target is reached first. Positions
may be held across sessions (``{side}_eod_exit: false``) for up to ``{side}_horizon`` minutes of
trading time.

**The classifier is honest by construction.** Calendar-month folds, expanding window: month
``m``'s events are scored by a model fit ONLY on events whose outcome was complete before
``m``'s first bar (a multi-day label that ends inside the fold never trains it). The trade rule
is the fold's own training-probability quantile (``{side}_trade_q``), so no threshold is tuned on
the months it trades. Logistic regression is the default (``model: lr``); gradient boosting
(``model: gbm``) is the challenger it must beat out of sample.

Everything that decides a trade is known at the event bar's close: the profiles exclude the
current session, the session ATR and the dollar-volume floor are shifted one session, and every
feature reads bars up to ``t``. Research-only computations (placebos, the event study, the
label-vs-engine reconciliation) live in ``research.py``, which this module never imports.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel

_SIDES = ("both", "long", "short")
_TRIGGERS = ("volume", "range", "both", "either")
_IMPULSES = ("any", "up", "down")
_MODELS = ("lr", "gbm")
#: Reg SHO Rule 201 trigger: this far below the prior close, a name is uptick-only for the rest
#: of the session and the whole next one. The strategy simply does not short it.
_SSR_DROP = 0.10
#: Relative-volume level that counts as "a burst" for the side-agnostic context features
#: (earlier bursts today, market breadth) — fixed, so a feature never moves with a side's knobs.
_CONTEXT_MULT = 2.0
#: Minutes in a regular session: the unit of the barrier's √(horizon) scaling.
_SESSION_MINUTES = 390
#: Fewest training events a fold needs before its model is fit.
_MIN_FIT_EVENTS = 50
#: Feature columns, in the order the classifier sees them (see :func:`context`).
FEATURES = ("log_rvol", "log_rrange", "burst_atr", "burst_sign", "clv", "vwap_atr", "sess_atr",
            "day_pos", "gap_atr", "prev_hi_atr", "prev_lo_atr", "trend_atr", "ret_5d", "mkt_k",
            "breadth", "tod", "to_close", "n_prior", "atr_pct", "dv_rank")


# ---------------------------------------------------------------------------
# Bars, sessions and the session-scale ATR (same helpers as level_flow).
# ---------------------------------------------------------------------------
def _minutes_per_bar(bar: str) -> int:
    """Minutes in one execution bar (``"1d"`` counts as one RTH day).

    Args:
        bar: Panel bar size (``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``).

    Returns:
        Minutes per bar.

    Raises:
        KeyError: On a bar size that is neither ``1d``/``1h`` nor ``<n>m``.
    """
    if bar == "1d":
        return _SESSION_MINUTES
    if bar == "1h":
        return 60
    if bar.endswith("m") and bar[:-1].isdigit():
        return int(bar[:-1])
    raise KeyError(f"unknown bar size {bar!r}")


def _bars(minutes: float, bar: str, *, exact: bool = False) -> int:
    """Convert a window in minutes to a count of execution bars.

    Args:
        minutes: Window length in minutes.
        bar: Panel bar size.
        exact: Require a whole multiple of the bar (a candle knob), else round up.

    Returns:
        The window in bars, at least 1.

    Raises:
        ValueError: If ``exact`` and ``minutes`` is not a multiple of the bar.
    """
    per = _minutes_per_bar(bar)
    if exact and (minutes < per or minutes % per):
        raise ValueError(f"{minutes}-minute windows cannot be built from {bar} bars")
    return max(1, math.ceil(minutes / per))


def _sessions(index: pd.DatetimeIndex, tz: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar session id (0-based, chronological) and bar number within the session."""
    day = pd.Series(index.tz_convert(tz).date, index=index)
    codes = pd.factorize(day, sort=True)[0]
    return codes, day.groupby(codes).cumcount().to_numpy()


def _session_atr(panel: Panel, codes: np.ndarray, atr_days: int) -> pd.DataFrame:
    """Session-scale Wilder ATR on the session grid, shifted one session (known at the open).

    Args:
        panel: Price panel.
        codes: Session id per bar (:func:`_sessions`).
        atr_days: Smoothing window in sessions.

    Returns:
        ``(session x symbol)`` frame in price units, NaN through the warm-up.
    """
    hi = panel.high.groupby(codes).max()
    lo = panel.low.groupby(codes).min()
    prev = panel.close.groupby(codes).last().shift(1)
    tr = np.fmax(np.fmax((hi - lo).to_numpy(), (hi - prev).abs().to_numpy()),
                 (lo - prev).abs().to_numpy())
    tr = pd.DataFrame(tr, index=hi.index, columns=hi.columns)
    return tr.ewm(alpha=1.0 / atr_days, min_periods=atr_days).mean().shift(1)


def _f32(frame) -> np.ndarray:
    """A frame as a float32 array (half the memory of the float64 panel frames)."""
    return np.asarray(frame, dtype=np.float32)


# ---------------------------------------------------------------------------
# The shared context: detector inputs and every feature, computed once per panel.
# ---------------------------------------------------------------------------
def context(panel: Panel, *, burst_minutes: int, profile_days: int, atr_days: int,
            dv_days: int, trend_minutes: int, skip_open: int, skip_close: int,
            warmup_sessions: int, min_price: float, min_dollar_volume: float, **_) -> dict:
    """Everything both sub-strategies read, as arrays on the ``(bar x symbol)`` grid.

    Detector inputs: ``vol_ratio`` (volume over the last ``burst_minutes`` / the same clock's
    median over the prior ``profile_days`` sessions), ``rng_ratio`` (the window's high-low range
    over close, against its own same-clock median) and ``move`` (the window's open-to-close
    return, its sign is the burst's direction). ``window`` is where an event may fire: past
    ``skip_open``, before ``skip_close``, after ``warmup_sessions``, above both floors. ``inside``
    marks bars whose burst window lies inside one session. ``feat`` holds every column of
    :data:`FEATURES` (bar-grid float32), all known at the bar's close.

    Args:
        panel: Price panel (1m, or a coarser bar derived from it).
        burst_minutes: Burst window in minutes (a whole multiple of the bar).
        profile_days: Sessions in the same-clock baselines.
        atr_days: Session-ATR smoothing window in sessions.
        dv_days: Sessions of median dollar volume behind the floor and ``dv_rank``.
        trend_minutes: Length of the pre-burst trend feature, in minutes.
        skip_open: No events in the first N minutes of a session.
        skip_close: No events in the last N minutes of a session.
        warmup_sessions: No events in the first N sessions.
        min_price: Price floor at the event bar.
        min_dollar_volume: Floor on the median session dollar volume of the prior sessions.
        **_: Other strategy params, ignored.

    Returns:
        Dict of arrays and scalars (see above), plus ``codes``, ``bar_no``, ``n_left`` (bars
        to the session's last bar), ``mkt`` (equal-weight index level at each close), ``ssr``,
        ``k`` (burst bars), ``index``, ``symbols``, ``bar``, ``tz``.
    """
    bar, tz = panel.bar, panel.tz
    index, cols = panel.close.index, panel.close.columns
    c, o, h, lo, v = panel.close, panel.open_, panel.high, panel.low, panel.volume
    codes, bar_no = _sessions(index, tz)
    n_sess = int(codes.max()) + 1
    k = _bars(burst_minutes, bar, exact=True)
    mpb = _minutes_per_bar(bar)
    inside = bar_no >= k - 1

    def sess(frame: pd.DataFrame) -> np.ndarray:
        """Session-grid frame spread onto the bar grid."""
        return frame.to_numpy()[codes]

    # --- the burst window, against the same clock's history ----------------------------------
    vol_k = v.rolling(k, min_periods=k).sum()
    vol_ratio = vol_k / bt.volume_profile(v, tz, days=profile_days).rolling(k, min_periods=k).sum()
    hi_k = h.rolling(k, min_periods=k).max()
    lo_k = lo.rolling(k, min_periods=k).min()
    rng_k = (hi_k - lo_k) / c
    rng_ratio = rng_k / bt.volume_profile(rng_k, tz, days=profile_days)
    burst_open = o.shift(k - 1)
    move = c / burst_open - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        clv = _f32(2.0 * (c - lo_k) / (hi_k - lo_k) - 1.0)
    del vol_k, hi_k, lo_k

    # --- session context --------------------------------------------------------------------
    atr_s = _session_atr(panel, codes, atr_days)
    atr = sess(atr_s)
    sess_open = sess(o.groupby(codes).first())
    prev_close_s = c.groupby(codes).last().shift(1)
    prev_close = sess(prev_close_s)
    prev_hi = sess(h.groupby(codes).max().shift(1))
    prev_lo = sess(lo.groupby(codes).min().shift(1))
    ret_5d = sess(prev_close_s / prev_close_s.shift(5) - 1.0)
    dv_s = (c * v).groupby(codes).sum().rolling(dv_days, min_periods=dv_days).median().shift(1)
    dv_med = sess(dv_s)
    dv_rank = sess(dv_s.rank(axis=1, pct=True))
    sess_hi = h.groupby(codes).cummax().to_numpy()
    sess_lo = lo.groupby(codes).cummin().to_numpy()
    vwap = ((c * v).groupby(codes).cumsum() / v.groupby(codes).cumsum()).to_numpy()
    ssr_now = sess_lo <= (1.0 - _SSR_DROP) * prev_close
    ssr_prev = pd.DataFrame(ssr_now, index=index).groupby(codes).last().shift(1, fill_value=False)
    ssr = ssr_now | ssr_prev.to_numpy(dtype=bool)[codes]
    tb = _bars(trend_minutes, bar)
    trend = (c.shift(k) - c.shift(k + tb)).to_numpy()

    # --- the market and the side-agnostic burst context ------------------------------------
    mret = c.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    mkt = (1.0 + mret).cumprod().to_numpy()
    mkt_k = np.r_[np.full(k, np.nan), mkt[k:] / mkt[:-k] - 1.0]
    busy = (vol_ratio.to_numpy() >= _CONTEXT_MULT) & inside[:, None]
    mv = move.to_numpy()
    counted = np.isfinite(vol_ratio.to_numpy()).sum(axis=1)
    with np.errstate(invalid="ignore"):
        breadth = ((busy & (mv > 0)).sum(axis=1) - (busy & (mv < 0)).sum(axis=1)) / counted
    n_prior = pd.DataFrame(busy, index=index).groupby(codes).cumsum().to_numpy() - busy
    n_left = pd.Series(codes).groupby(codes).transform("size").to_numpy() - 1 - bar_no

    cn = c.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        feat = {
            "log_rvol": _f32(np.log(vol_ratio.to_numpy())),
            "log_rrange": _f32(np.log(rng_ratio.to_numpy())),
            "burst_atr": _f32((cn - burst_open.to_numpy()) / atr),
            "burst_sign": _f32(np.sign(mv)),
            "clv": clv,
            "vwap_atr": _f32((cn - vwap) / atr),
            "sess_atr": _f32((cn - sess_open) / atr),
            "day_pos": _f32((cn - sess_lo) / (sess_hi - sess_lo)),
            "gap_atr": _f32((sess_open - prev_close) / atr),
            "prev_hi_atr": _f32((cn - prev_hi) / atr),
            "prev_lo_atr": _f32((cn - prev_lo) / atr),
            "trend_atr": _f32(trend / atr),
            "ret_5d": _f32(ret_5d),
            "mkt_k": _f32(mkt_k),
            "breadth": _f32(breadth),
            "tod": _f32(bar_no * mpb),
            "to_close": _f32(n_left * mpb),
            "n_prior": _f32(n_prior),
            "atr_pct": _f32(atr / cn),
            "dv_rank": _f32(dv_rank),
        }
    window = (bt.session_mask(index, tz, skip_open=skip_open, skip_close=skip_close).to_numpy()
              & inside & (codes >= warmup_sessions))[:, None]
    window = window & (cn >= min_price) & (dv_med >= min_dollar_volume)
    return {"vol_ratio": _f32(vol_ratio), "rng_ratio": _f32(rng_ratio), "move": _f32(mv),
            "window": window, "inside": inside, "ssr": ssr, "feat": feat, "mkt": mkt,
            "codes": codes, "bar_no": bar_no, "n_left": n_left, "k": k,
            "index": index, "symbols": list(cols), "bar": bar, "tz": tz, "n_sessions": n_sess}


def _recent(raw: np.ndarray, n: int) -> np.ndarray:
    """True where ``raw`` fired on any of the previous ``n`` bars of the same column."""
    cs = np.cumsum(raw, axis=0, dtype=np.int32)
    prev = np.zeros_like(cs)
    prev[1:] = cs[:-1]                                    # count up to t-1
    before = np.zeros_like(cs)
    if n + 1 <= len(cs):
        before[n + 1:] = cs[:-n - 1]                       # count up to t-1-n
    return (prev - before) > 0


def detect(ctx: dict, *, trigger: str, impulse: str, vol_mult: float, range_mult: float,
           cooldown: int) -> np.ndarray:
    """One sub-strategy's candidate events: bursts after quiet, inside the tradable window.

    A raw burst is a bar whose burst window clears the side's thresholds (``trigger`` picks
    volume, range, both or either) and whose move has the required sign (``impulse``). A raw
    burst is an **event** only if no raw burst fired on the previous ``cooldown`` bars of that
    name (the first bar of a cluster, the look may reach into the previous session) and it lies
    in ``ctx["window"]``.

    Args:
        ctx: :func:`context` output.
        trigger: ``volume`` | ``range`` | ``both`` | ``either``.
        impulse: ``any`` | ``up`` | ``down`` — the burst move's required sign.
        vol_mult: Relative-volume threshold.
        range_mult: Relative-range threshold.
        cooldown: Quiet bars required before a burst counts.

    Returns:
        ``(bar x symbol)`` bool array.
    """
    with np.errstate(invalid="ignore"):
        vol_ok = ctx["vol_ratio"] >= vol_mult
        rng_ok = ctx["rng_ratio"] >= range_mult
        trig = {"volume": vol_ok, "range": rng_ok, "both": vol_ok & rng_ok,
                "either": vol_ok | rng_ok}[trigger]
        mv = ctx["move"]
        imp = {"any": mv != 0, "up": mv > 0, "down": mv < 0}[impulse] & np.isfinite(mv)
    raw = trig & imp & ctx["inside"][:, None]
    return raw & ~_recent(raw, cooldown) & ctx["window"]


def features(ctx: dict, rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    """The :data:`FEATURES` of the events at ``(rows, cols)``, one float32 row each.

    An infinite value (a zero baseline) becomes NaN: the models read NaN as missing, never inf.
    """
    out = {}
    for name in FEATURES:
        a = ctx["feat"][name]
        x = a[rows] if a.ndim == 1 else a[rows, cols]            # market / clock features: 1-D
        out[name] = np.where(np.isfinite(x), x, np.nan).astype(np.float32)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Outcomes: what the engine would book for each event.
# ---------------------------------------------------------------------------
def outcomes(panel: Panel, ctx: dict, rows: np.ndarray, cols: np.ndarray, *, side: int,
             horizon: int, frac: np.ndarray, eod: bool) -> pd.DataFrame:
    """Barrier outcome of each event, booked the way vectorbt books ``sl_stop``/``tp_stop``.

    The event at bar ``t`` fills at bar ``f = t+1``'s open. Target and stop sit ``frac`` of
    bar ``f``'s CLOSE away (vectorbt's ``stop_entry_price="close"``), on the side's direction
    and against it. From bar ``f+1`` on, each bar is checked stop first: a bar whose low (long)
    reaches the stop exits at the stop, or at the open if it gapped through; otherwise a bar
    whose high reaches the target exits at the target, or at the open if it gapped through.
    With no barrier by bar ``L = f + horizon`` (or the session's last bar under ``eod``), the
    trade exits at ``L``'s open — the time or end-of-day exit. Verified against the engine by
    ``tests/test_burst_scan.py`` and the diagnostics notebook's reconciliation.

    Args:
        panel: The price panel the context was built on.
        ctx: :func:`context` output (sessions, the market index).
        rows: Event (signal) bar per event.
        cols: Symbol column per event.
        side: ``+1`` long, ``-1`` short.
        horizon: Time exit in bars after the fill.
        frac: Barrier distance per event, as a fraction of the fill bar's close.
        eod: Cap the trade at the fill session's last bar.

    Returns:
        One row per event: ``entry_px, exit_row, exit_px, reason`` (1 target, -1 stop, 0 time),
        ``win``, ``ret`` (signed), ``mkt_ret`` (the equal-weight market over the trade),
        ``dret`` (signed, market-demeaned), ``mfe``/``mae`` (signed best/worst excursion),
        ``bars``, ``nights`` (sessions crossed) and ``complete`` (False when the data ended
        before the trade could).
    """
    o, h, lo, c = (x.to_numpy(dtype=float) for x in (panel.open_, panel.high, panel.low,
                                                     panel.close))
    n_bars = len(o)
    n = len(rows)
    f = np.minimum(rows + 1, n_bars - 1)
    ok = rows + 1 < n_bars
    entry = np.where(ok, o[f, cols], np.nan)
    ref = c[f, cols]
    frac = np.asarray(frac, dtype=float)
    stop, target = ref * (1.0 - side * frac), ref * (1.0 + side * frac)
    last = f + horizon
    if eod:
        last = np.minimum(last, f + ctx["n_left"][f])
    complete = ok & (last <= n_bars - 1) & np.isfinite(stop) & np.isfinite(entry)
    last = np.minimum(last, n_bars - 1)
    exit_row = last.copy()
    exit_px = o[last, cols]
    reason = np.zeros(n, np.int8)
    best, worst = entry.copy(), entry.copy()
    alive = ok & np.isfinite(stop)
    for m in range(1, horizon + 1):
        act = np.flatnonzero(alive & (f + m <= last))
        if not len(act):
            break
        j, cc = f[act] + m, cols[act]
        op, hi, low = o[j, cc], h[j, cc], lo[j, cc]
        if side > 0:
            best[act], worst[act] = np.fmax(best[act], hi), np.fmin(worst[act], low)
            sl, tp = low <= stop[act], hi >= target[act]
            px = np.where(sl, np.fmin(op, stop[act]), np.fmax(op, target[act]))
        else:
            best[act], worst[act] = np.fmin(best[act], low), np.fmax(worst[act], hi)
            sl, tp = hi >= stop[act], low <= target[act]
            px = np.where(sl, np.fmax(op, stop[act]), np.fmin(op, target[act]))
        tp &= ~sl
        done = sl | tp
        hit = act[done]
        exit_row[hit], exit_px[hit] = j[done], px[done]
        reason[hit] = np.where(tp[done], 1, -1)
        alive[hit] = False
    ret = side * (exit_px / entry - 1.0)
    mkt = ctx["mkt"]
    x_ref = np.where(reason != 0, exit_row, np.maximum(exit_row - 1, 0))
    mkt_ret = mkt[x_ref] / mkt[np.maximum(f - 1, 0)] - 1.0
    codes = ctx["codes"]
    return pd.DataFrame({
        "entry_px": entry, "exit_row": exit_row, "exit_px": exit_px, "reason": reason,
        "win": reason == 1, "ret": ret, "mkt_ret": mkt_ret, "dret": ret - side * mkt_ret,
        "mfe": side * (best / entry - 1.0), "mae": side * (worst / entry - 1.0),
        "bars": exit_row - f, "nights": codes[exit_row] - codes[f], "complete": complete,
    })


# ---------------------------------------------------------------------------
# The classifier: calendar-month walk-forward, purged of labels still running.
# ---------------------------------------------------------------------------
def make_model(kind: str):
    """A fresh, deterministic classifier (scikit-learn imported here, not at module load).

    Args:
        kind: ``lr`` (median-impute, standardize, logistic regression) | ``gbm`` (histogram
            gradient boosting, which reads NaN natively).

    Returns:
        An unfitted estimator with ``fit`` / ``predict_proba``.
    """
    if kind == "lr":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), LogisticRegression(max_iter=1000))
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=0)


def month_starts(index: pd.DatetimeIndex, tz: str) -> tuple[np.ndarray, np.ndarray]:
    """Calendar month of every bar (exchange-local ``YYYY-MM`` codes) and each month's first bar.

    Returns:
        ``(month_of_bar, first_bar_of_month)`` — the second indexed by the first's codes.
    """
    months = index.tz_convert(tz).tz_localize(None).to_period("M")
    code, uniques = pd.factorize(months, sort=True)
    first = np.searchsorted(code, np.arange(len(uniques)))
    return code, first


def walk_forward(X: np.ndarray, y: np.ndarray, rows: np.ndarray, exit_rows: np.ndarray,
                 complete: np.ndarray, ctx: dict, *, model: str, trade_q: float,
                 warmup_sessions: int, min_train_sessions: int) -> pd.DataFrame:
    """Out-of-sample P(win) for every event, one calendar month at a time.

    Month ``m`` is predicted by a model fit on the events whose outcome was complete AND ended
    before ``m``'s first bar — the purge, so a multi-day label that runs into the month never
    trains it. Months starting before ``warmup_sessions + min_train_sessions`` sessions are not
    predicted, nor is a month whose training labels are all one class or fewer than
    :data:`_MIN_FIT_EVENTS`. An event is taken when its probability reaches the ``1 - trade_q``
    quantile of the fold's own training probabilities. Because every fold depends only on
    events before it, cutting the panel at a later session changes nothing earlier.

    Args:
        X: Feature matrix, one row per event (:func:`features`).
        y: Win labels.
        rows: Event bar per event.
        exit_rows: Exit bar per event (:func:`outcomes`).
        complete: Whether each outcome finished inside the data.
        ctx: :func:`context` output (index, tz, session codes).
        model: ``lr`` | ``gbm`` (:func:`make_model`).
        trade_q: Share of the most confident events to take.
        warmup_sessions: Sessions before the first event can exist.
        min_train_sessions: Sessions of events before the first predicted month.

    Returns:
        ``p`` (NaN = not predicted), ``thr`` (the fold's threshold), ``picked``, ``fold`` (the
        month's index into :func:`month_starts`) and ``n_train`` per event.
    """
    n = len(rows)
    out = pd.DataFrame({"p": np.full(n, np.nan), "thr": np.full(n, np.nan),
                        "picked": np.zeros(n, bool), "fold": np.full(n, -1),
                        "n_train": np.zeros(n, int)})
    if not n:
        return out
    month_of_bar, first = month_starts(ctx["index"], ctx["tz"])
    ev_month = month_of_bar[rows]
    y = np.asarray(y, dtype=float)
    usable = np.asarray(complete, bool) & np.isfinite(y)
    for m in np.unique(ev_month):
        start = first[m]
        if ctx["codes"][start] - warmup_sessions < min_train_sessions:
            continue
        train = usable & (exit_rows < start)
        test = ev_month == m
        yt = y[train]
        if train.sum() < _MIN_FIT_EVENTS or len(np.unique(yt)) < 2:
            continue
        est = make_model(model).fit(X[train], yt.astype(int))
        thr = float(np.quantile(est.predict_proba(X[train])[:, 1], 1.0 - trade_q))
        p = est.predict_proba(X[test])[:, 1]
        out.loc[test, "p"], out.loc[test, "thr"] = p, thr
        out.loc[test, "picked"], out.loc[test, "fold"] = p >= thr, m
        out.loc[test, "n_train"] = int(train.sum())
    return out


def spaced(rows: np.ndarray, cols: np.ndarray, gap: int) -> np.ndarray:
    """Keep events more than ``gap`` bars after the previous kept one of the same symbol.

    Applied to the TRADED events of one side with ``gap`` = its horizon, so a position is never
    re-entered while its time exit is pending (a second entry would be ignored by the engine,
    and its ``exit_after`` would later close the wrong position).

    Returns:
        Bool mask over the events.
    """
    keep = np.zeros(len(rows), bool)
    last_col, last_row = -1, 0
    for i in np.lexsort((rows, cols)):
        if cols[i] != last_col:
            last_col, last_row = cols[i], -(gap + 1)
        if rows[i] - last_row > gap:
            keep[i], last_row = True, rows[i]
    return keep


# ---------------------------------------------------------------------------
# The strategy.
# ---------------------------------------------------------------------------
def signals(
    panel: Panel,
    *,
    sides: str = "both",
    long_trigger: str = "both",
    long_impulse: str = "any",
    long_vol_mult: float = 3.0,
    long_range_mult: float = 2.0,
    long_horizon: int = 390,
    long_eod_exit: bool = False,
    long_move_atr: float = 1.0,
    long_trade_q: float = 0.33,
    short_trigger: str = "both",
    short_impulse: str = "any",
    short_vol_mult: float = 3.0,
    short_range_mult: float = 2.0,
    short_horizon: int = 390,
    short_eod_exit: bool = False,
    short_move_atr: float = 1.0,
    short_trade_q: float = 0.33,
    burst_minutes: int = 10,
    profile_days: int = 10,
    skip_open: int = 15,
    skip_close: int = 15,
    cooldown: int = 60,
    trend_minutes: int = 60,
    atr_days: int = 14,
    dv_days: int = 20,
    warmup_sessions: int = 20,
    min_price: float = 5.0,
    min_dollar_volume: float = 1_000_000.0,
    model: str = "lr",
    min_train_sessions: int = 50,
    max_positions: int = 10,
    ssr_guard: bool = True,
) -> dict:
    """Execution-timed entries, exits, barriers and sizing for both sub-strategies.

    Args:
        panel: Intraday price panel (1m stored, or a coarser bar derived from it).
        sides: ``both`` | ``long`` | ``short`` — which sub-strategies trade.
        long_trigger: ``volume`` | ``range`` | ``both`` | ``either`` — what makes a long burst.
        long_impulse: ``any`` | ``up`` | ``down`` — the burst move a long event needs
            (``up`` = continuation, ``down`` = fade).
        long_vol_mult: Relative volume over the burst window a long event needs.
        long_range_mult: Relative range over the burst window a long event needs.
        long_horizon: Long time exit in minutes of trading time (may cross sessions).
        long_eod_exit: Flat longs at the session's last bar.
        long_move_atr: Long target = stop distance, in session ATRs × √(horizon/390).
        long_trade_q: Share of the long classifier's most confident events it takes.
        short_trigger: The short sub-strategy's ``trigger``.
        short_impulse: The short sub-strategy's ``impulse`` (``down`` = continuation).
        short_vol_mult: The short sub-strategy's relative-volume threshold.
        short_range_mult: The short sub-strategy's relative-range threshold.
        short_horizon: Short time exit in minutes of trading time.
        short_eod_exit: Flat shorts at the session's last bar.
        short_move_atr: Short barrier distance, as ``long_move_atr``.
        short_trade_q: Share of the short classifier's most confident events it takes.
        burst_minutes: Burst window in minutes (a whole multiple of the bar).
        profile_days: Sessions in the same-clock volume and range baselines.
        skip_open: No events in the first N minutes (≥ ``burst_minutes``).
        skip_close: No events in the last N minutes.
        cooldown: Quiet minutes a burst needs before it counts.
        trend_minutes: Length of the pre-burst trend feature, in minutes.
        atr_days: Session-ATR smoothing window in sessions.
        dv_days: Sessions of median dollar volume behind the floor and ``dv_rank``.
        warmup_sessions: No events in the first N sessions.
        min_price: Price floor at the event bar.
        min_dollar_volume: Floor on the prior sessions' median session dollar volume.
        model: ``lr`` | ``gbm`` — the classifier both sub-strategies fit.
        min_train_sessions: Sessions of events before the first predicted month.
        max_positions: Wallet slots; each entry deploys ``1/max_positions`` of the free cash.
        ssr_guard: Never short a name down 10% from its prior close, that session and the next.

    Returns:
        ``{entries, exits, short_entries, short_exits, sl_stop, tp_stop, cash_frac}``.
    """
    return _compose(panel, **{k: v for k, v in locals().items() if k != "panel"})["signals"]


def _side_params(p: dict, side: str) -> dict:
    """One sub-strategy's knobs, prefix stripped (``long_vol_mult`` -> ``vol_mult``)."""
    return {k[len(side) + 1:]: v for k, v in p.items() if k.startswith(side + "_")}


def _run_side(panel: Panel, ctx: dict, p: dict, side: str) -> pd.DataFrame:
    """Detect, describe, label and score one sub-strategy's events (the events table)."""
    q = _side_params(p, side)
    sign = 1 if side == "long" else -1
    bar = ctx["bar"]
    ev = detect(ctx, trigger=q["trigger"], impulse=q["impulse"], vol_mult=q["vol_mult"],
                range_mult=q["range_mult"], cooldown=_bars(p["cooldown"], bar))
    if side == "short" and p["ssr_guard"]:
        ev &= ~ctx["ssr"]
    rows, cols = np.nonzero(ev)
    table = features(ctx, rows, cols)
    frac = (q["move_atr"] * table["atr_pct"].to_numpy(dtype=float)
            * math.sqrt(q["horizon"] / _SESSION_MINUTES))
    oc = outcomes(panel, ctx, rows, cols, side=sign, horizon=_bars(q["horizon"], bar),
                  frac=frac, eod=q["eod_exit"])
    wf = walk_forward(table.to_numpy(dtype=float), oc["win"].to_numpy(), rows,
                      oc["exit_row"].to_numpy(), oc["complete"].to_numpy(), ctx,
                      model=p["model"], trade_q=q["trade_q"],
                      warmup_sessions=p["warmup_sessions"],
                      min_train_sessions=p["min_train_sessions"])
    head = pd.DataFrame({"row": rows, "col": cols, "ts": ctx["index"][rows],
                         "symbol": np.asarray(ctx["symbols"], dtype=object)[cols],
                         "session": ctx["codes"][rows], "frac": frac})
    table = pd.concat([head, table, oc, wf], axis=1)
    picked = table["picked"].to_numpy() & np.isfinite(frac)
    keep = np.zeros(len(table), bool)
    keep[picked] = spaced(rows[picked], cols[picked], _bars(q["horizon"], bar))
    table["traded"] = keep
    return table


def _compose(panel: Panel, **p) -> dict:
    """Everything :func:`signals` computes, intermediates included (see :func:`compose`)."""
    ctx = context(panel, **p)
    shape = panel.close.shape
    index, columns = panel.close.index, panel.close.columns
    active = ("long", "short") if p["sides"] == "both" else (p["sides"],)
    events, raw = {}, {}
    for side in ("long", "short"):
        raw[side] = np.zeros(shape, bool)
        if side in active:
            events[side] = t = _run_side(panel, ctx, p, side)
            sel = t[t["traded"]]
            raw[side][sel["row"].to_numpy(), sel["col"].to_numpy()] = True
    clash = raw["long"] & raw["short"]                     # one bar, both sides: take neither
    frac = np.full(shape, np.nan)
    for side, t in events.items():
        raw[side] &= ~clash
        t["clash"] = clash[t["row"].to_numpy(), t["col"].to_numpy()] & t["traded"]
        if len(t):
            t["traded"] &= ~t["clash"]
            sel = t[t["traded"]]
            frac[sel["row"].to_numpy(), sel["col"].to_numpy()] = sel["frac"].to_numpy()

    def frame(a) -> pd.DataFrame:
        return pd.DataFrame(a, index=index, columns=columns)

    end = bt.session_end(index, panel.tz).to_numpy()[:, None]
    out = {}
    for side, (e_key, x_key) in (("long", ("entries", "exits")),
                                 ("short", ("short_entries", "short_exits"))):
        q = _side_params(p, side)
        entries = bt.next_bar(frame(raw[side]))
        exits = bt.exit_after(entries, _bars(q["horizon"], panel.bar))
        if q["eod_exit"]:
            exits |= end
        out[e_key], out[x_key] = entries, exits
    barrier = frame(frac).shift(1)                          # read at the FILL bar
    out |= {"sl_stop": barrier, "tp_stop": barrier, "cash_frac": 1.0 / p["max_positions"]}
    return {"signals": out, "events": events, "raw": {s: frame(r) for s, r in raw.items()},
            "clash": int(clash.sum()), "ctx": ctx, "params": p}


_DEFAULTS = {name: prm.default for name, prm in inspect.signature(signals).parameters.items()
             if prm.default is not inspect.Parameter.empty}


def compose(panel: Panel, **params) -> dict:
    """Expose every intermediate for the notebooks.

    Same computation as :func:`signals` (defaults under any overrides — notebooks pass
    ``{**strat.params, **overrides}``, never a bare call). Returns ``signals`` (the engine
    dict), ``events`` (per active side: one row per candidate event with its position, symbol,
    every feature, its outcome, its out-of-sample ``p``/``thr``/``picked``, ``clash`` (picked
    by both sides on one bar, so traded by neither) and ``traded``),
    ``raw`` (per side: the traded signal-bar frame, before the next-bar shift), ``clash``
    (same-bar long/short collisions dropped), ``ctx`` (:func:`context`) and ``params``.

    Args:
        panel: Price panel.
        **params: Overrides of :func:`signals`' keyword arguments.

    Returns:
        The intermediates dict described above.
    """
    return _compose(panel, **{**_DEFAULTS, **params})


def valid(params: dict) -> bool:
    """Reject incoherent combos (tolerates partial dicts from sweep grids).

    Args:
        params: Candidate parameter combo (may hold only the swept keys).

    Returns:
        True when the combo is coherent.
    """
    g = {**_DEFAULTS, **params}.get
    sides_ok = all(
        g(f"{s}_trigger") in _TRIGGERS and g(f"{s}_impulse") in _IMPULSES
        and g(f"{s}_vol_mult") >= 0 and g(f"{s}_range_mult") >= 0
        and 1 <= g(f"{s}_horizon") <= 5 * _SESSION_MINUTES
        and g(f"{s}_move_atr") > 0 and 0 < g(f"{s}_trade_q") < 1
        for s in ("long", "short"))
    return (sides_ok and g("sides") in _SIDES and g("model") in _MODELS
            and g("skip_open") >= g("burst_minutes") and g("cooldown") >= g("burst_minutes")
            and g("warmup_sessions") >= max(g("profile_days"), g("atr_days"), g("dv_days"), 6)
            and g("min_train_sessions") >= 1 and g("max_positions") >= 1)


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    open_: pd.Series | None = None,
    tz: str = "America/New_York",
    bar: str = "1m",
    **params,
) -> dict:
    """Per-symbol overlays: session VWAP, relative volume and both sides' detector events.

    The events drawn are the DETECTORS' (before any classifier — a model needs the whole
    universe's history, one symbol cannot fit it), so a chart shows what each side considers.

    Args:
        close: One symbol's adjusted close.
        volume: One symbol's volume.
        high: One symbol's adjusted high.
        low: One symbol's adjusted low.
        open_: One symbol's adjusted open.
        tz: The panel's exchange timezone.
        bar: The panel's bar size.
        **params: :func:`signals` keyword overrides.

    Returns:
        ``{"price", "volume", "volume_marks"}`` overlay series for ``plots.plot_symbol``.
    """
    def col(series: pd.Series | None) -> pd.DataFrame:
        return (close if series is None else series).to_frame("s")

    one = Panel(close=close.to_frame("s"), open_=col(open_), high=col(high), low=col(low),
                volume=volume.to_frame("s"), bar=bar, tz=tz)
    p = {**_DEFAULTS, **{k: v for k, v in params.items() if k in _DEFAULTS},
         "min_dollar_volume": 0.0, "dv_days": 1}
    ctx = context(one, **p)
    marks = np.zeros(len(close), bool)
    for side in ("long", "short"):
        q = _side_params(p, side)
        marks |= detect(ctx, trigger=q["trigger"], impulse=q["impulse"], vol_mult=q["vol_mult"],
                        range_mult=q["range_mult"], cooldown=_bars(p["cooldown"], bar))[:, 0]
    vw = (close * volume).groupby(ctx["codes"]).cumsum() / volume.groupby(ctx["codes"]).cumsum()
    return {"price": {"session VWAP": vw},
            "volume": {"relative volume (burst window)": pd.Series(ctx["vol_ratio"][:, 0],
                                                                   index=close.index)},
            "volume_marks": pd.Series(marks, index=close.index)}
