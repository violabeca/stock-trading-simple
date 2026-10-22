"""Darts forecast: a learned forecast of the next bars' return, walk-forward, retrained regularly.

The strategy is a FORECASTER plus a TRADING RULE.

**The forecaster.** Every ``bar_minutes`` (default 5; 60 = hourly) of the regular session it
predicts the return of each name over the next ``horizon`` bars. With ``overnight`` the horizon
may span sessions and the target is the close-to-close return, overnight gaps included:
60-minute bars and a 35-bar horizon predict the close 5 sessions later. It is a Darts model:

* ``gbm``: scikit-learn histogram gradient boosting, the LightGBM algorithm, used as a regressor;
* ``gbm_clf``: the same, as a classifier of up vs down;
* ``tide``: TiDE, an MLP encoder-decoder;
* ``tft``: the Temporal Fusion Transformer;
* ``linear``: a ridge regression on the same lagged inputs (the simplest learned baseline);
* ``tsmixer``: TSMixer, an all-MLP mixer of time steps and inputs.

All four see the same inputs, put on ONE time grid of ``bar_minutes`` bars. Darts needs every
input on the target's time steps, so each input is translated onto the grid:

* **Intraday** (from the 1-minute bars inside each grid bar, known at its close): return,
  realized volatility, range, close location, signed-volume flow, volume against its
  time-of-day norm, return since the open, distance to the session VWAP, and the other names'
  average return (a market proxy).
* **Daily** (known at the previous close, repeated on every bar of the session): previous-day,
  5-day and 20-day returns in volatility units, 20-day volatility, position in the 20-day range,
  the previous day's close location and relative volume, and today's overnight gap (known at
  the open).
* **Options** (end of day, known before the session; ``panel.features['opt_*']`` attached by
  ``research.attach_options`` from the free sources of ``burst_scan/options_free_data.ipynb``):
  ATM implied vol, skew, term structure, IV - HV, OCC customer put/call and volume surge,
  put/call open interest and volume, the open-interest change, dealer gamma imbalance. Each
  source gets a 0/1 "has data" input (Darts refuses NaN, so a missing value enters as 0).
* **Seasonality** (``seasonal``): the name's mean realized forward return from the same clock
  time on the previous sessions whose outcome is complete (a return over a half hour tends to
  repeat at the same half hour on later days: Heston, Korajczyk & Sadka 2010).
* **Future covariates** (known in advance): time of day, day of week, the monthly-expiry week.
* **Static**: which name (one-hot), so one global model learns all five.

**Walk-forward, retrained on a schedule.** At each retrain point (every ``retrain_minutes`` of
trading time; default 390 = before every session) the model is fit ONLY on the last
``train_days`` sessions before that point (with ``expanding``: on all history before it). It
then forecasts every decision bar until the next retrain point, each forecast from bars up to
that decision only. The deep models can reserve the
last ``val_days`` of each training window as a VALIDATION slice for early stopping, and later
retrains can fine-tune the previous weights instead of starting over. ``recency_half_life``
weights recent sessions more (the "the current regime lasts a few more days" idea).
``lag_history`` lets the first labels of each training window read their lagged target from before
the window (else the first ``lookback + horizon - 1`` labels of every window are dropped). Without
``overnight`` a training target counts only if its decision bar is in the same session (no
"previous close -> next open" pairs); with it the target is the close-to-close return over the
horizon, overnight gaps included. Every training label is realized before its retrain point.

**The trading rule** (cheap; the only searched knobs):

* Trade a name when the forecast return exceeds ``edge_bp`` (and, for models with a probability,
  P(direction) clears ``p_min``).
* Fill at the next 1-minute bar's open after the decision bar, and hold ``hold_bars`` grid bars.
  A new same-side forecast extends the hold; an opposite one closes it, and the new side opens
  one minute later. Everything is flat by the close, so an ``overnight`` forecast is not
  tradable yet (:func:`valid` rejects it).
* The stop is ``stop_vol`` × the name's typical move over the hold.
* Size risks ``risk_per_trade`` of the wallet at the stop, capped at one of ``max_positions``
  slots.
* A **model-health gate** (``health_days``) stands aside on days after the model's own recent
  forecasts stopped working: a regime-change brake, read causally from completed outcomes only.

The panel is the STORED 1-minute panel (``data.bar: 1m``); ``bar_minutes`` only sets the forecast
grid, so the 1-minute detail still reaches the model and fills stay at 1-minute resolution. On the
NYSE half-days (:data:`HALF_DAYS`) the stored "regular" session runs past the 13:00 close (engine
proposal #32); those minutes are ignored here. Darts is an optional dependency (``uv sync --extra
darts``) imported only inside the model functions; everything else is pure numpy / pandas.
``research.py`` (notebooks only) holds the I/O.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple.data import Panel

MODELS = ("gbm", "gbm_clf", "tide", "tft", "linear", "tsmixer")
#: The neural networks (quantile outputs, epochs, early stopping, fine-tuning).
DEEP = ("tide", "tft", "tsmixer")
OPTION_SETS = ("none", "iv", "all")
SIDES = ("both", "long", "short")
BAR_CHOICES = (1, 5, 10, 15, 30, 60)   # 60: a session's last bucket is 30 minutes
SESSION_MINUTES = 390
OPEN_MINUTE = 9 * 60 + 30
#: NYSE early closes (13:00 New York) inside the stored 1m history and its year ahead. The stored
#: "regular" panel still holds 13:00-15:59 on these days (post-close prints and volume-0
#: placeholders; engine proposal #32), so they are masked here.
HALF_DAYS = frozenset({"2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24"})
HALF_DAY_CLOSE = 13 * 60
#: NYSE full closures (weekday holidays) from the stored history's start through 2027. Only the
#: points PAST the panel end use them (``time_features`` continues the clock there, e.g. for the
#: latest forecast's target time); inside the panel the sessions come from the data. Extend before
#: 2028.
HOLIDAYS = frozenset({
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"})
#: Option inputs by SOURCE (keys in panel.features are "opt_" + name); each source gets a flag.
OPTION_SOURCES = {"dolt_iv": ("atm_iv", "skew", "term", "d_atm_iv"), "dolt_hv": ("iv_hv",),
                  "occ": ("cust_pc", "vol_surge"), "oi": ("oi_pc", "vol_pc", "d_oi", "gex")}
#: Which sources an ``options`` setting reads.
OPTION_SET_SOURCES = {"none": (), "iv": ("dolt_iv", "dolt_hv", "occ"),
                      "all": ("dolt_iv", "dolt_hv", "occ", "oi")}
INTRADAY = ("ret", "rv", "range", "clv", "flow", "rvol", "ret_open", "vwap_dist")
DAILY = ("prev_ret", "ret5", "ret20", "vol20", "pos20", "prev_clv", "prev_rvol", "gap")
TIME = ("tod_sin", "tod_cos", "tod_frac", "dow", "opex_week")
#: Sessions behind the ``seasonal`` input.
SEASON_DAYS = 20
#: Forecast-relevant parameters: a change re-runs the walk-forward; the rest is the trading rule.
MODEL_KEYS = ("model", "bar_minutes", "horizon", "lookback", "cov_lags", "train_days", "val_days",
              "retrain_minutes", "n_epochs", "finetune_epochs", "recency_half_life", "options",
              "use_daily", "use_market", "vol_days", "gbm_iter", "device", "seed", "overnight",
              "expanding", "seasonal", "lag_history")
#: Forecast knobs added later: off when a parameter dict predates them.
LATE_KEYS = ("overnight", "expanding", "seasonal", "lag_history")
_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 6


# ---------------------------------------------------------------------------------------------
# The grid.
# ---------------------------------------------------------------------------------------------
def sessions(index: pd.DatetimeIndex, tz: str) -> tuple[np.ndarray, np.ndarray]:
    """Per 1-minute bar: session number (0, 1, ... by exchange-local date) and minutes since the
    09:30 open by the CLOCK (a missing minute does not shift the later ones).

    Returns:
        ``(codes, minute)`` int arrays.
    """
    local = index.tz_convert(tz)
    codes = pd.factorize(np.asarray(local.date), sort=True)[0]
    minute = np.asarray(local.hour * 60 + local.minute) - OPEN_MINUTE
    return codes, minute


def regular_rows(panel: Panel) -> np.ndarray:
    """True for 1-minute rows inside the real regular session (False after the 13:00 close of a
    half-day in :data:`HALF_DAYS`)."""
    local = panel.close.index.tz_convert(panel.tz)
    day = np.asarray(local.strftime("%Y-%m-%d"))
    late = np.asarray(local.hour * 60 + local.minute) >= HALF_DAY_CLOSE
    return ~(np.isin(day, list(HALF_DAYS)) & late)


def grid(panel: Panel, bar_minutes: int) -> dict:
    """The forecast grid: ``bar_minutes`` buckets of each session, aligned to its 09:30 open.

    A bucket's DECISION ROW is its last 1-minute bar: everything about the bucket is known at that
    bar's close, and a trade decided there fills at the next 1-minute bar's open. Every session
    has the same ``n_pos`` positions by the clock (a 60-minute grid: 09:30, 10:30, ..., and a
    30-minute 15:30 bucket), so ``k * n_pos`` buckets later is the same clock time ``k``
    sessions later (a half-day's post-13:00 buckets stay in the index, empty).

    Returns:
        A dict of per-bucket arrays: ``start`` / ``end`` (first / last 1-minute row), ``code``
        (session), ``pos`` (bucket number in the session, by the clock), ``regular`` (inside the
        real session), ``left`` (regular buckets after it in the session; -1 after a half-day's
        close), ``first`` (first bucket of its session), ``date`` (session date), plus
        ``sess_first`` (first bucket of each session), ``n_pos`` (buckets per full session),
        ``codes_1m`` / ``regular_1m`` / ``last_1m`` (per 1-minute row: session, inside the real
        session, the session's last REAL bar).
    """
    codes, minute = sessions(panel.close.index, panel.tz)
    pos = minute // bar_minutes
    key = codes.astype(np.int64) * 10_000 + pos
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    end = np.r_[start[1:] - 1, len(key) - 1]
    reg_1m = regular_rows(panel)
    code, bpos, reg = codes[start], pos[start], reg_1m[start]
    n_reg = pd.Series(reg.astype(int)).groupby(code).transform("sum").to_numpy()
    rank = pd.Series(reg.astype(int)).groupby(code).cumsum().to_numpy() - 1
    left = np.where(reg, n_reg - 1 - rank, -1)
    first = np.r_[True, code[1:] != code[:-1]]
    sess_first = np.flatnonzero(first)
    dates = np.asarray(panel.close.index.tz_convert(panel.tz).date)
    # the last REAL bar of each session (the 12:59 bar on a half-day)
    rows = pd.Series(np.where(reg_1m, np.arange(len(codes)), -1))
    last_real = rows.groupby(codes).transform("max").to_numpy()
    last_1m = np.zeros(len(codes), bool)
    last_1m[np.unique(last_real[last_real >= 0])] = True
    return {"start": start, "end": end, "code": code, "pos": bpos, "regular": reg, "left": left,
            "first": first, "date": dates[start], "sess_first": sess_first,
            "n_pos": int(math.ceil(SESSION_MINUTES / bar_minutes)), "bar_minutes": bar_minutes,
            "codes_1m": codes, "regular_1m": reg_1m, "last_1m": last_1m}


def _reduce(values: np.ndarray, start: np.ndarray, how: str) -> np.ndarray:
    """Per-bucket reduction of a (minutes x names) array with NaN treated as missing."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if how == "sum":
        out = np.add.reduceat(np.where(ok, v, 0.0), start, axis=0)
    elif how == "max":
        out = np.maximum.reduceat(np.where(ok, v, -np.inf), start, axis=0)
    elif how == "min":
        out = np.minimum.reduceat(np.where(ok, v, np.inf), start, axis=0)
    else:
        raise ValueError(how)
    cnt = np.add.reduceat(ok.astype(float), start, axis=0)
    out[cnt == 0] = np.nan
    return out


def _arrays(panel: Panel, g: dict) -> tuple[np.ndarray, ...]:
    """O, H, L, C, V as float arrays with the half-days' post-close minutes set to NaN."""
    keep = g["regular_1m"][:, None]
    return tuple(np.where(keep, x.to_numpy(float), np.nan) for x in
                 (panel.open_, panel.high, panel.low, panel.close, panel.volume))


# ---------------------------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------------------------
def intraday_features(panel: Panel, g: dict, *, vol_days: int = 20) -> dict:
    """Per-bucket intraday features, each a (buckets x names) float array known at the bucket's
    close, plus the causal volatility ``scale`` (RMS of bucket returns over the previous
    ``vol_days`` sessions) used to put returns in comparable units. A session starts from its own
    open: the overnight gap is the daily ``gap`` input, never part of a bucket's return.

    Args:
        panel: The 1-minute panel.
        g: Its grid (:func:`grid`).
        vol_days: Sessions behind ``scale``.

    Returns:
        ``{"ret", "rv", "range", "clv", "flow", "rvol", "ret_open", "vwap_dist", "mkt",
        "scale", "close"}`` - ``ret`` / ``ret_open`` / ``vwap_dist`` / ``rv`` / ``range`` /
        ``mkt`` are RAW log units here; :func:`build_inputs` scales them. ``close`` is each
        bucket's last price (the last valid close at or before its decision row).
    """
    Op, H, L, C, V = _arrays(panel, g)
    codes_1m, start = g["codes_1m"], g["start"]
    # 1-minute returns against the last VALID close of the session (a missing minute's move is
    # kept); a session's first valid bar is its own open -> close (no overnight gap)
    cf = pd.DataFrame(C).groupby(codes_1m).ffill().to_numpy()
    prev = pd.DataFrame(cf).groupby(codes_1m).shift(1).to_numpy()
    carried = pd.DataFrame(C).ffill()
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = np.where(np.isfinite(prev), np.log(C / prev), np.log(C / Op))
    ret = _reduce(r1, start, "sum")
    rv = np.sqrt(_reduce(r1 ** 2, start, "sum"))
    hi, lo = _reduce(H, start, "max"), _reduce(L, start, "min")
    c_end = cf[g["end"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        rng = np.log(hi / lo)
        clv = np.where(hi > lo, ((c_end - lo) - (hi - c_end)) / (hi - lo), 0.0)
    vol = _reduce(V, start, "sum")
    flow = _reduce(np.sign(C - Op) * V, start, "sum") / np.where(vol > 0, vol, np.nan)
    # volume against the same bucket of the previous 10 sessions (causal)
    code, pos = g["code"], g["pos"]
    lv = pd.DataFrame(np.log1p(vol))
    norm = lv.groupby(pos).transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    rvol = (lv - norm).to_numpy()
    # since the open and against the session VWAP
    ret_open = pd.DataFrame(np.nan_to_num(ret)).groupby(code).cumsum().to_numpy()
    ret_open[~np.isfinite(ret)] = np.nan
    tp = (H + L + C) / 3.0
    pv = pd.DataFrame(np.nan_to_num(tp * V)).groupby(codes_1m).cumsum().to_numpy()
    vv = pd.DataFrame(np.nan_to_num(V)).groupby(codes_1m).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = pv[g["end"]] / np.where(vv[g["end"]] > 0, vv[g["end"]], np.nan)
        vwap_dist = np.log(c_end / vwap)
    empty = ~np.isfinite(ret)                        # no valid minute: a half-day's masked close
    clv[empty], vwap_dist[empty] = np.nan, np.nan
    # the other names' average return (leave-one-out market proxy; NaN if no other name has data)
    fin = np.isfinite(ret)
    others = fin.sum(axis=1, keepdims=True) - fin
    mkt = (np.nansum(ret, axis=1, keepdims=True) - np.nan_to_num(ret)) / np.where(
        others > 0, others, np.nan)
    # causal volatility scale: RMS bucket return over the previous vol_days sessions
    ms = pd.DataFrame(ret ** 2).groupby(code).mean()
    scale = np.sqrt(ms.rolling(vol_days, min_periods=max(3, vol_days // 2)).mean().shift(1))
    return {"ret": ret, "rv": rv, "range": rng, "clv": clv, "flow": flow, "rvol": rvol,
            "ret_open": ret_open, "vwap_dist": vwap_dist, "mkt": mkt,
            "scale": scale.to_numpy()[code], "close": carried.to_numpy()[g["end"]]}


def daily_features(panel: Panel, g: dict) -> dict:
    """Per-bucket daily features: values of the PREVIOUS sessions (known at the open), except
    ``gap`` (today's FIRST bar's open vs the previous close, known at the open; NaN when the
    session's first bar is missing). (buckets x names) arrays."""
    Op, H, L, C, V = _arrays(panel, g)
    codes_1m, code = g["codes_1m"], g["code"]
    grp = lambda x: pd.DataFrame(x, columns=panel.symbols).groupby(codes_1m)  # noqa: E731
    h, l_, c, v = grp(H).max(), grp(L).min(), grp(C).last(), grp(V).sum()
    first_row = np.r_[True, codes_1m[1:] != codes_1m[:-1]]
    o = pd.DataFrame(Op[first_row], index=c.index, columns=panel.symbols)
    r = np.log(c / c.shift(1))
    vol20 = r.rolling(20, min_periods=10).std()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "prev_ret": r / vol20,
            "ret5": np.log(c / c.shift(5)) / (vol20 * np.sqrt(5)),
            "ret20": np.log(c / c.shift(20)) / (vol20 * np.sqrt(20)),
            "vol20": np.log(vol20),
            "pos20": (c - l_.rolling(20, min_periods=10).min())
            / (h.rolling(20, min_periods=10).max() - l_.rolling(20, min_periods=10).min()),
            "prev_clv": ((c - l_) - (h - c)) / (h - l_).replace(0, np.nan),
            "prev_rvol": np.log(v / v.rolling(20, min_periods=10).mean().shift(1)),
        }
        res = {k: f.shift(1).to_numpy()[code] for k, f in out.items()}     # previous session
        gap = np.log(o / c.shift(1)) / vol20.shift(1)                      # known at the open
    res["gap"] = gap.to_numpy()[code]
    return res


def option_features(panel: Panel, g: dict, option_set: str) -> dict:
    """Per-bucket option features from ``panel.features['opt_<name>']`` (date x symbol frames,
    ALREADY lagged: the value on date d is known before d's open). Missing frames / dates -> NaN.
    """
    names = [n for src in OPTION_SET_SOURCES[option_set] for n in OPTION_SOURCES[src]]
    feats = panel.features or {}
    dates = pd.to_datetime(pd.Series(g["date"])).dt.normalize()
    out = {}
    for n in names:
        f = feats.get("opt_" + n)
        if f is None:
            out[n] = np.full((len(dates), len(panel.symbols)), np.nan)
            continue
        f = f.copy()
        idx = pd.DatetimeIndex(pd.to_datetime(f.index))
        f.index = (idx.tz_convert(None) if idx.tz is not None else idx).normalize()
        out[n] = f.reindex(columns=panel.symbols).reindex(dates.to_numpy()).to_numpy(float)
    return out


def time_features(g: dict, extra: int = 0) -> np.ndarray:
    """FUTURE covariates known in advance, (buckets + extra) x 5: sin / cos / fraction of the
    session clock, weekday / 4, and the monthly-expiry week (Monday to Friday of the month's
    third Friday, from the calendar, so a holiday third Friday still flags its week). The
    ``extra`` points past the panel end continue the clock onto the following sessions (weekdays
    that are not in :data:`HOLIDAYS`)."""
    step = g["pos"][-1] + 1 + np.arange(extra)
    pos = np.r_[g["pos"], step % g["n_pos"]]
    last = pd.Timestamp(g["date"][-1])
    session = pd.offsets.CustomBusinessDay(holidays=sorted(HOLIDAYS))
    d = pd.to_datetime(pd.Series([*g["date"], *(last + int(k) * session
                                                for k in step // g["n_pos"])]))
    ang = 2 * np.pi * pos / g["n_pos"]
    month1 = d.dt.to_period("M").dt.to_timestamp()
    third_fri = month1 + pd.to_timedelta((4 - month1.dt.weekday) % 7 + 14, unit="D")
    opex = ((d >= third_fri - pd.Timedelta(days=4)) & (d <= third_fri)).to_numpy()
    return np.stack([np.sin(ang), np.cos(ang), pos / g["n_pos"], d.dt.weekday.to_numpy() / 4.0,
                     opex.astype(float)], axis=1).astype(np.float32)


def seasonal_input(g: dict, target: np.ndarray, valid: np.ndarray, horizon: int, *,
                   days: int = SEASON_DAYS) -> np.ndarray:
    """The intraday-seasonality input: at each decision bar, the name's mean realized forward
    target (the label of a decision ``horizon`` bars earlier, in the target's units) from the SAME
    clock position on the previous ``days`` sessions whose outcome is complete at the decision -
    sessions at least ``horizon / n_pos`` back, so a multi-day outcome has ended. NaN with fewer
    than 5 of them. Heston, Korajczyk & Sadka (2010) find a stock's half-hour return repeating at
    the same half hour on later days.

    Args:
        g: The grid.
        target: The label array of :func:`build_inputs` (label at the bar it is realized).
        valid: Its usable mask.
        horizon: The horizon in grid bars.
        days: Sessions averaged.

    Returns:
        A (buckets x names) array.
    """
    n, m = target.shape
    fwd = np.full((n, m), np.nan)                       # each decision's realized outcome
    fwd[:-horizon] = np.where(valid[horizon:], target[horizon:], np.nan)
    code, pos = g["code"], g["pos"]
    cube = np.full((code[-1] + 1, max(g["n_pos"], pos.max() + 1), m), np.nan)
    cube[code, pos] = fwd
    ok = np.isfinite(cube)
    zero = np.zeros((1, *cube.shape[1:]))
    total = np.concatenate([zero, np.cumsum(np.where(ok, cube, 0.0), axis=0)])
    count = np.concatenate([zero, np.cumsum(ok, axis=0)])
    back = max(1, math.ceil(horizon / g["n_pos"]))      # the latest session whose outcome is known
    hi = np.clip(np.arange(len(cube)) - back + 1, 0, None)
    lo = np.clip(hi - days, 0, None)
    k = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(k >= 5, (total[hi] - total[lo]) / k, np.nan)
    return mean[code, pos]


def build_inputs(panel: Panel, *, bar_minutes: int = 5, horizon: int = 1, options: str = "all",
                 use_daily: bool = True, use_market: bool = True, vol_days: int = 20,
                 overnight: bool = False, seasonal: bool = False) -> dict:
    """Everything the forecaster sees, on the grid, per name.

    The label at bucket p is the forward return of a decision at p - horizon. Without
    ``overnight`` it is the backward ``horizon``-sum of the z-scored bucket returns and counts
    only when decided inside the same session. With ``overnight`` it is log(close at p / close
    at p - horizon) - the close-to-close return over the horizon, overnight gaps included -
    divided by ``tscale`` at the decision: the name's causal close-to-close volatility over the
    previous ``vol_days`` sessions times sqrt(horizon sessions), so one unit is one typical move
    over the horizon. It counts when the decision and the label buckets are real (regular) ones
    exactly ``horizon`` positions of the session clock apart. The inputs are the same either way:
    a session's bucket returns start from its open, and the night enters as the daily ``gap``.
    ``seasonal`` adds :func:`seasonal_input` (``seas``, built from these labels).

    Returns:
        ``{"g": grid, "names", "y": z-scored bucket return (buckets x names), "target": the label
        above, "valid": target usable, "scale" (of the bucket returns), "tscale" (of the target:
        ``scale`` unless ``overnight``), "close": each bucket's last price, "past": {name:
        (buckets x k) array}, "past_names", "groups": {feature: group}, "future": (buckets +
        horizon + 1) x 5, "future_names", "horizon", "bar_minutes", "overnight",
        "options_attached"}``.
    """
    g = grid(panel, bar_minutes)
    intr = intraday_features(panel, g, vol_days=vol_days)
    scale = intr["scale"]
    with np.errstate(invalid="ignore", divide="ignore"):
        y = intr["ret"] / scale
    feats, groups = {}, {}
    for k in INTRADAY:
        x = intr[k]
        if k in ("ret", "rv", "range", "ret_open", "vwap_dist"):
            with np.errstate(invalid="ignore", divide="ignore"):
                x = x / scale
        feats[k], groups[k] = x, "intraday"
    if use_market:
        ms = np.sqrt(pd.DataFrame(intr["mkt"] ** 2).groupby(g["code"]).mean()
                     .rolling(vol_days, min_periods=3).mean().shift(1)).to_numpy()[g["code"]]
        with np.errstate(invalid="ignore", divide="ignore"):
            feats["mkt"], groups["mkt"] = intr["mkt"] / ms, "market"
    if use_daily:
        for k, x in daily_features(panel, g).items():
            feats[k], groups[k] = x, "daily"
    opt = option_features(panel, g, options)
    for k, x in opt.items():
        feats["opt_" + k], groups["opt_" + k] = x, "options"
    # one 0/1 "has data" input per option SOURCE (Darts refuses NaN, so every input is filled with
    # 0 in _covariates; the flag tells a missing source from a real 0)
    for src in OPTION_SET_SOURCES[options]:
        cols = [opt[n] for n in OPTION_SOURCES[src]]
        feats["has_" + src] = np.isfinite(np.stack(cols)).any(axis=0).astype(float)
        groups["has_" + src] = "flag"
    tscale = scale
    if overnight:      # close-to-close over the horizon, gaps included, in typical moves
        close, pad = intr["close"], np.full((horizon, scale.shape[1]), np.nan)
        daily = pd.DataFrame(close).groupby(g["code"]).last()      # each session's last price
        dvol = np.log(daily / daily.shift(1)).rolling(vol_days, min_periods=max(3, vol_days // 2))
        tscale = (dvol.std().shift(1) * math.sqrt(horizon / g["n_pos"])).to_numpy()[g["code"]]
        with np.errstate(invalid="ignore", divide="ignore"):
            tgt = np.vstack([pad, np.log(close[horizon:] / close[:-horizon])
                             / tscale[:-horizon]])
        decided = np.r_[np.zeros(horizon, bool), g["regular"][:-horizon]]   # decision p - H
        # H buckets must be H positions of the session clock: a missing hour (a data hole, or a
        # half-day stored without its post-close rows) would stretch the horizon unnoticed
        clock = g["code"] * g["n_pos"] + g["pos"]
        aligned = np.r_[np.zeros(horizon, bool), clock[horizon:] - clock[:-horizon] == horizon]
        valid = np.isfinite(tgt) & (decided & aligned & g["regular"])[:, None]
    else:
        tgt = pd.DataFrame(y).rolling(horizon, min_periods=horizon).sum().to_numpy()
        same = pd.Series(g["code"]).shift(horizon).to_numpy() == g["code"]  # decision bar p - H
        valid = np.isfinite(tgt) & same[:, None]
    if seasonal:
        feats["seas"], groups["seas"] = seasonal_input(g, tgt, valid, horizon), "seasonal"
    names = list(feats)
    past = {s: np.stack([np.clip(feats[k][:, j], -20, 20) for k in names], axis=1)
            .astype(np.float32) for j, s in enumerate(panel.symbols)}
    attached = any(k.startswith("opt_") for k in (panel.features or {}))
    return {"g": g, "names": list(panel.symbols), "y": y, "target": tgt, "valid": valid,
            "scale": scale, "tscale": tscale, "close": intr["close"], "past": past,
            "past_names": names,
            "groups": groups, "future": time_features(g, extra=horizon + 1),
            "future_names": list(TIME), "horizon": horizon, "bar_minutes": bar_minutes,
            "overnight": overnight, "options_attached": attached}


# ---------------------------------------------------------------------------------------------
# The model.
# ---------------------------------------------------------------------------------------------
def _quiet() -> None:
    """Silence torch / Lightning chatter (their own warnings and loggers only - never the caller's,
    so ``forecast()``'s missing-options warning still shows). Call AFTER importing them: Lightning
    resets its logger level on import."""
    warnings.filterwarnings("ignore", module=r"(torch|pytorch_lightning|lightning)\b")
    for n in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning.fabric",
              "lightning_fabric", "darts"):
        logging.getLogger(n).setLevel(logging.ERROR)


def _early_stopping():
    from pytorch_lightning.callbacks import EarlyStopping
    return EarlyStopping(monitor="val_loss", patience=2, min_delta=0.0)


def fast_bins(base):
    """``base`` (scikit-learn's histogram gradient boosting regressor or classifier) with a fast
    path for 0/1 sample weights - the walk-forward's "usable label" mask.

    scikit-learn 1.9 bins a weighted fit with 254 weighted percentiles per input, one sort each:
    a minute per fit for a window of 1,000 lagged inputs. With 0/1 weights its bin edges are the
    unweighted 'averaged_inverted_cdf' percentiles of the rows kept, so those are computed
    directly, about 18x faster: the same edges (an edge can rarely differ in its last bit, a
    rounding of a midpoint between two data values, which no value can fall on), so the same
    model (``tests/test_darts_forecast.py::test_fast_bins_is_bit_identical``). Any other weights
    (a recency half-life), and fits above the binner's 200,000-row subsample (where scikit-learn
    draws a weighted random subsample), take the stock path. It overrides a private method; that
    test fails if a scikit-learn upgrade changes it.
    """
    class Fast(base):
        def _bin_data(self, X, sample_weight, is_training_data):
            if (is_training_data and sample_weight is not None
                    and X.shape[0] <= self._bin_mapper.subsample
                    and np.isin(sample_weight, (0.0, 1.0)).all()):
                self._bin_mapper.fit(X[sample_weight > 0])
                return self._bin_mapper.transform(X)
            return super()._bin_data(X, sample_weight, is_training_data)

    Fast.__name__ = Fast.__qualname__ = f"Fast{base.__name__}"
    return Fast


def make_model(kind: str, *, horizon: int, lookback: int, cov_lags: int, past_names: list[str],
               groups: dict, n_epochs: int = 10, val_days: int = 0, gbm_iter: int = 100,
               device: str = "cpu", seed: int = 0, n_samples: int = 15_000):
    """A fresh Darts model of ``kind``; imports Darts lazily.

    Every model forecasts ONE point, ``horizon`` grid bars after its input window ends
    (``output_chunk_length=1``, ``output_chunk_shift=horizon-1``): the backward ``horizon``-sum
    target at that point is the forward return of the decision bar. Intraday / market inputs enter
    with ``cov_lags`` lags; daily, option, flag and seasonal inputs (constant within a session, or
    already about the forecast window) with their latest value only. The gbm trees use a FIXED
    ``gbm_iter`` (no internal early stopping - its split would be random rows of an autocorrelated
    series). ``linear`` is a ridge regression on the same lags, its penalty 0.1 x the training
    samples (a mild shrinkage: the inputs are in comparable units, most near 1).
    """
    from darts.utils.likelihood_models import QuantileRegression

    shift = horizon - 1
    if kind in ("gbm", "gbm_clf", "linear"):
        from darts.models import SKLearnClassifierModel, SKLearnModel
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge

        _quiet()
        lags_past = {n: (cov_lags if groups[n] in ("intraday", "market") else 1)
                     for n in past_names}
        # 200 per leaf on a ~15k-row window; with a long horizon consecutive labels share most of
        # their window, so a leaf must span several horizons to hold several independent outcomes
        leaf = int(np.clip(n_samples // 75, 20, 200))
        if horizon > 1:
            leaf = max(leaf, min(5 * horizon, n_samples // 10))
        hp = dict(max_iter=gbm_iter, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=leaf,
                  l2_regularization=1.0, early_stopping=False, random_state=seed)
        common = dict(lags=lookback, lags_past_covariates=lags_past, lags_future_covariates=[0],
                      output_chunk_length=1, output_chunk_shift=shift, use_static_covariates=True)
        if kind == "linear":
            return SKLearnModel(model=Ridge(alpha=0.1 * max(n_samples, 1)), **common)
        if kind == "gbm":
            return SKLearnModel(model=fast_bins(HistGradientBoostingRegressor)(**hp), **common)
        return SKLearnClassifierModel(model=fast_bins(HistGradientBoostingClassifier)(**hp),
                                      **common)

    import pytorch_lightning  # noqa: F401  (imported first so _quiet's levels stick)

    trainer = {"accelerator": device, "enable_progress_bar": False, "enable_model_summary": False,
               "logger": False, "enable_checkpointing": False}
    if val_days > 0:
        trainer["callbacks"] = [_early_stopping()]
    common = dict(input_chunk_length=lookback, output_chunk_length=1, output_chunk_shift=shift,
                  likelihood=QuantileRegression([0.1, 0.25, 0.5, 0.75, 0.9]), n_epochs=n_epochs,
                  batch_size=int(np.clip(n_samples // 40, 32, 512)), random_state=seed,
                  use_static_covariates=True,
                  pl_trainer_kwargs=trainer, optimizer_kwargs={"lr": 1e-3})
    if kind == "tide":
        from darts.models import TiDEModel
        _quiet()
        return TiDEModel(hidden_size=64, num_encoder_layers=1, num_decoder_layers=1,
                         decoder_output_dim=8, temporal_decoder_hidden=16, dropout=0.1, **common)
    if kind == "tft":
        from darts.models import TFTModel
        _quiet()
        return TFTModel(hidden_size=16, lstm_layers=1, num_attention_heads=2, dropout=0.1,
                        add_relative_index=False, **common)
    if kind == "tsmixer":
        from darts.models import TSMixerModel
        _quiet()
        return TSMixerModel(hidden_size=32, ff_size=32, num_blocks=2, dropout=0.1, **common)
    raise ValueError(f"model must be one of {MODELS}, got {kind!r}")


def schedule(g: dict, *, train_days: int, retrain_minutes: int, warmup: int) -> np.ndarray:
    """Retrain points (bucket indices). The first is the first bucket of session
    ``warmup + train_days``; then every ``retrain_minutes`` of trading time - a multiple of 390
    retrains before every k-th session, a smaller multiple of ``bar_minutes`` retrains intraday,
    0 trains ONCE (a static model)."""
    sf = g["sess_first"]
    k0 = warmup + train_days
    if k0 >= len(sf):
        return np.array([], dtype=int)
    if retrain_minutes == 0:
        return np.array([sf[k0]])
    if retrain_minutes % SESSION_MINUTES == 0:
        return sf[k0::retrain_minutes // SESSION_MINUTES].astype(int)
    step = retrain_minutes // g["bar_minutes"]
    return np.arange(sf[k0], len(g["code"]), step, dtype=int)


def _series(inp: dict, upto: int, kind: str, lo: int = 0, *, target=None):
    """Darts target series per name over buckets [lo, upto) (``target`` overrides the values)."""
    from darts import TimeSeries

    tgt_all = inp["target"] if target is None else target
    out = []
    for j, s in enumerate(inp["names"]):
        tg = tgt_all[lo:upto, j]
        if kind == "gbm_clf":
            tg = (tg > 0).astype(np.float32)
        tg = np.nan_to_num(tg).astype(np.float32)
        stat = pd.DataFrame({f"is_{n}": [float(n == s)] for n in inp["names"]})
        out.append(TimeSeries.from_times_and_values(pd.RangeIndex(lo, upto), tg[:, None],
                                                    columns=["target"]).with_static_covariates(stat))
    return out


def _covariates(inp: dict, kind: str, *, past=None):
    """Full-length past (per name) and future covariate series. Models slice what they need; a
    forecast never reads a past covariate after its decision bar
    (tests/test_darts_forecast.py::test_forecast_ignores_everything_after_its_decision_bar).
    NaN -> 0 (Darts refuses NaN inputs); the has_* flags mark missing option sources."""
    from darts import TimeSeries

    n = len(inp["g"]["code"])
    src = inp["past"] if past is None else past
    pc = [TimeSeries.from_times_and_values(pd.RangeIndex(0, n), np.nan_to_num(src[s]),
                                           columns=inp["past_names"]) for s in inp["names"]]
    fut = TimeSeries.from_times_and_values(pd.RangeIndex(0, len(inp["future"])), inp["future"],
                                           columns=inp["future_names"])
    return pc, [fut] * len(inp["names"])


def series_start(lo: int, lookback: int, horizon: int, lag_history: bool) -> int:
    """Where a training window's target series starts: at its first label ``lo``, or with
    ``lag_history`` ``lookback + horizon - 1`` buckets earlier, so its first labels can read their
    lagged target (a sample's target lags end ``horizon`` buckets before its label). The labels
    before ``lo`` get weight 0 (:func:`_weights`'s ``first``)."""
    return max(0, lo - lookback - horizon + 1) if lag_history else lo


def _weights(inp: dict, lo: int, upto: int, half_life: float, first: int | None = None):
    """Sample weights over [lo, upto): 0 for unusable targets and for labels before ``first``
    (the lag history of :func:`series_start`), else 0.5 ** (sessions back / half_life) when
    ``half_life`` > 0 (recent sessions count more), else 1."""
    from darts import TimeSeries

    code = inp["g"]["code"]
    age = code[upto - 1] - code[lo:upto]
    base = 0.5 ** (age / half_life) if half_life > 0 else np.ones(upto - lo)
    if first is not None:
        base = np.where(np.arange(lo, upto) >= first, base, 0.0)
    return [TimeSeries.from_times_and_values(
        pd.RangeIndex(lo, upto), (base * inp["valid"][lo:upto, j]).astype(np.float32)[:, None],
        columns=["w"]) for j in range(len(inp["names"]))]


def predict_segment(est, inp: dict, kind: str, first: int, last: int, *, past=None,
                    target=None) -> dict:
    """Forecasts of a FITTED model for decision bars ``first .. last`` (inclusive): each from
    buckets up to its decision bar, for the point ``horizon`` bars later (Darts' ``start`` is the
    bar after the decision). ``past`` / ``target`` override the inputs (importance studies).

    Returns:
        ``{"s": decision bars, "edge": (len, names) forecast in z units, "p_up", "q10", "q90"}``.
    """
    H = inp["horizon"]
    n, m = inp["target"].shape
    hi = min(last + H + 1, n)
    pc, fut = _covariates(inp, kind, past=past)
    full = _series(inp, hi, kind, 0, target=target)
    prob = kind == "gbm_clf" or kind in DEEP
    hf = est.historical_forecasts(full, past_covariates=pc, future_covariates=fut,
                                  start=int(first + 1), forecast_horizon=1, retrain=False,
                                  last_points_only=True, verbose=False,
                                  predict_likelihood_parameters=prob)
    out = {k: np.full((last - first + 1, m), np.nan) for k in ("edge", "p_up", "q10", "q90")}
    for j, f in enumerate(hf):
        s = np.asarray(f.time_index, dtype=int) - H
        keep = (s >= first) & (s <= last)
        vals, rows = f.values()[keep], s[keep] - first
        if kind in ("gbm", "linear"):
            out["edge"][rows, j] = vals[:, 0]
        elif kind == "gbm_clf":
            pu = vals[:, -1]
            out["p_up"][rows, j] = pu
            spread = 1.0 if inp.get("overnight") else math.sqrt(H)   # the target's typical size
            out["edge"][rows, j] = (2 * pu - 1) * math.sqrt(2 / math.pi) * spread
        else:                                      # quantiles 0.1, 0.25, 0.5, 0.75, 0.9
            vals = np.sort(vals, axis=1)           # an undertrained net can cross them
            out["edge"][rows, j] = vals[:, 2]
            out["q10"][rows, j], out["q90"][rows, j] = vals[:, 0], vals[:, 4]
            out["p_up"][rows, j] = _p_up(vals)
    out["s"] = np.arange(first, last + 1)
    return out


def walk_forward(inp: dict, *, model: str = "gbm", lookback: int = 12, cov_lags: int = 6,
                 train_days: int = 40, val_days: int = 0, retrain_minutes: int = 390,
                 n_epochs: int = 10, finetune_epochs: int = 0, recency_half_life: float = 0.0,
                 gbm_iter: int = 100, device: str = "cpu", seed: int = 0,
                 expanding: bool = False, lag_history: bool = False, warmup: int = 21,
                 verbose: bool = False) -> dict:
    """Out-of-sample forecasts for every decision bar after the first retrain point.

    At retrain point ``T`` the model is fit on the buckets of the last ``train_days`` sessions
    before ``T`` - with ``expanding``, on every bucket from the end of the ``warmup`` to ``T``
    (``train_days`` is then only the history required before the first fit). With ``lag_history``
    the window's first labels read their lagged target from before it (:func:`series_start`);
    without, the first ``lookback + horizon - 1`` labels of the window have no sample. For the deep
    models (:data:`DEEP`) with ``val_days`` > 0 the last ``val_days`` sessions are the validation
    slice for early stopping - a FRESH early-stopping state per fit - and are not fitted; its
    labels start ``horizon``
    buckets after the fitted ones end, so no validation return overlaps a training return. Every
    training label is the return up to a bucket before ``T``. With ``finetune_epochs`` > 0 a deep
    model is fit in full only at the first point and fine-tuned for ``finetune_epochs`` epochs
    afterwards. It then forecasts each decision bar ``s`` in ``[T, T_next)`` from buckets up to
    ``s``: the predicted point is ``s + horizon``, i.e. the return over buckets ``s+1 ..
    s+horizon``.

    Returns:
        ``{"edge": forecast forward return in z units (buckets x names, NaN where none), "p_up":
        P(forward return > 0) (classifier probability or read off the quantiles; NaN for
        ``gbm``), "q10", "q90", "fit": index of the fit that made each forecast, "points":
        retrain points, "log": DataFrame per fit, "model": the last fitted model}``.
    """
    g, H = inp["g"], inp["horizon"]
    n, m = inp["target"].shape
    pts = schedule(g, train_days=train_days, retrain_minutes=retrain_minutes, warmup=warmup)
    edge, p_up, q10, q90, fit_id = (np.full((n, m), np.nan) for _ in range(5))
    past, fut = _covariates(inp, model)
    deep = model in DEEP
    est, log = None, []
    sf = g["sess_first"]
    for k, T in enumerate(pts):
        t0 = time.time()
        sess = g["code"][T]
        lo = sf[min(warmup, sess)] if expanding else sf[max(0, sess - train_days)]
        T_next = pts[k + 1] if k + 1 < len(pts) else n
        last = min(T_next - 1, n - 1 - H)          # the last decision with its outcome stored
        if last < T:
            continue
        fresh = est is None or not deep or finetune_epochs == 0
        if fresh:
            est = make_model(model, horizon=H, lookback=lookback, cov_lags=cov_lags,
                             past_names=inp["past_names"], groups=inp["groups"], n_epochs=n_epochs,
                             val_days=val_days, gbm_iter=gbm_iter, device=device, seed=seed,
                             n_samples=(T - lo) * m)
        fit_hi, kw = T, {}
        if deep and val_days > 0:
            cut = sf[max(0, sess - val_days)]       # validation labels: cut .. T - 1
            fit_hi = cut - H + 1                    # last training label cut - H: windows disjoint
            val_lo = max(lo, cut - H - lookback + 1)
            est.trainer_params["callbacks"] = [_early_stopping()]       # fresh state per fit
            kw = dict(val_series=_series(inp, T, model, val_lo), val_past_covariates=past,
                      val_future_covariates=fut,
                      val_sample_weight=_weights(inp, val_lo, T, recency_half_life))
        if fit_hi - lo <= lookback + H + 10:
            continue
        s0 = series_start(lo, lookback, H, lag_history)
        tr = _series(inp, fit_hi, model, s0)
        w = _weights(inp, s0, fit_hi, recency_half_life, first=lo)
        if deep:
            kw["epochs"] = n_epochs if fresh else finetune_epochs
            kw["verbose"] = False
        est.fit(tr, past_covariates=past, future_covariates=fut, sample_weight=w, **kw)
        _quiet()
        seg = predict_segment(est, inp, model, T, last)
        s = seg["s"]
        edge[s], p_up[s], q10[s], q90[s] = seg["edge"], seg["p_up"], seg["q10"], seg["q90"]
        fit_id[s] = np.where(np.isfinite(seg["edge"]), k, np.nan)
        log.append({"fit": k, "retrain_bucket": int(T), "session": str(g["date"][T]),
                    "train_from": str(g["date"][lo]),
                    "train_last": str(g["date"][max(lo, fit_hi - 1)]),
                    "val_days": int(val_days if deep else 0),
                    "samples": int((fit_hi - lo) * m), "fresh": fresh,
                    "seconds": round(time.time() - t0, 2)})
        if verbose and (k % 10 == 0 or k == len(pts) - 1):
            print(f"fit {k + 1}/{len(pts)} {g['date'][T]} ({time.time() - t0:.1f}s)")
    return {"edge": edge, "p_up": p_up, "q10": q10, "q90": q90, "fit": fit_id, "points": pts,
            "log": pd.DataFrame(log), "model": est}


def _p_up(q: np.ndarray) -> np.ndarray:
    """P(x > 0) from the 0.1 / 0.25 / 0.5 / 0.75 / 0.9 quantiles: a linear CDF between them (tied
    quantiles share the mean of their levels), flat tails clipped to [0.05, 0.95]."""
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    out = np.full(len(q), np.nan)
    for i, row in enumerate(q):
        if not np.all(np.isfinite(row)):
            continue
        r = np.maximum.accumulate(row)
        vals, inv = np.unique(r, return_inverse=True)
        lv = np.bincount(inv, weights=levels) / np.bincount(inv)
        cdf0 = np.interp(0.0, vals, lv, left=0.05, right=0.95) if len(vals) > 1 else (
            0.05 if vals[0] > 0 else (0.95 if vals[0] < 0 else 0.5))
        out[i] = 1 - cdf0
    return out


# ---------------------------------------------------------------------------------------------
# Forecast cache and the trading rule.
# ---------------------------------------------------------------------------------------------
def fingerprint(panel: Panel) -> str:
    """A full-content identity of the panel (all five frames, the index, the option inputs) for
    the forecast caches: any changed bar or input gives a new key."""
    h = hashlib.sha1()
    h.update(",".join(panel.symbols).encode())
    h.update(pd.util.hash_pandas_object(pd.Series(panel.close.index), index=False).to_numpy())
    for frame in (panel.open_, panel.high, panel.low, panel.close, panel.volume):
        h.update(np.ascontiguousarray(np.nan_to_num(frame.to_numpy(float), nan=-1.2345)).tobytes())
    for k in sorted(panel.features or {}):
        f = panel.features[k]
        h.update(k.encode())
        h.update(pd.util.hash_pandas_object(f, index=True).to_numpy())
    return h.hexdigest()[:20]


def forecast(panel: Panel, **p) -> dict:
    """:func:`build_inputs` + :func:`walk_forward` for the forecast-relevant params ``p`` (see
    ``MODEL_KEYS``), memoised in-process: an Optuna study that only moves the trading rule pays
    for the walk-forward once. Warns when option inputs are requested but none are attached (a
    headless run: the option inputs are then 0 with their flags off - a different experiment)."""
    mp = {k: p.get(k, False) if k in LATE_KEYS else p[k] for k in MODEL_KEYS}
    key = (fingerprint(panel), tuple(sorted(mp.items())))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    inp = build_inputs(panel, bar_minutes=mp["bar_minutes"], horizon=mp["horizon"],
                       options=mp["options"], use_daily=mp["use_daily"],
                       use_market=mp["use_market"], vol_days=mp["vol_days"],
                       overnight=mp["overnight"], seasonal=mp["seasonal"])
    if mp["options"] != "none" and not inp["options_attached"]:
        warnings.warn(f"darts_forecast: options={mp['options']!r} but no panel.features['opt_*'] "
                      "is attached (research.attach_options): the option inputs are all missing.",
                      stacklevel=2)
    wf = walk_forward(inp, **{k: mp[k] for k in ("model", "lookback", "cov_lags", "train_days",
                                                 "val_days", "retrain_minutes", "n_epochs",
                                                 "finetune_epochs", "recency_half_life",
                                                 "gbm_iter", "device", "seed", "expanding",
                                                 "lag_history")},
                      warmup=max(21, mp["vol_days"] + 1))
    out = {"inputs": inp, **wf, "panel_id": key[0]}
    _CACHE[key] = out
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return out


def health(fc: dict, *, days: int) -> np.ndarray:
    """Per session: the rank correlation of the model's forecasts with the realized forward
    returns, over the decisions of the previous ``days`` sessions (pooled over names) whose
    outcome was complete before the session opened. NaN without history. (sessions,) array."""
    inp = fc["inputs"]
    code, sf = inp["g"]["code"], inp["g"]["sess_first"]
    H = inp["horizon"]
    n_sess = int(code.max()) + 1
    out = np.full(n_sess, np.nan)
    if days <= 0:
        return out
    real = np.full_like(inp["target"], np.nan)
    real[:-H] = inp["target"][H:]                           # realized forward return of decision s
    ok = np.isfinite(fc["edge"]) & np.isfinite(real) & np.vstack(
        [inp["valid"][H:], np.zeros((H, real.shape[1]), bool)])
    rows, cols = np.nonzero(ok)
    e_all, r_all, c_all = fc["edge"][rows, cols], real[rows, cols], code[rows]
    for d in range(n_sess):
        use = (c_all >= d - days) & (c_all < d) & (rows + H < sf[d])    # outcome in by the open
        if use.sum() >= 30:
            out[d] = pd.Series(e_all[use]).rank().corr(pd.Series(r_all[use]).rank())
    return out


def trade_frames(panel: Panel, fc: dict, *, sides: str = "both", edge_bp: float = 10.0,
                 p_min: float = 0.5, hold_bars: int = 0, stop_vol: float = 2.0,
                 risk_per_trade: float = 0.001, max_positions: int = 5, skip_open: int = 15,
                 skip_close: int = 15, eod_exit: bool = True, health_days: int = 5,
                 health_min: float = 0.0) -> dict:
    """The trading rule on the 1-minute panel, from the forecasts of :func:`forecast`.

    Each qualifying decision asks to hold its side from the next 1-minute bar for ``hold_bars``
    grid bars. The book follows that INTENDED POSITION path per name: the newest decision wins, a
    same-side one extends the hold, an opposite one closes it and the new side opens one minute
    later (one order per bar), and with ``eod_exit`` the position is flat from the session's last
    real bar. Entries / exits are the path's starts / ends, plus an entry at every decision's fill
    bar while the path holds its side: the engine ignores it while the position is open, and it
    re-opens one a stop has closed (a fresh decision is a fresh trade).

    Returns the engine's signal dict: ``entries`` / ``exits`` / ``short_entries`` /
    ``short_exits``, ``sl_stop`` (fraction of entry price, the entering decision's) and a dense
    ``cash_frac`` frame (risk-per-trade sizing, capped at one slot).
    """
    inp, g = fc["inputs"], fc["inputs"]["g"]
    R, H = inp["bar_minutes"], inp["horizon"]
    hold = hold_bars or H
    n1, m = panel.close.shape
    edge_bp_arr = fc["edge"] * inp.get("tscale", inp["scale"]) * 1e4    # target units -> bp
    p = fc["p_up"]
    has_p = np.isfinite(p)
    elapsed = (g["pos"] + 1) * R                     # minutes since the open, at the bucket close
    room = hold + (0 if eod_exit else 1)             # without the EOD exit the hold must end first
    ok_time = g["regular"] & (elapsed >= skip_open) & (g["left"] * R >= skip_close) & \
        (g["left"] >= room)
    ok = ok_time[:, None] & np.isfinite(edge_bp_arr)
    if health_days > 0:
        hs = health(fc, days=health_days)[g["code"]]
        ok &= (np.nan_to_num(hs, nan=-np.inf) > health_min)[:, None]
    longs = ok & (edge_bp_arr > edge_bp) & (~has_p | (p >= p_min))
    shorts = ok & (edge_bp_arr < -edge_bp) & (~has_p | (1 - p >= p_min))
    if sides == "long":
        shorts[:] = False
    elif sides == "short":
        longs[:] = False
    # the decision's side, stop and size, placed on its FILL row (the 1m bar after the decision)
    move = inp["scale"] * math.sqrt(hold)                          # typical log move over the hold
    stop_b = stop_vol * move if stop_vol > 0 else np.full_like(move, np.nan)
    slot = 1.0 / max_positions
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_b = np.where(np.isfinite(stop_b) & (stop_b > 0),
                          np.minimum(slot, risk_per_trade / stop_b), slot)
    idx, cols = panel.close.index, panel.close.columns
    frame = lambda a: pd.DataFrame(a, index=idx, columns=cols)     # noqa: E731
    rr, cc = np.nonzero(longs | shorts)                            # (bucket, name) decisions
    dec_row = g["end"][rr]                                         # the decision bar (1m row)
    raw_l, raw_s = np.zeros((n1, m), bool), np.zeros((n1, m), bool)
    raw_l[dec_row, cc] = longs[rr, cc]
    raw_s[dec_row, cc] = shorts[rr, cc]
    fill_l = bt.next_bar(frame(raw_l)).to_numpy()                  # executes at the next 1m open
    fill_s = bt.next_bar(frame(raw_s)).to_numpy()
    side_1m = np.where(fill_l, 1.0, np.where(fill_s, -1.0, np.nan))
    stop_1m, frac_1m = np.full((n1, m), np.nan), np.full((n1, m), np.nan)
    fill_row = np.minimum(dec_row + 1, n1 - 1)                     # the same row next_bar marks
    stop_1m[fill_row, cc] = stop_b[rr, cc]
    frac_1m[fill_row, cc] = np.nan_to_num(frac_b, nan=slot)[rr, cc]
    hold_min = hold * R
    codes = g["codes_1m"]
    carry = lambda a: pd.DataFrame(a).groupby(codes).ffill(limit=hold_min - 1).to_numpy()  # noqa: E731
    want = np.nan_to_num(carry(side_1m))                           # newest decision wins
    stop_now, frac_now = carry(stop_1m), carry(frac_1m)
    want[~g["regular_1m"]] = 0.0
    if eod_exit:
        want[g["last_1m"]] = 0.0                                   # flat at the last real bar
    prev = np.vstack([np.zeros((1, m)), want[:-1]])
    want[(prev != 0) & (want != 0) & (np.sign(prev) != np.sign(want))] = 0.0   # switch via flat
    prev = np.vstack([np.zeros((1, m)), want[:-1]])
    long_in, short_in = want > 0, want < 0
    return {"entries": frame(long_in & ((prev <= 0) | fill_l)),
            "exits": frame((prev > 0) & (want <= 0)),
            "short_entries": frame(short_in & ((prev >= 0) | fill_s)),
            "short_exits": frame((prev < 0) & (want >= 0)),
            "sl_stop": frame(stop_now),
            "cash_frac": frame(np.where(np.isfinite(frac_now), frac_now, slot))}


def signals(
    panel: Panel,
    *,
    model: str = "gbm",
    bar_minutes: int = 5,
    horizon: int = 1,
    lookback: int = 12,
    cov_lags: int = 6,
    train_days: int = 40,
    val_days: int = 0,
    retrain_minutes: int = 390,
    n_epochs: int = 10,
    finetune_epochs: int = 0,
    recency_half_life: float = 0.0,
    options: str = "all",
    use_daily: bool = True,
    use_market: bool = True,
    vol_days: int = 20,
    gbm_iter: int = 100,
    device: str = "cpu",
    seed: int = 0,
    overnight: bool = False,
    expanding: bool = False,
    seasonal: bool = False,
    lag_history: bool = False,
    sides: str = "both",
    edge_bp: float = 10.0,
    p_min: float = 0.5,
    hold_bars: int = 0,
    stop_vol: float = 2.0,
    risk_per_trade: float = 0.001,
    max_positions: int = 5,
    skip_open: int = 15,
    skip_close: int = 15,
    eod_exit: bool = True,
    health_days: int = 5,
    health_min: float = 0.0,
) -> dict:
    """Walk-forward Darts forecasts turned into execution-timed long / short trades.

    Args:
        panel: The STORED 1-minute regular-hours panel (``panel.features['opt_*']`` optional).
        model: ``gbm`` | ``gbm_clf`` | ``tide`` | ``tft`` (see the module docstring).
        bar_minutes: Forecast grid in minutes: 1, 5, 10, 15, 30 or 60 (60: the session's last
            bucket is 30 minutes).
        horizon: Grid bars ahead the forecast covers (and the default hold).
        lookback: Target lags (gbm) / input window in grid bars (tide, tft).
        cov_lags: Lags of the intraday and market inputs (gbm).
        train_days: Sessions in each training window (rolling).
        val_days: tide / tft only - last sessions of each window held out for early stopping.
        retrain_minutes: Retrain every this many trading minutes (390 = each session, 0 = once).
        n_epochs: tide / tft epochs of a full fit.
        finetune_epochs: tide / tft - after the first fit, fine-tune this many epochs instead of
            refitting from scratch (0 = always refit).
        recency_half_life: Sessions after which a training sample's weight halves (0 = equal).
        options: ``none`` | ``iv`` | ``all`` option inputs.
        use_daily: Include the daily inputs.
        use_market: Include the other names' return (market proxy).
        vol_days: Sessions behind the volatility scale.
        gbm_iter: Boosting iterations (gbm, gbm_clf), fixed.
        device: ``cpu`` | ``mps`` for tide / tft.
        seed: Model seed.
        overnight: The horizon may span sessions and the target is the close-to-close return,
            overnight gaps included (multi-day forecasts; not tradable by this rule yet).
        expanding: Train on all history before each retrain point (``train_days`` = the history
            required before the first fit).
        seasonal: Add the seasonality input (:func:`seasonal_input`).
        lag_history: Let each training window's first labels read their lagged target from
            before the window (:func:`series_start`).
        sides: ``both`` | ``long`` | ``short``.
        edge_bp: Minimum |forecast return| in bp to trade.
        p_min: Minimum P(direction) for models that give one (0.5 = off).
        hold_bars: Grid bars to hold (0 = ``horizon``).
        stop_vol: Stop at this many typical moves over the hold (0 = no stop).
        risk_per_trade: Fraction of the wallet lost at the stop; size = min(slot, risk / stop).
        max_positions: Slots: the largest entry is 1 / max_positions of free cash.
        skip_open: No decisions in the first N minutes of the session.
        skip_close: No decisions in the last N minutes (a hold never crosses the close).
        eod_exit: Flat at the session's last real bar.
        health_days: Model-health gate window in sessions (0 = off).
        health_min: Trade only when the trailing forecast/outcome rank correlation exceeds it.

    Returns:
        ``entries`` / ``exits`` / ``short_entries`` / ``short_exits`` / ``sl_stop`` /
        ``cash_frac`` (a dense per-bar frame: one shared wallet, risk-per-trade sizing).
    """
    fc = forecast(panel, model=model, bar_minutes=bar_minutes, horizon=horizon,
                  lookback=lookback, cov_lags=cov_lags, train_days=train_days, val_days=val_days,
                  retrain_minutes=retrain_minutes, n_epochs=n_epochs,
                  finetune_epochs=finetune_epochs, recency_half_life=recency_half_life,
                  options=options, use_daily=use_daily, use_market=use_market, vol_days=vol_days,
                  gbm_iter=gbm_iter, device=device, seed=seed, overnight=overnight,
                  expanding=expanding, seasonal=seasonal, lag_history=lag_history)
    return trade_frames(panel, fc, sides=sides, edge_bp=edge_bp, p_min=p_min,
                        hold_bars=hold_bars, stop_vol=stop_vol, risk_per_trade=risk_per_trade,
                        max_positions=max_positions, skip_open=skip_open, skip_close=skip_close,
                        eod_exit=eod_exit, health_days=health_days, health_min=health_min)


def forecast_valid(params: dict) -> bool:
    """Reject impossible FORECASTER knobs: an unknown model or option set, a grid bar outside
    :data:`BAR_CHOICES`, a horizon below 1 (or of a session or more without ``overnight``: no
    label would stay inside a session), a retrain cadence that is neither whole sessions nor an
    intraday multiple of the bar, and a rolling training window too short for the walk-forward to
    fit (it needs more than ``lookback + 2 * horizon + 10`` buckets besides its validation)."""
    p = params
    R = p.get("bar_minutes", 5)
    rm = p.get("retrain_minutes", 390)
    H, n_pos = p.get("horizon", 1), math.ceil(SESSION_MINUTES / R)
    train, val = p.get("train_days", 40), p.get("val_days", 0)
    return (p.get("model", "gbm") in MODELS and p.get("options", "all") in OPTION_SETS
            and R in BAR_CHOICES and H >= 1 and (p.get("overnight", False) or n_pos > H)
            and (rm == 0 or rm % SESSION_MINUTES == 0 or (rm < SESSION_MINUTES and rm % R == 0))
            and train > val and (p.get("expanding", False)
                                 or (train - val) * n_pos > p.get("lookback", 12) + 2 * H + 10))


def valid(params: dict) -> bool:
    """Reject impossible TRADING combinations: bad forecaster knobs (:func:`forecast_valid`), an
    unknown side, a probability floor or slot count that cannot work, and a hold that cannot fit
    the session - the book is flat by the close, so an ``overnight`` forecast cannot be traded."""
    p = params
    R = p.get("bar_minutes", 5)
    hold = p.get("hold_bars", 0) or p.get("horizon", 1)
    return (forecast_valid(p) and not p.get("overnight", False)
            and p.get("sides", "both") in SIDES and p.get("skip_close", 15) >= 0
            and 0.5 <= p.get("p_min", 0.5) < 1.0 and p.get("max_positions", 5) >= 1
            and hold * R < SESSION_MINUTES - p.get("skip_open", 15))
