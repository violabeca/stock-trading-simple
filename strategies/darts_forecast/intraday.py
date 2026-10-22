"""darts_forecast intraday helpers (``darts_forecast_intraday.ipynb``): which bar size, input
span and output span let a Darts model predict the PATH of the rest of a trading day best, with
the night left out?

The cross-night path explorer (``explorer.py``) let paths run through the night. Here every
prediction stays inside one regular session (09:30-16:00 New York):

1. the target is each bar's INTRADAY move (a session's first bar from its own open: the overnight
   gap is an input, never a target) in typical moves for that stock and time of day, from ONE
   1-minute profile that every bar size shares (:func:`scales`);
2. every example is one STOCK-DAY: the previous session's last bars (input only) and the whole
   session (:func:`fit`), so no window reaches past the close or across a night. A "today" flag
   (0 on the previous session's bars, 1 on today's) and the clock tell the model when each input
   bar is from; the day's own facts (the gap, yesterday, the weekday, the calendar days since the
   previous session) are static inputs;
3. every configuration is graded on ONE truth: the log price change from the FILL (the first
   trade after the decision) to fixed checkpoints 10 minutes to 6 hours later, in one typical-move
   unit (:func:`outcomes`, :func:`grade`), its scale fit month by month on earlier months only,
   beside two baselines treated the same way (the time-of-day mean path and a ridge regression on
   a few summaries, :func:`baselines`);
4. the whole grid of bar size x input span x output span is measured (:func:`search`), a planted
   effect checks that the pipeline can find one (:func:`planted`), a model-free map shows which
   part of the day predicts which (:func:`atlas`), and ONE configuration, fixed by a rule before
   the search, is read once on TEST.

Half-days and the sessions right after them are left out (:func:`usable`). Loaded by path from
the notebook. It builds on ``explorer.py`` (the stock picks, the window, the networks), which it
loads once and exposes as :data:`ex` (with ``ex.rs``: research.py).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """A helper module of this folder, loaded by path once (shared with the explorer)."""
    key = f"darts_{name}"
    path = HERE / f"{name}.py"
    mod = sys.modules.get(key)
    if mod is not None and Path(getattr(mod, "__file__", "")) == path:
        return mod
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


ex = _sibling("explorer")
rs, SM = ex.rs, ex.SM
SESSION = 390
#: Part of every saved result's key: editing this file recomputes them.
INTRADAY_VERSION = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:10]
#: The grid the search measures (every cell): minutes per bar, minutes read, minutes predicted.
GRANULARITIES = (1, 2, 3, 5, 10)
INPUT_SPANS = (30, 60, 120, 240, 390)          # 390: a whole session (today + yesterday's end)
OUTPUT_SPANS = (10, 30, 60, 120, 240, 360)     # 360: from 10:00 to the close
#: Decisions every 30 minutes from 10:00 to 15:30 (minutes after the 09:30 open): every bar
#: size closes a bar there. A decision is graded when its whole output span ends by 16:00.
DECISION_MINUTES = tuple(range(30, 361, 30))
#: The graded points of a path: minutes after the decision (each configuration up to its span).
CHECKPOINTS = (10, 20, 30, 60, 90, 120, 180, 240, 300, 360)
BLOCK = 10          # the truth is a sum of 10-minute blocks; the luck band flips each block
FILL_WITHIN = 10    # a decision without a trade in the next 10 minutes cannot fill: skipped
STRIDE_MINUTES = 10  # a training window about every 10 minutes of the clock, at every bar size
ES_DAYS = 5         # sessions held out at the end of the training data (early stop, first scale)
#: Sessions the TEST refit holds out for its scale (the last ES_DAYS of them also stop its
#: training early): about a month, as VALIDATION's later months get; TEST is one month, so no
#: earlier graded month can add to it.
TEST_CAL_DAYS = 20
CLIP = 10.0         # a TRAINING target beyond 10 typical moves counts as 10 (never the truth)
MODELS = ("tide", "tsmixer", "dlinear")
#: The fixed settings: sessions behind the scales, minutes smoothed in the time-of-day profile,
#: seeds.
SETTINGS = {"vol_days": 20, "smooth": 5, "seed": 0, "device": "cpu", "lr": 1e-3}
#: The ridge baseline's penalties, one chosen per checkpoint by cross-validation over sessions.
RIDGE_ALPHAS = tuple(10.0 ** np.arange(0, 6.5, 0.5))
#: The planted effects of the positive control: correlation with the 30-minute move.
PLANTED = (0.05, 0.1, 0.2)
#: The smaller settings of a smoke run (``knobs(smoke=True)``).
SMOKE = {"names": 4, "months": 3, "train_months": 1, "granularities": (5, 10),
         "inputs": (30, 60), "outputs": (10, 30), "epochs": 1, "draws": 30,
         "compare": ("dlinear", "tide"), "default": (10, 60, 30), "planted": (0.2,)}
#: Inputs read bar by bar (past covariates): known at the bar's close.
PAST = ("ret", "rv", "range", "clv", "flow", "rvol", "ret_open", "vwap_dist", "mkt")
#: Inputs fixed for the day (static covariates): known at the open.
DAILY = ("prev_ret", "ret5", "ret20", "vol20", "pos20", "prev_clv", "prev_rvol", "gap")
#: Inputs known in advance for every bar (future covariates).
CLOCK = ("tod_sin", "tod_cos", "tod_frac", "today")
WORDS = {
    "tod_sin": "time of day (sine)", "tod_cos": "time of day (cosine)",
    "tod_frac": "time of day (fraction of the session: 0 = 09:30)",
    "today": "1 on today's bars, 0 on the previous session's",
    "dow": "day of the week", "days_since": "calendar days since the previous session "
    "(3 after a weekend)", "opex_week": "monthly option-expiry week",
}
_OHLC = ("open_", "high", "low", "close")
_PREP: dict = {}
_GI: dict = {}
_BASE: dict = {}


# ---------------------------------------------------------------------------------------------
# The run settings.
# ---------------------------------------------------------------------------------------------
def knobs(*, smoke: bool, **run) -> dict:
    """The run's settings: the notebook's knobs, with the smaller smoke settings on top when
    ``smoke`` (:data:`SMOKE`).

    Args:
        smoke: A quick end-to-end test.
        **run: ``model``, ``window``, ``train_months``, ``test_months``, ``granularities``,
            ``inputs``, ``outputs``, ``epochs``, ``draws``, ``compare``, ``default``,
            ``planted``, ``round_trip_bp``.

    Returns:
        The settings dict, plus ``smoke``.
    """
    k = {**run, "smoke": smoke}
    if smoke:
        a, b = (pd.Timestamp(x) for x in k["window"])
        start = (b + pd.Timedelta(days=1) - pd.DateOffset(months=SMOKE["months"])).date()
        k.update(window=(str(max(a.date(), start)), str(b.date())),
                 **{x: SMOKE[x] for x in ("train_months", "granularities", "inputs", "outputs",
                                          "epochs", "draws", "compare", "default", "planted")})
    return k


def knob_table(k: dict) -> pd.DataFrame:
    """The run's settings as a table (one row per knob)."""
    show = {"smoke": "quick end-to-end test (smaller everything)",
            "model": "the Darts model of the search",
            "window": "the months studied (TRAIN -> VALIDATION -> TEST)",
            "train_months": "months the first fit learns from",
            "test_months": "last months, read once at the end",
            "granularities": "bar sizes measured (minutes)",
            "inputs": "input spans measured (minutes read; 390 = a whole session)",
            "outputs": "output spans measured (minutes predicted, never past 16:00)",
            "epochs": "at most this many training passes (stopped early)",
            "draws": "make-believe futures behind each luck band",
            "compare": "models compared in §6",
            "default": "(bar minutes, input minutes, output minutes) of §5-§7",
            "planted": "planted effect sizes of §7 (correlation with the 30-minute move)",
            "round_trip_bp": "cost of a round trip (bp) behind the break-even line"}
    return pd.DataFrame([{"knob": n, "value": str(k.get(n)), "meaning": w}
                         for n, w in show.items()]).set_index("knob")


# ---------------------------------------------------------------------------------------------
# The data: placeholders masked, minute arrays, the time-of-day scale, the graded truth.
# ---------------------------------------------------------------------------------------------
def full_clock(panel):
    """The panel on the full exchange clock: 390 one-minute rows per session (09:30-15:59 New
    York); a minute missing from the data becomes a minute without a trade."""
    idx = panel.close.index
    days = pd.DatetimeIndex(sorted(set(idx.tz_convert(panel.tz).normalize().tz_localize(None))))
    clock = days.repeat(SESSION) + pd.to_timedelta(
        np.tile(np.arange(SESSION) + 570, len(days)), unit="min")
    want = clock.tz_localize(panel.tz).tz_convert(idx.tz)
    if len(want) == len(idx) and bool((want == idx).all()):
        return panel
    out = {k: getattr(panel, k).reindex(want) for k in _OHLC}
    return replace(panel, **out, volume=panel.volume.reindex(want).fillna(0.0))


def prepare(panel):
    """The 1-minute panel the study reads: on the full clock (:func:`full_clock`), with IBKR's
    volume-0 placeholder minutes (a flat copy of the previous close: nobody traded) masked as no
    trade (``burst_scan/research.py`` does the same for the extended hours). A placeholder left in
    is a stale price that the next trade predictably corrects: free, untradable "skill".

    Returns:
        The prepared panel (``panel`` is not modified).
    """
    p = full_clock(panel)
    none = p.volume.fillna(0.0) == 0
    return replace(p, **{k: getattr(p, k).mask(none) for k in _OHLC})


def minute_arrays(panel) -> dict:
    """A prepared panel as (sessions x 390 x stocks) arrays.

    Returns:
        ``open`` / ``close`` / ``volume`` (NaN prices without a trade or after a half-day's 13:00
        close), ``cf`` (the session's last traded close so far, NaN before its first trade),
        ``r1`` (each minute's intraday log move: from the session's last traded close, its first
        trade from its own open, so the night is never inside; NaN without a trade), ``dates``
        (session dates, "YYYY-MM-DD") and ``names``.
    """
    n = len(panel.close) // SESSION
    shape = (n, SESSION, len(panel.symbols))
    reg = SM["regular_rows"](panel).reshape(n, SESSION)[..., None]
    # one memory layout whatever the frames' own: a sum over a differently laid-out copy of the
    # same numbers rounds differently in the last bit, which the truncation check would flag
    arr = lambda k: np.ascontiguousarray(getattr(panel, k).to_numpy(float)).reshape(shape)  # noqa: E731
    op, C = (np.where(reg, arr(k), np.nan) for k in ("open_", "close"))
    V = arr("volume")
    ok = np.isfinite(C)
    last = np.maximum.accumulate(np.where(ok, np.arange(SESSION)[None, :, None], -1), axis=1)
    cf = np.take_along_axis(C, np.maximum(last, 0), axis=1)
    cf[last < 0] = np.nan
    prev = np.concatenate([np.full((n, 1, shape[2]), np.nan), cf[:, :-1]], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = np.where(np.isfinite(prev), np.log(C / prev), np.log(C / op))
    local = panel.close.index[::SESSION].tz_convert(panel.tz)
    return {"open": op, "close": C, "volume": V, "cf": cf, "r1": r1,
            "dates": np.array([str(d) for d in local.date]), "names": list(panel.symbols)}


def day_open_gap(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per session and stock: the first traded open, and the overnight gap (log of that open over
    the previous session's last traded close; NaN on the first session)."""
    op, cf = m["open"], m["cf"]
    o1 = np.take_along_axis(op, np.isfinite(op).argmax(axis=1)[:, None, :], axis=1)[:, 0]
    prev = np.r_[np.full((1, op.shape[2]), np.nan), cf[:-1, SESSION - 1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return o1, np.log(o1 / prev)


def scales(m: dict, *, vol_days: int = SETTINGS["vol_days"],
           smooth: int = SETTINGS["smooth"]) -> dict:
    """The typical-move unit, shared by every bar size and the truth: a stock's typical 1-minute
    move times the market's time-of-day profile, both from EARLIER sessions only.

    ``sigma`` (sessions x stocks) is the root-mean-square 1-minute intraday move over the previous
    ``vol_days`` sessions. ``prof2`` (sessions x 390) is the squared-move profile of the clock:
    each minute's squared move relative to its stock's ``sigma``, averaged over the stocks and
    the previous ``vol_days`` sessions, smoothed over ``smooth`` neighbouring minutes and scaled
    to average 1 over the session (pooling the stocks keeps it from chasing one stock's news). A
    stretch of minutes [a, b) then has the typical move ``sigma * sqrt(cum[b] - cum[a])``.

    Returns:
        ``{"sigma", "prof2", "cum"}`` (``cum``: sessions x 391, the running sum of ``prof2``).
    """
    n = len(m["dates"])
    r2 = m["r1"] ** 2
    lo = max(3, vol_days // 2)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        day = np.nanmean(r2, axis=1)
        sig = np.sqrt(pd.DataFrame(day).rolling(vol_days, min_periods=lo).mean().shift(1)
                      .to_numpy())
        rel = np.nanmean(r2 / sig[:, None, :] ** 2, axis=2)
    prof = pd.DataFrame(rel).rolling(vol_days, min_periods=lo).mean().shift(1)
    prof = prof.T.rolling(smooth, center=True, min_periods=1).mean().T.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof = prof / np.nanmean(prof, axis=1, keepdims=True)
    prof = np.where(np.isfinite(prof), prof, 1.0)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(prof, axis=1)], axis=1)
    return {"sigma": sig, "prof2": prof, "cum": cum}


def usable(dates) -> np.ndarray:
    """Sessions the study trains on and grades: neither a half-day (closed at 13:00) nor the
    session right after one (its input would read a 13:00 close as yesterday's end), nor the
    first session (no previous one)."""
    half = np.isin(np.asarray(dates), list(SM["HALF_DAYS"]))
    return ~half & ~np.r_[True, half[:-1]]


def outcomes(m: dict, sc: dict) -> dict:
    """The graded truth of every decision (session x decision time x stock), the same for every
    configuration.

    A decision at minute ``t`` (its bar closes there) fills at the open of the first traded minute
    in ``t .. t + FILL_WITHIN - 1`` (none: the decision is skipped, ``ok`` False). Its path is
    the log price change from that fill to the last traded close before each checkpoint, built
    from 10-minute blocks (``B``; block 0 starts at the fill), never past 16:00. The unit ``U`` is
    the typical move over those minutes (:func:`scales`); ``z = A / U``. The first 1-3 minutes'
    bounce off the decision bar's last trade (bid-ask, stale prints) is not in it.

    Returns:
        Per decision row: ``s`` (session), ``t`` (minute), ``k`` (stock), ``ok``, ``usable``,
        ``B`` (rows x 36 block moves), ``key`` (rows x 36: the clock block, shared by every stock
        and decision, that each block's luck coin is keyed on), ``A`` / ``U`` / ``z`` (rows x
        checkpoints, NaN past 16:00), ``p0`` (the decision bar's close), ``month``.
    """
    n, _, N = m["close"].shape
    s, t, k = (x.ravel() for x in np.meshgrid(np.arange(n), np.array(DECISION_MINUTES),
                                              np.arange(N), indexing="ij"))
    op, cf = m["open"], m["cf"]
    win = t[:, None] + np.arange(FILL_WITHIN)
    traded = np.isfinite(m["close"][s[:, None], win, k[:, None]]) \
        & np.isfinite(op[s[:, None], win, k[:, None]])
    ok = traded.any(axis=1)
    fill = t + traded.argmax(axis=1)
    j = np.arange(1, SESSION // BLOCK - 2)                   # 36 blocks: up to 6 hours
    ends = t[:, None] + BLOCK * j - 1
    e = cf[s[:, None], np.minimum(ends, SESSION - 1), k[:, None]]
    e[ends >= SESSION] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        logp = np.log(np.c_[op[s, fill, k], e])
    B = np.diff(logp, axis=1)
    B[~ok] = np.nan
    cps = np.array(CHECKPOINTS)
    A = np.cumsum(B, axis=1)[:, cps // BLOCK - 1]
    stop = t[:, None] + cps
    cum = sc["cum"]
    with np.errstate(invalid="ignore"):
        U = sc["sigma"][s, k][:, None] * np.sqrt(
            cum[s[:, None], np.minimum(stop, SESSION)] - cum[s, t][:, None])
    U[stop > SESSION] = np.nan
    key = s[:, None] * (SESSION // BLOCK) + np.minimum(t[:, None] // BLOCK + j - 1,
                                                       SESSION // BLOCK - 1)
    dates = m["dates"]
    return {"s": s, "t": t, "k": k, "ok": ok, "usable": usable(dates)[s] & ok, "B": B,
            "key": key, "A": A, "U": U, "z": A / U, "p0": cf[s, t - 1, k],
            "month": np.array([d[:7] for d in dates])[s]}


def prepared(panel) -> dict:
    """Everything configuration-free, built once per panel content (kept in-process, the last 4):
    the prepared panel, its minute arrays, the scales and the truth."""
    key = SM["fingerprint"](panel)
    if key not in _PREP:
        p = prepare(panel)
        m = minute_arrays(p)
        sc = scales(m)
        _PREP[key] = {"key": key, "panel": p, "m": m, "sc": sc, "oc": outcomes(m, sc)}
        while len(_PREP) > 4:
            _PREP.pop(next(iter(_PREP)))
    return _PREP[key]


# ---------------------------------------------------------------------------------------------
# The model's view: bars of one size, the stock-day series, fit and predict.
# ---------------------------------------------------------------------------------------------
def grid_inputs(pp: dict, g: int) -> dict:
    """Everything a model on ``g``-minute bars reads and predicts (kept in-process per panel and
    bar size).

    The target ``y`` is each bar's intraday log move (the sum of its minutes' ``r1``) divided by
    its typical move ``u`` (:func:`scales`, the bar's minutes), clipped at ±:data:`CLIP`. The bar
    inputs are ``strategy.build_inputs``' intraday and market ones (:data:`PAST`), the day's
    inputs its daily ones (:data:`DAILY`, the values at the session's first bar).

    Returns:
        ``y`` / ``valid`` / ``u`` (bars x stocks), ``past`` ({stock: bars x 9}), ``daily``
        (sessions x stocks x 8), ``clock`` (bars x 3: sin, cos, fraction of the session),
        ``dow`` / ``opex`` / ``days_since`` (per session), ``sf`` (each session's first bar),
        ``n_pos`` (bars per session), ``g``, ``names``, ``dates``.
    """
    key = (pp["key"], g)
    if key in _GI:
        return _GI[key]
    if SESSION % g:
        raise ValueError(f"{g}-minute bars do not divide the 390-minute session")
    m, sc = pp["m"], pp["sc"]
    n, N, n_pos = len(m["dates"]), len(m["names"]), SESSION // g
    inp = SM["build_inputs"](pp["panel"], bar_minutes=g, horizon=1, options="none",
                             vol_days=SETTINGS["vol_days"])
    gr = inp["g"]
    if len(gr["code"]) != n * n_pos or not (gr["pos"] == np.tile(np.arange(n_pos), n)).all():
        raise ValueError(f"the {g}-minute grid is not {n_pos} bars per session")
    r = m["r1"].reshape(n, n_pos, g, N)
    rb = np.where(np.isfinite(r).any(axis=2), np.nansum(r, axis=2), np.nan)
    u = sc["sigma"][:, None, :] * np.sqrt(np.diff(sc["cum"][:, ::g], axis=1))[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.clip(rb / u, -CLIP, CLIP).reshape(n * n_pos, N)
    names = inp["past_names"]
    ip, idl = [names.index(x) for x in PAST], [names.index(x) for x in DAILY]
    sf = np.arange(n) * n_pos
    tf = SM["time_features"](gr)
    days = pd.to_datetime(pd.Series(m["dates"]))
    out = {"y": y, "valid": np.isfinite(y), "u": u.reshape(n * n_pos, N),
           "past": {k: inp["past"][s][:, ip] for k, s in enumerate(m["names"])},
           "daily": np.stack([inp["past"][s][sf][:, idl] for s in m["names"]], axis=1),
           "clock": tf[:, :3], "dow": tf[sf, 3], "opex": tf[sf, 4],
           "days_since": np.r_[1, days.diff().dt.days.to_numpy()[1:]].astype(float),
           "sf": sf, "n_pos": n_pos, "g": g, "names": m["names"], "dates": m["dates"],
           "_fut": {}, "_stat": {}, "_day": {}}
    _GI[key] = out
    while len(_GI) > 6:
        _GI.pop(next(iter(_GI)))
    return out


def static(gi: dict, k: int, s: int) -> pd.DataFrame:
    """The static inputs of stock ``k`` on session ``s`` (one row): which stock (one-hot), the
    day's :data:`DAILY` inputs, the weekday, the calendar days since the previous session and the
    option-expiry week (all known at the open; a missing value is 0)."""
    if (k, s) not in gi["_stat"]:
        row = {f"is_{x}": float(i == k) for i, x in enumerate(gi["names"])}
        row.update(zip(DAILY, np.nan_to_num(gi["daily"][s, k]).astype(float), strict=True))
        row.update(dow=float(gi["dow"][s]), days_since=gi["days_since"][s] / 4.0,
                   opex_week=float(gi["opex"][s]))
        gi["_stat"][(k, s)] = pd.DataFrame({x: [v] for x, v in row.items()})
    return gi["_stat"][(k, s)]


def future(gi: dict, s: int, L: int):
    """The inputs known in advance of session ``s``'s stock-day series (every stock's): the clock
    (:data:`CLOCK`) of the previous session's last ``L - 1`` bars and of the whole session, and
    the "today" flag (0 on the previous session's bars, 1 on today's). It ends at the close, so a
    prediction past 16:00 is refused by Darts itself."""
    if (s, L) not in gi["_fut"]:
        from darts import TimeSeries

        a, b = gi["sf"][s] - (L - 1), gi["sf"][s] + gi["n_pos"]
        today = (np.arange(a, b) >= gi["sf"][s]).astype(np.float32)[:, None]
        gi["_fut"][(s, L)] = TimeSeries.from_times_and_values(
            pd.RangeIndex(a, b), np.hstack([gi["clock"][a:b], today]), columns=list(CLOCK))
    return gi["_fut"][(s, L)]


def past_series(gi: dict) -> list:
    """Per stock, its bar-by-bar inputs (:data:`PAST`) as one Darts series over the whole panel
    (built once)."""
    if "_pc" not in gi:
        from darts import TimeSeries

        n = len(gi["y"])
        gi["_pc"] = [TimeSeries.from_times_and_values(pd.RangeIndex(0, n),
                                                      np.nan_to_num(gi["past"][k]),
                                                      columns=list(PAST))
                     for k in range(len(gi["names"]))]
    return gi["_pc"]


def day_series(gi: dict, k: int, s: int, L: int, end: int | None = None):
    """Stock ``k``'s stock-day series of session ``s``: the targets of the previous session's last
    ``L - 1`` bars and of the session up to bar ``end`` (default: its last), on the global bar
    index (the past and future inputs share it), with the day's static inputs. Every window of
    ``L`` input bars then holds at least one of today's bars, so every predicted bar is today's.
    The whole day is built once (kept with the inputs); a shorter one is a slice of it."""
    if (k, s, L) not in gi["_day"]:
        from darts import TimeSeries

        a, b = gi["sf"][s] - (L - 1), gi["sf"][s] + gi["n_pos"]
        y = np.nan_to_num(gi["y"][a:b, k:k + 1]).astype(np.float32)
        gi["_day"][(k, s, L)] = TimeSeries.from_times_and_values(
            pd.RangeIndex(a, b), y, columns=["y"]).with_static_covariates(static(gi, k, s))
    full = gi["_day"][(k, s, L)]
    return full if end is None else full[:end - (gi["sf"][s] - (L - 1)) + 1]


def make_model(kind: str, L: int, H: int, *, epochs: int, n_samples: int, seed: int = 0,
               device: str = "cpu"):
    """The explorer's networks (``explorer.make_path_model``), DLinear with a moving-average
    kernel no longer than the input (the largest odd number up to min(25, ``L``))."""
    if kind != "dlinear":
        return ex.make_path_model(kind, L, H, epochs=epochs, n_samples=n_samples, seed=seed,
                                  device=device)
    import pytorch_lightning  # noqa: F401  (imported first so the quieting sticks)
    from darts.models import DLinearModel

    SM["_quiet"]()
    kernel = min(25, L) - (1 - min(25, L) % 2)
    trainer = {"accelerator": device, "enable_progress_bar": False, "enable_model_summary": False,
               "logger": False, "enable_checkpointing": False, "num_sanity_val_steps": 0,
               "callbacks": [SM["_early_stopping"](), ex._KeepBest()]}
    return DLinearModel(input_chunk_length=L, output_chunk_length=H, n_epochs=epochs,
                        batch_size=int(np.clip(n_samples // 40, 32, 512)), random_state=seed,
                        use_static_covariates=True, pl_trainer_kwargs=trainer,
                        optimizer_kwargs={"lr": SETTINGS["lr"]}, kernel_size=max(kernel, 1))


def fit(gi: dict, kind: str, L: int, H: int, train, held, *, epochs: int, seed: int = 0,
        device: str = "cpu") -> tuple:
    """ONE model for every stock, fit on the stock-day series of the ``train`` sessions; the
    ``held`` sessions (after them) stop training early (the weights of the best epoch are kept).
    A window is taken about every :data:`STRIDE_MINUTES` minutes of the clock at every bar size
    (Darts counts the stride back from the close); a bar without a trade weighs 0.

    Returns:
        ``(model, info)``; ``info``: ``stride``, ``samples`` (per epoch), ``epochs`` (run),
        ``best_val_loss``, ``seconds``, ``train`` / ``held`` (first and last date), ``sessions``.
    """
    from darts import TimeSeries

    pc = past_series(gi)
    w = gi["valid"].astype(np.float32)
    N = len(gi["names"])

    def part(sessions):
        ser, wts, fut, pas = [], [], [], []
        for s in sessions:
            a, b = gi["sf"][s] - (L - 1), gi["sf"][s] + gi["n_pos"]
            for k in range(N):
                ser.append(day_series(gi, k, s, L))
                wts.append(TimeSeries.from_times_and_values(pd.RangeIndex(a, b), w[a:b, k:k + 1],
                                                            columns=["w"]))
                fut.append(future(gi, s, L))
                pas.append(pc[k])
        return ser, wts, fut, pas

    series, weights, fut, pas = part(train)
    val, val_w, vfut, vpas = part(held)
    stride = max(1, round(STRIDE_MINUTES / gi["g"]))
    samples = len(series) * math.ceil((gi["n_pos"] - H) / stride)
    est = make_model(kind, L, H, epochs=epochs, n_samples=samples, seed=seed, device=device)
    kw = dict(future_covariates=fut, val_future_covariates=vfut) \
        if est.supports_future_covariates else {}
    t0 = time.time()
    est.fit(series, past_covariates=pas, sample_weight=weights, val_series=val,
            val_past_covariates=vpas, val_sample_weight=val_w, stride=stride, verbose=False, **kw)
    SM["_quiet"]()
    keep = next(c for c in est.trainer_params["callbacks"] if hasattr(c, "state"))
    if keep.state is not None:
        est.model.load_state_dict(keep.state)
    run = int(est.epochs_trained) if hasattr(est, "epochs_trained") else epochs
    d = gi["dates"]
    return est, {"stride": stride, "samples": samples, "epochs": run,
                 "best_val_loss": round(keep.best, 5), "seconds": round(time.time() - t0, 1),
                 "train": (str(d[train[0]]), str(d[train[-1]])),
                 "held": (str(d[held[0]]), str(d[held[-1]])), "sessions": len(train)}


def decision_bar(gi: dict, s, t):
    """The bar that closes at minute ``t`` of session ``s`` (its close is the decision)."""
    return gi["sf"][s] + np.asarray(t) // gi["g"] - 1


def predict(est, gi: dict, s, t, k, L: int, H: int, *, batch: int = 1024) -> np.ndarray:
    """The fitted model's next ``H`` bar moves (typical moves) from each decision (session ``s``,
    minute ``t``, stock ``k``), read from its last ``L`` bars only: the series and the bar inputs
    end at the decision bar, and Darts is asked for exactly ``output_chunk_length`` bars (more
    would read later inputs).

    Returns:
        A (decisions x H) array.
    """
    if est.output_chunk_length != H:
        raise ValueError(f"asked for {H} bars from a model built for {est.output_chunk_length}")
    if not len(s):
        return np.zeros((0, H), np.float32)
    b = decision_bar(gi, s, t)
    pc = past_series(gi)
    series = [day_series(gi, kk, ss, L, end=bb) for ss, bb, kk in zip(s, b, k, strict=True)]
    past = [pc[kk][bb - L + 1:bb + 1] for bb, kk in zip(b, k, strict=True)]
    kw = {"future_covariates": [future(gi, ss, L) for ss in s]} \
        if est.supports_future_covariates else {}
    out = est.predict(n=H, series=series, past_covariates=past, batch_size=batch, verbose=False,
                      **kw)
    return np.stack([p.values()[:, 0] for p in out]).astype(np.float32)


def checkpoint_moves(pred: np.ndarray, gi: dict, s, t, k, span: int) -> np.ndarray:
    """Predicted bar moves (typical moves of each bar) as the predicted LOG move from the
    decision to every checkpoint up to ``span`` minutes (bar ``j`` ends ``j * g`` minutes later;
    a checkpoint between two bar closes, as 10 minutes on 3-minute bars, is read on the straight
    line between them); NaN past ``span``.

    Returns:
        A (decisions x checkpoints) array.
    """
    H = pred.shape[1]
    b = decision_bar(gi, s, t)
    u = gi["u"][np.asarray(b)[:, None] + np.arange(1, H + 1), np.asarray(k)[:, None]]
    P = np.c_[np.zeros(len(pred)), np.cumsum(pred * u, axis=1)]
    x = np.array(CHECKPOINTS) / gi["g"]
    j0, j1 = np.floor(x).astype(int), np.ceil(x).astype(int)
    out = np.full((len(pred), len(CHECKPOINTS)), np.nan)
    on = (np.array(CHECKPOINTS) <= span) & (j1 <= H)
    out[:, on] = P[:, j0[on]] + (x[on] - j0[on]) * (P[:, j1[on]] - P[:, j0[on]])
    return out


# ---------------------------------------------------------------------------------------------
# Decisions, the scale and the grade.
# ---------------------------------------------------------------------------------------------
def session_index(dates, a: str, b: str) -> np.ndarray:
    """Usable sessions (:func:`usable`) from ``a`` to ``b`` (inclusive), as indices."""
    d = np.asarray(dates)
    return np.flatnonzero(usable(d) & (d >= a) & (d <= b))


def training_sessions(sp: dict, dates, part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The sessions a fit for ``part`` learns from, the :data:`ES_DAYS` after them that stop its
    training early, and the sessions its first scale is fit on. VALIDATION is graded by a fit on
    TRAIN (its first month's scale from those ES_DAYS sessions; later months add the graded months
    before them). TEST is graded by a refit on TRAIN and VALIDATION that holds out the last
    :data:`TEST_CAL_DAYS` sessions (at most a third of them) for its scale, the last ES_DAYS of
    them also stopping its training.

    Returns:
        ``(train, early-stopping, calibration)`` session indices.
    """
    last = sp["train"][1] if part == "validation" else sp["validation"][1]
    use = session_index(dates, sp["train"][0], last)
    cal = ES_DAYS if part == "validation" else max(ES_DAYS, min(TEST_CAL_DAYS, len(use) // 3))
    return use[:-cal], use[-ES_DAYS:], use[-cal:]


def rows_for(oc: dict, sessions, span: int) -> np.ndarray:
    """The decision rows graded on ``sessions`` for an output of ``span`` minutes: a fill, a
    usable session, and the whole span before 16:00."""
    return np.flatnonzero(oc["usable"] & np.isin(oc["s"], sessions)
                          & (oc["t"] + span <= SESSION))


def calibration(pred: np.ndarray, act: np.ndarray) -> float:
    """The scale ``beta`` (0 .. 1) that makes predicted paths agree best with the actual ones the
    way they are graded: each checkpoint's squared errors divided by its sum of squared actual
    moves (so long checkpoints do not dominate). 0 says the paths did not help at all ("no
    change"), 1 keeps them as they are; a network's paths come out too loud (it fits noise)."""
    p, a = np.nan_to_num(pred), np.nan_to_num(act)
    w = 1.0 / np.maximum((a * a).sum(axis=0), 1e-12)
    den = float(((p * p).sum(axis=0) * w).sum())
    return float(np.clip(((p * a).sum(axis=0) * w).sum() / den, 0.0, 1.0)) if den > 0 else 0.0


def chained_beta(p_held, a_held, p, a, month) -> tuple[np.ndarray, dict]:
    """The scale of every graded row, fit only on what was known before its month: the held-out
    sessions plus the graded months before it (the first month: the held-out sessions alone).

    Returns:
        ``(beta per row, {month: beta})``.
    """
    out, by = np.zeros(len(p)), {}
    for mo in sorted(set(month)):
        before = month < mo
        by[mo] = calibration(np.r_[p_held, p[before]], np.r_[a_held, a[before]])
        out[month == mo] = by[mo]
    return out, by


def grade(pred: np.ndarray, act: np.ndarray, k: np.ndarray, names, *, key=None, B=None,
          U=None, draws: int = 0, seed: int = 0) -> dict:
    """The grade of predicted paths against the actual ones at the checkpoints (both in typical
    moves; NaN columns are not graded).

    Per stock and checkpoint: R² = 1 - Σ (predicted - actual)² / Σ actual², the out-of-sample R²
    against the "no change" forecast; a stock's score is its mean over the checkpoints, the grade
    the mean over the stocks (equal weights). Direction: ``corr``, the correlation of predicted
    and actual change per stock and checkpoint, averaged (a scale cannot move it). Its luck band:
    the same predictions against ``draws`` make-believe futures in which every 10-minute block
    keeps its size but gets a random sign, one coin per clock block for all stocks and decisions
    (``key``, ``B``, ``U`` from :func:`outcomes`), drawn in mirror pairs (a coin and its opposite
    give opposite correlations, so luck's centre is exactly 0); ``z`` = luck-SDs above it.

    Returns:
        ``r2``, ``r2_cp`` (mean over stocks, per graded checkpoint), ``r2_by_name``,
        ``share_pos``, ``worst``, ``corr``, ``corr_cp``, ``n``, ``cps`` (graded checkpoint
        positions) and with ``draws`` ``null_mean`` / ``null_sd`` / ``z``.
    """
    on = np.flatnonzero(np.isfinite(act).all(axis=0) & np.isfinite(pred).all(axis=0))
    if not len(pred) or not len(on):
        return {"r2": np.nan, "r2_cp": np.full(len(CHECKPOINTS), np.nan), "corr": np.nan,
                "corr_cp": np.full(len(CHECKPOINTS), np.nan), "n": 0, "cps": on,
                "share_pos": np.nan, "worst": np.nan, "r2_by_name": pd.Series(dtype=float)}
    p, a = pred[:, on].astype(np.float64), act[:, on].astype(np.float64)
    r2, corr = ex._r2(p, a, k), ex._corr(p, a, k)
    by_name = pd.Series(np.nanmean(r2, axis=1), index=[names[x] for x in np.unique(k)])
    full = lambda v: np.r_[v, np.full(len(CHECKPOINTS) - len(v), np.nan)]  # noqa: E731
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {"r2": float(by_name.mean()), "r2_cp": full(np.nanmean(r2, axis=0)),
               "r2_by_name": by_name, "share_pos": float((by_name > 0).mean()),
               "worst": float(by_name.min()), "corr": float(np.nanmean(corr)),
               "corr_cp": full(np.nanmean(corr, axis=0)), "n": len(p), "cps": on}
    if draws and np.isfinite(out["corr"]):
        rng = np.random.default_rng(seed)
        width = on.max() + 1
        last = np.array(CHECKPOINTS)[on] // BLOCK - 1
        b = np.nan_to_num(B[:, :np.array(CHECKPOINTS)[width - 1] // BLOCK])
        kk = key[:, :b.shape[1]]
        null = []
        for d0 in range(0, max(1, draws // 2), 13):
            coin = rng.choice(np.array([-1.0, 1.0]),
                              size=(min(13, draws // 2 - d0), int(kk.max()) + 1))
            flip = np.cumsum(coin[:, kk] * b[None], axis=2)[:, :, last] / U[None][:, :, on]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                c = np.nanmean(ex._corr(p, flip, k), axis=(1, 2))
            null.extend(np.r_[c, -c])                   # each coin and its mirror image
        mu, sd = float(np.mean(null)), float(np.std(null))
        out.update(null_mean=mu, null_sd=sd, z=(out["corr"] - mu) / sd if sd > 0 else np.nan)
    return out


def by_time(pred: np.ndarray, act: np.ndarray, k: np.ndarray, t: np.ndarray, names) -> dict:
    """The R² grade of each decision time on its own: ``{minute: R²}``."""
    return {int(x): grade(pred[t == x], act[t == x], k[t == x], names)["r2"]
            for x in np.unique(t)}


def bootstrap(pred: np.ndarray, act: np.ndarray, s: np.ndarray, k: np.ndarray, *, other=None,
              n: int = 500, seed: int = 0) -> tuple[float, float]:
    """A 90% interval of the R² grade (or of its lead over ``other``'s predictions of the same
    decisions) by resampling whole sessions, every stock and decision of a session together."""
    on = np.isfinite(act).all(axis=0) & np.isfinite(pred).all(axis=0)
    days, d = np.unique(s, return_inverse=True)
    stocks, j = np.unique(k, return_inverse=True)

    def sums(p):
        e = np.zeros((len(days), len(stocks), on.sum()))
        b = np.zeros_like(e)
        np.add.at(e, (d, j), (p[:, on] - act[:, on]) ** 2)
        np.add.at(b, (d, j), act[:, on] ** 2)
        return e, b

    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(days), np.full(len(days), 1 / len(days)), size=n).astype(float)

    def r2(p):
        e, b = sums(p)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (1 - np.einsum("nd,dsc->nsc", w, e) / np.einsum("nd,dsc->nsc", w, b)) \
                .mean(axis=2).mean(axis=1)

    x = r2(pred) - (r2(other) if other is not None else 0.0)
    return float(np.nanpercentile(x, 5)), float(np.nanpercentile(x, 95))


def mde(null_sd: float, k: int = 1) -> float:
    """The smallest R² this grade detects 80% of the time: a correctly scaled forecast whose
    direction correlation clears the bar of the best of ``k`` tries (``explorer.best_of_bar``;
    1.96 for one) by 0.84 luck-SDs; R² ≈ correlation²."""
    return float(((ex.best_of_bar(k) + 0.84) * null_sd) ** 2)


# ---------------------------------------------------------------------------------------------
# Baselines: the time-of-day mean path and a ridge on summaries, scaled like the model.
# ---------------------------------------------------------------------------------------------
def ridge_features(pp: dict) -> tuple[np.ndarray, list[str]]:
    """Per decision row, a dozen summaries known at the decision, from the 1-minute data (the
    same for every bar size): the stock's own move over the last 1, 5, 15, 30 and 60 minutes and
    since the open, its distance to the session VWAP, the other stocks' average move over the
    last 1, 5 and 30 minutes, the last 30 minutes' volume against the same 30 minutes of the
    previous 20 sessions, whether the last minute traded (and that times the market's last
    move: a stale price catching up), today's gap and yesterday's move (in daily volatilities),
    and which decision time and which stock (one-hot). The gap counts only where the day's first
    trade came before the decision (a first trade inside the fill window would BE the fill).

    Returns:
        ``(rows x features, names)``.
    """
    m, sc, oc = pp["m"], pp["sc"], pp["oc"]
    s, t, k = oc["s"], oc["t"], oc["k"]
    cf, C, V = m["cf"], m["close"], m["volume"]
    n, _, N = cf.shape
    sig = sc["sigma"]
    o1, gap_raw = day_open_gap(m)
    traded = np.isfinite(m["open"])
    first = np.where(traded.any(axis=1), traded.argmax(axis=1), SESSION)   # sessions x stocks
    now = cf[s, t - 1, k]
    feats, names = [], []
    with np.errstate(invalid="ignore", divide="ignore"):
        for w in (1, 5, 15, 30, 60):
            back = np.where(t - 1 - w >= 0, cf[s, np.maximum(t - 1 - w, 0), k], o1[s, k])
            feats.append(np.log(now / back) / (sig[s, k] * np.sqrt(w)))
            names.append(f"ret_{w}m")
        feats.append(np.log(now / o1[s, k]) / (sig[s, k] * np.sqrt(t)))
        names.append("ret_open")
        pv = np.nancumsum(np.nan_to_num(C) * V, axis=1)
        vv = np.nancumsum(np.where(np.isfinite(C), V, 0.0), axis=1)
        feats.append(np.log(now / (pv[s, t - 1, k] / vv[s, t - 1, k])) / (sig[s, k] * np.sqrt(t)))
        names.append("vwap_dist")
        own = {w: f for w, f in zip((1, 5, 15, 30, 60), feats[:5], strict=True)}
        tab = {w: np.full((n, len(DECISION_MINUTES), N), np.nan) for w in (1, 5, 30)}
        ti = np.searchsorted(np.array(DECISION_MINUTES), t)
        for w in tab:
            tab[w][s, ti, k] = own[w]
        for w in (1, 5, 30):
            tot = np.nansum(tab[w], axis=2, keepdims=True)
            cnt = np.isfinite(tab[w]).sum(axis=2, keepdims=True)
            loo = (tot - np.nan_to_num(tab[w])) / np.maximum(cnt - np.isfinite(tab[w]), 1)
            feats.append(loo[s, ti, k])
            names.append(f"mkt_{w}m")
        v30 = np.add.reduceat(V, np.arange(0, SESSION, 30), axis=1)       # sessions x 13 x stocks
        base = pd.DataFrame(v30.reshape(n, -1)).rolling(20, min_periods=5).mean().shift(1) \
            .to_numpy().reshape(v30.shape)
        last30 = v30[s, t // 30 - 1, k]
        feats.append(np.log1p(last30) - np.log1p(base[s, t // 30 - 1, k]))
        names.append("rvol_30m")
        stale = (~np.isfinite(C[s, t - 1, k])).astype(float)
        feats += [stale, stale * feats[names.index("mkt_1m")]]
        names += ["no_trade_last_minute", "no_trade_x_mkt_1m"]
        close = cf[:, SESSION - 1]
        dret = np.log(close / np.r_[np.full((1, N), np.nan), close[:-1]])
        dvol = pd.DataFrame(dret).rolling(20, min_periods=10).std().shift(1).to_numpy()
        gap = np.where(first[s, k] < t, (gap_raw / dvol)[s, k], np.nan)
        prev = np.r_[np.full((1, N), np.nan), dret[:-1]] / dvol
        feats += [gap, prev[s, k]]
        names += ["gap", "prev_ret"]
    for x in DECISION_MINUTES:
        feats.append((t == x).astype(float))
        names.append(f"at_{x}")
    for i, x in enumerate(m["names"]):
        feats.append((k == i).astype(float))
        names.append(f"is_{x}")
    X = np.stack(feats, axis=1)
    return np.where(np.isfinite(X), X, 0.0), names


def baselines(panel, sp: dict, span: int, part: str = "validation") -> dict:
    """The two baselines of an output span, graded exactly like a model (same decisions, same
    checkpoints, the same month-by-month scale):

    - ``seasonal``: the average path from that decision time in typical moves, over every stock
      and the previous 20 usable sessions (the market's time-of-day pattern);
    - ``ridge``: one ridge regression per checkpoint on :func:`ridge_features`, fit on the same
      sessions as the model (TRAIN without its held-out week; for TEST, TRAIN and VALIDATION
      without the held-out stretch of :data:`TEST_CAL_DAYS` sessions), its
      penalty chosen by 5-fold cross-validation over whole sessions (a day's decisions share
      most of their outcome: rows are not independent, days nearly are).

    Returns:
        ``{name: {"pred" (scaled), "raw", "beta", "score"}}`` for the rows of
        :func:`rows_for` (kept in-process).
    """
    pp = prepared(panel)
    key = (pp["key"], span, part, json.dumps([sp["train"], sp["validation"], sp["test"]]))
    if key in _BASE:
        return _BASE[key]
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    oc, dates = pp["oc"], pp["m"]["dates"]
    train, _, cal = training_sessions(sp, dates, part)
    graded = session_index(dates, *sp[part])
    rh, rg = rows_for(oc, cal, span), rows_for(oc, graded, span)
    cps = np.array(CHECKPOINTS) <= span
    z = np.where(cps[None], oc["z"], np.nan)
    # the seasonal path: per (session, decision time), the stocks' mean, then the previous 20
    n, T = len(dates), len(DECISION_MINUTES)
    ti = np.searchsorted(np.array(DECISION_MINUTES), oc["t"])
    cube = np.full((n, T, len(oc["k"]) // (n * T), len(CHECKPOINTS)), np.nan)
    cube[oc["s"], ti, oc["k"]] = np.where(oc["usable"][:, None], z, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        day = np.nanmean(cube, axis=2).reshape(n, -1)
    seas = pd.DataFrame(day).rolling(20, min_periods=5).mean().shift(1).to_numpy() \
        .reshape(n, T, -1)[oc["s"], ti]
    # the ridge: fit on the training sessions' decisions, one checkpoint at a time
    X, _ = ridge_features(pp)
    fit_rows = rows_for(oc, train, 0)
    scaler = StandardScaler().fit(X[fit_rows])
    Xs = scaler.transform(X)
    ridge = np.full_like(z, np.nan)
    for c in np.flatnonzero(cps):
        r = fit_rows[np.isfinite(oc["z"][fit_rows, c])]
        days = len(np.unique(oc["s"][r]))
        folds = list(GroupKFold(min(5, days)).split(Xs[r], groups=oc["s"][r])) \
            if days >= 2 else None
        est = RidgeCV(alphas=RIDGE_ALPHAS, cv=folds).fit(Xs[r], oc["z"][r, c])
        ridge[:, c] = est.predict(Xs)
    out = {}
    for name, raw in (("seasonal", np.where(cps[None], seas, np.nan)), ("ridge", ridge)):
        beta, by = chained_beta(raw[rh][:, cps], z[rh][:, cps], raw[rg][:, cps], z[rg][:, cps],
                                oc["month"][rg])
        pred = beta[:, None] * raw[rg]
        out[name] = {"pred": pred, "raw": raw[rg], "beta": by,
                     "score": grade(pred, z[rg], oc["k"][rg], pp["m"]["names"])}
    _BASE[key] = out
    while len(_BASE) > 32:
        _BASE.pop(next(iter(_BASE)))
    return out


# ---------------------------------------------------------------------------------------------
# One configuration, the grid, the model comparison and the TEST read.
# ---------------------------------------------------------------------------------------------
def run(panel, sp: dict, cell, *, kind: str, epochs: int, draws: int = 200,
        part: str = "validation", seed: int = 0, device: str = "cpu") -> dict:
    """One configuration end to end: fit ONE model on the training sessions' stock-days
    (:func:`training_sessions`), predict the path from every graded decision of ``part`` (and of
    the sessions its first scale is fit on), turn it into moves at the checkpoints
    (:func:`checkpoint_moves`),
    scale it month by month (:func:`chained_beta`) and grade it (:func:`grade`) beside the
    unscaled paths (direction, luck band) and the baselines. Saved in the cache folder, keyed by
    the configuration, the split, the panel's content, the code and the module's constants (the
    name carries strategy.py's version, so ``research.prune_cache`` keeps what is reachable).

    Args:
        panel: The 1-minute panel (``explorer.load_window``; prepared here).
        sp: Its split.
        cell: ``(bar minutes, input minutes, output minutes)``.
        kind: The Darts model (:data:`MODELS`).
        epochs: At most this many training passes.
        draws: Luck-band futures (0: none).
        part: ``"validation"`` or ``"test"``.
        seed: Model and luck seed.
        device: ``cpu`` | ``mps``.

    Returns:
        ``cell``, ``kind``, ``part``, ``bars`` (input, output), ``rows`` (graded rows of the
        truth table), ``pred`` / ``raw`` (scaled / unscaled, typical moves at the checkpoints),
        ``beta`` ({month: scale}), ``score``, ``score_raw`` (with the luck band), ``r2_time``
        ({decision minute: R²}), ``base`` / ``base_cp`` / ``base_corr`` (each baseline's R², its R²
        per checkpoint, its direction), ``fit`` (its info plus ``first_graded``).
    """
    g, Lm, Sm = (int(x) for x in cell)
    if Lm % g:
        raise ValueError(f"an input of {Lm} minutes is not a whole number of {g}-minute bars")
    L, H = Lm // g, math.ceil(Sm / g)
    ident = json.dumps({"cell": [g, Lm, Sm], "kind": kind, "epochs": epochs, "draws": draws,
                        "part": part, "seed": seed, "device": device,
                        "split": [sp["train"], sp["validation"], sp["test"]],
                        "panel": SM["fingerprint"](panel),
                        "code": [rs.CODE_VERSION, ex.EXPLORER_VERSION, ex.RESEARCH_VERSION,
                                 INTRADAY_VERSION],
                        "constants": [DECISION_MINUTES, CHECKPOINTS, BLOCK, FILL_WITHIN,
                                      STRIDE_MINUTES, ES_DAYS, TEST_CAL_DAYS, CLIP, SETTINGS]},
                       sort_keys=True, default=str)
    path = rs.CACHE / (f"intraday-{kind}-{g}m-in{Lm}-out{Sm}-{part}-c{rs.CODE_VERSION[:6]}-"
                       f"{hashlib.sha1(ident.encode()).hexdigest()[:12]}.pkl")   # prune_cache keeps
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    pp = prepared(panel)
    oc, names, dates = pp["oc"], pp["m"]["names"], pp["m"]["dates"]
    gi = grid_inputs(pp, g)
    gi["_day"].clear()                 # one configuration's stock-day series at a time (memory)
    train, held, cal = training_sessions(sp, dates, part)
    est, info = fit(gi, kind, L, H, train, held, epochs=epochs, seed=seed, device=device)
    rh, rg = rows_for(oc, cal, Sm), rows_for(oc, session_index(dates, *sp[part]), Sm)

    def moves(r):
        p = predict(est, gi, oc["s"][r], oc["t"][r], oc["k"][r], L, H)
        return checkpoint_moves(p, gi, oc["s"][r], oc["t"][r], oc["k"][r], Sm) / oc["U"][r]

    raw_h, raw = moves(rh), moves(rg)
    act = np.where(np.isfinite(raw), oc["z"][rg], np.nan)
    cps = np.array(CHECKPOINTS) <= Sm
    beta, by = chained_beta(raw_h[:, cps], oc["z"][rh][:, cps], raw[:, cps], act[:, cps],
                            oc["month"][rg])
    pred = beta[:, None] * raw
    k = oc["k"][rg]
    base = baselines(panel, sp, Sm, part)
    out = {"cell": (g, Lm, Sm), "kind": kind, "part": part, "bars": (L, H), "rows": rg,
           "pred": pred.astype(np.float32), "raw": raw.astype(np.float32), "beta": by,
           "score": grade(pred, act, k, names),
           "score_raw": grade(raw, act, k, names, key=oc["key"][rg], B=oc["B"][rg],
                              U=oc["U"][rg], draws=draws, seed=seed),
           "r2_time": by_time(pred, act, k, oc["t"][rg], names),
           "base": {b: v["score"]["r2"] for b, v in base.items()},
           "base_cp": {b: v["score"]["r2_cp"] for b, v in base.items()},
           "base_corr": {b: v["score"]["corr"] for b, v in base.items()},
           "fit": {**info, "first_graded": str(dates[oc["s"][rg].min()]) if len(rg) else ""}}
    rs.CACHE.mkdir(exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh)
    os.replace(tmp, path)
    return out


def train_paths(sessions: int, span: int) -> float:
    """Independent training paths per stock: how many non-overlapping ``span``-minute paths the
    training sessions hold (finer bars add overlapping windows, not more of these)."""
    return sessions * max(1, (SESSION - BLOCK) // span)


def _row(r: dict) -> dict:
    """One configuration's numbers (a search / table row)."""
    g, Lm, Sm = r["cell"]
    sc, raw = r["score"], r["score_raw"]
    best_base = max(r["base"].values(), key=lambda v: v if np.isfinite(v) else -np.inf)
    return {"granularity": g, "input": Lm, "prediction": Sm, "input_bars": r["bars"][0],
            "output_bars": r["bars"][1], "r2": sc["r2"],
            "beta": float(np.mean(list(r["beta"].values()))) if r["beta"] else np.nan,
            "beta_first": next(iter(r["beta"].values()), np.nan), "r2_raw": raw["r2"],
            "corr": raw["corr"], "z": raw.get("z", np.nan),
            "null_mean": raw.get("null_mean", np.nan), "null_sd": raw.get("null_sd", np.nan),
            "share_pos": sc["share_pos"], "worst": sc["worst"],
            "ridge_r2": r["base"].get("ridge", np.nan),
            "seasonal_r2": r["base"].get("seasonal", np.nan),
            "vs_best_baseline": sc["r2"] - best_base, "paths": raw["n"],
            "train_paths": train_paths(r["fit"]["sessions"], Sm), "r2_cp": sc["r2_cp"],
            "corr_cp": raw["corr_cp"], "ridge_cp": r["base_cp"]["ridge"],
            "seasonal_cp": r["base_cp"]["seasonal"], "r2_time": r["r2_time"],
            "epochs": r["fit"]["epochs"],
            "stride": r["fit"]["stride"], "fit_s": r["fit"]["seconds"]}


def search(panel, sp: dict, *, kind: str, grans, inputs, outputs, epochs: int,
           draws: int = 200, seed: int = 0, device: str = "cpu", n_trials: int | None = None,
           verbose: bool = True) -> tuple[pd.DataFrame, object]:
    """Every configuration of the grid (bar size x input span x output span) measured on
    VALIDATION, through Optuna's grid sampler: a random order, so a run cut short is still a
    balanced sample; ``n_trials`` caps it. The grid is small, and measuring all of it gives each
    knob's effect on balanced data and an honest bar for the best of all the tries. TEST is never
    read here.

    Returns:
        ``(results, study)``: one row per configuration (``trial``, :func:`_row`'s numbers,
        ``seconds``) and the Optuna study.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = {"granularity": list(grans), "input": list(inputs), "prediction": list(outputs)}
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GridSampler(space, seed=seed))
    total = math.prod(len(v) for v in space.values())
    goal = min(n_trials or total, total)
    rows = []

    def objective(tr) -> float:
        cell = tuple(tr.suggest_categorical(x, v) for x, v in space.items())
        t0 = time.time()
        r = run(panel, sp, cell, kind=kind, epochs=epochs, draws=draws, seed=seed, device=device)
        row = {"trial": len(rows) + 1, **_row(r), "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        if verbose:
            print(f"{len(rows):>3}/{goal}  {cell[0]:>2}-min bars  input {cell[1]:>3} min  predict "
                  f"{cell[2]:>3} min  R² {row['r2']:+.4f} (scale {row['beta']:.2f}; ridge "
                  f"{row['ridge_r2']:+.4f}; direction z {row['z']:+.1f})  "
                  f"{row['seconds']:.0f} s")
        return float(row["r2"])

    study.optimize(objective, n_trials=goal)       # the grid sampler stops when the grid is done
    return pd.DataFrame(rows), study


def best(res: pd.DataFrame, objective: str = "r2", k: int = 1) -> pd.DataFrame:
    """The ``k`` best configurations by ``objective``, ties broken by direction (``z``)."""
    return res.sort_values([objective, "z"], ascending=False, na_position="last").head(k)


def best_per(res: pd.DataFrame, key: str) -> pd.DataFrame:
    """The best configuration of each value of ``key`` (the ranking of :func:`best`)."""
    return best(res, k=len(res)).drop_duplicates(key).sort_values(key)


def compare(panel, sp: dict, models, cell, *, epochs: int, draws: int = 200, seed: int = 0,
            verbose: bool = True) -> pd.DataFrame:
    """Every model of ``models`` on ONE configuration, graded on VALIDATION.

    Returns:
        One row per model (:func:`_row`'s numbers).
    """
    rows = []
    for mdl in models:
        t0 = time.time()
        r = run(panel, sp, cell, kind=mdl, epochs=epochs, draws=draws, seed=seed)
        rows.append({"model": mdl, **_row(r)})
        if verbose:
            print(f"{mdl:>8}: R² {r['score']['r2']:+.4f}  ({time.time() - t0:.0f} s)")
    return pd.DataFrame(rows).set_index("model")


def importance(study) -> pd.Series:
    """How much each knob moves the grade (Optuna's fANOVA over the completed trials)."""
    return ex.importance(study).rename(lambda x: x.replace("length", "span"))


# ---------------------------------------------------------------------------------------------
# The positive control and the map of the day.
# ---------------------------------------------------------------------------------------------
def _known_drift(m: dict, oc: dict) -> np.ndarray:
    """Per decision row and checkpoint, the planted drift (per unit of ``kappa``, log units) that
    the moves before the decision already fix: minute ``t + i`` drifts by 1/30 of the move over
    its previous 30 minutes, of which the part before ``t`` is known (none from ``t + 30`` on)."""
    r = np.nan_to_num(m["r1"])
    cs = np.concatenate([np.zeros((r.shape[0], 1, r.shape[2])), np.cumsum(r, axis=1)], axis=1)
    s, t, k = oc["s"], oc["t"], oc["k"]
    lo = np.maximum(t[:, None] + np.arange(30) - 30, 0)
    known = np.cumsum(cs[s, t, k][:, None] - cs[s[:, None], lo, k[:, None]], axis=1) / 30.0
    return known[:, np.minimum(np.array(CHECKPOINTS), 30) - 1]


def planted(panel, sp: dict, rho: float) -> tuple:
    """The panel with a known intraday effect planted, the kind a model can only see in its bar
    inputs: every minute, each stock drifts by ``kappa / 30`` times its own move over the previous
    30 minutes (since the open, early in the session), a 30-minute momentum. ``kappa`` is set per
    stock so that, on TRAIN, the part of the next 30 minutes' drift that the past already fixes
    correlates ``rho`` with the 30-minute move (R² ≈ ``rho``² there, less further out: the drift
    fades after 30 minutes). The drift accumulates into every later price, so the overnight gaps do
    not change.

    Returns:
        ``(planted panel, {"rho", "kappa" (per stock), "known"})``: ``known`` is the fixed part of
        each decision's drift per unit ``kappa`` (:func:`oracle` grades it).
    """
    panel = full_clock(panel)
    pp = prepared(panel)
    m, oc = pp["m"], pp["oc"]
    n, _, N = m["r1"].shape
    known = _known_drift(m, oc)
    tr = rows_for(oc, session_index(m["dates"], *sp["train"]), 30)
    c30 = CHECKPOINTS.index(30)
    kappa = np.zeros(N)
    for k in range(N):
        r = tr[oc["k"][tr] == k]
        d = known[r, c30] / oc["U"][r, c30]
        if np.std(d) > 0:
            kappa[k] = rho / math.sqrt(1 - rho ** 2) * np.nanstd(oc["z"][r, c30]) / np.std(d)
    r1 = np.nan_to_num(m["r1"])
    cs = np.concatenate([np.zeros((n, 1, N)), np.cumsum(r1, axis=1)], axis=1)
    minute = np.arange(SESSION)
    move30 = cs[:, minute] - cs[:, np.maximum(minute - 30, 0)]          # before each minute
    reg = SM["regular_rows"](panel).reshape(n, SESSION)
    inc = kappa[None, None, :] * move30 / 30.0 * reg[..., None]
    factor = np.exp(np.cumsum(inc.reshape(n * SESSION, N), axis=0))
    out = {f: getattr(panel, f) * factor for f in _OHLC}
    return replace(panel, **out), {"rho": rho, "kappa": dict(zip(m["names"], kappa, strict=True)),
                                   "known": known}


def oracle(panel_planted, sp: dict, info: dict, span: int) -> float:
    """The R² grade (VALIDATION) of the planted drift's known part itself: the most a model could
    recover of what was planted (it reads planted prices, so a little less)."""
    pp = prepared(panel_planted)
    m, oc = pp["m"], pp["oc"]
    kap = np.array([info["kappa"][x] for x in m["names"]])
    rg = rows_for(oc, session_index(m["dates"], *sp["validation"]), span)
    on = np.array(CHECKPOINTS) <= span
    pred = np.where(on[None], kap[oc["k"][rg]][:, None] * info["known"][rg] / oc["U"][rg],
                    np.nan)
    return grade(pred, np.where(on[None], oc["z"][rg], np.nan), oc["k"][rg], m["names"])["r2"]


def atlas(panel, sp: dict, *, parts=("train", "validation"), draws: int = 200,
          seed: int = 0) -> dict:
    """Which part of the day predicts which, without a model: every session's night (the gap)
    and its thirteen half-hours (the first from the open), each stock's moves divided by their
    own spread, correlated across all stock-days of ``parts``. The luck band flips the sign of
    the later part, one coin per day for every stock, in mirror pairs: the size of every move,
    the day's volatility and the stocks' co-movement stay, only the direction link goes (a
    shuffle of days would also break each day's shared volatility, which narrows the band: in
    pure noise with day-to-day volatility it put 10-22% of cells beyond 1.96). Also: the first
    ``x`` minutes against the rest of the day (``x = 0``: the night alone against the session).

    Returns:
        ``labels``, ``corr`` (14 x 14), ``z`` (14 x 14: luck-SDs above luck's centre, 0),
        ``first`` (a table: ``x``, ``corr``, ``z``), ``days``, ``stocks``.
    """
    pp = prepared(panel)
    m = pp["m"]
    cf = m["cf"]
    N = cf.shape[2]
    use = np.concatenate([session_index(m["dates"], *sp[p]) for p in parts])
    o1, night = day_open_gap(m)
    ends = cf[:, 29::30]                                               # 13 half-hour closes
    with np.errstate(invalid="ignore", divide="ignore"):
        half = np.diff(np.log(np.concatenate([o1[:, None], ends], axis=1)), axis=1)
        X = np.concatenate([night[:, None], half], axis=1)[use]       # days x 14 x stocks
        X = X / np.sqrt(np.nanmean(X ** 2, axis=0, keepdims=True))
        own = lambda v: v / np.sqrt(np.nanmean(v ** 2, axis=0, keepdims=True))  # noqa: E731
        firsts = {0: own(night[use])}
        rest = {0: own(np.log(cf[use, SESSION - 1] / o1[use]))}
        for x in (30, 60, 120, 180):
            firsts[x] = own(np.log(cf[use, x - 1] / o1[use]))
            rest[x] = own(np.log(cf[use, SESSION - 1] / cf[use, x - 1]))

    def corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            return np.nan
        a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
        den = math.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else np.nan

    def matrix(Y):
        return np.array([[corr(X[:, i], Y[:, j]) for j in range(14)] for i in range(14)])

    rng = np.random.default_rng(seed)
    obs = matrix(X)
    first_obs = {x: corr(firsts[x], rest[x]) for x in firsts}
    null, null_first = [], {x: [] for x in firsts}
    for _ in range(max(1, draws // 2)):
        coin = rng.choice(np.array([-1.0, 1.0]), size=(len(use), 1))   # one per day, every stock
        c = matrix(X * coin[:, :, None])
        null += [c, -c]                                   # each coin and its mirror image
        for x in firsts:
            v = corr(firsts[x], rest[x] * coin)
            null_first[x] += [v, -v]
    null = np.array(null)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (obs - null.mean(axis=0)) / null.std(axis=0)
    labels = ["night"] + [f"{(570 + 30 * i) // 60}:{(570 + 30 * i) % 60:02d}" for i in range(13)]
    first = pd.DataFrame([{"first minutes": x, "corr with the rest of the day": first_obs[x],
                           "z": (first_obs[x] - np.mean(null_first[x])) / np.std(null_first[x])
                           if np.std(null_first[x]) > 0 else np.nan}
                          for x in firsts]).set_index("first minutes")
    return {"labels": labels, "corr": obs, "z": z, "first": first, "days": len(use),
            "stocks": N}


# ---------------------------------------------------------------------------------------------
# Checks.
# ---------------------------------------------------------------------------------------------
def series_checks(panel, sp: dict, cell) -> list[tuple]:
    """The stock-day series of one configuration as Darts will read them, checked against their
    definition: every training series is the previous session's last bars plus one whole
    session, on consecutive global bars; its "today" flag is 1 exactly on that session's bars;
    the clock runs from 0 at 09:30; no series touches a half-day or the day after one.

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    g, Lm, _ = cell
    L = Lm // g
    pp = prepared(panel)
    gi = grid_inputs(pp, g)
    dates = pp["m"]["dates"]
    train, held, _ = training_sessions(sp, dates, "validation")
    bad_len = bad_flag = bad_prev = 0
    for s in np.r_[train, held]:
        ser, fut = day_series(gi, 0, s, L), future(gi, s, L)
        idx = ser.time_index
        flag = fut.values()[:, CLOCK.index("today")]
        bad_len += len(ser) != L - 1 + gi["n_pos"] or not (np.diff(idx) == 1).all() \
            or not fut.time_index.equals(idx)
        bad_flag += not ((flag == 1) == (np.asarray(idx) >= gi["sf"][s])).all()
        prev = np.asarray(idx)[np.asarray(idx) < gi["sf"][s]] // gi["n_pos"]
        bad_prev += not (prev == s - 1).all()
    clock0 = fut.values()[L - 1, CLOCK.index("tod_frac")]
    ok = usable(dates)
    excluded = [str(d) for i, d in enumerate(dates)
                if not ok[i] and sp["train"][0] <= d <= sp["test"][1]]
    used = np.r_[train, held, *(session_index(dates, *sp[x]) for x in ("validation", "test"))]
    half = set(SM["HALF_DAYS"])
    bad = [str(dates[i]) for i in used if dates[i] in half or (i > 0 and dates[i - 1] in half)]
    return [
        ("every stock-day series is the previous session's last bars + one whole session",
         bad_len == 0, f"{len(train) + len(held)} sessions checked ({L - 1} + {gi['n_pos']} "
         f"bars of {g} minutes each)"),
        ("the 'today' flag is 1 exactly on the session's own bars, the clock 0 at 09:30",
         bad_flag == 0 and abs(clock0) < 1e-6,
         f"time of day at the session's first bar: {clock0:.3f}"),
        ("the input's earlier part is the previous session, never two sessions back",
         bad_prev == 0, f"input of {Lm} minutes ≤ one session of 390"),
        ("no session trained on or graded is a half-day or the session after one", not bad,
         (f"used: {', '.join(bad)}" if bad else
          f"{len(used)} sessions; left out: {', '.join(excluded) or 'none in the window'}")),
    ]


def gap_invariance(panel, g: int, *, seed: int = 0) -> list[tuple]:
    """No night inside any target: every session's prices are multiplied by its own random factor
    (a different overnight gap every night, nothing within a day changes). Every target, graded
    move, scale and bar-by-bar input must stay the same; only the day's inputs (the gap,
    yesterday's move) may change, and the gap must (else the check could not fail).

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    panel = full_clock(panel)
    rng = np.random.default_rng(seed)
    n, N = len(panel.close) // SESSION, len(panel.symbols)
    f = np.repeat(np.exp(rng.normal(0.0, 0.05, (n, N))), SESSION, axis=0)
    moved = replace(panel, **{k: getattr(panel, k) * f for k in _OHLC})
    a, b = prepared(panel), prepared(moved)
    ga, gb = grid_inputs(a, g), grid_inputs(b, g)
    same = lambda x, y: bool(np.allclose(x, y, rtol=1e-6, atol=1e-6, equal_nan=True))  # noqa: E731
    past_same = all(same(ga["past"][k], gb["past"][k]) for k in range(N))
    gap_i = DAILY.index("gap")
    moved_gap = not same(ga["daily"][..., gap_i], gb["daily"][..., gap_i])
    return [
        ("the graded truth, the targets and the scales ignore the night",
         same(a["oc"]["z"], b["oc"]["z"]) and same(ga["y"], gb["y"])
         and same(a["sc"]["sigma"], b["sc"]["sigma"]),
         f"every night moved by a random ±5% ({g}-minute bars)"),
        ("every bar-by-bar input ignores the night", past_same, ", ".join(PAST)),
        ("the gap input did move (the check can fail)", moved_gap, "daily input 'gap'"),
    ]


def truncation(panel, sp: dict, cell, *, kind: str, epochs: int, cut_session: int = 10,
               seed: int = 0) -> list[tuple]:
    """Nothing reads the future: every 1-minute bar after a cut (12:30 of the ``cut_session``-th
    usable VALIDATION session) is replaced by random bars and the WHOLE configuration re-run
    (inputs, scales, fit, baselines, predictions). Every path the network or a baseline predicted
    at or before the cut must be identical, and so must the first month's scale; the later paths
    must change (else the check could not fail).

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    ck = _sibling("checks")
    panel = full_clock(panel)
    base = run(panel, sp, cell, kind=kind, epochs=epochs, draws=0, seed=seed)
    pp = prepared(panel)
    dates = pp["m"]["dates"]
    s_cut = session_index(dates, *sp["validation"])[cut_session]
    rows = np.zeros(len(panel.close), bool)
    rows[s_cut * SESSION + 180:] = True                    # from the 12:30 minute on
    alt_panel = ck.scrambled(panel, rows, seed)
    alt = run(alt_panel, sp, cell, kind=kind, epochs=epochs, draws=0, seed=seed)
    oc = pp["oc"]
    both = np.intersect1d(base["rows"], alt["rows"])
    ib, ia = (np.searchsorted(r["rows"], both) for r in (base, alt))
    s, t = oc["s"][both], oc["t"][both]
    early = (s < s_cut) | ((s == s_cut) & (t <= 180))
    same = np.array_equal(base["raw"][ib[early]], alt["raw"][ia[early]], equal_nan=True)
    moved = not np.array_equal(base["raw"][ib[~early]], alt["raw"][ia[~early]], equal_nan=True)
    first = next(iter(base["beta"]))
    bb, ba = (baselines(p, sp, cell[2]) for p in (panel, alt_panel))
    base_same = all(np.array_equal(bb[x]["raw"][ib[early]], ba[x]["raw"][ia[early]],
                                   equal_nan=True) for x in bb)
    return [("paths predicted before the cut are unchanged (network and baselines)",
             bool(early.any()) and same and base_same
             and base["beta"][first] == alt["beta"][first],
             f"cut after {dates[s_cut]} 12:30; {int(early.sum()):,} paths compared; first "
             f"month's scale {base['beta'][first]:.3f}"),
            ("paths predicted after the cut did change (the check can fail)", moved,
             f"{int((~early).sum()):,} later paths")]


def run_checks(r: dict, sp: dict, panel) -> list[tuple]:
    """The boundaries of one result, as built: training sessions end before the held-out ones,
    those before the first graded decision; every graded decision sits at a decision time and its
    whole span ends by 16:00 inside its part.

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    pp = prepared(panel)
    oc, dates = pp["oc"], pp["m"]["dates"]
    part = sp[r["part"]]
    s, t = oc["s"][r["rows"]], oc["t"][r["rows"]]
    f = r["fit"]
    d = dates[s]
    return [
        ("training sessions end before the held-out ones, those before the graded part",
         f["train"][1] < f["held"][0] <= f["held"][1] < part[0] <= f["first_graded"],
         f"training {f['train'][0]} → {f['train'][1]}; held out {f['held'][0]} → "
         f"{f['held'][1]}; first graded {f['first_graded']}"),
        ("every graded path starts at a decision time and ends by 16:00 of its own day",
         bool(np.isin(t, DECISION_MINUTES).all() and (t + r["cell"][2] <= SESSION).all()),
         f"{len(s):,} paths from {len(np.unique(t))} decision times"),
        ("every graded path lies inside its part", bool((d >= part[0]).all()
                                                        and (d <= part[1]).all()),
         f"{r['part'].upper()} {part[0]} → {part[1]}"),
    ]


# ---------------------------------------------------------------------------------------------
# Tables, words, pictures' data.
# ---------------------------------------------------------------------------------------------
def placeholder_table(panel) -> pd.DataFrame:
    """Per stock, the share of minutes stored as a volume-0 copy of the previous close (masked as
    no trade, :func:`prepare`)."""
    reg = SM["regular_rows"](panel)
    v = panel.volume[reg]
    return pd.DataFrame({"no-trade minutes %": (100 * (v == 0).mean()).round(1)}) \
        .rename_axis("stock")


def profile(panel, sp: dict) -> pd.DataFrame:
    """The day's shape on TRAIN: per minute of the session, the typical 1-minute move relative to
    the day's average (the square root of :func:`scales`' profile, averaged over TRAIN)."""
    pp = prepared(panel)
    tr = session_index(pp["m"]["dates"], *sp["train"])
    clock = pd.date_range("2000-01-01 09:31", periods=SESSION, freq="1min").strftime("%H:%M")
    return pd.DataFrame({"relative typical move": np.sqrt(pp["sc"]["prof2"][tr].mean(axis=0))},
                        index=pd.Index(clock, name="minute ending"))


def move_table(panel, sp: dict, round_trip_bp: float) -> pd.DataFrame:
    """What a move at each checkpoint is worth (TRAIN, the median stock): its typical size in
    basis points, and the R² a forecast needs before trading its sign at every decision pays
    ``round_trip_bp`` (a correlation ρ earns ρ · σ · √(2/π) per trade; R² ≈ ρ²)."""
    pp = prepared(panel)
    oc = pp["oc"]
    tr = rows_for(oc, session_index(pp["m"]["dates"], *sp["train"]), 0)
    rows = []
    for i, c in enumerate(CHECKPOINTS):
        a = oc["A"][tr, i]
        sd = pd.Series(a).groupby(oc["k"][tr]).std().median() * 1e4
        rho = round_trip_bp / (sd * math.sqrt(2 / math.pi))
        rows.append({"minutes after the decision": c, "typical move (bp)": round(sd, 1),
                     "R² to pay the round trip": round(min(rho, 1.0) ** 2, 4)})
    return pd.DataFrame(rows).set_index("minutes after the decision")


def bounce_table(panel, sp: dict, grans) -> pd.DataFrame:
    """Why the grade starts at the fill: per bar size, the lag-1 autocorrelation of intraday bar
    moves on TRAIN (negative: the bid-ask bounce and stale prints that the next trade
    corrects; the decision bar's last trade is not a price anyone can still trade at)."""
    pp = prepared(panel)
    tr = session_index(pp["m"]["dates"], *sp["train"])
    rows = []
    for g in grans:
        gi = grid_inputs(pp, g)
        y = gi["y"].reshape(len(pp["m"]["dates"]), gi["n_pos"], -1)[tr]
        a, b = y[:, :-1].reshape(-1, y.shape[2]), y[:, 1:].reshape(-1, y.shape[2])
        ac = np.array([np.corrcoef(a[ok, j], b[ok, j])[0, 1] for j in range(y.shape[2])
                       for ok in [np.isfinite(a[:, j]) & np.isfinite(b[:, j])]])
        rows.append({"bar size (min)": g, "lag-1 autocorrelation (mean)": round(ac.mean(), 3),
                     "stocks negative": f"{int((ac < 0).sum())} of {len(ac)}"})
    return pd.DataFrame(rows).set_index("bar size (min)")


def inputs_table(L: int, g: int) -> pd.DataFrame:
    """Every input of one forecast: kind (bar by bar, known in advance, fixed for the day), what
    it is, when it is known, and how much of it the model reads."""
    rows = [{"input": "y (the target)", "kind": "the series itself",
             "what it is": "the bar's intraday move in typical moves (no night inside)",
             "known at": "the bar's close", "read": f"{L} bars ({L * g} min)"}]
    for x in PAST:
        what, known, _ = ex.FEATURES.get(x, (x, "the bar's close", ""))
        rows.append({"input": x, "kind": "bar by bar (past)", "what it is": what,
                     "known at": known, "read": f"{L} bars"})
    for x in CLOCK:
        rows.append({"input": x, "kind": "known in advance (future)", "what it is": WORDS[x],
                     "known at": "in advance", "read": f"{L} bars + the predicted bars"})
    for x in DAILY:
        what, known, _ = ex.FEATURES.get(x, (x, "the open", ""))
        rows.append({"input": x, "kind": "fixed for the day (static)", "what it is": what,
                     "known at": known, "read": "once"})
    for x in ("dow", "days_since", "opex_week"):
        rows.append({"input": x, "kind": "fixed for the day (static)", "what it is": WORDS[x],
                     "known at": "in advance", "read": "once"})
    rows.append({"input": "stock", "kind": "fixed (static)", "what it is": "which stock "
                 "(one-hot)", "known at": "always", "read": "once"})
    return pd.DataFrame(rows).set_index("input")


def excluded_table(sp: dict, dates) -> pd.DataFrame:
    """The sessions inside the window that are left out, and why."""
    d = list(dates)
    rows = []
    for i, x in enumerate(d):
        if sp["train"][0] <= x <= sp["test"][1] and not usable(d)[i]:
            why = "half-day (closed 13:00)" if x in SM["HALF_DAYS"] else \
                "the session after a half-day"
            rows.append({"session": x, "why": why})
    return pd.DataFrame(rows, columns=["session", "why"]).set_index("session")


def results_table(res: pd.DataFrame, k_family: int) -> pd.DataFrame:
    """A search's results, best first, in readable units; the smallest detectable R² is for one
    pre-registered test and for the best of the ``k_family`` configurations read."""
    x = best(res, k=len(res))
    return pd.DataFrame({
        "bar size (min)": x["granularity"], "input (min)": x["input"],
        "predict (min)": x["prediction"], "bars in / out": x["input_bars"].astype(str) + " / "
        + x["output_bars"].astype(str),
        "R² vs no change": x["r2"].round(4), "scale β": x["beta"].round(2),
        "stocks R² > 0": (100 * x["share_pos"]).round(0).astype(int).astype(str) + "%",
        "ridge R²": x["ridge_r2"].round(4), "time-of-day R²": x["seasonal_r2"].round(4),
        "R² unscaled": x["r2_raw"].round(4), "direction corr": x["corr"].round(3),
        "direction z": x["z"].round(1),
        "detectable R² (one test)": x["null_sd"].map(lambda v: round(mde(v), 4)),
        f"detectable R² (best of {k_family})": x["null_sd"].map(lambda v: round(mde(v, k_family),
                                                                                4)),
        "training paths / stock": x["train_paths"].round(0).astype(int),
        "seconds": x["seconds"]}).reset_index(drop=True).rename_axis("rank") \
        .rename(index=lambda i: i + 1)


def search_words(res: pd.DataFrame, k_family: int) -> str:
    """The search in plain words: the winner, how many configurations beat "no change", the
    baselines and luck, whether the winner clears the bar for the best of all the tries."""
    top = best(res).iloc[0]
    bar = ex.best_of_bar(k_family)
    pos, clear = int((res["r2"] > 0).sum()), int((res["z"] >= 1.96).sum())
    beat = int(((res["r2"] > 0) & (res["vs_best_baseline"] > 0)).sum())
    zero = int((res["beta"] == 0).sum())
    lines = [] if top["r2"] > 0 else [
        "- **No configuration's paths agree with the actual ones better than \"no change\"** "
        f"(the best R² is {top['r2']:+.4f}); {zero} of {len(res)} were scaled to \"no change\" "
        "outright."]
    lines += [
        f"- **Winner on VALIDATION:** {top['granularity']:.0f}-minute bars, input "
        f"{top['input']:.0f} minutes, predict {top['prediction']:.0f} minutes: R² "
        f"{top['r2']:+.4f} against \"no change\" ({100 * top['share_pos']:.0f}% of the stocks "
        f"above 0, scale {top['beta']:.2f}); ridge {top['ridge_r2']:+.4f}, time-of-day mean "
        f"{top['seasonal_r2']:+.4f}; direction corr {top['corr']:+.3f}, z {top['z']:+.1f}.",
        f"- **Better than \"no change\":** {pos} of {len(res)}; **better than \"no change\" "
        f"and both baselines:** {beat}; **direction above luck (z ≥ 1.96):** {clear}.",
        f"- **The bar for the best of {k_family} tries:** z {bar:.1f}. The best direction z "
        f"({res['z'].max():+.1f}) is " + ("**above it**." if res["z"].max() >= bar else
                                          "**below it**: what the best of this many tries looks "
                                          "like by luck alone."),
        f"- **What this data could see:** the winner's R² would have to reach "
        f"{mde(top['null_sd']):.4f} to be detected as one pre-registered test, "
        f"{mde(top['null_sd'], k_family):.4f} as the best of {k_family}.",
    ]
    return "\n".join(lines)


def test_words(r: dict, label: str) -> str:
    """The TEST read in plain words: whether it held (TEST R² above 0 and direction z ≥ 1.96),
    with the R² TEST could have detected; its scale is fit on the :data:`TEST_CAL_DAYS` sessions
    before TEST."""
    t = _row(r)
    held = np.isfinite(t["r2"]) and t["r2"] > 0 and t["z"] >= 1.96
    return (f"- **{label}** ({t['granularity']:.0f} min, input {t['input']:.0f} min, predict "
            f"{t['prediction']:.0f} min): TEST R² {t['r2']:+.4f} (ridge {t['ridge_r2']:+.4f}), "
            f"direction z {t['z']:+.1f} → " + ("**held**." if held else "**did not hold**.")
            + f" TEST could detect R² ≥ {mde(t['null_sd']):.4f}.")


def example(panel, picks: pd.DataFrame, names, sp: dict, *, nth: int = 4,
            clock: str = "10:00") -> tuple[str, str]:
    """The worked decision: the first fast large cap among ``names`` (``explorer.example``) at
    ``clock`` on the ``nth`` usable VALIDATION session (10:00: every output span, up to the 6
    hours to the close, is graded there)."""
    name, _ = ex.example(picks, names, sp)
    day = session_index(prepared(panel)["m"]["dates"], *sp["validation"])[nth]
    return name, f"{prepared(panel)['m']['dates'][day]} {clock}"


def example_paths(panel, results, name: str, ts) -> list[dict]:
    """The worked decision's predicted price paths, one per result: the decision bar's close
    carried along each configuration's predicted bar moves (scaled and unscaled), at its bar
    closes.

    Returns:
        ``[{"label", "times", "price", "price_raw", "granularity", "beta"}]`` (results without
        that decision are left out).
    """
    pp = prepared(panel)
    oc, m = pp["oc"], pp["m"]
    t0 = pd.Timestamp(ts)
    s = list(m["dates"]).index(str(t0.date()))
    t = (t0.hour * 60 + t0.minute) - 570
    k = m["names"].index(name)
    out = []
    for label, r in results:
        g, _, Sm = r["cell"]
        hit = np.flatnonzero((oc["s"][r["rows"]] == s) & (oc["t"][r["rows"]] == t)
                             & (oc["k"][r["rows"]] == k))
        if not len(hit):
            continue
        row = r["rows"][hit[0]]
        U = oc["U"][row]
        x = np.array(CHECKPOINTS)
        on = np.isfinite(r["raw"][hit[0]])
        p0 = oc["p0"][row]
        times = [t0] + [t0 + pd.Timedelta(minutes=int(c)) for c in x[on]]
        beta = float(r["beta"][oc["month"][row]])          # the scale this decision's path got
        out.append({"label": label, "granularity": g, "beta": beta, "times": times,
                    "price": np.r_[p0, p0 * np.exp(r["pred"][hit[0], on] * U[on])],
                    "price_raw": np.r_[p0, p0 * np.exp(r["raw"][hit[0], on] * U[on])]})
    return out


def day_picture(panel, name: str, ts, cell) -> dict:
    """One stock-day series as a model with ``cell``'s bars and spans reads it at the decision
    ``ts`` (``charts.stock_day``): every bar's close and time from the series' first bar (the
    previous session's end) to the close, the "today" flag and the time of day it is given, and
    where the input window, the decision, the predicted span and the checkpoints fall.

    Returns:
        ``times``, ``price``, ``today``, ``tod_frac``, ``first`` (today's first bar), ``start``
        (the input window's first bar), ``decision``, ``end`` (the last predicted bar), all as
        positions in ``times``; ``checkpoints`` (times), ``name``, ``cell``, ``bars``.
    """
    g, Lm, Sm = cell
    L, H = Lm // g, math.ceil(Sm / g)
    pp = prepared(panel)
    gi = grid_inputs(pp, g)
    m = pp["m"]
    t0 = pd.Timestamp(ts)
    s = list(m["dates"]).index(str(t0.date()))
    t = t0.hour * 60 + t0.minute - 570
    k = m["names"].index(name)
    a, b = gi["sf"][s] - (L - 1), gi["sf"][s] + gi["n_pos"]
    j = np.arange(a, b)
    day = pd.to_datetime(pd.Series(m["dates"][j // gi["n_pos"]]))
    times = pd.DatetimeIndex(day + pd.to_timedelta(570 + (j % gi["n_pos"] + 1) * g, unit="min"))
    close = m["cf"].reshape(-1, len(m["names"]))[(j // gi["n_pos"]) * SESSION
                                                  + (j % gi["n_pos"] + 1) * g - 1, k]
    fut = future(gi, s, L).values()
    dec = int(decision_bar(gi, s, t)) - a
    return {"times": times, "price": close, "today": fut[:, CLOCK.index("today")],
            "tod_frac": fut[:, CLOCK.index("tod_frac")], "first": gi["sf"][s] - a,
            "start": dec - L + 1, "decision": dec, "end": min(dec + H, b - a - 1),
            "checkpoints": [t0 + pd.Timedelta(minutes=c) for c in CHECKPOINTS if c <= Sm],
            "name": name, "cell": cell, "bars": (L, H)}


def grid_table(grans, inputs, outputs, sessions: int) -> pd.DataFrame:
    """The grid in bars: per bar size, how many bars each input and output span is (an output
    span that is not a whole number of bars is rounded up and graded only up to its span), and in
    the last row the independent training paths per stock each output span holds over
    ``sessions`` training sessions: the same at every bar size."""
    rows = []
    for g in grans:
        row = {"bar size (min)": str(g)}
        row.update({f"input {x} min": x // g for x in inputs})
        row.update({f"predict {x} min": math.ceil(x / g) for x in outputs})
        rows.append(row)
    last = {"bar size (min)": "independent training paths / stock"}
    last.update({f"input {x} min": "" for x in inputs})
    last.update({f"predict {x} min": train_paths(sessions, x) for x in outputs})
    return pd.DataFrame([*rows, last]).set_index("bar size (min)")


def atlas_words(at: dict) -> str:
    """The map of the day (:func:`atlas`) in plain words: its strongest cell against the bar for
    the best of all its cells, how many cells pass 1.96, and the first ``x`` minutes against the
    rest of the day."""
    n = len(at["labels"])
    up = np.triu(np.ones((n, n), bool), 1)
    z = np.where(up, at["z"], np.nan)
    i, j = np.unravel_index(np.nanargmax(np.abs(z)), z.shape)
    k = int(up.sum())
    bar = ex.best_of_bar(k)
    first = ", ".join(f"{'the night alone' if x == 0 else f'{x} min'}: corr "
                      f"{r['corr with the rest of the day']:+.3f} (z {r['z']:+.1f})"
                      for x, r in at["first"].iterrows())
    return "\n".join([
        f"- **The strongest cell:** {at['labels'][i]} → {at['labels'][j]}: corr "
        f"{at['corr'][i, j]:+.3f}, z {z[i, j]:+.1f}; the bar for the best of {k} cells by luck is "
        f"|z| {bar:.1f}: " + ("**above it**." if abs(z[i, j]) >= bar else
                              "**below it**: what the best of this many cells looks like by luck."),
        f"- **Cells beyond |z| 1.96:** {int((np.abs(z) >= 1.96).sum())} of {k} (luck alone gives "
        f"about {0.05 * k:.0f}).",
        f"- **The first x minutes against the rest of the day:** {first}."])


def planted_table(rows: list[dict]) -> pd.DataFrame:
    """The positive control as a table: per planted size, what was planted (the oracle's R²),
    what the network and the ridge recovered, and the network's direction against luck."""
    return pd.DataFrame(rows).set_index("planted correlation")


#: The models in one line each (the notebook's §6).
MODEL_WORDS = ex.MODEL_WORDS


def models_table(models) -> pd.DataFrame:
    """The models compared, in words."""
    return ex.models_table(models)
