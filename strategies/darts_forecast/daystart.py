"""darts_forecast day-start helpers (``darts_forecast_daystart.ipynb``): a Darts model that reads a
fixed stretch of the trading day from a fixed morning time and predicts the next stretch of the
SAME day.

Every example is one STOCK-DAY at ``g``-minute bars: the bars from ``09:30 + O`` to
``09:30 + O + L`` are the input, the next ``H`` minutes the output, and ``O + L + H <= 390``, so no
input or output ever reaches past the close or across a night. The four knobs (bar size ``g``,
morning offset ``O``, input ``L``, output ``H``, all in minutes; :data:`GRANULARITIES`,
:data:`OFFSETS`, :data:`INPUTS`, :data:`OUTPUTS`) form a grid that is measured cell by cell
(:func:`search`).

The model reads the target series (each bar's intraday move in typical moves of its clock
position, :func:`bars`), optionally the bar's volume (a past input) and optionally two numbers
about the previous days (:func:`previous`), static inputs beside the stock's one-hot. Within one
cell every example starts at the same clock time, so a bar's position in the window IS its clock
time: no calendar input is needed. One model per cell learns all the stocks; it is fit on TRAIN,
scaled by beta and graded on VALIDATION with the explorer's path grade (R² against "no change",
direction against an outcome-flip luck band, an AR(1) baseline), and read once on TEST.

It builds on ``explorer.py`` (the stock picks, the window, the networks, the grade), loaded once and
exposed as :data:`ex` (with ``ex.rs``: research.py). It edits no other module and imports nothing
from ``intraday.py`` (the minute arrays below follow its ``prepare`` / ``minute_arrays``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
DAYSTART_VERSION = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:10]
#: The grid, in minutes: bar size, morning offset (the input starts this long after 09:30),
#: input, output. A cell needs whole bars and O + L + H <= 390 (:func:`cells`).
GRANULARITIES = (1, 2, 3, 5, 10, 15, 30)
OFFSETS = (0, 15, 30, 60)
INPUTS = (15, 30, 60, 120)
OUTPUTS = (30, 60, 120, 240, 360)
#: Days held out at the end of the fit days, for early stopping and the scale beta (at most a
#: third of them). TEST holds out more: the days before it are holiday-thin late December.
HELD_DAYS = {"validation": 10, "test": 20}
VOL_DAYS = 20          # sessions behind every typical size (bar moves, volume, the day inputs)
CLIP = 10.0            # a bar move beyond 10 typical moves counts as 10 (training and truth)
CLIP_IN = {"volume": 5.0, "prev": 4.0}
#: The previous-day inputs, in their (fixed) column order.
PREV = ("gap", "prev_move")
#: A smoke run: everything small (merged into the notebook's settings; its 4 stocks come from
#: ``explorer.default_names``).
SMOKE = {"window": ("2025-11-01", "2026-01-31"), "train_months": 1,
         "granularities": (5, 30), "offsets": (0, 30), "inputs": (30, 60), "outputs": (60, 120),
         "epochs": 3, "draws": 30, "default": (5, 30, 30, 60), "train_from": None}
_OHLC = ("open_", "high", "low", "close")
_PREP: dict = {}
_BARS: dict = {}


# ---------------------------------------------------------------------------------------------
# The data: one minute cube, bars of every size, the previous-day inputs.
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


def prepared(panel) -> dict:
    """Everything the grid shares, built once per panel content (kept in-process, the last 3).

    The panel goes on the full clock (:func:`full_clock`) and IBKR's volume-0 placeholder minutes
    (a flat copy of the last price: nobody traded) count as no trade, as do the minutes after a
    half-day's 13:00 close. Every array is laid out one way (C order) before any sum, so the same
    numbers always round the same way (the leak check compares bit for bit).

    Returns:
        ``key``, ``dates`` (session dates, "YYYY-MM-DD"), ``names``, ``usable`` (not a half-day),
        ``regular`` (sessions x 390 x 1), ``volume`` / ``r1`` / ``cf`` (sessions x 390 x stocks:
        volume, each minute's intraday log move from the session's last traded close (its first
        trade from its own open, so no night is inside; NaN without a trade), the last traded
        close so far), ``first`` (the first traded minute, 390 without one), ``close`` (each
        session's last trade, carried over a day without one), ``gap`` (the night in typical
        nights, clipped) and ``day_rms`` (the typical close-to-close day move), sessions x stocks.
    """
    key = SM["fingerprint"](panel)
    if key in _PREP:
        return _PREP[key]
    p = full_clock(panel)
    n, N = len(p.close) // SESSION, len(p.symbols)
    shape = (n, SESSION, N)
    reg = SM["regular_rows"](p).reshape(n, SESSION)[..., None]
    arr = lambda k: np.ascontiguousarray(getattr(p, k).to_numpy(float)).reshape(shape)  # noqa: E731
    vol = np.where(reg, np.nan_to_num(arr("volume")), 0.0)
    op, C = (np.where(vol > 0, arr(k), np.nan) for k in ("open_", "close"))
    ok = np.isfinite(C)
    last = np.maximum.accumulate(np.where(ok, np.arange(SESSION)[None, :, None], -1), axis=1)
    cf = np.take_along_axis(C, np.maximum(last, 0), axis=1)
    cf[last < 0] = np.nan
    prev = np.concatenate([np.full((n, 1, N), np.nan), cf[:, :-1]], axis=1)
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), SESSION)
    o1 = np.take_along_axis(op, np.minimum(first, SESSION - 1)[:, None, :], axis=1)[:, 0]
    close = pd.DataFrame(cf[:, -1]).ffill().to_numpy()
    before = np.r_[np.full((1, N), np.nan), close[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        r1 = np.where(np.isfinite(prev), np.log(C / prev), np.log(C / op))
        raw_gap, day = np.log(o1 / before), np.log(close / before)
    lo = VOL_DAYS // 2
    night = 1.4826 * pd.DataFrame(np.abs(raw_gap)).rolling(VOL_DAYS, min_periods=lo).median()
    day_rms = np.sqrt(pd.DataFrame(day ** 2).rolling(VOL_DAYS, min_periods=lo).mean())
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.clip(raw_gap / night.shift(1).to_numpy(), -CLIP_IN["prev"], CLIP_IN["prev"])
    dates = np.array([str(d) for d in p.close.index[::SESSION].tz_convert(p.tz).date])
    out = {"key": key, "dates": dates, "names": list(p.symbols),
           "usable": ~np.isin(dates, list(SM["HALF_DAYS"])), "regular": reg, "volume": vol,
           "r1": r1, "cf": cf, "first": first, "close": close, "gap": gap,
           "day_rms": day_rms.shift(1).to_numpy()}
    _PREP[key] = out
    while len(_PREP) > 3:
        _PREP.pop(next(iter(_PREP)))
    return out


def bars(pp: dict, g: int) -> dict:
    """Everything on ``g``-minute bars (kept in-process per panel and bar size), as (sessions x
    390/g x stocks) arrays.

    ``y`` is each bar's intraday log move (the sum of its minutes' moves: a session's first bar
    runs from its first traded open, a bar without a trade moves 0, which keeps the cumulative path
    exact) in TYPICAL MOVES: divided by ``u``, the root-mean-square of the same clock position's
    moves over the previous :data:`VOL_DAYS` sessions, floored at a quarter of the stock's typical
    bar over those sessions, and clipped at ±:data:`CLIP`. ``volume`` is log(1 + the bar's
    volume) minus log(1 + the stock's average bar volume over the previous sessions), clipped at
    ±5: the busy open and close stay in it. A half-day's afternoon is NaN.

    Returns:
        ``{"g", "nb", "y", "u", "volume", "close" (the last traded close at each bar's end),
        "pin" (``{"y", "g": {"code"}}``: the flat, NaN-free target that ``explorer.actual_paths``
        and ``explorer.ar1`` read)}``.
    """
    key = (pp["key"], g)
    if key in _BARS:
        return _BARS[key]
    if SESSION % g:
        raise ValueError(f"{g}-minute bars do not tile the 390-minute session")
    n, _, N = pp["r1"].shape
    nb = SESSION // g
    reg = pp["regular"].reshape(n, nb, g, 1).any(axis=2)
    r = np.where(reg, np.nansum(pp["r1"].reshape(n, nb, g, N), axis=2), np.nan)
    lo = VOL_DAYS // 2
    ms = pd.DataFrame((r ** 2).reshape(n, nb * N)).rolling(VOL_DAYS, min_periods=lo).mean()
    ms = ms.shift(1).to_numpy().reshape(n, nb, N)
    vb = pp["volume"].reshape(n, nb, g, N).sum(axis=2)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        u = np.fmax(np.sqrt(ms), 0.25 * np.sqrt(np.nanmean(ms, axis=1, keepdims=True)))
        y = np.clip(r / u, -CLIP, CLIP)
        avg = pd.DataFrame(np.nanmean(np.where(reg, vb, np.nan), axis=1)).rolling(
            VOL_DAYS, min_periods=lo).mean().shift(1).to_numpy()
        vin = np.clip(np.log1p(vb) - np.log1p(avg)[:, None, :], -CLIP_IN["volume"],
                      CLIP_IN["volume"])
    out = {"g": g, "nb": nb, "y": y, "u": u, "volume": np.where(reg, vin, np.nan),
           "close": pp["cf"].reshape(n, nb, g, N)[:, :, -1],
           "pin": {"y": np.nan_to_num(y).reshape(n * nb, N),
                   "g": {"code": np.repeat(np.arange(n), nb)}}}
    _BARS[key] = out
    while len(_BARS) > 8:
        _BARS.pop(next(iter(_BARS)))
    return out


def previous(pp: dict, prev_days: int = 1) -> dict:
    """The two previous-day inputs per session and stock, both known at 09:30 (examples of daily
    information the model can take; not chosen by a study).

    * ``gap``: log(the session's first traded open / the previous session's last trade), divided
      by the typical night (1.4826 x the median absolute gap of the previous :data:`VOL_DAYS`
      nights: one earnings night cannot shrink the next twenty), clipped at ±4.
    * ``prev_move``: log(the previous session's last trade / the last trade ``prev_days`` sessions
      before it), divided by sqrt(``prev_days``) x the typical close-to-close day move of the
      previous :data:`VOL_DAYS` sessions, clipped at ±4.

    Returns:
        ``{"gap", "prev_move"}``, (sessions x stocks) arrays.
    """
    c, x = pp["close"], int(prev_days)
    pad = lambda m: np.r_[np.full((m, c.shape[1]), np.nan), c[:-m]]  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.log(pad(1) / pad(x + 1)) / (np.sqrt(x) * pp["day_rms"])
    return {"gap": pp["gap"], "prev_move": np.clip(move, -CLIP_IN["prev"], CLIP_IN["prev"])}


def statics(pp: dict, cell, *, prev: bool, prev_days: int = 1) -> np.ndarray:
    """The previous-day inputs of one configuration, (sessions x stocks x 2) in the order of
    :data:`PREV` (no columns without ``prev``). The gap is 0 where the stock had not traded before
    the decision (09:30 + O + L): its first open is not known there yet. NaN is 0."""
    if not prev:
        return np.zeros((*pp["close"].shape, 0))
    _, off, L, _ = cell
    pv = previous(pp, prev_days)
    gap = np.where(pp["first"] < off + L, pv["gap"], 0.0)
    return np.nan_to_num(np.stack([gap, pv["prev_move"]], axis=-1))


# ---------------------------------------------------------------------------------------------
# The grid, the examples, fit and predict.
# ---------------------------------------------------------------------------------------------
def cells(grans=GRANULARITIES, offsets=OFFSETS, inputs=INPUTS, outputs=OUTPUTS) -> list[tuple]:
    """Every (bar, offset, input, output) cell in minutes with whole bars inside one session."""
    return [(g, o, i, h) for g in grans for o in offsets for i in inputs for h in outputs
            if not (o % g or i % g or h % g) and o + i + h <= SESSION]


def clock(minutes: int) -> str:
    """The New York clock time ``minutes`` after the 09:30 open, "HH:MM"."""
    return f"{(570 + minutes) // 60:02d}:{(570 + minutes) % 60:02d}"


def label(cell) -> str:
    """A configuration in words."""
    g, off, L, H = cell
    return f"{g}-min bars, read {clock(off)}-{clock(off + L)}, predict to {clock(off + L + H)}"


def warmup(train_from: str | None, window_start: str) -> int | None:
    """The sessions ``explorer.load_window`` must load before the window so that training can
    start at ``train_from``: the exchange sessions from ``train_from`` to the day before the window
    (at least the explorer's 21); None (the explorer's default) without ``train_from``."""
    if not train_from:
        return None
    days = pd.bdate_range(train_from, pd.Timestamp(window_start) - pd.Timedelta(days=1))
    return max(21, int((~days.strftime("%Y-%m-%d").isin(list(SM["HOLIDAYS"]))).sum()))


def split_days(pp: dict, sp: dict, part: str,
               fit_from: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The usable sessions (indices) a fit for ``part`` learns from, the held-out ones after them
    (early stopping and beta: the last :data:`HELD_DAYS` of the fit days, at most a third), and
    the graded ones. VALIDATION is graded by a fit on TRAIN, TEST by a refit on TRAIN and
    VALIDATION. The fit days start at ``fit_from`` (default: TRAIN's first day; an earlier day
    reaches into the panel's warm-up), but never before the typical sizes have
    ``VOL_DAYS // 2`` earlier sessions. Half-days are never used.

    Returns:
        ``(fit, held, graded)`` session indices.
    """
    d, use = pp["dates"], pp["usable"]
    last = sp["train"][1] if part == "validation" else sp["validation"][1]
    ready = np.arange(len(d)) >= VOL_DAYS // 2
    days = np.flatnonzero(use & ready & (d >= (fit_from or sp["train"][0])) & (d <= last))
    held = max(1, min(HELD_DAYS[part], len(days) // 3))
    graded = np.flatnonzero(use & (d >= sp[part][0]) & (d <= sp[part][1]))
    return days[:-held], days[-held:], graded


def examples(b: dict, pp: dict, cell, days, *, full: bool, volume: bool, prev: bool,
             prev_days: int = 1) -> tuple:
    """The Darts series of every (day, stock): the target bars from 09:30 + O, the input and the
    output (``full``: EXACTLY ``(L + H) / g`` bars, one training sample each; Darts sizes the
    samples of every series by the longest one) or the input only (to predict from), the volume
    input over the same bars, and the static inputs (the stock's one-hot, then :data:`PREV`) in
    one fixed column order (Darts checks only their number).

    Returns:
        ``(series, volume series or None, day per series, stock per series)``.
    """
    from darts import TimeSeries

    g, off, L, H = cell
    a, n = off // g, (L + H) // g if full else L // g
    names = pp["names"]
    st = statics(pp, cell, prev=prev, prev_days=prev_days)
    cols = [f"is_{x}" for x in names] + (list(PREV) if prev else [])
    eye = np.eye(len(names))
    y = np.nan_to_num(b["y"]).astype(np.float32)
    vi = np.nan_to_num(b["volume"]).astype(np.float32)
    idx = pd.RangeIndex(0, n)
    ser, vol, dd, kk = [], [], [], []
    for d in days:
        for k in range(len(names)):
            frame = pd.DataFrame([np.r_[eye[k], st[d, k]]], columns=cols)
            ser.append(TimeSeries.from_times_and_values(idx, y[d, a:a + n, k:k + 1], columns=["y"])
                       .with_static_covariates(frame))
            if volume:
                vol.append(TimeSeries.from_times_and_values(idx, vi[d, a:a + n, k:k + 1],
                                                            columns=["volume"]))
            dd.append(d)
            kk.append(k)
    return ser, (vol if volume else None), np.array(dd, int), np.array(kk, int)


def fit(b: dict, pp: dict, cell, kind: str, fit_days, held_days, *, volume: bool, prev: bool,
        prev_days: int, epochs: int, seed: int = 0, device: str = "cpu") -> tuple:
    """ONE model for every stock (``explorer.make_path_model``), fit on the stock-days of
    ``fit_days``; the ``held_days`` stop its training early and the best epoch's weights are kept.

    Returns:
        ``(model, info)``; ``info``: ``epochs`` (run), ``best_val_loss``, ``seconds``,
        ``stride`` (1), ``samples``, ``fit`` / ``held`` (first and last date).
    """
    g, _, L, H = cell
    kw = dict(volume=volume, prev=prev, prev_days=prev_days, full=True)
    ser, vol, _, _ = examples(b, pp, cell, fit_days, **kw)
    vser, vvol, _, _ = examples(b, pp, cell, held_days, **kw)
    est = ex.make_path_model(kind, L // g, H // g, epochs=epochs, n_samples=len(ser), seed=seed,
                             device=device)
    cov = dict(past_covariates=vol, val_past_covariates=vvol) if volume else {}
    t0 = time.time()
    est.fit(ser, val_series=vser, verbose=False, **cov)
    SM["_quiet"]()
    run_ = int(est.epochs_trained) if hasattr(est, "epochs_trained") else epochs  # before predict
    keep = next(c for c in est.trainer_params["callbacks"] if hasattr(c, "state"))
    if keep.state is not None:
        est.model.load_state_dict(keep.state)
    d = pp["dates"]
    return est, {"epochs": run_, "best_val_loss": round(keep.best, 5),
                 "seconds": round(time.time() - t0, 1), "stride": 1, "samples": len(ser),
                 "fit": (str(d[fit_days[0]]), str(d[fit_days[-1]])),
                 "held": (str(d[held_days[0]]), str(d[held_days[-1]]))}


def predict(est, b: dict, pp: dict, cell, days, *, volume: bool, prev: bool, prev_days: int,
            batch: int = 1024) -> tuple:
    """The model's next ``H / g`` bar moves (typical moves) of every (day, stock), read from that
    day's input bars only (Darts is asked for exactly ``output_chunk_length`` bars).

    Returns:
        ``(paths (examples x H/g), s, j)``: ``s`` is the decision bar on one global bar number
        (day x bars per session + bar), the same for every stock that day; ``j`` the stock.
    """
    g, off, L, H = cell
    ser, vol, dd, kk = examples(b, pp, cell, days, full=False, volume=volume, prev=prev,
                                prev_days=prev_days)
    s = dd * b["nb"] + (off + L) // g - 1
    if not ser:
        return np.zeros((0, H // g), np.float32), s, kk
    kw = {"past_covariates": vol} if volume else {}
    out = est.predict(n=H // g, series=ser, batch_size=batch, verbose=False, **kw)
    return np.stack([p.values()[:, 0] for p in out]).astype(np.float32), s, kk


# ---------------------------------------------------------------------------------------------
# One configuration end to end, the grid, the input check, the checks.
# ---------------------------------------------------------------------------------------------
def run(panel, sp: dict, cell, *, kind: str = "tide", volume: bool = True, prev: bool = True,
        prev_days: int = 1, fit_from: str | None = None, epochs: int = 50, draws: int = 200,
        part: str = "validation", seed: int = 0, device: str = "cpu") -> dict:
    """One configuration end to end: fit ONE model on the fit days (:func:`split_days`), scale its
    paths by beta on the held-out days (``explorer.calibration``), predict the path of every graded
    stock-day and grade it with ``explorer.score_paths`` beside the unscaled paths (direction, luck
    band) and the AR(1) baseline. Saved in the cache folder, keyed by the configuration, the
    inputs, the split, the panel's content, the code and the constants (the name carries
    strategy.py's version, so ``research.prune_cache`` keeps it).

    Args:
        panel: The 1-minute panel (``explorer.load_window``).
        sp: Its split.
        cell: ``(bar, offset, input, output)`` in minutes.
        kind: The Darts model (``explorer.PATH_MODELS``).
        volume: Read the bar's volume.
        prev: Read the two previous-day inputs (:func:`previous`).
        prev_days: The sessions behind ``prev_move``.
        fit_from: The first training day (None: TRAIN's first; :func:`split_days`).
        epochs: At most this many training passes.
        draws: Luck-band futures (0: none).
        part: ``"validation"`` or ``"test"``.
        seed: Model and luck seed.
        device: ``cpu`` | ``mps``.

    Returns:
        The explorer's result layout (``cell`` in BARS: (g, L/g, H/g), ``score``, ``score_raw``,
        ``beta``, ``ar1``, ``fit``, ``s`` / ``j``, ``pred`` / ``raw``), plus ``minutes`` (the cell),
        ``volume``, ``prev``, ``prev_days``, ``rho`` and ``checks`` (:func:`_checks`).
    """
    g, off, L, H = (int(x) for x in cell)
    cell = (g, off, L, H)
    if cell not in cells((g,), (off,), (L,), (H,)):
        raise ValueError(f"{cell}: needs whole {g}-minute bars and O + L + H <= {SESSION}")
    prev_days = int(prev_days) if prev else 0
    ident = json.dumps({"cell": cell, "kind": kind, "volume": volume, "prev": prev,
                        "prev_days": prev_days, "fit_from": fit_from, "epochs": epochs,
                        "draws": draws, "part": part, "seed": seed, "device": device,
                        "split": [sp["train"], sp["validation"], sp["test"]],
                        "panel": SM["fingerprint"](panel),
                        "code": [rs.CODE_VERSION, ex.EXPLORER_VERSION, ex.RESEARCH_VERSION,
                                 DAYSTART_VERSION],
                        "constants": [HELD_DAYS, VOL_DAYS, CLIP, CLIP_IN]},
                       sort_keys=True, default=str)
    tag = ("v" if volume else "") + (f"p{prev_days}" if prev else "") or "y"
    path = rs.CACHE / (f"daystart-{kind}-{g}m-o{off}-in{L}-out{H}-{tag}-{part}-"
                       f"c{rs.CODE_VERSION[:6]}-{hashlib.sha1(ident.encode()).hexdigest()[:12]}"
                       ".pkl")
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    pp = prepared(panel)
    b = bars(pp, g)
    fit_d, held_d, graded = split_days(pp, sp, part, fit_from)
    kw = dict(volume=volume, prev=prev, prev_days=prev_days)
    est, info = fit(b, pp, cell, kind, fit_d, held_d, epochs=epochs, seed=seed, device=device,
                    **kw)
    Hb, pin, names = H // g, b["pin"], pp["names"]
    ph, sh, jh = predict(est, b, pp, cell, held_d, **kw)
    beta = ex.calibration(ph, ex.actual_paths(pin, sh, jh, Hb)) if len(sh) else 0.0
    raw, s, j = predict(est, b, pp, cell, graded, **kw)
    act = ex.actual_paths(pin, s, j, Hb)
    pred = ex.scaled(raw, beta)
    rho = ex.ar1(pin, fit_d[0] * b["nb"], (fit_d[-1] + 1) * b["nb"])
    out = {"cell": (g, L // g, Hb), "minutes": cell, "kind": kind, "part": part,
           "volume": volume, "prev": prev, "prev_days": prev_days, "fit_from": fit_from,
           "s": s, "j": j,
           "pred": pred, "raw": raw, "beta": beta, "rho": rho,
           "score": ex.score_paths(pred, act, s, j, names),
           "score_raw": ex.score_paths(raw, act, s, j, names, draws=draws, seed=seed),
           "ar1": ex.score_paths(ex.ar1_paths(pin, s, j, rho, Hb), act, s, j, names),
           "fit": {**info, "first_graded": str(pp["dates"][graded[0]]) if len(graded) else ""}}
    out["checks"] = _checks(out, pp, sp, fit_d, held_d, graded)
    rs.CACHE.mkdir(exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh)
    os.replace(tmp, path)
    return out


def _checks(r: dict, pp: dict, sp: dict, fit_d, held_d, graded) -> list[tuple]:
    """The boundaries of one result, as built: the fit days start at ``fit_from`` (else TRAIN's
    first day) and end before the
    held-out days, those before the graded part; every example stays inside one regular session.
    (That the gap is withheld until a stock has traded is pinned by the tests: on liquid stocks
    no graded day is late, so a row here could not fail.)

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    d, part = pp["dates"], sp[r["part"]]
    g, off, L, H = r["minutes"]
    used = np.r_[fit_d, held_d, graded]
    half = sorted(set(d[used]) & set(SM["HALF_DAYS"]))
    late = int((pp["first"][graded] >= off + L).sum())
    return [
        ("fit days start at their first day and end before the held-out days, those before the "
         "graded part",
         bool(len(graded)) and (r["fit_from"] or sp["train"][0]) <= d[fit_d[0]]
         and d[fit_d[-1]] < d[held_d[0]] <= d[held_d[-1]] < part[0] <= d[graded[0]]
         and d[graded[-1]] <= part[1],
         f"fit {d[fit_d[0]]} → {d[fit_d[-1]]} ({len(fit_d)} days), held out {d[held_d[0]]} → "
         f"{d[held_d[-1]]} ({len(held_d)}), graded {r['part'].upper()} "
         f"{d[graded[0]] if len(graded) else '-'} → {d[graded[-1]] if len(graded) else '-'} "
         f"({len(graded)})"),
        ("every example stays inside one regular session (no half-day)",
         off + L + H <= SESSION and not half,
         f"{label(r['minutes'])}; {len(used)} days; {late} graded stock-days without a trade "
         "before the decision (their gap withheld)" + (f"; half-days: {half}" if half else "")),
    ]


def row(r: dict) -> dict:
    """One configuration's numbers: ``explorer._row`` (bars) plus the cell in minutes and the
    inputs read."""
    g, off, L, H = r["minutes"]
    return {**ex._row(r), "offset_min": off, "input_min": L, "output_min": H,
            "volume": r["volume"], "prev": r["prev"]}


def search(panel, sp: dict, *, grans, offsets, inputs, outputs, verbose: bool = True,
           **kw) -> pd.DataFrame:
    """Every cell of the grid (:func:`cells`) measured once on VALIDATION (each fit takes
    seconds, so no optimizer is needed). TEST is never read here.

    Args:
        panel: The panel.
        sp: Its split.
        grans: Bar sizes.
        offsets: Morning offsets.
        inputs: Input lengths.
        outputs: Output lengths (all in minutes).
        verbose: One progress line per cell.
        **kw: :func:`run`'s settings (``kind``, ``volume``, ``prev``, ``prev_days``, ``epochs``,
            ``draws``, ``seed``).

    Returns:
        One row per cell (:func:`row`, ``seconds``).
    """
    todo, rows = cells(grans, offsets, inputs, outputs), []
    for i, cell in enumerate(todo, start=1):
        t0 = time.time()
        rows.append({**row(run(panel, sp, cell, **kw)), "seconds": round(time.time() - t0, 1)})
        if verbose:
            x = rows[-1]
            print(f"{i:>3}/{len(todo)}  {label(cell)}: R² {x['r2']:+.4f} (scale "
                  f"{x['beta']:.2f}; direction z {x['z']:+.1f})  {x['seconds']:.0f} s")
    return pd.DataFrame(rows)


def ablation(panel, sp: dict, configs, *, part: str = "validation", **kw) -> pd.DataFrame:
    """Each optional input switched off in turn, for every configuration of ``configs``
    (``[(label, cell)]``): its row and the change (Δ) of R², direction corr and z against the
    configuration as set. Read the direction: the scale often makes R² exactly 0 either way.

    Returns:
        One row per configuration and inputs read.
    """
    variants = [("as set", kw["volume"], kw["prev"])]
    if kw["volume"]:
        variants.append(("without volume", False, kw["prev"]))
    if kw["prev"]:
        variants.append(("without the previous days", kw["volume"], False))
    rows = []
    for name, cell in configs:
        base = None
        for inputs, v, p in variants:
            x = row(run(panel, sp, cell, part=part, **{**kw, "volume": v, "prev": p}))
            base = base or x
            rows.append({"config": name, "inputs": inputs, "part": part, **x,
                         **{f"Δ{m}": x[m] - base[m] for m in ("r2", "corr", "z")}})
    return pd.DataFrame(rows)


def truncation(panel, sp: dict, cell, *, cut_session: int = 10, **kw) -> list[tuple]:
    """Nothing reads the future: every 1-minute bar from the decision (09:30 + O + L) of the
    graded VALIDATION day ``cut_session`` (0-based) on is replaced by random bars
    (``checks.scrambled``) and the WHOLE configuration re-run (inputs, scales, fit, beta,
    predictions). Every path
    predicted up to that decision, and beta, must be identical; the later paths must change (else
    the check could not fail).

    Args:
        panel: The panel.
        sp: Its split.
        cell: ``(bar, offset, input, output)`` in minutes.
        cut_session: Which graded VALIDATION day is cut.
        **kw: :func:`run`'s settings (the main run's, so its saved result is the base).

    Returns:
        ``[(check, ok, detail)]`` rows.
    """
    ck = _sibling("checks")
    panel = full_clock(panel)
    g, off, L, _ = cell
    base = run(panel, sp, cell, **kw)
    pp = prepared(panel)
    d_cut = split_days(pp, sp, "validation")[2][cut_session]
    rows = np.zeros(len(panel.close), bool)
    rows[d_cut * SESSION + off + L:] = True
    alt = run(ck.scrambled(panel, rows, kw.get("seed", 0)), sp, cell, **kw)
    early = base["s"] // (SESSION // g) <= d_cut
    same = np.array_equal(base["raw"][early], alt["raw"][early]) and base["beta"] == alt["beta"]
    moved = not np.array_equal(base["raw"][~early], alt["raw"][~early])
    return [("paths predicted up to the cut are unchanged, and so is the scale",
             bool(early.any()) and same,
             f"cut at {pp['dates'][d_cut]} {clock(off + L)}; {int(early.sum()):,} paths compared; "
             f"scale {base['beta']:.3f}"),
            ("paths predicted after the cut did change (the check can fail)", moved,
             f"{int((~early).sum()):,} later paths")]


# ---------------------------------------------------------------------------------------------
# Tables, words and the worked picture's data.
# ---------------------------------------------------------------------------------------------
def results_table(res: pd.DataFrame) -> pd.DataFrame:
    """The grid's results, best first, in minutes."""
    x = ex.best(res, k=len(res))
    return pd.DataFrame({
        "bar (min)": x["granularity"], "offset (min)": x["offset_min"],
        "input (min)": x["input_min"], "predict (min)": x["output_min"],
        "R² vs no change (scaled)": x["r2"].round(4), "scale β": x["beta"].round(2),
        "stocks R² > 0": (100 * x["share_pos"]).round(0).astype(int).astype(str) + "%",
        "R² unscaled": x["r2_raw"].round(4), "direction corr": x["corr"].round(3),
        "direction z vs luck": x["z"].round(1), "AR(1) R²": x["ar1_r2"].round(4),
        "paths": x["paths"], "seconds": x["seconds"]}).reset_index(drop=True).rename_axis(
        "rank").rename(index=lambda i: i + 1)


def words(res: pd.DataFrame) -> str:
    """The grid in plain words: the winner, how many cells beat "no change" and luck, and whether
    the best direction clears the bar for the best of all the cells."""
    top = ex.best(res).iloc[0]
    bar = ex.best_of_bar(len(res))
    cell = tuple(int(top[c]) for c in ("granularity", "offset_min", "input_min", "output_min"))
    best_z = res["z"].max()
    return "\n".join([
        f"- **Winner on VALIDATION:** {label(cell)}: R² {top['r2']:+.4f} against \"no change\" "
        f"({100 * top['share_pos']:.0f}% of the stocks above 0, scale {top['beta']:.2f}); "
        f"direction corr {top['corr']:+.3f}, z {top['z']:+.1f}; AR(1) R² {top['ar1_r2']:+.4f}.",
        f"- **Better than \"no change\" (R² > 0):** {int((res['r2'] > 0).sum())} of {len(res)} "
        f"cells; {int((res['beta'] == 0).sum())} scaled to \"no change\" outright; **direction "
        f"above luck (z ≥ 2):** {int((res['z'] >= 2).sum())}.",
        f"- **The bar for the best of {len(res)} cells:** z {bar:.1f}. The best direction z "
        f"({best_z:+.1f}) is " + ("**above it**." if best_z >= bar else
                                   "**below it**: what the best of this many tries looks like "
                                   "by luck alone."),
    ])


def example_paths(panel, results, name: str, day: str) -> list[dict]:
    """The worked decision's predicted price paths, one per result (``[(label, run)]``): the
    decision's last traded close (else the last trade before it) carried along the scaled and the
    unscaled predicted bar moves, each converted with its bar's typical move (known at the
    decision).

    Returns:
        ``[{"label", "times", "price", "price_raw", "granularity", "beta"}]`` for
        ``charts.path_picture`` (results without that decision are left out).
    """
    pp = prepared(panel)
    k = pp["names"].index(name)
    d = int(np.flatnonzero(pp["dates"] == day)[0])
    out = []
    for lab, r in results:
        g, off, L, H = r["minutes"]
        b = bars(pp, g)
        hit = np.flatnonzero((r["s"] == d * b["nb"] + (off + L) // g - 1) & (r["j"] == k))
        if not len(hit):
            continue
        p0 = pp["cf"][d, off + L - 1, k]
        p0 = p0 if np.isfinite(p0) else pp["close"][d - 1, k]
        u = np.nan_to_num(b["u"][d, (off + L) // g:(off + L + H) // g, k])
        times = pd.date_range(pd.Timestamp(str(day)) + pd.Timedelta(minutes=570 + off + L),
                              periods=H // g + 1, freq=f"{g}min")
        out.append({"label": lab, "granularity": g, "beta": r["beta"], "times": times,
                    "price": np.r_[p0, p0 * np.exp(np.cumsum(r["pred"][hit[0]] * u))],
                    "price_raw": np.r_[p0, p0 * np.exp(np.cumsum(r["raw"][hit[0]] * u))]})
    return out
