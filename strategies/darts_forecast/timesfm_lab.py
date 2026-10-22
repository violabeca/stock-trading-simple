"""TimesFM helpers (``darts_forecast_timesfm_marimo.py``): Google's pre-trained TimesFM 3.0
forecaster (Darts ``TimesFM3Model``), used ZERO-SHOT: nothing is trained or tuned on these stocks.
It reads a stock's recent closes and returns the next bars as three quantile paths (10 / 50 /
90 %), which are graded against the forecast "no change".

Four steps, one function each:

* :func:`bars`: every stock's regular-session closes and volumes on a full clock grid, the nights
  cut out (the model reads ONE long series in which the next session's first bar follows the
  close), stamped at the bar's CLOSE time;
* :func:`origins`: the moments a forecast is made, each with a full context before it and its whole
  horizon in the data (intraday: at fixed clock times, never across a night; multi-day: after each
  session's close);
* :func:`forecast`: the model's quantile paths for every origin, cached on disk (``var/``);
* :func:`score`: the paths against what the price did, beside "no change", with a luck band from
  flipping the sign of the actual moves (one coin per block of sessions, shared by every stock).

The figures (:func:`forecast_figure`, :func:`gallery`, :func:`skill_figure`,
:func:`coverage_figure`, :func:`end_scatter`) use the repo's palette and layout and cut the nights
out of the time axis.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_simple.plots import _GRID, _INK, _MUTED, SERIES, _base, _intraday_breaks

HERE = Path(__file__).resolve().parent
CACHE = HERE.parents[1] / "var" / "timesfm_lab"  # saved forecasts (git-ignored)
HUB = "google/timesfm-3.0-pytorch"
#: The weights' snapshot, downloaded 2026-10-16: pinned so a new upload never changes a result
#: silently, and so a cached snapshot loads without asking the Hub anything.
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
QUANTILES = (0.1, 0.5, 0.9)  # a subset of the nine the model was trained on
SETUPS = ("price only", "price + volume")  # what the model reads (volume as a past covariate)
VERSION = 1  # bump when forecast() changes what it computes (part of the disk-cache key)
OPEN = 9 * 60 + 30  # the regular session's open, minutes after midnight New York
CHUNK = 256  # origins per predict call (one progress step)
TIE_BP = 0.01  # a median end move under this (bp) calls no direction (float32 rounding of flat)
_BAND = "rgba(42, 120, 214, 0.16)"  # SERIES[0], light: the 10-90 % band


def _sibling(name: str):
    """A helper module of this folder, loaded by path once (shared with the other notebooks)."""
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


rs = _sibling("research")
charts = _sibling("charts")


# ---------------------------------------------------------------------------------------------
# Data and origins.
# ---------------------------------------------------------------------------------------------
def bars(panel, minutes: int = 1) -> dict:
    """Every stock's regular-session closes and volumes on a full clock grid of ``minutes``-minute
    bars, sessions one after the other (the nights cut out).

    A missing minute takes the previous close and zero volume (an IBKR volume-0 placeholder minute
    is already such a copy); a half-day ends at 13:00 (``strategy.HALF_DAYS``: the stored session
    runs on). Coarser bars are built from the minutes inside each session, aligned to 09:30.

    Args:
        panel: A 1-minute panel (``research.load``).
        minutes: Bar size in minutes (1 keeps the minutes).

    Returns:
        ``{"time"}``: each bar's CLOSE time (New York wall clock, naive), ``"day"``: session
        number, ``"close"`` / ``"volume"``: bars x stocks arrays, ``"symbols"``, ``"minutes"``,
        ``"per_day"``: bars in a full session, ``"filled"``: share of grid minutes without a
        stored bar, per stock.
    """
    keep = rs.SM["regular_rows"](panel)
    wall = panel.close.index.tz_convert(panel.tz).tz_localize(None)
    close = pd.DataFrame(panel.close.to_numpy(float), index=wall, columns=panel.symbols)[keep]
    volume = pd.DataFrame(panel.volume.to_numpy(float), index=wall, columns=panel.symbols)[keep]
    days = close.index.normalize().unique()
    half = set(rs.SM["HALF_DAYS"])
    grid = pd.DatetimeIndex(np.concatenate([
        (d + pd.Timedelta(minutes=OPEN)
         + pd.to_timedelta(np.arange(210 if f"{d:%Y-%m-%d}" in half else 390), "min")).to_numpy()
        for d in days]))
    close = close.reindex(grid)
    filled = close.isna().mean()
    close, volume = close.ffill(), volume.reindex(grid).fillna(0.0)
    day = pd.factorize(grid.normalize())[0]
    stamp = pd.Series(grid + pd.Timedelta(minutes=1))  # a 1-minute bar's close time
    if minutes > 1:
        minute = ((grid - grid.normalize()).total_seconds() // 60 - OPEN).astype(int)
        keys = [day, np.asarray(minute) // int(minutes)]
        close, volume = close.groupby(keys).last(), volume.groupby(keys).sum()
        stamp = stamp.groupby(keys).last()  # the bucket's last minute's close
        day = close.index.get_level_values(0).to_numpy()
    return {"time": pd.DatetimeIndex(stamp.to_numpy()), "day": np.asarray(day),
            "close": close.to_numpy(float), "volume": volume.to_numpy(float),
            "symbols": list(panel.symbols), "minutes": int(minutes),
            "per_day": -(-390 // int(minutes)), "filled": filled}


def origins(B: dict, *, context: int, horizon: int, clocks=None, limit: int | None = None
            ) -> pd.DataFrame:
    """The forecasts to make: one row per (stock, origin bar ``t``), the forecast made when bar
    ``t`` closes from the ``context`` bars up to and including it, for the next ``horizon`` bars.

    Args:
        B: :func:`bars` output.
        context: Bars the model reads.
        horizon: Bars it forecasts.
        clocks: Intraday: the clock times a forecast is made each day (``"10:00"``, the close
            time of the last bar read); the horizon must end in the same session. None:
            multi-day, one forecast after each session's last bar, the horizon running on into
            the next sessions.
        limit: At most this many rows, evenly spaced over the ordered list (the budget).

    Returns:
        Columns ``symbol``, ``col`` (stock column in ``B``), ``t``, ``time`` (the origin),
        ``day`` (session number), in time order.
    """
    n, day = len(B["time"]), B["day"]
    if clocks is None:
        t = np.flatnonzero(np.r_[day[1:] != day[:-1], True])  # each session's last bar
    else:
        tod = B["time"] - B["time"].normalize()
        t = np.flatnonzero(tod.isin(pd.to_timedelta([f"{c}:00" for c in clocks])))
    t = t[(t + 1 >= context) & (t + horizon < n)]
    if clocks is not None:
        t = t[day[t + horizon] == day[t]]  # never across a night
    rows = [(s, c, int(u)) for u in t for c, s in enumerate(B["symbols"])
            if np.isfinite(B["close"][u - context + 1, c])]
    out = pd.DataFrame(rows, columns=["symbol", "col", "t"])
    if limit is not None and len(out) > limit:
        out = out.iloc[np.unique(np.linspace(0, len(out) - 1, int(limit)).round().astype(int))]
    out = out.reset_index(drop=True)
    out["time"], out["day"] = B["time"][out["t"]], day[out["t"]]
    return out


# ---------------------------------------------------------------------------------------------
# The model.
# ---------------------------------------------------------------------------------------------
def _device(device: str) -> str:
    """``device``, or "cpu" when the Apple GPU (``"mps"``) is not available."""
    import torch

    return "cpu" if device == "mps" and not torch.backends.mps.is_available() else device


def _model(context: int, horizon: int, *, volume: bool, device: str, batch: int):
    """TimesFM 3.0 with the pinned weights, ready to predict: ``context`` bars in, ``horizon``
    bars out in ONE pass (Google's own decoding; a longer output length or a variable input
    length changes the paths by a few bp). Downloads the weights (~1.3 GB, non-commercial
    licence) on first use."""
    from darts import TimeSeries
    from darts.models import TimesFM3Model
    from darts.utils.likelihood_models import QuantileRegression

    m = TimesFM3Model(
        input_chunk_length=context, output_chunk_length=horizon, accept_license=True,
        hub_model_name=HUB, hub_model_revision=REVISION,
        likelihood=QuantileRegression(list(QUANTILES)), batch_size=batch,
        pl_trainer_kwargs={"accelerator": _device(device), "enable_progress_bar": False,
                           "logger": False, "enable_model_summary": False})
    dummy = TimeSeries.from_values(np.linspace(1.0, 2.0, context + horizon, dtype=np.float32))
    m.fit(dummy, past_covariates=dummy if volume else None)  # loads the weights; trains nothing
    return m


def _path(B: dict, org: pd.DataFrame, context: int, horizon: int, volume: bool) -> Path:
    """The cache file of one run: every setting and the exact numbers read."""
    h = hashlib.sha1(json.dumps([VERSION, REVISION, context, horizon, volume, B["minutes"],
                                 B["symbols"], list(QUANTILES)]).encode())
    for a in (B["close"], B["volume"], B["time"].asi8, org[["col", "t"]].to_numpy()):
        h.update(np.ascontiguousarray(a).tobytes())
    tag = "price+volume" if volume else "price"
    return CACHE / f"tfm-{B['minutes']}m-L{context}-H{horizon}-{tag}-{h.hexdigest()[:12]}.npy"


def cached(B: dict, org: pd.DataFrame, *, context: int, horizon: int, volume: bool) -> bool:
    """Whether :func:`forecast` would load these forecasts from disk."""
    return _path(B, org, context, horizon, volume).exists()


def forecast(B: dict, org: pd.DataFrame, *, context: int, horizon: int, volume: bool,
             device: str = "mps", batch: int = 64, progress=None) -> np.ndarray:
    """The model's quantile paths for every origin, in prices.

    The model reads the close (and, with ``volume``, ``log(1 + volume)`` as a past covariate) of
    the ``context`` bars up to the origin. Loaded only for the run and freed after it; the result
    is saved under ``var/timesfm_lab`` and read back for the same settings and data.

    Args:
        B: :func:`bars` output.
        org: :func:`origins` output.
        context: Bars read (as in :func:`origins`).
        horizon: Bars forecast.
        volume: Also read the volume.
        device: ``"mps"`` (Apple GPU, the same numbers as the CPU to ~1e-5 and faster) or
            ``"cpu"``.
        batch: Forecasts per network pass.
        progress: Called once per :data:`CHUNK` origins (a progress bar's ``update``).

    Returns:
        ``(len(org), horizon, 3)`` float32: the 10 / 50 / 90 % quantile of each future bar's close.
    """
    path = _path(B, org, context, horizon, volume)
    if path.exists():
        return np.load(path)
    import gc

    import torch
    from darts import TimeSeries

    model = _model(context, horizon, volume=volume, device=device, batch=batch)
    logv = np.log1p(B["volume"])
    out = []
    for a in range(0, len(org), CHUNK):
        part = org.iloc[a:a + CHUNK]
        win = [(slice(t - context + 1, t + 1), c)
               for t, c in zip(part["t"], part["col"], strict=True)]
        series = [TimeSeries.from_values(B["close"][w].astype(np.float32)) for w in win]
        covs = ([TimeSeries.from_values(logv[w].astype(np.float32)) for w in win]
                if volume else None)
        pred = model.predict(n=horizon, series=series, past_covariates=covs,
                             predict_likelihood_parameters=True, verbose=False)
        out.append(np.stack([p.values() for p in pred]).astype(np.float32))
        if progress is not None:
            progress()
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    q = np.concatenate(out) if out else np.empty((0, horizon, len(QUANTILES)), np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, q)
    return q


# ---------------------------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------------------------
def _band(real: float, flipped: np.ndarray) -> dict:
    """A statistic beside its luck band: the 2.5 / 97.5 % of the flipped draws and the z."""
    sd = float(flipped.std())
    return {"value": float(real), "lo": float(np.quantile(flipped, 0.025)),
            "hi": float(np.quantile(flipped, 0.975)),
            "z": (float(real) - float(flipped.mean())) / sd if sd > 0 else 0.0}


def score(B: dict, org: pd.DataFrame, q: np.ndarray, *, block: int = 1, flips: int = 1000,
          seed: int = 0) -> dict:
    """The forecasts against what the price did. Every move is a log return from the origin's
    close, in basis points.

    * **path error**: the median path's mean absolute error over all bars, as a share saved
      against the "no change" path (0 % = no better, positive = better), and per lead time;
    * **direction**: the share of forecasts whose median ends on the right side of the origin
      price (ties left out), and the correlation of the predicted and the actual end move;
    * **calibration**: the share of actual closes inside the 10-90 % band (80 % if calibrated),
      below the 10 % line and above the 90 % line (10 % each), per lead time.

    The luck band flips the sign of the ACTUAL moves (every move keeps its size): one coin per
    ``block`` sessions, shared by every stock and origin in it (a day's moves share news;
    overlapping multi-day horizons share bars), drawn in mirror pairs so the band is centred.
    It tests whether the forecasts know the DIRECTION; a lean (always "up" in a rising market)
    shows as ``pred_up`` against ``act_up``, not in the band.

    Args:
        B: :func:`bars` output.
        org: :func:`origins` output.
        q: :func:`forecast` output.
        block: Sessions per coin.
        flips: Sign draws (even).
        seed: Coin seed.

    Returns:
        A dict of statistics (``skill``, ``hit``, ``corr`` with their bands; per-lead arrays
        ``skill_step`` / ``skill_lo`` / ``skill_hi`` / ``inside`` / ``below`` / ``above``; per
        forecast ``ratio`` (path error / no-change error), ``f_end``, ``y_end``).
    """
    t, c = org["t"].to_numpy(), org["col"].to_numpy()
    H = q.shape[1]
    p0 = B["close"][t, c]
    act = B["close"][t[:, None] + np.arange(1, H + 1), c[:, None]]
    y = 1e4 * np.log(act / p0[:, None])
    f = 1e4 * np.log(q[:, :, 1] / p0[:, None])
    em, en, ef = np.abs(f - y), np.abs(y), np.abs(f + y)  # model, no change, model if flipped
    blocks = pd.factorize(org["day"].to_numpy() // max(int(block), 1))[0]
    half = np.random.default_rng(seed).choice([-1.0, 1.0], size=(flips // 2, blocks.max() + 1))
    s = np.vstack([half, -half])[:, blocks]  # (flips, forecasts)
    up = (s > 0).astype(float)
    err_flip = up @ em + (1 - up) @ ef  # (flips, lead): the path error under each draw
    base = en.sum(0)
    fe, ye = f[:, -1], y[:, -1]
    call = np.where(np.abs(fe) < TIE_BP, 0.0, fe)  # the direction the median calls (0: none)
    agree = np.sign(call) * np.sign(ye)
    used = agree != 0
    fc = fe - fe.mean()
    Y = s * ye
    Yc = Y - Y.mean(1, keepdims=True)
    corr_flip = (Yc @ fc) / (np.sqrt((Yc ** 2).sum(1)) * np.sqrt((fc ** 2).sum()) + 1e-12)
    flip_step = 1 - err_flip / base
    return {
        "n": len(org),
        "skill": _band(1 - em.sum() / base.sum(), 1 - err_flip.sum(1) / base.sum()),
        "hit": _band((agree[used] > 0).mean(), ((s[:, used] * agree[used]) > 0).mean(1)),
        "corr": _band(np.corrcoef(fe, ye)[0, 1], corr_flip),
        "skill_step": 1 - em.sum(0) / base,
        "skill_lo": np.quantile(flip_step, 0.025, axis=0),
        "skill_hi": np.quantile(flip_step, 0.975, axis=0),
        "inside": ((q[:, :, 0] <= act) & (act <= q[:, :, 2])).mean(0),
        "below": (act < q[:, :, 0]).mean(0), "above": (act > q[:, :, 2]).mean(0),
        "pred_move": float(np.median(np.abs(fe))), "act_move": float(np.median(np.abs(ye))),
        "band_width": float(np.median(1e4 * np.log(q[:, -1, 2] / q[:, -1, 0]))),
        "pred_up": float((call > 0).mean()), "act_up": float((ye > 0).mean()),
        "ratio": em.sum(1) / np.maximum(en.sum(1), 1e-9), "f_end": fe, "y_end": ye,
    }


def summary(scores: dict) -> pd.DataFrame:
    """The scorecard: one row per input setup, the luck band beside each statistic."""
    rows = []
    for name, sc in scores.items():
        sk, hit, cr = sc["skill"], sc["hit"], sc["corr"]
        rows.append({
            "reads": name, "forecasts": sc["n"],
            "path error saved vs no change": f"{100 * sk['value']:+.2f}%",
            "path luck band": f"{100 * sk['lo']:+.2f} … {100 * sk['hi']:+.2f}%",
            "z (path)": round(sk["z"], 2),
            "direction right": f"{100 * hit['value']:.1f}%",
            "direction luck band": f"{100 * hit['lo']:.1f} … {100 * hit['hi']:.1f}%",
            "z (direction)": round(hit["z"], 2),
            "end-move corr": f"{cr['value']:+.3f}",
            "corr luck band": f"{cr['lo']:+.3f} … {cr['hi']:+.3f}",
            "inside 10-90% band": f"{100 * sc['inside'].mean():.1f}%",
            "below 10% / above 90%": f"{100 * sc['below'].mean():.1f}% / "
                                     f"{100 * sc['above'].mean():.1f}%",
            "end move, predicted / actual (median bp)": f"{sc['pred_move']:.0f} / "
                                                         f"{sc['act_move']:.0f}",
            "band width at the end (bp)": f"{sc['band_width']:.0f}",
            "says up / went up": f"{100 * sc['pred_up']:.0f}% / {100 * sc['act_up']:.0f}%",
        })
    return pd.DataFrame(rows).set_index("reads")


def verdict(scores: dict) -> str:
    """The scorecard in one markdown line per setup: path error and direction against their
    luck bands, and whether the 10-90 % band is honest (75-85 % inside reads "about right")."""
    def where(b):
        return ("above its luck band" if b["value"] > b["hi"] else
                "below its luck band" if b["value"] < b["lo"] else "inside its luck band")

    lines = []
    for name, sc in scores.items():
        sk, hit, cover = sc["skill"], sc["hit"], 100 * sc["inside"].mean()
        band = ("about right" if 75 <= cover <= 85 else
                "too narrow: overconfident" if cover < 75 else "too wide: underconfident")
        lines.append(
            f"- **{name}**: {'beats' if sk['value'] > 0 else 'does not beat'} \"no change\" "
            f"({100 * sk['value']:+.2f}% of the path error, {where(sk)}); direction right "
            f"{100 * hit['value']:.1f}% ({where(hit)}); {cover:.0f}% of the closes inside the "
            f"10-90% band ({band}).")
    return "\n".join(lines)


def by_stock(B: dict, org: pd.DataFrame, Q: dict, *, block: int = 1) -> pd.DataFrame:
    """The scorecard's main numbers per stock and input setup (z against each stock's own luck
    band: a few hundred forecasts each, so expect wide bands)."""
    rows = []
    for sym in B["symbols"]:
        m = (org["symbol"] == sym).to_numpy()
        if m.sum() < 10:
            continue
        for name, q in Q.items():
            sc = score(B, org[m].reset_index(drop=True), q[m], block=block)
            rows.append({"stock": sym, "reads": name, "forecasts": int(m.sum()),
                         "path error saved": f"{100 * sc['skill']['value']:+.2f}%",
                         "z (path)": round(sc["skill"]["z"], 2),
                         "direction right": f"{100 * sc['hit']['value']:.1f}%",
                         "z (direction)": round(sc["hit"]["z"], 2),
                         "end-move corr": f"{sc['corr']['value']:+.3f}",
                         "inside 10-90% band": f"{100 * sc['inside'].mean():.1f}%"})
    return pd.DataFrame(rows)


def picks(sc: dict, at=(2.5, 20, 40, 60, 80, 97.5)) -> list[int]:
    """Forecasts spread over the path-error ranking (best first), for :func:`gallery`: an honest
    sample of how the forecasts look, not a hand-picked one."""
    r = sc["ratio"]
    order = np.argsort(r, kind="stable")
    return [int(order[min(len(r) - 1, int(round(p / 100 * (len(r) - 1))))]) for p in at]


# ---------------------------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------------------------
_STYLE = {SETUPS[0]: {"color": SERIES[0], "width": 2.2},
          SETUPS[1]: {"color": SERIES[1], "width": 2.0, "dash": "dash"}}


def _forecast_traces(B: dict, org: pd.DataFrame, Q: dict, i: int, before: int, legend: bool):
    """The traces of one forecast (band, medians, no change, actual price) and its time span."""
    t, c = int(org["t"].iat[i]), int(org["col"].iat[i])
    H = next(iter(Q.values())).shape[1]
    lo = max(0, t - before)
    x = charts._wall(B["time"][lo:t + H + 1])
    fx = charts._wall(B["time"][t:t + H + 1])  # the forecast starts at the origin's close
    p0 = float(B["close"][t, c])
    q0 = Q[SETUPS[0]][i]
    tr = [
        go.Scatter(x=fx, y=np.r_[p0, q0[:, 2]], mode="lines", line={"width": 0},
                   name="90% quantile", legendgroup="band", showlegend=False,
                   hovertemplate="%{y:$.2f}"),
        go.Scatter(x=fx, y=np.r_[p0, q0[:, 0]], mode="lines", line={"width": 0}, fill="tonexty",
                   fillcolor=_BAND, name="10-90% band (price only)", legendgroup="band",
                   showlegend=legend, hovertemplate="%{y:$.2f}"),
        go.Scatter(x=[fx[0], fx[-1]], y=[p0, p0], mode="lines", name="no change",
                   legendgroup="flat", showlegend=legend, hoverinfo="skip",
                   line={"color": _MUTED, "width": 1, "dash": "dot"}),
    ]
    for name, q in Q.items():
        tr.append(go.Scatter(x=fx, y=np.r_[p0, q[i][:, 1]], mode="lines",
                             name=f"median: {name}", legendgroup=name,
                             showlegend=legend, line=_STYLE[name], hovertemplate="%{y:$.2f}"))
    tr.append(go.Scatter(x=x, y=B["close"][lo:t + H + 1, c], mode="lines", name="actual",
                         legendgroup="actual", showlegend=legend,
                         line={"color": _INK, "width": 1.5}, hovertemplate="%{y:$.2f}"))
    return tr, B["time"][lo:t + H + 1], t


def _origin_line(fig: go.Figure, B: dict, t: int, *, row="all", col=1, label=True) -> None:
    """A dotted vertical line where the forecast is made (after the traces: a shape on a subplot
    without traces is dropped silently)."""
    x = charts._wall([B["time"][t]])[0]
    fig.add_vline(x=x, line={"color": _MUTED, "width": 1, "dash": "dot"}, row=row, col=col)
    if label:
        fig.add_annotation(x=x, y=1.0, yref="paper", text="forecast made", showarrow=False,
                           xanchor="right", yanchor="bottom", font={"color": _MUTED, "size": 11})


def forecast_figure(B: dict, org: pd.DataFrame, Q: dict, i: int, *, before: int,
                    title: str) -> go.Figure:
    """One forecast up close: the actual price (black) from ``before`` bars before the origin to
    the horizon's end, each setup's median path, the price-only 10-90 % band, the "no change"
    line, and the volume below (dark: read before the forecast; light: after it).

    Args:
        B: :func:`bars` output.
        org: :func:`origins` output.
        Q: ``{setup: forecast(...)}``.
        i: Row of ``org``.
        before: Context bars drawn (the model reads more).
        title: Figure title.

    Returns:
        The figure.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.77, 0.23],
                        vertical_spacing=0.04)
    traces, span, t = _forecast_traces(B, org, Q, i, before, legend=True)
    for tr in traces:
        fig.add_trace(tr, row=1, col=1)
    lo = t + 1 - len(span) + next(iter(Q.values())).shape[1]  # first bar drawn
    vol = B["volume"][lo:lo + len(span), int(org["col"].iat[i])]
    cut = t - lo + 1  # bars up to and including the origin: read before the forecast
    fig.add_trace(go.Bar(x=charts._wall(span[:cut]), y=vol[:cut], name="volume, before",
                         marker={"color": _MUTED}, showlegend=False,
                         hovertemplate="%{y:,.0f}"), row=2, col=1)
    fig.add_trace(go.Bar(x=charts._wall(span[cut:]), y=vol[cut:], name="volume, after",
                         marker={"color": _GRID}, showlegend=False,
                         hovertemplate="%{y:,.0f}"), row=2, col=1)
    _origin_line(fig, B, t)
    _base(fig, title=title, height=580)
    fig.update_layout(hovermode="x unified", bargap=0.1)
    fig.update_xaxes(type="date", rangebreaks=_intraday_breaks(span),
                     range=list(charts._wall([span[0], span[-1]])))
    fig.update_yaxes(title_text="price ($)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="volume", row=2, col=1)
    return charts.below(fig, -0.08)


def gallery(B: dict, org: pd.DataFrame, Q: dict, rows: list[int], sc: dict, *, before: int,
            title: str, cols: int = 3) -> go.Figure:
    """Several forecasts side by side (:func:`picks`: from the best to the worst path error),
    each titled with its stock, origin and path error against "no change".

    Args:
        B: :func:`bars` output.
        org: :func:`origins` output.
        Q: ``{setup: forecast(...)}``.
        rows: Rows of ``org``.
        sc: :func:`score` of the price-only setup (its ``ratio``).
        before: Context bars drawn per panel.
        title: Figure title.
        cols: Panels per row.

    Returns:
        The figure.
    """
    n_rows = -(-len(rows) // cols)
    names = [f"{org['symbol'].iat[i]} · {org['time'].iat[i]:%a %d %b %H:%M} · error "
             f"{sc['ratio'][i]:.2f}× no change" for i in rows]
    fig = make_subplots(rows=n_rows, cols=cols, subplot_titles=names, horizontal_spacing=0.05,
                        vertical_spacing=0.12)
    for k, i in enumerate(rows):
        r, cl = k // cols + 1, k % cols + 1
        traces, span, t = _forecast_traces(B, org, Q, i, before, legend=k == 0)
        for tr in traces:
            fig.add_trace(tr, row=r, col=cl)
        _origin_line(fig, B, t, row=r, col=cl, label=False)
        fig.update_xaxes(type="date", rangebreaks=_intraday_breaks(span), nticks=4,
                         range=list(charts._wall([span[0], span[-1]])), row=r, col=cl)
    _base(fig, title=title, height=330 * n_rows + 90)
    fig.update_annotations(font={"size": 11, "color": _INK})
    fig.update_yaxes(tickprefix="$")
    return charts.below(fig, -0.06)


def skill_figure(scores: dict, x: np.ndarray, *, xlabel: str, title: str) -> go.Figure:
    """Path error saved against "no change" by lead time, each setup with its luck band (the
    sign-flip draws' 2.5-97.5 %). Above zero the forecast beats "no change" at that lead."""
    fig = go.Figure()
    for name, sc in scores.items():
        color = _STYLE[name]["color"]
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]],
                                 y=100 * np.r_[sc["skill_hi"], sc["skill_lo"][::-1]],
                                 fill="toself", fillcolor=_rgba(color, 0.12), line={"width": 0},
                                 name=f"luck band ({name})", legendgroup=name, showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=100 * sc["skill_step"], mode="lines", name=name,
                                 legendgroup=name, line=_STYLE[name], hovertemplate="%{y:+.2f}%"))
    fig.add_hline(y=0, line={"color": _INK, "width": 1},
                  annotation_text="no change", annotation_position="bottom right")
    _base(fig, title=title, height=420)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="path error saved vs no change (%)", ticksuffix="%")
    return charts.below(fig, -0.18)


def coverage_figure(scores: dict, x: np.ndarray, *, xlabel: str, title: str) -> go.Figure:
    """Calibration by lead time: left, the share of actual closes inside the 10-90 % band (80 %
    if the band is honest); right, below the 10 % line (solid) and above the 90 % line (dotted),
    10 % each if calibrated. Too few inside = the model is overconfident."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=["inside the 10-90% band",
                                        "below the 10% line (solid) / above the 90% line (dotted)"])
    for name, sc in scores.items():
        color = _STYLE[name]["color"]
        fig.add_trace(go.Scatter(x=x, y=100 * sc["inside"], mode="lines", name=name,
                                 legendgroup=name, line=_STYLE[name],
                                 hovertemplate="%{y:.1f}%"), row=1, col=1)
        for key, dash in (("below", None), ("above", "dot")):
            fig.add_trace(go.Scatter(x=x, y=100 * sc[key], mode="lines", legendgroup=name,
                                     name=f"{name}, {key}", showlegend=False,
                                     line={"color": color, "width": 1.8, "dash": dash},
                                     hovertemplate="%{y:.1f}%"), row=1, col=2)
    for col, level in ((1, 80), (2, 10)):
        fig.add_hline(y=level, line={"color": _INK, "width": 1, "dash": "dash"}, row=1, col=col,
                      annotation_text=f"calibrated: {level}%", annotation_position="top right")
    _base(fig, title=title, height=420)
    fig.update_layout(hovermode="x unified")
    fig.update_annotations(font={"size": 11})
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(ticksuffix="%")
    return charts.below(fig, -0.2)


def end_scatter(scores: dict, org: pd.DataFrame, symbols: list[str], *, title: str) -> go.Figure:
    """Each forecast's predicted move to the horizon's end (median, x) against the actual move
    (y), one panel per setup, one color per stock. Points in the upper-right and lower-left
    quadrants got the direction right; the axes differ in scale (the median moves far less)."""
    fig = make_subplots(rows=1, cols=len(scores), horizontal_spacing=0.08,
                        subplot_titles=[f"{name}: direction right "
                                        f"{100 * sc['hit']['value']:.1f}%, corr "
                                        f"{sc['corr']['value']:+.3f}"
                                        for name, sc in scores.items()])
    sym = org["symbol"].to_numpy()
    for k, sc in enumerate(scores.values()):
        for j, s in enumerate(symbols):
            m = sym == s
            cls = go.Scattergl if m.sum() > 2000 else go.Scatter
            fig.add_trace(cls(x=sc["f_end"][m], y=sc["y_end"][m], mode="markers", name=s,
                              legendgroup=s, showlegend=k == 0,
                              marker={"color": SERIES[j % len(SERIES)], "size": 4,
                                      "opacity": 0.55},
                              hovertemplate=f"{s}<br>predicted %{{x:+.0f}} bp<br>"
                                            "actual %{y:+.0f} bp<extra></extra>"),
                          row=1, col=k + 1)
        fig.add_hline(y=0, line={"color": _MUTED, "width": 1}, row=1, col=k + 1)
        fig.add_vline(x=0, line={"color": _MUTED, "width": 1}, row=1, col=k + 1)
    _base(fig, title=title, height=470)
    fig.update_annotations(font={"size": 12})
    fig.update_xaxes(title_text="predicted move to the end (bp)")
    fig.update_yaxes(title_text="actual move (bp)", col=1)
    return charts.below(fig, -0.16)


def _rgba(hex_color: str, alpha: float) -> str:
    """A ``#rrggbb`` color with an alpha, for fills."""
    r, g, b = (int(hex_color[k:k + 2], 16) for k in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


# ---------------------------------------------------------------------------------------------
# Size: does TimesFM's band know how big the move will be, better than a cheap volatility band?
# ---------------------------------------------------------------------------------------------
#: The standard normal's 10 / 50 / 90 % quantiles: the cheap bands are Gaussian in log price.
Z = np.array([-1.2815515655446004, 0.0, 1.2815515655446004])
REFERENCE = "clock × recent"  # the cheap band TimesFM is compared with (declared before the run)
PRIMARY = f"TimesFM band, centred ({SETUPS[0]})"  # the TimesFM row the verdict reads
_SIZE_STYLE = {
    f"TimesFM as is ({SETUPS[0]})": {"color": SERIES[0], "width": 1.6, "dash": "dot"},
    PRIMARY: {"color": SERIES[0], "width": 2.4},
    f"TimesFM band, centred ({SETUPS[1]})": {"color": SERIES[1], "width": 2.0, "dash": "dash"},
    "clock profile": {"color": SERIES[2], "width": 2.0},
    REFERENCE: {"color": SERIES[4], "width": 2.0},
}


def _slots(B: dict) -> np.ndarray:
    """Each bar's position in its session (0: the first bar, whose return holds the night)."""
    day = B["day"]
    first = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    return np.arange(len(day)) - first[day]


def vol_bands(B: dict, org: pd.DataFrame, horizon: int, *, level: int, days: int = 20) -> dict:
    """Two cheap competitors for TimesFM's band: Gaussian 10 / 50 / 90 % quantiles of the log move
    from the origin's close, centred on "no change", every future bar's variance taken from
    realized volatility. Nothing is fitted or searched; every number is known at the origin.

    * ``clock profile``: the mean squared return of the same bar of the session (the same clock
      time; a session's first bar holds the night) over the ``days`` sessions before the origin's
      session (at least half of them present);
    * ``clock × recent``: that profile scaled by how busy the stock is now: the realized variance
      of the last ``level`` bars over what the profile expects for them (each session's first
      bar, the night, left out of both sums).

    Args:
        B: :func:`bars` output.
        org: :func:`origins` output.
        horizon: Bars forecast.
        level: Bars behind "recent".
        days: Sessions behind the clock profile.

    Returns:
        ``{"clock profile": (N, H, 3), "clock × recent": (N, H, 3)}`` quantiles of the move in
        bp, and ``"ok"``: (N,) origins with enough history for both.
    """
    C = B["close"]
    r2 = np.full(C.shape, np.nan)
    r2[1:] = np.log(C[1:] / C[:-1]) ** 2
    day, slot = B["day"], _slots(B)
    nd, ns, S = day.max() + 1, slot.max() + 1, C.shape[1]
    grid = np.full((nd, ns, S), np.nan)
    grid[day, slot] = r2
    have = np.isfinite(grid)
    cs = np.concatenate([np.zeros((1, ns, S)), np.cumsum(np.where(have, grid, 0.0), 0)])
    cn = np.concatenate([np.zeros((1, ns, S)), np.cumsum(have, 0)])
    d, lo = np.arange(nd), np.maximum(np.arange(nd) - days, 0)
    cnt = cn[d] - cn[lo]  # sessions [d - days, d): never the origin's own session
    prof = np.where(cnt >= days // 2, (cs[d] - cs[lo]) / np.maximum(cnt, 1), np.nan)
    t, c, d0 = org["t"].to_numpy(), org["col"].to_numpy(), org["day"].to_numpy()
    fut = t[:, None] + np.arange(1, horizon + 1)
    sd = np.sqrt(np.cumsum(prof[d0[:, None], slot[fut], c[:, None]], 1))
    past = t[:, None] - np.arange(level)  # the last `level` bars, the origin's included
    day_bar = slot[past] > 0
    real = np.where(day_bar, r2[past, c[:, None]], 0.0).sum(1)
    expect = np.where(day_bar, prof[d0[:, None], slot[past], c[:, None]], 0.0).sum(1)
    ok = np.isfinite(sd[:, -1]) & np.isfinite(real) & np.isfinite(expect) & (expect > 0)
    clock = 1e4 * Z * sd[..., None]
    ratio = np.where(ok, real / np.where(expect > 0, expect, 1.0), np.nan)
    return {"clock profile": clock, REFERENCE: clock * np.sqrt(ratio)[:, None, None], "ok": ok}


def _pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """The quantile (pinball) loss of quantile ``q`` at level ``tau`` for outcome ``y``."""
    u = y - q
    return u * (tau - (u < 0))


def _cluster_se(d: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Standard error of the mean of ``d`` (rows = forecasts, any trailing shape) with the rows
    of one block (sessions sharing news, or overlapping horizons) treated as one draw."""
    nb = blocks.max() + 1
    onehot = np.zeros((nb, len(d)))
    onehot[blocks, np.arange(len(d))] = 1.0
    resid = onehot @ d - onehot.sum(1).reshape(-1, *[1] * (d.ndim - 1)) * d.mean(0)
    return np.sqrt(nb / (nb - 1) * (resid ** 2).sum(0)) / len(d)


def size_compare(B: dict, org: pd.DataFrame, Q: dict, bands: dict, *, block: int = 1) -> dict:
    """TimesFM's 10-90 % band against the cheap bands of :func:`vol_bands`, on the same forecasts
    (those with enough history for the cheap bands), scored by the band loss: the pinball loss of
    the 10 % plus the 90 % quantile, in bp (lower is better; it rewards a band that is narrow
    where the move is small and wide where it is large, and charges every close outside it).

    TimesFM is scored as is and with its band CENTRED on "no change" (each quantile divided by its
    median): the centred band keeps what TimesFM knows about the SIZE of the move and drops the
    wandering median (section 1: no direction), as the cheap bands have none. Skill is the share
    of :data:`REFERENCE`'s loss saved; z is the paired difference over its clustered standard
    error (one block per ``block`` sessions, shared by every stock).

    Args:
        B: :func:`bars` output.
        org: :func:`origins` output.
        Q: ``{setup: forecast(...)}``.
        bands: :func:`vol_bands` output for the same origins.
        block: Sessions per cluster.

    Returns:
        ``table`` (one row per forecaster), ``lead`` (per-lead skill, its standard error and the
        share inside the band), ``deciles`` (predicted against realized size), ``stocks`` (skill
        per stock), ``n`` and ``dropped``.
    """
    ok = bands["ok"]
    t, c = org["t"].to_numpy()[ok], org["col"].to_numpy()[ok]
    H = bands[REFERENCE].shape[1]
    p0 = B["close"][t, c]
    y = 1e4 * np.log(B["close"][t[:, None] + np.arange(1, H + 1), c[:, None]] / p0[:, None])
    fc = {f"TimesFM as is ({SETUPS[0]})": 1e4 * np.log(Q[SETUPS[0]][ok] / p0[:, None, None])}
    for name, q in Q.items():
        fc[f"TimesFM band, centred ({name})"] = 1e4 * np.log(q[ok] / q[ok][:, :, 1:2])
    fc.update({k: v[ok] for k, v in bands.items() if k != "ok"})
    blocks = pd.factorize(org["day"].to_numpy()[ok] // max(int(block), 1))[0]
    loss = {k: _pinball(y, v[..., 0], 0.1) + _pinball(y, v[..., 2], 0.9) for k, v in fc.items()}
    ref = loss[REFERENCE]
    sym = org["symbol"].to_numpy()[ok]
    rows, lead, dec, stocks = [], {}, [], []
    for k, v in fc.items():
        gain = ref - loss[k]  # positive: k's band loss is lower
        g = gain.mean(1)
        se = _cluster_se(g, blocks)
        sd_end = (v[:, -1, 2] - v[:, -1, 0]) / (2 * Z[2])
        inside = (v[..., 0] <= y) & (y <= v[..., 2])
        rows.append({
            "band": k, "band loss (bp)": round(float(loss[k].mean()), 2),
            f"saved vs {REFERENCE}": f"{100 * g.mean() / ref.mean():+.2f}%",
            "z": round(float(g.mean() / se), 2) if se > 0 else 0.0,
            "inside 10-90%": f"{100 * inside.mean():.1f}%",
            "inside at the 1st bar": f"{100 * inside[:, 0].mean():.1f}%",
            "width at the end (median bp)": round(float(np.median(v[:, -1, 2] - v[:, -1, 0]))),
            "width ↔ |move| (rank corr)": round(float(
                pd.Series(sd_end).corr(pd.Series(np.abs(y[:, -1])), method="spearman")), 3),
        })
        lead[k] = {"skill": 1 - loss[k].mean(0) / ref.mean(0),
                   "se": _cluster_se(gain, blocks) / ref.mean(0), "inside": inside.mean(0)}
        bins = pd.qcut(sd_end, 10, labels=False, duplicates="drop")
        for b in np.unique(bins) if not k.startswith("TimesFM as is") else ():  # = centred width
            m = bins == b
            dec.append({"band": k, "decile": int(b) + 1, "predicted": float(sd_end[m].mean()),
                        "realized": float(np.sqrt(np.mean(y[m, -1] ** 2)))})
        for s in np.unique(sym):
            m = sym == s
            se_s = _cluster_se(g[m], pd.factorize(blocks[m])[0])
            stocks.append({"stock": s, "band": k, "forecasts": int(m.sum()),
                           f"saved vs {REFERENCE}": f"{100 * g[m].mean() / ref[m].mean():+.2f}%",
                           "z": round(float(g[m].mean() / se_s), 2) if se_s > 0 else 0.0,
                           "inside 10-90%": f"{100 * inside[m].mean():.1f}%"})
    return {"table": pd.DataFrame(rows).set_index("band"), "lead": lead,
            "deciles": pd.DataFrame(dec), "stocks": pd.DataFrame(stocks), "n": int(ok.sum()),
            "dropped": int((~ok).sum())}


def size_verdict(cmp: dict) -> str:
    """The size test in one markdown line: the pre-declared comparison (:data:`PRIMARY` against
    :data:`REFERENCE`) and whether it clears z > 2."""
    row = cmp["table"].loc[PRIMARY]
    saved, z = row[f"saved vs {REFERENCE}"], row["z"]
    word = ("**beats**" if z > 2 else "**loses to**" if z < -2 else "**cannot be told from**")
    return (f"TimesFM's band (centred, {SETUPS[0]}) {word} the cheap *{REFERENCE}* band: it saves "
            f"{saved} of its band loss, z {z:+.2f}, on {cmp['n']:,} forecasts "
            f"({cmp['dropped']:,} dropped: under 10 sessions of history for the cheap bands).")


def size_figure(cmp: dict, x: np.ndarray, *, xlabel: str, title: str) -> go.Figure:
    """By lead time: left, each band's loss saved against :data:`REFERENCE` (above 0: better),
    the TimesFM band's ±1.96 clustered standard errors shaded; right, the share of closes inside
    each 10-90 % band (80 % if honest)."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, column_widths=[0.55, 0.45],
                        subplot_titles=[f"band loss saved vs {REFERENCE}",
                                        "closes inside the 10-90% band"])
    p = cmp["lead"][PRIMARY]
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]],
                             y=100 * np.r_[p["skill"] + 1.96 * p["se"],
                                           (p["skill"] - 1.96 * p["se"])[::-1]],
                             fill="toself", fillcolor=_rgba(SERIES[0], 0.14), line={"width": 0},
                             name="±1.96 s.e.", legendgroup=PRIMARY, showlegend=False,
                             hoverinfo="skip"), row=1, col=1)
    for k, v in cmp["lead"].items():
        style = _SIZE_STYLE[k]
        fig.add_trace(go.Scatter(x=x, y=100 * v["skill"], mode="lines", name=k, legendgroup=k,
                                 line=style, hovertemplate="%{y:+.1f}%"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=100 * v["inside"], mode="lines", name=k, legendgroup=k,
                                 showlegend=False, line=style, hovertemplate="%{y:.1f}%"),
                      row=1, col=2)
    fig.add_hline(y=0, line={"color": _INK, "width": 1}, row=1, col=1)
    fig.add_hline(y=80, line={"color": _INK, "width": 1, "dash": "dash"}, row=1, col=2,
                  annotation_text="calibrated: 80%", annotation_position="bottom right")
    _base(fig, title=title, height=460)
    fig.update_layout(hovermode="x unified")
    fig.update_annotations(font={"size": 11})
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(ticksuffix="%")
    return charts.below(fig, -0.2)


def deciles_figure(cmp: dict, *, title: str) -> go.Figure:
    """Does a band know WHICH moves will be big? The forecasts cut into ten groups by the size
    each band predicts for the horizon's end (its 10-90 % width as a standard deviation); x = the
    group's predicted size, y = the size the moves really had (root mean square). On the diagonal:
    right on average; a steeper line across the groups: it tells big moves from small ones."""
    fig = go.Figure()
    dec = cmp["deciles"]
    top = float(dec[["predicted", "realized"]].to_numpy().max()) * 1.05
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="predicted = realized",
                             line={"color": _MUTED, "width": 1, "dash": "dash"},
                             hoverinfo="skip"))
    for k, part in dec.groupby("band", sort=False):
        fig.add_trace(go.Scatter(x=part["predicted"], y=part["realized"], mode="lines+markers",
                                 name=k, line=_SIZE_STYLE[k], marker={"size": 6},
                                 hovertemplate="decile %{text}<br>predicted %{x:.0f} bp<br>"
                                               "realized %{y:.0f} bp<extra></extra>",
                                 text=part["decile"]))
    _base(fig, title=title, height=480)
    fig.update_xaxes(title_text="predicted size of the end move (bp, one standard deviation)",
                     range=[0, top])
    fig.update_yaxes(title_text="realized size (bp, root mean square)", range=[0, top])
    return charts.below(fig, -0.16)
