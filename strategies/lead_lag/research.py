"""lead_lag research helpers: everything the exploration notebook computes, no figures.

There is no ``strategy.py`` yet: this folder is a go/no-go study of whether one stock's move
predicts another's LATER move (README). It is loaded by path from ``lead_lag_explore.ipynb`` (the
burst_scan pattern) and never through ``strategies.load``, so nothing here can reach a backtest.

Everything works on a **cube**: one float32 array per quantity shaped ``(session, bar, symbol)``,
so a lag can never reach into the previous session. A daily panel is ONE session whose bars are
days — its lags cross days, as they should.

* **Data** — :func:`mask_placeholders`, :func:`cube`, :func:`load_intraday`. The leader signal
  ``x`` is the last-trade close-to-close return; the follower target ``y`` is the bar's own
  first-to-last print, ``log(close / open)`` — what a trade entered at the first print after a
  signal earns. A thin name's stale close catches up in the gap BETWEEN bars, which ``y``
  excludes; a close-to-close map is largely a map of who prints rarely.
* **Units** — :func:`scale` (causal: trailing RMS × time-of-day profile), :func:`lagged`.
* **The map** — :func:`pair_t`: a session- (or week-) clustered t for every leader × follower at
  once, one float32 matmul per cluster.
* **Controls** — :func:`granger` (per-name OLS innovations), :func:`portfolios` (group columns).
* **Tradability** — :func:`forward` (entry at the next open, never past the close), :func:`spread`.
* **Reporting** — :func:`day_t`, :func:`checks_table`, :func:`scorecard` (burst_scan's, copied:
  that folder is not committed).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from trading_simple.data import Panel, load_panel, resample_panel

#: The windows read — ALL available data (user decision, 2026-09-28). ``[start, split)`` is
#: discovery, ``[split, end]`` validation; ``size`` is the half-year whose median dollar volume
#: ranks names point-in-time (strictly before ``start``).
WINDOWS = {
    "intraday": {"start": "2025-06-18", "split": "2026-01-01", "end": "2026-06-26",
                 "size": ("2025-01-01", "2025-06-17")},
    "daily": {"start": "2020-07-01", "split": "2023-07-01", "end": "2026-06-12",
              "size": ("2020-01-01", "2020-06-30"), "long_run": "2015-01-01"},
}
#: The first bar a later strategy may treat as out-of-sample: this study read everything before.
FIRST_OOS = {"intraday": "2026-06-29", "daily": "2026-06-15"}
BARS = ("1m", "5m", "1d")
#: Per bar size: ``L`` the primary lag window (bars), ``K`` the market lag window, ``hold`` the
#: tradability horizon (bars), ``skip_open`` minutes whose target is blanked, ``days`` the trailing
#: scale window (sessions; days at 1d), ``rolls`` null maps and ``min_roll`` their minimum shift
#: (sessions; days at 1d).
SPEC = {
    "1m": {"L": 5, "K": 15, "hold": 5, "skip_open": 5, "days": 5, "rolls": 20, "min_roll": 5},
    "5m": {"L": 3, "K": 12, "hold": 3, "skip_open": 5, "days": 5, "rolls": 20, "min_roll": 5},
    "1d": {"L": 1, "K": 5, "hold": 1, "skip_open": 0, "days": 60, "rolls": 10, "min_roll": 63},
}
#: The slow daily map's lags: supply-chain drift is a monthly effect (Cohen-Frazzini 2008).
SLOW = (2, 22)
#: Mega caps with 1m bars only from May/June 2026: a pre-registered leader GROUP, never screened.
MEGA = ("AAPL", "MSFT", "NVDA", "TSLA")
#: |T| a pair must reach to count as a discovery.
T_CUT = 4.5
#: Pre-registered pass bars (README "Pre-registration"); the notebook's CHECKS read only these.
THRESH = {
    "null_std": (0.85, 1.20),   # V1: std of the rolled maps' T
    "null_tail": 3.4e-5,        # V1: share of rolled |T| >= T_CUT (5x the normal tail)
    "canary": 16,               # V2: of 20 leads planted at the break-even correlation
    "market_t": 3.0,            # C1: pooled Granger t, fast market -> the rest
    "fdr": 0.20,                # C2: mean rolled count / real count
    "busiest": 0.25,            # C2: share of discoveries owned by the busiest follower
    "lag_rho": 0.30,            # C3 market: Spearman of per-name lag t, discovery vs validation
    "val_t": 2.0,               # C3 market: pooled Granger t on validation
    "top_val_t": 2.0,           # C3 pairs: mean signed validation T of the top-100
    "top_sign": 0.65,           # C3 pairs: share of the top-100 keeping their sign
    "size_rho": -0.20,          # C4: Spearman(lag t, log market cap) at most this
    "size_t": -3.0,             # C4: its rank-regression t given stale share and price
    "big_small": 3.0,           # C4: T(big quintile -> small) at least
    "small_big": 1.0,           # C4: T(small -> big) at most
    "industry_pct": 0.99,       # C5a: same-industry minus cross-industry mean T vs permutations
    "hou_t": 2.0,               # C5b: big->small minus small->big inside industries, t
    "link_pct": 0.60,           # C5c: linked leaders' mean percentile within their follower
    "link_p": 0.005,            # C5c: its random-leader p
    "trade_t": 2.0,             # C6: net edge per trade, cluster t at delay 1
}
#: E[|Z| given |Z| > 2] for a standard normal: the mean signal size of a trade taken at |z| > 2.
Z_TRADE = 2.373


# ---------------------------------------------------------------------------
# Data.
# ---------------------------------------------------------------------------
def mask_placeholders(panel: Panel) -> Panel:
    """Blank the bars without a trade before any candle is derived from them.

    IBKR stores a minute with fewer than 100 shares traded as a flat placeholder at the previous
    close (volume 0; every one of 2,230 such minutes in a 2025-10 sample). Left in, it makes a thin
    name's 5-minute open the stale price of a minute nobody traded; blanked, ``resample_panel``'s
    first/last skip it and every candle opens and closes on real prints.

    Args:
        panel: A price panel (any bar).

    Returns:
        A copy with open/high/low/close/volume NaN wherever volume is not positive.
    """
    live = panel.volume > 0
    return replace(panel, close=panel.close.where(live), open_=panel.open_.where(live),
                   high=panel.high.where(live), low=panel.low.where(live),
                   volume=panel.volume.where(live))


def cube(panel: Panel, *, symbols: list[str] | None = None, skip_open: int = 0) -> dict:
    """A panel as (session, bar, symbol) float32 arrays: leader signal, target and gap.

    Bars are placed by their clock position in the session (time since the session's first
    stored bar ÷ bar length), so every session shares one clock and a half day is NaN-padded.
    A daily panel becomes one session whose bars are its days.

    Args:
        panel: Price panel (1m, a derived ``<n>m`` or 1d); ``volume > 0`` marks a print.
        symbols: Column order (names without data become all-NaN columns); default the panel's.
        skip_open: Minutes after each session's first bar whose target ``y`` is blanked (the
            opening auction's minutes); ignored for daily bars.

    Returns:
        ``x`` last-trade log return (NaN on a session's first bar and before a name's first
        print, 0 on a bar without a print); ``y`` log(close/open) of bars with a print; ``gap``
        log(open / previous last trade) of bars with a print (daily: the overnight move);
        ``present`` (session, bar) bars that exist; ``stamps`` their naive-UTC timestamps (NaT
        padding); ``prints``/``slots`` per symbol and ``stale = 1 - prints / slots`` (intraday:
        present bars of the sessions the name printed in; daily: bars after its first print);
        ``price`` median close; ``symbols``; ``bar``.
    """
    syms = list(symbols) if symbols is not None else panel.symbols
    index = panel.close.index
    c, o, v = (f.reindex(columns=syms).to_numpy(np.float64)
               for f in (panel.close, panel.open_, panel.volume))
    with np.errstate(invalid="ignore"):
        printed = (v > 0) & (c > 0) & (o > 0)     # a zero or missing price is no print
    daily = panel.bar == "1d"
    if daily:
        codes, pos = np.zeros(len(index), int), np.arange(len(index))
    else:
        codes = pd.factorize(pd.Series(index.tz_convert(panel.tz).date), sort=True)[0]
        ns = index.asi8
        first = pd.Series(ns).groupby(codes).transform("min").to_numpy()
        pos = (ns - first) // (int(panel.bar[:-1]) * 60 * 10**9)
    shape = (codes.max() + 1, pos.max() + 1, len(syms))
    grid = np.full(shape, np.nan)
    C, Op, P = grid.copy(), grid.copy(), grid
    C[codes, pos] = np.where(printed, c, np.nan)
    Op[codes, pos] = np.where(printed, o, np.nan)
    P[codes, pos] = printed
    present = np.zeros(shape[:2], bool)
    present[codes, pos] = True
    stamps = np.full(shape[:2], np.datetime64("NaT"), "datetime64[ns]")
    stamps[codes, pos] = index.tz_convert("UTC").tz_localize(None).to_numpy()
    # Last trade, carried forward inside the session only; padding stays NaN.
    last = np.where(np.isfinite(C), np.arange(shape[1])[None, :, None], 0)
    np.maximum.accumulate(last, axis=1, out=last)
    C_ff = np.take_along_axis(C, last, axis=1)
    C_ff[~present] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.full(shape, np.nan)
        x[:, 1:] = np.log(C_ff[:, 1:] / C_ff[:, :-1])
        y = np.log(C / Op)
        gap = np.full(shape, np.nan)
        gap[:, 1:] = np.log(Op[:, 1:] / C_ff[:, :-1])
    if not daily and skip_open:
        y[:, :math.ceil(skip_open / int(panel.bar[:-1]))] = np.nan
    hit = np.nan_to_num(P)
    if daily:
        slots = np.isfinite(C_ff).sum(axis=(0, 1))
    else:
        active = hit.sum(axis=1) > 0                                     # (session, symbol)
        slots = (active * present.sum(axis=1)[:, None]).sum(axis=0)
    prints = hit.sum(axis=(0, 1))
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns
        stale = np.where(slots > 0, 1.0 - prints / slots, np.nan)
        price = np.nanmedian(c, axis=0)
    f32 = lambda a: a.astype(np.float32)  # noqa: E731
    return {"x": f32(x), "y": f32(y), "gap": f32(gap), "present": present, "stamps": stamps,
            "prints": prints, "slots": slots, "stale": stale, "price": price,
            "symbols": syms, "bar": panel.bar}


def _concat(parts: list[dict]) -> dict:
    """Stack cubes of consecutive months along the session axis (bars padded to the widest)."""
    width = max(p["x"].shape[1] for p in parts)

    def stack(key: str, fill):
        out = []
        for p in parts:
            a = p[key]
            pad = np.full((a.shape[0], width - a.shape[1], *a.shape[2:]), fill, a.dtype)
            out.append(np.concatenate([a, pad], axis=1))
        return np.concatenate(out)

    out = {k: stack(k, np.nan) for k in ("x", "y", "gap")}
    out["present"] = stack("present", False)
    out["stamps"] = stack("stamps", np.datetime64("NaT"))
    out["prints"] = sum(p["prints"] for p in parts)
    out["slots"] = sum(p["slots"] for p in parts)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN columns
        out["stale"] = np.where(out["slots"] > 0, 1.0 - out["prints"] / out["slots"], np.nan)
        out["price"] = np.nanmedian(np.stack([p["price"] for p in parts]), axis=0)
    out["symbols"], out["bar"] = parts[0]["symbols"], parts[0]["bar"]
    return out


def load_intraday(engine, symbols: list[str], start, end, *, minutes=(5,), skip_open: int = 0,
                  exchanges: dict[str, str] | None = None) -> dict[str, dict]:
    """Regular-session 1m bars month by month, as cubes at 1m and at each derived bar size.

    One month of every name at a time keeps the peak under ~1 GB, where one ``load_panel`` of
    the whole window holds several (float64, pivoted six times). Placeholders are masked BEFORE
    resampling (:func:`mask_placeholders`); months split at session boundaries, so nothing is
    lost at the seams.

    Args:
        engine: Database engine.
        symbols: Column order of every cube.
        start: First session (date).
        end: Last session (date); its whole session is included.
        minutes: Derived bar sizes (each > 1), e.g. ``(5,)``.
        skip_open: Minutes of each session whose target is blanked (see :func:`cube`).
        exchanges: Per-symbol venue overrides (``settings.exchanges``).

    Returns:
        ``{"1m": cube, "5m": cube, ...}``.
    """
    first = pd.Timestamp(start, tz="UTC")
    stop = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    months = pd.date_range(first.normalize().replace(day=1), stop, freq="MS")
    bounds = [first, *months[months > first], stop]
    parts: dict[int, list] = {1: [], **{int(m): [] for m in minutes}}
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        try:
            p = load_panel(engine, list(symbols), "1m", a, b - pd.Timedelta(seconds=1),
                           exchanges=exchanges)
        except ValueError:                                # no bars in this slice
            continue
        p = mask_placeholders(p)
        for m, out in parts.items():
            out.append(cube(p if m == 1 else resample_panel(p, m), symbols=list(symbols),
                            skip_open=skip_open))
        del p
    return {f"{m}m": _concat(out) for m, out in parts.items()}


def rows(c: dict, start, end=None) -> slice:
    """The contiguous flattened rows of the sessions (daily: days) in ``[start, end)``.

    Args:
        c: A cube.
        start: First date (inclusive).
        end: End date (exclusive); None = to the end.

    Returns:
        A slice over ``c[...].reshape(-1, n_symbols)`` rows.
    """
    width = c["x"].shape[1]
    daily = c["x"].shape[0] == 1
    when = pd.DatetimeIndex(c["stamps"][0] if daily else c["stamps"][:, 0])
    lo = int(when.searchsorted(pd.Timestamp(start)))
    hi = len(when) if end is None else int(when.searchsorted(pd.Timestamp(end)))
    return slice(lo, hi) if daily else slice(lo * width, hi * width)


def clusters(c: dict, by: str = "session") -> np.ndarray:
    """Cluster id of every flattened row: sessions intraday; ISO weeks or months at 1d.

    Args:
        c: A cube.
        by: ``"session"`` (intraday), ``"week"`` or ``"month"`` (daily).

    Returns:
        Non-decreasing integer ids, one per row.
    """
    n_s, width = c["x"].shape[:2]
    if n_s > 1:
        return np.repeat(np.arange(n_s), width)
    when = pd.DatetimeIndex(c["stamps"][0])
    if by == "month":
        key = when.year * 12 + when.month
    else:
        iso = when.isocalendar()
        key = iso["year"].to_numpy() * 100 + iso["week"].to_numpy()
    return pd.factorize(np.asarray(key), sort=True)[0]


# ---------------------------------------------------------------------------
# Units.
# ---------------------------------------------------------------------------
def scale(r: np.ndarray, *, days: int, fit=None, clip: float = 5.0) -> np.ndarray:
    """Returns in causal volatility units: ``r / (trailing RMS × time-of-day profile)``, clipped.

    The trailing RMS is each name's root-mean-square return over the previous ``days``
    sessions — the current one excluded, so a value never depends on its own session. The
    profile is the pooled RMS of those units at each bar of the session over the ``fit``
    sessions: the open's minutes move several times more than midday's, and unscaled they would
    carry the whole map. A daily cube (one session) scales by the trailing ``days`` days.

    Args:
        r: (session, bar, symbol) returns; NaN = no value.
        days: Trailing window in sessions (days at 1d).
        fit: Index (slice or boolean mask) of the sessions the profile is fitted on; None = all.
        clip: Absolute cap in the resulting units.

    Returns:
        float32 array shaped like ``r``; NaN in the first ``days`` sessions (warm-up).
    """
    if r.shape[0] == 1:                               # daily: every day is a one-bar session
        return scale(r.transpose(1, 0, 2), days=days, clip=clip).transpose(1, 0, 2)
    ok = np.isfinite(r)
    r = np.where(ok, r, np.nan)
    ss = np.where(ok, np.square(r, dtype=np.float64), 0.0).sum(axis=1)       # (session, symbol)
    cs = np.vstack([np.zeros((1, ss.shape[1])), np.cumsum(ss, axis=0)])
    cn = np.vstack([np.zeros((1, ss.shape[1])), np.cumsum(ok.sum(axis=1), axis=0)])
    ms = np.full(ss.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ms[days:] = (cs[days:-1] - cs[:-1 - days]) / (cn[days:-1] - cn[:-1 - days])
        ms[~(ms > 0)] = np.nan              # a name that did not move for `days` has no scale
        z = r / np.sqrt(ms)[:, None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # One wild value (a name waking up after a quiet spell) must not set the profile.
            sample = np.clip(z if fit is None else z[fit], -4 * clip, 4 * clip)
            prof = np.sqrt(np.nanmean(np.square(sample), axis=(0, 2)))
            prof = prof / np.sqrt(np.nanmean(np.square(prof)))
        out = np.clip(z / prof[None, :, None], -clip, clip)
    return out.astype(np.float32)


def lagged(x: np.ndarray, first: int, last: int) -> np.ndarray:
    """Sum of ``x`` over lags ``first..last`` along the bar axis, inside each session.

    ``out[s, b] = x[s, b - first] + ... + x[s, b - last]``, lags before the session's first
    bar dropped (a daily cube is one session, so there the sum crosses days). NaN counts as 0.
    ``first = 0`` includes the bar itself (a signal known at its close).

    Args:
        x: (session, bar, ...) array.
        first: Smallest lag (>= 0).
        last: Largest lag (>= first).

    Returns:
        float32 array shaped like ``x``.
    """
    z = np.nan_to_num(np.asarray(x, np.float32))
    csum = np.concatenate([np.zeros_like(z[:, :1]), np.cumsum(z, axis=1)], axis=1)
    b = np.arange(z.shape[1])
    # sum of bars b-last .. b-first = csum[b-first+1] - csum[b-last]; both clipped at the
    # session's first bar, so a lag reaching before it contributes nothing.
    return csum[:, np.clip(b - first + 1, 0, None)] - csum[:, np.clip(b - last, 0, None)]


# ---------------------------------------------------------------------------
# The map.
# ---------------------------------------------------------------------------
def pair_t(X: np.ndarray, Y: np.ndarray, cl: np.ndarray, *, roll: int = 0) -> np.ndarray:
    """Cluster-robust t of ``Σ X_i·Y_j`` for every leader ``i`` × follower ``j`` at once.

    ``T = Σ_c S_c / sqrt(Σ_c S_c²)`` with ``S_c`` the products summed inside cluster ``c``
    (a session, or a week at 1d): the score test of "no alignment", valid when the volatility of
    many names moves together — a plain ``ρ·√n`` is not (≈40x fake discoveries in a simulation
    with shared volatility bursts; tests). NaN counts as 0, so a name without data in a cluster
    adds nothing to it.

    Args:
        X: Leader regressors (rows × leaders), or a cube (flattened).
        Y: Follower targets (rows × followers), or a cube (flattened).
        cl: Non-decreasing cluster id per row.
        roll: Rows to shift ``X`` by (the null: whole sessions intraday, >= 63 days at 1d).

    Returns:
        float32 (leaders × followers) T; NaN where a follower or leader has no data.
    """
    X = np.nan_to_num(np.asarray(X, np.float32).reshape(-1, X.shape[-1]))
    Y = np.nan_to_num(np.asarray(Y, np.float32).reshape(-1, Y.shape[-1]))
    if roll:
        X = np.roll(X, roll, axis=0)
    cuts = np.flatnonzero(np.diff(cl)) + 1
    S = np.zeros((X.shape[1], Y.shape[1]))
    V = np.zeros_like(S)
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(cl)], strict=True):
        s = X[a:b].T @ Y[a:b]
        S += s
        V += np.square(s, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (S / np.sqrt(V)).astype(np.float32)


# ---------------------------------------------------------------------------
# Controls.
# ---------------------------------------------------------------------------
def granger(y: np.ndarray, shared, own: np.ndarray | None = None,
            cl: np.ndarray | None = None) -> dict:
    """Per-column OLS ``y_j ~ [1, shared, own_j]``: innovations, coefficients, cluster t.

    With the market, its lags and the name's own lags as regressors, the residual ``u_j`` is the
    part of a follower's move no market channel explains, and ``Σ_t E_i u_j`` is exactly the
    numerator of the Granger coefficient of a leader regressor ``E_i`` (Frisch-Waugh), so the
    pair map on ``u`` is a Granger map. Rows where ``y_j`` is NaN are skipped for that column;
    NaN regressors count as 0.

    Args:
        y: Targets, (rows × names) or a cube.
        shared: Regressors common to every column: a list of (rows,) or (session, bar) arrays
            (may be empty).
        own: Optional per-column regressor shaped like ``y`` (e.g. the name's own lag sum).
        cl: Non-decreasing cluster id per row; None skips the t.

    Returns:
        ``{"u": residuals shaped like y (NaN where y is), "coef": (names × k), "t": (names × k)}``
        with columns ``[intercept, *shared, own]``.
    """
    shape = y.shape
    Y = np.asarray(y).reshape(-1, shape[-1])
    A = np.column_stack([np.ones(len(Y))] + [np.nan_to_num(np.ravel(a).astype(np.float64))
                                             for a in shared])
    W = None if own is None else np.asarray(own).reshape(len(Y), -1)
    k = A.shape[1] + (W is not None)
    coef, tst = np.full((shape[-1], k), np.nan), np.full((shape[-1], k), np.nan)
    u = np.full(Y.shape, np.nan, np.float32)
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j])
        if ok.sum() <= 2 * k:
            continue
        yj = Y[ok, j].astype(np.float64)
        X = A[ok] if W is None else np.column_stack([A[ok], np.nan_to_num(W[ok, j])])
        xtx_inv = np.linalg.pinv(X.T @ X)
        b = xtx_inv @ (X.T @ yj)
        e = yj - X @ b
        u[ok, j], coef[j] = e, b
        if cl is not None:
            c = cl[ok]
            starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
            g = np.add.reduceat(X * e[:, None], starts, axis=0)
            n_g = len(starts)
            var = np.diag(xtx_inv @ (g.T @ g) @ xtx_inv) * n_g / max(n_g - 1, 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                tst[j] = b / np.sqrt(var)
    return {"u": u.reshape(shape), "coef": coef, "t": tst}


def portfolios(x: np.ndarray, labels) -> tuple[np.ndarray, list]:
    """NaN-aware equal-weight columns, one per distinct label (None/NaN labels left out).

    Args:
        x: (rows × names) or a cube.
        labels: One label per name (quintile, industry, ...).

    Returns:
        ``(P, groups)``: P shaped like ``x`` with the last axis = groups (NaN where no member
        has a value), and the sorted group labels.
    """
    lab = pd.Series(list(labels))
    groups = sorted(lab.dropna().unique().tolist())
    H = np.stack([(lab == g).to_numpy() for g in groups], axis=1).astype(np.float32)
    X = np.asarray(x, np.float32).reshape(-1, x.shape[-1])
    ok = np.isfinite(X)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = (np.where(ok, X, 0.0) @ H) / (ok.astype(np.float32) @ H)
    return P.reshape(*x.shape[:-1], len(groups)), groups


# ---------------------------------------------------------------------------
# Tradability.
# ---------------------------------------------------------------------------
def forward(x: np.ndarray, y: np.ndarray, h: int, *, delay: int = 1) -> np.ndarray:
    """Log return of a trade decided at bar ``t``'s close, entered at the open of ``t + delay``.

    Held ``h`` bars: exit at the close of ``t + delay + h - 1``, so the return is
    ``y[t+delay] + x[t+delay+1] + ... + x[t+delay+h-1]``. NaN when the entry bar has no print or
    the hold would run past the session's last bar (daily: the data's), so a trade never holds
    overnight intraday. ``delay = 2`` is the latency check (one bar late).

    Args:
        x: Raw last-trade returns (a cube).
        y: Raw first-to-last print returns of the same cube.
        h: Bars held (>= 1).
        delay: Bars from the signal's close to the entry open (>= 1).

    Returns:
        float32 cube, indexed by the SIGNAL bar ``t``.
    """
    out = np.full(y.shape, np.nan, np.float32)
    n = y.shape[1] - delay - h + 1
    if n <= 0:
        return out
    acc = y[:, delay:delay + n].astype(np.float32)
    for s in range(1, h):
        acc = acc + x[:, delay + s:delay + s + n]
    out[:, :n] = acc
    return out


def spread(pred: np.ndarray, fwd: np.ndarray, cl: np.ndarray, *, q: float = 0.1,
           min_names: int = 20) -> dict:
    """Per-row top-minus-bottom-``q`` forward return and rank IC across names, cluster t's.

    Args:
        pred: Predictions (rows × names) or a cube.
        fwd: Forward returns shaped like ``pred``.
        cl: Cluster id per row.
        q: Tail fraction on each side.
        min_names: Rows with fewer valid names are skipped.

    Returns:
        ``{"spread", "spread_t", "ic", "ic_t", "rows", "deciles"}`` — mean per-row spread and
        IC with cluster t's, rows used, and the mean forward return of each prediction decile.
    """
    n = pred.shape[-1]
    P = np.asarray(pred, np.float64).reshape(-1, n)
    F = np.asarray(fwd, np.float64).reshape(-1, n)
    ok = np.isfinite(P) & np.isfinite(F)
    keep = ok.sum(axis=1) >= min_names
    P, F, ok, c = np.where(ok, P, np.nan)[keep], np.where(ok, F, np.nan)[keep], ok[keep], cl[keep]
    empty = {"spread": np.nan, "spread_t": np.nan, "ic": np.nan, "ic_t": np.nan, "rows": 0,
             "deciles": np.full(10, np.nan)}
    if not len(P):
        return empty
    rp = pd.DataFrame(P).rank(axis=1, pct=True).to_numpy()
    rf = pd.DataFrame(F).rank(axis=1, pct=True).to_numpy()
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        top = np.nanmean(np.where(rp > 1 - q, F, np.nan), axis=1)
        bot = np.nanmean(np.where(rp <= q, F, np.nan), axis=1)
        dp, df = rp - np.nanmean(rp, axis=1, keepdims=True), rf - np.nanmean(rf, axis=1,
                                                                            keepdims=True)
        ic = np.nansum(dp * df, axis=1) / np.sqrt(np.nansum(dp**2, axis=1)
                                                  * np.nansum(df**2, axis=1))
        bucket = np.clip(np.ceil(rp * 10), 1, 10)
        deciles = np.array([np.nanmean(np.nanmean(np.where(bucket == d, F, np.nan), axis=1))
                            for d in range(1, 11)])
    s, i = day_t(top - bot, c), day_t(ic, c)
    return {"spread": s["mean"], "spread_t": s["t"], "ic": i["mean"], "ic_t": i["t"],
            "rows": int(len(P)), "deciles": deciles}


# ---------------------------------------------------------------------------
# Reporting (copied from strategies/burst_scan/research.py).
# ---------------------------------------------------------------------------
def day_t(values, days) -> dict:
    """Mean of ``values`` with a t-statistic clustered by day (or any cluster id).

    Cluster-robust (CR1) standard error of a mean: events on one day share that day's news and
    move together, so treating them as independent overstates t by roughly √(events per day).

    Args:
        values: One value per event (NaN dropped).
        days: The event's cluster (any hashable per event).

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


def spearman(a, b) -> float:
    """Rank correlation of two equal-length vectors over the pairs where both are finite."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(pd.Series(a[ok]).rank(), pd.Series(b[ok]).rank())[0, 1])


def checks_table(checks: list[tuple]) -> pd.DataFrame:
    """The notebook's ``(check, value, expected, ok)`` list as a table; prints a one-line tally.

    Args:
        checks: One tuple per check.

    Returns:
        The table with ``ok`` as "OK" / "FAIL".
    """
    frame = pd.DataFrame(checks, columns=["check", "value", "expected", "ok"])
    frame["ok"] = np.where(frame["ok"].astype(bool), "OK", "FAIL")
    n_fail = int((frame["ok"] == "FAIL").sum())
    print(f"CHECKS: {len(frame) - n_fail} OK, {n_fail} FAIL")
    return frame


def _rounded(value):
    """JSON-safe copy: floats to 4 decimals, NaN to None, numpy scalars to Python."""
    if isinstance(value, dict):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 4)
    return value


def scorecard(payload: dict) -> None:
    """Print the notebook's one-line machine-readable result: ``SCORECARD {json}``."""
    print("SCORECARD " + json.dumps(_rounded(payload)))
