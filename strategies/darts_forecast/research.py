"""darts_forecast research helpers for the notebooks: loading, the option inputs, saved forecasts,
evaluation, the retraining study, input importance, the coin-flip null, and the notebooks' own
tables (the trading rule's funnel, backtests, stress, monthly returns) and small date and format
utilities. The last section serves ``darts_forecast_walkthrough.ipynb`` (1-hour bars, a 5-day
horizon): the multi-day forecast table, the obvious-guess baselines, the outcome-flip luck band,
the price and direction scores, the leak detector, the latest forecast, the verdict and the gaps.

Loaded by path from the notebooks; ``strategy.py`` never imports it. Unlike ``signals()`` this
module does I/O: it reads the shared database (prices) and the free options store
(``strategies/burst_scan/options_data``, collected by ``burst_scan/options_free_data.ipynb``).
Its siblings are loaded the same way: ``charts.py`` (the figure helpers) and ``checks.py`` (the
integrity and leakage checks) never load this module; ``explorer.py`` (the explorer notebook's
helpers) builds on it and loads it once, as ``explorer.rs``.

The data split lives in :data:`WINDOWS`; every notebook prints it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import text

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import load_panel

STRAT = strategies.load("darts_forecast")
SM = STRAT.signals.__globals__                      # strategy.py's own functions
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"                              # git-ignored: saved forecasts
OPTIONS_STORE = strategies.STRATEGIES_DIR / "burst_scan" / "options_data"
#: Part of every cache key: editing strategy.py invalidates every saved forecast.
CODE_VERSION = hashlib.sha1((HERE / "strategy.py").read_bytes()).hexdigest()[:10]
#: The data split, inclusive SESSION dates. TUNE = everything a knob may be chosen on (walk-forward:
#: each forecast is out of sample, but the researcher has seen these months). TEST = the frozen
#: config, scored once. HOLDOUT = read once, only behind READ_HOLDOUT = True.
WINDOWS = {"tune": ("2025-06-18", "2026-01-31"),
           "test": ("2026-02-01", "2026-04-30"),
           "holdout": ("2026-05-01", None)}
_OF = None


def windows_table() -> pd.DataFrame:
    """The split as a table (window, from, to, what it is for)."""
    use = {"tune": "fit + forecast walk-forward; knobs may be chosen here",
           "test": "the frozen config, scored ONCE (diagnostics notebook)",
           "holdout": "read ONCE behind READ_HOLDOUT = True"}
    return pd.DataFrame([{"window": k, "from": a, "to": b or "last stored bar", "use": use[k]}
                         for k, (a, b) in WINDOWS.items()])


def round_trip_bp(settings, panel) -> float:
    """The engine's round-trip cost in bp at this strategy's order size (2 legs of
    ``bt.strategy_costs``: commission incl. the $1 minimum, spread, slippage)."""
    fees, slip = bt.strategy_costs(STRAT, panel, settings)
    return 2 * (fees + slip) * 1e4


# ---------------------------------------------------------------------------------------------
# Loading and the option inputs.
# ---------------------------------------------------------------------------------------------
def options_module():
    """``burst_scan/options_free.py`` (the free options collector), loaded by path."""
    global _OF
    if _OF is None:
        spec = importlib.util.spec_from_file_location(
            "options_free", strategies.STRATEGIES_DIR / "burst_scan" / "options_free.py")
        _OF = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_OF)
    return _OF


def option_frames(sessions, symbols, store: Path = OPTIONS_STORE, *, stale_days: int = 5) -> dict:
    """The option inputs as ``{"opt_<name>": DataFrame (session date x symbol)}``, indexed at UTC
    midnight of each session date like ``features.py``'s frames.

    A value computed from day ``e``'s END-OF-DAY data is attached to the first session AFTER
    ``e`` (strictly later: it is only known after ``e``'s close), and never more than
    ``stale_days`` calendar days after ``e`` (the DoltHub history has gaps of up to two weeks;
    a stale value is missing, not carried). A file dated on a market holiday inside the panel's
    span is dropped: DoltHub writes one as a copy of the session before, which would read as a
    zero day-over-day change. ``d_atm_iv`` / ``d_oi`` are day-over-day changes only across at most 4
    calendar days (a weekend plus a holiday), never across a data gap.

    Args:
        sessions: The panel's session dates.
        symbols: The panel's symbols.
        store: The options store.
        stale_days: Maximum calendar days between the data date and the session.

    Returns:
        Frames for atm_iv, skew, term, d_atm_iv, iv_hv, cust_pc, vol_surge, oi_pc, vol_pc, d_oi, gex
        (missing sources simply produce no frame).
    """
    of = options_module()
    sess = pd.DatetimeIndex(pd.to_datetime(pd.Series(sessions))).normalize()

    def trading(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        days = pd.to_datetime(frame["date"]).dt.normalize()
        return frame[~(days.between(sess.min(), sess.max()) & ~days.isin(sess))]

    def near(frame: pd.DataFrame) -> pd.Series:      # the previous row is <= 4 calendar days back
        return pd.to_datetime(frame["date"]).groupby(frame["underlying"]).diff() <= pd.Timedelta(
            days=4)

    parts = []
    dolt = trading(of.load_history("dolthub_chain", store))
    iv = None
    if len(dolt):
        iv = of.iv_features(dolt).sort_values(["underlying", "date"]).reset_index(drop=True)
        iv["d_atm_iv"] = iv.groupby("underlying")["atm_iv"].diff().where(near(iv))
        parts.append(iv[["date", "underlying", "atm_iv", "skew", "term", "d_atm_iv"]])
    dvol = trading(of.load_history("dolthub_vol", store))
    if len(dvol):
        hv = dvol.rename(columns={"act_symbol": "underlying"})
        hv["iv_hv"] = hv["iv_current"].astype(float) - hv["hv_current"].astype(float)
        parts.append(hv[["date", "underlying", "iv_hv"]])
    occ = trading(of.load_history("occ_volume", store))
    if len(occ):
        parts.append(of.occ_features(occ)[["date", "underlying", "cust_pc", "vol_surge"]])
    md = trading(of.load_history("marketdata_chain", store))
    if len(md):
        mdf = of.md_features(md, iv)                 # sorted by underlying, date
        mdf["d_oi"] = mdf["d_oi"].where(near(mdf))
        parts.append(mdf[["date", "underlying", "oi_pc", "vol_pc", "d_oi", "gex"]])
    sess = pd.DataFrame({"session": pd.to_datetime(pd.Series(sessions)).dt.normalize()})
    sess = sess.drop_duplicates().sort_values("session")
    out = {}
    for part in parts:
        part = part[part["underlying"].isin(symbols)].copy()
        part["date"] = pd.to_datetime(part["date"]).dt.normalize()
        for col in [c for c in part.columns if c not in ("date", "underlying")]:
            wide = part.pivot_table(index="date", columns="underlying", values=col).sort_index()
            wide = wide.reset_index().rename(columns={"date": "data_date"})
            m = pd.merge_asof(sess, wide.assign(key=wide["data_date"]),
                              left_on="session", right_on="key", direction="backward",
                              allow_exact_matches=False, tolerance=pd.Timedelta(days=stale_days))
            frame = m.set_index("session").reindex(columns=list(symbols)).astype(float)
            frame.index = pd.DatetimeIndex(frame.index).tz_localize("UTC")
            frame.index.name = "date"
            out["opt_" + col] = frame
    return out


def attach_options(panel, store: Path = OPTIONS_STORE):
    """A copy of ``panel`` with the option inputs in ``panel.features`` (:func:`option_frames`)."""
    dates = pd.Series(panel.close.index.tz_convert(panel.tz).date).unique()
    feats = {**(panel.features or {}), **option_frames(dates, panel.symbols, store)}
    return replace(panel, features=feats)


def adjustment_factors(engine, symbols, start, end=None) -> pd.DataFrame:
    """The stored DAILY bars' adjustment factor ``adj_close / close`` (session date x symbol).

    The stored 1-minute bars are not adjusted for dividends (their ``adj_close`` equals
    ``close``), so a multi-day return across an ex-dividend date shows the dividend as a loss.
    This factor steps on every ex-date (and split); dates without a daily bar get no factor.

    Args:
        engine: Database engine.
        symbols: The symbols.
        start: First date.
        end: Last timestamp (inclusive), or None for everything stored.

    Returns:
        A DataFrame indexed by naive session date, one column per symbol.
    """
    q = text("SELECT symbol, ts, close, adj_close FROM prices_ohlcv WHERE bar = '1d' "
             "AND symbol = ANY(:syms) AND ts >= :a "
             "AND ts <= COALESCE(CAST(:b AS timestamptz), ts)")
    with engine.connect() as c:
        df = pd.read_sql(q, c, params={"syms": list(symbols), "a": pd.Timestamp(start),
                                       "b": pd.Timestamp(end) if end else None})
    df["date"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None).dt.normalize()
    df["f"] = (df["adj_close"] / df["close"]).where(df["close"] > 0)
    return df.pivot_table(index="date", columns="symbol", values="f").reindex(
        columns=list(symbols))


def adjust_dividends(panel, factors: pd.DataFrame):
    """A copy of a 1-minute panel with each session's prices multiplied by that session's daily
    adjustment factor (:func:`adjustment_factors`): a return across an ex-dividend date then
    includes the dividend (the holder's total return), and every other return is unchanged.
    A session without a factor (a hole in the daily bars, or after the last stored one) carries
    the previous session's: a factor of 1 there would be a fake jump of the cumulative factor
    (up to the dividends paid since). Sessions before the first factor keep their raw prices.
    The factors are taken relative to the panel's last session, so its latest prices are the
    traded ones (and a dividend paid after the panel ends changes nothing in it).

    Args:
        panel: The 1-minute panel.
        factors: :func:`adjustment_factors` of its symbols.

    Returns:
        The adjusted panel (volume unchanged).
    """
    days = pd.DatetimeIndex(panel.close.index.tz_convert(panel.tz).date)
    f = factors.reindex(columns=panel.symbols).reindex(days).ffill().fillna(1.0).to_numpy()
    f = f / f[-1:]                                     # relative to the last session: traded prices

    def scaled(x):
        return pd.DataFrame(x.to_numpy() * f, index=x.index, columns=x.columns)

    return replace(panel, open_=scaled(panel.open_), high=scaled(panel.high),
                   low=scaled(panel.low), close=scaled(panel.close))


def load(settings, engine, *, start: str | None = None, end: str = WINDOWS["tune"][1],
         options: bool = True, symbols: list[str] | None = None, adjust: bool = False):
    """The strategy's 1-minute panel (config ``data`` block) with the option inputs attached.

    Args:
        settings: Global settings.
        engine: Database engine.
        start: First session (default: the config's ``data.start``).
        end: LAST SESSION, inclusive (default: the TUNE window's end). Pass ``WINDOWS["test"][1]``
            to include the test months (warm: the walk-forward simply continues) or ``"all"`` for
            everything stored (the HOLDOUT; only behind READ_HOLDOUT).
        options: Attach the option inputs.
        symbols: Override the universe.
        adjust: Adjust the prices for dividends with the daily bars' factors
            (:func:`adjust_dividends`). Needed for multi-day (``overnight``) targets; intraday
            returns do not change.
    """
    d = STRAT.data
    syms = symbols or list(settings.universe[d["universe"]])
    # load_panel keeps ts <= end: a bare date would be 00:00 UTC and drop that whole session
    stop = None if end == "all" else (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
    p = load_panel(engine, syms, "1m", start or d["start"], stop,
                   session=d.get("session", "regular"), exchanges=settings.exchanges)
    if adjust:      # daily bars are stamped 00:00 UTC of their date: ts <= end reads through end
        last = None if end == "all" else pd.Timestamp(end, tz="UTC")
        p = adjust_dividends(p, adjustment_factors(engine, syms, start or d["start"], last))
    return attach_options(p) if options else p


# ---------------------------------------------------------------------------------------------
# The options store, file by file (the data notebook re-derives the option inputs from it).
# ---------------------------------------------------------------------------------------------
def store_span(source: str, store: Path = OPTIONS_STORE) -> tuple:
    """First and last day stored for an options-history source, from the file names alone (no
    values are read).

    Args:
        source: The source's folder in the options store (e.g. ``"dolthub_chain"``).
        store: The options store.

    Returns:
        ``(first, last)`` as Timestamps, or ``(None, None)`` when the folder holds no file.
    """
    days = sorted(f.name[:10] for f in (store / source).glob("*.csv.gz"))
    return (pd.Timestamp(days[0]), pd.Timestamp(days[-1])) if days else (None, None)


def read_store(source: str, first: str, last: str, store: Path = OPTIONS_STORE) -> pd.DataFrame:
    """Every stored day of an options-history source between ``first`` and ``last`` (inclusive
    "YYYY-MM-DD" strings), as the raw rows ``burst_scan/options_free.py`` reads. A file without a
    ``date`` column gets one from its file name.

    Args:
        source: The source's folder in the options store.
        first: First day to read.
        last: Last day to read.
        store: The options store.

    Returns:
        The rows of every non-empty file in the range, or an empty frame.
    """
    of = options_module()
    out = []
    for f in sorted((store / source).glob("*.csv.gz")):
        if first <= f.name[:10] <= last:
            df = of._read(f)
            if len(df):
                if "date" not in df:
                    df.insert(0, "date", f.name[:10])
                out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def sessions_only(df: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Drop store days inside the span of ``sessions`` that are not sessions: DoltHub files a copy
    of the session before on market holidays (the data notebook's check (i)).
    :func:`option_frames` drops them too.

    Args:
        df: Raw store rows with a ``date`` column.
        sessions: The panel's session dates (naive, sorted).

    Returns:
        The rows dated on a session or outside the span.
    """
    if not len(df):
        return df
    d = pd.to_datetime(df["date"])
    return df[~(d.between(sessions[0], sessions[-1]) & ~d.isin(sessions))]


def gap_days(df: pd.DataFrame) -> pd.Series:
    """Calendar days from each row's previous stored day of the same underlying (NaN for its
    first row): a day-over-day change is only taken across at most 4 days.

    Args:
        df: Rows with ``date`` and ``underlying`` columns, sorted by underlying and date.

    Returns:
        The gap in days, aligned with ``df``.
    """
    return pd.to_datetime(df["date"]).groupby(df["underlying"]).diff().dt.days


def store_day(folder: str, day: str, names, store: Path = OPTIONS_STORE) -> pd.DataFrame:
    """The rows of ``names`` in one stored day, without the date and fetch stamps and sorted, so
    two days' files compare equal exactly when they hold the same data.

    Args:
        folder: The source's folder in the options store.
        day: The day ("YYYY-MM-DD").
        names: The underlyings to keep.
        store: The options store.

    Returns:
        The sorted rows, or an empty frame when no file is stored for that day.
    """
    df = read_store(folder, day, day, store)
    if not len(df):
        return df
    who = "underlying" if "underlying" in df else "act_symbol"
    df = df[df[who].isin(names)].drop(columns=["date", "fetched_at"], errors="ignore")
    return df.sort_values(list(df.columns)).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Forecasts: run, save, load.
# ---------------------------------------------------------------------------------------------
def model_params(params: dict) -> dict:
    """The forecast-relevant subset of a parameter dict (``strategy.MODEL_KEYS``); a dict made
    before the ``overnight`` / ``expanding`` knobs existed gets them off."""
    return {k: params.get(k, False) if k in SM["LATE_KEYS"] else params[k]
            for k in SM["MODEL_KEYS"]}


def run_forecast(panel, params: dict) -> dict:
    """``strategy.forecast`` for ``params`` (the in-process cache makes a repeat free; the result
    keeps the last fitted model in ``fc["model"]``)."""
    return SM["forecast"](panel, **model_params(params))


def _key(params: dict, panel, extra: str = "") -> str:
    """Cache file stem: a readable prefix + a hash of EVERY forecast knob, the panel's whole
    content (``strategy.fingerprint``) and the code version."""
    mp = model_params(params)
    ident = json.dumps({"params": mp, "panel": SM["fingerprint"](panel), "code": CODE_VERSION,
                        "extra": extra}, sort_keys=True, default=str)
    end = panel.close.index[-1].tz_convert(panel.tz).date()
    return (f"{mp['model']}-{mp['bar_minutes']}m-h{mp['horizon']}-r{mp['retrain_minutes']}"
            f"-to{end}{extra}-c{CODE_VERSION[:6]}-{hashlib.sha1(ident.encode()).hexdigest()[:12]}")


def save_forecast(fc: dict, params: dict, panel, extra: str = "") -> Path:
    """Pickle a forecast WITHOUT the fitted model to ``cache/<key>.pkl`` (git-ignored), atomically:
    an interrupted write never leaves a half file under the key."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{_key(params, panel, extra)}.pkl"
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({"params": model_params(params), "code": CODE_VERSION,
                     **{k: v for k, v in fc.items() if k != "model"}}, fh)
    os.replace(tmp, path)
    return path


def load_forecast(params: dict, panel, extra: str = "") -> dict | None:
    """The saved forecast for exactly these knobs, this panel content and this code version, or
    None (a missing or unreadable file is a miss). Its ``model`` is None."""
    try:
        with open(CACHE / f"{_key(params, panel, extra)}.pkl", "rb") as fh:
            fc = pickle.load(fh)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    return {**fc, "model": None}


def prune_cache() -> list[str]:
    """Delete the saved forecasts of other versions of strategy.py (no key can reach them again).
    Returns the deleted file names."""
    dead = [f for f in CACHE.glob("*.pkl") if f"-c{CODE_VERSION[:6]}-" not in f.name]
    for f in dead:
        f.unlink()
    return [f.name for f in dead]


def cached_forecast(panel, params: dict, extra: str = "", *, verbose: bool = True) -> dict:
    """Load the saved forecast, else run the walk-forward and save it."""
    fc = load_forecast(params, panel, extra)
    if fc is not None:
        return fc
    if verbose:
        print(f"walk-forward {_key(params, panel, extra)} ...")
    fc = run_forecast(panel, params)
    save_forecast(fc, params, panel, extra)
    return fc


def prime_cache(panel, params: dict, extra: str = "") -> dict:
    """Put the saved (or freshly computed) forecast for ``params`` on ``panel`` into
    ``strategy.forecast``'s in-process cache, so ``signals()`` / ``bt.evaluate`` / an Optuna study
    on THIS panel reuse it instead of re-running the walk-forward. The disk key holds the panel's
    content, so a forecast made on another panel is never installed (``extra``: the disk label,
    e.g. ``"-null0"`` for :func:`null_study`'s coin-flip forecasts). Returns the forecast."""
    fc = cached_forecast(panel, params, extra)
    SM["_CACHE"][(SM["fingerprint"](panel), tuple(sorted(model_params(params).items())))] = fc
    return fc


# ---------------------------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------------------------
def _in_window(dates: np.ndarray, window) -> np.ndarray:
    """``dates`` ("YYYY-MM-DD" strings) inside ``window`` (a WINDOWS name, a (from, to) pair with
    an inclusive or None end, or None for all)."""
    if window is None:
        return np.ones(len(dates), bool)
    a, b = WINDOWS[window] if isinstance(window, str) else window
    return (dates >= a) & ((dates <= b) if b else True)


def day_t(values, days) -> dict:
    """Mean with a t-statistic clustered by trading day (the names share each day's market)."""
    v = pd.Series(np.asarray(values, float))
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


def _cells(fc: dict, window=None) -> tuple[np.ndarray, np.ndarray]:
    """The (decision grid bar, name) cells :func:`table` scores, in its row order: a forecast
    exists, its label is usable (``inputs["valid"]``), and both the decision and its outcome fall
    in ``window`` (see :func:`_in_window`)."""
    inp = fc["inputs"]
    g, H = inp["g"], inp["horizon"]
    n, m = inp["target"].shape
    real = np.full((n, m), np.nan)
    real[:-H] = inp["target"][H:]
    usable = np.zeros((n, m), bool)
    usable[:-H] = inp["valid"][H:]
    rows, cols = np.nonzero(np.isfinite(fc["edge"]) & np.isfinite(real) & usable)

    def day(b):
        return pd.to_datetime(pd.Series(g["date"][b])).dt.strftime("%Y-%m-%d").to_numpy()

    keep = _in_window(day(rows), window) & _in_window(day(rows + H), window)
    return rows[keep], cols[keep]


def table(fc: dict, window=None) -> pd.DataFrame:
    """One row per scored forecast: date, month, name, minute (the decision's clock time, New
    York), edge (z), edge_bp, p_up, real (realized forward return, z), real_bp, fit, and block
    (the group its error bars cluster by: the session, or for a multi-day horizon a run of as
    many sessions as the horizon spans, since neighbouring days then share most of their
    outcome). Scored = a forecast exists, its label is usable (``inputs["valid"]``: inside the
    session for intraday horizons), and both the decision and its outcome fall in ``window``
    (see :func:`_in_window`), so loading a later window never changes this one's rows."""
    rows, cols = _cells(fc, window)
    return _frame(fc, rows, cols)


def _frame(fc: dict, rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    """:func:`table`'s columns for the cells ``rows`` x ``cols`` (from :func:`_cells`)."""
    inp = fc["inputs"]
    g, H = inp["g"], inp["horizon"]
    real = inp["target"][rows + H, cols]                          # the realized forward return
    dates = pd.to_datetime(pd.Series(g["date"][rows]))
    scale = inp.get("tscale", inp["scale"])[rows, cols]          # the target's units, in returns
    span = math.ceil(H / g["n_pos"]) if inp.get("overnight") else 1
    return pd.DataFrame({
        "date": dates.dt.strftime("%Y-%m-%d").to_numpy(),
        "month": dates.dt.strftime("%Y-%m").to_numpy(),
        "name": np.asarray(inp["names"])[cols],
        "minute": np.minimum(570 + (g["pos"][rows] + 1) * inp["bar_minutes"], 960),
        "edge": fc["edge"][rows, cols], "edge_bp": fc["edge"][rows, cols] * scale * 1e4,
        "p_up": fc["p_up"][rows, cols], "real": real,
        "real_bp": real * scale * 1e4, "fit": fc["fit"][rows, cols],
        "block": g["code"][rows] // span})


def rank_ic(t: pd.DataFrame) -> dict:
    """Pooled rank correlation of edge with the realized return, t clustered by ``block`` (by day
    for an intraday horizon; see :func:`table`)."""
    if len(t) < 30:
        return {"ic": np.nan, "t": np.nan, "n": len(t), "days": 0}
    a, b = t["edge"].rank(), t["real"].rank()
    za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    r = day_t(za * zb, t["block"] if "block" in t else t["date"])
    return {"ic": r["mean"], "t": r["t"], "n": r["n"], "days": r["days"]}


def summary(t: pd.DataFrame, *, cost_bp: float = 14.0) -> dict:
    """The forecast's headline numbers on a table from :func:`table`.

    ic / ic_t: pooled rank IC (day-clustered t); hit: sign agreement; auc: P(edge ranks an up
    move above a down move); mse_skill: 1 - MSE / MSE of the zero forecast (z units; > 0 means
    better than "no move"); sign_bp / sign_t: mean of sign(edge) x realized bp per forecast,
    before costs; net_bp: after a ``cost_bp`` round trip if every forecast were traded (default:
    the engine's 14 bp at this strategy's $2,000 orders, see :func:`round_trip_bp`)."""
    from sklearn.metrics import roc_auc_score

    out = rank_ic(t)
    if len(t) < 30:
        return out
    s = np.sign(t["edge"])
    out["hit"] = float((s == np.sign(t["real"]))[t["real"] != 0].mean())
    up = (t["real"] > 0).astype(int)
    out["auc"] = float(roc_auc_score(up, t["edge"])) if up.nunique() == 2 else np.nan
    out["mse_skill"] = float(1 - ((t["edge"] - t["real"]) ** 2).mean() / (t["real"] ** 2).mean())
    pnl = day_t(s * t["real_bp"], t["block"] if "block" in t else t["date"])
    out["sign_bp"], out["sign_t"] = pnl["mean"], pnl["t"]
    out["net_bp"] = pnl["mean"] - cost_bp
    return out


def by(t: pd.DataFrame, key: str, *, cost_bp: float = 14.0) -> pd.DataFrame:
    """:func:`summary` per group (``month`` | ``name`` | ``hour`` | ``fit`` | any column)."""
    t = t.assign(hour=(t["minute"] // 60).astype(int))
    return pd.DataFrame({k: summary(g, cost_bp=cost_bp) for k, g in t.groupby(key)}).T


def calibration(t: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Forecast deciles (0 = most negative): mean forecast vs mean realized return (bp), hit."""
    q = pd.qcut(t["edge"].rank(method="first"), bins, labels=False)
    return t.groupby(q).agg(edge_bp=("edge_bp", "mean"), real_bp=("real_bp", "mean"),
                            hit=("real", lambda r: float((r > 0).mean())), n=("real", "size"))


def rolling_ic(t: pd.DataFrame, days: int = 10) -> pd.Series:
    """Per-session rank IC, smoothed over ``days`` sessions (``days=1``: the raw daily IC)."""
    daily = t.groupby("date").apply(lambda d: d["edge"].rank().corr(d["real"].rank())
                                    if len(d) >= 10 else np.nan)
    return daily.rolling(days, min_periods=min(days, max(2, days // 2))).mean()


def null_z(real: dict, null: pd.DataFrame, metric: str = "ic") -> float:
    """How far a real :func:`summary` stands from the coin-flip runs of :func:`null_study`, from
    STANDARD ERRORS (a few seeds cannot give a spread): (real - null mean) / sqrt(SE_real^2 +
    SE_null^2 / k) with SE = value / t, capped at the real t against exactly 0. A coin-flip market
    has no direction, so its mean is 0 plus seed noise (or a leak, which this catches): a lucky
    negative null mean must not lift the real result. ``metric``: ``ic`` or ``sign_bp``."""
    tcol = {"ic": "t", "sign_bp": "sign_t"}[metric]
    se_r = abs(real[metric] / real[tcol]) if real.get(tcol) else np.nan
    se_n = float((null[metric] / null[tcol]).abs().mean()) if len(null) else np.nan
    den = math.sqrt(se_r ** 2 + se_n ** 2 / max(len(null), 1))
    z = float((real[metric] - null[metric].mean()) / den) if den > 0 else np.nan
    return min(z, float(real[tcol]))


def xs_ic(t: pd.DataFrame) -> pd.Series:
    """Per-session mean of the cross-sectional rank IC: at each decision bar the names are ranked
    against each other, which removes the market's move and the model's overall forecast level.

    Args:
        t: A forecast table from :func:`table`.

    Returns:
        One IC per session date (NaN where every name got the same rank).
    """
    keys = [t["date"], t["minute"]]
    ra, rb = t.groupby(keys)["edge"].rank(), t.groupby(keys)["real"].rank()
    da, db = ra - ra.groupby(keys).transform("mean"), rb - rb.groupby(keys).transform("mean")
    den = np.sqrt((da**2).groupby(keys).sum() * (db**2).groupby(keys).sum())
    return ((da * db).groupby(keys).sum() / den.replace(0, np.nan)).groupby(level=0).mean()


def univariate(src: dict, g: dict, after_warmup: np.ndarray, future: np.ndarray, horizon: int,
               names) -> tuple[pd.DataFrame, list[str]]:
    """Each input on its own: its pooled rank IC with the realized return of the next ``horizon``
    grid bars, t clustered by day (:func:`day_t`), plus each name's own rank IC.

    The rows are the decision bars after the warm-up whose outcome is usable and ends inside the
    session. A past input is read at the decision bar, a future (clock) input at the predicted bar.

    Args:
        src: A ``strategy.build_inputs`` result (the real panel's, or a coin-flip panel's).
        g: The REAL panel's grid (``inp["g"]``): its sessions and clock decide the rows.
        after_warmup: Grid bars past the warm-up (bool, one per grid bar).
        future: The real panel's future covariates (``inp["future"]``).
        horizon: Grid bars the target covers.
        names: The panel's names, in ``src["past"]`` order.

    Returns:
        ``(table, constant)``: one row per input (``ic``, ``t``, ``n``, ``days`` and one column per
        name), and the inputs that are constant on those rows (no IC).
    """
    H = horizon
    real = np.full(src["target"].shape, np.nan)
    real[:-H] = src["target"][H:]  # the realized forward return of a decision at s
    usable = np.zeros(real.shape, bool)
    usable[:-H] = src["valid"][H:]
    ok = usable & np.isfinite(real) & after_warmup[:, None]  # valid: same session if intraday
    r, c = np.nonzero(ok)
    fwd, days = real[r, c], g["date"][r]
    X = np.stack([src["past"][s] for s in names])[c, r]  # rows x inputs
    feats = {k: X[:, i] for i, k in enumerate(src["past_names"])}
    feats |= {k: future[r + H, i] for i, k in enumerate(src["future_names"])}  # the predicted bar
    out, constant = {}, []
    for k, x in feats.items():
        keep = np.isfinite(x)
        if np.unique(x[keep]).size < 2:
            constant.append(k)
            continue
        a = pd.Series(x[keep]).rank().to_numpy()
        b = pd.Series(fwd[keep]).rank().to_numpy()
        res = day_t((a - a.mean()) / a.std() * (b - b.mean()) / b.std(), days[keep])
        per = {
            s: pd.Series(x[keep & (c == jj)]).rank().corr(pd.Series(fwd[keep & (c == jj)]).rank())
            for jj, s in enumerate(names)
        }
        out[k] = {"ic": res["mean"], "t": res["t"], "n": res["n"], "days": res["days"], **per}
    return pd.DataFrame(out).T, constant


# ---------------------------------------------------------------------------------------------
# Studies: retraining cadence, input importance, the coin-flip null.
# ---------------------------------------------------------------------------------------------
def retrain_study(panel, params: dict, cadences=(0, 7800, 1950, 390), window="tune", *,
                  cost_bp: float = 14.0, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """The same model retrained every ``cadence`` trading minutes (0 = once, 7800 = every 20
    sessions, 1950 = weekly, 390 = daily, 60 = hourly ...). Returns (summary table, {cadence:
    forecast table}). Forecasts are cached on disk per cadence."""
    rows, tables = [], {}
    for c in cadences:
        p = {**params, "retrain_minutes": c}
        if not SM["forecast_valid"](p):
            raise ValueError(f"retrain_minutes={c} is not a valid forecaster cadence on the "
                             f"{p['bar_minutes']}-minute grid (strategy.forecast_valid)")
        fc = cached_forecast(panel, p, verbose=verbose)
        t = table(fc, window)
        tables[c] = t
        rows.append({"retrain_minutes": c, "fits": int(len(fc["log"])),
                     "fit_seconds": float(fc["log"]["seconds"].sum()) if len(fc["log"]) else 0.0,
                     **summary(t, cost_bp=cost_bp)})
    return pd.DataFrame(rows).set_index("retrain_minutes"), tables


def group_importance(panel, params: dict, *, repeats: int = 3, seed: int = 0,
                     window="tune") -> pd.DataFrame:
    """Permutation importance by INPUT GROUP, strictly out of sample.

    ONE model is fit at the first retrain point (``retrain_minutes = 0``, a static model), so every
    later decision bar is out of sample for it. Each repeat draws one permutation of the window's
    sessions after that fit and applies it, whole days at a time, to one group at a time - the
    intraday, market, daily, options and flag inputs, and the model's own lagged target ("target
    lags") - then re-forecasts every scored bar. ``ic_drop`` = the unshuffled IC minus the mean
    shuffled IC: how much the forecast leans on that group. It is large only if the model has an
    edge to lose; on a model with no edge every drop is noise around 0.

    Returns:
        One row per group (``group``, ``inputs``, ``ic_shuffled``, ``ic_drop``), largest drop first;
        ``attrs["base_ic"]`` is the static model's unshuffled IC.
    """
    p = {**params, "retrain_minutes": 0}
    fc = run_forecast(panel, p)
    inp, est = fc["inputs"], fc["model"]
    if est is None or not len(fc["points"]):
        return pd.DataFrame()
    code, H = inp["g"]["code"], inp["horizon"]
    first, last = int(fc["points"][0]), len(code) - 1 - H
    ds = pd.to_datetime(pd.Series(inp["g"]["date"])).dt.strftime("%Y-%m-%d").to_numpy()
    days = np.unique(code[first:][_in_window(ds[first:], window)])
    rows_of = {d: np.flatnonzero(code == d) for d in days}

    def ic(past=None, target=None) -> float:
        seg = SM["predict_segment"](est, inp, p["model"], first, last, past=past, target=target)
        edge = np.full_like(fc["edge"], np.nan)
        edge[seg["s"]] = seg["edge"]
        return rank_ic(table({**fc, "edge": edge}, window))["ic"]

    base = ic()
    groups = sorted(set(inp["groups"].values()))
    rng = np.random.default_rng(seed)
    drops = {g_: [] for g_ in [*groups, "target lags"]}
    for _ in range(repeats):
        perm = rng.permutation(days)
        while len(days) > 1 and (perm == days).any():     # every day takes another day's inputs
            perm = rng.permutation(days)
        pairs = [(rows_of[d], rows_of[s]) for d, s in zip(days, perm, strict=True)]
        for grp in drops:
            cols = [i for i, nme in enumerate(inp["past_names"]) if inp["groups"][nme] == grp]
            past = {s: x.copy() for s, x in inp["past"].items()} if cols else None
            target = inp["target"].copy() if grp == "target lags" else None
            for dst, src in pairs:
                k = min(len(dst), len(src))
                if target is not None:
                    target[dst[:k]] = inp["target"][src[:k]]
                for s in past or {}:
                    past[s][dst[:k, None], cols] = inp["past"][s][src[:k, None], cols]
            drops[grp].append(ic(past=past, target=target))
    n_in = {g_: sum(v == g_ for v in inp["groups"].values()) for g_ in groups}
    out = pd.DataFrame([{"group": g_, "inputs": n_in.get(g_, params["lookback"]),
                         "ic_shuffled": float(np.mean(v)), "ic_drop": base - float(np.mean(v))}
                        for g_, v in drops.items()])
    out.attrs["base_ic"] = base
    return out.sort_values("ic_drop", ascending=False).reset_index(drop=True)


def sign_surrogate(panel, seed: int = 0, *, common: bool = True):
    """The coin-flip market: every 1-minute bar's move (from the previous VALID close, gaps
    included) mirrored with probability 1/2 and the path re-chained from each name's first valid
    open. Volatility, volume and the clock stay; direction is destroyed. ``common=True`` uses ONE
    sign per minute for all names (keeps their co-movement - the right null for anything pooled
    across names or using the market input)."""
    o, h, lo, c = (x.to_numpy(float) for x in (panel.open_, panel.high, panel.low, panel.close))
    n, m = c.shape
    seen = np.isfinite(o)
    first_o = np.where(seen.any(axis=0), o[seen.argmax(axis=0), np.arange(m)], np.nan)
    prev = np.vstack([np.full((1, m), np.nan), pd.DataFrame(c).ffill().to_numpy()[:-1]])
    prev = np.where(np.isfinite(prev), prev, first_o)
    a, hh, ll, cc = (np.log(x / prev) for x in (o, h, lo, c))
    rng = np.random.default_rng(seed)
    if common:
        sign = np.repeat(np.where(rng.random(n) < 0.5, -1.0, 1.0)[:, None], m, 1)
    else:
        sign = np.where(rng.random((n, m)) < 0.5, -1.0, 1.0)
    level = np.log(first_o) + np.vstack([np.zeros((1, m)),
                                         np.cumsum(np.nan_to_num(sign * cc), axis=0)[:-1]])
    moves = {"open_": sign * a, "high": np.where(sign > 0, hh, -ll),
             "low": np.where(sign > 0, ll, -hh), "close": sign * cc}
    return replace(panel, **{k: pd.DataFrame(np.exp(level + x), index=panel.close.index,
                                             columns=panel.close.columns)
                             for k, x in moves.items()})


def null_study(panel, params: dict, seeds=(0, 1, 2), window="tune", *,
               cost_bp: float = 14.0) -> pd.DataFrame:
    """The whole walk-forward on coin-flip markets: what the summary looks like with NO direction
    to find. Judge the real result against it with :func:`null_z`, not against 0.5 / 0."""
    rows = []
    for s in seeds:
        fc = cached_forecast(sign_surrogate(panel, s), params, extra=f"-null{s}")
        rows.append({"seed": s, **summary(table(fc, window), cost_bp=cost_bp)})
    return pd.DataFrame(rows).set_index("seed")


# ---------------------------------------------------------------------------------------------
# The grid and the walk-forward, as tables (the data notebook's split and inventory).
# ---------------------------------------------------------------------------------------------
def loaded(features: dict, cols, sessions_utc: pd.DatetimeIndex) -> np.ndarray:
    """Per session: does any name have any of these inputs?

    Args:
        features: ``panel.features``.
        cols: Feature names (``"opt_*"``), or None for the 1-minute bars (always there).
        sessions_utc: The sessions, as ``panel.features`` indexes them (UTC midnight).

    Returns:
        One bool per session.
    """
    if cols is None:
        return np.ones(len(sessions_utc), bool)
    have = [features[c].reindex(sessions_utc).notna().any(axis=1).to_numpy()
            for c in cols if c in features]
    return np.any(have, axis=0) if have else np.zeros(len(sessions_utc), bool)


def runs(mask) -> list[tuple[int, int]]:
    """``(first, last)`` index of every run of True in a boolean array.

    Args:
        mask: A boolean array.

    Returns:
        The runs, in order (empty when nothing is True).
    """
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]], strict=True))


def x_of(b0: int, b1: int, g: dict) -> tuple[float, float]:
    """Grid bars ``[b0, b1)`` as ``(start, width)`` in sessions: session number + the fraction of
    the day, so sessions sit side by side (nights and weekends do not exist on the grid).

    Args:
        b0: First grid bar.
        b1: One past the last grid bar.
        g: The grid (``inp["g"]``).

    Returns:
        ``(start, width)``.
    """
    code, n_pos = g["code"], g["n_pos"]
    x0 = code[b0] + g["pos"][b0] / n_pos
    x1 = code[b1 - 1] + (g["pos"][b1 - 1] + 1) / n_pos
    return float(x0), float(x1 - x0)


def when(t: int, g: dict, bar_minutes: int) -> str:
    """When the retrain at grid bar ``t`` happens, in words ("2025-12-18 before the open" or
    "... after 10:35").

    Args:
        t: The retrain point (a grid bar).
        g: The grid.
        bar_minutes: Minutes per grid bar.

    Returns:
        The date and the moment.
    """
    pos = g["pos"][t]
    return f"{g['date'][t]} " + (
        "before the open" if pos == 0 else f"after {clock(pos - 1, bar_minutes)}")


def slices(t: int, t_next: int, val: int, g: dict, train_days: int, colors) -> list[tuple]:
    """``strategy.walk_forward``'s slices of the fit at retrain point ``t``, in grid bars.

    Args:
        t: The retrain point.
        t_next: The next retrain point (or the grid's end).
        val: VALIDATION sessions (0 for gbm).
        g: The grid.
        train_days: Sessions in each training window.
        colors: The (TRAIN, VALIDATION, FORECAST) colors, carried through for the chart.

    Returns:
        ``[(first, end, color, name)]`` for TRAIN, VALIDATION and FORECAST (``end`` excluded; an
        empty slice has ``end <= first``).
    """
    code, sf = g["code"], g["sess_first"]
    s = int(code[t])
    lo = int(sf[max(0, s - train_days)])
    fit_hi = int(sf[max(0, s - val)]) if val else t
    c_train, c_val, c_fc = colors
    return [
        (lo, fit_hi, c_train, "TRAIN (fitted)"),
        (fit_hi, t, c_val, "VALIDATION (early stopping)"),
        (t, t_next, c_fc, "FORECAST (out of sample)"),
    ]


def missing(values: np.ndarray, names) -> dict:
    """Percent missing per name, as Table 3's columns.

    Args:
        values: names x grid bars.
        names: The panel's names, in row order.

    Returns:
        ``{"<name> % missing": percent}``.
    """
    return {f"{s} % missing": 100 * np.isnan(values[j]).mean() for j, s in enumerate(names)}


# ---------------------------------------------------------------------------------------------
# The trading rule, decision by decision (the strategy notebook's funnel and position path).
# ---------------------------------------------------------------------------------------------
def decisions(r: dict, fc: dict) -> dict:
    """The rule ``r``'s gates, stage by stage, on the grid (grid bars x names): the tests of
    ``strategy.trade_frames`` (``checks.path_misses`` confirms they reproduce its position path).

    Args:
        r: The trading-rule knobs (``sides``, ``edge_bp``, ``p_min``, ``hold_bars``,
            ``skip_open``, ``skip_close``, ``eod_exit``, ``health_days``, ``health_min``).
        fc: The forecast.

    Returns:
        Boolean arrays: ``hours`` (inside the trading hours, a forecast exists), ``edge`` (also
        beyond ``edge_bp`` and ``p_min``, either side), ``long`` and ``short`` (also through the
        health gate: the rule's decisions).
    """
    inp = fc["inputs"]
    g = inp["g"]
    edge_bp = fc["edge"] * inp.get("tscale", inp["scale"]) * 1e4  # forecast over the horizon, bp
    R, room = inp["bar_minutes"], (r["hold_bars"] or inp["horizon"]) + (0 if r["eod_exit"] else 1)
    hours = (
        g["regular"]
        & ((g["pos"] + 1) * R >= r["skip_open"])
        & (g["left"] * R >= r["skip_close"])
        & (g["left"] >= room)
    )
    out = {"hours": hours[:, None] & np.isfinite(edge_bp)}
    p, has_p = fc["p_up"], np.isfinite(fc["p_up"])
    long_ = out["hours"] & (edge_bp > r["edge_bp"]) & (~has_p | (p >= r["p_min"]))
    short = out["hours"] & (edge_bp < -r["edge_bp"]) & (~has_p | (1 - p >= r["p_min"]))
    long_ &= r["sides"] != "short"
    short &= r["sides"] != "long"
    out["edge"] = long_ | short
    if r["health_days"] > 0:
        hs = SM["health"](fc, days=r["health_days"])[g["code"]]
        gate = (np.nan_to_num(hs, nan=-np.inf) > r["health_min"])[:, None]
        long_, short = long_ & gate, short & gate
    return {**out, "long": long_, "short": short}


def held(e, x) -> np.ndarray:
    """The holding per bar from an entry / exit frame pair. An entry while already held is
    ``trade_frames``' re-entry marker (it re-opens a trade a stop closed): it starts no holding.

    Args:
        e: Entries (bars x names, bool).
        x: Exits of the same side.

    Returns:
        bars x names, True while held.
    """
    state = pd.DataFrame(np.where(e, 1.0, np.where(x, 0.0, np.nan))).ffill().fillna(0.0)
    return state.to_numpy() > 0


def holding_starts(sig: dict) -> np.ndarray:
    """Entries that start a holding: the bar before was not held on that side.

    Args:
        sig: ``strategy.trade_frames`` output (``entries`` / ``exits`` and the short pair).

    Returns:
        bars x names, True where a holding starts.
    """
    n1, m = sig["entries"].shape
    out = np.zeros((n1, m), bool)
    for ent, ext in (("entries", "exits"), ("short_entries", "short_exits")):
        e, h = sig[ent].to_numpy(), held(sig[ent].to_numpy(), sig[ext].to_numpy())
        out |= e & ~np.vstack([np.zeros((1, m), bool), h[:-1]])
    return out


def funnel(r: dict, stats: dict, panel, fc: dict, start: str) -> pd.Series:
    """How many forecasts survive each gate of a rule, the holdings they form and the trades the
    engine made, from ``start`` on.

    Args:
        r: The full trading rule.
        stats: The rule's backtest numbers (for ``n_trades``).
        panel: The panel the forecast was made on.
        fc: The forecast.
        start: First scored session ("YYYY-MM-DD").

    Returns:
        One count per stage.
    """
    d, sig = decisions(r, fc), SM["trade_frames"](panel, fc, **r)
    starts = holding_starts(sig)
    inp = fc["inputs"]
    edge_bp = fc["edge"] * inp.get("tscale", inp["scale"]) * 1e4
    dates = pd.to_datetime(pd.Series(inp["g"]["date"]))
    on_grid = (dates >= pd.Timestamp(start)).to_numpy()[:, None]  # grid bars from start on
    in_window = panel.close.index >= pd.Timestamp(start, tz="UTC")
    return pd.Series(
        {
            "forecasts (decision bars x names)": int((np.isfinite(edge_bp) & on_grid).sum()),
            "+ inside the trading hours": int((d["hours"] & on_grid).sum()),
            "+ |forecast| > edge_bp": int((d["edge"] & on_grid).sum()),
            "+ the health gate = the rule's decisions": int(
                ((d["long"] | d["short"]) & on_grid).sum()
            ),
            "holdings: runs of same-side decisions": int(starts[in_window].sum()),
            "trades the engine executed": int(stats["n_trades"]),
        }
    )


# ---------------------------------------------------------------------------------------------
# Backtests and their tables (the strategy and diagnostics notebooks).
# ---------------------------------------------------------------------------------------------
def scored(params: dict, *, strategy, p, start: str, sett) -> tuple:
    """One warm backtest: the portfolio plus ``bt.evaluate``'s numbers, from ONE run.

    ``bt._run_scored`` is ``bt.evaluate``'s first half (signals on the whole panel, the book from
    ``start``, fees from ``bt.strategy_costs`` at the strategy's order size, its fill model); the
    stats are its second half, with the benchmark at the same costs and fill (the strategy
    notebook's §13 checks that ``bt.evaluate`` itself returns the same numbers). The notebook binds
    its defaults with ``functools.partial``.

    Args:
        params: Parameter overrides on top of ``strategy.params``.
        strategy: The strategy to run.
        p: The panel.
        start: First scored session.
        sett: Settings (the cost block).

    Returns:
        ``(portfolio, stats)``.
    """
    pf, window, fees, slip = bt._run_scored(strategy, p, params, sett, start)
    stats = bt.portfolio_stats(pf)
    stats |= bt.active_stats(
        pf, window, fees=fees, slippage=slip, fill=strategy.fill, total_return=stats["total_return"]
    )
    return pf, stats


def with_costs(settings, mult: float = 1.0, spread: float | None = None,
               slippage: float | None = None):
    """A copy of ``settings`` with its (pydantic) cost block scaled or overridden.

    Args:
        settings: Global settings.
        mult: Multiplies the commission, its minimum, the spread and the slippage.
        spread: A quoted spread in bp instead of the scaled one.
        slippage: A slippage per leg in bp instead of the scaled one.

    Returns:
        The new settings.
    """
    k = settings.costs
    new = k.model_copy(
        update={
            "commission_per_share": k.commission_per_share * mult,
            "min_commission": k.min_commission * mult,
            "spread_bps": k.spread_bps * mult if spread is None else spread,
            "slippage_bps": k.slippage_bps * mult if slippage is None else slippage,
        }
    )
    return settings.model_copy(update={"costs": new})


def stress(label: str, params: dict | None = None, *, base: dict, panel, start: str, strategy,
           sett) -> dict:
    """One full ``bt.evaluate`` of a candidate under a changed condition, plus the mean net return
    of its trades in bp (costs and fill model included, as in the book). The diagnostics notebook
    binds the candidate, the panel and the window with ``functools.partial``.

    Args:
        label: The row's name.
        params: Parameter changes on top of ``base``.
        base: The candidate's parameters.
        panel: The panel.
        start: First scored session.
        strategy: The strategy (e.g. a copy with ``fill="pessimistic"``).
        sett: Settings (e.g. :func:`with_costs`).

    Returns:
        One row of the stress table.
    """
    prm = {**base, **(params or {})}
    st = bt.evaluate(strategy, panel, prm, settings=sett, score_from=start)
    pf = bt.run_strategy(strategy, panel, prm, settings=sett, score_from=start)
    keep = ("alpha_cagr", "cagr", "beta", "excess_cagr", "volatility", "max_drawdown", "win_rate")
    return {
        "case": label,
        **{m: st[m] for m in keep},
        "n_trades": st["n_trades"],
        "bp_per_trade": 1e4 * bt.trade_records(pf)["return"].mean(),
    }


def monthly(book, init_cash: float, tz: str) -> pd.Series:
    """A book's return per calendar month (in ``tz``), from its equity curve; the first month is
    measured from the starting cash.

    Args:
        book: A vectorbt portfolio.
        init_cash: The starting cash.
        tz: The panel's timezone (the months are New York months).

    Returns:
        One return per "YYYY-MM" (an unnamed index, so a table of several books shows none).
    """
    v = bt.equity_curve(book, init_cash=init_cash)["value"]
    end = v.groupby(v.index.tz_convert(tz).strftime("%Y-%m")).last()
    return (end / end.shift(1).fillna(init_cash) - 1.0).rename_axis(None)


def book(x: dict) -> dict:
    """The scorecard's numbers for one book, rounded to 4 decimals.

    Args:
        x: A book's stats.

    Returns:
        alpha_cagr, beta, excess_cagr, volatility, trades_per_year.
    """
    keys = ("alpha_cagr", "beta", "excess_cagr", "volatility", "trades_per_year")
    return {m: round(float(x[m]), 4) for m in keys}


def as_params(row, space: dict) -> dict:
    """A trial row's searched parameters, cast back to the types ``signals()`` expects.

    Args:
        row: A row of ``opt.trials_frame`` (or of the candidates menu).
        space: ``optuna.space``.

    Returns:
        ``{knob: value}``: a categorical as its declared choice (3, never 3.0), an int as an int,
        a float rounded to 10 places (0.85, not Optuna's 0.8500000000000001).
    """
    out = {}
    for k, spec in space.items():
        if spec["type"] == "categorical":  # the declared choice itself: 3, never 3.0
            out[k] = next(c for c in spec["choices"] if c == row[k])
        elif spec["type"] == "int":
            out[k] = int(round(float(row[k])))
        else:
            out[k] = round(float(row[k]), 10)  # 0.85, not Optuna's 0.8500000000000001
    return out


def candidate(choice, cand_path: Path) -> tuple[dict, str]:
    """The candidate's full parameter dict and where it came from.

    ``"config"`` = config.yaml; ``"pick"`` / a trial number = a row of the candidates file
    ``darts_forecast.ipynb`` writes: config.yaml, then the file's ``forecast`` block (the forecast
    knobs its study ran on), then the row's ``params`` (its searched trading-rule knobs).

    Args:
        choice: ``"config"``, ``"pick"`` or a trial number.
        cand_path: The candidates file (``studies/<STUDY>.candidates.yaml``).

    Returns:
        ``(params, source)``.

    Raises:
        FileNotFoundError: No candidates file (and ``choice`` is not ``"config"``).
        ValueError: The trial is not in the file.
    """
    if choice == "config":
        return dict(STRAT.params), "config.yaml"
    if not cand_path.exists():
        raise FileNotFoundError(f"no {cand_path.name}: run darts_forecast.ipynb (or use 'config')")
    snap = yaml.safe_load(cand_path.read_text()) or {}
    rows = {r["trial"]: r for r in snap.get("front") or []}
    want = snap.get("pick") if choice == "pick" else int(choice)
    if want not in rows:
        if choice != "pick":
            raise ValueError(f"trial {want} is not in {cand_path.name}: {sorted(rows)}")
        print(f"{cand_path.name} holds no row for the study's pick ({want}): using config.yaml")
        return dict(STRAT.params), f"config.yaml ({cand_path.name} has no admissible pick)"
    if not snap.get("admissible", True):
        print("note: the study marked its pick NOT admissible: no trial above its trade floor had")
        print("alpha > 0 on TUNE, so by its own rule this row should not be traded")
    prm = {**STRAT.params, **(snap.get("forecast") or {}), **rows[want]["params"]}
    return prm, f"{cand_path.name}, trial {want}"


def coverage(features: dict | None, options: str, a, b) -> pd.Series:
    """Share of (session, name) cells with data, sessions ``a`` .. ``b``: each option input the
    candidate reads, and its source's ``has_<source>`` flag (any of the source's inputs present:
    the 0/1 input that tells the model a missing source from a real 0).

    Args:
        features: ``panel.features``.
        options: The candidate's ``options`` knob (``"none"`` | ``"iv"`` | ``"all"``).
        a: First session.
        b: Last session.

    Returns:
        One share per ``(source, input)``.
    """
    lo, hi = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
    feats, out = features or {}, {}
    for src in SM["OPTION_SET_SOURCES"][options]:
        have = {
            n: feats[f"opt_{n}"].loc[lo:hi].notna().to_numpy()
            for n in SM["OPTION_SOURCES"][src]
            if f"opt_{n}" in feats
        }
        flag = np.logical_or.reduce(list(have.values())) if have else np.zeros(1, bool)
        out[(src, f"has_{src}")] = float(flag.mean())
        out.update({(src, f"opt_{n}"): float(h.mean()) for n, h in have.items()})
    return pd.Series(out, dtype=float)


def marks(frame: pd.DataFrame) -> set:
    """The ``(bar, name)`` cells where a signal frame is True.

    Args:
        frame: A boolean signal frame (bars x names).

    Returns:
        The set of ``(timestamp, name)`` pairs.
    """
    rows, cols = np.nonzero(frame.to_numpy())
    return set(zip(frame.index[rows], frame.columns[cols], strict=True))


def how_closed(r, open_: pd.DataFrame, slip: float, planned: dict, opposite: dict,
               following: dict, last_real: set) -> str:
    """How a trade ended, read off the book's own signal path.

    The stop: an exit its path did not plan, or a planned exit that did not fill at the bar's open
    (the stop fired inside that bar and the engine fills it first). Otherwise the session's last
    real bar (``eod_exit``), an opposite decision (whose side opens one bar later), or the hold ran
    out.

    Args:
        r: One row of ``bt.trade_records``.
        open_: The panel's open prices.
        slip: Slippage per leg (an order at the open fills at open x (1 ± slip)).
        planned: ``{"Long" | "Short": marks of that side's planned exits}`` (:func:`marks`).
        opposite: ``{"Long" | "Short": marks of the opposite side's entries}``.
        following: ``{bar: the next bar}``.
        last_real: Each session's last real 1-minute bar.

    Returns:
        "open at the end", "stop", "session close", "reversal" or "time exit".
    """
    if r["status"] != "Closed":
        return "open at the end"
    at_open = abs(abs(r["exit_price"] / open_.at[r["exit_ts"], r["symbol"]] - 1) - slip)
    if (r["exit_ts"], r["symbol"]) not in planned[r["direction"]] or not at_open < 1e-9:
        return "stop"
    if r["exit_ts"] in last_real:
        return "session close"
    if (following.get(r["exit_ts"]), r["symbol"]) in opposite[r["direction"]]:
        return "reversal"
    return "time exit"


# ---------------------------------------------------------------------------------------------
# Small date and format utilities (the notebooks' tables, labels and axes).
# ---------------------------------------------------------------------------------------------
def ymd(ts) -> str:
    """A date as YYYY-MM-DD.

    Args:
        ts: Anything ``pd.Timestamp`` reads.

    Returns:
        The date text.
    """
    return f"{pd.Timestamp(ts):%Y-%m-%d}"


def ny_day(ts, tz: str) -> pd.Timestamp:
    """A database timestamp as its calendar day in ``tz`` (naive midnight). A naive timestamp is
    read as UTC, like the database's.

    Args:
        ts: The timestamp.
        tz: The panel's timezone.

    Returns:
        The local day.
    """
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tz is None else ts
    return ts.tz_convert(tz).tz_localize(None).normalize()


def ny(ts, tz: str) -> pd.DatetimeIndex:
    """Timestamps as wall-clock time in ``tz``, naive (plotly draws naive times as given).

    Args:
        ts: tz-aware timestamps.
        tz: The panel's timezone.

    Returns:
        The naive local times.
    """
    return pd.DatetimeIndex(ts).tz_convert(tz).tz_localize(None)


def iso(t) -> list[str]:
    """Naive New York times as strings (plotly.js would shift epoch ms to the viewer's timezone;
    one day needs no thinning).

    Args:
        t: Naive timestamps.

    Returns:
        "YYYY-MM-DD HH:MM:SS" strings.
    """
    return list(t.strftime("%Y-%m-%d %H:%M:%S"))


def clock(pos, bar_minutes: int, end: bool = True) -> str:
    """New York clock time of a grid bar's close (``end=True``) or open, from its position in the
    session (0 = the bar from 09:30).

    Args:
        pos: The grid bar's position in its session.
        bar_minutes: Minutes per grid bar.
        end: The close (True) or the open (False).

    Returns:
        "HH:MM".
    """
    m = 570 + (int(pos) + int(end)) * bar_minutes
    return f"{m // 60:02d}:{m % 60:02d}"


def lag_txt(n: int) -> str:
    """The lags of an input as text: ``s`` is the decision grid bar.

    Args:
        n: Number of lags.

    Returns:
        "s (latest)" or "s-5 .. s (6)".
    """
    return "s (latest)" if n == 1 else f"s-{n - 1} .. s ({n})"


def pct(v) -> str:
    """A signed percentage, or "n/a" for None or a non-finite value.

    Args:
        v: A fraction.

    Returns:
        "+1.2%" or "n/a".
    """
    return "n/a" if v is None or not np.isfinite(v) else f"{v:+.1%}"


def fmt_space(spec: dict) -> str:
    """One readable line for an ``optuna.space`` entry.

    Args:
        spec: The entry (``type``, ``choices`` or ``low`` / ``high``, ``log``, ``step``).

    Returns:
        "a | b | c" or "low – high (log)" / "low – high, step s".
    """
    if spec["type"] == "categorical":
        return " | ".join(map(str, spec["choices"]))
    tail = " (log)" if spec.get("log") else (f", step {spec['step']}" if spec.get("step") else "")
    return f"{spec['low']} – {spec['high']}{tail}"


def cadence_label(c: int) -> str:
    """The model notebook's name of a retraining cadence (trading minutes): "once", "every
    session", "weekly", "every 20 sessions", "every 60 min (intraday)". (The strategy notebook
    uses :func:`cadence_name`, whose words differ.)

    Args:
        c: The cadence in trading minutes (0 = once).

    Returns:
        The label.
    """
    if c == 0:
        return "once"
    if c % 390:
        return f"every {c} min (intraday)"
    return {1: "every session", 5: "weekly"}.get(c // 390, f"every {c // 390} sessions")


def cadence_name(c: int) -> str:
    """The strategy notebook's name of a retrain cadence: "once", "daily", "every 5 sessions",
    "every 60 min". (The model notebook uses :func:`cadence_label`.)

    Args:
        c: The cadence in trading minutes (0 = once).

    Returns:
        The name.
    """
    if c == 0:
        return "once"
    if c % 390:
        return f"every {c} min"
    return "daily" if c == 390 else f"every {c // 390} sessions"


def sess(day: str, k: int, sessions: list[str]) -> str:
    """The session ``k`` sessions after ``day`` (``k < 0``: before).

    Args:
        day: A session ("YYYY-MM-DD").
        k: The offset in sessions.
        sessions: The panel's sessions, in order.

    Returns:
        The session.
    """
    return sessions[sessions.index(day) + k]


def sessions_span(a: int, b: int, sessions) -> str:
    """The span of session numbers ``a`` .. ``b`` as "first → last (n)", or "—" when it is empty.

    Args:
        a: First session number.
        b: Last session number.
        sessions: Session dates as text, by number.

    Returns:
        The span.
    """
    return f"{sessions[a]} → {sessions[b]} ({b - a + 1})" if b >= a else "—"


def fit_windows(r, val: int, sessions: list[str]) -> str:
    """One fit's TRAIN (and VALIDATION) sessions, as text.

    Args:
        r: A row of the fit log (``train_from``, ``train_last``, ``session``).
        val: VALIDATION sessions (0 for gbm).
        sessions: The panel's sessions, in order.

    Returns:
        "TRAIN a → b" (+ " + VALIDATION c → d").
    """
    out = f"TRAIN {r['train_from']} → {r['train_last']}"
    if val:
        out += (f" + VALIDATION {sess(r['train_last'], 1, sessions)} → "
                f"{sess(r['session'], -1, sessions)}")
    return out


# ---------------------------------------------------------------------------------------------
# Verdict helpers.
# ---------------------------------------------------------------------------------------------
def checks_table(checks: list[tuple]) -> pd.DataFrame:
    """``[(check, ok, detail)]`` as a table."""
    return pd.DataFrame(checks, columns=["check", "ok", "detail"])


def state(ok) -> str:
    """A check's outcome: "pass", "fail", or "not run" (None or NaN).

    Args:
        ok: The check's result.

    Returns:
        The outcome.
    """
    if ok is None or (isinstance(ok, float) and np.isnan(ok)):
        return "not run"
    return "pass" if bool(ok) else "fail"


def scorecard(payload: dict) -> None:
    """Print ONE machine-readable line: ``SCORECARD {json}`` (floats rounded at every depth)."""
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list | tuple):
            return [clean(x) for x in v]
        if isinstance(v, float | np.floating):
            return round(float(v), 4) if np.isfinite(v) else None
        return int(v) if isinstance(v, np.integer) else v
    print("SCORECARD " + json.dumps(clean(payload), default=str))


# ---------------------------------------------------------------------------------------------
# The walkthrough (darts_forecast_walkthrough.ipynb): 1-hour bars, a forecast 5 sessions ahead.
# ---------------------------------------------------------------------------------------------
#: The walkthrough's forecaster: 1-hour bars, the close 35 bars (5 sessions) later, nights
#: included, every fit on ALL earlier outcomes, a fresh fit before every session (its §0 says why).
WALK_DEFAULTS = {"model": "gbm", "bar_minutes": 60, "horizon": 35, "overnight": True,
                 "expanding": True, "train_days": 40, "retrain_minutes": 390, "lookback": 7,
                 "cov_lags": 1, "val_days": 0, "options": "iv"}
#: The forecaster's knobs in plain words (the walkthrough's knob table).
KNOB_WORDS = {
    "model": "the forecaster: gbm = gradient-boosted decision trees predicting the change",
    "bar_minutes": "minutes per bar, built from the stored 1-minute bars (60: 7 bars a session)",
    "horizon": "bars ahead to predict (35 hourly bars = 5 sessions = same clock time next week)",
    "overnight": "the change is measured close to close, so the nights are included",
    "lookback": "the model also sees its own target as it stood at the last N bars",
    "cov_lags": "hourly inputs enter with their last N bars (daily and option inputs: latest)",
    "train_days": "sessions of history required before the first fit",
    "expanding": "every fit learns from ALL earlier outcomes already known (no rolling window)",
    "retrain_minutes": "refit every N trading minutes (390 = before every session, 0 = once)",
    "val_days": "sessions held out for early stopping (0 = none; the trees have a fixed size)",
    "options": "option inputs (iv: implied volatility, skew, term structure, implied minus "
               "historical volatility, put/call volume from the OCC clearing house)",
    "use_daily": "include the daily inputs (1-, 5-, 20-day returns, volatility, gap ...)",
    "use_market": "include the other names' average return (a market proxy)",
    "vol_days": "sessions behind the volatility scale (one 'typical move')",
    "gbm_iter": "boosting rounds of the trees (fixed)",
    "seed": "the model's random seed",
}
#: Each input in plain words (the walkthrough's input table).
INPUT_WORDS = {
    "ret": "the last hour's return", "rv": "the last hour's volatility (from its 1-minute moves)",
    "range": "the last hour's high-low range",
    "clv": "where the last hour closed in its range (-1 = low, +1 = high)",
    "flow": "buying vs selling volume in the last hour",
    "rvol": "the last hour's volume vs the same hour of the previous 10 sessions",
    "ret_open": "the return since the session's open",
    "vwap_dist": "the price vs the session's volume-weighted average price",
    "mkt": "the other four names' average return in the last hour",
    "prev_ret": "yesterday's return", "ret5": "the last 5 sessions' return",
    "ret20": "the last 20 sessions' return", "vol20": "the last 20 sessions' volatility",
    "pos20": "where yesterday's close sits in the 20-day high-low range",
    "prev_clv": "where yesterday closed in its range",
    "prev_rvol": "yesterday's volume vs its 20-day average",
    "gap": "today's overnight gap (first trade vs yesterday's close)",
    "opt_atm_iv": "30-day at-the-money implied volatility (the move the option prices expect)",
    "opt_skew": "out-of-the-money put minus call implied volatility (25 delta): the price of "
                "crash protection",
    "opt_term": "near-expiry minus 30-day implied volatility",
    "opt_d_atm_iv": "change of the implied volatility since the previous day",
    "opt_iv_hv": "implied minus historical volatility (expected vs past moves)",
    "opt_cust_pc": "customers' put / call option volume (OCC, the options clearing house)",
    "opt_vol_surge": "option volume vs its 20-day average",
    "opt_oi_pc": "put / call open interest", "opt_vol_pc": "put / call option volume",
    "opt_d_oi": "change of the open interest", "opt_gex": "dealer gamma imbalance",
    "has_dolt_iv": "1 if the implied-volatility inputs (DoltHub data) have data (0: they "
                   "enter as 0)",
    "has_dolt_hv": "1 if implied minus historical volatility (DoltHub) has data",
    "has_occ": "1 if the OCC inputs have data", "has_oi": "1 if the open-interest inputs have data",
}
#: The unit each input is read in (the worked example's input table).
INPUT_UNITS = {
    "ret": "typical hourly moves", "rv": "typical hourly moves", "range": "typical hourly moves",
    "ret_open": "typical hourly moves", "vwap_dist": "typical hourly moves",
    "clv": "-1 (low) .. +1 (high)", "flow": "-1 (all selling) .. +1 (all buying)",
    "rvol": "log ratio (0 = normal)", "mkt": "typical hourly moves of the market",
    "prev_ret": "typical daily moves", "ret5": "typical 5-day moves",
    "ret20": "typical 20-day moves", "vol20": "log of the daily volatility (-3.0 = 5% a day)",
    "pos20": "0 (20-day low) .. 1 (20-day high)", "prev_clv": "-1 (low) .. +1 (high)",
    "prev_rvol": "log ratio (0 = normal)", "gap": "typical daily moves",
    "opt_atm_iv": "volatility a year (0.5 = 50%)", "opt_skew": "volatility a year",
    "opt_term": "volatility a year", "opt_d_atm_iv": "volatility a year",
    "opt_iv_hv": "volatility a year", "opt_cust_pc": "log ratio (0 = as many puts as calls)",
    "opt_vol_surge": "log ratio (0 = normal)",
}
_NAMES = {"edge": "model (gbm)", "p_up": "classifier P(up)", "b_up": "always up",
          "b_drift": "drift", "b_mom": "momentum", "b_rev": "reversal"}


def walk_params(**over) -> dict:
    """The walkthrough's parameters: config.yaml, then :data:`WALK_DEFAULTS`, then ``over``.

    Args:
        **over: Knobs to change (e.g. ``model="gbm_clf"`` or ``retrain_minutes=1950``).

    Returns:
        The full parameter dict (forecaster and trading-rule knobs).

    Raises:
        ValueError: The forecaster knobs are impossible (``strategy.forecast_valid``).
    """
    p = {**STRAT.params, **WALK_DEFAULTS, **over}
    if not SM["forecast_valid"](p):
        raise ValueError(f"impossible forecaster knobs: {model_params(p)}")
    return p


def knob_table(params: dict) -> pd.DataFrame:
    """The forecaster's knobs with their values and what they mean in plain words.

    Args:
        params: A parameter dict (:func:`walk_params`).

    Returns:
        One row per knob of :data:`KNOB_WORDS` (``value``, ``meaning``).
    """
    return pd.DataFrame([{"knob": k, "value": str(params[k]), "meaning": w}
                         for k, w in KNOB_WORDS.items()]).set_index("knob")


def input_kw(params: dict) -> dict:
    """``strategy.build_inputs``' keyword arguments from a parameter dict.

    Args:
        params: A parameter dict.

    Returns:
        ``bar_minutes``, ``horizon``, ``options``, ``use_daily``, ``use_market``, ``vol_days``,
        ``overnight``, ``seasonal``.
    """
    keys = ("bar_minutes", "horizon", "options", "use_daily", "use_market", "vol_days")
    return {**{k: params[k] for k in keys}, "overnight": params.get("overnight", False),
            "seasonal": params.get("seasonal", False)}


def inputs(panel, params: dict) -> dict:
    """``strategy.build_inputs`` with the forecaster knobs of ``params`` (what every forecast
    sees, on the hourly grid).

    Args:
        panel: The 1-minute panel.
        params: A parameter dict.

    Returns:
        The inputs dict.
    """
    return SM["build_inputs"](panel, **input_kw(params))


def split_dates(panel, params: dict) -> dict:
    """The walk-forward's milestones on this panel: the warm-up, the first fit, the last graded
    decision (its outcome, ``horizon`` bars later, is the window's last session).

    Args:
        panel: The 1-minute panel (one window).
        params: A parameter dict.

    Returns:
        ``sessions`` ("YYYY-MM-DD"), ``warmup`` (sessions), ``first_fit`` and ``last_decision``.
    """
    sess = sorted({f"{d:%Y-%m-%d}" for d in panel.close.index.tz_convert(panel.tz).date})
    warmup = max(21, params["vol_days"] + 1)
    n_pos = math.ceil(SM["SESSION_MINUTES"] / params["bar_minutes"])
    span = math.ceil(params["horizon"] / n_pos) if params.get("overnight") else 0
    return {"sessions": sess, "warmup": warmup, "first_fit": sess[warmup + params["train_days"]],
            "last_decision": sess[-1 - span]}


def outcome_table(inp: dict, params: dict, window="tune", *,
                  earnings: dict | None = None) -> pd.DataFrame:
    """The graded decisions before any model: every decision from the first fit on whose 5-day
    outcome is usable and inside ``window`` (the rows a forecast will be graded on), as a
    :func:`horizon_table` with a zero forecast.

    Args:
        inp: ``strategy.build_inputs`` output.
        params: A parameter dict (``vol_days``, ``train_days``).
        window: The graded window.
        earnings: :func:`earnings_dates`.

    Returns:
        The table.
    """
    n, m = inp["target"].shape
    edge = np.zeros((n, m))
    first = inp["g"]["sess_first"][max(21, params["vol_days"] + 1) + params["train_days"]]
    edge[:first] = np.nan
    nan = np.full((n, m), np.nan)
    fc = {"inputs": inp, "edge": edge, "p_up": nan, "q10": nan, "q90": nan, "fit": nan}
    return horizon_table(fc, window, earnings=earnings)


def nights(inp: dict, name: str, ts) -> dict:
    """How much of one decision's 5-day change happened while the market was closed: the change
    split into the overnight gaps inside its window and the trading hours.

    Args:
        inp: ``strategy.build_inputs`` output.
        name: The stock.
        ts: The decision time (New York).

    Returns:
        ``total`` (log change), ``nights`` (their sum) and ``gaps`` (each night's log change).
    """
    s, j, H = bucket_at(inp, ts), inp["names"].index(name), inp["horizon"]
    close, first = inp["close"][:, j], inp["g"]["first"]
    day = inp["y"][:, j] * inp["scale"][:, j]                       # a bar's open-session return
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.r_[np.nan, np.log(close[1:] / close[:-1])]
    gaps = [float(move[p] - day[p]) for p in range(s + 1, s + H + 1) if first[p]]
    return {"total": float(np.log(close[s + H] / close[s])), "nights": float(np.sum(gaps)),
            "gaps": gaps}


def earnings_dates(engine, names, start, end) -> dict:
    """Earnings report dates per name, from the shared database's ``earnings`` table.

    The table stores no time of day: a report after the close moves the next session's open, one
    before the open moves the same session, so :func:`horizon_table` flags both sessions.

    Args:
        engine: Database engine.
        names: The symbols.
        start: First report date (inclusive).
        end: Last report date (inclusive): the window's end, so no later date is read.

    Returns:
        ``{name: ["YYYY-MM-DD", ...]}`` (an empty list for a name without a report).
    """
    q = text("SELECT DISTINCT symbol, report_date FROM earnings WHERE symbol = ANY(:s) "
             "AND report_date >= :a AND report_date <= :b ORDER BY symbol, report_date")
    with engine.connect() as c:
        df = pd.read_sql(q, c, params={"s": list(names), "a": pd.Timestamp(start).date(),
                                       "b": pd.Timestamp(end).date()})
    return {n: [ymd(d) for d in df.loc[df["symbol"] == n, "report_date"]] for n in names}


def dividend_table(panel, factors: pd.DataFrame) -> pd.DataFrame:
    """The ex-dividend dates inside a dividend-adjusted panel (:func:`load` with ``adjust=True``):
    the sessions where the daily bars' adjustment factor steps, with the overnight gap into each
    before and after the adjustment.

    Args:
        panel: The ADJUSTED 1-minute panel.
        factors: :func:`adjustment_factors` of its names over the same span.

    Returns:
        One row per ex-date: ``name``, ``ex-date``, ``dividend %`` (the factor's step), ``raw gap
        %`` (the overnight move in the stored, unadjusted bars: the dividend looks like a loss) and
        ``adjusted gap %`` (what the model sees).
    """
    step = np.log(factors / factors.shift(1)).reindex(columns=panel.symbols)
    day = np.asarray(panel.close.index.tz_convert(panel.tz).strftime("%Y-%m-%d"))
    first = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])         # each session's first bar
    open_at = dict(zip(day[first], first, strict=True))
    cols = ["name", "ex-date", "dividend %", "raw gap %", "adjusted gap %"]
    rows = []
    for name in panel.symbols:
        prev = panel.close[name].ffill().to_numpy()
        for d, x in step[name].items():
            i = open_at.get(ymd(d))
            if not i or not abs(x) > 1e-5:                    # outside the panel, or no step
                continue
            gap = math.log(panel.open_[name].iloc[i] / prev[i - 1])
            rows.append(dict(zip(cols, [name, ymd(d), 100 * math.expm1(x),
                                        100 * math.expm1(gap - x), 100 * math.expm1(gap)],
                                 strict=True)))
    return pd.DataFrame(rows, columns=cols)


def overnight_share(inp: dict) -> pd.DataFrame:
    """How much of each name's move happens while the market is closed.

    A session's close-to-close return splits into the overnight gap (the previous close to the
    first trade) and the day (the first trade to the close); 5 sessions split into 5 gaps and 5
    days. The nights' share is var(nights) / (var(nights) + var(days)).

    Args:
        inp: ``strategy.build_inputs`` output.

    Returns:
        Per name, in %: the nights' share of the 1-day and of the 5-day variance, the typical
        overnight gap (RMS) and the typical 5-day move (SD of the 5-day close-to-close change).
    """
    code = inp["g"]["code"]
    day = pd.DataFrame(inp["y"] * inp["scale"]).groupby(code).sum(min_count=1)  # open -> close
    close = pd.DataFrame(inp["close"]).groupby(code).last()
    cc = np.log(close / close.shift(1))

    def share(x: pd.DataFrame) -> float:
        return float(x["night"].var() / (x["night"].var() + x["day"].var()))

    rows = {}
    for j, name in enumerate(inp["names"]):
        d = pd.DataFrame({"night": cc[j] - day[j], "day": day[j], "cc": cc[j]}).dropna()
        w = d.rolling(5).sum().dropna()
        rows[name] = {"nights' share of 1-day variance %": 100 * share(d),
                      "nights' share of 5-day variance %": 100 * share(w),
                      "typical overnight gap %": 100 * float(np.sqrt((d["night"] ** 2).mean())),
                      "typical 5-day move %": 100 * float(w["cc"].std())}
    return pd.DataFrame(rows).T


def decision_times(g: dict) -> pd.DatetimeIndex:
    """Each grid bar's decision time: its close in New York (naive), from the session clock. The
    30-minute last bar closes at 16:00, a half-day's last real bar at 13:00.

    Args:
        g: The grid (``inp["g"]``).

    Returns:
        One timestamp per grid bar.
    """
    R = g["bar_minutes"]
    day = pd.to_datetime(pd.Series(g["date"]).astype(str))
    half = day.dt.strftime("%Y-%m-%d").isin(SM["HALF_DAYS"]).to_numpy()
    minute = np.minimum(570 + (g["pos"] + 1) * R, 960)
    cut = half & (570 + g["pos"] * R < SM["HALF_DAY_CLOSE"])          # holds the 13:00 close
    minute = np.where(cut, np.minimum(minute, SM["HALF_DAY_CLOSE"]), minute)
    return pd.DatetimeIndex(day + pd.to_timedelta(minute, unit="min"))


def bucket_at(inp: dict, ts) -> int:
    """The grid bar whose decision time (:func:`decision_times`) is ``ts``.

    Args:
        inp: ``strategy.build_inputs`` output.
        ts: A New York time, e.g. ``"2025-11-12 13:30"``.

    Returns:
        The grid bar.

    Raises:
        KeyError: No grid bar closes at ``ts``.
    """
    hit = np.flatnonzero(decision_times(inp["g"]) == pd.Timestamp(ts))
    if not len(hit):
        raise KeyError(f"no grid bar closes at {ts}")
    return int(hit[0])


def base_rates(t: pd.DataFrame) -> pd.DataFrame:
    """The 5-day changes the forecasts are graded on: how often they were up, their mean and their
    spread, per name and pooled (the "wind" every direction call is made in).

    Args:
        t: A :func:`horizon_table`.

    Returns:
        Per name and ``all``: graded forecasts, independent weeks (graded sessions / the horizon
        in sessions; blank for ``all``: the names move together, so they are not a sum, see
        :func:`effective_n`), up share, mean % and SD % of the 5-day change.
    """
    span = int((t["target_session"] - t["session"]).max())

    def one(x: pd.DataFrame) -> dict:
        return {"graded forecasts": len(x), "independent weeks": x["session"].nunique() / span,
                "up share": float((x["r_pct"] > 0).mean()), "mean %": 100 * x["r_pct"].mean(),
                "SD %": 100 * x["r_pct"].std()}

    out = {n: one(x) for n, x in t.groupby("name")}
    out["all"] = {**one(t), "independent weeks": np.nan}
    return pd.DataFrame(out).T


def input_table(inp: dict, params: dict) -> pd.DataFrame:
    """Everything one forecast reads: each input's group, meaning, when it is known and how many of
    its past values the model sees.

    Args:
        inp: ``strategy.build_inputs`` output.
        params: The parameters (``model``, ``lookback``, ``cov_lags``).

    Returns:
        One row per input, plus the target's own lags, the calendar and the name;
        ``attrs["numbers"]``: the numbers one gbm forecast reads.
    """
    deep = params["model"] in SM["DEEP"]
    known = {"intraday": "at the bar's close", "market": "at the bar's close",
             "daily": "at the previous close", "options": "the evening before (<= 5 days old)",
             "flag": "with its source", "seasonal": "at the decision (completed outcomes only)"}

    def lags(group: str) -> int:
        return params["lookback"] if deep else (
            params["cov_lags"] if group in ("intraday", "market") else 1)

    rows = [("the 5-day change itself", "target", "as it stood at each of the last bars",
             "at the bar's close", params["lookback"])]
    for k in inp["past_names"]:
        grp = inp["groups"][k]
        rows.append((k, grp, INPUT_WORDS.get(k, ""),
                     "at today's open" if k == "gap" else known[grp], lags(grp)))
    rows += [(", ".join(inp["future_names"]), "calendar",
              "clock time, weekday and option-expiry week of the TARGET bar", "in advance", 1),
             ("is_<name>", "name", "which stock: one model learns all five", "always", 1)]
    out = pd.DataFrame(rows, columns=["input", "group", "meaning", "known when", "lags"])
    out.attrs["numbers"] = int(out["lags"].iloc[:-2].sum() + len(inp["future_names"])
                               + len(inp["names"]))
    return out.set_index("input")


def one_forecast_inputs(inp: dict, name: str, ts) -> pd.DataFrame:
    """The numbers one forecast reads, at its decision bar: each past input's latest value (a
    missing value enters as 0, with its flag at 0) and the target's last values.

    Args:
        inp: ``strategy.build_inputs`` output.
        name: The stock.
        ts: The decision time (New York).

    Returns:
        One row per input: ``group``, ``value`` (NaN = missing), ``unit`` (:data:`INPUT_UNITS`),
        ``meaning``.
    """
    s, j = bucket_at(inp, ts), inp["names"].index(name)
    vals = inp["past"][name][s]
    rows = [(k, inp["groups"][k], float(v),
             "1 = has data, 0 = missing" if k.startswith("has_") else INPUT_UNITS.get(k, ""),
             INPUT_WORDS.get(k, ""))
            for k, v in zip(inp["past_names"], vals, strict=True)]
    rows.insert(0, ("5-day change, last bar", "target", float(inp["target"][s, j]),
                    "typical 5-day moves", "the change of the 5 sessions ending now"))
    return pd.DataFrame(rows, columns=["input", "group", "value", "unit", "meaning"]).set_index(
        "input")


def fit_log(fc: dict) -> pd.DataFrame:
    """The walk-forward's fits in plain words: when each was made, what it learned from and how
    much independent evidence that was.

    A fit made before session T learns from every decision whose outcome is known by then: from
    the end of the warm-up to ``horizon`` bars before T (the last ones are still open). (The
    walk-forward log's ``samples`` also counts the hours of those open sessions, which carry no
    outcome yet.)

    Args:
        fc: A forecast.

    Returns:
        One row per fit: ``fit before``, ``learns from decisions`` (first and last date),
        ``decisions × names with a known outcome``, ``weeks of known outcomes per name`` and
        ``seconds``.
    """
    inp, log = fc["inputs"], fc["log"]
    g, H = inp["g"], inp["horizon"]
    first = {str(g["date"][b]): int(b) for b in g["sess_first"]}
    lo = log["train_from"].map(first).to_numpy(int)                  # the first decision
    T = log["retrain_bucket"].to_numpy(int)
    last = T - 1 - H                                                  # the last known decision
    known = [int(inp["valid"][a + H:b].sum()) for a, b in zip(lo, T, strict=True)]
    weeks = (g["code"][last] - g["code"][lo] + 1) * g["n_pos"] / H
    return pd.DataFrame({
        "fit before": log["session"].to_numpy(),
        "learns from decisions": [f"{g['date'][a]} → {g['date'][b]}" for a, b in
                                  zip(lo, last, strict=True)],
        "decisions × names with a known outcome": known,
        "weeks of known outcomes per name": weeks,
        "seconds": log["seconds"].to_numpy()}, index=pd.Index(log["fit"] + 1, name="fit"))


def _earn(inp: dict, rows: np.ndarray, cols: np.ndarray, earnings: dict | None) -> np.ndarray:
    """True where a forecast's window (decision, decision + horizon] holds a report's first move:
    the first bar of the report day's session or of the next one (no time of day is stored)."""
    out = np.zeros(len(rows), bool)
    g, H = inp["g"], inp["horizon"]
    sf = g["sess_first"]
    days = np.array([str(d) for d in g["date"][sf]])
    for j, name in enumerate(inp["names"]):
        for day in (earnings or {}).get(name, []):
            k = int(np.searchsorted(days, day))
            for b in sf[k:k + 2]:
                out |= (cols == j) & (rows < b) & (b <= rows + H)
    return out


def horizon_table(fc: dict, window=None, *, earnings: dict | None = None) -> pd.DataFrame:
    """:func:`table`'s rows (the same selection) with what a multi-day forecast is read by: the
    times, the prices and the forecast as a % change.

    Args:
        fc: A forecast (:func:`cached_forecast`).
        window: As :func:`table`: both the decision and its outcome inside it.
        earnings: :func:`earnings_dates`, to flag the windows that hold a report.

    Returns:
        :func:`table`'s columns plus ``bucket`` / ``target_bucket`` (grid bars), ``session`` /
        ``target_session`` (session numbers), ``pos`` (the bar of the session), ``ts`` /
        ``target_ts`` (decision and target time, New York), ``price_now`` / ``price_then``,
        ``f_pct`` (the forecast change, exp(edge x tscale) - 1), ``r_pct`` (the actual change),
        ``pred_price``, ``q10_price`` / ``q90_price`` (NaN without quantiles) and ``earn``.
    """
    inp = fc["inputs"]
    g, H = inp["g"], inp["horizon"]
    rows, cols = _cells(fc, window)
    times, sc = decision_times(g), inp["tscale"][rows, cols]
    now, then = inp["close"][rows, cols], inp["close"][rows + H, cols]

    def price(z: str) -> np.ndarray:
        return now * np.exp(fc[z][rows, cols] * sc)

    return _frame(fc, rows, cols).assign(
        bucket=rows, target_bucket=rows + H, session=g["code"][rows],
        target_session=g["code"][rows + H], pos=g["pos"][rows], ts=times[rows],
        target_ts=times[rows + H], price_now=now, price_then=then,
        f_pct=np.expm1(fc["edge"][rows, cols] * sc), r_pct=then / now - 1,
        pred_price=price("edge"), q10_price=price("q10"), q90_price=price("q90"),
        earn=_earn(inp, rows, cols, earnings))


def one_forecast(t: pd.DataFrame, name: str, ts, horizon: str = "5-day") -> pd.Series:
    """One forecast end to end, in dollars and percent: what the model said and what happened.

    Args:
        t: A :func:`horizon_table`.
        name: The stock.
        ts: The decision time (New York).
        horizon: The horizon in words (the "typical move" line).

    Returns:
        One value per line (times, prices, the forecast and the actual change, the error, whether
        the direction was right).
    """
    r = t[(t["name"] == name) & (t["ts"] == pd.Timestamp(ts))].iloc[0]
    err = r["pred_price"] - r["price_then"]
    out = {"decision time": f"{r['ts']:%Y-%m-%d %H:%M}", "price now": f"${r['price_now']:,.2f}",
           "forecast change": f"{r['f_pct']:+.2%}",
           "predicted price": f"${r['pred_price']:,.2f}",
           f"one typical {horizon} move": f"±{r['edge_bp'] / r['edge'] / 1e4:.1%}",
           "target time": f"{r['target_ts']:%Y-%m-%d %H:%M}",
           "actual price": f"${r['price_then']:,.2f}", "actual change": f"{r['r_pct']:+.2%}",
           "error (predicted - actual)": f"{'+' if err >= 0 else '-'}${abs(err):,.2f} "
                                         f"({r['pred_price'] / r['price_then'] - 1:+.1%})",
           "direction right?": "yes" if (r["f_pct"] > 0) == (r["r_pct"] > 0) else "no"}
    if np.isfinite(r["p_up"]):
        out["P(up), classifier"] = f"{r['p_up']:.0%}"
    return pd.Series(out, name=name)


def score_table(sc: pd.DataFrame, group: str = "all") -> pd.DataFrame:
    """:func:`scores` as one readable row per forecast: the direction numbers in %, the per-name
    IC with the centre of its luck band, and the z of skill and IC against luck.

    Args:
        sc: :func:`scores` output.
        group: The group to show.

    Returns:
        One row per forecast.
    """
    x = sc[sc["group"] == group]
    v, mu, z = (x.pivot_table(index="forecast", columns="metric", values=c, sort=False)
                for c in ("value", "null_mean", "z"))
    out = pd.DataFrame({"says up %": 100 * v["up_share"], "hit %": 100 * v["hit"],
                        "chance %": 100 * v["chance"], "skill (pts)": 100 * v["skill"],
                        "skill z": z["skill"], "per-name IC": v["ic_name"],
                        "IC luck centre": mu["ic_name"], "per-name IC z": z["ic_name"],
                        "pooled IC": v["ic_pooled"]})
    out.index = [_NAMES.get(i, i) for i in out.index]
    for a, b in _NAMES.items():
        out.index = [i.replace(a, b) for i in out.index]
    return out


def null_rows(null: pd.DataFrame, sc: pd.DataFrame, metric: str = "ic_name") -> pd.DataFrame:
    """The leak detector's seeds and the real forecast as :func:`scores`-like rows (each with its
    own luck band), for ``charts.dots``.

    Args:
        null: :func:`null_scores`.
        sc: :func:`scores` of the real forecast.
        metric: ``ic_name`` or ``skill``.

    Returns:
        One row per coin-flip market and one for the real market.
    """
    rows = [{"forecast": f"coin-flip market {s}", "metric": metric,
             **{k: r[f"{metric} {k}"] for k in ("value", "null_mean", "null_sd", "z")}}
            for s, r in null.iterrows()]
    real = pick(sc, "edge", metric)
    rows.append({"forecast": "real market (the model)", "metric": metric,
                 **{k: real[k] for k in ("value", "null_mean", "null_sd", "z")}})
    return pd.DataFrame(rows)


def null_table(null: pd.DataFrame) -> pd.DataFrame:
    """:func:`null_scores` in plain words: one row per coin-flip market.

    Args:
        null: :func:`null_scores`.

    Returns:
        ``says up %``, ``per-name IC`` with its luck band's centre, SD and z, ``skill (pts)`` and
        its z.
    """
    out = pd.DataFrame({"says up %": 100 * null["up_share"], "per-name IC": null["ic_name value"],
                        "luck centre": null["ic_name null_mean"],
                        "luck SD": null["ic_name null_sd"], "IC z": null["ic_name z"],
                        "skill (pts)": 100 * null["skill value"], "skill z": null["skill z"]})
    out.index = [f"coin-flip market {s}" for s in null.index]
    return out


def found_chance(edge: float, sd: float) -> float:
    """How often a true edge of size ``edge`` would be found (z >= 1.96) when luck's SD is ``sd``.

    Args:
        edge: The true edge (e.g. a per-name IC of 0.1).
        sd: The luck band's SD of the metric.

    Returns:
        The probability.
    """
    from scipy.stats import norm

    return float(norm.cdf(edge / sd - 1.96))


def gap_numbers(t: pd.DataFrame, sc: pd.DataFrame, ledger: pd.DataFrame,
                dividends: pd.DataFrame, prices: pd.DataFrame, panel) -> dict:
    """The numbers :func:`gaps_table` fills in, from this run's tables.

    Args:
        t: The graded :func:`horizon_table`.
        sc: Its :func:`scores`.
        ledger: :func:`effective_n`.
        dividends: :func:`dividend_table`.
        prices: :func:`price_table`.
        panel: The panel (its half-days).

    Returns:
        The dict :data:`GAPS` reads.
    """
    days = {f"{d:%Y-%m-%d}" for d in panel.close.index.tz_convert(panel.tz).date}
    by = t.groupby("name")
    move = by["price_then"].last() / by["price_now"].first() - 1      # over the graded window
    sq = t["r_pct"] ** 2
    nochange = prices["R² price (no change)"].dropna()
    return {"weeks": float(ledger.loc["independent weeks per name", "value"]),
            "mde": 2.8 * pick(sc)["null_sd"], "n_div": len(dividends),
            "max_div": float(dividends["dividend %"].max()) if len(dividends) else 0.0,
            "earn_share": float(t["earn"].mean()), "earn_sq": float(sq[t["earn"]].sum() / sq.sum()),
            "mu": float(move.get("MU", np.nan)), "coin": float(move.get("COIN", np.nan)),
            "up": float(pick(sc, "b_up", "hit")["value"]), "half": len(days & SM["HALF_DAYS"]),
            "warmup": max(21, STRAT.params["vol_days"] + 1), "rows": len(t),
            "eff": float(ledger.loc["independent outcomes", "value"]),
            "r2_lo": float(nochange.min()), "r2_hi": float(nochange.max()),
            "drift_pooled": float(pick(sc, "b_drift", "ic_pooled")["value"]),
            "drift_name": float(pick(sc, "b_drift", "ic_name")["value"])}


def verdict_words(v: dict, sc: pd.DataFrame, ledger: pd.DataFrame) -> str:
    """The verdict in plain sentences: the status with the primary number, its luck and its 95%
    band; the direction calls against chance; what this data could NOT rule out (the band's top:
    an after-the-data statement), with the edge the design would have found reliably (a
    before-the-data one) for scale.

    Args:
        v: :func:`verdict`.
        sc: The :func:`scores` it read.
        ledger: :func:`effective_n` (the independent weeks per name).

    Returns:
        The text.
    """
    p, s = v["primary"], pick(sc, "edge", "skill")
    hit, chance = pick(sc, "edge", "hit")["value"], pick(sc, "edge", "chance")["value"]
    weeks = float(ledger.loc["independent weeks per name", "value"])
    top = (f"no edge at all (the band's top is {p['hi']:+.2f})" if p["hi"] <= 0 else
           f"a small edge, up to a per-name IC of about {p['hi']:+.2f} (the top of its 95% band)"
           if p["hi"] < 0.2 else f"an edge as large as a per-name IC of {p['hi']:+.2f}")
    return (f"Verdict: {v['status'].upper()}. The primary number, the per-name rank correlation "
            f"of the forecast with the actual 5-day change, is {p['value']:+.3f}: luck alone "
            f"centres on {p['null_mean']:+.3f} with an SD of {p['null_sd']:.3f}, so z = "
            f"{p['z']:+.1f}, and its 95% band (the true values this data cannot rule out) runs "
            f"from {p['lo']:+.2f} to {p['hi']:+.2f}. "
            f"The model called the direction right {hit:.1%} of the time against {chance:.1%} "
            f"for a coin with the same up/down mix, a direction skill of {100 * s['value']:+.1f} "
            f"points (z = {s['z']:+.1f}). "
            f"What this data cannot rule out: {top}. For scale: with {weeks:.0f} independent "
            f"weeks per stock, only an edge of about {v['detectable']:.2f} would have been found "
            f"reliably (4 times in 5). Nothing here speaks for other months, other stocks or "
            f"trading costs.")


def walk_summary(params: dict, t: pd.DataFrame, sc: pd.DataFrame, ledger: pd.DataFrame,
                 null: pd.DataFrame, cad: pd.DataFrame, latest: pd.DataFrame, v: dict,
                 checks: pd.DataFrame) -> dict:
    """The walkthrough's machine-readable record (the ``SCORECARD`` line).

    Args:
        params: The parameters.
        t: The graded table.
        sc: Its :func:`scores`.
        ledger: :func:`effective_n`.
        null: :func:`null_scores`.
        cad: :func:`cadence_scores`.
        latest: :func:`latest_forecast`.
        v: :func:`verdict`.
        checks: :func:`checks_table` of every check.

    Returns:
        The payload for :func:`scorecard`.
    """
    def val(f, m):
        return pick(sc, f, m)["value"]

    p = pick(sc)
    return {
        "notebook": "darts_forecast_walkthrough",
        "forecaster": {k: params[k] for k in ("model", "bar_minutes", "horizon", "lookback",
                                              "cov_lags", "train_days", "expanding",
                                              "retrain_minutes", "options", "overnight")},
        "graded": [f"{t['ts'].min():%Y-%m-%d}", f"{t['ts'].max():%Y-%m-%d}"], "rows": len(t),
        "independent_outcomes": float(ledger.loc["independent outcomes", "value"]),
        "ic_name": p["value"], "ic_name_z": p["z"], "ic_band": [p["lo"], p["hi"]],
        "skill": val("edge", "skill"), "skill_z": pick(sc, "edge", "skill")["z"],
        "hit": val("edge", "hit"), "chance": val("edge", "chance"),
        "baselines_ic_name": {b: val(b, "ic_name") for b in ("b_drift", "b_mom", "b_rev")},
        "baselines_ic_name_z": {b: pick(sc, b, "ic_name")["z"] for b in ("b_drift", "b_mom",
                                                                          "b_rev")},
        "always_up_hit": val("b_up", "hit"),
        "leak_combined_z": null.attrs["combined_z"],
        "cadence_ic_name": {f: v_ for f, v_ in cad.query("metric == 'ic_name'")[
            ["forecast", "value"]].itertuples(index=False)},
        "detectable_ic": v["detectable"], "latest_forecast_pct": latest[
            "forecast change %"].round(2).to_dict(),
        "checks_pass": bool(checks["ok"].all()), "verdict": v["status"]}


def _col(t: pd.DataFrame, inp: dict) -> np.ndarray:
    """Each row's name as its column in the inputs."""
    return t["name"].map({n: i for i, n in enumerate(inp["names"])}).to_numpy()


def add_baselines(t: pd.DataFrame, inp: dict) -> pd.DataFrame:
    """The obvious guesses on the forecast's rows, each made with what was known at the decision,
    in the forecast's units (typical 5-day moves):

    * ``b_up``: always up;
    * ``b_drift``: the name's mean 5-day change over every outcome completed by the decision;
    * ``b_mom``: the last 5 days continue (the change over the ``horizon`` bars up to the decision);
    * ``b_rev``: they reverse (minus ``b_mom``);
    * ``base_rate``: the name's share of up outcomes completed by the decision (the fair P(up)).

    Args:
        t: A :func:`horizon_table`.
        inp: Its forecast's inputs.

    Returns:
        A copy of ``t`` with the five columns.
    """
    H = inp["horizon"]
    lab = np.where(inp["valid"], inp["target"], np.nan)     # an outcome is known at its label bar
    seen = np.cumsum(np.isfinite(lab), axis=0)
    den = np.where(seen > 0, seen, np.nan)
    drift = np.cumsum(np.nan_to_num(lab), axis=0) / den
    rate = np.cumsum(lab > 0, axis=0) / den
    close = inp["close"]
    mom = np.full_like(close, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mom[H:] = np.log(close[H:] / close[:-H]) / inp["tscale"][H:]
    s, j = t["bucket"].to_numpy(), _col(t, inp)
    return t.assign(b_up=1.0, b_drift=drift[s, j], b_mom=mom[s, j], b_rev=-mom[s, j],
                    base_rate=rate[s, j])


def add_p_up(t: pd.DataFrame, fc_clf: dict) -> pd.DataFrame:
    """The classifier's P(up) on the forecast's rows (same decision bars and names).

    Args:
        t: A :func:`horizon_table`.
        fc_clf: The ``gbm_clf`` forecast on the same panel and grid.

    Returns:
        A copy of ``t`` with ``p_up`` filled in (NaN where the classifier made no forecast).
    """
    return t.assign(p_up=fc_clf["p_up"][t["bucket"].to_numpy(), _col(t, fc_clf["inputs"])])


def flip_outcomes(inp: dict, t: pd.DataFrame, draws: int = 1000, seed: int = 0, *,
                  signs: np.ndarray | None = None, guesses: bool = True) -> dict:
    """Make-believe futures for the luck band. Every bar's real close-to-close move (a session's
    first bar holds the night) keeps its size but gets a random sign - ONE coin per bar for all
    names, so they still move together - and each row's outcome is the signed sum of its own
    ``horizon`` bars, in the row's units (divided by ``tscale``). Overlap, co-movement and the
    size of every move stay; direction and trend are gone.

    The obvious guesses made from the price path (:func:`add_baselines`' drift, momentum and
    reversal) are rebuilt inside every make-believe path as well: a guess and the outcomes that
    follow it are pieces of the same path, which pushes its no-skill score away from 0 in a short
    sample, and a band that kept the guess fixed would read that push as skill (or anti-skill).
    The model's forecasts stay fixed; the leak detector (:func:`null_scores`) re-runs the whole
    pipeline instead.

    Args:
        inp: The forecast's inputs.
        t: A :func:`horizon_table` (its ``bucket`` and ``name``).
        draws: Make-believe futures.
        seed: Random seed.
        signs: ``(draws, grid bars)`` signs to use instead of random ones (all +1 gives back
            ``t["real"]`` and :func:`add_baselines`' columns).
        guesses: Also rebuild the path guesses.

    Returns:
        ``{"outcome": (draws, len(t)) outcomes}`` and, with ``guesses``, ``b_drift`` / ``b_mom``
        / ``b_rev``: each guess as it would have been in each make-believe future (same shape).
    """
    H, close, ts = inp["horizon"], inp["close"], inp["tscale"]
    move = np.zeros_like(close)
    with np.errstate(invalid="ignore", divide="ignore"):
        move[1:] = np.nan_to_num(np.log(close[1:] / close[:-1]))
    s, j = t["bucket"].to_numpy(), _col(t, inp)
    scale, at_decision = ts[s, j], np.full_like(ts, np.nan)
    at_decision[H:] = ts[:-H]                           # each label's scale: at its decision
    if signs is None:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(draws, len(close)))
    out = {k: np.empty((len(signs), len(t)))
           for k in (("outcome", "b_drift", "b_mom") if guesses else ("outcome",))}
    for a in range(0, len(signs), 100):                     # 100 futures at a time
        cum = np.cumsum(signs[a:a + 100, :, None] * move[None], axis=1)
        out["outcome"][a:a + 100] = (cum[:, s + H, j] - cum[:, s, j]) / scale
        if guesses:
            back = np.full_like(cum, np.nan)                # the change over the last H bars
            back[:, H:] = cum[:, H:] - cum[:, :-H]
            out["b_mom"][a:a + 100] = back[:, s, j] / scale
            lab = np.where(inp["valid"], back / at_decision, np.nan)   # each completed outcome
            seen = np.cumsum(np.isfinite(lab), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                drift = np.cumsum(np.nan_to_num(lab), axis=1) / np.where(seen > 0, seen, np.nan)
            out["b_drift"][a:a + 100] = drift[:, s, j]
    if guesses:
        out["b_rev"] = -out["b_mom"]
    return out


def _spearman(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank correlation of forecast ``f`` with each row of ``y`` (sets x rows): ``f`` is one
    vector for every set, or one per set (sets x rows). NaN where ``f`` is constant."""
    from scipy.stats import rankdata

    rf = rankdata(np.atleast_2d(f), axis=1)
    rf -= rf.mean(axis=1, keepdims=True)
    ry = rankdata(y, axis=1)
    ry -= ry.mean(axis=1, keepdims=True)
    den = np.sqrt((ry ** 2).sum(axis=1) * (rf ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ry * rf).sum(axis=1) / den, np.nan)


def _metrics(f: np.ndarray, y: np.ndarray, names: np.ndarray) -> dict:
    """Direction metrics of forecast ``f`` (one per row, or one per row and outcome set) against
    each outcome set in ``y`` (sets x rows): per name, then averaged over the names, plus the
    pooled rank IC."""
    call, up = f > 0, y > 0
    per = {k: [] for k in ("up_share", "hit", "chance", "ic_name")}
    for n in np.unique(names):
        r = names == n
        q, b = call[..., r].mean(axis=-1), up[:, r].mean(axis=1)
        per["up_share"].append(np.broadcast_to(q, b.shape))
        per["hit"].append((up[:, r] == call[..., r]).mean(axis=1))
        per["chance"].append(q * b + (1 - q) * (1 - b))
        per["ic_name"].append(_spearman(f[..., r], y[:, r]))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {k: np.nanmean(np.stack(v), axis=0) for k, v in per.items()}
    out["skill"] = out["hit"] - out["chance"]
    out["ic_pooled"] = _spearman(f, y)
    return out


def _groups(t: pd.DataFrame, f: np.ndarray, by, bins: int) -> dict:
    """Row masks per group: all rows (``by=None``), each value of a column, or ``"confidence"``:
    ``bins`` equal groups of |forecast| within each name, weakest first."""
    if by is None:
        return {"all": np.ones(len(t), bool)}
    if by != "confidence":
        return {k: (t[by] == k).to_numpy() for k in sorted(t[by].unique())}
    pct = pd.Series(np.abs(f)).groupby(t["name"].to_numpy()).rank(pct=True).to_numpy()
    grp = np.minimum((pct * bins).astype(int), bins - 1)
    labels = ("weak", "medium", "strong") if bins == 3 else [f"{i + 1}/{bins}" for i in range(bins)]
    return {f"{i + 1}. {lab}": grp == i for i, lab in enumerate(labels)}


def scores(t: pd.DataFrame, flips: dict, *, cols=("edge", "b_up", "b_drift", "b_mom", "b_rev"),
           by=None, bins: int = 3, pairs=(("edge", "b_drift"), ("edge", "b_mom"))) -> pd.DataFrame:
    """Direction metrics of each forecast column, with the luck band of each.

    Metrics, per name and averaged over the names (so a name's trend never counts as skill):
    ``up_share`` (share of "up" calls), ``hit`` (share of right signs), ``chance`` (the hit
    rate of a coin calling "up" as often, in a market going up as often: q·b + (1-q)(1-b)),
    ``skill`` = hit - chance, ``ic_name`` (the rank correlation of forecast and outcome within
    each name: THE primary number) and ``ic_pooled`` (all rows at once: rewards a name's
    trend). The luck band re-scores each forecast against every make-believe future of
    :func:`flip_outcomes` - a path guess (drift, momentum, reversal) as rebuilt in that future,
    every other forecast as it is: ``null_mean`` / ``null_sd``, ``z`` = (value - null_mean) /
    null_sd and ``lo`` / ``hi`` = value -/+ 1.96 null_sd (the 95% band around the measured
    value). ``"a - b"`` rows are paired differences, judged against the differences in the same
    futures. A ``p_up`` column (the classifier) joins as P(up) - 0.5.

    Args:
        t: A :func:`horizon_table` with :func:`add_baselines` (and :func:`add_p_up`).
        flips: :func:`flip_outcomes` for ``t``'s rows.
        cols: The forecast columns.
        by: None, ``"confidence"`` (terciles of |forecast| within each name) or a column of ``t``.
        bins: Groups for ``"confidence"``.
        pairs: ``(a, b)`` forecasts to compare (both must be in ``cols``).

    Returns:
        Long format: ``forecast``, ``group``, ``metric``, ``value``, ``null_mean``,
        ``null_sd``, ``z``, ``lo``, ``hi``.
    """
    cols = list(cols)
    if "p_up" in t and t["p_up"].notna().all() and "p_up" not in cols:
        cols.append("p_up")
    y, names, res = t["real"].to_numpy()[None], t["name"].to_numpy(), {}
    for c in cols:
        f = t[c].to_numpy(float) - (0.5 if c == "p_up" else 0.0)
        fn = flips.get(c, f)                        # rebuilt in each future (a path guess) or fixed
        for grp, r in _groups(t, f, by, bins).items():
            res[c, grp] = (_metrics(f[r], y[:, r], names[r]),
                           _metrics(fn[..., r], flips["outcome"][:, r], names[r]))
    for a, b in pairs:
        for grp in sorted({g_ for c, g_ in res if c == a}):
            if (b, grp) in res:
                (ra, na), (rb, nb) = res[a, grp], res[b, grp]
                res[f"{a} - {b}", grp] = ({k: ra[k] - rb[k] for k in ra},
                                          {k: na[k] - nb[k] for k in na})
    rows = []
    with warnings.catch_warnings():                       # a constant forecast has no IC
        warnings.simplefilter("ignore", RuntimeWarning)
        for (c, grp), (real, null) in res.items():
            for k in ("up_share", "hit", "chance", "skill", "ic_name", "ic_pooled"):
                v, mu, sd = float(real[k][0]), float(np.nanmean(null[k])), float(
                    np.nanstd(null[k]))
                z = (v - mu) / sd if sd > 0 else np.nan
                rows.append({"forecast": c, "group": grp, "metric": k, "value": v,
                             "null_mean": mu, "null_sd": sd, "z": z, "lo": v - 1.96 * sd,
                             "hi": v + 1.96 * sd})
    return pd.DataFrame(rows)


def pick(sc: pd.DataFrame, forecast: str = "edge", metric: str = "ic_name",
         group: str = "all") -> pd.Series:
    """One row of :func:`scores` (``value``, ``null_mean``, ``null_sd``, ``z``, ``lo``, ``hi``).

    Args:
        sc: :func:`scores` output.
        forecast: The forecast column (or ``"a - b"``).
        metric: The metric.
        group: The group.

    Returns:
        The row.
    """
    hit = sc[(sc["forecast"] == forecast) & (sc["metric"] == metric) & (sc["group"] == group)]
    return hit.iloc[0]


def reliability(t: pd.DataFrame, col: str = "p_up", bins: int = 5,
                flips: dict | None = None) -> pd.DataFrame:
    """Is a probability of "up" honest? The forecasts in ``bins`` equal groups by P(up): the mean
    P(up) against the share that really went up, with error bars from ``flips`` (the measured
    share -/+ 1.96 SD of the shares in the make-believe futures).

    No luck band is given for the Brier score: the make-believe futures go up half the time, so
    they would expect any spread-out P(up) to lose to the trailing up share anyway.

    Args:
        t: A :func:`horizon_table` with :func:`add_baselines` and :func:`add_p_up`.
        col: The probability column.
        bins: Equal-count groups.
        flips: :func:`flip_outcomes` for ``t``'s rows (for the error bars).

    Returns:
        One row per group: ``mean P(up)``, ``real up share``, ``n``, ``lo`` / ``hi``. ``attrs``:
        ``brier`` (mean squared error of P(up) against up = 1 / down = 0), ``brier_coin`` (0.25,
        always 50%), ``brier_trailing`` (the name's past up share, ``base_rate``: the fair
        reference), ``skill_vs_trailing`` (1 - brier / brier_trailing: the improvement over
        it) and ``sharpness`` (SD of P(up)).
    """
    p, base = t[col].to_numpy(float), t["base_rate"].to_numpy(float)
    up = (t["real"] > 0).to_numpy()
    grp = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False).to_numpy()
    fu = flips["outcome"] > 0 if flips is not None else None
    rows = []
    for b in range(bins):
        m = grp == b
        sd = float(fu[:, m].mean(axis=1).std()) if fu is not None else np.nan
        share = float(up[m].mean())
        rows.append({"group": b + 1, "mean P(up)": float(p[m].mean()), "real up share": share,
                     "n": int(m.sum()), "lo": share - 1.96 * sd, "hi": share + 1.96 * sd})
    out = pd.DataFrame(rows).set_index("group")
    brier, trail = float(((p - up) ** 2).mean()), float(((base - up) ** 2).mean())
    out.attrs = {"brier": brier, "brier_coin": 0.25, "brier_trailing": trail,
                 "skill_vs_trailing": 1 - brier / trail, "sharpness": float(p.std())}
    return out


def price_table(t: pd.DataFrame) -> pd.DataFrame:
    """The predicted price against the actual price, next to "no change" (today's price as the
    forecast), per name and pooled - the numbers behind "a price chart flatters".

    Args:
        t: A :func:`horizon_table`.

    Returns:
        Per name and ``all``: ``R² price (model)`` / ``R² price (no change)`` (1 - squared error /
        the actual price's variance; per name only), ``error % (model)`` / ``error % (no
        change)`` (RMS price error in % of the actual price), ``Theil U`` (their ratio: below 1 =
        better than no change) and ``R² change`` (of the 5-day change, against forecasting 0).
    """
    def one(x: pd.DataFrame) -> dict:
        act = x["price_then"]

        def r2(p: pd.Series) -> float:
            return float(1 - ((act - p) ** 2).sum() / ((act - act.mean()) ** 2).sum())

        def err(p: pd.Series) -> float:
            return 100 * float(np.sqrt((((p - act) / act) ** 2).mean()))

        e_m, e_0 = err(x["pred_price"]), err(x["price_now"])
        return {"R² price (model)": r2(x["pred_price"]), "R² price (no change)": r2(x["price_now"]),
                "error % (model)": e_m, "error % (no change)": e_0, "Theil U": e_m / e_0,
                "R² change": float(1 - ((x["r_pct"] - x["f_pct"]) ** 2).sum()
                                   / (x["r_pct"] ** 2).sum())}

    out = {n: one(x) for n, x in t.groupby("name")}
    out["all"] = {**one(t), "R² price (model)": np.nan, "R² price (no change)": np.nan}
    return pd.DataFrame(out).T


def lag_profile(t: pd.DataFrame, name: str, max_lag: int = 10) -> pd.DataFrame:
    """How far the predicted price sits from the actual price k sessions before its target time,
    k = 0 .. ``max_lag``. A forecast that is "today's price, nudged" sits closest at k = the
    horizon in sessions: the price when it was made.

    Args:
        t: A :func:`horizon_table`.
        name: The stock.
        max_lag: The largest k.

    Returns:
        One row per k: ``sessions back``, ``typical gap %`` (mean |predicted / actual - 1|), ``n``.
    """
    x = t[t["name"] == name]
    price = {**x.set_index(["session", "pos"])["price_now"].to_dict(),
             **x.set_index(["target_session", "pos"])["price_then"].to_dict()}
    rows = []
    for k in range(max_lag + 1):
        past = np.array([price.get((s - k, p), np.nan)
                         for s, p in zip(x["target_session"], x["pos"], strict=True)])
        rows.append({"sessions back": k, "n": int(np.isfinite(past).sum()),
                     "typical gap %": 100 * float(np.nanmean(np.abs(x["pred_price"] / past - 1)))})
    return pd.DataFrame(rows)


def effective_n(t: pd.DataFrame) -> pd.DataFrame:
    """The evidence ledger: how many independent outcomes the graded rows are worth.

    Forecasts an hour apart share all but one hour of their outcome, so a name holds one
    independent outcome per horizon (5 sessions); the names move together, so they count as
    m / (1 + (m - 1) x their mean correlation).

    Args:
        t: A :func:`horizon_table`.

    Returns:
        One value per line of the ledger.
    """
    span = int((t["target_session"] - t["session"]).max())
    weeks = float((t.groupby("name")["session"].nunique() / span).mean())
    c = t.pivot_table(index="bucket", columns="name", values="r_pct").corr().to_numpy()
    m = len(c)
    rho = float(c[~np.eye(m, dtype=bool)].mean())
    eff = m / (1 + (m - 1) * rho)
    return pd.Series({"graded forecasts (rows)": len(t),
                      "sessions with graded decisions": t["session"].nunique(),
                      "independent weeks per name": weeks,
                      "names": m, "mean correlation between the names": rho,
                      "effective names": eff, "independent outcomes": weeks * eff},
                     name="value").to_frame()


def coverage_table(t: pd.DataFrame) -> pd.DataFrame:
    """How honest a quantile model's 80% range is (its 10% and 90% quantiles): how often the actual
    price ended inside, below and above it, and how wide it was - for all forecasts and split by
    earnings windows.

    Args:
        t: A :func:`horizon_table` of a quantile model (TiDE / TFT).

    Returns:
        Rows ``all`` / ``earnings windows`` / ``other``: ``forecasts``, ``inside %`` (honest:
        80), ``below %`` / ``above %`` (10 each), ``width %`` (of the price).
    """
    def one(x: pd.DataFrame) -> dict:
        a, lo, hi = x["price_then"], x["q10_price"], x["q90_price"]
        return {"forecasts": len(x), "inside %": 100 * ((a >= lo) & (a <= hi)).mean(),
                "below %": 100 * (a < lo).mean(), "above %": 100 * (a > hi).mean(),
                "width %": 100 * ((hi - lo) / x["price_now"]).mean()}

    parts = {"all": t, "earnings windows": t[t["earn"]], "other": t[~t["earn"]]}
    return pd.DataFrame({k: one(x) for k, x in parts.items() if len(x)}).T


def cadence_scores(panel, params: dict, t: pd.DataFrame, flips: dict,
                   cadences=(0, 1950, 390), window="tune") -> pd.DataFrame:
    """Does retraining help? The same forecaster refit once, weekly or daily (each walk-forward
    cached on disk), scored on exactly ``t``'s rows against the same make-believe futures;
    ``"<cadence> - once"`` rows are the paired differences.

    Args:
        panel: The panel.
        params: The parameters (``retrain_minutes`` is replaced).
        t: The reference :func:`horizon_table` (its rows and outcomes).
        flips: :func:`flip_outcomes` for ``t``'s rows.
        cadences: Retrain cadences in trading minutes, the control (0 = once) first.
        window: The graded window.

    Returns:
        :func:`scores` output with one forecast per cadence (:func:`cadence_label`);
        ``attrs["fits"]``: fits and fit seconds per cadence.

    Raises:
        ValueError: The cadences do not forecast the same rows.
    """
    tab, fits = t[["bucket", "name", "real", "session"]].copy(), {}
    keys = pd.MultiIndex.from_frame(t[["bucket", "name"]])
    for c in cadences:
        fc = cached_forecast(panel, {**params, "retrain_minutes": c})
        edge = horizon_table(fc, window).set_index(["bucket", "name"])["edge"]
        tab[cadence_label(c)] = edge.reindex(keys).to_numpy()
        fits[cadence_label(c)] = (len(fc["log"]), float(fc["log"]["seconds"].sum()))
    labels = list(fits)
    if tab[labels].isna().any().any():
        raise ValueError("the cadences do not forecast the same rows")
    out = scores(tab, flips, cols=labels, pairs=[(c, labels[0]) for c in labels[1:]])
    out.attrs["fits"] = fits
    return out


def cadence_words(cad: pd.DataFrame) -> str:
    """The retraining question answered in plain words, from :func:`cadence_scores`: does any
    schedule beat the control (training once) on the primary number, and how do the
    direction-skill differences read? Several differences share one control, so a |z| near 2
    among them is called borderline, never proof.

    Args:
        cad: :func:`cadence_scores` output (the control first).

    Returns:
        The text.
    """
    names = list(dict.fromkeys(cad["forecast"]))
    plain, diffs = [f for f in names if " - " not in f], [f for f in names if " - " in f]
    ic, sk = ({f: pick(cad, f, m) for f in names} for m in ("ic_name", "skill"))
    ctrl = plain[0]

    def vs(f: str, m: dict, scale: float = 1.0, unit: str = "") -> str:
        digits = 2 if scale == 1 else 1
        return f"{f.split(' - ')[0]} {scale * m[f]['value']:+.{digits}f}{unit} (z {m[f]['z']:+.1f})"

    out = (f"Retraining {'helped' if any(ic[f]['z'] >= 2 for f in diffs) else 'did not help'}. "
           f"Per-name IC (the primary number): "
           + ", ".join(f"{f} {ic[f]['value']:+.2f}" for f in plain)
           + f"; against training {ctrl}: " + ", ".join(vs(f, ic) for f in diffs)
           + (": all within luck. " if all(abs(ic[f]["z"]) < 2 for f in diffs) else ". "))
    out += (f"Direction skill against training {ctrl}: "
            + ", ".join(vs(f, sk, 100, " points") for f in diffs) + ". ")
    if any(abs(sk[f]["z"]) >= 1.96 for f in diffs):
        lean = "refitting chases noise" if np.mean([sk[f]["value"] for f in diffs]) < 0 else (
            "refitting helps")
        out += (f"These are {len(diffs)} related comparisons that share one control, so a |z| "
                f"near 2 is borderline: a hint that {lean}, not proof. ")
    c = sk[ctrl]
    return out + (f"Training {ctrl} itself: {100 * c['value']:+.1f} points of skill (z "
                  f"{c['z']:+.1f}), " + ("no better than chance either." if abs(c["z"]) < 2
                                         else "beyond luck."))


def null_scores(panel, params: dict, seeds=(0, 1, 2), window="tune", *,
                draws: int = 200) -> pd.DataFrame:
    """The leak detector: the whole pipeline (inputs, walk-forward, forecasts) on coin-flip copies
    of the market (:func:`sign_surrogate`: every 1-minute move gets a random sign, one per minute
    for all names), each scored like the real forecast against its OWN luck band. With no
    direction to find, every seed must land inside its band; a clearly positive average means
    something reads the future.

    Args:
        panel: The real panel.
        params: The parameters.
        seeds: Coin-flip markets (each walk-forward is cached on disk).
        window: The graded window.
        draws: Make-believe futures per seed.

    Returns:
        Per seed: ``up_share`` and, for ``ic_name`` and ``skill``, the value, the luck band's
        mean and SD, and z. ``attrs``: ``combined_z`` (the mean z x sqrt(seeds)) and ``clean``
        (combined_z < 2).
    """
    rows = []
    for s in seeds:
        fc = cached_forecast(sign_surrogate(panel, s), params, extra=f"-null{s}")
        t = horizon_table(fc, window)
        flips = flip_outcomes(fc["inputs"], t, draws, seed=s, guesses=False)
        sc = scores(t, flips, cols=("edge",), pairs=())
        sc = sc.set_index("metric")
        rows.append({"seed": s, "up_share": sc.loc["up_share", "value"],
                     **{f"{m} {k}": sc.loc[m, k] for m in ("ic_name", "skill")
                        for k in ("value", "null_mean", "null_sd", "z")}})
    out = pd.DataFrame(rows).set_index("seed")
    z = float(out["ic_name z"].mean() * math.sqrt(len(out)))
    out.attrs = {"combined_z": z, "clean": bool(z < 2)}
    return out


def _upto(panel, day: str):
    """The panel's 1-minute bars through session ``day`` (inclusive)."""
    keep = np.asarray(panel.close.index.tz_convert(panel.tz).strftime("%Y-%m-%d")) <= day
    return replace(panel, **{k: getattr(panel, k)[keep]
                             for k in ("open_", "high", "low", "close", "volume")})


def _fit_last(inp: dict, params: dict):
    """One fresh model fit on every label realized by the last bar (the walk-forward's training
    window, one step past its last fit: expanding = from the end of the warm-up)."""
    g, kind = inp["g"], params["model"]
    n, m = inp["target"].shape
    sess, sf = int(g["code"][-1]) + 1, g["sess_first"]               # the next session
    lo = sf[min(max(21, params["vol_days"] + 1), sess - 1)] if params["expanding"] else (
        sf[max(0, sess - params["train_days"])])
    est = SM["make_model"](kind, horizon=inp["horizon"], lookback=params["lookback"],
                           cov_lags=params["cov_lags"], past_names=inp["past_names"],
                           groups=inp["groups"], n_epochs=params["n_epochs"], val_days=0,
                           gbm_iter=params["gbm_iter"], device=params["device"],
                           seed=params["seed"], n_samples=(n - lo) * m)
    pc, fut = SM["_covariates"](inp, kind)
    kw = {"epochs": params["n_epochs"], "verbose": False} if kind in SM["DEEP"] else {}
    s0 = SM["series_start"](lo, params["lookback"], inp["horizon"],
                            params.get("lag_history", False))
    est.fit(SM["_series"](inp, n, kind, s0), past_covariates=pc, future_covariates=fut,
            sample_weight=SM["_weights"](inp, s0, n, params["recency_half_life"], first=lo),
            **kw)
    SM["_quiet"]()
    return est


def _predict_last(est, inp: dict, kind: str) -> tuple[np.ndarray, int]:
    """The fitted model's forecast from the last bar, for the point ``horizon`` bars later.

    Returns:
        ``(values, target grid bar)``: per name, the forecast (z units) or, for a classifier,
        P(up).
    """
    n = len(inp["g"]["code"])
    pc, fut = SM["_covariates"](inp, kind)
    pred = est.predict(n=1, series=SM["_series"](inp, n, kind, 0), past_covariates=pc,
                       future_covariates=fut,
                       predict_likelihood_parameters=kind == "gbm_clf" or kind in SM["DEEP"],
                       verbose=False)
    vals = np.array([p.values()[0] for p in pred])
    out = vals[:, -1] if kind == "gbm_clf" else (
        np.sort(vals, axis=1)[:, 2] if kind in SM["DEEP"] else vals[:, 0])
    return out, int(pred[0].time_index[0])


def refit_forecast(panel, params: dict, day: str, inp: dict) -> tuple[int, np.ndarray]:
    """The next session's forecasts from ONE model fit on the panel cut after ``day``
    (:func:`latest_forecast`'s fit): a model that physically lacks every later bar. In an honest
    walk-forward they equal its own forecasts of that session (``checks.refit_check``).

    Args:
        panel: The panel.
        params: The parameters.
        day: The last session the fit may see ("YYYY-MM-DD").
        inp: The whole panel's inputs (the forecasts read the next session's inputs).

    Returns:
        ``(the next session's first grid bar, (its bars, names) forecasts)``.
    """
    short = SM["build_inputs"](_upto(panel, day), **input_kw(params))
    g = inp["g"]
    first = int(g["sess_first"][int(short["g"]["code"][-1]) + 1])
    seg = SM["predict_segment"](_fit_last(short, params), inp, params["model"], first,
                                first + g["n_pos"] - 1)
    return first, seg["edge"]


def latest_forecast(panel, params: dict, *, clf: bool = False, asof: str | None = None):
    """The end of the pipeline: the forecast made at the LAST bar, about the bar ``horizon``
    later (whose outcome is not in the data). The walk-forward stops ``horizon`` bars before the
    end (it only forecasts decisions with a known outcome), so this fits ONE fresh model on every
    outcome known at the last bar, the walk-forward's own training window, and predicts with it.

    Args:
        panel: The panel.
        params: The parameters.
        clf: Also fit the ``gbm_clf`` classifier for P(up).
        asof: Forecast from the last bar of this session instead ("YYYY-MM-DD"): nothing after it
            is read.

    Returns:
        Per name: ``decision time``, ``target time`` (the same clock time ``horizon`` bars later,
        on business days: one session early when a market holiday falls in between, like the
        model's own calendar input, ``strategy.time_features``), ``price now``, ``forecast
        change %``, ``predicted price``, ``P(up)``, ``typical 10% change`` / ``typical 90%
        change`` (of the name's past 5-day changes, %).
        ``attrs["target_bar"]``: the grid bar the model's forecast is stamped at (last + horizon).
    """
    if asof is not None:
        panel = _upto(panel, asof)
    inp = SM["build_inputs"](panel, **input_kw(params))
    g, H = inp["g"], inp["horizon"]
    edge, at = _predict_last(_fit_last(inp, params), inp, params["model"])
    p_up = np.full(len(inp["names"]), np.nan)
    if clf:
        p_up = _predict_last(_fit_last(inp, {**params, "model": "gbm_clf"}), inp, "gbm_clf")[0]
    step = int(g["pos"][-1]) + H                       # time_features' clock past the panel end
    session = pd.offsets.CustomBusinessDay(holidays=sorted(SM["HOLIDAYS"]))
    day = pd.Timestamp(g["date"][-1]) + (step // g["n_pos"]) * session
    target = decision_times({**g, "pos": np.array([step % g["n_pos"]]), "date": np.array([day])})
    sc, now = inp["tscale"][-1], inp["close"][-1]
    past = inp["close"][H:] / inp["close"][:-H] - 1
    past = np.where(inp["valid"][H:], past, np.nan)
    out = pd.DataFrame({
        "decision time": decision_times(g)[-1], "target time": target[0], "price now": now,
        "forecast change %": 100 * np.expm1(edge * sc),
        "predicted price": now * np.exp(edge * sc), "P(up)": p_up,
        "typical 10% change": 100 * np.nanpercentile(past, 10, axis=0),
        "typical 90% change": 100 * np.nanpercentile(past, 90, axis=0)},
        index=pd.Index(inp["names"], name="name"))
    out.attrs = {"target_bar": at, "last_bar": len(g["code"]) - 1}
    return out


def verdict(sc: pd.DataFrame, null: pd.DataFrame, breakdowns: dict, *,
            test: pd.DataFrame | None = None) -> dict:
    """The walkthrough's pre-registered rule (its §0), applied literally to the gbm forecast.

    * "promising": the primary number (per-name rank IC) has luck z >= 2, the direction skill is
      above 0, the per-name IC beats every baseline's, the leak detector is clean, the per-name
      IC is above 0 in at least 4 of 5 names, and it survives dropping the earnings windows
      (z >= 1);
    * "no evidence of a large edge": z < 1 and the band excludes an IC of 0.2;
    * "inconclusive": anything else; "void" when the leak detector fires.

    A TEST read can only confirm a "promising" result (z >= 1.65 there); a miss is "not confirmed".

    Args:
        sc: :func:`scores` of the TUNE rows.
        null: :func:`null_scores`.
        breakdowns: ``{"name": scores(..., by="name"), "earnings": scores(..., by="earn")}``.
        test: :func:`scores` of the TEST rows, if read.

    Returns:
        ``{"status", "tests": {test: passed}, "primary": the primary row, "detectable": the
        IC edge this design would find 4 times in 5 (2.8 x the band's SD: a before-the-data
        number; what the data rule out is the band's top, ``primary["hi"]``)}``.
    """
    prim, skill = pick(sc, "edge", "ic_name"), pick(sc, "edge", "skill")
    base = [pick(sc, b, "ic_name")["value"] for b in ("b_drift", "b_mom", "b_rev")]
    per = breakdowns["name"].query("forecast == 'edge' and metric == 'ic_name'")["value"]
    calm = pick(breakdowns["earnings"], "edge", "ic_name", False)
    tests = {"primary z >= 2": prim["z"] >= 2,
             "direction skill above 0": skill["value"] > 0,
             "per-name IC above every baseline's": all(prim["value"] > b for b in base),
             "leak detector clean": null.attrs["clean"],
             "per-name IC above 0 in >= 4 of 5 names": int((per > 0).sum()) >= 4,
             "survives dropping the earnings windows (z >= 1)": calm["z"] >= 1}
    tests = {k: bool(v) for k, v in tests.items()}
    if not tests["leak detector clean"]:
        status = "void: the leak detector fired"
    elif all(tests.values()):
        status = "promising"
        if test is not None:
            t_ = pick(test, "edge", "ic_name")
            ok = t_["value"] > 0 and t_["z"] >= 1.65
            status += ", confirmed on TEST" if ok else " on TUNE, not confirmed on TEST"
    elif prim["z"] < 1 and prim["hi"] < 0.2:
        status = "no evidence of a large edge"
    else:
        status = "inconclusive"
    return {"status": status, "tests": tests, "primary": prim,
            "detectable": 2.8 * prim["null_sd"]}


#: The walkthrough's gap list: (area, gap, what this run measured, the fix). The measured text
#: is filled from the notebook's numbers (:func:`gaps_table`).
GAPS = [
    ("data", "short hourly history: 1-minute bars start 2025-06-18",
     "{weeks:.0f} graded weeks per name; the smallest edge found reliably (4 times in 5) is "
     "an IC of {mde:.2f}",
     "more names (about 510 symbols have 1-minute bars over TUNE, no backfill needed); a "
     "daily-bar model (daily bars since 2015, about 25x the evidence); an IBKR 1-hour backfill"),
    ("data", "the 1-minute bars are not dividend-adjusted",
     "{n_div} ex-dates in TUNE, the largest {max_div:.2f}%: fixed here (load(adjust=True))",
     "engine proposal: adjust intraday bars in load_panel with the daily factor"),
    ("data", "earnings: a few weeks carry the big moves, and no input knows a report is due",
     "{earn_share:.0%} of graded forecasts hold a report; they carry {earn_sq:.0%} of the "
     "squared moves", "an earnings calendar with report times; a 'report inside the window' "
     "input (known in advance)"),
    ("data", "one regime: a strong trend for most names",
     "MU {mu:+.0%}, COIN {coin:+.0%} over the graded window; 'always up' hits {up:.0%}",
     "judge skill within names and against drift (done here); more history"),
    ("data", "option open interest (MarketData) starts 2025-10-03, mid-TUNE",
     "options = 'iv' here (full coverage)", "use 'all' only once it covers the training window"),
    ("data", "no intraday market index (SPY 1-minute bars are not stored)",
     "the market input is the other four names", "store an index or ETF; or SPY daily inputs"),
    ("data", "half-days come from a hand list (strategy.HALF_DAYS)",
     "{half} half-days masked in TUNE; the list ends 2026-12-24", "extend it past 2026; engine "
     "proposal #32 (mark post-close bars)"),
    ("data", "the stored daily bars stop at 2026-09-11", "later sessions would stay unadjusted",
     "refresh the daily bars"),
    ("code", "the trading rule cannot hold overnight (valid() rejects overnight)",
     "this forecast cannot be backtested yet", "extend trade_frames to multi-day holds (and its "
     "forecast-to-bp conversion: it uses the bar scale, the forecast is in tscale units)"),
    ("code", "the warm-up rebuilds the daily inputs from 1-minute bars",
     "{warmup} sessions of TUNE are only warm-up", "daily inputs from the stored daily bars"),
    ("code", "the latest forecast needs its own fit (the walk-forward stops H bars early)",
     "research.latest_forecast does it here (its fit = the walk-forward's, checked in §6)",
     "fold it into strategy.py"),
    ("code", "the calendar past the data's end ignores holidays",
     "a market holiday in the next 5 sessions would put the latest forecast's target date, and "
     "the model's weekday input for it, one session early",
     "an NYSE holiday list beside HALF_DAYS, used by strategy.time_features (proposal)"),
    ("method", "overlapping 5-day outcomes: few independent ones",
     "{rows:,} graded rows are about {eff:.0f} independent outcomes",
     "luck bands from re-dealt futures (done); more names or daily bars for power"),
    ("method", "day-clustered t-statistics are wrong at this horizon",
     "measured 24-34% false alarms at a nominal 5%: never shown here",
     "the luck band (2-3% false alarms)"),
    ("method", "price charts flatter any forecast",
     "'no change' explains R² {r2_lo:.2f}-{r2_hi:.2f} of the price level",
     "judge the change, never the level (done)"),
    ("method", "drift in a trending sample looks like skill",
     "drift's pooled IC {drift_pooled:+.2f} vs per-name {drift_name:+.2f}",
     "per-name metrics and the drift baseline (done)"),
    ("method", "forking paths: every knob tried on TUNE is another look",
     "the knobs and the verdict rule were fixed in §0", "TEST once, with the knobs frozen"),
]


def gaps_table(measured: dict) -> pd.DataFrame:
    """The gaps this walkthrough found, with what this run measured for each.

    Args:
        measured: The numbers :data:`GAPS`' texts need (weeks, mde, n_div, max_div, earn_share,
            earn_sq, mu, coin, up, half, warmup, rows, eff, r2_lo, r2_hi, drift_pooled,
            drift_name).

    Returns:
        One row per gap: ``area``, ``gap``, ``measured here``, ``fix``.
    """
    return pd.DataFrame([(a, g_, m.format(**measured), f) for a, g_, m, f in GAPS],
                        columns=["area", "gap", "measured here", "fix"])
