"""Tori Trades' trendline bounce on 21 stocks: defaults, a knob grid and a placebo.

Fixed before the first run (2026-10-19, the user asked to run it for 20 more stocks than NOW,
report the learnings and explore the knobs, above all how close a low must come to a line to be
a bounce and how far through it price must go to be a break):

- **Stocks**: NOW and META MSFT NVDA GOOGL PLTR JPM GS COF LLY GEHC CAT GEV FLEX COST KMX EXPE
  FTI CCJ COIN RGTI: every sector, 1-minute candles since mid-2025 or earlier.
- **Window**: 2026-03-30 to 2026-09-29, the six months all of them have 1-minute candles for;
  the 15-minute trading chart, the line knobs at the notebook's defaults.
- **A · defaults**: ``band`` 0.25, ``room`` 0.5, ``give`` 0, sides long, short and both.
- **B · knob grid**, long and short separately: ``band`` (how close a low/high must come to the
  line to be a signal) x ``room`` (how far through the line the stop sits: the break that stops
  the trade) x ``give`` (how far beyond the line a close must be to end it) =
  {0.1, 0.25, 0.5, 1} x {0.25, 0.5, 1, 2} x {0, 0.5, 1} typical candles.
- **C · placebo**, long and short at the defaults: every line moved ``shift`` in
  {-2, -1, -0.5, 0.5, 1, 2} typical candles toward price. If her lines mark something real,
  moved lines should trade worse than the true ones (shift 0, from A).
- **Score**: a stock's score in a stretch is the compounded return of the trades entered in it
  (the sum of log(1 + return), costs included; a stock without trades scores 0); a setting's
  score is the mean over the 21 stocks. The knobs are **picked** on the trades entered before
  2026-06-29 (discovery) and **judged** on those entered from then on (validation).

Added after reading A to C (2026-10-19), fixed before running it:

- **D · two improvements from her own rules**, at the default knobs: ``trend`` 50 (a long only
  above the 50-session daily average, a short only below: "always trading with the trend"),
  sides long, short and both; ``min_span`` 2, 5, 10 or 20 sessions between a line's points A and
  B (only the higher charts' lines), long and short; and both filters together (``trend`` 50,
  ``min_span`` 5), both sides. A variant **counts** only if it beats the default of the same
  sides on discovery AND on validation.

Usage::

    uv run python strategies/tori_trades_trendline/study.py run [--workers 7] [--stages D]
    uv run python strategies/tori_trades_trendline/study.py report

Results go to ``var/tori_trades_trendline/study/`` (git-ignored): ``trades.pkl``, ``runs.pkl``
and ``report.txt``; ``cache/`` keeps every stock's drawn lines (``--stages`` reruns only those
stages and replaces their rows).
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "var" / "tori_trades_trendline" / "study"
_spec = importlib.util.spec_from_file_location("tori_bounce_study", HERE / "bounce.py")
bounce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bounce)

STOCKS = ["NOW", "META", "MSFT", "NVDA", "GOOGL", "PLTR", "JPM", "GS", "COF", "LLY", "GEHC",
          "CAT", "GEV", "FLEX", "COST", "KMX", "EXPE", "FTI", "CCJ", "COIN", "RGTI"]
START, END, SPLIT = "2026-03-30", "2026-09-29", "2026-06-29"
DEFAULTS = {"band": 0.25, "room": 0.5, "give": 0.0, "shift": 0.0, "trend": 0, "min_span": 0.0}
STAGES = "ABCD"


def configs() -> pd.DataFrame:
    """Every replay of the study: its stage, sides and trading knobs, one row per run (new
    stages are appended, so a run's number never changes)."""
    rows = [{"stage": "A", "sides": s, **DEFAULTS} for s in ("long", "short", "both")]
    for s, band, room, give in itertools.product(("long", "short"), (0.1, 0.25, 0.5, 1.0),
                                                 (0.25, 0.5, 1.0, 2.0), (0.0, 0.5, 1.0)):
        rows.append({"stage": "B", "sides": s, **DEFAULTS, "band": band, "room": room,
                     "give": give})
    for s, shift in itertools.product(("long", "short"), (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)):
        rows.append({"stage": "C", "sides": s, **DEFAULTS, "shift": shift})
    rows += [{"stage": "D", "sides": s, **DEFAULTS, "trend": 50} for s in ("long", "short", "both")]
    rows += [{"stage": "D", "sides": s, **DEFAULTS, "min_span": span}
             for s, span in itertools.product(("long", "short"), (2.0, 5.0, 10.0, 20.0))]
    rows.append({"stage": "D", "sides": "both", **DEFAULTS, "trend": 50, "min_span": 5.0})
    return pd.DataFrame(rows).rename_axis("run").reset_index()


def one_stock(symbol: str, stages: str = STAGES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The study's replays of ``stages`` on one stock (one process; the lines are drawn once per
    setup candle, kept in ``cache/<symbol>.pkl`` and shared by all its replays).

    Returns:
        ``(trades, runs)``: every trade, and one summary row per run, both tagged with the
        stock and the run.
    """
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    settings = load_settings()
    daily, minutes = bounce.td.load(make_engine(settings), symbol)
    end = pd.Timestamp(END)
    daily = daily[daily.index <= end]
    minutes = minutes[minutes.index.tz_localize(None).normalize() <= end]
    stored = OUT / "cache" / f"{symbol}.pkl"
    cache = pd.read_pickle(stored) if stored.exists() else {}
    trades, runs = [], []
    todo = configs()
    for row in todo[todo["stage"].isin(list(stages))].itertuples():
        result = bounce.replay(daily, minutes, start=START, costs=settings.costs,
                               sides=row.sides, band=row.band, room=row.room, give=row.give,
                               shift=row.shift, trend=row.trend, min_span=row.min_span,
                               cache=cache)
        trades.append(result["trades"].assign(stock=symbol, run=row.run))
        runs.append({"stock": symbol, "run": row.run, **result["summary"]})
    stored.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(cache, stored)
    return pd.concat(trades, ignore_index=True), pd.DataFrame(runs)


def run(workers: int, stages: str = STAGES) -> None:
    """Run the study's ``stages``, one stock per process, and save the trades and the run
    summaries (replacing the earlier rows of those stages' runs)."""
    OUT.mkdir(parents=True, exist_ok=True)
    began, trades, runs = time.time(), [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for symbol, (t, r) in zip(STOCKS, pool.map(one_stock, STOCKS, [stages] * len(STOCKS)),
                                  strict=True):
            trades.append(t)
            runs.append(r)
            print(f"{symbol}: {len(r)} runs, {len(t)} trades ({time.time() - began:.0f} s)",
                  flush=True)
    trades, runs = pd.concat(trades, ignore_index=True), pd.concat(runs, ignore_index=True)
    if (OUT / "trades.pkl").exists() and set(stages) != set(STAGES):
        done = set(runs["run"])
        old_t, old_r = pd.read_pickle(OUT / "trades.pkl"), pd.read_pickle(OUT / "runs.pkl")
        trades = pd.concat([old_t[~old_t["run"].isin(done)], trades], ignore_index=True)
        runs = pd.concat([old_r[~old_r["run"].isin(done)], runs], ignore_index=True)
    trades.to_pickle(OUT / "trades.pkl")
    runs.to_pickle(OUT / "runs.pkl")


def scores(trades: pd.DataFrame, since=None, until=None) -> pd.DataFrame:
    """Each run's score per stock: the compounded return (sum of log(1 + return)) of the trades
    entered in ``[since, until)``; one row per run, one column per stock, 0 without trades."""
    t = trades
    if since is not None:
        t = t[t["entry time"] >= pd.Timestamp(since)]
    if until is not None:
        t = t[t["entry time"] < pd.Timestamp(until)]
    per = np.log1p(t["return"]).groupby([t["run"], t["stock"]]).sum().unstack("stock")
    return per.reindex(index=configs()["run"], columns=STOCKS).fillna(0.0)


def report() -> str:
    """The study's tables, as text (also saved to ``report.txt``)."""
    trades, runs, cfg = (pd.read_pickle(OUT / "trades.pkl"), pd.read_pickle(OUT / "runs.pkl"),
                         configs().set_index("run"))
    full, disc, val = scores(trades), scores(trades, until=SPLIT), scores(trades, since=SPLIT)
    out = []

    def show(title, frame):
        out.append(f"\n## {title}\n\n{frame.to_string()}\n")

    def stats(rows):
        """Mean, standard error over stocks and count of positive stocks of score rows."""
        return pd.DataFrame({"mean": rows.mean(axis=1), "se": rows.std(axis=1, ddof=1)
                             / np.sqrt(rows.shape[1]), "stocks > 0": (rows > 0).sum(axis=1)})

    a = cfg.index[cfg["stage"] == "A"]
    r = runs[runs["run"].isin(a)].merge(cfg[["sides"]], left_on="run", right_index=True)
    per = r.pivot(index="stock", columns="sides", values="strategy return")
    per["buy and hold"] = r.groupby("stock")["buy and hold return"].first()
    per["trades both"] = r[r["sides"] == "both"].set_index("stock")["trades"]
    show("A · defaults: return over the window per stock (costs included)",
         per.reindex(STOCKS).round(3))
    t = trades[trades["run"].isin(a)].merge(cfg[["sides"]], left_on="run", right_index=True)
    show("A · defaults: trades by mode and side", t.groupby(["sides", "side"]).agg(
        trades=("R", "size"), per_stock=("R", lambda v: len(v) / len(STOCKS)),
        win_rate=("return", lambda v: (v > 0).mean()), mean_R=("R", "mean"),
        median_R=("R", "median"), mean_return=("return", "mean"),
        mean_candles=("candles", "mean")).round(3))
    show("A · defaults: score (log, mean over stocks) on the whole window",
         stats(full.loc[a]).assign(sides=cfg.loc[a, "sides"]).round(4))
    show("A · defaults: exits", t.groupby(["side", "reason"]).agg(
        trades=("R", "size"), mean_R=("R", "mean")).round(3))

    for side in ("long", "short"):
        b = cfg.index[(cfg["stage"] == "B") & (cfg["sides"] == side)]
        grid = cfg.loc[b, ["band", "room", "give"]].assign(
            discovery=disc.loc[b].mean(axis=1), validation=val.loc[b].mean(axis=1),
            whole=full.loc[b].mean(axis=1), val_stocks_pos=(val.loc[b] > 0).sum(axis=1))
        grid["val_rank"] = grid["validation"].rank(ascending=False).astype(int)
        show(f"B · {side}: the 10 best settings on discovery, and how they did on validation",
             grid.sort_values("discovery", ascending=False).head(10).round(4))
        default = grid[(grid["band"] == 0.25) & (grid["room"] == 0.5) & (grid["give"] == 0)]
        show(f"B · {side}: the default setting", default.round(4))
        rho = grid[["discovery", "validation"]].corr(method="spearman").iloc[0, 1]
        out.append(f"B · {side}: rank correlation of discovery and validation scores over the "
                   f"48 settings: {rho:+.2f}\n")
        for knob in ("band", "room", "give"):
            show(f"B · {side}: mean score by {knob} (over the other knobs)",
                 grid.groupby(knob)[["discovery", "validation", "whole"]].mean().round(4))

    for side in ("long", "short"):
        c = cfg.index[(cfg["sides"] == side) & ((cfg["stage"] == "C") | (cfg["stage"] == "A"))]
        tt = trades[trades["run"].isin(c)].merge(cfg[["shift"]], left_on="run", right_index=True)
        table = tt.groupby("shift").agg(trades=("R", "size"), mean_R=("R", "mean"),
                                        win_rate=("return", lambda v: (v > 0).mean()),
                                        mean_return=("return", "mean"))
        table = table.join(stats(full.loc[c]).set_index(cfg.loc[c, "shift"]))
        show(f"C · {side}: placebo, lines moved toward price by shift typical candles "
             "(0 = her lines)", table.round(4))

    d = cfg.index[cfg["stage"] == "D"]
    if d.isin(trades["run"]).any():
        base = {s: cfg.index[(cfg["stage"] == "A") & (cfg["sides"] == s)][0]
                for s in ("long", "short", "both")}
        rows = cfg.loc[[*base.values(), *d], ["sides", "trend", "min_span"]].assign(
            discovery=disc.mean(axis=1), validation=val.mean(axis=1), whole=full.mean(axis=1),
            stocks_pos=(full > 0).sum(axis=1),
            trades_per_stock=trades.groupby("run").size() / len(STOCKS))
        rows["beats default"] = [
            bool(r.discovery > rows.loc[base[r.sides], "discovery"]
                 and r.validation > rows.loc[base[r.sides], "validation"])
            if i not in base.values() else "(default)" for i, r in rows.iterrows()]
        show("D · improvements (rows 0-2 are the defaults); a variant counts when it beats the "
             "default of the same sides on discovery AND validation", rows.round(4))
    text = "".join(out)
    (OUT / "report.txt").write_text(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["run", "report"])
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--stages", default=STAGES, help="which stages to run, e.g. D")
    args = parser.parse_args()
    if args.what == "run":
        run(args.workers, args.stages)
    print(report())
    sys.exit(0)
