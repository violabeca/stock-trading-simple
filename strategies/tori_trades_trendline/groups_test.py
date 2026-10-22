"""Do Tori Trades' lines win more often than lines moved off them, in some GROUP of stocks?

Step 2 of the user's plan (2026-10-20): look at more stocks, group them by how they move (the
``classic_signals`` stock groups), and check whether some group shows an edge, before any more
optimization. Step 1 (the win/loss objective, 21 stocks) found no edge overall and one weak hint:
on shorts with her exits, a stock's advantage over moved lines in the first half of the window
repeated in the second (rank correlation +0.48).

Fixed before the first replay:

- **Universe**: the 200 stocks with the most dollar volume per session among those with
  1-minute candles since before 2025-10-01 (the ``classic_signals`` pool), priced at $5 or
  more, with at least 100 sessions in the grouping window.
- **Groups**: 10, by the 11 price-character numbers of ``classic_signals/groups.py`` (scores of
  ranks, a Ward tree), measured on 2025-07-01 to 2026-03-27 only, before the test window; the
  grouping's stability (each half of that window alone) is reported beside it.
- **Test**: her default rules on the 15-minute chart (``bounce.replay``), longs and shorts apart,
  on her lines and on the same lines moved 0.5, 1 and 2 typical candles toward and away from price
  (the placebo), 2026-03-30 to 2026-09-28. Trades entered before 2026-06-29 are the discovery
  half, the others the validation half.
- **Measure**: per group, side and half, the win rate (a trade wins when it makes money after
  costs) on her lines minus on the moved lines, per stock; the group's mean of that over its
  stocks, with a t over the stocks. Beside it the user's score: wins minus losses per stock and
  week, and its excess over what the moved lines' win rate would give; and the net return per
  trade on her lines and on the moved ones.
- **Pass**: her lines win more often in both halves, with t >= 2 on discovery and t >= 1.65 on
  validation (20 group-side tests: about 0.5 pass discovery by luck, validation sorts them out).

Usage::

    uv run python strategies/tori_trades_trendline/groups_test.py universe
    uv run python strategies/tori_trades_trendline/groups_test.py run [--workers 7]
    uv run python strategies/tori_trades_trendline/groups_test.py report

Everything goes to ``var/tori_trades_trendline/groups/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "var" / "tori_trades_trendline" / "groups"
_spec = importlib.util.spec_from_file_location("tori_bounce_groups", HERE / "bounce.py")
bounce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bounce)
td = bounce.td

CHARACTER = ("2025-07-01", "2026-03-27")      # where the groups are measured: before the test
START, SPLIT, END = "2026-03-30", "2026-06-29", "2026-09-28"
STOCKS, GROUPS, MIN_SESSIONS, MIN_PRICE = 200, 10, 100, 5.0
SHIFTS = (0.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def _groups_module():
    """``classic_signals/groups.py`` (it loads its own siblings)."""
    spec = importlib.util.spec_from_file_location(
        "classic_signals_groups_for_tori", ROOT / "strategies" / "classic_signals" / "groups.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def universe() -> pd.DataFrame:
    """Pick the stocks and form the groups (see the module docstring); saves ``universe.pkl``
    (members with their group) and prints the groups and their quality."""
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    g = _groups_module()
    settings = load_settings()
    engine = make_engine(settings)
    pool = g.pool(engine)
    early = pool[pd.to_datetime(pool["first"]) < pd.Timestamp("2025-10-01", tz="UTC")]
    sums = g.compute(engine, early)                     # kept on disk: read, not recomputed
    begin, finish = (pd.Timestamp(d) for d in CHARACTER)
    cut = {s: f[(f.index >= begin) & (f.index <= finish)] for s, f in sums.items()}
    cut = {s: f for s, f in cut.items() if len(f) >= MIN_SESSIONS}
    costs = _costs(settings)
    feats = g.characters(cut, costs)
    feats = feats[feats["price"] >= MIN_PRICE].dropna(subset=list(g.FEATURES))
    feats = feats.sort_values("dollar volume per session", ascending=False).head(STOCKS)
    sc = g.scores(feats[list(g.FEATURES)])
    halves = [g.scores(g.characters({s: cut[s] for s in sc.index}, costs, part=p)
                       [list(g.FEATURES)]) for p in ("first half", "second half")]
    result = g.grouping(sc, GROUPS, halves=halves)
    quality = g.quality(sc, *halves, ks=[GROUPS])
    members = result["members"].join(feats[["price", "dollar volume per session"]])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"members": members, "groups": result["groups"], "quality": quality,
                  "centre": result["centre"]}, OUT / "universe.pkl")
    print(result["groups"].to_string())
    print(quality.round(3).to_string())
    return members


def _costs(settings) -> dict:
    """The classic_signals cost dictionary (its ``simulate.costs``)."""
    spec = importlib.util.spec_from_file_location(
        "classic_signals_simulate_for_tori",
        ROOT / "strategies" / "classic_signals" / "simulate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.costs(settings)


def one_stock(symbol: str) -> pd.DataFrame:
    """Every replay of the test on one stock: her rules, long and short apart, her lines and the
    moved ones; one row per trade, tagged with the side traded and the shift."""
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    settings = load_settings()
    try:
        daily, minutes = td.load(make_engine(settings), symbol)
    except (ValueError, KeyError):
        return pd.DataFrame()
    end = pd.Timestamp(END)
    daily = daily[daily.index <= end]
    minutes = minutes[minutes.index.tz_localize(None).normalize() <= end]
    sessions = td.clock(daily)
    book = bounce.Chartbook(daily, minutes, list(td.TIMEFRAMES)[:6], sessions)
    cache, out = {}, []
    for sides in ("long", "short"):
        for shift in SHIFTS:
            try:
                r = bounce.replay(daily, minutes, start=START, costs=settings.costs,
                                  sides=sides, shift=shift, book=book, cache=cache)
            except ValueError:                    # no candle before the start, or none after
                return pd.DataFrame()
            t = r["trades"]
            if len(t):
                out.append(t[["side", "entry time", "exit time", "return", "R", "candles",
                              "reason"]].assign(stock=symbol, shift=shift))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run(workers: int) -> None:
    """Replay every stock of the universe, a stock per process; saves ``trades.pkl``."""
    members = pd.read_pickle(OUT / "universe.pkl")["members"]
    began, parts = time.time(), []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, t in enumerate(pool.map(one_stock, members.index), start=1):
            parts.append(t)
            if i % 10 == 0:
                print(f"{i} of {len(members)} stocks ({time.time() - began:.0f} s)", flush=True)
    trades = pd.concat([p for p in parts if len(p)], ignore_index=True)
    trades.to_pickle(OUT / "trades.pkl")
    print(f"done: {trades['stock'].nunique()} stocks, {len(trades)} trades, "
          f"{time.time() - began:.0f} s")


def _test(trades: pd.DataFrame) -> pd.DataFrame:
    """Per stock: trades, win rate on her lines and on the moved lines, their difference,
    wins minus losses per week and its excess over the moved lines' win rate, net bp per trade."""
    weeks = {"discovery": 13.0, "validation": 13.0}
    rows = []
    for (stock, half), g in trades.groupby(["stock", "half"]):
        hers, moved = g[g["shift"] == 0], g[g["shift"] != 0]
        if not len(hers) or not len(moved):
            continue
        p_hers, p_moved = (hers["return"] > 0).mean(), (moved["return"] > 0).mean()
        signs = np.where(hers["return"] > 0, 1, -1).sum()
        rows.append({"stock": stock, "half": half, "trades": len(hers),
                     "win rate": p_hers, "moved win rate": p_moved, "edge": p_hers - p_moved,
                     "wins-losses/week": signs / weeks[half],
                     "excess wins-losses/week": (signs - len(hers) * (2 * p_moved - 1))
                     / weeks[half],
                     "bp/trade": 1e4 * hers["return"].mean(),
                     "moved bp/trade": 1e4 * moved["return"].mean()})
    return pd.DataFrame(rows)


def report() -> str:
    """The test's tables, as text (also saved to ``report.txt``)."""
    stored = pd.read_pickle(OUT / "universe.pkl")
    members = stored["members"]
    trades = pd.read_pickle(OUT / "trades.pkl")
    trades["half"] = np.where(trades["entry time"] < pd.Timestamp(SPLIT), "discovery",
                              "validation")
    out = [f"\n## The groups (measured {CHARACTER[0]} to {CHARACTER[1]})\n\n"
           f"{stored['groups'].to_string()}\n\n{stored['quality'].round(3).to_string()}\n"]
    rows = []
    for side in ("long", "short"):
        per = _test(trades[trades["side"] == side]).merge(
            members[["group"]], left_on="stock", right_index=True)
        for label, part in [("all", per), *[(str(int(k)), v) for k, v in per.groupby("group")]]:
            line = {"side": side, "group": label}
            for half in ("discovery", "validation"):
                h = part[part["half"] == half]
                e = h["edge"]
                t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 2 else np.nan
                line |= {f"{half} stocks": len(h), f"{half} trades/stock/week":
                         h["trades"].mean() / 13.0, f"{half} win rate": h["win rate"].mean(),
                         f"{half} edge": e.mean(), f"{half} t": t,
                         f"{half} excess W-L/week": h["excess wins-losses/week"].mean(),
                         f"{half} bp/trade": h["bp/trade"].mean(),
                         f"{half} moved bp/trade": h["moved bp/trade"].mean()}
            line["passes"] = bool(line["discovery edge"] > 0 and line["discovery t"] >= 2
                                  and line["validation edge"] > 0
                                  and line["validation t"] >= 1.65)
            rows.append(line)
    table = pd.DataFrame(rows).set_index(["side", "group"])
    for half in ("discovery", "validation"):
        cols = [c for c in table.columns if c.startswith(half)]
        part = table[cols].rename(columns=lambda c: c.split(" ", 1)[1]).round(3)
        out.append(f"\n## {half}: her lines against the moved lines (edge = win rate minus the "
                   f"moved lines' win rate, mean over the group's stocks; t over the stocks)\n\n"
                   f"{part.to_string()}\n")
    out.append("\n## Passes (t >= 2 on discovery and t >= 1.65 on validation, both edges > 0)"
               f"\n\n{table['passes'].to_string()}\n")
    text = "".join(out)
    (OUT / "report.txt").write_text(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("what", choices=["universe", "run", "report"])
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    if args.what == "universe":
        universe()
    elif args.what == "run":
        run(args.workers)
    else:
        print(report())
