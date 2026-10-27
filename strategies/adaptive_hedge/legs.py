"""Leg attribution of a saved adaptive_hedge label (re-runs its book with ``lab.book``; no new
variant). Per test and calendar year: each leg's daily contribution — the name's weight in the
book at the previous close (asset value / book value) times its return — stitched, as an
annualized arithmetic contribution, with the leg's beta to the equal-weight market and its alpha
(contribution − beta × market). Long + short − costs ≈ the book.

    uv run python strategies/adaptive_hedge/legs.py <label> [TB,T3,T1,T2]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/janmarti/Documents/code/stock-trading-simple")
spec = importlib.util.spec_from_file_location("lab", REPO / "strategies/proposals_lab/lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
KEY = "adaptive_hedge"


def _stats(x: pd.Series, mkt: pd.Series) -> dict:
    x, mkt = x.align(mkt, join="left", fill_value=0.0)
    var = float(mkt.var())
    beta = float(x.cov(mkt) / var) if var > 0 else np.nan
    ann = 252.0
    return {"contrib": float(x.mean() * ann), "beta": beta,
            "alpha": float((x.mean() - beta * mkt.mean()) * ann)}


def legs(label: str, tests=("TB", "T3", "T1", "T2")) -> dict:
    res = lab.load_results(KEY, label)
    strat = lab.load_strategy(KEY, res["meta"]["overrides"])
    settings = lab._settings()
    out, daily = {}, {}
    for t in tests:
        parts = []
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            if res["meta"].get("hedge"):
                p = lab.with_spy(p)
            r = lab.book(strat, p, settings, te0, signals=strat.signals(p, **strat.params))
            pf, w = r["pf"], r["window"]
            held = pf.asset_value(group_by=False).div(pf.value(), axis=0).shift(1).fillna(0.0)
            rets = w.close.pct_change(fill_method=None).reindex(held.index).fillna(0.0)
            contrib = held * rets[held.columns]
            parts.append(pd.DataFrame({
                "long": contrib.where(held > 0, 0.0).sum(axis=1),
                "short": contrib.where(held < 0, 0.0).sum(axis=1),
                "book": r["returns"], "market": r["bench"],
                "short gross": -held.clip(upper=0).sum(axis=1),
                "long gross": held.clip(lower=0).sum(axis=1)}))
        d = pd.concat(parts)
        daily[t] = d
        row = {leg: _stats(d[leg], d["market"]) for leg in ("long", "short", "book")}
        row["costs + cash"] = float((d["book"] - d["long"] - d["short"]).mean() * 252)
        row["gross"] = {"long": float(d["long gross"].mean()),
                        "short": float(d["short gross"].mean())}
        years = {}
        for y, g in d.groupby(d.index.year):
            years[int(y)] = {"long": round(float(g["long"].sum()), 3),
                             "short": round(float(g["short"].sum()), 3),
                             "book": round(float((1 + g["book"]).prod() - 1), 3),
                             "market": round(float((1 + g["market"]).prod() - 1), 3)}
        row["years (sum of daily contributions)"] = years
        out[t] = row
    print(json.dumps(out, indent=1, default=float))
    path = lab.RESULTS / KEY / f"legs-{label}.pkl"
    pd.to_pickle(daily, path)
    return out


if __name__ == "__main__":
    legs(sys.argv[1], tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else
         ("TB", "T3", "T1", "T2"))
