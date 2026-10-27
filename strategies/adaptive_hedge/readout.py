"""Read saved lab results for adaptive_hedge (no backtests run here).

    uv run python strategies/adaptive_hedge/readout.py <label> [<label> ...]

Per label and test: the lab's summary (alpha, P, beta, CAGR, buy-and-hold, SPY CAGR over the same
days, volatility, max drawdown and buy-and-hold's, late / 2x-cost and top-5-out alpha), the mean
long / short exposure, and the proposal's own bars (P2 drawdown ratio, P3 2022 net exposure on
2022-06-30 and the 2022 return against buy-and-hold).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/janmarti/Documents/code/stock-trading-simple")
spec = importlib.util.spec_from_file_location("lab", REPO / "strategies/proposals_lab/lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
KEY = "adaptive_hedge"


def _dd(r: pd.Series) -> float:
    curve = (1 + r).cumprod()
    return float((curve / curve.cummax() - 1).min())


def table(label: str) -> pd.DataFrame:
    res = lab.load_results(KEY, label)
    s = lab.summary(res)
    spy = lab.spy().pct_change(fill_method=None)
    rows = {}
    for t, col in s.items():
        rows_t = res[t]
        r = pd.concat([x["returns"] for x in rows_t])
        b = pd.concat([x["bench"] for x in rows_t]).reindex(r.index).fillna(0.0)
        sp = spy.reindex(r.index).fillna(0.0)
        years = lab.years_of(r.index)
        rows[t] = {"alpha": col["alpha"], "P": col["P"], "beta": col["beta"],
                   "cagr": col["cagr"], "B&H": col["B&H"],
                   "SPY": float(np.expm1(np.log1p(sp).sum() / years)),
                   "vol": col["vol"], "maxDD": col["maxDD"], "B&H maxDD": _dd(b),
                   "late": col.get("late alpha", np.nan),
                   "2x": col.get("2x costs alpha", np.nan),
                   "top5out": col["top5-out alpha"],
                   "long": float(np.mean([x["long"] for x in rows_t])),
                   "short": float(np.mean([x["short"] for x in rows_t])),
                   "orders/yr": col["orders/yr"]}
    return pd.DataFrame(rows).T


def year_2022(label: str) -> dict:
    """P3: net exposure on 2022-06-30 (T1's first period) and the 2022 calendar-year return
    against buy-and-hold."""
    res = lab.load_results(KEY, label)
    if "T1" not in res:
        return {}
    r = pd.concat([x["returns"] for x in res["T1"]])
    b = pd.concat([x["bench"] for x in res["T1"]]).reindex(r.index).fillna(0.0)
    inside = r.index.year == 2022
    first = res["T1"][0]
    return {"2022 book": float((1 + r[inside]).prod() - 1),
            "2022 B&H": float((1 + b[inside]).prod() - 1),
            "net exposure mean 2022-H1": first["long"] + first["short"]}


def paired_train(base: str, label: str, tests=("TB", "T3")) -> dict:
    """Paired alpha delta of ``label`` over ``base`` on the stitched TRAIN era (TB + T3), the
    same days and bootstrap blocks (``lab.paired``)."""
    def stitch(res):
        return lab.stitched([(r["returns"], r["bench"]) for t in tests if t in res
                             for r in res[t]])
    a, b = stitch(lab.load_results(KEY, label)), stitch(lab.load_results(KEY, base))
    idx = a["returns"].index
    return lab.paired(a["returns"], b["returns"].reindex(idx).fillna(0.0),
                      b["bench"].reindex(idx).fillna(0.0), b["years"])


if __name__ == "__main__":
    args = sys.argv[1:]
    base = None
    if "--base" in args:
        i = args.index("--base")
        base, args = args[i + 1], args[:i] + args[i + 2:]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for lbl in args:
            print(f"== {lbl}")
            print(table(lbl).to_string(float_format=lambda x: f"{x:+.3f}"))
            print({k: round(v, 3) for k, v in year_2022(lbl).items()})
            if base and lbl != base:
                print("TRAIN (TB+T3 stitched) vs", base, paired_train(base, lbl))
