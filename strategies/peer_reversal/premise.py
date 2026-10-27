"""Premise check for peer_reversal (RESEARCH_LOG §0, step 2): TRAIN periods only.

Does the peer-relative 5-day move predict the next 5 or 10 sessions against the market, and is
the effect larger on stress days? For every TRAIN period (E0, TB, T3) and decision bar t in it:

* the signal: ``compose()``'s signal with the config's values (``peers`` cluster / sector /
  market);
* the outcome: the beta-adjusted return from the fill, ``open[t+1] → open[t+1+h]``, minus the
  name's trailing 126-bar beta times the equal-weight mean of the same returns;
* the daily Spearman IC of −signal against the outcome (positive = reversal), and the
  bottom-tenth minus top-tenth mean outcome (the decile spread).

The t-statistics are Newey-West with ``h`` lags (overlapping outcomes). Run from the repo root:
``uv run python strategies/peer_reversal/premise.py [h ...]``.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys

import numpy as np
import pandas as pd

REPO = "/Users/janmarti/Documents/code/stock-trading-simple"


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lab = _load("lab", f"{REPO}/strategies/proposals_lab/lab.py")
S = _load("peer_reversal_strategy", f"{REPO}/strategies/peer_reversal/strategy.py")


def nw_t(x: pd.Series, lags: int) -> float:
    """Newey-West t-statistic of the mean of ``x`` (Bartlett weights)."""
    x = x.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(var / n))


def daily_rows(test: str, te0: str, te1: str, horizons, **over) -> pd.DataFrame:
    """One row per decision bar of a period: IC and decile spread per horizon, stress flag."""
    strat = lab.load_strategy("peer_reversal")
    p = lab.panel(test, te0, lab.strategy_features(strat))
    keys = inspect.signature(S.compose).parameters
    c = S.compose(p, **{k: v for k, v in {**strat.params, **over}.items() if k in keys})
    ret = p.close.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    beta = ret.rolling(126, min_periods=63).cov(mkt).div(mkt.rolling(126, min_periods=63).var(),
                                                          axis=0)
    idx = p.close.index
    inside = (idx >= pd.Timestamp(te0, tz="UTC")) & (idx <= pd.Timestamp(te1, tz="UTC"))
    rank = c["rank"]
    out = {"stress": c["stress"][inside]}
    for h in horizons:
        fwd = p.open_.shift(-(1 + h)) / p.open_.shift(-1) - 1.0
        fwd = fwd.where(c["eligible"])
        adj = fwd.sub(beta.mul(fwd.mean(axis=1), axis=0))
        sig = -c["signal"].where(c["eligible"] & adj.notna())
        ic = sig.rank(axis=1).corrwith(adj.where(sig.notna()).rank(axis=1), axis=1)
        low = adj.where(rank <= 0.1 + 1e-9).mean(axis=1)
        high = adj.where(rank >= 0.9 - 1e-9).mean(axis=1)
        out[f"ic{h}"] = ic[inside]
        out[f"spread{h}"] = (low - high)[inside]
    return pd.DataFrame(out)


def main(horizons=(5, 10)) -> None:
    """Print the premise table for cluster, sector and market peers on TRAIN."""
    rows = []
    for peers in ("cluster", "sector", "market"):
        frames = [daily_rows(t, te0, te1, horizons, peers=peers)
                  for t in lab.TRAIN for te0, te1 in lab.periods(t)]
        d = pd.concat(frames)
        d = d[~d.index.duplicated()]
        for h in horizons:
            for regime, mask in (("all", d.index == d.index),
                                 ("stress", d["stress"].to_numpy()),
                                 ("calm", ~d["stress"].to_numpy())):
                x = d[mask]
                rows.append({"peers": peers, "h": h, "days": regime, "n": int(x[f"ic{h}"].count()),
                             "IC": x[f"ic{h}"].mean(), "IC t": nw_t(x[f"ic{h}"], h),
                             "spread %": 100 * x[f"spread{h}"].mean(),
                             "spread t": nw_t(x[f"spread{h}"], h)})
        by_year = d.groupby(d.index.year)[[f"ic{h}" for h in horizons]
                                          + [f"spread{h}" for h in horizons]].mean()
        print(f"\n{peers} peers, by year (IC, spread as a fraction):")
        print(by_year.round(4).to_string())
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200):
        print("\nTRAIN (E0 + TB + T3), decision bars inside the periods:")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main(tuple(int(a) for a in sys.argv[1:]) or (5, 10))
