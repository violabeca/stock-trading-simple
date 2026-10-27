"""Trade anatomy of a peer_reversal config on TRAIN periods only (RESEARCH_LOG, design input).

For every position the book opens inside a TRAIN period (E0, TB, T3): the gross trade return
from its fill (``open[entry + 1]``) to its exit fill (``open[exit + 1]``), side-signed and net
of the equal-weight market over the same days. No costs are charged; the yardstick is the lab's
round trip at a $1,000 order, 24 bp a unit of gross. Splits:

* by side, by the stress state at entry;
* by the formation window's busiest-day volume ÷ normal volume (terciles);
* by how extreme the entry rank was (terciles).

Plus the "reversal curve" of every entry: the mean side-signed, market-adjusted return from the
fill to ``open[entry + 1 + k]``, k = 1..21, whatever the book did.

``uv run python strategies/peer_reversal/anatomy.py '<json overrides>'``
"""

from __future__ import annotations

import importlib.util
import inspect
import json
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


def trades(test: str, te0: str, params: dict, curve: int = 21):
    """One row per position opened in the period, and each one's reversal curve."""
    p = lab.panel(test, te0, ["short_interest", "fund_shares"])
    keys = inspect.signature(S.compose).parameters
    c = S.compose(p, **{k: v for k, v in params.items() if k in keys})
    longs, shorts, _ = S._select(c["rank"], c["shortable"], c["enter_ok"], c["slot_size"] > 0,
                                 q_in=params["q_in"], q_out=params["q_out"],
                                 slots=params["slots"], max_hold=params["max_hold"])
    idx = p.close.index
    o = p.open_.to_numpy(dtype=float)
    mkt = np.nanmean(o[1:] / o[:-1] - 1.0, axis=1)
    mlog = np.r_[0.0, np.cumsum(np.log1p(np.nan_to_num(mkt)))]          # open-to-open market
    vol = p.volume.reindex_like(p.close)
    skip, form = params["skip"], params["formation"]
    busiest = vol.shift(skip).rolling(form, min_periods=1).max()
    normal = vol.shift(skip + form).rolling(63, min_periods=31).median()
    vratio = (busiest / normal).to_numpy(dtype=float)
    rank = c["rank"].to_numpy(dtype=float)
    stress = c["stress"].to_numpy(dtype=bool)
    lo, hi = int(np.searchsorted(idx, pd.Timestamp(te0, tz="UTC"))), len(idx) - 1
    rows, curves = [], []
    for side, held in ((1, longs), (-1, shorts)):
        h = held.to_numpy(dtype=bool)
        start = h & ~np.vstack([np.zeros((1, h.shape[1]), bool), h[:-1]])
        for t, j in zip(*np.nonzero(start), strict=True):
            if t < lo - 1:                       # fills before the period are not its trades
                continue
            end = t + 1
            while end < len(h) and h[end, j]:
                end += 1
            a, z = t + 1, min(end + 1, hi)       # entry and exit fill at the next open
            if a >= z or not np.isfinite(o[a, j]) or not np.isfinite(o[z, j]):
                continue
            adj = lambda b, a=a, j=j: (o[b, j] / o[a, j] - 1.0  # noqa: E731
                                       - np.expm1(mlog[b] - mlog[a]))
            rows.append({"side": side, "symbol": held.columns[j], "entry": idx[t],
                         "bars": z - a, "stress": bool(stress[t]), "rank": rank[t, j],
                         "vratio": vratio[t, j], "gross": side * adj(z),
                         "raw": side * (o[z, j] / o[a, j] - 1.0)})
            curves.append([side * adj(a + k) if a + k <= hi and np.isfinite(o[a + k, j])
                           else np.nan for k in range(1, curve + 1)])
    return pd.DataFrame(rows), pd.DataFrame(curves, columns=range(1, curve + 1))


def main(over: dict) -> None:
    """Print the TRAIN trade anatomy of the config (``over`` on top of config.yaml)."""
    params = lab.load_strategy("peer_reversal", over).params
    parts = [trades(t, te0, params) for t in lab.TRAIN for te0, _ in lab.periods(t)]
    e = pd.concat([x for x, _ in parts], ignore_index=True)
    cv = pd.concat([y for _, y in parts], ignore_index=True)
    e["volume"] = pd.qcut(e["vratio"].rank(method="first"), 3, labels=["low", "mid", "high"])
    extreme = np.where(e["side"] > 0, e["rank"], 1 - e["rank"])
    e["extreme"] = pd.qcut(pd.Series(extreme).rank(method="first"), 3,
                           labels=["most", "mid", "least"])
    print(f"{over}: TRAIN trades {len(e)}, mean bars {e['bars'].mean():.1f}, gross market-"
          f"adjusted {1e4 * e['gross'].mean():+.1f} bp a trade (raw {1e4 * e['raw'].mean():+.1f}"
          f" bp); the round trip at $1,000 orders costs 24 bp")
    for key in ("side", "stress", "volume", "extreme"):
        g = e.groupby(key, observed=True)
        tab = pd.DataFrame({"trades": g.size(), "bp/trade": 1e4 * g["gross"].mean(),
                            "t": g["gross"].mean() / g["gross"].std() * np.sqrt(g.size()),
                            "bars": g["bars"].mean(), "vratio": g["vratio"].median()})
        print(f"\nby {key}:\n{tab.round(2).to_string()}")
    side, st = e["side"].to_numpy(), e["stress"].to_numpy()
    tab = pd.DataFrame({"all": cv.mean(), "long": cv[side > 0].mean(),
                        "short": cv[side < 0].mean(), "stress": cv[st].mean(),
                        "calm": cv[~st].mean()}) * 1e4
    print("\nreversal curve, bp from the fill to open[entry + 1 + k]:")
    print(tab.round(1).T.to_string())


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
