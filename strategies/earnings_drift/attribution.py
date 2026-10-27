"""Leg attribution of one earnings_drift config (diagnostic; RESEARCH_LOG §4).

Re-runs the config's book per period (the lab's warm ``book``), splits each day's gross P&L
(held weight at the previous close × the close-to-close return, before costs) into the long
and the short leg, and prints per test: each leg's yearly contribution, the equal-weight
market's return, and the biggest single-name contributors of a chosen period.

    uv run python strategies/earnings_drift/attribution.py '<overrides>' [tests]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("lab", REPO / "strategies/proposals_lab/lab.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)


def legs(overrides: dict, tests) -> None:
    """Print the long / short legs' gross yearly contribution per test and per period."""
    strat = lab.load_strategy("earnings_drift", overrides)
    settings = lab._settings()
    for t in tests:
        rows, frames = [], []
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            res = lab.book(strat, p, settings, te0)
            pf, w = res["pf"], res["window"]
            held = pf.asset_value(group_by=False).div(pf.value(), axis=0).shift(1).fillna(0.0)
            ret = w.close.pct_change(fill_method=None).reindex(held.index).fillna(0.0)
            pnl = held * ret[held.columns]
            long_, short_ = pnl.where(held > 0, 0.0).sum(axis=1), pnl.where(held < 0, 0.0).sum(
                axis=1)
            frames.append(pd.DataFrame({"long": long_, "short": short_, "book": res["returns"],
                                        "market": res["bench"]}))
            yrs = lab.years_of(held.index)
            rows.append({"period": te0, "long": long_.sum() / yrs, "short": short_.sum() / yrs,
                         "market": float(np.expm1(np.log1p(res["bench"]).sum() / yrs))})
        f = pd.concat(frames)
        yrs = lab.years_of(f.index)
        print(f"{t}: long leg {f['long'].sum() / yrs:+.2%}/yr, short leg "
              f"{f['short'].sum() / yrs:+.2%}/yr (gross, simple sums), market "
              f"{np.expm1(np.log1p(f['market']).sum() / yrs):+.2%}/yr")
        print("   per period:", " ".join(f"{r['period'][:7]} L{r['long']:+.3f} S{r['short']:+.3f}"
                                         for r in rows))


if __name__ == "__main__":
    over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    legs(over, sys.argv[2].split(",") if len(sys.argv) > 2 else lab.OPEN)
