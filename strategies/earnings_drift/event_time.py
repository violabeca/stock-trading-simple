"""Event-time drift on TRAIN only (RESEARCH_LOG §5): where after an event does its drift sit?

For every scored TRAIN event (E0, TB, T3; prices <= 2021-12-31) in the top or bottom fifth:
the market-adjusted return (stock − equal-weight market of the period's names) split into

* ``pre_next``: from the close of the bar after the usable bar to the close before the name's
  NEXT report session (the drift between announcements);
* ``at_next``: the next report's own reaction window, close before it → close after it;

and the cumulative market-adjusted path from the open after the usable bar at fixed horizons.
Informational, like the premise check.

    uv run python strategies/earnings_drift/event_time.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ed_premise", HERE / "premise.py")
premise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(premise)
lab = premise.lab
STEPS = (5, 10, 20, 40, 55, 60, 65, 70, 80, 90)


def study() -> None:
    """Print the two-part split and the cumulative paths, top vs bottom fifth."""
    strat = lab.load_strategy("earnings_drift")
    keys = ("surprise", "w_ear", "ear_end", "trail", "min_events", "min_price", "cent_max")
    events = strat.signals.__globals__["events"]
    feats = lab.strategy_features(strat)
    panels = [(te0, te1, lab.panel(t, te0, feats)) for t in lab.TRAIN
              for te0, te1 in lab.periods(t)]
    opens, closes, days = {}, {}, {}
    for _, _, p in panels:
        for s in p.close.columns:
            opens[s], closes[s] = p.open_[s], p.close[s]
            days[s] = p.features["earn_day"][s]
    o, c, d = (pd.DataFrame(x).sort_index() for x in (opens, closes, days))
    keep = o.index <= premise.CUTOFF
    o, c, d = o[keep], c[keep], d[keep]
    rows = []
    for te0, te1, p in panels:
        ev = events(p, **{k: strat.params[k] for k in keys})
        ev = ev[(ev["date"] >= pd.Timestamp(te0, tz="UTC"))
                & (ev["date"] <= pd.Timestamp(te1, tz="UTC")) & ev["pct"].notna()]
        ev = ev[(ev["pct"] >= 0.8 - 1e-9) | (ev["pct"] <= 0.2 + 1e-9)]
        names = list(p.close.columns)
        mo = (o[names].shift(-1) / o[names] - 1).mean(axis=1)       # open-to-open market
        mc = c[names].pct_change(fill_method=None).mean(axis=1)    # close-to-close market
        mo_idx = (1 + mo.fillna(0.0)).cumprod().shift(1, fill_value=1.0)
        mc_idx = (1 + mc.fillna(0.0)).cumprod()
        pos = {x: i for i, x in enumerate(o.index)}
        for date, sym, pct in zip(ev["date"], ev["symbol"], ev["pct"], strict=True):
            i = pos[date]
            if i + 1 >= len(o.index):
                continue
            a = o.index[i + 1]
            nxt = np.flatnonzero(d[sym].iloc[i + 1:].notna().to_numpy())
            row = {"side": "top" if pct >= 0.5 else "bottom", "date": date}
            for h in STEPS:
                if i + 1 + h < len(o.index):
                    z = o.index[i + 1 + h]
                    row[f"h{h}"] = (o.at[z, sym] / o.at[a, sym] - 1) - (
                        mo_idx[z] / mo_idx[a] - 1)
            if len(nxt) and i + 2 + nxt[0] < len(o.index):
                k = i + 1 + nxt[0]                                    # next report session
                b0, b1 = o.index[k - 1], o.index[k + 1]
                row["gap"] = k - i
                row["pre_next"] = (c.at[b0, sym] / c.at[a, sym] - 1) - (   # close u+1 → p'-1
                    mc_idx[b0] / mc_idx[a] - 1)
                row["at_next"] = (c.at[b1, sym] / c.at[b0, sym] - 1) - (mc_idx[b1] / mc_idx[b0] - 1)
            rows.append(row)
    df = pd.DataFrame(rows)
    q = df["date"].dt.tz_convert(None).dt.to_period("Q")
    print(f"{len(df)} extreme TRAIN events; sessions from usable bar to the next report: "
          f"median {df['gap'].median():.0f}, 10-90% {df['gap'].quantile(.1):.0f}-"
          f"{df['gap'].quantile(.9):.0f}")
    for col in ["pre_next", "at_next", *[f"h{h}" for h in STEPS]]:
        top = df[df["side"] == "top"].groupby(q)[col].mean()
        bot = df[df["side"] == "bottom"].groupby(q)[col].mean()
        sp = (top - bot).dropna()
        print(f"  {col:9s} top {top.mean():+.4f} bottom {bot.mean():+.4f} spread "
              f"{sp.mean():+.4f} (t {sp.mean() / sp.std() * np.sqrt(len(sp)):+.1f}, "
              f"{int((sp > 0).sum())}/{len(sp)} quarters > 0)")


if __name__ == "__main__":
    study()
