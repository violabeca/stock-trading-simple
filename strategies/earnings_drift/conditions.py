"""TRAIN-only screen of conditioning variables for the iterations (RESEARCH_LOG §6).

Top-fifth minus bottom-fifth market-adjusted spread (open after the usable bar → h sessions
later, beta-adjusted as in ``premise.py``) of the U0 events, split by:

* ``volume``: abnormal announcement volume (report session + the next, over 2 × the 60-session
  median before the report) above / below the trailing events' median;
* ``dv``: the name's 63-session dollar volume in the upper / lower half of the universe;
* ``agree``: SUE and EAR ranks on the same side of 0.5 / opposite sides;
* ``dtc``: days-to-cover above / below the bar's cross-sectional median (2018+ only).

    uv run python strategies/earnings_drift/conditions.py
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


def screen() -> None:
    """Print the conditional spreads per horizon."""
    ev = premise.train_events()
    strat = lab.load_strategy("earnings_drift")
    feats = lab.strategy_features(strat)
    extra = []
    for t in lab.TRAIN:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, feats)
            sub = ev[(ev["test"] == t) & (ev["date"] >= pd.Timestamp(te0, tz="UTC"))
                     & (ev["date"] <= pd.Timestamp(te1, tz="UTC"))]
            vol = p.volume.to_numpy(dtype=float)
            base = p.volume.rolling(60, min_periods=20).median().to_numpy(dtype=float)
            dtc = p.features.get("short_interest")
            med = dtc.median(axis=1) if dtc is not None else None
            for idx, b, j in zip(sub.index, sub["bar"], sub["col"], strict=True):
                rp = b - 2                                       # the report session
                abn = (vol[rp, j] + vol[rp + 1, j]) / (2 * base[rp - 2, j])
                d = dtc.iat[b, j] if dtc is not None else np.nan
                extra.append({"i": idx, "abvol": abn,
                              "crowded": (d > med.iat[b]) if np.isfinite(d) else np.nan})
    ex = pd.DataFrame(extra).set_index("i")
    ev = ev.join(ex)
    quarter = ev["date"].dt.tz_convert(None).dt.to_period("Q")
    ev["vol_hi"] = ev["abvol"] > ev.groupby(quarter)["abvol"].transform("median")
    ev["agree"] = (ev["r_sue"] - 0.5) * (ev["r_ear"] - 0.5) > 0
    print(f"{len(ev)} events; abnormal volume median {ev['abvol'].median():.2f}, "
          f"10-90% {ev['abvol'].quantile(.1):.2f}-{ev['abvol'].quantile(.9):.2f}")
    splits = {"volume high": ev["vol_hi"], "volume low": ~ev["vol_hi"],
              "dv upper": ev["dv_half"] == "upper", "dv lower": ev["dv_half"] == "lower",
              "agree": ev["agree"], "disagree": ~ev["agree"],
              "crowded (2018+)": ev["crowded"] == True,  # noqa: E712
              "not crowded (2018+)": ev["crowded"] == False}  # noqa: E712
    for h in premise.HORIZONS:
        col = f"fwd{h}"
        for name, mask in splits.items():
            sub = ev[mask & ev[col].notna()]
            q = sub["date"].dt.tz_convert(None).dt.to_period("Q")
            top = sub[sub["pct"] >= 0.8 - 1e-9].groupby(q)[col].mean()
            bot = sub[sub["pct"] <= 0.2 + 1e-9].groupby(q)[col].mean()
            sp = (top - bot).dropna()
            print(f"  h={h:2d} {name:20s} n top/bot {int((sub['pct'] >= 0.8 - 1e-9).sum()):4d}/"
                  f"{int((sub['pct'] <= 0.2 + 1e-9).sum()):4d}  top {top.mean():+.4f} "
                  f"bot {bot.mean():+.4f} spread {sp.mean():+.4f} (t {premise._t(sp):+.1f}, "
                  f"{int((sp > 0).sum())}/{len(sp)})")


if __name__ == "__main__":
    screen()
