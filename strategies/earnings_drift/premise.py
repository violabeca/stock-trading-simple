"""Premise check for earnings_drift (RESEARCH_LOG §0): TRAIN events only, prices <= 2021-12-31.

For every scored event whose usable bar lies in a TRAIN period (E0, TB, T3; each period's own
point-in-time names), the forward return from the open after the usable bar to the open
``h`` sessions later, minus the name's trailing-126-bar beta × the equal-weight market's return
over the same days. Prints, per horizon, the rank IC of the score and of each component (mean
over calendar quarters, t across quarters) and the top-fifth minus bottom-fifth spread, overall
and per dollar-volume half of the universe. Informational: it decides nothing alone.

    uv run python strategies/earnings_drift/premise.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("lab", REPO / "strategies/proposals_lab/lab.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)

CUTOFF = pd.Timestamp("2021-12-31", tz="UTC")
HORIZONS = (20, 40, 60)


def train_events(overrides: dict | None = None, horizons=HORIZONS) -> pd.DataFrame:
    """Every TRAIN period's scored events with beta-adjusted forward returns per horizon.

    Args:
        overrides: Config overrides (validated by the lab).
        horizons: Forward horizons in sessions (columns ``fwd<h>``).

    Returns:
        The events (:func:`strategy.events` columns) with ``beta``, ``dv``, ``dv_half``,
        ``fwd<h>`` and ``test``.
    """
    strat = lab.load_strategy("earnings_drift", overrides)
    keys = ("surprise", "w_ear", "ear_end", "trail", "min_events", "min_price", "cent_max")
    events = strat.signals.__globals__["events"]
    feats = lab.strategy_features(strat)
    panels = [(t, te0, te1, lab.panel(t, te0, feats)) for t in lab.TRAIN
              for te0, te1 in lab.periods(t)]
    # one price frame per name: the latest TRAIN panel holding it (all rows <= 2021-12-31)
    opens, closes = {}, {}
    for _, _, _, p in panels:
        for s in p.close.columns:
            opens[s], closes[s] = p.open_[s], p.close[s]
    o = pd.DataFrame(opens).sort_index()
    c = pd.DataFrame(closes).sort_index()
    o, c = o[o.index <= CUTOFF], c[c.index <= CUTOFF]
    rows = []
    for t, te0, te1, p in panels:
        ev = events(p, **{k: strat.params[k] for k in keys})
        inside = (ev["date"] >= pd.Timestamp(te0, tz="UTC")) & (
            ev["date"] <= pd.Timestamp(te1, tz="UTC"))
        ev = ev[inside & ev["pct"].notna()].copy()
        ret = p.close.pct_change(fill_method=None)
        mkt = ret.mean(axis=1)
        beta = ret.rolling(126, min_periods=63).cov(mkt).div(
            mkt.rolling(126, min_periods=63).var(), axis=0)
        oo = o[p.close.columns]                            # the period's names, open to open
        daily = (oo.shift(-1) / oo - 1).mean(axis=1)       # equal-weight open-to-open market
        mkt_idx = (1 + daily.fillna(0.0)).cumprod().shift(1, fill_value=1.0)
        pos = {d: i for i, d in enumerate(o.index)}
        ev["beta"] = [beta.iat[b, j] for b, j in zip(ev["bar"], ev["col"], strict=True)]
        ev["dv"] = [(p.close.iloc[max(b - 63, 0):b, j] * p.volume.iloc[max(b - 63, 0):b, j])
                    .median() for b, j in zip(ev["bar"], ev["col"], strict=True)]
        ev["dv_half"] = (ev.groupby(ev["date"].dt.tz_convert(None).dt.to_period("Q"))["dv"]
                         .rank(pct=True) > 0.5).map({True: "upper", False: "lower"})
        for h in horizons:
            fwd, mk = [], []
            for d, s in zip(ev["date"], ev["symbol"], strict=True):
                i = pos[d]
                if i + 1 + h >= len(o.index):
                    fwd.append(np.nan)
                    mk.append(np.nan)
                    continue
                a, z = o.index[i + 1], o.index[i + 1 + h]
                fwd.append(o.at[z, s] / o.at[a, s] - 1)
                mk.append(mkt_idx.get(z, np.nan) / mkt_idx.get(a, np.nan) - 1)
            ev[f"fwd{h}"] = np.array(fwd) - ev["beta"].fillna(1.0).to_numpy() * np.array(mk)
        ev["test"] = t
        rows.append(ev)
    return pd.concat(rows, ignore_index=True)


def _t(x: pd.Series) -> float:
    """t-statistic of a mean across quarters."""
    return float(x.mean() / x.std() * np.sqrt(len(x)))


def summarize(ev: pd.DataFrame, label: str) -> None:
    """Quarterly rank IC and fifth spread, overall and by dollar-volume half."""
    print(f"\n== {label}: {len(ev)} scored TRAIN events, "
          f"{ev['date'].min().date()} .. {ev['date'].max().date()}")
    q = ev["date"].dt.tz_convert(None).dt.to_period("Q")
    for h in HORIZONS:
        col = f"fwd{h}"
        for part, sub in [("all", ev), ("upper dv half", ev[ev["dv_half"] == "upper"]),
                          ("lower dv half", ev[ev["dv_half"] == "lower"])]:
            sub = sub[sub[col].notna()]
            qq = sub["date"].dt.tz_convert(None).dt.to_period("Q")
            ics = {k: sub.groupby(qq).apply(
                lambda g, k=k, col=col: g[k].corr(g[col], method="spearman"))
                for k in ("pct", "r_sue", "r_ear")}
            top = sub[sub["pct"] >= 0.8 - 1e-9].groupby(qq)[col].mean()
            bot = sub[sub["pct"] <= 0.2 + 1e-9].groupby(qq)[col].mean()
            spread = (top - bot).dropna()
            line = " ".join(f"IC {k} {v.mean():+.3f} (t {_t(v):+.1f})" for k, v in ics.items())
            print(f"  h={h:2d} {part:14s} n={len(sub):5d} {line} | top-bot fifth "
                  f"{spread.mean():+.4f} (t {_t(spread):+.1f},"
                  f" {int((spread > 0).sum())}/{len(spread)} quarters > 0)"
                  f" top {top.mean():+.4f} bot {bot.mean():+.4f}")
    per_test = ev.groupby("test").apply(
        lambda g: (g.loc[g["pct"] >= 0.8 - 1e-9, "fwd60"].mean()
                   - g.loc[g["pct"] <= 0.2 + 1e-9, "fwd60"].mean()))
    print("  top-bot fifth fwd60 by test:", per_test.round(4).to_dict())
    print("  quarters:", int(q.nunique()))


if __name__ == "__main__":
    import json
    over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    summarize(train_events(over), f"overrides {over}")
