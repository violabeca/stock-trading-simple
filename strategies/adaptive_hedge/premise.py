"""Premise checks for adaptive_hedge on TRAIN only (TB + T3; RESEARCH_LOG.md §0).

Informational, decides nothing alone. Run from the repo root:

    uv run python strategies/adaptive_hedge/premise.py [TB,T3] ['<overrides json>'] [tag]

1. Long signal: Spearman rank IC and top-minus-bottom decile spread of the long rank (minus
   days-to-cover over 42 bars, inside 5 dollar-volume groups) against the forward h-bar
   beta-adjusted return (next open -> close t+h; 126-bar beta to the period's equal-weight names),
   on non-overlapping dates every h bars inside each period's scored window.
2. Short starts: the forward h-bar return after a start (next open -> close t+h), raw, minus the
   equal-weight market and beta-adjusted; averaged per date, then per block of h bars (t over
   blocks).
3. The adaptive premise: the daily share of names making a short start against the market's
   trailing 21-bar return; and the U0 book's short count and short holding time.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/janmarti/Documents/code/stock-trading-simple")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lab = _load("lab", REPO / "strategies/proposals_lab/lab.py")
ah = _load("adaptive_hedge_strategy", REPO / "strategies/adaptive_hedge/strategy.py")


def _beta(close: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    ret = close.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    return ret.rolling(window, min_periods=63).cov(mkt).div(
        mkt.rolling(window, min_periods=63).var(), axis=0)


def _t(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def period(test: str, te0: str, te1: str, params: dict, horizons=(21, 63)) -> dict:
    p = lab.panel(test, te0, ["short_interest", "fund_shares"])
    parts = ah.compose(p, **params)
    close, open_ = p.close, p.open_
    idx = close.index
    beta = _beta(close)
    first = int(np.searchsorted(idx, pd.Timestamp(te0, tz="UTC")))
    out = {"ic": {}, "spread": {}, "events": {}}
    for h in horizons:
        fwd = close.shift(-h) / open_.shift(-1) - 1.0             # fill t+1 open -> close t+h
        mkt = fwd.mean(axis=1)
        adj = fwd - beta.mul(mkt, axis=0)
        rel = fwd.sub(mkt, axis=0)
        ics, spreads = [], []
        for t in range(first, len(idx) - h, h):                   # non-overlapping dates
            s, y = parts["score"].iloc[t], adj.iloc[t]
            ok = s.notna() & y.notna()
            if ok.sum() < 50:
                continue
            ics.append(s[ok].rank().corr(y[ok].rank()))
            top, bot = y[ok][s[ok] >= s[ok].quantile(0.9)], y[ok][s[ok] <= s[ok].quantile(0.1)]
            spreads.append(top.mean() - bot.mean())
        out["ic"][h], out["spread"][h] = ics, spreads
        ev = parts["short_start"] & parts["short_ok"]
        crowd = parts["score"] <= 0.5                              # variant B's half
        mom = (close.shift(params["rs_skip"]) / close.shift(params["rs_window"]) - 1.0) < 0
        rows = []
        for t in range(first, len(idx) - h):
            names = ev.columns[ev.iloc[t].to_numpy()]
            if len(names):
                row = {"t": t, "n": len(names), "raw": fwd.iloc[t][names].mean(),
                       "rel": rel.iloc[t][names].mean(), "adj": adj.iloc[t][names].mean()}
                for tag, mask in (("crowded", crowd), ("momentum", mom)):
                    for yes in (True, False):
                        sub = names[mask.iloc[t][names].to_numpy() == yes]
                        key = f"{tag}{'+' if yes else '-'}"
                        row[f"rel {key}"] = rel.iloc[t][sub].mean() if len(sub) else np.nan
                        row[f"n {key}"] = len(sub)
                rows.append(row)
        out["events"][h] = rows
    # the adaptive premise: breadth of breakdowns vs the market's trailing month
    win = idx >= pd.Timestamp(te0, tz="UTC")
    ret = close.pct_change(fill_method=None)
    mkt21 = (1 + ret.mean(axis=1)).rolling(21).apply(np.prod, raw=True) - 1
    starts = (parts["short_start"] & parts["short_ok"]).sum(axis=1) / close.notna().sum(axis=1)
    b = ah.book(p, **params)
    shorts = b["shorts"]
    holds = [(pd.Timestamp(c) - pd.Timestamp(o)).days for side, _, o, c, _ in b["exits"]
             if side == "short" and pd.Timestamp(c) >= pd.Timestamp(te0, tz="UTC")]
    causes = pd.Series([x[4] for x in b["exits"] if x[0] == "short"
                        and pd.Timestamp(x[3]) >= pd.Timestamp(te0, tz="UTC")]).value_counts()
    out["breadth"] = pd.DataFrame({"starts": starts[win], "mkt21": mkt21[win],
                                   "n_short": shorts.sum(axis=1)[win],
                                   "n_long": b["longs"].sum(axis=1)[win]})
    out["holds_days"] = holds
    out["short_exit_causes"] = causes.to_dict()
    return out


def main(tests=("TB", "T3"), overrides: dict | None = None, tag: str = "") -> dict:
    strat = lab.load_strategy("adaptive_hedge", overrides)
    params = strat.params
    res = {t: {te0: period(t, te0, te1, params) for te0, te1 in lab.periods(t)} for t in tests}
    summary = {}
    for h in (21, 63):
        ics = [x for t in res for q in res[t].values() for x in q["ic"][h]]
        spr = [x for t in res for q in res[t].values() for x in q["spread"][h]]
        ev = pd.DataFrame([dict(r, block=f"{t}{te0}-{r['t'] // h}") for t in res
                           for te0, q in res[t].items() for r in q["events"][h]])
        blocks = ev.groupby("block")[["raw", "rel", "adj"]].mean()
        summary[f"long IC h{h}"] = {"mean": float(np.nanmean(ics)), "t": _t(ics),
                                    "dates": len(ics),
                                    "decile spread": float(np.nanmean(spr)), "spread t": _t(spr)}
        summary[f"short starts h{h}"] = {
            "events": int(ev["n"].sum()), "dates": len(ev), "blocks": len(blocks),
            **{f"{k} mean": float(blocks[k].mean()) for k in ("raw", "rel", "adj")},
            **{f"{k} t": _t(blocks[k]) for k in ("raw", "rel", "adj")}}
        per_test = {}
        for t in res:
            e = ev[ev["block"].str.startswith(t)]
            bl = e.groupby("block")[["raw", "rel", "adj"]].mean()
            per_test[t] = {k: round(float(bl[k].mean()), 4) for k in ("raw", "rel", "adj")}
        summary[f"short starts h{h}"]["by test"] = per_test
        splits = {}
        for key in ("crowded+", "crowded-", "momentum+", "momentum-"):
            bl = ev.groupby("block")[f"rel {key}"].mean()
            splits[key] = {"rel mean": round(float(bl.mean()), 4), "t": round(_t(bl), 2),
                           "events": int(ev[f"n {key}"].sum())}
        summary[f"short starts h{h}"]["splits (minus the market)"] = splits
    br = pd.concat([q["breadth"] for t in res for q in res[t].values()])
    br = br[~br.index.duplicated()]
    month = br.index.tz_convert(None).to_period("M")
    mret = br["mkt21"].groupby(month).last()
    by_month = br.groupby(month)[["starts", "n_short", "n_long"]].mean().assign(mkt21=mret)
    up, down = by_month[by_month["mkt21"] > 0], by_month[by_month["mkt21"] <= 0]
    holds = [x for t in res for q in res[t].values() for x in q["holds_days"]]
    causes = pd.concat([pd.Series(q["short_exit_causes"], dtype=float) for t in res
                        for q in res[t].values()]).groupby(level=0).sum()
    summary["breadth"] = {
        "corr(starts, trailing 21-bar market) daily": float(br["starts"].corr(br["mkt21"],
                                                                                method="spearman")),
        "starts share/day: mean": float(br["starts"].mean()),
        "starts share/day in up months": float(up["starts"].mean()),
        "starts share/day in down months": float(down["starts"].mean()),
        "U0 shorts held: mean": float(br["n_short"].mean()),
        "U0 shorts held: share of bars at the 20 cap": float((br["n_short"] >= 20).mean()),
        "U0 shorts held in up months": float(up["n_short"].mean()),
        "U0 shorts held in down months": float(down["n_short"].mean()),
        "short holds (calendar days) median": float(np.median(holds)) if holds else np.nan,
        "short holds mean": float(np.mean(holds)) if holds else np.nan,
        "short exits": {k: int(v) for k, v in causes.items()},
        "corr(n_short, trailing 21-bar market) daily": float(br["n_short"].corr(
            br["mkt21"], method="spearman"))}
    print(json.dumps(summary, indent=1, default=float))
    print(by_month.round(3).to_string())
    out = lab.RESULTS / "adaptive_hedge" / f"premise{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary,
                               "by_month": by_month.reset_index().astype(str).to_dict("records")},
                              indent=1, default=float))
    return summary


if __name__ == "__main__":
    main(tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("TB", "T3"),
         json.loads(sys.argv[2]) if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else "")
