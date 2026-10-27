"""Daily-return correlation and 50/50 blends of quality_junk labels with another lab book.

``correlate.py <label> <other strategy> <other label> [--v2]``: per test, the correlation of the
two books' daily returns, of their excess over the market (book − buy-and-hold) and of their
market residuals (each regressed on buy-and-hold), on common days.
``correlate.py blend <label> <other strategy> <other label> [--v2]``: the daily 50/50 blend of
the two books, stitched per period like the lab (alpha, P, volatility, max drawdown, beta,
alpha per unit of volatility).

Round 1 (RESEARCH_LOG.md §5 and §10) ran these on ``informed_flow_ls_p1/v6_si`` in lab v1's
``results/`` (read-only; the round-1 review reproduced every number with the ``blend`` logic
below). The lab README asks agents not to touch another agent's results, so round 2 did NOT
re-run them on lab v2 (``--v2`` reads ``results2/``); the diversification numbers in the docs
are lab v1's.
"""

import pickle
import sys

import pandas as pd

ROOT = "/Users/janmarti/Documents/code/stock-trading-simple/var/proposals_lab"


def _load(strategy, label, v2=False):
    with open(f"{ROOT}/{'results2' if v2 else 'results'}/{strategy}/{label}.pkl", "rb") as f:
        return pickle.loads(f.read())


def series(strategy, label, test, v2=False):
    r = _load(strategy, label, v2)
    if test not in r:
        return None
    ret = pd.concat([x["returns"] for x in r[test]])
    bench = pd.concat([x["bench"] for x in r[test]])
    return ret, bench


def residual_corr(label, other, olabel, tests=("TB", "T3", "T1", "T2"), v2=False):
    """Correlation of the two books' market residuals (each regressed on buy-and-hold)."""
    out = {}
    for t in tests:
        a, b = series("quality_junk", label, t, v2), series(other, olabel, t, v2)
        if a is None or b is None:
            continue
        df = pd.concat({"qj": a[0], "other": b[0], "mkt": a[1]}, axis=1).dropna()
        res = {}
        for k in ("qj", "other"):
            beta = df[k].cov(df["mkt"]) / df["mkt"].var()
            res[k] = df[k] - beta * df["mkt"]
        out[t] = float(res["qj"].corr(res["other"]))
    return out


def blend(label, other, olabel, tests=("TB", "T3", "T1", "T2"), v2=False):
    """Daily 50/50 blend of the two books, stitched per period against quality_junk's
    buy-and-hold (both books trade the same universes), next to the other book alone."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lab", "/Users/janmarti/Documents/code/stock-trading-simple/strategies/proposals_lab/"
        "lab.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    qj, ot = _load("quality_junk", label, v2), _load(other, olabel, v2)
    out = {}
    for t in tests:
        if t not in qj or t not in ot:
            continue
        pairs, alone = [], []
        for rq, ro in zip(qj[t], ot[t], strict=True):
            assert rq["period"] == ro["period"]
            r = 0.5 * rq["returns"] + 0.5 * ro["returns"].reindex(rq["returns"].index).fillna(0.0)
            pairs.append((r, rq["bench"]))
            alone.append((ro["returns"], ro["bench"]))
        st, so = lab.stitched(pairs), lab.stitched(alone)
        out[t] = {"blend alpha": st["alpha"], "P": st["P(alpha > 0)"], "vol": st["volatility"],
                  "max DD": st["max drawdown"], "beta": st["beta"],
                  "alpha / vol": st["alpha"] / st["volatility"],
                  "other alpha / vol": so["alpha"] / so["volatility"]}
        print(t, {k: round(v, 3) for k, v in out[t].items()})
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--v2"]
    v2 = "--v2" in sys.argv
    if args and args[0] == "blend":
        blend(*args[1:4], v2=v2)
        sys.exit()
    label, other, olabel = args[:3]
    for t in ("E0", "TB", "T3", "T1", "T2"):
        a, b = series("quality_junk", label, t, v2), series(other, olabel, t, v2)
        if a is None or b is None:
            continue
        df = pd.concat({"qj": a[0], "other": b[0], "mkt": a[1]}, axis=1).dropna()
        ex = df[["qj", "other"]].sub(df["mkt"], axis=0)
        print(f"{t}: days {len(df)}  corr {df['qj'].corr(df['other']):+.2f}  "
              f"corr vs market qj {df['qj'].corr(df['mkt']):+.2f} other "
              f"{df['other'].corr(df['mkt']):+.2f}  excess corr {ex['qj'].corr(ex['other']):+.2f}")
    print("residual corr:", {k: round(v, 2) for k, v in residual_corr(label, other, olabel,
                                                                       v2=v2).items()})
