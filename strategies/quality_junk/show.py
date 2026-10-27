"""Print the lab's summary of saved quality_junk labels.

``show.py <label> [tests]``: per-test summary, per-period alphas, the stitched TRAIN record.
``show.py delta <base> <label>...``: paired alpha Δ of each label against ``base`` on the
stitched TRAIN record (E0 + TB + T3) and on T1, T2 (the iteration bar's three numbers).
"""

import importlib.util
import json
import sys

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "lab", "/Users/janmarti/Documents/code/stock-trading-simple/strategies/proposals_lab/lab.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)
NAME = "quality_junk"


def _pairs(res, tests):
    return [(r["returns"], r["bench"]) for t in tests if t in res for r in res[t]]


def delta(base: str, label: str) -> dict:
    """Paired Δ alpha (and P) of ``label`` vs ``base``: stitched TRAIN, T1, T2."""
    a, b = lab.load_results(NAME, label), lab.load_results(NAME, base)
    out = {}
    for name, tests in (("TRAIN", lab.TRAIN), ("T1", ["T1"]), ("T2", ["T2"])):
        sa, sb = lab.stitched(_pairs(a, tests)), lab.stitched(_pairs(b, tests))
        idx = sa["returns"].index
        pr = lab.paired(sa["returns"], sb["returns"].reindex(idx).fillna(0.0),
                        sb["bench"].reindex(idx).fillna(0.0), sb["years"])
        out[name] = {"alpha": round(sa["alpha"], 4), "base alpha": round(sb["alpha"], 4),
                     "delta": round(pr["delta alpha"], 4),
                     "P(delta>0)": round(pr["P(delta > 0)"], 3)}
    return out


if __name__ == "__main__":
    if sys.argv[1] == "delta":
        for lab_name in sys.argv[3:]:
            print(lab_name, "vs", sys.argv[2], json.dumps(delta(sys.argv[2], lab_name)))
        sys.exit()
    res = lab.load_results(NAME, sys.argv[1])
    tests = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    s = lab.summary(res, tests)
    rows = {t: {k: v for k, v in col.items() if k != "periods"} for t, col in s.items()}
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(rows).T.round(3).to_string())
    for t, col in s.items():
        print(t, json.dumps(col["periods"]))
    train = _pairs(res, lab.TRAIN)
    if train:
        st = lab.stitched(train)
        print("TRAIN stitched:", {k: round(v, 4) for k, v in st.items()
                                  if isinstance(v, float)})
