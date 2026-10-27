"""The locked read, in ONE batch (the cross-strategy round's last step, 2026-10-23).

Every design's DECLARED final and the component books (as controls) run once on the LOCKED
tests T4 (names 1001-1500) and TL (names 1501-2000), 2022-01 -> 2026-10-16, under labels
``<label>-LOCK``; the overall final must be declared in ``FINAL_DECLARATION.md`` beside this
file before this script runs (it refuses to start otherwise). Reported per book and test:
alpha, P, beta, CAGR vs buy-and-hold, max drawdown, the late-fill alpha, and the 2022 bear and
Feb-Apr 2025 episodes inside those names.

usage: uv run python strategies/proposals_lab/final_batch.py '<json list of
       {"strategy": ..., "label": ..., "overrides": {...}, "hedge": bool, "role": ...}>'
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("proposals_lab", HERE / "lab.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)

EPISODES = {"2022 bear": ("2022-01-03", "2022-10-12"), "Feb-Apr 2025": ("2025-02-19", "2025-04-08")}


def episode_returns(rows) -> dict:
    r = pd.concat([x["returns"] for x in rows])
    b = pd.concat([x["bench"] for x in rows]).reindex(r.index).fillna(0.0)
    out = {}
    for k, (a, z) in EPISODES.items():
        m = (r.index >= pd.Timestamp(a, tz="UTC")) & (r.index <= pd.Timestamp(z, tz="UTC"))
        out[k] = f"{(1 + r[m]).prod() - 1:+.1%} ({(1 + b[m]).prod() - 1:+.1%})"
    return out


def main(items: list[dict]) -> None:
    if not (HERE / "FINAL_DECLARATION.md").exists():
        raise SystemExit("declare the overall final in FINAL_DECLARATION.md before the locked read")
    rows = []
    for it in items:
        name = f"{it['label']}-LOCK"
        path = lab.results_path(it["strategy"], name)
        res = lab.load_results(it["strategy"], name) if path.exists() else lab.run(
            it["strategy"], name, it.get("overrides", {}), ["T4", "TL"],
            hedge=it.get("hedge", False))
        for t in ("T4", "TL"):
            s = lab.stitched([(x["returns"], x["bench"]) for x in res[t]])
            late = lab.stitched([(x["late"], x["bench"]) for x in res[t]])["alpha"]
            rows.append({"role": it.get("role", ""), "book": f"{it['strategy']}/{it['label']}",
                         "test": t, "alpha": round(s["alpha"], 4), "P": round(s["P(alpha > 0)"], 3),
                         "beta": round(s["beta"], 2), "CAGR": round(s["cagr"], 4),
                         "B&H CAGR": round(s["buy & hold"], 4),
                         "maxDD": round(s["max drawdown"], 3),
                         "late alpha": round(late, 4), **episode_returns(res[t])})
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.to_string(index=False))
    out.to_csv(lab.RESULTS / "final_batch.csv", index=False)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
