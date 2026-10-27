"""Target-versus-realized reconciliation of saved adaptive_hedge labels (round-2 repair; it runs
no new variant and selects nothing).

The lab sends a name an order only on the bar its target changes (``bt.rotation_targets``). An
order that vectorbt refuses or part-fills, or drops at a NaN price, is never sent again, so the
book can hold what the strategy no longer wants, or miss what it wants. On lab v2, under
``lock_cash`` and near full gross, this happens when free cash runs out. A cover is the worst case:
vectorbt lets a cover spend the short's locked proceeds plus an equal margin (2 x the entry
value), so a short that has more than doubled needs free cash to be bought back.

This script rebuilds every period exactly as ``lab.run`` does (signals once, ``lab.book``), with
vectorbt's order logs switched on, and checks the rebuild against the saved returns. It then
reports:

* every order refused or part-filled, by type (``lab._call_order``'s classes), with free cash;
* every order dropped at a NaN fill price (a target the book never holds: a "dead" slot);
* every STUCK position (held while its in-force target is 0, or held on the wrong side), with its
  bars and its P&L from the due bar's open on;
* the time-averaged exposure the book misses because of refused, part-filled or dropped entries;
* per test (and the TRAIN era, TB + T3): the stitched alpha as saved, and with the stuck P&L
  removed. The second number is an estimate of the strategy's own exits: each exit fills when
  the strategy decided it, and the cash it frees stays idle. With ``--base``, the paired delta
  against a base label is given both ways.

T4 and TL are locked tests, and this script refuses them.

    uv run python strategies/adaptive_hedge/reconcile.py <label>[,<label>...] [TB,T3,T1,T2] \
        [--base <label>]
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

REPO = Path("/Users/janmarti/Documents/code/stock-trading-simple")
spec = importlib.util.spec_from_file_location("lab", REPO / "strategies/proposals_lab/lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)
KEY = "adaptive_hedge"
OUT = lab.RESULTS / KEY / "repair"
LOCKED = {"T4", "TL"}
CLASSES = ("long cut", "short cover", "new short", "long buy", "none")
TINY = 1e-9

_from_orders = vbt.Portfolio.from_orders.__func__


def _logged(cls, *args, **kwargs):
    """``Portfolio.from_orders`` with the order logs on. Logs never change the simulation."""
    kwargs["log"] = True
    return _from_orders(cls, *args, **kwargs)


vbt.Portfolio.from_orders = classmethod(_logged)


def _classes(req: pd.DataFrame) -> np.ndarray:
    """Each request's class against the name's in-force target before the bar (the rules of
    ``lab._call_order``): 0 long cut, 1 short cover, 2 new short, 3 long buy, 4 none."""
    t = req.to_numpy(dtype=float)
    prev = pd.DataFrame(t).ffill().shift(1).fillna(0.0).to_numpy()
    has = ~np.isnan(t)
    x = np.where(has, t, prev)
    cat = np.full(t.shape, 4, dtype=np.int64)
    cat[has & (x > 0) & (x > prev)] = 3
    cat[has & (x < 0) & (x < prev)] = 2
    cat[has & (prev < 0) & (x >= prev) & (x <= 0)] = 1
    cat[has & (prev > 0) & (x <= prev) & (x >= 0)] = 0
    return cat


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(first, last)`` positions of every run of True in a 1-d mask."""
    out, k, n = [], 0, len(mask)
    while k < n:
        if mask[k]:
            e = k
            while e + 1 < n and mask[e + 1]:
                e += 1
            out.append((k, e))
            k = e + 1
        else:
            k += 1
    return out


def period(strat, settings, test: str, te0: str, saved: dict, hedge: bool) -> dict:
    """Rebuild one period's book with logs and reconcile its targets with its positions.

    Args:
        strat: The loaded strategy with the label's overrides.
        settings: The lab's settings (base costs).
        test: Test name (an open test).
        te0: The period's first scored day.
        saved: The saved result row of this period (its ``returns`` are checked).
        hedge: Whether the label merged SPY into the panel.

    Returns:
        ``rebuild`` (largest |rebuilt − saved| daily return), ``orders`` (the order logs with
        ``class``, ``need``, ``got``), ``stuck`` (list of runs), ``stuck_pnl`` (daily stuck P&L
        as a share of the previous bar's book value), ``dead`` (targets the book never holds),
        ``missing`` (time-averaged missing long and short exposure) and ``returns``.
    """
    p = lab.panel(test, te0, lab.strategy_features(strat))
    if hedge:
        p = lab.with_spy(p)
    sig = strat.signals(p, **strat.params)
    res = lab.book(strat, p, settings, te0, signals=sig)
    rebuild = float(np.max(np.abs(res["returns"].to_numpy() - saved["returns"].to_numpy())))
    pf, w = res["pf"], res["window"]
    index, cols = w.close.index, list(pf.wrapper.columns)
    logs = pf.logs.records_readable
    lg = logs[logs["Request Size"].notna()].copy()
    req = lg.pivot_table(index="Timestamp", columns="Column", values="Request Size",
                         aggfunc="last").reindex(index=index, columns=cols)
    cat = pd.DataFrame(_classes(req), index=index, columns=cols)
    row_of = {d: k for k, d in enumerate(index)}
    col_of = {c: k for k, c in enumerate(cols)}
    lg["class"] = [CLASSES[cat.iat[row_of[d], col_of[c]]]
                   for d, c in zip(lg["Timestamp"], lg["Column"], strict=True)]
    lg["need"] = lg["Request Size"] * lg["Value"] / lg["Val Price"] - lg["Position"]
    side = np.where(lg["Result Side"] == "Buy", 1.0, np.where(lg["Result Side"] == "Sell", -1.0,
                                                               0.0))
    lg["got"] = side * lg["Result Size"].fillna(0.0)

    tgt = req.ffill().fillna(0.0)
    held = pf.assets().reindex(index=index, columns=cols).fillna(0.0)
    value = pf.value().reindex(index)
    close = w.close.reindex(columns=cols).ffill()
    open_ = w.open_.reindex(columns=cols)
    stuck_mask = ((tgt.abs() < TINY) & (held.abs() > TINY)) | ((tgt > 0) & (held < -TINY)) | (
        (tgt < 0) & (held > TINY))
    pnl = pd.Series(0.0, index=index)
    stuck = []
    for c in stuck_mask.columns[stuck_mask.any()]:
        m = stuck_mask[c].to_numpy()
        for s0, e in _runs(m):
            total = 0.0
            for k in range(max(s0, 1), e + 1):
                if k == s0:     # the exit should have filled at this bar's open
                    d = held[c].iat[k] * (close[c].iat[k] - open_[c].iat[k]) / value.iat[k - 1]
                else:
                    d = held[c].iat[k - 1] * (close[c].iat[k] - close[c].iat[k - 1]) / value.iat[
                        k - 1]
                d = 0.0 if not np.isfinite(d) else float(d)
                pnl.iat[k] += d
                total += d
            due = lg[(lg["Column"] == c) & (lg["Timestamp"] == index[s0])]
            why = (f"{due['Result Status'].iat[0]}/{due['Result Status Info'].iat[0]}"
                   if len(due) else "no order")
            fill = (float(due["got"].iat[0] / due["need"].iat[0])
                    if len(due) and abs(due["need"].iat[0]) > TINY else float("nan"))
            stuck.append({"name": c, "side": "short" if held[c].iat[s0] < 0 else "long",
                          "from": index[s0], "bars": e - s0 + 1, "status": why,
                          "fill": fill, "free cash": float(due["Free Cash"].iat[0])
                          if len(due) else float("nan"),
                          "weight": float(held[c].iat[s0] * close[c].iat[s0] / value.iat[s0]),
                          "weight at end": float(held[c].iat[e] * close[c].iat[e] / value.iat[e]),
                          "pnl": total})

    real = lg[(lg["class"] != "none") & lg["need"].notna() & (lg["need"].abs() > 1e-6)]
    refused = real[real["Result Status"] != "Filled"]
    partial = real[(real["Result Status"] == "Filled") & (real["got"] / real["need"] < 0.99)]
    nan_price = lg[(lg["Result Status Info"] == "PriceNaN") & (lg["Request Size"].abs() > TINY)]
    dead = (tgt.abs() > TINY) & (held.abs() < TINY)
    dead_names = {c: int(dead[c].sum()) for c in dead.columns[dead.any()]}

    # missing exposure: what refused / part-filled entries left out, carried to the name's next
    # order; plus the targets of orders dropped at a NaN price
    miss = pd.DataFrame(0.0, index=index, columns=cols)
    short_of = real[real["class"].isin(["long buy", "new short"])]
    short_of = short_of[(short_of["got"] / short_of["need"]) < 0.99]
    for _, z in short_of.iterrows():
        c, k = z["Column"], row_of[z["Timestamp"]]
        nxt = req[c].iloc[k + 1:].first_valid_index()
        end = row_of[nxt] if nxt is not None else len(index)
        miss.iloc[k:end, col_of[c]] += (z["need"] - z["got"]) * close[c].iloc[k:end].to_numpy()
    miss = miss.div(value, axis=0)
    for _, z in nan_price.iterrows():
        c, k = z["Column"], row_of[z["Timestamp"]]
        nxt = req[c].iloc[k + 1:].first_valid_index()
        end = row_of[nxt] if nxt is not None else len(index)
        miss.iloc[k:end, col_of[c]] += z["Request Size"]
    missing = {"long": float(miss.clip(lower=0).sum(axis=1).mean()) + 0.0,
               "short": float(-miss.clip(upper=0).sum(axis=1).mean()) + 0.0}
    return {"rebuild": rebuild, "orders": len(pf.orders.records), "refused": refused,
            "partial": partial, "nan_price": nan_price, "stuck": stuck, "stuck_pnl": pnl,
            "dead": dead_names, "missing": missing, "returns": res["returns"],
            "bench": res["bench"]}


def _fmt_orders(frame: pd.DataFrame, what: str) -> list[str]:
    """One printed line per order record: class, name, day, status, fill ratio, free cash."""
    out = []
    for _, z in frame.iterrows():
        ratio = z["got"] / z["need"] if abs(z["need"]) > TINY else float("nan")
        out.append(f"    {what:8s} {z['class']:11s} {z['Column']:6s} {z['Timestamp'].date()} "
                   f"{z['Result Status']}/{z['Result Status Info']} filled {ratio:.2f} of "
                   f"{z['need']:+.3f} sh (target {z['Request Size']:+.4f}), free cash "
                   f"{z['Free Cash']:.2f}")
    return out


def label_run(label: str, tests: list[str]) -> dict:
    """Reconcile every period of ``tests`` for one saved label and print the findings.

    Returns:
        ``{test: [period dicts]}`` in the order of ``lab.periods(test)``.
    """
    res = lab.load_results(KEY, label)
    strat = lab.load_strategy(KEY, res["meta"]["overrides"])
    settings = lab._settings()
    out, tot = {}, {"refused": {}, "partial": {}, "stuck": 0, "nan_price": 0}
    for t in tests:
        rows = []
        for i, (te0, _) in enumerate(lab.periods(t)):
            r = period(strat, settings, t, te0, res[t][i], bool(res["meta"].get("hedge")))
            for key in ("refused", "partial"):
                for cls, n in r[key]["class"].value_counts().items():
                    tot[key][cls] = tot[key].get(cls, 0) + int(n)
            tot["stuck"] += len(r["stuck"])
            tot["nan_price"] += len(r["nan_price"])
            a0 = lab.period_alpha(r["returns"], r["bench"])
            a1 = lab.period_alpha(r["returns"] - r["stuck_pnl"], r["bench"])
            lines = [f"{label} {t} {te0}: rebuild max|diff| {r['rebuild']:.1e}; orders "
                     f"{r['orders']}; refused {len(r['refused'])}, part-filled "
                     f"{len(r['partial'])}, NaN-price drops {len(r['nan_price'])}; missing "
                     f"exposure long {r['missing']['long']:.2%} short {r['missing']['short']:.2%}"
                     f"; period alpha {a0:+.4f}" + (f" -> stuck P&L removed {a1:+.4f}"
                                                    if r["stuck"] else "")]
            lines += _fmt_orders(r["refused"], "REFUSED")
            lines += _fmt_orders(r["partial"], "partial")
            for _, z in r["nan_price"].iterrows():
                lines.append(f"    NaN-price drop {z['Column']} {z['Timestamp'].date()} "
                             f"(target {z['Request Size']:+.4f})")
            for s in r["stuck"]:
                lines.append(
                    f"    STUCK {s['side']} {s['name']} from {s['from'].date()}: {s['bars']} bars "
                    f"({s['status']}, filled {s['fill']:.2f}, free cash {s['free cash']:.2f}); "
                    f"weight {s['weight']:+.4f} -> {s['weight at end']:+.4f}; P&L while stuck "
                    f"{s['pnl']:+.2%} of the book")
            if r["dead"]:
                lines.append(f"    targets never held (bars): {r['dead']}")
            print("\n".join(lines), flush=True)
            rows.append(r)
        out[t] = rows
    print(f"{label} TOTAL refused {tot['refused']}; part-filled {tot['partial']}; NaN-price "
          f"drops {tot['nan_price']}; stuck runs {tot['stuck']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"reconcile-{label}.pkl"
    keep = pickle.loads(path.read_bytes()) if path.exists() else {}   # other tests are kept
    keep.update({t: [{k: r[k] for k in ("stuck", "stuck_pnl", "dead", "missing", "rebuild")}
                     for r in rows] for t, rows in out.items()})
    path.write_bytes(pickle.dumps(keep))
    return out


def _stitch(rows, adjusted: bool):
    """``lab.stitched`` over a test's periods, with each period's stuck P&L removed if
    ``adjusted``."""
    return lab.stitched([((r["returns"] - r["stuck_pnl"]) if adjusted else r["returns"],
                          r["bench"]) for r in rows])


def _paired(rows, base_rows, adjusted: bool) -> tuple[float, float, str]:
    """Paired alpha delta against a base label, as ``lab.report`` computes it.

    Returns:
        ``(delta, P(delta > 0), "periods better/periods")``.
    """
    s, b = _stitch(rows, adjusted), _stitch(base_rows, adjusted)
    pr = lab.paired(s["returns"], b["returns"].reindex(s["returns"].index).fillna(0.0),
                    b["bench"].reindex(s["returns"].index).fillna(0.0), b["years"])
    d = [lab.period_alpha((x["returns"] - x["stuck_pnl"]) if adjusted else x["returns"],
                          x["bench"])
         - lab.period_alpha((y["returns"] - y["stuck_pnl"]) if adjusted else y["returns"],
                            y["bench"]) for x, y in zip(rows, base_rows, strict=True)]
    return pr["delta alpha"], pr["P(delta > 0)"], f"{sum(v > 0 for v in d)}/{len(d)}"


def summarize(results: dict, base: str | None) -> None:
    """Per label and test: stitched alpha as saved and with the stuck P&L removed, plus the
    paired delta against ``base`` (as saved, and with both sides' stuck P&L removed)."""
    for label, out in results.items():
        groups = {t: rows for t, rows in out.items()}
        if "TB" in out and "T3" in out:
            groups["TRAIN (TB+T3)"] = out["TB"] + out["T3"]
        for t, rows in groups.items():
            s0, s1 = _stitch(rows, False), _stitch(rows, True)
            line = (f"{label:9s} {t:13s} alpha {s0['alpha']:+.2%} (P {s0['P(alpha > 0)']:.2f}) "
                    f"-> stuck removed {s1['alpha']:+.2%} (P {s1['P(alpha > 0)']:.2f})")
            if base and label != base and base in results:
                bo = results[base]
                brows = (bo["TB"] + bo["T3"]) if t.startswith("TRAIN") else bo.get(t)
                if brows is not None:
                    d0, p0, n0 = _paired(rows, brows, False)
                    d1, p1, n1 = _paired(rows, brows, True)
                    line += (f"; - {base}: {d0:+.2%} (P {p0:.2f}, {n0}) -> {d1:+.2%} "
                             f"(P {p1:.2f}, {n1})")
            print(line, flush=True)


def main(argv: list[str]) -> None:
    """CLI: ``<label>[,<label>...] [tests] [--base <label>]`` (see the module docstring)."""
    args = [a for a in argv if not a.startswith("--")]
    base = argv[argv.index("--base") + 1] if "--base" in argv else None
    if base in args:
        args.remove(base)
    labels = args[0].split(",")
    tests = args[1].split(",") if len(args) > 1 else ["TB", "T3", "T1", "T2"]
    if LOCKED & set(tests):
        raise SystemExit(f"{sorted(LOCKED & set(tests))}: locked tests are never reconciled here")
    order = ([base] if base and base not in labels else []) + labels
    results = {label: label_run(label, tests) for label in order}
    summarize(results, base)


if __name__ == "__main__":
    main(sys.argv[1:])
