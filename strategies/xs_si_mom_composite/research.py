"""xs_si_mom_composite research helpers (loaded by path; ``strategy.py`` never imports this).

    uv run python strategies/xs_si_mom_composite/research.py audit <labels> [tests]
    uv run python strategies/xs_si_mom_composite/research.py premise [TB,T3]
    uv run python strategies/xs_si_mom_composite/research.py grid '<base overrides>' [tests]
    uv run python strategies/xs_si_mom_composite/research.py bars
    uv run python strategies/xs_si_mom_composite/research.py delta <label> <base> [tests]
    uv run python strategies/xs_si_mom_composite/research.py stuck <labels> [tests] [base]
    uv run python strategies/xs_si_mom_composite/research.py exposure [tests]
    uv run python strategies/xs_si_mom_composite/research.py noise_delta [label] [base]

``audit``: does the lab's runner hold the book the strategy asks for? Mechanics only — each
period rebuilt with vectorbt's order log (:func:`rebuild`) and asserted equal to ``lab.book``;
orders asked, rejected, part-filled, dropped at a NaN price, off-target name-days (a target
not held, held on the wrong side, or a position held against a zero target: :func:`off_target`),
the gross never held, the stuck orders' gap P&L (:func:`gap_pnl`), and the book invariants on
the real weights. No return level is read.
``premise``: TRAIN month-end rank IC and top-15 minus bottom-15 spread of each label's score
against the next 21 bars' beta-adjusted return (informational). ``grid``: the pre-registered
exhaustive w_mom grid (robust score, PBO). ``bars``: the pre-registered bars of RESEARCH_LOG §0.6
from the saved labels. ``delta``: paired Δ of one label against another per test and on the
stitched TRAIN record (``<strategy>/<label>`` reads another strategy's saved result).
``stuck``: every refused or part-filled order, how long its position stayed off target and the
P&L of that gap from the due bar's OPEN (when the order should have filled), each test's alpha
with the gaps' P&L removed and, with ``base``, the paired Δ against it both ways (an estimate;
the lab number stays the record). ``exposure``: positions only — the books' net sector exposure
and how many of the composite's names each single-sleeve book also holds. ``noise_delta``: the
paired Δ of two labels' ``lab.noise`` copies, which drop the same names (RESEARCH_LOG §12.0).
``audit`` and ``rebuild`` are resid_momentum/research.py's with the strategy name as an argument.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd

REPO = "/Users/janmarti/Documents/code/stock-trading-simple"
NAME = "xs_si_mom_composite"
#: The design's labels (RESEARCH_LOG §0.3); their overrides on config.yaml.
LABELS = {"A_U0": {}, "C_si": {"long_sleeves": "si", "short_sleeves": "si"},
          "C_mom": {"long_sleeves": "mom", "short_sleeves": "mom"}, "C_mix": {"combine": "book"},
          "I1": {"short_gate": "prof"}, "TUNED": {"w_mom": 1.5}, "I2": {"neutral": "sector"}}
_LAB = {}
#: The lab's locked tests: never built, read or reconciled here (the orchestrator reads them).
LOCKED = ("T4", "TL")


def _open_only(tests) -> tuple:
    """``tests`` as a tuple, refusing the locked ones."""
    if set(tests) & set(LOCKED):
        raise SystemExit(f"{sorted(set(tests) & set(LOCKED))}: locked tests are never read here")
    return tuple(tests)


def _lab():
    """The proposals lab (``strategies/proposals_lab/lab.py``), loaded once by path."""
    if "lab" not in _LAB:
        spec = importlib.util.spec_from_file_location(
            "lab", f"{REPO}/strategies/proposals_lab/lab.py")
        lab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lab)
        _LAB["lab"] = lab
    return _LAB["lab"]


def _load(label: str) -> dict:
    """A saved lab v2 label of this strategy, or ``<strategy>/<label>`` of another (read-only)."""
    strategy, _, name = label.rpartition("/")
    return _lab().load_results(strategy or NAME, name)


def rebuild(strat, window, sig: dict, settings, log: bool = True):
    """One period's book rebuilt exactly as the lab (v2) builds it, with vectorbt's order log:
    ``lab.run_portfolio``'s call order (long cuts, short covers, new shorts, long buys) and its
    borrow fees, inside the engine's ``retrim_k`` loop. Returns the final Portfolio."""
    import vectorbt as vbt

    from trading_simple import backtest as bt
    lab = _lab()
    fees, slip = bt.strategy_costs(strat, window, settings)
    price, slip = bt.fill_prices(window, slip, strat.fill)
    weights = sig["weights"]
    targets = bt.rotation_targets(weights)
    if "borrow" in sig:
        fees = lab.borrow_fees(targets, sig["borrow"], fees)
    bt.configure()

    def build(tg):
        return vbt.Portfolio.from_orders(
            window.close, size=tg, size_type="targetpercent", price=price, fees=fees,
            slippage=slip, init_cash=settings.backtest.init_cash, group_by=True,
            cash_sharing=True, lock_cash=True, call_seq=lab._call_order(tg), freq=window.freq,
            log=log)

    k = sig.get("retrim_k")
    return build(targets) if k is None else bt._retrim_loop(build, targets, weights, k * weights)


def off_target(assets: pd.DataFrame, weights: pd.DataFrame, close: pd.DataFrame,
               tiny: float = 1e-9) -> pd.DataFrame:
    """Name-days, on priced bars, on which the book does not hold its target's side.

    Both directions count: a target not held (a refused entry) or held on the wrong side, AND
    a position held against a zero target (an exit that did not fill: the stuck short of a
    part-filled cover). A position under ``tiny`` shares counts as flat.

    Args:
        assets: The book's shares at each bar's close (``pf.assets()``).
        weights: The target weights in force on each bar (after the open fill).
        close: The window's closes; unpriced name-days are not counted.
        tiny: Shares below which a position is flat.

    Returns:
        A boolean frame shaped like ``assets``.
    """
    held = np.sign(assets.where(assets.abs() > tiny, 0.0))
    want = np.sign(weights.reindex_like(assets).fillna(0.0))
    return held.ne(want) & close.reindex_like(assets).notna()


def gap_pnl(shares: float, open_, close, prev_value, at: int, nxt: int) -> np.ndarray:
    """Daily P&L of ``shares`` held off target, as a share of the previous bar's book value.

    The order was due at bar ``at``'s open and the name's next order fills at bar ``nxt``'s
    open (``nxt == len(close)``: none inside the window). Had the due order filled, the gap
    would have traded at ``at``'s open, so that bar books ``shares × (close − open)``, the bars
    after it close to close, and bar ``nxt`` the overnight ``open − previous close`` before the
    next order trades the gap away; in dollars the pieces telescope to ``shares × (open[nxt] −
    open[at])`` (to the window's last close when ``nxt`` is past it). This is
    adaptive_hedge/reconcile.py's booking. The costs of trading the gap at the due open are not
    charged (about 0.1% of its notional).

    Args:
        shares: Position minus the shares the due order asked for (< 0: short of target).
        open_, close: The name's fill (open) prices and its forward-filled closes.
        prev_value: The book's value at the previous bar's close (the initial cash on bar 0).
        at: Bar of the due order.
        nxt: Bar of the name's next order, or ``len(close)``.

    Returns:
        One value per bar, zero outside ``at .. nxt``.
    """
    o, c, v = (np.asarray(x, dtype=float) for x in (open_, close, prev_value))
    end = min(nxt, len(c))
    move = np.zeros(len(c))
    move[at] = c[at] - o[at]
    move[at + 1:end] = np.diff(c[at:end])
    if nxt < len(c):
        move[nxt] = o[nxt] - c[nxt - 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = shares * move / v
    return np.where(np.isfinite(out), out, 0.0)


def invariants(weights: pd.DataFrame, *, slots: int, gross: float, constant: bool) -> dict:
    """The book invariants on one weight frame: names a side, gross, one weight per hold."""
    w = weights.fillna(0.0)
    side = np.sign(w)
    hold = side.ne(side.shift(fill_value=0.0)).cumsum()
    multi = 0
    if constant:
        for c in w.columns[(w != 0).any()]:
            held = side[c] != 0
            multi += int((w[c][held].groupby(hold[c][held]).nunique() > 1).sum())
    return {"max longs": int((w > 0).sum(axis=1).max()),
            "max shorts": int((w < 0).sum(axis=1).max()),
            "max gross": float(w.abs().sum(axis=1).max()),
            "holds with >1 weight": multi,
            "ok": bool((w > 0).sum(axis=1).max() <= slots and (w < 0).sum(axis=1).max() <= slots
                       and w.abs().sum(axis=1).max() <= gross + 1e-9 and multi == 0)}


def audit(labels=("A_U0", "C_si", "C_mom", "C_mix"), tests=("TB", "T1")) -> pd.DataFrame:
    """Mechanics only, no returns: does the engine hold the book each label asks for?

    Every period is rebuilt with :func:`rebuild` and asserted equal to ``lab.book`` (day 0
    counted from the initial cash as ``lab.book`` does). From vectorbt's log, per label and
    test: orders asked (a non-zero change), rejected, partly filled (below 99.5% of the shares
    asked), dropped at a NaN price, the share of orders rejected or part-filled (the §0.5
    rule's 1%), split by kind (long buy, long sale, new short, cover), off-target name-days
    (:func:`off_target`: both a target not held and a position held against a zero target),
    the mean gross never held and the stuck orders' summed gap P&L from their due open
    (:func:`_gaps`, a share of the book); plus :func:`invariants` on the scored window's weights.
    """
    from trading_simple import backtest as bt
    lab = _lab()
    tests = _open_only(tests)
    settings = lab._settings()
    rows = {}
    for label in labels:
        over = LABELS[label]
        strat = lab.load_strategy(NAME, over)
        for t in tests:
            acc = {"orders asked": 0, "rejected": 0, "partly filled": 0, "no price": 0,
                   "off-target days": 0, "stuck orders": 0, "gap P&L": 0.0,
                   "missing gross x bars": 0.0, "bars": 0,
                   "short kinds": {}, "invariants ok": True, "max gross": 0.0,
                   "max longs": 0, "max shorts": 0, "holds with >1 weight": 0}
            for te0, _ in lab.periods(t):
                p = lab.panel(t, te0, lab.strategy_features(strat))
                sig = strat.signals(p, **strat.params)
                window, s = bt._score_window(p, sig, te0)
                inv = invariants(s["weights"], slots=strat.params["slots"],
                                 gross=strat.params["gross"],
                                 constant=strat.params["combine"] == "rank")
                acc["invariants ok"] &= inv["ok"]
                for k in ("max gross", "max longs", "max shorts"):
                    acc[k] = max(acc[k], inv[k])
                acc["holds with >1 weight"] += inv["holds with >1 weight"]
                pf = rebuild(strat, window, s, settings)
                ref = lab.book(strat, p, settings, te0, signals=sig)["returns"]
                value = bt.equity_curve(pf)["value"]
                mine = value / value.shift(1).fillna(settings.backtest.init_cash) - 1.0
                assert float((mine - ref).abs().max()) < 1e-12, (label, t, te0)
                lg = pf.logs.values
                lg = lg[np.isfinite(lg["req_size"])]
                with np.errstate(invalid="ignore", divide="ignore"):
                    want = lg["req_size"] * lg["value"] / lg["val_price"] - lg["position"]
                got = np.where(lg["res_status"] == 0, lg["res_size"], 0.0)   # 0 = Filled
                nonzero = np.abs(want) > 1e-9 * np.maximum(np.abs(lg["position"]), 1.0)
                reject = (lg["res_status"] == 2) & nonzero                   # 2 = Rejected
                noprice = ((lg["res_status"] == 1) & np.isin(lg["res_status_info"], (1, 2))
                           & ((lg["req_size"] != 0) | (lg["position"] != 0)))
                partial = (lg["res_status"] == 0) & nonzero & (np.abs(got) < 0.995 * np.abs(want))
                short = reject | partial
                pos = lg["position"]
                kind = np.where((pos < 0) & (want > 0), "cover",
                                np.where((pos > 0) & (want < 0), "long sale",
                                         np.where(want < 0, "new short", "long buy")))
                for k in np.unique(kind[short]):
                    acc["short kinds"][str(k)] = acc["short kinds"].get(str(k), 0) + int(
                        (kind[short] == k).sum())
                gap = (np.abs(want) - np.abs(got)) * lg["val_price"] / lg["value"]
                n = len(window.close)
                for i in np.flatnonzero(short):
                    c, at = lg["col"][i], lg["idx"][i]
                    later = np.sort(lg["idx"][(lg["col"] == c) & (lg["idx"] > at)])
                    acc["missing gross x bars"] += float(gap[i]) * float(
                        (later[0] if len(later) else n) - at)
                acc["orders asked"] += int(nonzero.sum())
                acc["rejected"] += int(reject.sum())
                acc["partly filled"] += int(partial.sum())
                acc["no price"] += int(noprice.sum())
                acc["off-target days"] += int(off_target(pf.assets(), s["weights"],
                                                         window.close).to_numpy().sum())
                stuck_rows, gap_ret = _gaps(pf, window, settings.backtest.init_cash)
                acc["stuck orders"] += len(stuck_rows)
                acc["gap P&L"] += float(gap_ret.sum())
                acc["bars"] += n
            acc["mean missing gross"] = acc.pop("missing gross x bars") / max(acc["bars"], 1)
            acc["refused or partial share"] = (acc["rejected"] + acc["partly filled"]) / max(
                acc["orders asked"], 1)
            rows[(label, t)] = acc
            print(label, t, json.dumps(acc, default=str), flush=True)
    out = pd.DataFrame(rows).T
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.to_string())
    rule = out["refused or partial share"].astype(float)
    print(f"§0.5 rule: max refused-or-partial share {rule.max():.4%} "
          f"-> gross {'0.85' if rule.max() > 0.01 else 'stays 0.90'}")
    return out


def premise(tests=("TB", "T3"), labels=("A_U0", "C_si", "C_mom"), horizon: int = 21) -> dict:
    """TRAIN month-end rank IC and top-15 minus bottom-15 spread (informational).

    At every month end inside a test period's window whose next ``horizon`` bars are in the
    panel: each label's book score (the smoothed rank that drives ``_select``) against the
    beta-adjusted forward return ``close(t + h) / close(t) − 1 − β_t × m``, with ``m`` the
    compounded equal-weight return of the eligible names over the same bars and ``β_t`` p1's
    trailing 126-bar beta known at ``t``.
    """
    lab = _lab()
    tests = _open_only(tests)
    mod = lab.load_strategy(NAME, {}).signals.__globals__
    out = {}
    for label in labels:
        strat = lab.load_strategy(NAME, LABELS[label])
        prm = strat.params
        lo = prm["long_sleeves"].split("+")
        w = {"si": prm["w_si"], "mom": prm["w_mom"], "prof": prm["w_prof"]}
        recs = []
        for t in tests:
            for te0, te1 in lab.periods(t):
                p = lab.panel(t, te0, lab.strategy_features(strat))
                r = mod["ranks"](p, set(lo), dtc_window=prm["dtc_window"],
                                 neutral_groups=prm["neutral_groups"],
                                 min_price=prm["min_price"])
                ok = r["base"]["eligible"]
                score = mod["side_score"](r["ranks"], lo, w, ok, prm["score_smooth"])
                close = p.close
                ret = close.pct_change(fill_method=None)
                growth = (1 + ret.where(ok).mean(axis=1).fillna(0.0)).cumprod()
                idx = close.index
                month = idx.year * 12 + idx.month
                ends = np.flatnonzero(np.r_[month[1:] != month[:-1], False])
                inside = (idx >= pd.Timestamp(te0, tz="UTC")) & (idx <= pd.Timestamp(te1,
                                                                                       tz="UTC"))
                for a in ends:
                    b = a + horizon
                    if not inside[a] or b >= len(idx):
                        continue
                    fwd = close.iloc[b] / close.iloc[a] - 1
                    mkt = growth.iloc[b] / growth.iloc[a] - 1
                    y = fwd - r["base"]["beta"].iloc[a] * mkt
                    x = score.iloc[a]
                    keep = x.notna() & y.notna()
                    x, y = x[keep], y[keep]
                    if len(x) < 50:
                        continue
                    order = x.sort_values()
                    recs.append({"test": t, "date": idx[a], "ic": x.corr(y, method="spearman"),
                                 "spread": y[order.index[-15:]].mean()
                                 - y[order.index[:15]].mean()})
        tab = pd.DataFrame(recs)

        def summ(g):
            def tstat(s):
                return float(s.mean() / s.std(ddof=1) * np.sqrt(len(s)))
            return {"months": len(g), "IC": round(float(g["ic"].mean()), 4),
                    "IC t": round(tstat(g["ic"]), 2), "spread/21d": round(float(
                        g["spread"].mean()), 4), "spread t": round(tstat(g["spread"]), 2)}
        out[label] = {"all": summ(tab), **{t: summ(g) for t, g in tab.groupby("test")}}
    print(json.dumps(out, indent=1))
    return out


GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)   #: the pre-registered w_mom points


def grid(base: dict | None = None, values=GRID,
         tests=("TB", "T3"), prefix: str = "G") -> dict:
    """The pre-registered exhaustive w_mom grid on ``tests`` (base books only).

    Each point is saved as the label ``<prefix>_w<w>`` (base overrides + ``w_mom``); its per-
    period alphas form one row of the points × periods matrix. Returns each point's robust
    score (mean − ½ sd, ``lab.robust``) and mean alpha, the best point and the PBO.
    """
    lab = _lab()
    tests = _open_only(tests)
    base = base or {}
    rows, out = [], {}
    for v in values:
        label = f"{prefix}_w{v:.2f}"
        over = {**base, "w_mom": float(v)}
        path = lab.results_path(NAME, label)
        res = lab.load_results(NAME, label) if path.exists() else lab.run(
            NAME, label, over, list(tests), late=False)
        alphas = [lab.period_alpha(r["returns"], r["bench"]) for t in tests for r in res[t]]
        rows.append(alphas)
        out[label] = {"w_mom": float(v), "robust": lab.robust(alphas),
                      "mean": float(np.mean(alphas)), "periods": [round(a, 4) for a in alphas]}
    m = np.array(rows)
    best = max(out, key=lambda k: out[k]["robust"])
    summary = {"points": out, "best": best, "best w_mom": out[best]["w_mom"],
               "pbo": lab.pbo(m)}
    print(json.dumps(summary, indent=1))
    return summary


def _pair(rows_a, rows_b) -> dict:
    """Paired Δ alpha of two labels' period rows on the same days and bootstrap blocks (the
    benchmark is ``rows_a``'s), both stitched alphas and how many periods ``rows_a`` won."""
    lab = _lab()
    sa = lab.stitched([(r["returns"], r["bench"]) for r in rows_a])
    sb = lab.stitched([(r["returns"], r["bench"]) for r in rows_b])
    idx = sa["returns"].index
    pr = lab.paired(sa["returns"], sb["returns"].reindex(idx).fillna(0.0),
                    sa["bench"].reindex(idx).fillna(0.0), sa["years"])
    better = [lab.period_alpha(x["returns"], x["bench"])
              - lab.period_alpha(y["returns"], y["bench"])
              for x, y in zip(rows_a, rows_b, strict=True)]
    return {"alpha": sa["alpha"], "base alpha": sb["alpha"], "Δ": pr["delta alpha"],
            "P(Δ>0)": pr["P(delta > 0)"],
            "better": f"{sum(d > 0 for d in better)}/{len(better)}"}


def delta(label: str, base: str, tests=("TB", "T3", "T1", "T2")) -> dict:
    """Paired Δ alpha of ``label`` minus ``base`` per test and on the stitched TRAIN (TB + T3)
    record (the same days and bootstrap blocks; the benchmark is ``label``'s)."""
    a, b = _load(label), _load(base)
    out = {}
    for t in tests:
        if t in a and t in b:
            out[t] = _pair(a[t], b[t])
    train = [t for t in ("TB", "T3") if t in a and t in b]
    if train:
        out["TRAIN"] = _pair([r for t in train for r in a[t]], [r for t in train for r in b[t]])
    print(f"{label} − {base}:")
    for k, v in out.items():
        print(f"  {k:6s} alpha {v['alpha']:+.4f}  base {v['base alpha']:+.4f}  "
              f"Δ {v['Δ']:+.4f} (P {v['P(Δ>0)']:.3f}, better {v['better']})")
    return out


def _gaps(pf, window, init_cash: float):
    """The refused or part-filled orders of one rebuilt period and each one's daily gap P&L.

    An order's gap is the position it left minus the shares it asked for (vectorbt's log:
    ``req_size × value / val_price``), held from its due bar's OPEN until the name's next order
    (:func:`gap_pnl`; the strategy fills at the open). Returns ``(rows, gap)``: one dict per
    order (date, name, kind, signed shares wanted / got, the gap in shares, bars stuck, the
    gap's P&L as a share of the book and the due bar's part of it) and the summed daily gap
    return. Removing ``gap`` from the book's returns estimates the book had every such order
    filled at its due open, the cash it would have freed (or used) idle.
    """
    lg = pf.logs.values
    lg = lg[np.isfinite(lg["req_size"])]
    with np.errstate(invalid="ignore", divide="ignore"):
        target = lg["req_size"] * lg["value"] / lg["val_price"]
    want = target - lg["position"]
    filled = lg["res_status"] == 0                                    # 0 = Filled
    got = np.where(filled, lg["new_position"] - lg["position"], 0.0)
    nonzero = np.abs(want) > 1e-9 * np.maximum(np.abs(lg["position"]), 1.0)
    bad = ((lg["res_status"] == 2) | (filled & (np.abs(got) < 0.995 * np.abs(want)))) & nonzero
    assets, value = pf.assets(), pf.value()
    prev = value.shift(1).fillna(init_cash).to_numpy()
    index, n = window.close.index, len(window.close)
    gap_ret = np.zeros(n)
    rows = []
    for i in np.flatnonzero(bad):
        c, at = int(lg["col"][i]), int(lg["idx"][i])
        name = window.close.columns[c]
        later = np.sort(lg["idx"][(lg["col"] == c) & (lg["idx"] > at)])
        nxt = int(later[0]) if len(later) else n
        shares = float(assets[name].iloc[at] - target[i])
        g = gap_pnl(shares, window.open_[name], window.close[name].ffill(), prev, at, nxt)
        gap_ret += g
        pos = lg["position"][i]
        rows.append({"date": str(index[at].date()), "name": name,
                     "kind": "cover" if pos < 0 < want[i] else "new short" if want[i] < 0
                     <= pos else "long sale" if pos > 0 > want[i] else "long buy",
                     "wanted": round(float(want[i]), 2), "got": round(float(got[i]), 2),
                     "gap shares": round(shares, 2),
                     "status info": int(lg["res_status_info"][i]), "bars stuck": int(nxt - at),
                     "gap P&L / book": round(float(g.sum()), 4),
                     "of which the due bar": round(float(g[at]), 4)})
    return rows, pd.Series(gap_ret, index=index)


def stuck(labels=("A_U0",), tests=("TB", "T3", "T1", "T2"), base: str | None = None) -> dict:
    """Refused or part-filled orders per label and test (:func:`_gaps`), and the alpha with
    their gap P&L removed: each such order filled at its due bar's open, the cash it would have
    freed (or used) idle — adaptive_hedge's repair estimate, applied to refused entries too.

    Every rebuilt period is asserted equal to the saved returns. TRAIN (TB + T3) is stitched
    when both are in ``tests``. With ``base``, the paired Δ of each label against it, as run
    and with both sides' gaps removed. The saved lab results stay the record.
    """
    from trading_simple import backtest as bt
    lab = _lab()
    tests = _open_only(tests)
    settings = lab._settings()
    init = settings.backtest.init_cash
    runs, out = {}, {}
    for label in dict.fromkeys([*labels, *([base] if base else [])]):
        strat = lab.load_strategy(NAME, LABELS[label])
        res = _load(label)
        for t in tests:
            fixed, orders = [], []
            for row, (te0, _) in zip(res[t], lab.periods(t), strict=True):
                p = lab.panel(t, te0, lab.strategy_features(strat))
                sig = strat.signals(p, **strat.params)
                window, s = bt._score_window(p, sig, te0)
                pf = rebuild(strat, window, s, settings)
                value = pf.value()
                mine = value / value.shift(1).fillna(init) - 1.0
                assert float((mine - row["returns"]).abs().max()) < 1e-12, (label, t, te0)
                rows, gap = _gaps(pf, window, init)
                orders += rows
                fixed.append({**row, "returns": row["returns"] - gap.reindex(
                    row["returns"].index).fillna(0.0)})
            runs[(label, t)] = (res[t], fixed, orders)
        if "TB" in tests and "T3" in tests:
            tb, t3 = runs[(label, "TB")], runs[(label, "T3")]
            runs[(label, "TRAIN")] = (tb[0] + t3[0], tb[1] + t3[1], tb[2] + t3[2])
    for (label, t), (rows_run, rows_fixed, orders) in runs.items():
        a, b = lab._stitch(rows_run), lab._stitch(rows_fixed)
        out[(label, t)] = {"orders": orders, "alpha as run": a["alpha"],
                           "P as run": a["P(alpha > 0)"], "alpha stuck removed": b["alpha"],
                           "P stuck removed": b["P(alpha > 0)"]}
        if base and label != base:
            run_d = _pair(rows_run, runs[(base, t)][0])
            fix_d = _pair(rows_fixed, runs[(base, t)][1])
            out[(label, t)].update({
                f"Δ vs {base} as run": run_d["Δ"], "P as run Δ": run_d["P(Δ>0)"],
                f"Δ vs {base} stuck removed": fix_d["Δ"], "P stuck removed Δ": fix_d["P(Δ>0)"],
                "better stuck removed": fix_d["better"]})
        print(label, t, json.dumps(out[(label, t)], default=str), flush=True)
    return out


def exposure(tests=("TB", "T3", "T1", "T2"), labels=("A_U0", "C_si", "C_mom", "C_mix")) -> dict:
    """Positions only (no returns): each book's mean Σ|net weight per sector| and its largest
    sector, and how many of A_U0's held names C_si's and C_mom's books also hold, same side."""
    lab = _lab()
    tests = _open_only(tests)
    codes = json.loads((lab.VAR / "sectors.json").read_text())
    names = {float(v): k for k, v in codes.items()}
    strats = {k: lab.load_strategy(NAME, LABELS[k]) for k in labels}
    out = {}
    for t in tests:
        acc: dict = {}
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strats[labels[0]]))
            idx = p.close.index
            win = (idx >= pd.Timestamp(te0, tz="UTC")) & (idx <= pd.Timestamp(te1, tz="UTC"))
            sec = p.features["sector"].iloc[-1].map(names).fillna("none")
            w = {k: s.signals(p, **s.params)["weights"][win] for k, s in strats.items()}
            for k, x in w.items():
                net = x.T.groupby(sec).sum().T.abs()
                acc.setdefault(f"{k} Σ|net sector|", []).append(float(net.sum(axis=1).mean()))
                acc.setdefault(f"{k} largest sector", []).append(float(net.max(axis=1).mean()))
            a = w["A_U0"]
            for k in ("C_si", "C_mom"):
                for side, sign in (("long", 1), ("short", -1)):
                    both = ((a * sign > 0) & (w[k] * sign > 0)).sum(axis=1).mean()
                    acc.setdefault(f"A_U0 {side}s also in {k}", []).append(float(both))
        out[t] = {k: round(float(np.mean(v)), 3) for k, v in acc.items()}
        print(t, json.dumps(out[t]), flush=True)
    return out


def noise_sd(label: str = "A_U0", k: int = 5) -> dict:
    """``lab.noise``'s sd per test from the saved copies, plus the stitched TRAIN's."""
    lab = _lab()
    res = _load(label)
    copies = sorted((lab.RESULTS / NAME).glob(f"{label}-noise{k}-*.pkl"))
    out = {}
    for t in [*[t for t in lab.OPEN if t in res], "TRAIN"]:
        d = []
        for f in copies:
            r = pd.read_pickle(f)
            if t == "TRAIN":
                rows_c = [x for tt in ("TB", "T3") for x in r[tt]]
                rows_l = [x for tt in ("TB", "T3") for x in res[tt]]
            else:
                rows_c, rows_l = r[t], res[t]
            d.append(lab._stitch(rows_c)["alpha"] - lab._stitch(rows_l)["alpha"])
        if len(d) > 1:
            out[t] = {"deltas": [round(x, 4) for x in d], "sd": float(np.std(d, ddof=1))}
    return out


def noise_delta(label: str = "A_U0", base: str = "C_mix", k: int = 5) -> dict:
    """Paired Δ(``label`` copy − ``base`` copy) for each pair of ``lab.noise`` copies.

    ``lab.noise`` seeds copy ``c`` with ``100 + c`` and draws the dropped names test by test
    from the panel's columns, so two labels' copies ``c`` drop the same names; this is checked
    (same seed, same ``drop``, identical benchmark rows, which are the equal-weight book of the
    names kept). Per stitched TRAIN (TB + T3), T1 and T2: the labels' own Δ, each copy pair's
    Δ, P and periods won, and the copies' mean and sd. Informational (RESEARCH_LOG §12.0 (4)).
    """
    lab = _lab()
    pairs = []
    while all((lab.RESULTS / NAME / f"{x}-noise{k}-{len(pairs)}.pkl").exists()
              for x in (label, base)):
        a, b = (_load(f"{x}-noise{k}-{len(pairs)}") for x in (label, base))
        assert (a["meta"]["seed"], a["meta"]["drop"]) == (b["meta"]["seed"], b["meta"]["drop"])
        assert [t for t in a if t in lab.OPEN] == [t for t in b if t in lab.OPEN]
        for t in (t for t in a if t in lab.OPEN):
            for x, y in zip(a[t], b[t], strict=True):
                assert x["bench"].equals(y["bench"]), (t, x["period"])   # the same names
        pairs.append((a, b))
    full = (_load(label), _load(base))
    out = {}
    for g, tests in {"TRAIN": ("TB", "T3"), "T1": ("T1",), "T2": ("T2",)}.items():
        def rows(res, tests=tests):
            return [x for t in tests for x in res[t]]
        own = _pair(rows(full[0]), rows(full[1]))
        ds = [_pair(rows(a), rows(b)) for a, b in pairs]
        d = np.array([x["Δ"] for x in ds])
        out[g] = {"labels Δ": round(own["Δ"], 4), "labels P": own["P(Δ>0)"],
                  "copies Δ": [round(x, 4) for x in d], "copies P": [x["P(Δ>0)"] for x in ds],
                  "copies better": [x["better"] for x in ds],
                  "copies mean": round(float(d.mean()), 4) if len(d) else None,
                  "copies sd": round(float(d.std(ddof=1)), 4) if len(d) > 1 else None}
    print(json.dumps({"label": label, "base": base, "copy pairs": len(pairs), **out}, indent=1))
    return out


def bars(label: str = "A_U0") -> dict:
    """The pre-registered bars (RESEARCH_LOG §0.6) for ``label`` against the controls."""
    lab = _lab()
    res = {k: _load(k) for k in (label, "C_si", "C_mom", "C_mix")}

    def alpha(r, tests):
        return lab.stitched([(x["returns"], x["bench"]) for t in tests for x in r[t]])["alpha"]

    level = {k: {"TRAIN": alpha(r, ("TB", "T3")), "T1": alpha(r, ("T1",)),
                 "T2": alpha(r, ("T2",))} for k, r in res.items()}
    nz = noise_sd("A_U0")
    mix = delta(label, "C_mix")
    b2 = {t: mix[t]["Δ"] for t in ("TRAIN", "T1", "T2")}
    best = {t: max(level["C_si"][t], level["C_mom"][t]) for t in ("TRAIN", "T1", "T2")}
    dm = {k: lab.down_months(r) for k, r in res.items()}
    ep = lab.episodes(res[label])
    g6 = bool(ep and all(v["book"] > v["market"] for v in ep.values())
              and dm[label].get("book mean", -1) > 0)
    out = {"levels": level, "noise sd": {t: v["sd"] for t, v in nz.items()},
           "bar2 Δ vs C_mix": b2,
           "bar2": bool(min(b2.values()) >= 0 and b2["T1"] > 2 * nz["T1"]["sd"]),
           "best single sleeve": best,
           "bar3": bool(level[label]["TRAIN"] >= best["TRAIN"] and level[label]["T2"] >= best["T2"]
                        and level[label]["T1"] >= best["T1"] - 0.01),
           "down months": {k: v.get("book mean") for k, v in dm.items()},
           "G6": g6, "bar4": bool(g6 and dm[label]["book mean"] > dm["C_si"]["book mean"])}
    print(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "audit":
        audit(tuple(a[1].split(",")) if len(a) > 1 else ("A_U0", "C_si", "C_mom", "C_mix"),
              tuple(a[2].split(",")) if len(a) > 2 else ("TB", "T1"))
    elif a and a[0] == "premise":
        premise(tuple(a[1].split(",")) if len(a) > 1 else ("TB", "T3"))
    elif a and a[0] == "grid":
        grid(json.loads(a[1]) if len(a) > 1 else {},
             tests=tuple(a[2].split(",")) if len(a) > 2 else ("TB", "T3"))
    elif a and a[0] == "bars":
        bars(a[1] if len(a) > 1 else "A_U0")
    elif a and a[0] == "delta":
        delta(a[1], a[2], tuple(a[3].split(",")) if len(a) > 3 else ("TB", "T3", "T1", "T2"))
    elif a and a[0] == "stuck":
        stuck(tuple(a[1].split(",")) if len(a) > 1 else ("A_U0",),
              tuple(a[2].split(",")) if len(a) > 2 else ("TB", "T3", "T1", "T2"),
              a[3] if len(a) > 3 else None)
    elif a and a[0] == "exposure":
        exposure(tuple(a[1].split(",")) if len(a) > 1 else ("TB", "T3", "T1", "T2"))
    elif a and a[0] == "noise_sd":
        print(json.dumps(noise_sd(a[1] if len(a) > 1 else "A_U0"), indent=1))
    elif a and a[0] == "noise_delta":
        noise_delta(a[1] if len(a) > 1 else "A_U0", a[2] if len(a) > 2 else "C_mix")
    else:
        raise SystemExit(__doc__)
