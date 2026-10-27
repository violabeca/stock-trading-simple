"""xs_cross_leg research helpers (loaded by path; ``strategy.py`` never imports this).

    uv run python strategies/xs_cross_leg/research.py audit [labels] [tests]
    uv run python strategies/xs_cross_leg/research.py stuck <labels> [tests] [base]
    uv run python strategies/xs_cross_leg/research.py delta <label> <base> [tests]
    uv run python strategies/xs_cross_leg/research.py betas [labels] [tests]
    uv run python strategies/xs_cross_leg/research.py legs <label> [tests]
    uv run python strategies/xs_cross_leg/research.py overlap [tests]
    uv run python strategies/xs_cross_leg/research.py noise_sd [label]
    uv run python strategies/xs_cross_leg/research.py bars [label]

``audit``: does the lab's runner hold the book the strategy asks for? Mechanics only, no return
level. Each period is rebuilt with vectorbt's order log (:func:`rebuild`) and asserted equal to
``lab.book``. From the log: orders asked, refused, part-filled, dropped at a NaN price,
off-target name-days both ways (:func:`off_target`), the gross never held and the stuck
orders' gap P&L from their due open (:func:`_gaps`). On the weights: the book invariants
(:func:`invariants`) and the flips. Against the research preview (:func:`preview`, the targets
replayed frictionless outside vectorbt): daily correlation, tracking error, the yearly drag
(costs, borrow and failed orders) and the largest daily gaps. ``stuck``: every refused or
part-filled order and each test's alpha with the gaps' P&L removed (an estimate; the lab
number stays the record), paired against ``base`` both ways. ``delta``: paired Δ alpha per test
and on the stitched TRAIN (TB + T3) record (``<strategy>/<label>`` reads another strategy's
saved result, read-only). ``betas``: per period, each leg's and the book's ex-ante beta and the
realized beta. ``legs``: each leg's daily contribution (adaptive_hedge/legs.py's pattern).
``overlap``: positions only — how many of a label's names the single-sleeve books also hold,
and how often the long mask binds. ``bars``: the pre-registered bars of RESEARCH_LOG §0.6.

``rebuild``, ``off_target``, ``gap_pnl`` and ``_gaps`` are xs_si_mom_composite/research.py's
review-fixed versions (its RESEARCH_LOG §12) with this strategy's name. Locked tests (T4, TL)
are refused everywhere: the orchestrator reads them.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd

REPO = "/Users/janmarti/Documents/code/stock-trading-simple"
NAME = "xs_cross_leg"
CORE = "xs_si_mom_composite"
#: This design's labels (RESEARCH_LOG §0.2, §0.3, §0.7); their overrides on config.yaml.
LABELS = {"C_U0": {}, "C_rev": {"long_sleeves": "mom", "short_sleeves": "si"},
          "I1": {"short_sleeves": "si+mom"}, "I2b": {"short_sizing": "beta"}}
#: The composite's labels this design reads (read-only; the same strategy.py and panels).
CORE_LABELS = {"A_U0": {}, "C_si": {"long_sleeves": "si", "short_sleeves": "si"},
               "C_mom": {"long_sleeves": "mom", "short_sleeves": "mom"}}
#: The lab's locked tests: never built, read or reconciled here (the orchestrator reads them).
LOCKED = ("T4", "TL")
OPEN = ("TB", "T3", "T1", "T2")
_LAB = {}


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


def _split(label: str) -> tuple[str, str]:
    """``(strategy, label)`` of ``label`` or ``<strategy>/<label>`` (this strategy by default)."""
    strategy, _, name = label.rpartition("/")
    return strategy or NAME, name


def _load(label: str) -> dict:
    """A saved lab v2 label of this strategy, or ``<strategy>/<label>`` of another (read-only)."""
    return _lab().load_results(*_split(label))


def _strategy(label: str):
    """The loaded strategy of a label: overrides from its saved meta when it exists, else from
    :data:`LABELS` (this strategy) or :data:`CORE_LABELS` (the composite)."""
    lab = _lab()
    strategy, name = _split(label)
    path = lab.results_path(strategy, name)
    if path.exists():
        over = lab.load_results(strategy, name)["meta"]["overrides"]
    else:
        over = LABELS[name] if strategy == NAME else CORE_LABELS[name]
    return lab.load_strategy(strategy, over)


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


def preview(window, weights: pd.DataFrame, retrim_k=None, init: float = 1.0) -> pd.Series:
    """The research preview: the book the strategy asks for, replayed without vectorbt.

    Target-percent orders at each bar's open, only where a name's target changes
    (``bt.rotation_targets``); the book is valued at the bar's opens for sizing (a name without
    an open at its last close) and at the closes for the return. Shares are fractional, the cash
    is shared and never locked, and there are no fees, slippage or borrow. Every order fills: one
    due at a missing open waits for the name's next priced open. With ``retrim_k`` a long whose
    close weight reaches ``k ×`` its target is trimmed back at the next open (the engine's
    ``_retrim_loop``). Day 0 starts from ``init`` cash, as ``lab.book`` counts it. Without costs
    and failed orders it equals ``lab.run_portfolio`` to rounding (tested), so the engine's book
    minus this one is its costs, its borrow and every order it refused, part-filled or dropped.

    Args:
        window: The scored window's panel.
        weights: The target weights at each fill bar (the strategy's ``weights`` frame).
        retrim_k: The long cap (``None``: off).
        init: Starting cash.

    Returns:
        The preview's daily returns on the window's index.
    """
    from trading_simple import backtest as bt
    close = window.close
    w = weights.reindex(index=close.index, columns=close.columns).fillna(0.0)
    order = bt.rotation_targets(w).to_numpy(dtype=float)
    dense = w.to_numpy(dtype=float)
    o = window.open_.reindex_like(close).to_numpy(dtype=float)
    c = close.ffill().to_numpy(dtype=float)
    n, m = c.shape
    shares, cash, last, out = np.zeros(m), init, init, np.zeros(n)
    pending, over = np.full(m, np.nan), np.zeros(m, bool)
    val = np.full(m, np.nan)
    for t in range(n):
        val = np.where(np.isfinite(o[t]), o[t], val)
        value = cash + np.nansum(shares * val)
        pending = np.where(np.isfinite(order[t]), order[t],
                           np.where(over & np.isnan(pending), dense[t], pending))
        do = np.isfinite(pending) & np.isfinite(o[t])
        new = pending[do] * value / o[t, do]
        cash -= float(((new - shares[do]) * o[t, do]).sum())
        shares[do] = new
        pending[do] = np.nan
        now = cash + np.nansum(shares * c[t])
        out[t] = now / last - 1.0
        last, val = now, np.where(np.isfinite(c[t]), c[t], val)
        if retrim_k:
            with np.errstate(invalid="ignore", divide="ignore"):
                weight = shares * c[t] / now
            over = (dense[t] > 0) & (weight >= retrim_k * dense[t])
    return pd.Series(out, index=close.index)


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
    adaptive_hedge/reconcile.py's booking. The costs of trading the gap are not charged.

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


def _gaps(pf, window, init_cash: float):
    """The refused or part-filled orders of one rebuilt period and each one's daily gap P&L.

    An order's gap is the position it left minus the shares it asked for (vectorbt's log:
    ``req_size × value / val_price``), held from its due bar's OPEN until the name's next order
    (:func:`gap_pnl`). Returns ``(rows, gap)``: one dict per order (date, name, kind, signed
    shares wanted / got, the gap in shares, bars stuck, the gap's P&L as a share of the book and
    the due bar's part of it) and the summed daily gap return. Removing ``gap`` from the book's
    returns estimates the book had every such order filled at its due open, the cash it would
    have freed (or used) idle.
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


def invariants(weights: pd.DataFrame, *, slots: int, gross: float) -> dict:
    """The book invariants on one weight frame: names a side, gross, one weight per hold, and
    no name on both sides of the book on one bar."""
    w = weights.fillna(0.0)
    side = np.sign(w)
    hold = side.ne(side.shift(fill_value=0.0)).cumsum()
    multi = 0
    for c in w.columns[(w != 0).any()]:
        held = side[c] != 0
        multi += int((w[c][held].groupby(hold[c][held]).nunique() > 1).sum())
    out = {"max longs": int((w > 0).sum(axis=1).max()),
           "max shorts": int((w < 0).sum(axis=1).max()),
           "max gross": float(w.abs().sum(axis=1).max()),
           "holds with >1 weight": multi}
    out["ok"] = bool(out["max longs"] <= slots and out["max shorts"] <= slots
                     and out["max gross"] <= gross + 1e-9 and multi == 0)
    return out


def audit(labels=("C_U0", "C_rev"), tests=OPEN, top: int = 5) -> pd.DataFrame:
    """Mechanics only, no returns: does the engine hold the book each label asks for?

    Every period is rebuilt with :func:`rebuild` and asserted equal to ``lab.book`` (day 0
    counted from the initial cash as ``lab.book`` does). From vectorbt's log, per label and
    test: orders asked (a non-zero change), refused, partly filled (below 99.5% of the shares
    asked), dropped at a NaN price, the share refused or part-filled (the §0.5 rule's 1%),
    split by kind (long buy, long sale, new short, cover), off-target name-days
    (:func:`off_target`, both ways), the mean gross never held and the stuck orders' summed gap
    P&L from their due open (:func:`_gaps`, a share of the book); :func:`invariants` and the
    flips (short to long, long to short, in one order) on the scored window's weights; and the
    engine against :func:`preview`: daily correlation, annualized tracking error, the yearly
    drag (mean daily engine − preview × 252: costs, borrow, failed orders) and the ``top``
    largest daily gaps with the orders placed that day.
    """
    from trading_simple import backtest as bt
    lab = _lab()
    tests = _open_only(tests)
    settings = lab._settings()
    init = settings.backtest.init_cash
    rows = {}
    for label in labels:
        strat = lab.load_strategy(NAME, LABELS[label])
        for t in tests:
            acc = {"orders asked": 0, "refused": 0, "partly filled": 0, "no price": 0,
                   "off-target days": 0, "stuck orders": 0, "gap P&L": 0.0,
                   "missing gross x bars": 0.0, "bars": 0, "short kinds": {},
                   "invariants ok": True, "max gross": 0.0, "max longs": 0, "max shorts": 0,
                   "holds with >1 weight": 0, "flips short->long": 0, "flips long->short": 0}
            eng, pre, days = [], [], []
            for te0, _ in lab.periods(t):
                p = lab.panel(t, te0, lab.strategy_features(strat))
                sig = strat.signals(p, **strat.params)
                window, s = bt._score_window(p, sig, te0)
                w = s["weights"]
                inv = invariants(w, slots=strat.params["slots"], gross=strat.params["gross"])
                acc["invariants ok"] &= inv["ok"]
                for k in ("max gross", "max longs", "max shorts"):
                    acc[k] = max(acc[k], inv[k])
                acc["holds with >1 weight"] += inv["holds with >1 weight"]
                was = np.sign(w.shift(1, fill_value=0.0))
                acc["flips short->long"] += int(((was < 0) & (w > 0)).to_numpy().sum())
                acc["flips long->short"] += int(((was > 0) & (w < 0)).to_numpy().sum())
                pf = rebuild(strat, window, s, settings)
                ref = lab.book(strat, p, settings, te0, signals=sig)["returns"]
                value = bt.equity_curve(pf)["value"]
                mine = value / value.shift(1).fillna(init) - 1.0
                assert float((mine - ref).abs().max()) < 1e-12, (label, t, te0)
                eng.append(mine)
                pre.append(preview(window, w, s.get("retrim_k")).reindex(mine.index))
                changed = bt.rotation_targets(w).notna().sum(axis=1).reindex(mine.index)
                days.append(changed)
                lg = pf.logs.values
                lg = lg[np.isfinite(lg["req_size"])]
                with np.errstate(invalid="ignore", divide="ignore"):
                    want = lg["req_size"] * lg["value"] / lg["val_price"] - lg["position"]
                got = np.where(lg["res_status"] == 0, lg["new_position"] - lg["position"], 0.0)
                nonzero = np.abs(want) > 1e-9 * np.maximum(np.abs(lg["position"]), 1.0)
                refused = (lg["res_status"] == 2) & nonzero                  # 2 = Rejected
                noprice = ((lg["res_status"] == 1) & np.isin(lg["res_status_info"], (1, 2))
                           & ((lg["req_size"] != 0) | (lg["position"] != 0)))
                partial = (lg["res_status"] == 0) & nonzero & (np.abs(got) < 0.995 * np.abs(want))
                short = refused | partial
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
                acc["refused"] += int(refused.sum())
                acc["partly filled"] += int(partial.sum())
                acc["no price"] += int(noprice.sum())
                acc["off-target days"] += int(off_target(pf.assets(), w,
                                                         window.close).to_numpy().sum())
                stuck_rows, gap_ret = _gaps(pf, window, init)
                acc["stuck orders"] += len(stuck_rows)
                acc["gap P&L"] += float(gap_ret.sum())
                acc["bars"] += n
            e, q, d = pd.concat(eng), pd.concat(pre), pd.concat(days)
            diff = e - q
            acc["engine~preview corr"] = round(float(e.corr(q)), 5)
            acc["engine-preview TE/yr"] = round(float(diff.std() * np.sqrt(252)), 5)
            acc["engine-preview drag/yr"] = round(float(diff.mean() * 252), 5)
            big = diff.abs().nlargest(top).index
            acc["largest gaps"] = [(str(i.date()), round(float(diff[i]), 5), int(d[i]))
                                   for i in big]
            acc["mean missing gross"] = acc.pop("missing gross x bars") / max(acc["bars"], 1)
            acc["refused or partial share"] = (acc["refused"] + acc["partly filled"]) / max(
                acc["orders asked"], 1)
            rows[(label, t)] = acc
            print(label, t, json.dumps(acc, default=str), flush=True)
    out = pd.DataFrame(rows).T
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.drop(columns=["largest gaps", "short kinds"]).to_string())
    rule = out["refused or partial share"].astype(float)
    print(f"§0.5 rule ({', '.join(tests)}): max refused-or-partial share {rule.max():.4%} "
          f"-> gross {'0.85' if rule.max() > 0.01 else 'stays 0.90'}")
    return out


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
            "P(Δ>0)": pr["P(delta > 0)"], "better": f"{sum(d > 0 for d in better)}/{len(better)}"}


def delta(label: str, base: str, tests=OPEN, quiet: bool = False) -> dict:
    """Paired Δ alpha of ``label`` minus ``base`` per test and on the stitched TRAIN (TB + T3)
    record (the same days and bootstrap blocks; the benchmark is ``label``'s)."""
    tests = _open_only(tests)
    a, b = _load(label), _load(base)
    out = {t: _pair(a[t], b[t]) for t in tests if t in a and t in b}
    train = [t for t in ("TB", "T3") if t in a and t in b]
    if train:
        out["TRAIN"] = _pair([r for t in train for r in a[t]], [r for t in train for r in b[t]])
    if not quiet:
        print(f"{label} − {base}:")
        for k, v in out.items():
            print(f"  {k:6s} alpha {v['alpha']:+.4f}  base {v['base alpha']:+.4f}  "
                  f"Δ {v['Δ']:+.4f} (P {v['P(Δ>0)']:.3f}, better {v['better']})")
    return out


def stuck(labels=("C_U0",), tests=OPEN, base: str | None = None) -> dict:
    """Refused or part-filled orders per label and test (:func:`_gaps`), and the alpha with
    their gap P&L removed: each such order filled at its due bar's open, the cash it would have
    freed (or used) idle — adaptive_hedge's repair estimate, applied to refused entries too.

    Every rebuilt period is asserted equal to the saved returns. TRAIN (TB + T3) is stitched
    when both are in ``tests``. With ``base``, the paired Δ of each label against it, as run
    and with both sides' gaps removed. A label may be ``<strategy>/<label>`` (rebuilt read-only).
    The saved lab results stay the record.
    """
    from trading_simple import backtest as bt
    lab = _lab()
    tests = _open_only(tests)
    settings = lab._settings()
    init = settings.backtest.init_cash
    runs, out = {}, {}
    for label in dict.fromkeys([*labels, *([base] if base else [])]):
        strat, res = _strategy(label), _load(label)
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
                orders += [{**r, "period": te0} for r in rows]
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


def _beta_frame(p) -> pd.DataFrame:
    """``lab._ex_ante_beta``'s beta: trailing 126-bar beta of each name to the panel's equal-weight
    return, known the bar before (shifted one bar)."""
    ret = p.close.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    return ret.rolling(126, min_periods=63).cov(mkt).div(
        mkt.rolling(126, min_periods=63).var(), axis=0).shift(1)


def betas(labels=("C_U0", "C_rev"), tests=OPEN) -> dict:
    """Per period: the mean ex-ante beta of each leg (Σ target weight × beta over the long and
    the short names, as ``lab._ex_ante_beta`` measures the book) and of the book, the realized
    beta (the period's daily book returns on the equal-weight buy-and-hold, ``lab.alpha_of``)
    and the period's alpha. Positions are recomputed; returns are the saved ones."""
    lab = _lab()
    tests = _open_only(tests)
    out = {}
    for label in labels:
        res, strat = _load(label), _strategy(label)
        for t in tests:
            if t not in res:
                continue
            rows = []
            for row, (te0, te1) in zip(res[t], lab.periods(t), strict=True):
                p = lab.panel(t, te0, lab.strategy_features(strat))
                w = strat.signals(p, **strat.params)["weights"].fillna(0.0)
                b = _beta_frame(p).reindex(columns=w.columns)
                idx = p.close.index
                win = (idx >= pd.Timestamp(te0, tz="UTC")) & (idx <= pd.Timestamp(te1, tz="UTC"))
                lo = (w.clip(lower=0) * b).sum(axis=1)[win]
                sh = (w.clip(upper=0) * b).sum(axis=1)[win]
                r, bench = row["returns"], row["bench"]
                _, _, beta, alpha = lab.alpha_of(r.to_numpy(), bench.to_numpy(),
                                                 lab.years_of(r.index))
                rows.append({"period": te0, "ex-ante long": round(float(lo.mean()), 3),
                             "ex-ante short": round(float(sh.mean()), 3),
                             "ex-ante book": round(float((lo + sh).mean()), 3),
                             "realized beta": round(float(beta[0]), 3),
                             "alpha": round(float(alpha[0]), 4)})
            tab = pd.DataFrame(rows).set_index("period")
            within = int((tab["realized beta"].abs() <= 0.15).sum())
            out[(label, t)] = {"table": tab, "within ±0.15": f"{within}/{len(tab)}"}
            print(f"{label} {t}: realized |beta| <= 0.15 in {within}/{len(tab)} periods")
            with pd.option_context("display.width", 200):
                print(tab.to_string())
    return out


def _stats(x: pd.Series, mkt: pd.Series) -> dict:
    """A daily series' annualized arithmetic contribution, its beta to ``mkt`` and its alpha
    (contribution − beta × the market's), adaptive_hedge/legs.py's ``_stats``."""
    x, mkt = x.align(mkt, join="left", fill_value=0.0)
    var = float(mkt.var())
    beta = float(x.cov(mkt) / var) if var > 0 else np.nan
    return {"contrib": float(x.mean() * 252), "beta": beta,
            "alpha": float((x.mean() - beta * mkt.mean()) * 252)}


def legs(label: str, tests=OPEN) -> dict:
    """Leg attribution of a saved label (adaptive_hedge/legs.py's pattern; re-runs its book with
    ``lab.book``, no new variant): each leg's daily contribution — the name's weight in the book
    at the previous close (asset value / book value) times its return — as an annualized
    arithmetic contribution, its beta to the equal-weight market and its alpha (contribution −
    beta × market), per test, per period and per calendar year. Long + short + costs and cash ≈
    the book. ``<strategy>/<label>`` re-runs another strategy's book (read-only; saved here)."""
    lab = _lab()
    tests = _open_only(tests)
    strat = _strategy(label)
    settings = lab._settings()
    out, daily = {}, {}
    for t in tests:
        parts, per = [], []
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            r = lab.book(strat, p, settings, te0, signals=strat.signals(p, **strat.params))
            pf, w = r["pf"], r["window"]
            held = pf.asset_value(group_by=False).div(pf.value(), axis=0).shift(1).fillna(0.0)
            rets = w.close.pct_change(fill_method=None).reindex(held.index).fillna(0.0)
            contrib = held * rets[held.columns]
            d = pd.DataFrame({"long": contrib.where(held > 0, 0.0).sum(axis=1),
                              "short": contrib.where(held < 0, 0.0).sum(axis=1),
                              "book": r["returns"], "market": r["bench"],
                              "short gross": -held.clip(upper=0).sum(axis=1),
                              "long gross": held.clip(lower=0).sum(axis=1)})
            parts.append(d)
            per.append({"period": te0, **{f"{leg} alpha": round(_stats(d[leg], d["market"])
                                                                ["alpha"], 4)
                                          for leg in ("long", "short", "book")}})
        d = pd.concat(parts)
        daily[t] = d
        row = {leg: _stats(d[leg], d["market"]) for leg in ("long", "short", "book")}
        row["costs + cash"] = float((d["book"] - d["long"] - d["short"]).mean() * 252)
        row["gross"] = {"long": float(d["long gross"].mean()),
                        "short": float(d["short gross"].mean())}
        row["periods"] = per
        years = {}
        for y, g in d.groupby(d.index.year):
            years[int(y)] = {"long": round(float(g["long"].sum()), 3),
                             "short": round(float(g["short"].sum()), 3),
                             "book": round(float((1 + g["book"]).prod() - 1), 3),
                             "market": round(float((1 + g["market"]).prod() - 1), 3)}
        row["years (sum of daily contributions)"] = years
        out[t] = row
    if "TB" in daily and "T3" in daily:
        d = pd.concat([daily["TB"], daily["T3"]])
        out["TRAIN"] = {leg: _stats(d[leg], d["market"]) for leg in ("long", "short", "book")}
    print(json.dumps(out, indent=1, default=float))
    pd.to_pickle(daily, lab.RESULTS / NAME / f"legs-{label.replace('/', '__')}.pkl")
    return out


def overlap(tests=OPEN) -> dict:
    """Positions only (no returns): per test, the mean number of C_U0's held longs that C_si's
    book also holds long and of its shorts that C_mom's book also holds short (and the same for
    C_rev's legs), and how many names a bar sit in the MOM short entry zone while held long (the
    short mask binding)."""
    lab = _lab()
    tests = _open_only(tests)
    strats = {k: _strategy(k) for k in ("C_U0", "C_rev", f"{CORE}/C_si", f"{CORE}/C_mom")}
    core = strats["C_U0"].signals.__globals__["core"].__dict__     # the pinned core
    out = {}
    for t in tests:
        acc: dict = {}
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strats["C_U0"]))
            idx = p.close.index
            win = (idx >= pd.Timestamp(te0, tz="UTC")) & (idx <= pd.Timestamp(te1, tz="UTC"))
            w = {k: s.signals(p, **s.params)["weights"][win] for k, s in strats.items()}
            si, mom = w[f"{CORE}/C_si"], w[f"{CORE}/C_mom"]
            pairs = {"C_U0 longs also in C_si longs": (w["C_U0"] > 0) & (si > 0),
                     "C_U0 shorts also in C_mom shorts": (w["C_U0"] < 0) & (mom < 0),
                     "C_rev longs also in C_mom longs": (w["C_rev"] > 0) & (mom > 0),
                     "C_rev shorts also in C_si shorts": (w["C_rev"] < 0) & (si < 0)}
            for k, v in pairs.items():
                acc.setdefault(k, []).append(float(v.sum(axis=1).mean()))
            prm = strats["C_U0"].params
            r = core["ranks"](p, {"si", "mom"}, dtc_window=prm["dtc_window"],
                              neutral_groups=prm["neutral_groups"], min_price=prm["min_price"])
            ok = r["base"]["eligible"]
            wt = {"si": 1.0, "mom": 1.0, "prof": 1.0}
            mom_score = core["side_score"](r["ranks"], ["mom"], wt, ok, prm["score_smooth"])
            held_long = (w["C_U0"] > 0).shift(-1, fill_value=False)      # held at the decision
            zone = mom_score[win] <= 1 - prm["q_in"] + 1e-9
            acc.setdefault("C_U0 longs in the MOM short zone (per bar)", []).append(
                float((held_long & zone).sum(axis=1).mean()))
        out[t] = {k: round(float(np.mean(v)), 2) for k, v in acc.items()}
        print(t, json.dumps(out[t]), flush=True)
    return out


def noise_sd(label: str = "C_U0", k: int = 5) -> dict:
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


def bars(label: str = "C_U0") -> dict:
    """The pre-registered design bars (RESEARCH_LOG §0.6) for ``label``.

    (2) Δ(label − C_rev) > 0 on TRAIN, T1 and T2, P(Δ > 0) >= 0.80 on T1 and T2; (3) realized
    per-period beta within ±0.15 in >= 8 of the 10 T1 periods (reported); (4) alpha >= max(C_si,
    C_mom) on TRAIN and T2; (5) Δ(label − A_U0) >= 0 on TRAIN and T2, >= −1.0 pt on T1 (context).
    Bar (1), the gates, is ``lab.py verdict``.
    """
    lab = _lab()

    def level(lbl):
        r = _load(lbl)
        return {"TRAIN": lab.stitched([(x["returns"], x["bench"]) for t in ("TB", "T3")
                                       for x in r[t]])["alpha"],
                "T1": lab._stitch(r["T1"])["alpha"], "T2": lab._stitch(r["T2"])["alpha"]}

    lv = {k: level(k) for k in dict.fromkeys(
        (label, "C_rev", f"{CORE}/C_si", f"{CORE}/C_mom", f"{CORE}/A_U0"))}
    rev = delta(label, "C_rev", quiet=True)
    b2 = bool(all(rev[t]["Δ"] > 0 for t in ("TRAIN", "T1", "T2"))
              and rev["T1"]["P(Δ>0)"] >= 0.80 and rev["T2"]["P(Δ>0)"] >= 0.80)
    res = _load(label)
    per = [lab.alpha_of(r["returns"].to_numpy(), r["bench"].to_numpy(),
                        lab.years_of(r["returns"].index))[2][0] for r in res["T1"]]
    within = int(sum(abs(b) <= 0.15 for b in per))
    best = {t: max(lv[f"{CORE}/C_si"][t], lv[f"{CORE}/C_mom"][t]) for t in ("TRAIN", "T1", "T2")}
    b4 = bool(lv[label]["TRAIN"] >= best["TRAIN"] and lv[label]["T2"] >= best["T2"])
    integ = delta(label, f"{CORE}/A_U0", quiet=True)
    b5 = bool(integ["TRAIN"]["Δ"] >= 0 and integ["T2"]["Δ"] >= 0 and integ["T1"]["Δ"] >= -0.01)
    out = {"levels": lv,
           "bar2 Δ vs C_rev": {t: (round(rev[t]["Δ"], 4), round(rev[t]["P(Δ>0)"], 3),
                                   rev[t]["better"]) for t in ("TRAIN", "TB", "T3", "T1", "T2")},
           "bar2": b2,
           "bar3 T1 realized betas": [round(float(b), 3) for b in per],
           "bar3": f"{within}/{len(per)} within ±0.15 -> {'pass' if within >= 8 else 'fail'}",
           "bar4 best single sleeve": best, "bar4": b4,
           "bar5 Δ vs A_U0": {t: (round(integ[t]["Δ"], 4), round(integ[t]["P(Δ>0)"], 3),
                                  integ[t]["better"]) for t in ("TRAIN", "TB", "T3", "T1", "T2")},
           "bar5": b5}
    print(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "audit":
        audit(tuple(a[1].split(",")) if len(a) > 1 else ("C_U0", "C_rev"),
              tuple(a[2].split(",")) if len(a) > 2 else OPEN)
    elif a and a[0] == "stuck":
        stuck(tuple(a[1].split(",")) if len(a) > 1 else ("C_U0",),
              tuple(a[2].split(",")) if len(a) > 2 else OPEN, a[3] if len(a) > 3 else None)
    elif a and a[0] == "delta":
        delta(a[1], a[2], tuple(a[3].split(",")) if len(a) > 3 else OPEN)
    elif a and a[0] == "betas":
        betas(tuple(a[1].split(",")) if len(a) > 1 else ("C_U0", "C_rev"),
              tuple(a[2].split(",")) if len(a) > 2 else OPEN)
    elif a and a[0] == "legs":
        legs(a[1], tuple(a[2].split(",")) if len(a) > 2 else OPEN)
    elif a and a[0] == "overlap":
        overlap(tuple(a[1].split(",")) if len(a) > 1 else OPEN)
    elif a and a[0] == "noise_sd":
        print(json.dumps(noise_sd(a[1] if len(a) > 1 else "C_U0"), indent=1))
    elif a and a[0] == "bars":
        bars(a[1] if len(a) > 1 else "C_U0")
    else:
        raise SystemExit(__doc__)
