"""xs_three_book research helpers (loaded by path; ``strategy.py`` never imports this).

    uv run python strategies/xs_three_book/research.py audit '<overrides>' [TB,T3,T1,T2]
    uv run python strategies/xs_three_book/research.py reproduce [T1:2024-01-01,TB:2020-01-01]
    uv run python strategies/xs_three_book/research.py diagnose <label> [TB,T3,T1,T2]
    uv run python strategies/xs_three_book/research.py preview
    uv run python strategies/xs_three_book/research.py table <label,...> [--base <label>]
    uv run python strategies/xs_three_book/research.py bars

``audit``: does the lab's runner (``lab.run_portfolio``) hold the account the strategy asks for?
Mechanics and book invariants only, NO returns (the pre-registered execution rule reads it).
``reproduce``: each sleeve run alone through this strategy at its own gross must rebuild the
component's saved lab book (a plumbing check; prints the largest daily difference only).
``diagnose``: a saved label rebuilt period by period with vectorbt's order logs
(adaptive_hedge/reconcile.py's pattern): refused and part-filled orders by class, stuck
positions and their P&L, orders dropped at a NaN price, missing exposure, the sleeves' target
contributions, the order sizes at the declared $100k account and the hand re-pricing at the $1
minimum. ``preview``: the return-sum of the components' saved books (the design's preview).
``table`` / ``bars``: saved labels side by side with paired deltas; the pre-registered bars.
T4 and TL are locked: every function here refuses them.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys

import numpy as np
import pandas as pd

REPO = "/Users/janmarti/Documents/code/stock-trading-simple"
KEY = "xs_three_book"
OPEN4 = ("TB", "T3", "T1", "T2")
LOCKED = {"T4", "TL"}
#: The components' saved lab labels (read-only) and their nominal gross.
COMPONENT_LABELS = {"si": ("informed_flow_ls_p1", "v6_si", 0.98),
                    "mom": ("resid_momentum", "U0v2", 0.90),
                    "ah": ("adaptive_hedge", "TUNEDv2", 0.98),
                    "ah_u0": ("adaptive_hedge", "U0v2", 0.98)}
ACCOUNT = 100_000.0          # the declared account (config.yaml)
CLASSES = ("long cut", "short cover", "new short", "long buy", "none")
TINY = 1e-9
_LAB = {}


def _lab():
    """The proposals lab (``strategies/proposals_lab/lab.py``), loaded once by path."""
    if "lab" not in _LAB:
        spec = importlib.util.spec_from_file_location(
            "lab", f"{REPO}/strategies/proposals_lab/lab.py")
        lab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lab)
        _LAB["lab"] = lab
    return _LAB["lab"]


def _open_only(tests) -> list[str]:
    """The tests as a list; refuses the locked T4 and TL."""
    tests = list(tests)
    if LOCKED & set(tests):
        raise SystemExit(f"{sorted(LOCKED & set(tests))}: locked tests are never read here")
    return tests


def _panel(lab, test, te0, strat, hedge=True):
    """A lab period panel with the strategy's features (SPY merged when ``hedge``)."""
    p = lab.panel(test, te0, lab.strategy_features(strat))
    return lab.with_spy(p) if hedge else p


def _compose(strat, p):
    """The strategy's ``compose()`` on a panel, with the loaded params."""
    import inspect
    mod = strat.signals.__globals__
    keys = inspect.signature(mod["compose"]).parameters
    return mod["compose"](p, **{k: v for k, v in strat.params.items() if k in keys})


def _signals(strat, parts) -> dict:
    """``signals()``'s output built from a ``compose()`` result (the same frames)."""
    out = {"weights": parts["weights"], "borrow": parts["borrow"]}
    if strat.params.get("retrim_k", 0) > 0:
        out["retrim_k"] = strat.params["retrim_k"]
    return out


def rebuild(strat, window, sig: dict, settings, log: bool = True):
    """One period's book rebuilt exactly as the lab builds it (``lab.run_portfolio``: its call
    order and borrow fees, the retrim loop), with vectorbt's order log on (copied from
    resid_momentum/research.py)."""
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
    return build(targets) if k is None else bt._retrim_loop(build, targets, weights,
                                                             k * weights)


def _returns(pf, settings) -> pd.Series:
    """A book's daily returns, day 0 counted from the initial cash (as ``lab.book``)."""
    from trading_simple import backtest as bt
    value = bt.equity_curve(pf)["value"]
    return value / value.shift(1).fillna(settings.backtest.init_cash) - 1.0


def _classes(req: pd.DataFrame) -> np.ndarray:
    """Each request's class against the in-force target before the bar (``lab._call_order``'s
    rules): 0 long cut, 1 short cover, 2 new short, 3 long buy, 4 none."""
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


def orders_of(pf, window) -> pd.DataFrame:
    """Every request of a logged book: class, shares asked (``need``) and got, status, free
    cash, and the request's dollar size at the declared :data:`ACCOUNT` (from the marked book)."""
    logs = pf.logs.records_readable
    lg = logs[logs["Request Size"].notna()].copy()
    index, cols = window.close.index, list(pf.wrapper.columns)
    req = lg.pivot_table(index="Timestamp", columns="Column", values="Request Size",
                         aggfunc="last").reindex(index=index, columns=cols)
    cat = pd.DataFrame(_classes(req), index=index, columns=cols)
    row_of = {d: k for k, d in enumerate(index)}
    col_of = {c: k for k, c in enumerate(cols)}
    lg["class"] = [CLASSES[cat.iat[row_of[d], col_of[c]]]
                   for d, c in zip(lg["Timestamp"], lg["Column"], strict=True)]
    lg["need"] = lg["Request Size"] * lg["Value"] / lg["Val Price"] - lg["Position"]
    side = np.where(lg["Result Side"] == "Buy", 1.0,
                    np.where(lg["Result Side"] == "Sell", -1.0, 0.0))
    lg["got"] = side * lg["Result Size"].fillna(0.0)
    lg["dollars"] = (lg["need"] * lg["Val Price"]).abs() / lg["Value"] * ACCOUNT
    return lg, req


def reconcile_period(pf, window, saved_returns=None) -> dict:
    """Targets against realized positions for one logged book (adaptive_hedge/reconcile.py):
    refused and part-filled orders, NaN-price drops, stuck positions (held while the in-force
    target is 0 or on the other side) with their P&L from the due bar's open, targets never
    held, and the time-averaged exposure that refused or part-filled entries leave out."""
    lg, req = orders_of(pf, window)
    index, cols = window.close.index, list(pf.wrapper.columns)
    tgt = req.ffill().fillna(0.0)
    held = pf.assets().reindex(index=index, columns=cols).fillna(0.0)
    value = pf.value().reindex(index)
    close = window.close.reindex(columns=cols).ffill()
    open_ = window.open_.reindex(columns=cols)
    stuck_mask = (((tgt.abs() < 1e-12) & (held.abs() > TINY)) | ((tgt > 0) & (held < -TINY))
                  | ((tgt < 0) & (held > TINY)))
    pnl = pd.Series(0.0, index=index)
    stuck = []
    for c in stuck_mask.columns[stuck_mask.any()]:
        for s0, e in _runs(stuck_mask[c].to_numpy()):
            total = 0.0
            for k in range(max(s0, 1), e + 1):
                if k == s0:
                    d = held[c].iat[k] * (close[c].iat[k] - open_[c].iat[k]) / value.iat[k - 1]
                else:
                    d = held[c].iat[k - 1] * (close[c].iat[k] - close[c].iat[k - 1]) / \
                        value.iat[k - 1]
                d = float(d) if np.isfinite(d) else 0.0
                pnl.iat[k] += d
                total += d
            due = lg[(lg["Column"] == c) & (lg["Timestamp"] == index[s0])]
            stuck.append({"name": c, "side": "short" if held[c].iat[s0] < 0 else "long",
                          "from": index[s0], "bars": e - s0 + 1,
                          "status": (f"{due['Result Status'].iat[0]}/"
                                     f"{due['Result Status Info'].iat[0]}" if len(due)
                                     else "no order"),
                          "free cash": float(due["Free Cash"].iat[0]) if len(due) else np.nan,
                          "weight": float(held[c].iat[s0] * close[c].iat[s0] / value.iat[s0]),
                          "pnl": total})
    real = lg[(lg["class"] != "none") & lg["need"].notna() & (lg["need"].abs() > 1e-6)]
    refused = real[real["Result Status"] != "Filled"]
    ratio = real["got"] / real["need"]
    partial99 = real[(real["Result Status"] == "Filled") & (ratio < 0.99)]
    partial995 = real[(real["Result Status"] == "Filled") & (ratio < 0.995)]
    nan_price = lg[(lg["Result Status Info"] == "PriceNaN") & (lg["Request Size"].abs() > TINY)]
    dead = (tgt.abs() > 1e-12) & (held.abs() < TINY) & window.close.reindex(
        columns=cols).notna()
    miss = pd.DataFrame(0.0, index=index, columns=cols)
    row_of = {d: k for k, d in enumerate(index)}
    col_of = {c: k for k, c in enumerate(cols)}
    short_of = real[real["class"].isin(["long buy", "new short"]) & (ratio < 0.99)]
    for _, z in short_of.iterrows():
        c, k = z["Column"], row_of[z["Timestamp"]]
        nxt = req[c].iloc[k + 1:].first_valid_index()
        end = row_of[nxt] if nxt is not None else len(index)
        miss.iloc[k:end, col_of[c]] += (z["need"] - z["got"]) * close[c].iloc[k:end].to_numpy()
    miss = miss.div(value, axis=0)
    out = {"orders": int(len(pf.orders.records)), "refused": refused, "partial99": partial99,
           "partial995": partial995, "nan_price": nan_price, "stuck": stuck, "stuck_pnl": pnl,
           "dead": {c: int(dead[c].sum()) for c in dead.columns[dead.any()]},
           "off_days": int(((np.sign(held) != np.sign(tgt)) & (tgt.abs() > 1e-12)
                            & window.close.reindex(columns=cols).notna()).to_numpy().sum()),
           "missing": {"long": float(miss.clip(lower=0).sum(axis=1).mean()),
                       "short": float(-miss.clip(upper=0).sum(axis=1).mean())},
           "log": lg}
    if saved_returns is not None:
        out["rebuild"] = float(np.max(np.abs(_returns(pf, _lab()._settings()).to_numpy()
                                              - saved_returns.to_numpy())))
    return out


def _fmt(frame: pd.DataFrame, what: str) -> list[str]:
    """One printed line per order record: class, name, day, status, fill ratio, free cash."""
    out = []
    for _, z in frame.iterrows():
        r = z["got"] / z["need"] if abs(z["need"]) > TINY else float("nan")
        out.append(f"    {what:8s} {z['class']:11s} {z['Column']:6s} {z['Timestamp'].date()} "
                   f"{z['Result Status']}/{z['Result Status Info']} filled {r:.3f} of "
                   f"{z['need']:+.3f} sh (target {z['Request Size']:+.4f}), free cash "
                   f"{z['Free Cash']:.2f}")
    return out


def audit(overrides: dict | None = None, tests=OPEN4, strategy: str = KEY,
          hedge: bool = True) -> pd.DataFrame:
    """Mechanics and book invariants of a config on the lab's runner — NO returns.

    Per test: orders asked, rejected, part-filled (< 99.5% and < 99% of the shares asked),
    orders dropped at a NaN price, stuck runs, position-days off target, the mean gross that
    refused or part-filled entries leave unheld; and the invariants: the largest nominal gross
    (whole panel), the largest |SPY| weight, bars inside the window on which the coverage switch
    is off, the largest realized gross (Σ |asset value| / book value) and the smallest free
    cash as a share of the book. Every rebuild is checked equal to ``lab.book``.
    """
    from trading_simple import backtest as bt
    lab = _lab()
    tests = _open_only(tests)
    strat = lab.load_strategy(strategy, overrides or {})
    settings = lab._settings()
    rows = {}
    for t in tests:
        acc = {"orders": 0, "rejected": 0, "partial<99.5%": 0, "partial<99%": 0,
               "nan-price drops": 0, "stuck runs": 0, "off-target days": 0,
               "missing gross (mean)": 0.0, "max nominal gross": 0.0, "max |SPY|": 0.0,
               "switch-off bars": 0, "max realized gross": 0.0, "min free cash": 1.0,
               "max rebuild diff": 0.0}
        bars = 0
        for te0, _ in lab.periods(t):
            p = _panel(lab, t, te0, strat, hedge)
            parts = _compose(strat, p)
            sig = _signals(strat, parts)
            window, s = bt._score_window(p, sig, te0)
            pf = rebuild(strat, window, s, settings)
            ref = lab.book(strat, p, settings, te0, signals=sig)["returns"]
            diff = float((_returns(pf, settings) - ref).abs().max())
            r = reconcile_period(pf, window)
            n = len(window.close)
            inside = p.close.index >= window.close.index[0]
            w = sig["weights"]
            value = pf.value()
            gross = pf.asset_value(group_by=False).abs().sum(axis=1) / value
            free = pf.cash(free=True) / value
            acc["orders"] += r["orders"]
            acc["rejected"] += len(r["refused"])
            acc["partial<99.5%"] += len(r["partial995"])
            acc["partial<99%"] += len(r["partial99"])
            acc["nan-price drops"] += len(r["nan_price"])
            acc["stuck runs"] += len(r["stuck"])
            acc["off-target days"] += r["off_days"]
            acc["missing gross (mean)"] += (r["missing"]["long"] + r["missing"]["short"]) * n
            acc["max nominal gross"] = max(acc["max nominal gross"],
                                           float(w.abs().sum(axis=1).max()))
            if "SPY" in w.columns:
                acc["max |SPY|"] = max(acc["max |SPY|"], float(w["SPY"].abs().max()))
            acc["switch-off bars"] += int((~parts["on"][inside]).sum())
            acc["max realized gross"] = max(acc["max realized gross"], float(gross.max()))
            acc["min free cash"] = min(acc["min free cash"], float(free.min()))
            acc["max rebuild diff"] = max(acc["max rebuild diff"], diff)
            bars += n
            lines = [f"{t} {te0}: orders {r['orders']}, rejected {len(r['refused'])}, partial "
                     f"(<99.5%) {len(r['partial995'])}, NaN-price {len(r['nan_price'])}, "
                     f"stuck {len(r['stuck'])}, off-target days {r['off_days']}, max realized "
                     f"gross {gross.max():.3f}, min free cash {free.min():.3f}, rebuild "
                     f"{diff:.1e}"]
            lines += _fmt(r["refused"], "REFUSED") + _fmt(r["partial995"], "partial")
            for s0 in r["stuck"]:
                lines.append(f"    STUCK {s0['side']} {s0['name']} from {s0['from'].date()}: "
                             f"{s0['bars']} bars ({s0['status']}, free cash "
                             f"{s0['free cash']:.2f})")
            for _, z in r["nan_price"].iterrows():
                lines.append(f"    NaN-price drop {z['Column']} {z['Timestamp'].date()} "
                             f"(target {z['Request Size']:+.4f})")
            print("\n".join(lines), flush=True)
        acc["missing gross (mean)"] /= max(bars, 1)
        rows[t] = acc
    out = pd.DataFrame(rows).T
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.to_string())
    return out


def reproduce(where=(("T1", "2024-01-01"), ("TB", "2020-01-01"), ("T2", "2026-07-01"))):
    """Each sleeve alone at its own gross (budget = the component's nominal gross) through this
    strategy, against the component's saved lab book: the largest daily |difference| of the
    book's returns (not the returns themselves). Exact except where this account defers a
    target change to the next priced open and the component did not."""
    lab = _lab()
    settings = lab._settings()
    for key in ("si", "mom", "ah"):
        comp, label, g = COMPONENT_LABELS[key]
        saved = lab.load_results(comp, label)
        strat = lab.load_strategy(KEY, {"sleeves": key, "budgets": str(g), "gross": g})
        for t, te0 in where:
            i = [x for x, _ in lab.periods(t)].index(te0)
            p = _panel(lab, t, te0, strat)
            res = lab.book(strat, p, settings, te0)
            ref = saved[t][i]["returns"]
            d = float((res["returns"] - ref.reindex(res["returns"].index)).abs().max())
            print(f"reproduce {key} ({comp}/{label}) {t} {te0}: max |daily diff| {d:.2e}",
                  flush=True)


def _sleeve_legs(parts, window, held_weights, book_returns) -> pd.DataFrame:
    """Each sleeve's daily target contribution: its weight in force at the previous close ×
    the close-to-close return (drift-free, arithmetic); plus the hedge, the book and the
    realized long and short legs (the book's own positions, adaptive_hedge/legs.py)."""
    idx = window.close.index
    rets = window.close.pct_change(fill_method=None).fillna(0.0)
    out = {}
    for k, w in parts["sleeves"].items():
        prev = w.reindex(idx).shift(1).fillna(0.0)
        out[k] = (prev * rets[prev.columns]).sum(axis=1)
    if "SPY" in rets.columns:
        out["hedge"] = parts["hedge"].reindex(idx).shift(1).fillna(0.0) * rets["SPY"]
    contrib = held_weights * rets[held_weights.columns]
    out["long"] = contrib.where(held_weights > 0, 0.0).sum(axis=1)
    out["short"] = contrib.where(held_weights < 0, 0.0).sum(axis=1)
    out["book"] = book_returns
    out["long gross"] = held_weights.clip(lower=0).sum(axis=1)
    out["short gross"] = -held_weights.clip(upper=0).sum(axis=1)
    return pd.DataFrame(out)


def diagnose(label: str, tests=OPEN4, strategy: str = KEY) -> dict:
    """Rebuild every period of a saved label with logs: the reconciliation, the sleeve legs,
    the order sizes at :data:`ACCOUNT`, and (with the hedge on) the month-end ex-ante beta
    after the hedge. Saved to ``results2/<strategy>/diag-<label>.pkl``; prints the findings."""
    from trading_simple import backtest as bt
    lab = _lab()
    tests = _open_only(tests)
    res = lab.load_results(strategy, label)
    strat = lab.load_strategy(strategy, res["meta"]["overrides"])
    settings = lab._settings()
    path = lab.RESULTS / strategy / f"diag-{label}.pkl"
    keep = pickle.loads(path.read_bytes()) if path.exists() else {}
    tot = {"refused": {}, "partial995": {}, "partial99": {}, "stuck": 0, "nan_price": 0}
    for t in tests:
        rows = []
        for i, (te0, _) in enumerate(lab.periods(t)):
            p = _panel(lab, t, te0, strat, res["meta"].get("hedge", False))
            parts = _compose(strat, p)
            sig = _signals(strat, parts)
            window, s = bt._score_window(p, sig, te0)
            pf = rebuild(strat, window, s, settings)
            r = reconcile_period(pf, window, res[t][i]["returns"])
            held = pf.asset_value(group_by=False).div(pf.value(), axis=0).shift(1).fillna(0.0)
            legs = _sleeve_legs(parts, window, held, res[t][i]["returns"])
            lg = r.pop("log")
            real = lg[(lg["class"] != "none") & (lg["need"].abs() > 1e-6)]
            d = real["dollars"].to_numpy()
            shares = d / real["Val Price"].to_numpy()
            fee = bt.strategy_costs(strat, window, settings)[0] - settings.costs.spread_bps / 2e4
            small = d < 250.0
            years = lab.years_of(window.close.index)
            # the lab charges fee x order value; a real order pays max($1, $0.005 a share)
            extra = float((1.0 - fee * d[small]).sum() / ACCOUNT / years)
            under = d < 1000.0
            full = float((1.0 - fee * d[under]).sum() / ACCOUNT / years)
            net = float((np.maximum(settings.costs.min_commission,
                                    shares * settings.costs.commission_per_share)
                         - fee * d).sum() / ACCOUNT / years)
            post = None
            if strat.params.get("hedge_max", 0) > 0:
                rm = strat.signals.__globals__["component"]("mom")
                core = strat.signals.__globals__["component"]("si")._without(p, "SPY")
                rws = rm._full_rows(core.close)
                first = rm._month_ends(p.close.index, rws)[rws].shift(1, fill_value=False)
                first = first.reindex(p.close.index, fill_value=False)
                betas = strat.signals.__globals__["trailing_betas"](
                    p.close, core.close.columns, strat.params["beta_window"])
                after = parts["beta"] + parts["hedge"] * betas["SPY"]
                inside = p.close.index >= window.close.index[0]
                post = after[first & inside]
            rows.append({**{k: v for k, v in r.items() if k not in ("refused", "partial99",
                                                                   "partial995", "nan_price")},
                         "refused": r["refused"],
                         "partial995": r["partial995"], "nan_price": r["nan_price"],
                         "legs": legs, "order dollars": d, "orders < $250": int(small.sum()),
                         "extra $1 on < $250 (share of book a year)": extra,
                         "extra $1 on < $1000 (share of book a year)": full,
                         "net re-pricing, every order (share of book a year)": net,
                         "post-hedge beta": post, "returns": res[t][i]["returns"],
                         "bench": res[t][i]["bench"]})
            for k2 in ("refused", "partial995", "partial99"):
                for cls, n in r[k2]["class"].value_counts().items():
                    tot[k2][cls] = tot[k2].get(cls, 0) + int(n)
            tot["stuck"] += len(r["stuck"])
            tot["nan_price"] += len(r["nan_price"])
            lines = [f"{label} {t} {te0}: rebuild {r['rebuild']:.1e}; orders {r['orders']}; "
                     f"refused {len(r['refused'])}, partial<99.5% {len(r['partial995'])}, "
                     f"NaN-price {len(r['nan_price'])}, stuck {len(r['stuck'])}, off-target "
                     f"days {r['off_days']}, missing long {r['missing']['long']:.2%} short "
                     f"{r['missing']['short']:.2%}; orders < $250: {int(small.sum())} of "
                     f"{len(d)} (median ${np.median(d) if len(d) else 0:,.0f})"]
            lines += _fmt(r["refused"], "REFUSED") + _fmt(r["partial995"], "partial")
            for z in r["stuck"]:
                lines.append(f"    STUCK {z['side']} {z['name']} from {z['from'].date()}: "
                             f"{z['bars']} bars ({z['status']}, free cash {z['free cash']:.2f}"
                             f"); weight {z['weight']:+.4f}; P&L while stuck {z['pnl']:+.2%}")
            if r["dead"]:
                lines.append(f"    targets never held (bars): {r['dead']}")
            print("\n".join(lines), flush=True)
        keep[t] = rows
    path.write_bytes(pickle.dumps(keep))
    print(f"{label} TOTAL refused {tot['refused']}; partial<99.5% {tot['partial995']}; "
          f"partial<99% {tot['partial99']}; stuck runs {tot['stuck']}; NaN-price drops "
          f"{tot['nan_price']}", flush=True)
    return keep


def diag_summary(label: str, strategy: str = KEY) -> pd.DataFrame:
    """A saved diagnosis per test (and TRAIN = TB + T3): alpha as saved and with the stuck P&L
    removed, the sleeves' and legs' annualized contributions, the order-size re-pricing."""
    lab = _lab()
    keep = pickle.loads((lab.RESULTS / strategy / f"diag-{label}.pkl").read_bytes())
    groups = {t: keep[t] for t in OPEN4 if t in keep}
    if "TB" in keep and "T3" in keep:
        groups["TRAIN"] = keep["TB"] + keep["T3"]
    table = {}
    for t, rows in groups.items():
        s0 = lab.stitched([(r["returns"], r["bench"]) for r in rows])
        s1 = lab.stitched([(r["returns"] - r["stuck_pnl"].reindex(r["returns"].index)
                            .fillna(0.0), r["bench"]) for r in rows])
        legs = pd.concat([r["legs"] for r in rows])
        col = {"alpha": s0["alpha"], "alpha stuck removed": s1["alpha"],
               "stuck runs": sum(len(r["stuck"]) for r in rows),
               "stuck P&L": float(sum(r["stuck_pnl"].sum() for r in rows)),
               "refused": sum(len(r["refused"]) for r in rows),
               "partial<99.5%": sum(len(r["partial995"]) for r in rows),
               "orders < $250 / yr": sum(r["orders < $250"] for r in rows) / s0["years"],
               "re-price < $250 (pts/yr)": 100 * np.mean(
                   [r["extra $1 on < $250 (share of book a year)"] for r in rows]),
               "re-price < $1000 (pts/yr)": 100 * np.mean(
                   [r["extra $1 on < $1000 (share of book a year)"] for r in rows]),
               "net re-price, all orders (pts/yr)": 100 * np.mean(
                   [r["net re-pricing, every order (share of book a year)"] for r in rows])}
        mkt = pd.concat([r["bench"] for r in rows]).reindex(legs.index).fillna(0.0)
        for leg in [c for c in legs.columns if c not in ("book", "long gross", "short gross")]:
            x = legs[leg]
            var = float(mkt.var())
            beta = float(x.cov(mkt) / var) if var > 0 else np.nan
            col[f"{leg} contrib"] = float(x.mean() * 252)
            col[f"{leg} alpha"] = float((x.mean() - beta * mkt.mean()) * 252)
        col["long gross"] = float(legs["long gross"].mean())
        col["short gross"] = float(legs["short gross"].mean())
        post = [r["post-hedge beta"] for r in rows if r["post-hedge beta"] is not None]
        if post:
            col["max month-end post-hedge beta"] = float(pd.concat(post).max())
        table[t] = col
    out = pd.DataFrame(table)
    with pd.option_context("display.width", 250, "display.max_rows", 100):
        print(out.to_string(float_format=lambda x: f"{x:.4f}"))
    return out


def _series(res, tests, key="returns"):
    """A saved label's daily series over ``tests``, stitched and de-duplicated."""
    x = pd.concat([row[key] for t in tests for row in res[t]])
    return x[~x.index.duplicated()].sort_index()


def preview(tests=OPEN4) -> pd.DataFrame:
    """The return-sum preview from the components' SAVED lab books (not an engine book):
    Σ b_s / g_s × the component's daily return, each test stitched, benchmark = SI's. U0 =
    0.30 each; C_core = SI + MOM at 0.45; CRISIS = MOM + AH at 0.45; C_ahu0 = U0 with AH's U0v2.
    Late = the same sum of the components' +50 bp books."""
    lab = _lab()
    tests = _open_only(tests)
    res = {k: lab.load_results(c, lab_) for k, (c, lab_, _) in COMPONENT_LABELS.items()}
    g = {k: v[2] for k, v in COMPONENT_LABELS.items()}
    mixes = {"U0": {"si": 0.30, "mom": 0.30, "ah": 0.30},
             "C_core": {"si": 0.45, "mom": 0.45}, "CRISIS": {"mom": 0.45, "ah": 0.45},
             "C_ahu0": {"si": 0.30, "mom": 0.30, "ah_u0": 0.30}}
    groups = [(t, (t,)) for t in tests] + ([("TRAIN", ("TB", "T3"))]
                                           if {"TB", "T3"} <= set(tests) else [])
    table = {}
    for name, mix in mixes.items():
        col = {}
        for gname, ts in groups:
            idx = _series(res["si"], ts).index
            bench = _series(res["si"], ts, "bench").reindex(idx).fillna(0.0)
            for key in ("returns", "late"):
                r = sum(b / g[k] * _series(res[k], ts, key).reindex(idx).fillna(0.0)
                        for k, b in mix.items())
                s = lab.stitched([(r, bench)])
                if key == "returns":
                    col[f"{gname} alpha"] = s["alpha"]
                    col[f"{gname} P"] = s["P(alpha > 0)"]
                    col[f"{gname} vol"] = s["volatility"]
                    col[f"{gname} beta"] = s["beta"]
                    col[f"{gname} maxDD"] = s["max drawdown"]
                else:
                    col[f"{gname} late"] = s["alpha"]
        table[name] = col
    out = pd.DataFrame(table)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(out.to_string(float_format=lambda x: f"{x:+.4f}"))
    return out


def _train(res, key="returns"):
    """The ``(returns, bench)`` pairs of a label's TRAIN periods (TB + T3)."""
    return [(r[key], r["bench"]) for t in ("TB", "T3") if t in res for r in res[t]]


def _paired(lab, a_rows, b_rows):
    """Paired alpha delta (a − b) on the same days and bootstrap blocks (``lab.report``)."""
    s, b = lab.stitched(a_rows), lab.stitched(b_rows)
    pr = lab.paired(s["returns"], b["returns"].reindex(s["returns"].index).fillna(0.0),
                    b["bench"].reindex(s["returns"].index).fillna(0.0), b["years"])
    return pr["delta alpha"], pr["P(delta > 0)"]


def table(labels, base: str | None = None, strategy: str = KEY) -> pd.DataFrame:
    """Saved labels side by side: per test alpha (P), beta, vol, max DD, late / 2×-cost /
    top-5-out alpha, orders a year; the stitched TRAIN (TB + T3, G1); with ``base`` the paired
    Δ (P) per test and on the stitched TRAIN, and periods better."""
    lab = _lab()
    out = {}
    ref = lab.load_results(strategy, base) if base else None
    for name in labels:
        res = lab.load_results(strategy, name)
        col = {}
        summ = lab.summary(res)
        for t in [x for x in OPEN4 if x in res]:
            s = summ[t]
            col[f"{t} alpha"], col[f"{t} P"] = s["alpha"], s["P"]
            col[f"{t} beta"], col[f"{t} vol"], col[f"{t} maxDD"] = s["beta"], s["vol"], s["maxDD"]
            for k in ("late alpha", "2x costs alpha", "top5-out alpha"):
                if k in s:
                    col[f"{t} {k}"] = s[k]
            col[f"{t} orders/yr"] = s["orders/yr"]
            col[f"{t} ex-ante beta"] = s["ex-ante beta"]
            if ref is not None and name != base and t in ref:
                d, pd_ = _paired(lab, [(r["returns"], r["bench"]) for r in res[t]],
                                 [(r["returns"], r["bench"]) for r in ref[t]])
                col[f"{t} Δ"], col[f"{t} P(Δ>0)"] = d, pd_
                better = [lab.period_alpha(x["returns"], x["bench"])
                          - lab.period_alpha(y["returns"], y["bench"])
                          for x, y in zip(res[t], ref[t], strict=True)]
                col[f"{t} better"] = f"{sum(v > 0 for v in better)}/{len(better)}"
        if "TB" in res and "T3" in res:
            tr = lab.stitched(_train(res))
            col["TRAIN alpha"], col["TRAIN P"], col["TRAIN beta"] = (
                tr["alpha"], tr["P(alpha > 0)"], tr["beta"])
            for k in ("late", "2x costs"):
                if k in res["TB"][0]:
                    col[f"TRAIN {k}"] = lab.stitched(_train(res, k))["alpha"]
            if ref is not None and name != base and "TB" in ref and "T3" in ref:
                col["TRAIN Δ"], col["TRAIN P(Δ>0)"] = _paired(lab, _train(res), _train(ref))
        out[name] = col
    frame = pd.DataFrame(out)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(frame.to_string(float_format=lambda x: f"{x:+.4f}"))
    return frame


def noise_sd(label: str, strategy: str = KEY) -> dict:
    """The sd (ddof 1) and RMS of the noise copies' stitched alpha minus the label's, per test
    (``lab.noise``'s saved copies; nothing is re-run)."""
    lab = _lab()
    res = lab.load_results(strategy, label)
    out = {}
    files = sorted((lab.RESULTS / strategy).glob(f"{label}-noise5-*.pkl"))
    copies = [pickle.loads(f.read_bytes()) for f in files]
    for t in [x for x in OPEN4 if x in res]:
        base = lab.stitched([(r["returns"], r["bench"]) for r in res[t]])["alpha"]
        d = [lab.stitched([(r["returns"], r["bench"]) for r in c[t]])["alpha"] - base
             for c in copies if t in c]
        if d:
            out[t] = {"deltas": [round(x, 4) for x in d], "sd": float(np.std(d, ddof=1)),
                      "rms": float(np.sqrt(np.mean(np.square(d))))}
    print(json.dumps(out, indent=1))
    return out


def component_stats(tests=("TB", "T3", "T1", "T2")) -> pd.DataFrame:
    """The components' saved books (read-only): alpha, volatility, alpha ÷ volatility per test
    and on the stitched TRAIN (TB + T3)."""
    lab = _lab()
    out = {}
    for key in ("si", "mom", "ah"):
        comp, label, _ = COMPONENT_LABELS[key]
        res = lab.load_results(comp, label)
        col = {}
        for t in tests:
            s = lab.stitched([(r["returns"], r["bench"]) for r in res[t]])
            col[f"{t} alpha"], col[f"{t} vol"] = s["alpha"], s["volatility"]
            col[f"{t} a/vol"] = s["alpha"] / s["volatility"]
        tr = lab.stitched(_train(res))
        col["TRAIN alpha"], col["TRAIN vol"] = tr["alpha"], tr["volatility"]
        col["TRAIN a/vol"] = tr["alpha"] / tr["volatility"]
        out[f"{key} ({comp}/{label})"] = col
    frame = pd.DataFrame(out)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(frame.to_string(float_format=lambda x: f"{x:+.4f}"))
    return frame


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == "audit":
        audit(json.loads(a[1]) if len(a) > 1 else {},
              tuple(a[2].split(",")) if len(a) > 2 else OPEN4)
    elif cmd == "reproduce":
        where = (tuple(tuple(x.split(":")) for x in a[1].split(",")) if len(a) > 1 else
                 (("T1", "2024-01-01"), ("TB", "2020-01-01"), ("T2", "2026-07-01")))
        reproduce(where)
    elif cmd == "diagnose":
        diagnose(a[1], tuple(a[2].split(",")) if len(a) > 2 else OPEN4)
        diag_summary(a[1])
    elif cmd == "summary":
        diag_summary(a[1])
    elif cmd == "preview":
        preview()
    elif cmd == "table":
        base = a[a.index("--base") + 1] if "--base" in a else None
        table(a[1].split(","), base)
    elif cmd == "noise":
        noise_sd(a[1])
    elif cmd == "components":
        component_stats()
    else:
        raise SystemExit(__doc__)
