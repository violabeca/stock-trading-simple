"""xs_portable_alpha research helpers (loaded by path; ``strategy.py`` never imports this).

    uv run python strategies/xs_portable_alpha/research.py audit '<overrides>' [TB,T3,T1,T2]
    uv run python strategies/xs_portable_alpha/research.py reproduce
    uv run python strategies/xs_portable_alpha/research.py diagnose <label> [TB,T3,T1,T2]
    uv run python strategies/xs_portable_alpha/research.py preview [<label>]
    uv run python strategies/xs_portable_alpha/research.py table <label,...>
    uv run python strategies/xs_portable_alpha/research.py bars <label> [--base <label>]
    uv run python strategies/xs_portable_alpha/research.py noise <label>
    uv run python strategies/xs_portable_alpha/research.py e0

``audit``: does the lab's runner (``lab.run_portfolio``) hold the account the strategy asks for?
Mechanics and book invariants only, NO returns, on the base, late-fill and 2×-cost books.
``reproduce``: the account at ``core: 0`` must rebuild xs_three_book's saved ``C_core`` book (its
overlay at 0.90) to the last bit — a plumbing check that prints differences only. ``diagnose``:
a saved label rebuilt period by period with vectorbt's order logs (xs_three_book/research.py's
``rebuild`` and ``reconcile_period``): refused and part-filled orders, stuck positions and their
P&L, the core's realized weight, the legs and the order sizes at the declared $150k account.
``preview``: the judge's return sum of the components' SAVED books plus ``core`` × SPY; with a
label, the engine-vs-preview gap on that label's own benchmark. ``table`` / ``bars``: the design's
per-test table (CAGR against buy-and-hold and SPY, alpha net of the core's own alpha, episode loss
shares, drawdown ratios, the stitched 2018-26 record) and its pre-registered bars A1-A4.
T4 and TL are locked: every function here refuses them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/janmarti/Documents/code/stock-trading-simple")
KEY = "xs_portable_alpha"
OPEN4 = ("TB", "T3", "T1", "T2")
LOCKED = {"T4", "TL"}
SPY = "SPY"
ACCOUNT = 150_000.0          # the declared account (config.yaml)
#: xs_three_book's research helpers, reused for the order-log rebuild and the reconciliation;
#: pinned so that a change there cannot silently change this audit.
TB_RESEARCH = ("strategies/xs_three_book/research.py",
               "cce351fed8c0817f14ab73b2a87ae0d866750da8bde781b5da4aaa3a3f132401")
#: The overlay's components: saved lab label and nominal gross (read-only, for the preview).
COMPONENTS = {"si": ("informed_flow_ls_p1", "v6_si", 0.98), "mom": ("resid_momentum", "U0v2", 0.90)}
#: c = 0, the overlay alone at the full gross: xs_three_book's saved C_core (read-only).
C_CORE = ("xs_three_book", "C_core")
TINY = 1e-9
_MOD: dict = {}


def _load(alias: str, rel: str, sha: str | None = None):
    """A module loaded once by path (refused if its SHA-256 is not ``sha``)."""
    if alias not in _MOD:
        path = REPO / rel
        if sha is not None and hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{rel} changed since it was pinned: re-check and re-pin")
        spec = importlib.util.spec_from_file_location(alias, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MOD[alias] = module
    return _MOD[alias]


def _lab():
    """The proposals lab (``strategies/proposals_lab/lab.py``)."""
    return _load("lab", "strategies/proposals_lab/lab.py")


def _tbr():
    """xs_three_book's research helpers (pinned, :data:`TB_RESEARCH`)."""
    return _load("_xspa_tbr", *TB_RESEARCH)


def _open_only(tests) -> list[str]:
    """The tests as a list; refuses the locked T4 and TL."""
    tests = list(tests)
    if LOCKED & set(tests):
        raise SystemExit(f"{sorted(LOCKED & set(tests))}: locked tests are never read here")
    return tests


def _panel(lab, test, te0, strat):
    """A lab period panel with the strategy's features and SPY merged (``--hedge``)."""
    return lab.with_spy(lab.panel(test, te0, lab.strategy_features(strat)))


def _book_args(strat) -> dict:
    """xs_three_book ``compose()`` arguments of this account's overlay (``gross − core``)."""
    tb = strat.signals.__globals__["book"]()
    p = strat.params
    own = ("core", "core_vol", "sleeves", "gross")
    args = {**tb._DEFAULTS, **{k: v for k, v in p.items() if k not in own},
            "sleeves": p["sleeves"], "gross": p["gross"] - p["core"]}
    keys = inspect.signature(tb.compose).parameters
    return {k: v for k, v in args.items() if k in keys}


def _first_after_month_end(strat, close: pd.Series) -> pd.Series:
    """True on the first bar after each month end (resid_momentum ``_month_ends`` on SPY)."""
    rm = strat.signals.__globals__["book"]().component("mom")
    rows = close.notna()
    return rm._month_ends(close.index, rows).shift(1, fill_value=False)


def audit(overrides: dict | None = None, tests=OPEN4, books=("base", "late", "2x")):
    """Mechanics and book invariants of a config on the lab's runner — NO returns.

    Per period and book (base costs, +50 bp late fills, 2× costs): every rebuild is checked
    equal to ``lab.book``; orders asked, refused, part-filled (< 99.5% and < 99% of the shares
    asked), dropped at a NaN price, stuck runs, position-days off target, the mean gross that
    refused or part-filled entries leave unheld, the largest realized gross, the smallest free
    cash and the core's realized weight range and order count. Per period (signals): the overlay
    equals xs_three_book's book at ``gross − core`` exactly, the core's target is ``core`` (U0)
    or in ``{0, core/2, core}`` and moves only on the first bar after a month end (I1), the
    nominal gross is at most ``gross`` and the coverage switch is never off inside the window.
    """
    from trading_simple import backtest as bt
    lab, tbr = _lab(), _tbr()
    tests = _open_only(tests)
    strat = lab.load_strategy(KEY, overrides or {})
    p_ = strat.params
    settings = lab._settings()
    cost = {"base": settings, "late": lab.with_costs(settings, 1.0, 50.0),
            "2x": lab.with_costs(settings, 2.0)}
    tb = strat.signals.__globals__["book"]()
    args = _book_args(strat)
    c = p_["core"]
    rows = {}
    for t in tests:
        acc = {"periods": 0, "overlay max |diff|": 0.0, "core target ok": True,
               "max nominal gross": 0.0, "switch-off bars": 0, "max rebuild diff": 0.0}
        for b in books:
            acc.update({f"{b} {k}": 0 for k in ("orders", "refused", "partial<99.5%",
                                                 "partial<99%", "nan-price", "stuck runs",
                                                 "off-target days", "core orders")})
            acc.update({f"{b} missing gross": 0.0, f"{b} max realized gross": 0.0,
                        f"{b} min free cash": 1.0, f"{b} core weight min": 1.0,
                        f"{b} core weight max": 0.0})
        bars = 0
        for te0, _ in lab.periods(t):
            p = _panel(lab, t, te0, strat)
            sig = strat.signals(p, **p_)
            w = sig["weights"]
            parts = tb.compose(p, **args)
            names = [x for x in w.columns if x != SPY]
            ref = parts["weights"].reindex(index=w.index, columns=w.columns).fillna(0.0)
            ov = float(np.abs(w[names].to_numpy() - ref[names].to_numpy()).max())
            ov = max(ov, float(ref[SPY].abs().max()))           # no sleeve ever holds SPY
            window, s = bt._score_window(p, sig, te0)
            inside = p.close.index >= window.close.index[0]
            spy_w = w[SPY]
            if p_["core_vol"]:
                moved = spy_w.ne(spy_w.shift()) & (np.arange(len(spy_w)) > 0)
                ok = (set(np.round(spy_w.unique(), 12)) <= {0.0, round(c / 2, 12), round(c, 12)}
                      and not (moved & ~_first_after_month_end(strat, p.close[SPY])).any())
            else:
                ok = bool((spy_w == c).all())
            acc["periods"] += 1
            acc["overlay max |diff|"] = max(acc["overlay max |diff|"], ov)
            acc["core target ok"] = acc["core target ok"] and ok
            acc["max nominal gross"] = max(acc["max nominal gross"],
                                           float(w.abs().sum(axis=1).max()))
            acc["switch-off bars"] += int((~parts["on"][inside]).sum())
            n = len(window.close)
            bars += n
            line = [f"{t} {te0}: overlay |diff| {ov:.1e}, core target {'ok' if ok else 'BAD'}, "
                    f"nominal gross {w.abs().sum(axis=1).max():.3f}, switch-off bars "
                    f"{int((~parts['on'][inside]).sum())}"]
            for b in books:
                pf = tbr.rebuild(strat, window, s, cost[b])
                saved = lab.book(strat, p, cost[b], te0, signals=sig)["returns"]
                diff = float((tbr._returns(pf, cost[b]) - saved).abs().max())
                r = tbr.reconcile_period(pf, window)
                value = pf.value()
                assets = pf.asset_value(group_by=False)
                gross = assets.abs().sum(axis=1) / value
                free = pf.cash(free=True) / value
                core_w = assets[SPY] / value
                core_orders = int((pf.orders.records_readable["Column"] == SPY).sum())
                acc["max rebuild diff"] = max(acc["max rebuild diff"], diff)
                for k, v in (("orders", r["orders"]), ("refused", len(r["refused"])),
                             ("partial<99.5%", len(r["partial995"])),
                             ("partial<99%", len(r["partial99"])),
                             ("nan-price", len(r["nan_price"])), ("stuck runs", len(r["stuck"])),
                             ("off-target days", r["off_days"]), ("core orders", core_orders)):
                    acc[f"{b} {k}"] += v
                acc[f"{b} missing gross"] += (r["missing"]["long"] + r["missing"]["short"]) * n
                acc[f"{b} max realized gross"] = max(acc[f"{b} max realized gross"],
                                                     float(gross.max()))
                acc[f"{b} min free cash"] = min(acc[f"{b} min free cash"], float(free.min()))
                acc[f"{b} core weight min"] = min(acc[f"{b} core weight min"],
                                                  float(core_w.min()))
                acc[f"{b} core weight max"] = max(acc[f"{b} core weight max"],
                                                  float(core_w.max()))
                line.append(f"  {b:4s}: rebuild {diff:.1e}; orders {r['orders']} (core "
                            f"{core_orders}); refused {len(r['refused'])}, partial<99.5% "
                            f"{len(r['partial995'])}, NaN-price {len(r['nan_price'])}, stuck "
                            f"{len(r['stuck'])}, off-target days {r['off_days']}; realized gross "
                            f"max {gross.max():.3f}, free cash min {free.min():.3f}, core weight "
                            f"{core_w.min():.3f}-{core_w.max():.3f}")
                line += ["    " + x for x in tbr._fmt(r["refused"], "REFUSED")
                         + tbr._fmt(r["partial995"], "partial")]
                for z in r["stuck"]:
                    line.append(f"    STUCK {z['side']} {z['name']} from {z['from'].date()}: "
                                f"{z['bars']} bars ({z['status']}, free cash "
                                f"{z['free cash']:.2f})")
                for _, z in r["nan_price"].iterrows():
                    line.append(f"    NaN-price drop {z['Column']} {z['Timestamp'].date()} "
                                f"(target {z['Request Size']:+.4f})")
            print("\n".join(line), flush=True)
        for b in books:
            acc[f"{b} missing gross"] /= max(bars, 1)
        rows[t] = acc
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_rows", 100):
        print(out.to_string())
    return out


def reproduce(where=(("T1", "2024-01-01"), ("TB", "2020-01-01"), ("T2", "2026-07-01"))):
    """The account at ``core: 0`` (the overlay alone at 0.90) against xs_three_book's saved
    ``C_core`` lab book: the largest daily |difference| of the returns (not the returns)."""
    lab = _lab()
    settings = lab._settings()
    saved = lab.load_results(*C_CORE)
    strat = lab.load_strategy(KEY, {"core": 0.0})
    for t, te0 in where:
        i = [x for x, _ in lab.periods(t)].index(te0)
        res = lab.book(strat, _panel(lab, t, te0, strat), settings, te0)
        ref = saved[t][i]["returns"]
        d = float((res["returns"] - ref.reindex(res["returns"].index)).abs().max())
        print(f"reproduce core 0 vs {C_CORE[0]}/{C_CORE[1]} {t} {te0}: max |daily diff| {d:.2e}",
              flush=True)


def diagnose(label: str, tests=OPEN4) -> dict:
    """Rebuild every period of a saved label with logs: the reconciliation (refused and
    part-filled orders, stuck positions and their P&L, off-target days, missing exposure), the
    legs (core, overlay long, overlay short: each held weight at the previous close × the
    close-to-close return), the core's realized weight, and the order sizes at :data:`ACCOUNT`
    with the hand re-pricing at the $1 minimum. Saved to ``results2/<KEY>/diag-<label>.pkl``."""
    from trading_simple import backtest as bt
    lab, tbr = _lab(), _tbr()
    tests = _open_only(tests)
    res = lab.load_results(KEY, label)
    strat = lab.load_strategy(KEY, res["meta"]["overrides"])
    settings = lab._settings()
    path = lab.RESULTS / KEY / f"diag-{label}.pkl"
    keep = pickle.loads(path.read_bytes()) if path.exists() else {}
    tot = {"refused": {}, "partial995": {}, "partial99": {}, "stuck": 0, "nan_price": 0}
    for t in tests:
        rows = []
        for i, (te0, _) in enumerate(lab.periods(t)):
            p = _panel(lab, t, te0, strat)
            sig = strat.signals(p, **strat.params)
            window, s = bt._score_window(p, sig, te0)
            pf = tbr.rebuild(strat, window, s, settings)
            r = tbr.reconcile_period(pf, window, res[t][i]["returns"])
            value = pf.value()
            held = pf.asset_value(group_by=False).div(value, axis=0).shift(1).fillna(0.0)
            rets = window.close.pct_change(fill_method=None).fillna(0.0)
            contrib = held * rets[held.columns]
            names = [x for x in held.columns if x != SPY]
            legs = pd.DataFrame({
                "core": contrib[SPY],
                "long": contrib[names].where(held[names] > 0, 0.0).sum(axis=1),
                "short": contrib[names].where(held[names] < 0, 0.0).sum(axis=1),
                "book": res[t][i]["returns"], "core weight": held[SPY],
                "long gross": held[names].clip(lower=0).sum(axis=1),
                "short gross": -held[names].clip(upper=0).sum(axis=1)})
            lg = r.pop("log")
            real = lg[(lg["class"] != "none") & (lg["need"].abs() > 1e-6)]
            d = ((real["need"] * real["Val Price"]).abs() / real["Value"] * ACCOUNT).to_numpy()
            shares = d / real["Val Price"].to_numpy()
            fee = bt.strategy_costs(strat, window, settings)[0] - settings.costs.spread_bps / 2e4
            years = lab.years_of(window.close.index)
            small, under = d < 250.0, d < 1000.0
            # the lab charges fee x order value; a real order pays max($1, $0.005 a share)
            extra = float((1.0 - fee * d[small]).sum() / ACCOUNT / years)
            full = float((1.0 - fee * d[under]).sum() / ACCOUNT / years)
            net = float((np.maximum(settings.costs.min_commission,
                                    shares * settings.costs.commission_per_share)
                         - fee * d).sum() / ACCOUNT / years)
            rows.append({**{k: v for k, v in r.items() if k not in ("partial99",)},
                         "legs": legs, "order dollars": d, "orders < $250": int(small.sum()),
                         "re-price < $250": extra, "re-price < $1000": full,
                         "re-price all": net, "returns": res[t][i]["returns"],
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
                     f"{r['missing']['short']:.2%}; core weight {held[SPY].iloc[1:].min():.3f}-"
                     f"{held[SPY].max():.3f}; orders < $250: {int(small.sum())} of {len(d)} "
                     f"(median ${np.median(d) if len(d) else 0:,.0f})"]
            lines += tbr._fmt(r["refused"], "REFUSED") + tbr._fmt(r["partial995"], "partial")
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


def diag_summary(label: str) -> pd.DataFrame:
    """A saved diagnosis per test (and TRAIN = TB + T3): alpha as saved and with the stuck P&L
    removed, refusals, the legs' annualized contributions, the core's mean realized weight and
    the order re-pricing at :data:`ACCOUNT`."""
    lab = _lab()
    keep = pickle.loads((lab.RESULTS / KEY / f"diag-{label}.pkl").read_bytes())
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
               "core weight mean": float(legs["core weight"].mean()),
               "long gross": float(legs["long gross"].mean()),
               "short gross": float(legs["short gross"].mean()),
               "orders < $250 / yr": sum(r["orders < $250"] for r in rows) / s0["years"]}
        for k in ("re-price < $250", "re-price < $1000", "re-price all"):
            col[f"{k} (pts/yr)"] = 100 * float(np.mean([r[k] for r in rows]))
        for leg in ("core", "long", "short"):
            col[f"{leg} contrib (/yr)"] = float(legs[leg].mean() * 252)
        table[t] = col
    out = pd.DataFrame(table)
    with pd.option_context("display.width", 250, "display.max_rows", 100):
        print(out.to_string(float_format=lambda x: f"{x:.4f}"))
    return out


# --- reading: the per-test table, the bars -----------------------------------------------------

def _series(res, tests, key="returns") -> pd.Series:
    """A saved label's daily series over ``tests``, stitched and de-duplicated."""
    x = pd.concat([row[key] for t in tests for row in res[t]])
    return x[~x.index.duplicated()].sort_index()


def _spy_close() -> pd.Series:
    """SPY's adjusted close (``lab.spy()``), 2015 → the lab's last bar."""
    return _lab().spy()


def _core_multiplier(strat, index) -> pd.Series:
    """The core's multiplier in force on each of ``index``'s days: 1 (U0) or I1's ``m_t``
    (``strategy.core_scale`` on SPY's whole series, which equals the per-panel value on every
    day of a period: the rule is causal, test-pinned)."""
    if not strat.params["core_vol"]:
        return pd.Series(1.0, index=index)
    spy = _spy_close()
    return strat.signals.__globals__["core_scale"](spy).reindex(index)


def core_book(close: pd.Series, target: pd.Series) -> pd.Series:
    """Daily returns of an account that holds ONLY the core: SPY at ``target`` of its value,
    bought on the first bar and re-sized whenever the target changes, at that bar's close (the
    lab's SPY open = close), the rest in cash, no costs. Its alpha is the core's own alpha as
    the engine holds it (drift included)."""
    cash, held, before, prev = 1.0, 0.0, 1.0, None
    out = np.zeros(len(close))
    for k, (px, tg) in enumerate(zip(close.to_numpy(), target.to_numpy(), strict=True)):
        value = cash + held * px
        if prev is None or tg != prev:
            held, prev = tg * value / px, tg
            cash = value - held * px
        out[k], before = value / before - 1.0, value
    return pd.Series(out, index=close.index)


def _top_out_overlay(rows, k: int = 5) -> pd.Series:
    """The stitched returns with the daily P&L of the ``k`` biggest OVERLAY contributors
    removed (``lab.top_out``'s rule; SPY, the core, is never counted as a name)."""
    tot = pd.concat([r["contrib total"] for r in rows], axis=1).sum(axis=1).drop(SPY,
                                                                                errors="ignore")
    worst = tot.sort_values(ascending=False).index[:k]
    out = []
    for r in rows:
        d = r["contrib daily"]
        cut = d[[c for c in worst if c in d.columns]].sum(axis=1).reindex(r["returns"].index)
        out.append(r["returns"] - cut.fillna(0.0))
    x = pd.concat(out)
    return x[~x.index.duplicated()].sort_index()


def _net(lab, r, b, core_r, c, years, seed: int = 7) -> tuple[float, float]:
    """Alpha of ``r`` minus ``c`` × the alpha of ``core_r``, both against ``b`` on the same
    days, and its block-bootstrap P(> 0) on the same blocks as ``lab.bootstrap``."""
    r, b, q = (np.asarray(x, float) for x in (r, b, core_r))
    a = lab.alpha_of(r, b, years)[3][0] - c * lab.alpha_of(q, b, years)[3][0]
    idx = lab.block_index(len(r), seed=seed)
    boot = lab.alpha_of(r[idx], b[idx], years)[3] - c * lab.alpha_of(q[idx], b[idx], years)[3]
    return float(a), float((boot > 0).mean())


def _stats(lab, strat, res, tests) -> dict:
    """One label over ``tests`` (stitched): the design's per-test numbers (see :func:`table`)."""
    r = _series(res, tests)
    b = _series(res, tests, "bench").reindex(r.index).fillna(0.0)
    years = lab.years_of(r.index)
    c = strat.params["core"]
    spy = _spy_close()
    s = spy.pct_change().reindex(r.index).fillna(0.0)
    h = _core_multiplier(strat, r.index)
    st = lab.stitched([(r, b)])
    out = {"cagr": st["cagr"], "B&H": st["buy & hold"],
           "SPY cagr": float(np.expm1(np.log1p(s).sum() / years)),
           "alpha": st["alpha"], "P": st["P(alpha > 0)"], "beta": st["beta"],
           "vol": st["volatility"], "maxDD": st["max drawdown"],
           "B&H maxDD": float(((1 + b).cumprod() / (1 + b).cumprod().cummax() - 1).min()),
           "SPY alpha": float(lab.alpha_of(s.to_numpy(), b.to_numpy(), years)[3][0])}
    out["maxDD ratio"] = out["maxDD"] / out["B&H maxDD"]
    out["net alpha"], out["net P"] = _net(lab, r, b, h * s, c, years)
    cb = []
    for t in tests:
        for row in res[t]:
            idx = row["returns"].index
            cb.append(core_book(spy.reindex(idx).ffill(), c * h.reindex(idx)))
    cbr = pd.concat(cb)
    cbr = cbr[~cbr.index.duplicated()].sort_index().reindex(r.index).fillna(0.0)
    out["net alpha (core book)"], out["net P (core book)"] = _net(lab, r, b, cbr, 1.0, years)
    for key, name in (("late", "late"), ("2x costs", "2x")):
        if key in res[tests[0]][0]:
            x = _series(res, tests, key).reindex(r.index).fillna(0.0)
            out[f"net {name}"] = _net(lab, x, b, h * s, c, years)[0]
    x = _top_out_overlay([row for t in tests for row in res[t]]).reindex(r.index).fillna(0.0)
    out["net top5-out"] = _net(lab, x, b, h * s, c, years)[0]
    out["lab top5-out (raw)"] = lab.top_out([row for t in tests for row in res[t]])
    day = r.std() * np.sqrt(len(r) / years)
    out["sharpe"] = float(r.mean() * len(r) / years / day) if day > 0 else np.nan
    bd = b.std() * np.sqrt(len(b) / years)
    out["B&H sharpe"] = float(b.mean() * len(b) / years / bd) if bd > 0 else np.nan
    sd = s.std() * np.sqrt(len(s) / years)
    out["SPY sharpe"] = float(s.mean() * len(s) / years / sd) if sd > 0 else np.nan
    out["SPY maxDD"] = float(((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min())
    return out


def _label(name: str):
    """``(strategy, label)`` of a label name; ``other/label`` reads another strategy."""
    return tuple(name.split("/", 1)) if "/" in name else (KEY, name)


def _loaded(lab, strategy, label):
    """A saved label and its loaded strategy (a foreign label reads as ``core`` 0)."""
    res = lab.load_results(strategy, label)
    if strategy == KEY:
        return res, lab.load_strategy(KEY, res["meta"]["overrides"])

    class _Zero:                                    # a book with no core (xs_three_book)
        params = {"core": 0.0, "core_vol": False}
    return res, _Zero()


def table(labels, tests=("TB", "T3", "T1", "T2")) -> pd.DataFrame:
    """Saved labels side by side, per test and on TRAIN (TB + T3) and the stitched 2018-07 →
    2026-10 record (TB + T3 + T1): CAGR against buy-and-hold and SPY, raw alpha (P), SPY's own
    alpha, alpha net of ``core`` × SPY's alpha (P), net of the core-only book, net late / 2× /
    top-5-out (overlay names), beta, vol, max DD and its ratio to buy-and-hold's, Sharpe; then
    the four bear episodes (book, market, loss share) and the down months."""
    lab = _lab()
    tests = _open_only(tests)
    out = {}
    for name in labels:
        res, strat = _loaded(lab, *_label(name))
        groups = [(t, (t,)) for t in tests if t in res]
        if "TB" in res and "T3" in res:
            groups.append(("TRAIN", ("TB", "T3")))
        if all(t in res for t in ("TB", "T3", "T1")):
            groups.append(("2018-26", ("TB", "T3", "T1")))
        col = {}
        for g, ts in groups:
            for k, v in _stats(lab, strat, res, ts).items():
                col[f"{g} {k}"] = v
        for k, v in lab.episodes(res).items():
            col[f"{k} book"], col[f"{k} market"] = v["book"], v["market"]
            col[f"{k} loss share"] = v["book"] / v["market"] if v["market"] < 0 else np.nan
        dm = lab.down_months(res, tests=("TB", "T3", "T1"))
        if dm:
            col["down months book"], col["down months market"] = dm["book mean"], dm["market mean"]
            col["down months n"] = dm["down months"]
        out[name] = col
    frame = pd.DataFrame(out)
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(frame.to_string(float_format=lambda x: f"{x:+.4f}"))
    return frame


def bars(label: str, base: str | None = None) -> dict:
    """The pre-registered bars (RESEARCH_LOG.md §0) of one saved label: A1 (CAGR above
    buy-and-hold on T1 and T2), A2 (every bear episode beaten), A3 (max DD at most half of
    buy-and-hold's on TB, T1, T2), A4 (net alpha: TRAIN > 0, T1 > 0 with P >= 0.90, T2 > 0, G4
    luck >= 95, G5 net late and 2× > 0, G8 net top-5-out > 0); with ``base`` the I1 adoption
    conditions (A1-A4 AND every episode return and the T1 / T2 max DD better than the base's)."""
    lab = _lab()
    res, strat = _loaded(lab, KEY, label)
    st = {g: _stats(lab, strat, res, ts) for g, ts in
          (("TB", ("TB",)), ("T1", ("T1",)), ("T2", ("T2",)), ("TRAIN", ("TB", "T3")))}
    ep = lab.episodes(res)
    lk = None
    if any((lab.RESULTS / KEY).glob(f"luck-{label}-*.pkl")):
        lk = lab.luck(KEY, label)
    a = {"A1 T1 CAGR > B&H": st["T1"]["cagr"] > st["T1"]["B&H"],
         "A1 T2 CAGR > B&H": st["T2"]["cagr"] > st["T2"]["B&H"],
         "A2 every episode beaten": bool(ep) and len(ep) == 4
         and all(v["book"] > v["market"] for v in ep.values()),
         **{f"A3 {t} |maxDD| <= half B&H's": abs(st[t]["maxDD"]) <= 0.5 * abs(st[t]["B&H maxDD"])
            for t in ("TB", "T1", "T2")},
         "A4 TRAIN net > 0": st["TRAIN"]["net alpha"] > 0,
         "A4 T1 net > 0, P >= 0.90": st["T1"]["net alpha"] > 0 and st["T1"]["net P"] >= 0.90,
         "A4 T2 net > 0": st["T2"]["net alpha"] > 0,
         "A4 G4 luck >= 95": bool(lk and lk["percentile"] >= 95),
         "A4 G5 T1 net late and 2x > 0": st["T1"]["net late"] > 0 and st["T1"]["net 2x"] > 0,
         "A4 G8 T1 net top5-out > 0": st["T1"]["net top5-out"] > 0}
    values = {"T1 cagr / B&H": (st["T1"]["cagr"], st["T1"]["B&H"]),
              "T2 cagr / B&H": (st["T2"]["cagr"], st["T2"]["B&H"]),
              "episodes (book, market)": {k: (v["book"], v["market"]) for k, v in ep.items()},
              **{f"{t} maxDD / B&H": (st[t]["maxDD"], st[t]["B&H maxDD"])
                 for t in ("TB", "T1", "T2")},
              "TRAIN net": st["TRAIN"]["net alpha"],
              "T1 net (P)": (st["T1"]["net alpha"], st["T1"]["net P"]),
              "T2 net": st["T2"]["net alpha"], "luck": lk,
              "T1 net late / 2x": (st["T1"]["net late"], st["T1"]["net 2x"]),
              "T1 net top5-out": st["T1"]["net top5-out"]}
    out = {"bars": a, "values": values, "all A1-A4": all(a.values())}
    if base is not None:
        rb, sb = _loaded(lab, KEY, base)
        eb = lab.episodes(rb)
        sbt = {t: _stats(lab, sb, rb, (t,)) for t in ("T1", "T2")}
        out["adoption"] = {
            "A1-A4 pass": all(a.values()),
            "every episode better": all(ep[k]["book"] > eb[k]["book"] for k in eb),
            "T1 |maxDD| smaller": abs(st["T1"]["maxDD"]) < abs(sbt["T1"]["maxDD"]),
            "T2 |maxDD| smaller": abs(st["T2"]["maxDD"]) < abs(sbt["T2"]["maxDD"])}
        out["adopt"] = all(out["adoption"].values())
    print(json.dumps(out, indent=1, default=lambda x: x if not isinstance(x, float) else
                     round(x, 5)))
    return out


def preview(label: str | None = None, core: float = 0.45, tests=OPEN4) -> pd.DataFrame:
    """The judge's return-sum preview (not an engine book): ``(0.90 − core) / 2`` × each
    component's saved daily return over its nominal gross, plus ``core`` × SPY's daily return;
    against SI's saved benchmark (the judge's numbers) and, with ``label``, against that label's
    own benchmark beside the engine's alpha (the gap of the reconciliation bar)."""
    lab = _lab()
    tests = _open_only(tests)
    comp = {k: lab.load_results(s, lb) for k, (s, lb, _) in COMPONENTS.items()}
    res = lab.load_results(KEY, label) if label else None
    spy = _spy_close().pct_change()
    groups = [(t, (t,)) for t in tests]
    if {"TB", "T3"} <= set(tests):
        groups.append(("TRAIN", ("TB", "T3")))
    out = {}
    for g, ts in groups:
        idx = _series(comp["si"], ts).index
        bench = _series(comp["si"], ts, "bench").reindex(idx).fillna(0.0)
        pv = sum((0.90 - core) / 2 / gross * _series(comp[k], ts).reindex(idx).fillna(0.0)
                 for k, (_, _, gross) in COMPONENTS.items()) + core * spy.reindex(idx).fillna(0.0)
        s = lab.stitched([(pv, bench)])
        col = {"preview cagr": s["cagr"], "preview B&H (SI bench)": s["buy & hold"],
               "preview alpha (SI bench)": s["alpha"], "preview beta": s["beta"],
               "preview maxDD": s["max drawdown"]}
        if res is not None:
            r = _series(res, ts)
            b = _series(res, ts, "bench").reindex(r.index).fillna(0.0)
            e = lab.stitched([(r, b)])
            q = lab.stitched([(pv.reindex(r.index).fillna(0.0), b)])
            col.update({"engine alpha": e["alpha"], "preview alpha (label bench)": q["alpha"],
                        "gap engine - preview": e["alpha"] - q["alpha"],
                        "engine cagr": e["cagr"], "preview cagr (label days)": q["cagr"]})
        out[g] = col
    frame = pd.DataFrame(out)
    with pd.option_context("display.width", 250, "display.max_rows", 100):
        print(frame.to_string(float_format=lambda x: f"{x:+.4f}"))
    return frame


def e0(core: float = 0.45) -> dict:
    """The E0 disclosure (§0, not an engine book): E0 has no short interest, so the SI sleeve sits
    in cash there and the account would be ``(0.90 − core) / 2 / 0.90`` × MOM's saved ``U0v2``
    E0 book plus ``core`` × SPY. Raw alpha, SPY's own alpha and the net alpha against MOM's E0
    benchmark (no SPY in it), with P, and the late-fill net."""
    lab = _lab()
    mom = lab.load_results(*COMPONENTS["mom"][:2])
    r0 = _series(mom, ("E0",))
    b = _series(mom, ("E0",), "bench").reindex(r0.index).fillna(0.0)
    s = _spy_close().pct_change().reindex(r0.index).fillna(0.0)
    years = lab.years_of(r0.index)
    k = (0.90 - core) / 2 / COMPONENTS["mom"][2]
    out = {}
    for key in ("returns", "late"):
        r = k * _series(mom, ("E0",), key).reindex(r0.index).fillna(0.0) + core * s
        st = lab.stitched([(r, b)])
        net, p = _net(lab, r, b, s, core, years)
        out[key] = {"cagr": st["cagr"], "B&H": st["buy & hold"], "alpha": st["alpha"],
                    "SPY alpha": float(lab.alpha_of(s.to_numpy(), b.to_numpy(), years)[3][0]),
                    "net alpha": net, "net P": p}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == "audit":
        audit(json.loads(a[1]) if len(a) > 1 else {},
              tuple(a[2].split(",")) if len(a) > 2 else OPEN4)
    elif cmd == "reproduce":
        reproduce()
    elif cmd == "diagnose":
        diagnose(a[1], tuple(a[2].split(",")) if len(a) > 2 else OPEN4)
        diag_summary(a[1])
    elif cmd == "summary":
        diag_summary(a[1])
    elif cmd == "preview":
        preview(a[1] if len(a) > 1 else None)
    elif cmd == "table":
        table(a[1].split(","))
    elif cmd == "bars":
        bars(a[1], a[a.index("--base") + 1] if "--base" in a else None)
    elif cmd == "noise":
        _tbr().noise_sd(a[1], strategy=KEY)
    elif cmd == "e0":
        e0()
    else:
        raise SystemExit(__doc__)
