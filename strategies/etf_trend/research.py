"""etf_trend research harness — the proposals_lab yardstick on an ETF panel (2026-10-22).

The lab's equity tests do not apply to an ETF book, so this file defines its own half-year
periods on its own panel and reuses the lab's book, statistics, optimizer objective and PBO
(``lab.book``, ``lab.stitched``, ``lab.alpha_of``, ``lab.paired``, ``lab.robust``,
``lab.period_alpha``, ``lab.pbo``, ``lab.top_out``, ``lab._anatomy``). Everything is scored
against SPY buy-and-hold (primary) and the equal-weight buy-and-hold of the period's ETFs
(``lab.book``'s bench). Results: ``lab.RESULTS / "etf_trend" / <label>.pkl`` — ``results2/``
under lab v2 (round 2, labels ending in ``v2``); round 1's labels stay in ``results/``.

Round 2 (lab v2, 2026-10-23): every book is ``lab.book`` as the lab runs it — its portfolio
runner (short covers before new shorts), day 0 counted from the initial cash (the opening
orders' costs included), ``lab.borrow_fees`` — and SPY's buy-and-hold is built the same way
(:func:`spy_returns`). Two strategy-local cross-checks ride along: the borrow a real account
owes on the REALIZED shorts (``exact borrow``) and the late-fill stress on the risky legs only
(``late risky``). Every half-year re-buys its book on day 0 (all those orders are charged in
v2); "continuing orders" leaves them out, which is what one continuous account would place.

CLI (repo root): ``uv run python strategies/etf_trend/research.py <cmd> ...``

    panel                                   load the ETF panel from the DB once (cached)
    premise                                 the TRAIN-only premise check
    run <label> '<overrides>' [tests] [--costs2] [--read-locked]
    report <base> <label>... [--tests TRAIN,VA,VB,VALID]
    luck <label> [test] [n]                 shuffled-asset books (default VALID, 100)
    noise <label>                           drop-one-asset and rebalance-day copies
    tune <label> '<overrides>' '<space>' [n_trials]
    bear <label>...                         bear episodes, calendar years, down months
    verdict <label>                         the adapted gates G1, G2, G4, G5, G6, G8
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trading_simple import backtest as bt

REPO = Path("/Users/janmarti/Documents/code/stock-trading-simple")
HERE = REPO / "strategies" / "etf_trend"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lab = _load("lab", REPO / "strategies" / "proposals_lab" / "lab.py")
S = _load("etf_trend_strategy", HERE / "strategy.py")

KEY = "etf_trend"
OUT = lab.RESULTS / KEY
ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU",
        "XLV", "XLY", "VNQ", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "DBC", "UUP", "SHY"]
CASH = "SHY"
MIN_HISTORY = 252
TESTS = {"TRAIN": lab.halves("2004-01-01", "2015-07-01"),
         "VALID": lab.halves("2016-01-01", "2026-07-01"),
         "LOCKED": lab.halves("2000-01-01", "2003-07-01")}
SPLITS = {"VA": ("VALID", "2016-01-01", "2020-12-31"), "VB": ("VALID", "2021-01-01", lab.LAST)}
OPEN = ["TRAIN", "VALID"]
EPISODES = {"2008-09": ("2007-10-09", "2009-03-09"),
            "Q4 2018": ("2018-09-20", "2018-12-24"),
            "COVID crash": ("2020-02-19", "2020-03-23"),
            "2022 bear": ("2022-01-03", "2022-10-12"),
            "Feb-Apr 2025": ("2025-02-19", "2025-04-08")}
EXTRA_EPISODES = {"2011 debt ceiling": ("2011-04-29", "2011-10-03"),
                  "2015-16": ("2015-05-21", "2016-02-11"),
                  "2000-02": ("2000-03-24", "2002-10-09")}


def _ts(day) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


# --- data ---------------------------------------------------------------------------------------

def full_panel():
    """The 24 ETFs' daily bars 1993 -> LAST from the shared DB (cached in ``panel.pkl``)."""
    path = OUT / "panel.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    from trading_simple.data import load_panel
    p = load_panel(lab._engine(), ETFS, "1d", "1993-01-01", lab.LAST, session="regular")
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(p))
    return p


_PERIOD: dict = {}


def period_panel(te0: str, te1: str, drop: tuple = ()):
    """One period's panel: rows up to ``te1``, the ETFs with >= ``MIN_HISTORY`` closes before
    ``te0`` (an ETF joins at the first period start after its first 12 months), minus ``drop``."""
    key = (te0, te1, tuple(sorted(drop)))
    if key not in _PERIOD:
        p = full_panel()
        idx = p.close.index
        hist = p.close.loc[idx < _ts(te0)].notna().sum()
        live = [c for c in p.close.columns if hist[c] >= MIN_HISTORY and c not in drop]
        cut = idx <= _ts(te1)
        sl = lambda f: f.loc[cut, live]  # noqa: E731
        _PERIOD[key] = replace(p, close=sl(p.close), open_=sl(p.open_), high=sl(p.high),
                               low=sl(p.low), volume=sl(p.volume), features=None)
        if len(_PERIOD) > 64:
            _PERIOD.pop(next(iter(_PERIOD)))
    return _PERIOD[key]


def spy_returns(index, fees: float, slippage: float, fill: str = "open",
                source=None) -> pd.Series:
    """SPY buy-and-hold daily returns on a book's window ``index``, built as ``lab.book``
    builds its bench (lab v2): bought at the window's first fill at the book's own costs, day 0
    counted from the initial cash. SPY's bars come from ``source`` (default the full ETF
    panel), so a noise copy that drops SPY from the book keeps its benchmark."""
    src = full_panel() if source is None else source
    sl = lambda f: f.loc[index, ["SPY"]]  # noqa: E731
    q = replace(src, close=sl(src.close), open_=sl(src.open_), high=sl(src.high),
                low=sl(src.low), volume=sl(src.volume), features=None)
    total, r = bt.benchmark(q, fees=fees, slippage=slippage, fill=fill)
    r = r.reindex(index).fillna(0.0).copy()
    r.iloc[0] = (1.0 + r.iloc[0]) * (1.0 + total) / float(np.prod(1.0 + r.to_numpy())) - 1.0
    return r


def periods(test: str):
    return TESTS[test]


# --- one config through the periods -------------------------------------------------------------

def results_path(label: str) -> Path:
    return OUT / f"{label}.pkl"


#: Stamped into the meta of every result: round 2 = lab v2 (``lab.book``'s runner, day 0 and
#: borrow) and the review-fixed ``strategy.build_weights`` (gross fit on the trading assets
#: only, ``min_change``). Round 1's labels (U0, T1, I1-I6) and the interim lab-v1 re-runs of
#: the first fix (U0r, T1r, I1r-I6r) stay in ``results/``.
CODE = "round 2: lab v2 + build_weights fix (2026-10-23)"


def _first_bar_orders(pf, first) -> pd.Series:
    """True for the orders placed on the window's first bar: the half-year restart, where
    each period buys its whole book (charged under lab v2; one continuous account would not
    place them)."""
    o = pf.orders.records_readable
    return pd.to_datetime(o["Timestamp"]) == first


def fee_gap(pf, fees_frac: float, settings, account: float = 30_000.0, first=None) -> float:
    """Real minus engine commission+spread over a book's orders, as a share of an ``account``
    sized account (the engine charges a fee FRACTION priced at ``order_notional``; a real order
    pays max($1, shares × $0.005) whatever its size). Positive = the engine undercharges.
    ``first`` given: the orders on that bar (the half-year restart) are left out."""
    o = pf.orders.records_readable
    if first is not None and not o.empty:
        o = o[~_first_bar_orders(pf, first).to_numpy()]
    if o.empty:
        return 0.0
    c = settings.costs
    k = account / settings.backtest.init_cash
    value = o["Size"].abs() * o["Price"] * k
    real = np.maximum(c.min_commission, o["Size"].abs() * k * c.commission_per_share)
    real = real + value * c.spread_bps / 2.0 / 1e4
    return float((real - value * fees_frac).sum() / account)


def borrow_owed(pf, rate) -> pd.Series:
    """The borrow a real account accrues on the book's REALIZED shorts, in dollars per bar:
    ``rate`` (yearly; a scalar or a frame per name and bar) × the short's value at the previous
    close × the calendar days since it (weekends and holidays count, as a stock loan does)."""
    short = -pf.asset_value(group_by=False).clip(upper=0.0)
    idx = short.index
    if isinstance(rate, pd.DataFrame):
        rate = rate.reindex(index=idx, columns=short.columns).fillna(0.0)
    days = pd.Series(np.r_[0.0, np.diff(idx.asi8) / 86_400e9], index=idx)
    return (short.shift(1).fillna(0.0) * rate).sum(axis=1) * days / 365.0


def book(strat, p, settings, te0, sig: dict, risky_late_bps: float = 0.0,
         exact_borrow: bool = False) -> dict:
    """One period's book: ``lab.book`` (lab v2) unless a cross-check is asked for, plus SPY's
    buy-and-hold on the same window (``spy``) and the borrow it charged and owed.

    Args:
        strat, p, settings, te0, sig: As ``lab.book`` (``sig`` = the strategy's signals).
        risky_late_bps: That much extra slippage on every leg but the cash asset's, through
            the lab's own runner (``lab.run_portfolio``) with lab v2's day 0: the late-fill
            stress on the ETFs that can move against an order (a 1-3-year Treasury fund does
            not move 50 bp against a market-on-open order).
        exact_borrow: Run WITHOUT the ``borrow`` output and charge each day the borrow a real
            account owes on the REALIZED shorts (:func:`borrow_owed`) instead of the lab's
            fee-fraction model (``lab.borrow_fees``: the accrual charged on the order that
            buys a short back, in target terms). 0 for a long-only book.

    Returns:
        ``lab.book``'s dict plus ``spy`` (daily returns), ``borrow charged $`` (the borrow
        inside the book's fees) and ``borrow owed $`` (:func:`borrow_owed` on its shorts).
    """
    eng = {k: v for k, v in sig.items() if k != "borrow"} if exact_borrow else sig
    init = settings.backtest.init_cash
    if risky_late_bps:
        w, s = bt._score_window(p, eng, te0)
        fees, slip = bt.strategy_costs(strat, w, settings)
        price, slippage = bt.fill_prices(w, slip, strat.fill)
        extra = pd.DataFrame(np.tile([0.0 if c == CASH else risky_late_bps / 1e4
                                      for c in w.close.columns], (len(w.close), 1)),
                             index=w.close.index, columns=w.close.columns)
        pf = lab.run_portfolio(w, s[bt.WEIGHTS_KEY], price=price, fees=fees,
                               slippage=slippage + extra, init_cash=init,
                               borrow=s.get(bt.BORROW_KEY))
        value = bt.equity_curve(pf)["value"]
        res = {"returns": value / value.shift(1).fillna(init) - 1.0, "pf": pf, "window": w}
    else:
        res = lab.book(strat, p, settings, te0, signals=eng)
    w = res["window"]
    fees, slip = bt.strategy_costs(strat, w, settings)
    res["spy"] = spy_returns(w.close.index, fees, slip, strat.fill)
    rate = sig.get(bt.BORROW_KEY, 0.0)
    owed = borrow_owed(res["pf"], rate)
    if exact_borrow:
        value = res["pf"].value()
        res["returns"] = res["returns"] - (owed / value.shift(1).fillna(init)).reindex(
            res["returns"].index).fillna(0.0)
    o = res["pf"].orders.records_readable
    res["borrow charged $"] = 0.0 if exact_borrow or o.empty else float(
        o["Fees"].sum() - fees * (o["Size"].abs() * o["Price"]).sum())
    res["borrow owed $"] = float(owed.sum())
    return res


def _locked_guard(label: str) -> None:
    store = OUT / "locked_reads.json"
    reads = json.loads(store.read_text()) if store.exists() else {}
    if label in reads:
        raise SystemExit(f"LOCKED was already read for {label} on {reads[label]}: never twice")
    reads[label] = pd.Timestamp.now().isoformat(timespec="seconds")
    store.write_text(json.dumps(reads, indent=1))


def run(label: str, overrides: dict | None = None, tests=None, *, late: bool = True,
        costs2: bool = False, drop: int = 0, seed: int = 0, read_locked: bool = False) -> dict:
    """Run a config through ``tests`` (default TRAIN + VALID), period by period."""
    tests = tests or OPEN
    if "LOCKED" in tests:
        if not read_locked:
            raise SystemExit("LOCKED needs --read-locked (once, for the declared final)")
        _locked_guard(label)
    strat = lab.load_strategy(KEY, overrides)
    settings = lab._settings()
    late_s, c2_s = lab.with_costs(settings, 1.0, 50.0), lab.with_costs(settings, 2.0)
    path = results_path(label)
    old = pickle.loads(path.read_bytes()) if path.exists() else {}
    meta = {"overrides": overrides or {}, "drop": drop, "seed": seed, "code": CODE,
            "lab": lab.LAB_VERSION}
    if old and old.get("meta") != meta:
        raise SystemExit(f"{label} was saved with {old.get('meta')}, not {meta}")
    out = {**old, "meta": meta}
    rng = np.random.default_rng(seed)
    for t in tests:
        rows = []
        for te0, te1 in periods(t):
            gone = ()
            if drop:
                risky = [c for c in period_panel(te0, te1).close.columns if c != CASH]
                gone = tuple(rng.choice(risky, drop, replace=False))
            p = period_panel(te0, te1, gone)
            sig = strat.signals(p, **strat.params)
            res = book(strat, p, settings, te0, sig)
            row = {"period": te0, "end": te1, "returns": res["returns"], "bench": res["bench"],
                   "spy": res["spy"], "assets": list(p.close.columns),
                   "dropped": list(gone), **lab._anatomy(res)}
            w = res["window"]
            first = w.close.index[0]
            held = res["pf"].asset_value(group_by=False).div(res["pf"].value(), axis=0)
            row["exposure"] = held.drop(columns=[CASH], errors="ignore").sum(axis=1)
            row["cash share"] = held[CASH] if CASH in held.columns else held.iloc[:, 0] * 0
            row["orders continuing"] = int((~_first_bar_orders(res["pf"], first)).sum())
            frac = bt.strategy_costs(strat, w, settings)[0]
            row["fee gap"] = fee_gap(res["pf"], frac, settings)
            row["fee gap continuing"] = fee_gap(res["pf"], frac, settings, first=first)
            row["borrow charged $"] = res["borrow charged $"]
            row["borrow owed $"] = res["borrow owed $"]
            row["value mean"] = float(res["pf"].value().mean())
            if late:
                row["late"] = book(strat, p, late_s, te0, sig)["returns"]
                row["late risky"] = book(strat, p, settings, te0, sig,
                                         risky_late_bps=50.0)["returns"]
            if costs2:
                row["2x costs"] = book(strat, p, c2_s, te0, sig)["returns"]
            if (sig["weights"] < 0).to_numpy().any():       # the realized-short borrow
                row["exact borrow"] = book(strat, p, settings, te0, sig,
                                           exact_borrow=True)["returns"]
            rows.append(row)
            print(f"{KEY}/{label} {t} {te0}: alpha vs SPY "
                  f"{lab.period_alpha(row['returns'], row['spy']):+.3f}  vs EW "
                  f"{lab.period_alpha(row['returns'], row['bench']):+.3f}  orders "
                  f"{row['orders']}  assets {len(w.close.columns)}", flush=True)
        out[t] = rows
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def load_results(label: str) -> dict:
    return pickle.loads(results_path(label).read_bytes())


def _rows(res: dict, test: str) -> list[dict]:
    """A test's rows; VA / VB are VALID's periods inside their dates."""
    if test in SPLITS:
        src, a, z = SPLITS[test]
        return [r for r in res.get(src, []) if _ts(a) <= _ts(r["period"]) <= _ts(z)]
    return res.get(test, [])


def _vs(rows, bench: str = "spy", key: str = "returns") -> dict:
    return lab.stitched([(r[key], r[bench]) for r in rows])


def _swap(rows, bench: str):
    return [{**r, "bench": r[bench]} for r in rows]


def summary(res: dict, tests=("TRAIN", "VA", "VB", "VALID")) -> dict:
    out = {}
    for t in tests:
        rows = _rows(res, t)
        if not rows:
            continue
        s, e = _vs(rows), _vs(rows, "bench")
        col = {"alpha": s["alpha"], "P": s["P(alpha > 0)"], "beta": s["beta"], "cagr": s["cagr"],
               "SPY": s["buy & hold"], "vol": s["volatility"], "maxDD": s["max drawdown"],
               "alpha EW": e["alpha"], "beta EW": e["beta"], "EW": e["buy & hold"],
               "orders/yr": sum(r["orders"] for r in rows) / s["years"],
               "exposure": float(pd.concat([r["exposure"] for r in rows]).mean()),
               "fee gap bp/yr": 1e4 * sum(r.get("fee gap", 0.0) for r in rows) / s["years"],
               "top1-out alpha": lab.top_out(_swap(rows, "spy"), 1),
               "periods": [round(lab.period_alpha(r["returns"], r["spy"]), 3) for r in rows]}
        if "orders continuing" in rows[0]:
            yrs = s["years"]
            col["continuing orders/yr"] = sum(r["orders continuing"] for r in rows) / yrs
            col["fee gap continuing bp/yr"] = 1e4 * sum(r["fee gap continuing"]
                                                        for r in rows) / yrs
            for k in ("charged", "owed"):
                col[f"borrow {k} %/yr"] = 100 * sum(r[f"borrow {k} $"] / r["value mean"]
                                                    for r in rows) / yrs
        for key in ("late", "late risky", "2x costs", "exact borrow"):
            if key in rows[0]:
                col[f"{key} alpha"] = _vs(rows, key=key)["alpha"]
        out[t] = col
    return out


def report(base: str, labels: list[str], tests=("TRAIN", "VA", "VB", "VALID")) -> pd.DataFrame:
    """Every label against ``base`` on the same days: alpha vs SPY (and vs the EW bench), the
    paired bootstrap Δ vs SPY, periods better, episodes and the down-month mean."""
    ref = load_results(base)
    table = {}
    for name in [base, *labels]:
        res = load_results(name)
        col = {}
        for t in tests:
            rows = _rows(res, t)
            if not rows:
                continue
            s = _vs(rows)
            col[f"{t} alpha"] = s["alpha"]
            col[f"{t} P"] = s["P(alpha > 0)"]
            col[f"{t} beta"] = s["beta"]
            col[f"{t} cagr"] = s["cagr"]
            col[f"{t} SPY"] = s["buy & hold"]
            col[f"{t} maxDD"] = s["max drawdown"]
            col[f"{t} alpha EW"] = _vs(rows, "bench")["alpha"]
            for key in ("late", "late risky", "exact borrow"):
                if key in rows[0]:
                    col[f"{t} {key}"] = _vs(rows, key=key)["alpha"]
            col[f"{t} orders/yr"] = sum(r["orders"] for r in rows) / s["years"]
            if "orders continuing" in rows[0]:
                col[f"{t} continuing orders/yr"] = sum(r["orders continuing"]
                                                       for r in rows) / s["years"]
            base_rows = _rows(ref, t)
            if name != base and base_rows:
                b = _vs(base_rows)
                idx = s["returns"].index
                pr = lab.paired(s["returns"], b["returns"].reindex(idx).fillna(0.0),
                                s["bench"], s["years"])
                col[f"{t} Δ"] = pr["delta alpha"]
                col[f"{t} P(Δ>0)"] = pr["P(delta > 0)"]
                d = [lab.period_alpha(x["returns"], x["spy"])
                     - lab.period_alpha(y["returns"], y["spy"])
                     for x, y in zip(rows, base_rows, strict=False)]
                col[f"{t} better"] = f"{sum(v > 0 for v in d)}/{len(d)}"
        dm = down_months(res)
        if dm:
            col["SPY down months: book / SPY"] = (
                f"{dm['book mean']:+.2%} / {dm['spy mean']:+.2%} (n {dm['n']}, "
                f">0 {dm['book > 0']:.0%})")
        for k, v in episodes(res).items():
            col[k] = f"{v['book']:+.1%} (SPY {v['spy']:+.1%})"
        table[name] = col
    out = pd.DataFrame(table)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_rows", 300):
        print(out.to_string(float_format=lambda x: f"{x:.3f}"))
    return out


# --- bear table -----------------------------------------------------------------------------

def _record(res: dict, tests=("TRAIN", "VALID"), key: str = "returns"):
    rows = [r for t in tests for r in res.get(t, [])]
    r = pd.concat([x[key] for x in rows])
    b = pd.concat([x["spy"] for x in rows]).reindex(r.index).fillna(0.0)
    e = pd.concat([x["bench"] for x in rows]).reindex(r.index).fillna(0.0)
    keep = ~r.index.duplicated()
    return r[keep].sort_index(), b[keep].sort_index(), e[keep].sort_index()


def episodes(res: dict, extra: bool = False, tests=("TRAIN", "VALID")) -> dict:
    r, b, e = _record(res, tests)
    out = {}
    for k, (a, z) in {**EPISODES, **(EXTRA_EPISODES if extra else {})}.items():
        m = (r.index >= _ts(a)) & (r.index <= _ts(z))
        if m.sum() < 5:
            continue
        out[k] = {"book": float((1 + r[m]).prod() - 1), "spy": float((1 + b[m]).prod() - 1),
                  "EW": float((1 + e[m]).prod() - 1)}
    return out


def down_months(res: dict, tests=("TRAIN", "VALID")) -> dict:
    r, b, _ = _record(res, tests)
    key = r.index.tz_convert(None).to_period("M")
    bm, sm = (1 + r).groupby(key).prod() - 1, (1 + b).groupby(key).prod() - 1
    down = sm < 0
    if not down.any():
        return {}
    return {"n": int(down.sum()), "book mean": float(bm[down].mean()),
            "spy mean": float(sm[down].mean()), "book > 0": float((bm[down] > 0).mean()),
            "beta down": float(np.polyfit(sm[down], bm[down], 1)[0])}


def bear(labels: list[str], tests=("TRAIN", "VALID")) -> None:
    rows, yrs = {}, {}
    for lab_ in labels:
        res = load_results(lab_)
        ep = episodes(res, extra=True, tests=tests)
        r, b, e = _record(res, tests)
        if "SPY" not in rows:
            rows["SPY"] = {k: v["spy"] for k, v in ep.items()}
            rows["EW ETFs"] = {k: v["EW"] for k, v in ep.items()}
            for nm, ser in (("SPY", b), ("EW ETFs", e)):
                c = (1 + ser).cumprod()
                rows[nm]["CAGR"] = float(c.iloc[-1] ** (1 / lab.years_of(ser.index)) - 1)
                rows[nm]["max DD"] = float((c / c.cummax() - 1).min())
                yrs[nm] = (1 + ser).groupby(ser.index.year).prod() - 1
        rows[lab_] = {k: v["book"] for k, v in ep.items()}
        c = (1 + r).cumprod()
        rows[lab_]["CAGR"] = float(c.iloc[-1] ** (1 / lab.years_of(r.index)) - 1)
        rows[lab_]["max DD"] = float((c / c.cummax() - 1).min())
        yrs[lab_] = (1 + r).groupby(r.index.year).prod() - 1
        dm = down_months(res, tests)
        rows[lab_]["down-month mean"] = dm.get("book mean", np.nan)
        rows["SPY"]["down-month mean"] = dm.get("spy mean", np.nan)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print((pd.DataFrame(rows).T * 100).round(1).to_string())
        print((pd.DataFrame(yrs).T * 100).round(1).to_string())


# --- luck, noise, the optimizer -----------------------------------------------------------------

COMPOSE_KEYS = ("lookback", "skip", "blend", "hurdle", "vol_com", "mode", "cash", "gross",
                "vol_target", "cov_window", "budget", "overlay", "min_history", "offset", "rule",
                "assets")


def shuffled_signals(strat, p, rng, parts=None) -> dict:
    """The strategy's own states permuted across the period's risky assets, each asset keeping
    its own inverse-vol size; the cash asset absorbs the rest (the luck book)."""
    prm = strat.params
    parts = parts or S.compose(p, **{k: prm[k] for k in COMPOSE_KEYS})
    state = parts["state"]
    cols = state.columns.to_numpy()
    perm = cols[rng.permutation(len(cols))]
    shuf = state[perm].set_axis(cols, axis=1)
    w = S.build_weights(shuf, parts["size"], parts["decision"], p.close.columns,
                        cash=parts["cash"], gross=prm["gross"], band=prm["band"],
                        min_change=prm["min_change"])
    return {"weights": w, "borrow": float(prm["borrow"])}


def luck(label: str, test: str = "VALID", n: int = 100, seed: int = 29) -> dict:
    """``n`` shuffled-asset books per period (states permuted across assets; sizes kept);
    the real alpha vs SPY's percentile among them."""
    res = load_results(label)
    strat = lab.load_strategy(KEY, res["meta"]["overrides"])
    settings = lab._settings()
    stamp = (sorted(strat.params.items()), test, n, seed)
    if res["meta"].get("code"):                  # results of the review-fixed code
        stamp = (*stamp, res["meta"]["code"])
    key = hashlib.sha1(repr(stamp).encode()).hexdigest()[:10]
    cache = OUT / f"luck-{label}-{key}.pkl"
    rows = _rows(res, test)
    if cache.exists():
        books = pickle.loads(cache.read_bytes())
    else:
        rng = np.random.default_rng(seed)
        books = []
        src = SPLITS[test][0] if test in SPLITS else test
        for te0, te1 in periods(src):
            if not any(r["period"] == te0 for r in rows):
                continue
            p = period_panel(te0, te1)
            parts = S.compose(p, **{k: strat.params[k] for k in COMPOSE_KEYS})
            books.append([book(strat, p, settings, te0,
                               shuffled_signals(strat, p, rng, parts))["returns"]
                          for _ in range(n)])
            print(f"luck {label} {test} {te0}: {n} books", flush=True)
        cache.write_bytes(pickle.dumps(books))
    spy = [r["spy"] for r in rows]
    alphas = np.array([lab.stitched([(books[i][k], spy[i]) for i in range(len(books))])["alpha"]
                       for k in range(n)])
    real = _vs(rows)["alpha"]
    out = {"real": real, "median": float(np.median(alphas)),
           "p95": float(np.percentile(alphas, 95)),
           "percentile": float((alphas < real).mean() * 100)}
    print(json.dumps(out))
    return out


def noise(label: str, copies: int = 3) -> dict:
    """Path noise: drop one random risky ETF a period (``copies`` seeds) and move the monthly
    decision to the 6th / 11th / 16th trading day. Returns each kind's sd of Δalpha vs SPY."""
    res = load_results(label)
    over = res["meta"]["overrides"]
    tests = [t for t in OPEN if t in res]
    deltas = {"drop1": {t: [] for t in (*tests, "VA", "VB")},
              "offset": {t: [] for t in (*tests, "VA", "VB")}}
    jobs = [("drop1", f"{label}-drop1-{c}", over, 1, 100 + c) for c in range(copies)]
    jobs += [("offset", f"{label}-off{k}", {**over, "offset": k}, 0, 0) for k in (5, 10, 15)]
    for kind, name, ov, drop, seed in jobs:
        path = results_path(name)
        r = pickle.loads(path.read_bytes()) if path.exists() else run(
            name, ov, tests, late=False, drop=drop, seed=seed)
        for t in deltas[kind]:
            deltas[kind][t].append(_vs(_rows(r, t))["alpha"] - _vs(_rows(res, t))["alpha"])
    out = {kind: {t: {"deltas": [round(x, 4) for x in v], "sd": float(np.std(v, ddof=1))}
                  for t, v in d.items()} for kind, d in deltas.items()}
    print(json.dumps(out, indent=1))
    return out


def tune(label: str, overrides: dict, space: dict, n_trials: int = 120, tests=("TRAIN",),
         seed: int = 42) -> dict:
    """TPE over ``space`` on the TRAIN periods; every trial scored by ``lab.robust`` over its
    per-period alphas vs SPY. Journal ``studies/etf_trend-<label>.log`` (resumable); the
    trials' per-period alphas give the PBO."""
    import optuna
    settings = lab._settings()
    lab.STUDIES.mkdir(parents=True, exist_ok=True)
    name = f"{KEY}-{label}"
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        str(lab.STUDIES / f"{name}.log")))
    study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                sampler=optuna.samplers.TPESampler(
                                    seed=seed, n_startup_trials=min(20, n_trials // 3)),
                                load_if_exists=True)
    stamp = {"space": repr(space), "tests": list(tests), "overrides": repr(overrides),
             "code": CODE}
    if study.user_attrs and study.user_attrs.get("stamp") != stamp:
        raise ValueError(f"study {name} was run with another space/tests/overrides")
    study.set_user_attr("stamp", stamp)
    store = lab.STUDIES / f"{name}.alphas.pkl"
    matrix = pickle.loads(store.read_bytes()) if store.exists() else {}

    def suggest(trial):
        out = {}
        for k, s in space.items():
            if s["type"] == "int":
                out[k] = trial.suggest_int(k, s["low"], s["high"], step=s.get("step", 1))
            elif s["type"] == "float":
                out[k] = trial.suggest_float(k, s["low"], s["high"], step=s.get("step"))
            else:
                out[k] = trial.suggest_categorical(k, s["choices"])
        return out

    def objective(trial):
        params = {**overrides, **suggest(trial)}
        try:
            strat = lab.load_strategy(KEY, params)
        except ValueError:
            raise optuna.TrialPruned() from None
        alphas = []
        for t in tests:
            for te0, te1 in periods(t):
                p = period_panel(te0, te1)
                res = book(strat, p, settings, te0, strat.signals(p, **strat.params))
                alphas.append(lab.period_alpha(res["returns"], res["spy"]))
        matrix[trial.number] = alphas
        store.write_bytes(pickle.dumps(matrix))
        trial.set_user_attr("mean alpha", float(np.mean(alphas)))
        return lab.robust(alphas)

    left = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:
        study.optimize(objective, n_trials=left)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    m = np.array([matrix[t.number] for t in done if t.number in matrix])
    best = study.best_trial
    ranked = sorted(done, key=lambda t: -t.value)[:5]
    out = {"best": best.params, "score": best.value,
           "mean alpha": best.user_attrs.get("mean alpha"), "pbo": lab.pbo(m),
           "trials": len(done),
           "top5": [{"params": t.params, "score": round(t.value, 4)} for t in ranked]}
    print(json.dumps(out, indent=1, default=str))
    return out


# --- the gates ------------------------------------------------------------------------------

def verdict(label: str) -> pd.DataFrame:
    """The adapted gates G1, G2, G4, G5, G6, G8 for one saved label (luck from its cache)."""
    res = load_results(label)
    s = summary(res)
    tr, va = s.get("TRAIN", {}), s.get("VALID", {})
    lk = luck(label) if any(OUT.glob(f"luck-{label}-*.pkl")) else None
    ep = episodes(res)
    dm = down_months(res)
    rows = [
        ("G1 TRAIN alpha vs SPY > 0", tr.get("alpha"), bool(tr and tr["alpha"] > 0)),
        ("G2 VALID alpha vs SPY > 0, P >= 0.90", va.get("alpha"),
         bool(va and va["alpha"] > 0 and va["P"] >= 0.90)),
        ("G4 VALID beats >= 95% of shuffled-asset books", lk["percentile"] if lk else np.nan,
         bool(lk and lk["percentile"] >= 95)),
        ("G5 VALID late-fill and 2x-cost alpha > 0",
         min(va.get("late alpha", np.nan), va.get("2x costs alpha", np.inf)),
         bool(va.get("late alpha", -1) > 0 and va.get("2x costs alpha", 1) > 0)),
        ("G5 (info) as G5, late fills on the risky legs only",
         min(va.get("late risky alpha", np.nan), va.get("2x costs alpha", np.inf)),
         bool(va.get("late risky alpha", -1) > 0 and va.get("2x costs alpha", 1) > 0)),
        ("G6 beats SPY in every bear episode, down-month mean > 0", dm.get("book mean", np.nan),
         bool(ep and all(v["book"] > v["spy"] for v in ep.values())
              and dm.get("book mean", -1) > 0)),
        ("G8 VALID alpha without its best asset > 0", va.get("top1-out alpha"),
         bool(va.get("top1-out alpha", -1) > 0)),
    ]
    out = pd.DataFrame(rows, columns=["gate", "value", "passed"])
    print(out.to_string())
    print("VALID P(alpha>0):", va.get("P"), " episodes:",
          {k: (round(v["book"], 3), round(v["spy"], 3)) for k, v in ep.items()})
    return out


# --- premise check (TRAIN only) -------------------------------------------------------------

def premise(first: str = "2004-01-01", last: str = "2015-12-31") -> dict:
    """Does the 12-month trend state predict next month's excess return over SHY? Monthly, on
    the first trading day of each month, for every risky ETF live then (>= 252 closes):
    raw, vol-scaled and SPY-beta-adjusted next-month excess returns, on vs off."""
    p = full_panel()
    close = p.close
    dec = np.flatnonzero(S.decision_bars(close.index, 0))
    rets = close.pct_change(fill_method=None)
    vol = rets.ewm(com=60, min_periods=20).std() * np.sqrt(252)
    ex = rets.sub(rets[CASH], axis=0)
    spy_ex = ex["SPY"]
    beta = ex.rolling(252, min_periods=126).cov(spy_ex).div(
        spy_ex.rolling(252, min_periods=126).var(), axis=0)
    risky = [c for c in close.columns if c != CASH]
    count = close.notna().cumsum()
    recs = []
    for a, b in zip(dec[:-1], dec[1:], strict=True):
        day = close.index[a]
        if not (_ts(first) <= day <= _ts(last)):
            continue
        r12 = close.iloc[a] / close.iloc[a - 252] - 1 if a >= 252 else close.iloc[a] * np.nan
        hurdle = r12[CASH] if np.isfinite(r12[CASH]) else 0.0
        nxt = close.iloc[b] / close.iloc[a] - 1
        cash_nxt = nxt[CASH] if np.isfinite(nxt[CASH]) else 0.0
        for c in risky:
            if count[c].iloc[a] < 252 or not np.isfinite(r12[c]) or not np.isfinite(nxt[c]):
                continue
            y = nxt[c] - cash_nxt
            spy_y = nxt["SPY"] - cash_nxt
            recs.append({"month": day, "asset": c, "on": float(r12[c] > hurdle), "y": y,
                         "z": y / (vol[c].iloc[a] * np.sqrt(21 / 252)),
                         "resid": y - beta[c].iloc[a] * spy_y if np.isfinite(beta[c].iloc[a])
                         else np.nan})
    d = pd.DataFrame(recs)
    out = {"months": int(d["month"].nunique()), "obs": len(d), "share on": float(d["on"].mean())}
    for col in ("y", "z", "resid"):
        g = d.dropna(subset=[col]).groupby(["month", "on"])[col].mean().unstack()
        diff = (g[1.0] - g[0.0]).dropna()
        out[f"{col}: on - off (mean/month)"] = float(diff.mean())
        out[f"{col}: t over months"] = float(diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff)))
        out[f"{col}: on mean"] = float(d.loc[d["on"] == 1, col].mean())
        out[f"{col}: off mean"] = float(d.loc[d["on"] == 0, col].mean())
    # MOP pooled regression of the vol-scaled return on the sign, Fama-MacBeth t over months
    sgn = d.assign(s=2 * d["on"] - 1)
    slopes = sgn.groupby("month").apply(
        lambda g: np.polyfit(g["s"], g["z"], 1)[0] if g["s"].nunique() > 1 else np.nan,
        include_groups=False).dropna()
    out["z on sign: FM slope"] = float(slopes.mean())
    out["z on sign: FM t"] = float(slopes.mean() / slopes.std(ddof=1) * np.sqrt(len(slopes)))
    # pooled time-series: mean of z x sign (the TSMOM return per unit of risk), t over months
    ts = sgn.assign(pnl=sgn["s"] * sgn["z"]).groupby("month")["pnl"].mean()
    out["TSMOM z-return/month"] = float(ts.mean())
    out["TSMOM t"] = float(ts.mean() / ts.std(ddof=1) * np.sqrt(len(ts)))
    per = sgn.groupby("asset").apply(lambda g: np.corrcoef(g["s"], g["z"])[0, 1]
                                     if g["s"].nunique() > 1 else np.nan, include_groups=False)
    out["per-asset corr(sign, z): mean"] = float(per.mean())
    out["per-asset corr > 0"] = f"{int((per > 0).sum())}/{int(per.notna().sum())}"
    out["per asset"] = per.round(3).to_dict()
    print(json.dumps(out, indent=1, default=float))
    (OUT / "premise.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    os.environ.setdefault("PYTHONHASHSEED", "0")
    cmd = a[0]
    if cmd == "panel":
        p = full_panel()
        c = p.close
        print(pd.DataFrame({"first": c.apply(lambda s: s.first_valid_index()),
                            "last": c.apply(lambda s: s.last_valid_index()),
                            "bars": c.notna().sum(),
                            "holes": c.apply(lambda s: int(s.loc[s.first_valid_index():]
                                                           .isna().sum()))}).to_string())
    elif cmd == "premise":
        premise()
    elif cmd == "run":
        tests = None
        if len(a) > 3 and not a[3].startswith("--"):
            tests = a[3].split(",")
        run(a[1], json.loads(a[2]) if len(a) > 2 else {}, tests, costs2="--costs2" in a,
            read_locked="--read-locked" in a)
    elif cmd == "report":
        tests = ("TRAIN", "VA", "VB", "VALID")
        if "--tests" in a:
            i = a.index("--tests")
            tests, a = tuple(a[i + 1].split(",")), a[:i] + a[i + 2:]
        report(a[1], a[2:], tests)
    elif cmd == "luck":
        luck(a[1], a[2] if len(a) > 2 else "VALID", int(a[3]) if len(a) > 3 else 100)
    elif cmd == "noise":
        noise(a[1])
    elif cmd == "tune":
        tune(a[1], json.loads(a[2]), json.loads(a[3]), int(a[4]) if len(a) > 4 else 120)
    elif cmd == "bear":
        bear(a[1:])
    elif cmd == "verdict":
        verdict(a[1])
    else:
        raise SystemExit(__doc__)
