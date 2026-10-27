"""resid_momentum research helpers (loaded by path; ``strategy.py`` never imports this).

    uv run python strategies/resid_momentum/research.py premise [E0,TB,T3]
    uv run python strategies/resid_momentum/research.py legs '<overrides>' [tests]
    uv run python strategies/resid_momentum/research.py audit '<overrides>' [tests]

``premise``: the TRAIN-only premise check of RESEARCH_LOG.md §1 — at every month end, the rank
IC and the decile / top-20 spreads of each score against the NEXT month's beta-adjusted return
(``r_i − β_i × r_m``, β the trailing beta known at the decision), per test and per year.
``legs``: the long and the short side of a config run alone through the engine (diagnostic).
``audit``: does the lab's runner (v2: ``lab.run_portfolio``) hold the book the strategy asks
for? Mechanics only (rejected and partly filled orders, the gross they leave unheld), no
returns (RESEARCH_LOG "Review fixes", "Round 2").
``table <labels> [base]`` / ``crashes <labels>``: saved lab labels side by side (G1 with its
late, 2x-cost and noise versions; paired deltas) and the momentum-crash windows; a label
``<name>@v1`` reads round 1's lab-v1 result. ``orders``, ``betas``, ``windows``: the review's
cost check and the tests of its three causal claims.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd

REPO = "/Users/janmarti/Documents/code/stock-trading-simple"


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


def _load(lab, label: str) -> dict:
    """A saved lab label; ``<label>@v1`` reads lab v1's ``results/`` (round 1, for the
    record), a bare label lab v2's ``results2/``."""
    if label.endswith("@v1"):
        import pickle
        path = lab.VAR / "results" / "resid_momentum" / f"{label[:-3]}.pkl"
        return pickle.loads(path.read_bytes())
    return lab.load_results("resid_momentum", label)


def _noise_files(lab, label: str) -> list:
    """The saved noise copies of a label (v1 or v2, as :func:`_load`)."""
    folder = lab.VAR / ("results" if label.endswith("@v1") else "results2") / "resid_momentum"
    base = label[:-3] if label.endswith("@v1") else label
    return sorted(folder.glob(f"{base}-noise*-*.pkl"))


def month_table(lab, strat, test: str, te0: str, te1: str, score: str, **over) -> pd.DataFrame:
    """One row per decision month whose forward month lies inside the period: IC, spreads."""
    import inspect
    p = lab.panel(test, te0, lab.strategy_features(strat))
    mod = strat.signals.__globals__
    keys = inspect.signature(mod["compose"]).parameters
    parts = mod["compose"](p, **{k: v for k, v in {**strat.params, **over, "score": score}.items()
                                 if k in keys})
    reb = parts["rebalance"].copy()
    reb.iloc[np.flatnonzero(parts["rows"].to_numpy())[-1]] = True     # the panel's last full row
    at = np.flatnonzero(reb.to_numpy())
    close = p.close.where(parts["rows"], axis=0).ffill()
    growth = (1 + parts["market"].fillna(0.0)).cumprod()
    lo, hi = pd.Timestamp(te0, tz="UTC"), pd.Timestamp(te1, tz="UTC")
    out = []
    for a, b in zip(at[:-1], at[1:], strict=True):
        if not lo <= close.index[b] <= hi:
            continue
        x = parts["value"].iloc[a]
        fwd = close.iloc[b] / close.iloc[a] - 1
        mkt = growth.iloc[b] / growth.iloc[a] - 1
        y = fwd - parts["beta"].iloc[a] * mkt
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        if len(x) < 50:
            continue
        rx = x.rank(pct=True)
        top, bot = y[rx >= 0.9].mean(), y[rx <= 0.1].mean()
        order = x.sort_values()
        out.append({"date": close.index[a], "n": len(x), "ic": x.rank().corr(y.rank()),
                    "decile": top - bot, "top20": y[order.index[-20:]].mean()
                    - y[order.index[:20]].mean(), "long20": y[order.index[-20:]].mean(),
                    "short20": -y[order.index[:20]].mean()})
    return pd.DataFrame(out)


def _t(s: pd.Series) -> float:
    return float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))) if len(s) > 2 else float("nan")


def premise(tests=("E0", "TB", "T3"), **over) -> dict:
    """Mean monthly IC (t), decile and top-20 spreads (t) per test and year, for both scores."""
    lab = _lab()
    strat = lab.load_strategy("resid_momentum", {})
    result = {}
    for score in ("residual", "raw"):
        tables = []
        for t in tests:
            for te0, te1 in lab.periods(t):
                tab = month_table(lab, strat, t, te0, te1, score, **over)
                tab["test"] = t
                tables.append(tab)
        tab = pd.concat(tables, ignore_index=True)
        def summ(g):
            return {"months": len(g), "IC": round(float(g["ic"].mean()), 4),
                    "IC t": round(_t(g["ic"]), 2), "IC>0": round(float((g["ic"] > 0).mean()), 2),
                    "decile/mo": round(float(g["decile"].mean()), 4),
                    "decile t": round(_t(g["decile"]), 2),
                    "top20/mo": round(float(g["top20"].mean()), 4),
                    "top20 t": round(_t(g["top20"]), 2),
                    "long20/mo": round(float(g["long20"].mean()), 4),
                    "short20/mo": round(float(g["short20"].mean()), 4)}
        result[score] = {"all": summ(tab),
                         **{t: summ(g) for t, g in tab.groupby("test")},
                         **{str(y): summ(g) for y, g in tab.groupby(tab["date"].dt.year)}}
    print(json.dumps(result, indent=1))
    return result


def legs(overrides: dict | None = None, tests=("E0", "TB", "T3", "T1", "T2")) -> pd.DataFrame:
    """Leg attribution (a diagnostic, not a variant): the book's own signals run three times
    through the engine — whole, long side alone, short side alone — per test, stitched.
    Alpha is against the equal-weight market of the same names, as everywhere in the lab."""
    lab = _lab()
    strat = lab.load_strategy("resid_momentum", overrides or {})
    settings = lab._settings()
    rows = {}
    for t in tests:
        pairs = {"book": [], "long": [], "short": []}
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            sig = strat.signals(p, **strat.params)
            for name, w in (("book", sig["weights"]), ("long", sig["weights"].clip(lower=0)),
                            ("short", sig["weights"].clip(upper=0))):
                res = lab.book(strat, p, settings, te0, signals={**sig, "weights": w})
                pairs[name].append((res["returns"], res["bench"]))
        for name, pr in pairs.items():
            s = lab.stitched(pr)
            rows[(t, name)] = {"alpha": s["alpha"], "P": s["P(alpha > 0)"], "beta": s["beta"],
                               "cagr": s["cagr"], "vol": s["volatility"],
                               "maxDD": s["max drawdown"]}
    out = pd.DataFrame(rows).T
    print(out.round(4).to_string())
    return out


def rebuild(strat, window, sig: dict, settings, log: bool = True, runner: str = "lab"):
    """One period's book rebuilt exactly as the lab builds it, with vectorbt's order log on.

    ``runner="lab"`` (lab v2, the default): ``lab.run_portfolio`` — its per-bar call order
    (long cuts, short covers, new shorts, long buys) and its borrow fees (``lab.borrow_fees``).
    ``runner="engine"``: the shared engine's ``backtest._run_portfolio`` (``call_seq="auto"``,
    ``backtest._borrow_fees``), which lab v1 used. Both keep the ``retrim_k`` loop.

    Args:
        strat: The loaded strategy (fill model, order size).
        window: The scored window panel (``backtest._score_window``).
        sig: The window's signals (``weights``, ``borrow``, ``retrim_k``).
        settings: Lab settings (costs, initial cash).
        log: Keep vectorbt's per-request order log.
        runner: ``"lab"`` or ``"engine"``.

    Returns:
        The final ``vectorbt.Portfolio``.
    """
    import vectorbt as vbt

    from trading_simple import backtest as bt
    lab = _lab()
    fees, slip = bt.strategy_costs(strat, window, settings)
    price, slip = bt.fill_prices(window, slip, strat.fill)
    weights = sig["weights"]
    targets = bt.rotation_targets(weights)
    if "borrow" in sig:
        fees = (lab.borrow_fees if runner == "lab" else bt._borrow_fees)(
            targets, sig["borrow"], fees)
    bt.configure()

    def build(tg):
        seq = lab._call_order(tg) if runner == "lab" else "auto"
        return vbt.Portfolio.from_orders(
            window.close, size=tg, size_type="targetpercent", price=price, fees=fees,
            slippage=slip, init_cash=settings.backtest.init_cash, group_by=True,
            cash_sharing=True, lock_cash=True, call_seq=seq, freq=window.freq, log=log)

    k = sig.get("retrim_k")
    return build(targets) if k is None else bt._retrim_loop(build, targets, weights,
                                                             k * weights)


def audit(overrides: dict | None = None, tests=("E0", "TB", "T3", "T1", "T2")) -> pd.DataFrame:
    """Does the engine hold the book the strategy asks for? Mechanics only, no returns.

    Every period is rebuilt with :func:`rebuild` on lab v2's runner (checked equal to the
    lab's own book, day 0 counted from the initial cash as ``lab.book`` does). From
    vectorbt's log, per test: orders asked for (a non-NaN target), rejected (by reason),
    partly filled (filled below 99.5% of the shares asked for), and the gross they leave
    unheld: each shortfall (as a share of the book at the order) counted for the bars until
    that name's next order, averaged over the test's bars. ``off-target days`` counts
    name-days whose engine position has the wrong sign or none against a non-zero target (the
    review's measure; names without a close that day excluded).

    Args:
        overrides: Params merged into the config (e.g. ``{"phased": False}``).
        tests: Lab tests to audit.

    Returns:
        One row per test.
    """
    from trading_simple import backtest as bt
    lab = _lab()
    strat = lab.load_strategy("resid_momentum", overrides or {})
    settings = lab._settings()
    rows = {}
    for t in tests:
        acc = {"orders asked": 0, "rejected": 0, "partly filled": 0, "no price": 0,
               "off-target days": 0, "missing gross x bars": 0.0, "bars": 0, "reasons": {}}
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            sig = strat.signals(p, **strat.params)
            window, s = bt._score_window(p, sig, te0)
            pf = rebuild(strat, window, s, settings)
            ref = lab.book(strat, p, settings, te0, signals=sig)["returns"]
            value = bt.equity_curve(pf)["value"]
            mine = value / value.shift(1).fillna(settings.backtest.init_cash) - 1.0
            assert float((mine - ref).abs().max()) < 1e-12, (t, te0)   # the lab's own book
            lg = pf.logs.values                                  # numpy records
            asked = np.isfinite(lg["req_size"])
            lg = lg[asked]
            with np.errstate(invalid="ignore", divide="ignore"):
                want = lg["req_size"] * lg["value"] / lg["val_price"] - lg["position"]
            got = np.where(lg["res_status"] == 0, lg["res_size"], 0.0)  # 0 = Filled
            nonzero = np.abs(want) > 1e-9 * np.maximum(np.abs(lg["position"]), 1.0)
            reject = lg["res_status"] == 2                               # 2 = Rejected
            noprice = ((lg["res_status"] == 1) & np.isin(lg["res_status_info"], (1, 2))
                       & ((lg["req_size"] != 0) | (lg["position"] != 0)))  # Price/ValPrice NaN
            partial = ((lg["res_status"] == 0) & nonzero
                       & (np.abs(got) < 0.995 * np.abs(want)))
            short = (reject | partial) & nonzero
            gap = (np.abs(want) - np.abs(got)) * lg["val_price"] / lg["value"]
            nxt = {}
            for c in np.unique(lg["col"]):
                ii = np.sort(lg["idx"][lg["col"] == c])
                nxt[c] = ii
            n = len(window.close)
            for i in np.flatnonzero(short):
                c, at = lg["col"][i], lg["idx"][i]
                later = nxt[c][nxt[c] > at]
                acc["missing gross x bars"] += float(gap[i]) * float(
                    (later[0] if len(later) else n) - at)
            for code in np.unique(lg["res_status_info"][reject]):
                key = int(code)
                acc["reasons"][key] = acc["reasons"].get(key, 0) + int(
                    (lg["res_status_info"][reject] == code).sum())
            acc["orders asked"] += int(nonzero.sum())
            acc["rejected"] += int((reject & nonzero).sum())
            acc["partly filled"] += int(partial.sum())
            acc["no price"] += int(noprice.sum())
            w = s["weights"]
            priced = window.close.notna()
            acc["off-target days"] += int(((np.sign(pf.assets()) != np.sign(w)) & (w != 0)
                                           & priced).to_numpy().sum())
            acc["bars"] += n
        acc["mean missing gross"] = acc.pop("missing gross x bars") / max(acc["bars"], 1)
        rows[t] = acc
    out = pd.DataFrame(rows).T
    with pd.option_context("display.width", 200):
        print(out.to_string())
    return out


def order_sizes(overrides: dict | None = None, tests=("E0", "TB", "T3", "T1", "T2"),
                account: float = 40_000.0) -> pd.DataFrame:
    """Every order of the lab book scaled to an ``account``-dollar book: mean and median order,
    the share below $1,000, and the real commission (``max($1, $0.005 × shares)``) against the
    modeled one (the fee fraction's commission part at ``order_notional``), as a yearly share of
    the book (the review's cost check, per test). ``median $ ex day 0`` leaves out each
    period's opening rebuild (a lab artifact: a continuous book never re-buys itself), and
    ``account for $1000`` is the account whose median order, so measured, is $1,000."""
    from trading_simple import backtest as bt
    lab = _lab()
    strat = lab.load_strategy("resid_momentum", overrides or {})
    settings = lab._settings()
    c = settings.costs
    rows = {}
    for t in tests:
        dollars, later, real, modeled, years = [], [], 0.0, 0.0, 0.0
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            res = lab.book(strat, p, settings, te0)
            o = res["pf"].orders.records_readable
            book = res["pf"].value().reindex(o["Timestamp"]).to_numpy()
            d = (o["Size"] * o["Price"]).abs().to_numpy() / book * account
            shares = d / o["Price"].to_numpy()
            real += float(np.maximum(c.min_commission, shares * c.commission_per_share).sum())
            fee = bt.strategy_costs(strat, res["window"], settings)[0] - c.spread_bps / 2e4
            modeled += float((d * fee).sum())
            dollars.extend(d.tolist())
            later.extend(d[(o["Timestamp"] > res["window"].close.index[0]).to_numpy()].tolist())
            years += lab.years_of(res["window"].close.index)
        d = np.asarray(dollars)
        rows[t] = {"orders": len(d), "mean $": d.mean(), "median $": np.median(d),
                   "median $ ex day 0": np.median(later), "< $1000": (d < 1000).mean(),
                   "account for $1000": account * 1000.0 / np.median(later),
                   "real $/yr": real / years, "modeled $/yr": modeled / years,
                   "extra, % of book a year": (real - modeled) / years / account}
    out = pd.DataFrame(rows).T
    print(out.round(4).to_string())
    return out


def table(labels, base: str | None = None) -> pd.DataFrame:
    """Saved lab labels side by side: per test alpha (P), beta, max DD, late / 2×-cost /
    top-5-out alpha, orders a year; the TRAIN-stitched alpha (G1) with its late-fill, 2×-cost
    and noise-copy versions; and, with ``base``, the paired Δ (P(Δ > 0)) per test and on the
    stitched TRAIN. A label ``<name>@v1`` reads round 1's lab-v1 result (:func:`_load`)."""
    lab = _lab()
    out = {}

    def train(res, key="returns"):
        pairs = [(r[key], r["bench"]) for t in lab.TRAIN if t in res for r in res[t]]
        return lab.stitched(pairs) if pairs else None

    ref = _load(lab, base) if base else None
    for name in labels:
        res = _load(lab, name)
        col = {}
        for t, s in lab.summary(res).items():
            col[f"{t} alpha"] = f"{s['alpha']:+.2%} (P {s['P']:.3f})"
            col[f"{t} beta / maxDD"] = f"{s['beta']:+.3f} / {s['maxDD']:+.1%}"
            col[f"{t} late / 2x / top5-out"] = (f"{s.get('late alpha', np.nan):+.2%} / "
                                               f"{s.get('2x costs alpha', np.nan):+.2%} / "
                                               f"{s['top5-out alpha']:+.2%}")
            col[f"{t} orders/yr"] = f"{s['orders/yr']:.0f}"
            col[f"{t} periods"] = " ".join(f"{x:+.3f}" for x in s["periods"])
            if ref is not None and t in ref and name != base:
                b = lab._stitch(ref[t])
                m = lab._stitch(res[t])
                pr = lab.paired(m["returns"], b["returns"].reindex(m["returns"].index)
                                .fillna(0.0), b["bench"].reindex(m["returns"].index)
                                .fillna(0.0), b["years"])
                col[f"{t} Δ vs {base}"] = (f"{pr['delta alpha']:+.2%} "
                                           f"(P {pr['P(delta > 0)']:.3f})")
        tr = train(res)
        if tr is not None:
            col["TRAIN alpha (G1)"] = f"{tr['alpha']:+.2%} (P {tr['P(alpha > 0)']:.3f})"
            if "late" in res[lab.TRAIN[0]][0]:
                col["TRAIN late"] = f"{train(res, 'late')['alpha']:+.2%}"
            if "2x costs" in res[lab.TRAIN[0]][0]:
                col["TRAIN 2x costs"] = f"{train(res, '2x costs')['alpha']:+.2%}"
            noise = _noise_files(lab, name)
            if noise:
                import pickle
                col["TRAIN noise copies"] = " ".join(
                    f"{train(pickle.loads(f.read_bytes()))['alpha']:+.2%}" for f in noise)
            if ref is not None and name != base:
                b = train(ref)
                pr = lab.paired(tr["returns"], b["returns"].reindex(tr["returns"].index)
                                .fillna(0.0), b["bench"].reindex(tr["returns"].index)
                                .fillna(0.0), b["years"])
                col[f"TRAIN Δ vs {base}"] = (f"{pr['delta alpha']:+.2%} "
                                             f"(P {pr['P(delta > 0)']:.3f})")
        dm = lab.down_months(res)
        if dm:
            col["down months book / market"] = (f"{dm['book mean']:+.2%} / "
                                                f"{dm['market mean']:+.2%} (n "
                                                f"{dm['down months']}, >0 {dm['book > 0']:.0%})")
        episodes = lab.episodes(res) if any(t in res for t in ("E0", "TB", "T3", "T1")) else {}
        for k, v in episodes.items():
            col[k] = f"{v['book']:+.1%} ({v['market']:+.1%})"
        out[name] = col
    frame = pd.DataFrame(out)
    with pd.option_context("display.width", 250, "display.max_rows", 300,
                           "display.max_colwidth", 80):
        print(frame.to_string())
    return frame


def crash_windows(labels) -> pd.DataFrame:
    """Book (and market) return over the momentum-crash windows of RESEARCH_LOG §2."""
    lab = _lab()
    wins = [("Nov-Dec 2016", ["E0"], "2016-11-01", "2016-12-31"),
            ("Nov 2020", ["T3"], "2020-11-01", "2020-11-30"),
            ("Jan-Mar 2021", ["T3"], "2021-01-01", "2021-03-31"),
            ("Jul-Sep 2026 T1", ["T1"], "2026-07-01", "2026-09-30"),
            ("Jul-Sep 2026 T2", ["T2"], "2026-07-01", "2026-09-30"),
            ("Jul 1-Oct 16 2026 T1", ["T1"], "2026-07-01", "2026-10-16"),
            ("2024-H1 T1", ["T1"], "2024-01-01", "2024-06-30")]
    out = {}
    for name in labels:
        res = _load(lab, name)
        col = {}
        for w, tests, a, z in wins:
            if not all(t in res for t in tests):
                continue
            r = pd.concat([x["returns"] for t in tests for x in res[t]])
            m = pd.concat([x["bench"] for t in tests for x in res[t]])
            k = (r.index >= pd.Timestamp(a, tz="UTC")) & (r.index <= pd.Timestamp(z, tz="UTC"))
            col[w] = f"{(1 + r[k]).prod() - 1:+.2%} ({(1 + m[k]).prod() - 1:+.2%})"
        out[name] = col
    frame = pd.DataFrame(out)
    print(frame.to_string())
    return frame


def _parts_and_book(lab, strat, p):
    """``compose()``'s frames and the strategy's executed weights on one panel."""
    import inspect
    mod = strat.signals.__globals__
    keys = inspect.signature(mod["compose"]).parameters
    parts = mod["compose"](p, **{k: v for k, v in strat.params.items() if k in keys})
    return parts, strat.signals(p, **strat.params)["weights"]


def beta_check(overrides: dict | None = None, tests=("E0", "TB", "T3", "T1")) -> pd.DataFrame:
    """Claim (i) of the review: is the short leg's high beta an estimation error?

    At every decision inside a test period: the mean ex-ante beta (the trailing beta the score
    used) of the names held after it, per leg; their ex-post beta (OLS on the market over the
    next 126 bars, at least 63 observed); and the market's summed return over the formation
    window. If over-estimated betas were pushed to the short side by a rising market, the
    short-minus-long ex-ante gap would follow that sign, and the shorts' ex-post beta would sit
    well below the ex-ante one."""
    lab = _lab()
    strat = lab.load_strategy("resid_momentum", overrides or {})
    lookback, skip = strat.params["lookback"], strat.params["skip"]
    out = []
    for t in tests:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            parts, w = _parts_and_book(lab, strat, p)
            m, r, beta = parts["market"], parts["returns"], parts["beta"]
            idx = p.close.index
            lo, hi = pd.Timestamp(te0, tz="UTC"), pd.Timestamp(te1, tz="UTC")
            for d in np.flatnonzero(parts["rebalance"].to_numpy()):
                if not (lo - pd.Timedelta(days=31) <= idx[d] < hi) or d + 2 >= len(idx):
                    continue
                held = w.iloc[d + 2]                     # the book once both phases filled
                fm = float(np.log1p(m.iloc[d - lookback + 1:d - skip + 1]).sum())
                fut = slice(d + 1, min(d + 127, len(idx)))
                mf, rf = m.iloc[fut], r.iloc[fut]
                row = {"date": idx[d], "test": t, "formation market": fm}
                for side, mask in (("long", held > 0), ("short", held < 0)):
                    names = held.index[mask]
                    row[f"{side} ex-ante"] = float(beta.iloc[d][names].mean())
                    post = []
                    for c in names:
                        ok = rf[c].notna() & mf.notna()
                        if ok.sum() >= 63:
                            post.append(np.cov(rf[c][ok], mf[ok])[0, 1] / mf[ok].var())
                    row[f"{side} ex-post"] = float(np.mean(post)) if post else np.nan
                out.append(row)
    tab = pd.DataFrame(out).drop_duplicates("date").sort_values("date")
    tab["gap ex-ante"] = tab["short ex-ante"] - tab["long ex-ante"]
    tab["gap ex-post"] = tab["short ex-post"] - tab["long ex-post"]
    up = tab["formation market"] > 0
    summary = pd.DataFrame({
        "decisions": [int(up.sum()), int((~up).sum()), len(tab)],
        "short ex-ante": [tab.loc[up, "short ex-ante"].mean(), tab.loc[~up, "short ex-ante"]
                          .mean(), tab["short ex-ante"].mean()],
        "short ex-post": [tab.loc[up, "short ex-post"].mean(), tab.loc[~up, "short ex-post"]
                          .mean(), tab["short ex-post"].mean()],
        "long ex-ante": [tab.loc[up, "long ex-ante"].mean(), tab.loc[~up, "long ex-ante"]
                         .mean(), tab["long ex-ante"].mean()],
        "long ex-post": [tab.loc[up, "long ex-post"].mean(), tab.loc[~up, "long ex-post"]
                         .mean(), tab["long ex-post"].mean()],
        "gap ex-ante": [tab.loc[up, "gap ex-ante"].mean(), tab.loc[~up, "gap ex-ante"].mean(),
                        tab["gap ex-ante"].mean()],
        "gap ex-post": [tab.loc[up, "gap ex-post"].mean(), tab.loc[~up, "gap ex-post"].mean(),
                        tab["gap ex-post"].mean()]},
        index=["formation market up", "formation market down", "all"])
    print(summary.round(3).to_string())
    print("corr(gap ex-ante, formation market):",
          round(float(tab["gap ex-ante"].corr(tab["formation market"])), 3))
    return tab


def _sector_names() -> dict:
    from pathlib import Path
    codes = json.loads(Path(f"{REPO}/var/proposals_lab/sectors.json").read_text())
    return {float(v): k for k, v in codes.items()}


def window_split(overrides: dict | None = None, windows=None) -> pd.DataFrame:
    """Claims (ii) and (iii) of the review: P&L of the executed weights (Σ weight × close-to-
    close return, costs ignored) over a window, by leg and by sector, each split into its beta
    part (weight × ex-ante β × market) and its residual part (the rest)."""
    lab = _lab()
    strat = lab.load_strategy("resid_momentum", overrides or {})
    names = _sector_names()
    windows = windows or [("2022 bear", "T1", "2022-01-03", "2022-10-12"),
                          ("Nov-Dec 2016", "E0", "2016-11-01", "2016-12-31"),
                          ("Nov 2020", "T3", "2020-11-01", "2020-11-30"),
                          ("Jan-Mar 2021", "T3", "2021-01-01", "2021-03-31"),
                          ("Jul-Oct 16 2026", "T1", "2026-07-01", "2026-10-16")]
    out = []
    for label, t, a, z in windows:
        a_, z_ = pd.Timestamp(a, tz="UTC"), pd.Timestamp(z, tz="UTC")
        for te0, te1 in lab.periods(t):
            lo, hi = pd.Timestamp(te0, tz="UTC"), pd.Timestamp(te1, tz="UTC")
            if hi < a_ or lo > z_:
                continue
            p = lab.panel(t, te0, lab.strategy_features(strat))
            parts, w = _parts_and_book(lab, strat, p)
            k = (p.close.index >= max(a_, lo)) & (p.close.index <= min(z_, hi))
            held = w.shift(1).fillna(0.0)[k]             # weight in force over the bar
            r = parts["returns"].fillna(0.0)[k]
            bm = parts["beta"].shift(1).fillna(1.0)[k].mul(parts["market"].fillna(0.0)[k],
                                                           axis=0)
            sec = p.features["sector"].iloc[-1].map(names).fillna("unknown")
            for side, sign in (("long", held > 0), ("short", held < 0)):
                h = held.where(sign, 0.0)
                tot, beta_part = (h * r).sum(), (h * bm).sum()
                for s_name in sorted(set(sec)):
                    cols = sec.index[sec == s_name]
                    out.append({"window": label, "leg": side, "sector": s_name,
                                "pnl": float(tot[cols].sum()),
                                "beta part": float(beta_part[cols].sum()),
                                "name-days": int((h[cols] != 0).to_numpy().sum())})
    tab = pd.DataFrame(out).groupby(["window", "leg", "sector"], sort=False).sum().reset_index()
    tab["residual part"] = tab["pnl"] - tab["beta part"]
    legs = tab.groupby(["window", "leg"], sort=False)[["pnl", "beta part", "residual part"]].sum()
    print(legs.round(4).to_string())
    big = tab[tab["pnl"].abs() >= 0.005].sort_values(["window", "pnl"])
    print(big.round(4).to_string(index=False))
    return tab


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "premise":
        premise(tuple(a[1].split(",")) if len(a) > 1 else ("E0", "TB", "T3"))
    elif a and a[0] == "legs":
        legs(json.loads(a[1]) if len(a) > 1 else {},
             tuple(a[2].split(",")) if len(a) > 2 else ("E0", "TB", "T3", "T1", "T2"))
    elif a and a[0] == "audit":
        audit(json.loads(a[1]) if len(a) > 1 else {},
              tuple(a[2].split(",")) if len(a) > 2 else ("E0", "TB", "T3", "T1", "T2"))
    elif a and a[0] == "table":
        table(a[1].split(","), a[2] if len(a) > 2 else None)
    elif a and a[0] == "crashes":
        crash_windows(a[1].split(","))
    elif a and a[0] == "orders":
        order_sizes(json.loads(a[1]) if len(a) > 1 else {})
    elif a and a[0] == "betas":
        beta_check(json.loads(a[1]) if len(a) > 1 else {})
    elif a and a[0] == "windows":
        window_split(json.loads(a[1]) if len(a) > 1 else {})
    else:
        raise SystemExit(__doc__)
