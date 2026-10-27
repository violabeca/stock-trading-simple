"""quality_junk research helpers (not loaded by the strategy): audits, premise, anatomy, fixes.

Run from the repo root::

    uv run python strategies/quality_junk/research.py coverage [tests]
    uv run python strategies/quality_junk/research.py premise '<overrides>' [tests]
    uv run python strategies/quality_junk/research.py anatomy '<overrides>' [tests]
    uv run python strategies/quality_junk/research.py boundary <label> [tests]
    uv run python strategies/quality_junk/research.py boundary-report <label>
    uv run python strategies/quality_junk/research.py v1v2 <v1 label> <v2 label> [--engine]
    uv run python strategies/quality_junk/research.py bear <label>...
    uv run python strategies/quality_junk/research.py staleness [tests]
    uv run python strategies/quality_junk/research.py floor [tests]

``coverage`` is identity-only (which names have which data, per test period; no returns).
``premise`` is the protocol's TRAIN-only signal check: on every month-end decision bar, the
Spearman rank IC and the top-minus-bottom-tenth spread of the composite score (and of each sleeve)
against the next 21 sessions' BETA-ADJUSTED return, from the next open (the fill) — each name's
return minus its trailing 252-bar beta to the equal-weight universe times the universe's return.
``anatomy`` splits a config's book into its legs. The last four answer the 2026-10-23 review
(RESEARCH_LOG.md §12, ported to lab v2 in §13): ``boundary`` re-reads a saved label with each
half-year's first day charged as ONE continuous book would pay it (lab v2 charges a full rebuild;
v1 dropped the day) and checks that new shorts fill, ``bear`` splits the falling-market record
into what the book's beta implies and the rest, ``staleness`` counts frozen gross-profit levels,
``floor`` counts the names the $5 floors remove.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/janmarti/Documents/code/stock-trading-simple")
_spec = importlib.util.spec_from_file_location("lab", REPO / "strategies/proposals_lab/lab.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)
NAME = "quality_junk"
SECTORS = {v: k for k, v in json.loads((lab.VAR / "sectors.json").read_text()).items()}


def _helpers(overrides=None):
    strat = lab.load_strategy(NAME, overrides)
    return strat, strat.signals.__globals__


def coverage(tests=("E0", "TB", "T3", "T1", "T2")) -> pd.DataFrame:
    """Per test period, averaged over its bars: names, eligible share, the share of eligible names
    on gross profit / assets vs the ROE fallback, short-interest coverage, and the thinnest sector
    group."""
    strat, g = _helpers()
    rows = []
    for t in tests:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            parts = g["compose"](p, **{k: strat.params[k] for k in (
                "w_prof", "w_safe", "vol_window", "min_price", "beta_window")})
            idx = p.close.index
            win = (idx >= pd.Timestamp(te0, tz="UTC")) & (idx <= pd.Timestamp(te1, tz="UTC"))
            el = parts["eligible"][win]
            meas = parts["measure"][win].where(el)
            n = p.close.shape[1]
            sec = p.features["sector"].iloc[-1]
            per_sector = el.iloc[-1].groupby(sec).sum()
            prof = parts["raw"]["prof"][win].where(el)
            rows.append({"test": t, "period": te0, "names": n,
                         "eligible": float(el.sum(axis=1).mean() / n),
                         "gpa": float((meas == 0).sum(axis=1).mean() / n),
                         "roe": float((meas == 1).sum(axis=1).mean() / n),
                         # the two scales: per-bar median of each measure, averaged
                         "gpa median": float(prof.where(meas == 0).median(axis=1).mean()),
                         "roe median": float(prof.where(meas == 1).median(axis=1).mean()),
                         "no sector": int(sec.isna().sum()),
                         "short_interest": float(p.features["short_interest"][win].notna()
                                                 .sum(axis=1).mean() / n),
                         "thinnest sector": f"{SECTORS.get(per_sector.idxmin(), 'none')} "
                                            f"{int(per_sector.min())}"})
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200):
        print(out.round(3).to_string())
    return out


def _forward(p, h: int = 21):
    """Beta-adjusted return from the next open over ``h`` sessions, per decision bar row."""
    ret = p.close.pct_change(fill_method=None)
    mkt = ret.mean(axis=1)
    beta = ret.rolling(252, min_periods=126).cov(mkt).div(
        mkt.rolling(252, min_periods=126).var(), axis=0)
    fwd = p.open_.shift(-(h + 1)) / p.open_.shift(-1) - 1
    mfwd = fwd.mean(axis=1)
    return fwd - beta.mul(mfwd, axis=0)


def premise(overrides=None, tests=("E0", "TB", "T3"), h: int = 21) -> dict:
    """Month-end rank IC and tenth spread of the score and of each sleeve (TRAIN by default)."""
    strat, g = _helpers(overrides)
    keys = ("w_prof", "w_safe", "vol_window", "safety", "safe_by_sector", "min_price",
            "beta_window")
    recs = []
    for t in tests:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            parts = g["compose"](p, **{k: strat.params[k] for k in keys})
            sector = p.features["sector"].fillna(-1.0)
            hybrid = g["_rank"](parts["raw"]["prof"].where(parts["eligible"]), sector)
            growth = g["_prof_rank"](parts["raw"]["growth"].where(parts["eligible"]),
                                     parts["raw"]["gpa"].notna(), sector)
            # review R7: a GP level unchanged for over 200 days falls back to ROE where it exists
            raw = parts["raw"]
            fresh = raw["gpa"].notna() & ~(_age_since_change(p.features["fund_gp_ttm"]) > 200)
            uses = fresh | (raw["gpa"].notna() & raw["roe"].isna())
            prof_fresh = g["_prof_rank"](raw["gpa"].where(uses, raw["roe"]).where(
                parts["eligible"]), uses, sector)
            score_fresh = (w := strat.params)["w_prof"] * prof_fresh + w["w_safe"] * parts[
                "ranks"]["safe"]
            score_fresh = score_fresh.where(parts["eligible"]).rank(axis=1, pct=True)
            adj = _forward(p, h)
            idx = p.close.index
            dec = np.flatnonzero(g["decision_bars"](idx, "month"))
            first = int(np.searchsorted(idx, pd.Timestamp(te0, tz="UTC")))
            for i in dec:
                if i < first - 1 or i + h + 1 >= len(idx) or idx[i] > pd.Timestamp(te1, tz="UTC"):
                    continue
                y = adj.iloc[i]
                signals = {"score": parts["score"].iloc[i], "prof": parts["ranks"]["prof"].iloc[i],
                           "safe": parts["ranks"]["safe"].iloc[i],
                           "prof_hybrid_in_sector": hybrid.iloc[i],
                           "growth": growth.iloc[i], "prof_fresh": prof_fresh.iloc[i],
                           "score_fresh": score_fresh.iloc[i],
                           "prof_raw": parts["raw"]["prof"].where(parts["eligible"]).iloc[i]}
                rec = {"test": t, "date": idx[i]}
                for k, x in signals.items():
                    ok = x.notna() & y.notna()
                    if ok.sum() < 30:
                        continue
                    rec[f"{k} ic"] = float(x[ok].rank().corr(y[ok].rank()))
                    q = x[ok].rank(pct=True)
                    rec[f"{k} spread"] = float(y[ok][q > 0.9].mean() - y[ok][q <= 0.1].mean())
                recs.append(rec)
    df = pd.DataFrame(recs).set_index("date").sort_index()
    out = {}
    for col in [c for c in df.columns if c != "test"]:
        v = df[col].dropna()
        out[col] = {"mean": float(v.mean()), "t": float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))),
                    "months": int(len(v))}
    by = df.groupby(df.index.year)[["score ic", "prof ic", "safe ic", "score spread"]].mean()
    print(pd.DataFrame(out).T.round(4).to_string())
    print(by.round(4).to_string())
    return {"summary": out, "by_year": by, "frame": df}


def anatomy(overrides=None, tests=("T1",), hedge: bool = False) -> dict:
    """Leg-by-leg anatomy of one config: daily long and short P&L contributions (realized book
    weight at the previous close × the name's return), the mean dollar exposure and ex-ante beta
    of each side, and each side's mean weight by sector, test by test."""
    strat, g = _helpers(overrides)
    settings = lab._settings()
    out = {}
    for t in tests:
        legs, sectors, expo = [], [], []
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            if hedge:
                p = lab.with_spy(p)
            sig = strat.signals(p, **strat.params)
            res = lab.book(strat, p, settings, te0, signals=sig)
            pf, w = res["pf"], res["window"]
            held = pf.asset_value(group_by=False).div(pf.value(), axis=0).shift(1).fillna(0.0)
            rets = w.close.pct_change(fill_method=None).reindex(held.index)[held.columns]
            c = (held * rets.fillna(0.0))
            legs.append(pd.DataFrame({"long": c.where(held > 0, 0.0).sum(axis=1),
                                      "short": c.where(held < 0, 0.0).sum(axis=1),
                                      "book": res["returns"], "market": res["bench"]}))
            parts = g["compose"](p, **{k: strat.params[k] for k in (
                "w_prof", "w_safe", "vol_window", "safety", "safe_by_sector", "min_price",
                "beta_window")})
            beta = parts["beta"].reindex(index=held.index, columns=held.columns)
            expo.append(pd.DataFrame({"long $": held.clip(lower=0).sum(axis=1),
                                      "short $": -held.clip(upper=0).sum(axis=1),
                                      "long beta": (held.clip(lower=0) * beta).sum(axis=1),
                                      "short beta": (held.clip(upper=0) * beta).sum(axis=1)}))
            sec = p.features["sector"].iloc[-1].map(SECTORS).fillna("none")
            sectors.append(pd.DataFrame({
                "long": held.clip(lower=0).mean().groupby(sec.reindex(held.columns)).sum(),
                "short": held.clip(upper=0).mean().groupby(sec.reindex(held.columns)).sum()}))
        lg = pd.concat(legs)
        year = lg.index.year
        by_year = lg.groupby(year).sum()
        # each leg net of its own realized beta to the market (daily regression over the test)
        betas = {k: float(lg[k].cov(lg["market"]) / lg["market"].var()) for k in ("long", "short")}
        resid = pd.DataFrame({f"{k} net of beta": lg[k] - b * lg["market"]
                              for k, b in betas.items()})
        years = lab.years_of(lg.index)
        out[t] = {"by_year": by_year, "exposure": pd.concat(expo).mean(),
                  "sectors": sum(sectors) / len(sectors), "daily": lg, "leg betas": betas,
                  "resid_by_year": resid.groupby(year).sum(),
                  "resid_per_year": (resid.sum() / years).to_dict()}
        with pd.option_context("display.width", 200):
            print(t, "leg contributions by year (sum of daily, ~ return points):")
            print((by_year * 100).round(1).to_string())
            print(out[t]["exposure"].round(3).to_string())
            print((out[t]["sectors"] * 100).round(1).to_string())
            print(t, "realized leg betas:", {k: round(v, 3) for k, v in betas.items()})
            print((out[t]["resid_by_year"] * 100).round(1).to_string())
            print(t, "net of beta, points a year:",
                  {k: round(v * 100, 2) for k, v in out[t]["resid_per_year"].items()})
    return out


# --- 2026-10-23 review fixes (RESEARCH_LOG.md §12) ---------------------------------------------

#: The top 500 is ONE continuous book from 2016-07 to 2026-10; names 501-1000 start in 2022-01.
CHAINS = (("E0", "TB", "T3", "T1"), ("T2",))
FIX = lab.RESULTS / NAME / "fix"


def _starved(pf, weights: pd.DataFrame) -> tuple[int, int, float]:
    """``(new shorts, starved, mean short $)`` of one book: the orders that open a short (or
    flip a long short), how many of them sit under 80% of their target at the fill bar's close
    (vectorbt rejected or part-filled them for want of free cash under ``lock_cash``), and the
    book's mean realized short exposure (a share of its value)."""
    from trading_simple import backtest as bt
    tg = bt.rotation_targets(weights).to_numpy(dtype=float)
    prev = pd.DataFrame(tg).ffill().shift(1).fillna(0.0).to_numpy()
    new = ~np.isnan(tg) & (tg < 0) & (prev >= 0)
    real = pf.asset_value(group_by=False).div(pf.value(), axis=0).to_numpy()
    short = new & (real > 0.8 * np.nan_to_num(tg))      # tg < 0: less than 80% of it filled
    return int(new.sum()), int(short.sum()), float(-np.clip(real, None, 0.0).sum(axis=1).mean())


def _end_weights(pf) -> pd.Series:
    """Each name's realized weight in a book at its last close (a short < 0)."""
    return pf.asset_value(group_by=False).iloc[-1] / float(pf.value().iloc[-1])


def _crossing(new: pd.Series, old: pd.Series | None) -> tuple[float, float, float, int, float]:
    """What ONE continuous book trades at a half-year boundary, as shares of the book:
    ``(entering, leaving, kept, orders, gone)`` — the new book's weight in names it did not hold
    on that side, the old book's weight in names it no longer holds on that side, the new weight
    carried over free (held on the same side across the boundary), the orders (a side change is
    two), and the part of ``leaving`` in names that left the universe (``new``'s index), which
    any book must sell. ``old`` None (a chain's first period): the whole book is built."""
    universe, new = new.index, new[new != 0]
    if old is None:
        return float(new.abs().sum()), 0.0, 0.0, int(len(new)), 0.0
    names = new.index.union(old.index)
    a, b = new.reindex(names).fillna(0.0), old.reindex(names).fillna(0.0)
    same = (np.sign(a) == np.sign(b)) & (a != 0)
    enter, leave = (a != 0) & ~same, (b != 0) & ~same
    gone = leave & ~names.isin(universe)
    return (float(a[enter].abs().sum()), float(b[leave].abs().sum()),
            float(a[same].abs().sum()), int(enter.sum() + leave.sum()), float(b[gone].abs().sum()))


def boundary(label: str = "U0v2", tests=("E0", "TB", "T3", "T1", "T2")) -> dict:
    """Re-read a saved label with each half-year's first day charged as one continuous book.

    Review R1 (round 1) found that lab v1 measured returns from ``value[0]``, dropping every
    period's first day — the setup orders' fees, slippage and late-fill surcharge, the borrow the
    engine charges up front for shorts held to the period end, the first open-to-close move —
    for the book AND the benchmark. Lab v2 counts that day from the initial cash, which charges a
    full 40-name (and 500-name benchmark) rebuild every six months: conservative, since one
    continuous book would trade only what changes at the boundary. This re-runs the label's books
    (base, +50 bp late fills, 2× costs; checked bit-exact against the saved returns) and stitches
    each record several ways (:data:`READINGS`): ``lab`` (as saved; on v2 the full rebuild),
    ``full`` (the same rebuilt by hand, a check), ``boundary`` (the first day charges only what
    ONE continuous book trades at the boundary — :func:`_crossing`; the benchmark its universe
    change and re-weighting to 1/N — plus the day-0 borrow, the first day's move omitted for both
    as lab v1 did), ``boundary+move`` (the same with the first day's move: the continuous-book
    counterpart of the v2 reading). It also counts NEW shorts the engine's own runner
    (``call_seq="auto"``, lab v1) left short of their target against the lab v2 runner
    (``starved``: realized size at the fill bar's close under 80% of the target). Chains:
    :data:`CHAINS`. Never run on a locked test.
    """
    from trading_simple import backtest as bt
    assert not set(tests) & set(lab.LOCKED), "boundary() never reads a locked test"
    saved = lab.load_results(NAME, label)
    strat, g = _helpers(saved["meta"]["overrides"])
    settings = lab._settings()
    init = settings.backtest.init_cash
    cfg = {"base": settings, "late": lab.with_costs(settings, 1.0, 50.0),
           "2x": lab.with_costs(settings, 2.0)}
    saved_key = {"base": "returns", "late": "late", "2x": "2x costs"}
    recs, worst = {t: [] for t in tests}, 0.0
    for chain in CHAINS:
        prev = prev_b = None
        for t in [t for t in chain if t in tests]:
            for i, (te0, _) in enumerate(lab.periods(t)):
                p = lab.panel(t, te0, lab.strategy_features(strat))
                sig = strat.signals(p, **strat.params)
                win, wsig = bt._score_window(p, sig, te0)
                w0 = wsig["weights"].iloc[0]
                rec = {"period": te0}
                for k, s in cfg.items():
                    res = lab.book(strat, p, s, te0, signals=sig)
                    fees, slip = bt.strategy_costs(strat, win, s)
                    total, curve = bt.benchmark(win, fees=fees, slippage=slip, fill=strat.fill)
                    if saved_key[k] in saved[t][i]:
                        old = saved[t][i][saved_key[k]].reindex(res["returns"].index)
                        worst = max(worst, float(np.nanmax(np.abs(res["returns"] - old))))
                    rec[k] = {"returns": res["returns"], "bench": res["bench"], "c": fees + slip,
                              "r0": float(res["pf"].value().iloc[0]) / init - 1.0,
                              "b0": (1.0 + total) / float((1.0 + curve).prod()) - 1.0}
                    if k == "base":
                        pf, fees0, slip0 = res["pf"], fees, slip
                # the borrow charged on day-0 orders: the same book without its borrow frame
                nob = {k: v for k, v in wsig.items() if k != "borrow"}
                pf_nb = lab.engine_run(strat, win, nob, settings)[0]
                borrow0 = (float(pf_nb.value().iloc[0]) - float(pf.value().iloc[0])) / init
                # new shorts left under target: the lab v2 runner vs the engine's own (lab v1)
                pf_v1 = bt.run(win, wsig, fees=fees0, slippage=slip0, init_cash=init,
                               fill=strat.fill, whole_shares=strat.whole_shares)
                starved = {k: _starved(x, wsig["weights"]) for k, x in (("v2", pf), ("v1", pf_v1))}
                # the benchmark rebuilt exactly as bt.benchmark builds it, for its end weights
                wb = pd.DataFrame(1.0 / win.close.shape[1], index=win.close.index,
                                  columns=win.close.columns).where(win.close.notna())
                pf_b = bt.run(win, {"weights": wb}, fees=fees0, slippage=slip0, fill=strat.fill)
                wb0 = wb.iloc[0].fillna(0.0)
                enter, leave, kept, n_cross, gone = _crossing(w0, prev)
                # a regular month-end rebalance inside the window: traded weight per rebalance
                orders, value = pf.orders.records, pf.value()
                rot = orders[orders["idx"] > 0]
                traded = float((np.abs(rot["size"] * rot["price"])
                                / value.to_numpy()[rot["idx"]]).sum())
                d = g["decision_bars"](p.close.index, strat.params["rebalance"])
                n_reb = int((d[:-1] & (p.close.index[:-1] >= win.close.index[0])).sum())
                regular = traded / max(n_reb, 1)
                low = enter if prev is None else gone + regular
                if prev_b is None:
                    turn_b = float(wb0.sum())
                else:
                    u = wb0.index.union(prev_b.index)
                    turn_b = float((wb0.reindex(u).fillna(0.0)
                                    - prev_b.reindex(u).fillna(0.0)).abs().sum())
                move = float((w0 * (win.close.iloc[0] / win.open_.iloc[0] - 1.0)).sum())
                for k in cfg:
                    c = rec[k]["c"]
                    rec[k]["boundary0"] = -c * (enter + leave) - borrow0
                    rec[k]["boundary_b0"] = -c * turn_b
                    rec[k]["move0"] = rec[k]["r0"] + c * (kept - leave)
                    rec[k]["move_b0"] = rec[k]["b0"] + c * (float(wb0.sum()) - turn_b)
                    rec[k]["low0"] = -c * low - borrow0
                gross = pf.asset_value(group_by=False).abs().sum(axis=1) / pf.value()
                rec.update({
                    "setup": float(w0.abs().sum()), "enter": enter, "leave": leave, "kept": kept,
                    "bench setup": float(wb0.sum()), "bench turnover": turn_b,
                    "borrow0": borrow0, "orders": int(len(orders)),
                    "first-bar orders": int((orders["idx"] == 0).sum()),
                    "crossing orders": n_cross, "gone": gone, "regular": regular, "low": low,
                    "rebalances": n_reb,
                    "cost check": (move - rec["base"]["r0"] - borrow0)
                    / (rec["base"]["c"] * float(w0.abs().sum())),
                    "gross max": float(gross.max()), "gross mean": float(gross.mean()),
                    "nominal gross max": float(wsig["weights"].abs().sum(axis=1).max()),
                    "years": lab.years_of(win.close.index),
                    "new shorts": starved["v2"][0], "starved v2": starved["v2"][1],
                    "starved v1": starved["v1"][1], "short $ v2": starved["v2"][2],
                    "short $ v1": starved["v1"][2]})
                recs[t].append(rec)
                prev, prev_b = _end_weights(pf), _end_weights(pf_b)
                lab._PANELS.clear()                 # one panel in memory at a time
                print(f"{t} {te0}: setup {rec['setup']:.3f} enter {enter:.3f} leave {leave:.3f} "
                      f"kept {kept:.3f} gone {gone:.3f} regular {regular:.3f} | bench turnover "
                      f"{turn_b:.3f} | borrow0 {borrow0:+.5f} "
                      f"| orders {rec['orders']} (first bar {rec['first-bar orders']}, crossing "
                      f"{n_cross}) | cost check {rec['cost check']:.3f} | new shorts "
                      f"{rec['new shorts']}, starved v2 {rec['starved v2']} / v1 "
                      f"{rec['starved v1']}", flush=True)
    print(f"max |daily return - saved| over every book: {worst:.2e}")
    FIX.mkdir(parents=True, exist_ok=True)
    (FIX / f"boundary-{label}.pkl").write_bytes(pickle.dumps(recs))
    return _boundary_report(recs, label)


READINGS = {"lab": None, "full": ("r0", "b0"), "boundary": ("boundary0", "boundary_b0"),
            "boundary+move": ("move0", "move_b0"), "low": ("low0", "boundary_b0")}


def _restored(rows, setting: str, reading: str):
    """``(book, buy-and-hold)`` daily pairs of one setting under one reading
    (:data:`READINGS`: the first-day fields of the book and of the benchmark). The benchmark is
    always the base-cost one, as ``lab.verdict`` stitches the late and 2× books (the market does
    not pay the strategy's stress)."""
    first = READINGS[reading]
    pairs = []
    for rec in rows:
        r, b = rec[setting]["returns"].copy(), rec["base"]["bench"].copy()
        if first:
            r.iloc[0], b.iloc[0] = rec[setting][first[0]], rec["base"][first[1]]
        pairs.append((r, b))
    return pairs


def _boundary_report(recs: dict, label: str) -> dict:
    """Alpha (P) per test and reading, the gates the first day can move, turnover facts."""
    groups = {t: [t] for t in recs if recs[t]}
    if all(t in groups for t in lab.TRAIN):
        groups = {"TRAIN": list(lab.TRAIN), **groups}
    table, facts, out = {}, {}, {}
    for name, ts in groups.items():
        rows = [r for t in ts for r in recs[t]]
        for reading in READINGS:
            col = {}
            for k in ("base", "late", "2x"):
                st = lab.stitched(_restored(rows, k, reading))
                col[k] = round(st["alpha"] * 100, 2)
                if k == "base":
                    col["P"], col["beta"] = round(st["P(alpha > 0)"], 3), round(st["beta"], 3)
            table[(name, reading)] = col
            out[(name, reading)] = col
        years = sum(r["years"] for r in rows)
        boundary = sum(r["enter"] + r["leave"] for r in rows)
        facts[name] = {
            "boundary / full setup": sum(r["enter"] + r["leave"] for r in rows)
            / sum(r["setup"] for r in rows),
            "bench boundary / setup": sum(r["bench turnover"] for r in rows)
            / sum(r["bench setup"] for r in rows),
            "orders/yr": sum(r["orders"] for r in rows) / years,
            "first-bar/yr": sum(r["first-bar orders"] for r in rows) / years,
            "rotation/yr": sum(r["orders"] - r["first-bar orders"] for r in rows) / years,
            "crossing/yr": sum(r["crossing orders"] for r in rows) / years,
            "gone / boundary": sum(r["gone"] for r in rows) / max(boundary, 1e-12),
            "regular rebalance / setup": float(np.mean([r["regular"] / r["setup"]
                                                        for r in rows])),
            "nominal gross max": max(r["nominal gross max"] for r in rows),
            "day-0 borrow pts/yr": sum(r["borrow0"] for r in rows) / years * 100,
            "gross max": max(r["gross max"] for r in rows),
            "gross mean": float(np.mean([r["gross mean"] for r in rows])),
            "cost check": float(np.mean([r["cost check"] for r in rows])),
            "new shorts": sum(r.get("new shorts", 0) for r in rows),
            "starved v2": sum(r.get("starved v2", 0) for r in rows),
            "starved v1 runner": sum(r.get("starved v1", 0) for r in rows),
            "short $ v2": float(np.mean([r.get("short $ v2", np.nan) for r in rows])),
            "short $ v1 runner": float(np.mean([r.get("short $ v1", np.nan) for r in rows]))}
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(f"{label}: stitched alpha (%/yr) by first-day reading; benchmark at base costs")
        print(pd.DataFrame(table).T.to_string())
        print(pd.DataFrame(facts).round(3).to_string())
    if "T1" in recs and recs["T1"]:
        for reading in READINGS:
            late = lab.stitched(_restored(recs["T1"], "late", reading))["alpha"]
            two = lab.stitched(_restored(recs["T1"], "2x", reading))["alpha"]
            print(f"G5 ({reading}): T1 late {late:+.4f}, 2x {two:+.4f} -> "
                  f"{'pass' if late > 0 and two > 0 else 'FAIL'}")
    top = [r for t in CHAINS[0] if t in recs for r in recs[t]]
    if top:
        for reading in ("lab", "boundary", "boundary+move"):
            fake = {t: [{"returns": r, "bench": b} for r, b in _restored(recs[t], "base", reading)]
                    for t in CHAINS[0] if t in recs}
            ep = {k: (round(v["book"] * 100, 1), round(v["market"] * 100, 1))
                  for k, v in lab.episodes(fake).items()}
            dm = lab.down_months(fake)
            print(f"G6 ({reading}): episodes {ep}; down-month mean {dm.get('book mean', 0):+.4f}")
    return out


def boundary_report(label: str = "U0v2") -> dict:
    """Re-print :func:`boundary`'s report from its saved records (no book is re-run)."""
    return _boundary_report(pickle.loads((FIX / f"boundary-{label}.pkl").read_bytes()), label)


def v1v2(v1: str = "U0", v2: str = "U0v2", engine: bool = False,
         tests=("E0", "TB", "T3", "T1", "T2")) -> pd.DataFrame:
    """Where a label's numbers moved between lab v1 and lab v2 (round 2, RESEARCH_LOG.md §13).

    Readings, each stitched per test (alpha %/yr): ``v1`` (the saved lab-v1 label: v1 panels,
    the engine's runner, day 0 dropped); ``engine`` (``engine=True`` only, re-run: the v2 panels
    with the ENGINE's runner — ``call_seq="auto"``, the engine's borrow fees — and day 0
    dropped: what the v1 code makes of the rebuilt panels); ``v2 day0 dropped`` (the saved v2
    returns with each period's first day set to 0 for the book and the benchmark, as lab v1
    measured them); ``v2`` (as saved). v1 → engine is the panels (NYSE sessions only, ETFs out,
    the current DB), engine → dropped the lab's runner, dropped → v2 the first day."""
    from trading_simple import backtest as bt
    assert not set(tests) & set(lab.LOCKED), "v1v2() never reads a locked test"
    old = pickle.loads((lab.VAR / "results" / NAME / f"{v1}.pkl").read_bytes())
    new = lab.load_results(NAME, v2)
    strat = lab.load_strategy(NAME, new["meta"]["overrides"]) if engine else None
    settings = lab._settings()
    rows, per = {}, []
    for t in tests:
        pairs = {"v1": [(r["returns"], r["bench"]) for r in old[t]], "v2": [],
                 "v2 day0 dropped": []}
        if engine:
            pairs["engine"] = []
        for i, rec in enumerate(new[t]):
            r, b = rec["returns"], rec["bench"]
            pairs["v2"].append((r, b))
            r0, b0 = r.copy(), b.copy()
            r0.iloc[0] = b0.iloc[0] = 0.0
            pairs["v2 day0 dropped"].append((r0, b0))
            line = {"test": t, "period": rec["period"],
                    "v1": lab.period_alpha(old[t][i]["returns"], old[t][i]["bench"]),
                    "v2 day0 dropped": lab.period_alpha(r0, b0), "v2": lab.period_alpha(r, b)}
            if engine:
                te0 = rec["period"]
                p = lab.panel(t, te0, lab.strategy_features(strat))
                sig = strat.signals(p, **strat.params)
                win, wsig = bt._score_window(p, sig, te0)
                fees, slip = bt.strategy_costs(strat, win, settings)
                pf = bt.run(win, wsig, fees=fees, slippage=slip,
                            init_cash=settings.backtest.init_cash, fill=strat.fill,
                            whole_shares=strat.whole_shares)
                re = bt.equity_curve(pf)["value"].pct_change().fillna(0.0)
                _, braw = bt.benchmark(win, fees=fees, slippage=slip, fill=strat.fill)
                pairs["engine"].append((re, braw.reindex(re.index).fillna(0.0)))
                line["engine"] = lab.period_alpha(*pairs["engine"][-1])
                lab._PANELS.clear()
            per.append(line)
        rows[t] = {k: lab.stitched(v)["alpha"] * 100 for k, v in pairs.items()}
    order = ["v1", "engine", "v2 day0 dropped", "v2"] if engine else ["v1", "v2 day0 dropped",
                                                                       "v2"]
    out = pd.DataFrame(rows).T[order]
    per = pd.DataFrame(per).set_index(["test", "period"])[order] * 100
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(f"{v1} -> {v2}: stitched alpha, %/yr")
        print(out.round(2).to_string())
        print("per period, %/yr")
        print(per.round(1).to_string())
    return out


def bear(labels=("U0v2", "I1v2", "I4v2")) -> dict:
    """Split each label's falling-market record into what its beta implies and the rest (R2).

    For every episode (and the market's down months, E0 + TB + T3 + T1 stitched): the book's
    return, the market's, the book's own beta inside the episode (daily regression), the
    beta-implied part (the test's realized beta × the market's return) and the rest."""
    tops = ("E0", "TB", "T3", "T1")
    out = {}
    for name in labels:
        res = lab.load_results(NAME, name)
        beta_of = {t: lab._stitch(res[t])["beta"] for t in tops if t in res}
        r = pd.concat([x["returns"] for t in tops for x in res[t]])
        b = pd.concat([x["bench"] for t in tops for x in res[t]]).reindex(r.index).fillna(0.0)
        test_of = pd.Series([t for t in tops for x in res[t] for _ in range(len(x["returns"]))],
                            index=r.index)
        rows = {}
        for k, (a, z) in lab.EPISODES.items():
            m = (r.index >= pd.Timestamp(a, tz="UTC")) & (r.index <= pd.Timestamp(z, tz="UTC"))
            if m.sum() < 5:
                continue
            book, mkt = float((1 + r[m]).prod() - 1), float((1 + b[m]).prod() - 1)
            beta_t = beta_of[test_of[m].iloc[0]]
            inside = float(r[m].cov(b[m]) / b[m].var())
            rows[k] = {"book": book, "market": mkt, "episode beta": inside,
                       "test beta": beta_t, "beta part": beta_t * mkt,
                       "rest": book - beta_t * mkt}
        bk, mk = lab.monthly([x for t in tops for x in res[t]])
        down = mk < 0
        beta_all = lab.stitched([(x["returns"], x["bench"]) for t in tops for x in res[t]])["beta"]
        rows["down months"] = {"book": float(bk[down].mean()), "market": float(mk[down].mean()),
                               "episode beta": np.nan, "test beta": beta_all,
                               "beta part": beta_all * float(mk[down].mean()),
                               "rest": float(bk[down].mean()) - beta_all * float(mk[down].mean())}
        out[name] = pd.DataFrame(rows).T
        print(f"== {name} (book, market, beta parts in %; betas raw)")
        frame = out[name].copy()
        for c in ("book", "market", "beta part", "rest"):
            frame[c] = frame[c] * 100
        print(frame.round(2).to_string())
    return out


def _age_since_change(f: pd.DataFrame) -> pd.DataFrame:
    """Calendar days since each cell's value last changed (NaN where the value is NaN)."""
    days = np.asarray(f.index.tz_convert(None).normalize().asi8 // 86_400_000_000_000, float)
    d = pd.DataFrame(np.repeat(days[:, None], f.shape[1], axis=1), index=f.index,
                     columns=f.columns)
    changed = f.ne(f.shift()) & f.notna()
    return (d - d.where(changed).ffill()).where(f.notna())


_COMPOSE = ("w_prof", "w_safe", "vol_window", "safety", "safe_by_sector", "w_growth",
            "growth_window", "min_price", "beta_window", "short_max_si", "si_mid", "si_high",
            "borrow_gc", "borrow_mid", "borrow_high")


def _book_parts(strat, g, p):
    """``compose`` output, decision bars and U0's held frames for one panel."""
    P = strat.params
    parts = g["compose"](p, **{k: P[k] for k in _COMPOSE})
    decide = g["decision_bars"](p.close.index, P["rebalance"])
    longs, shorts, _ = g["_select"](parts["score"], p.close, parts["shortable"], decide,
                                    q_in=P["q_in"], q_out=P["q_out"], slots=P["slots"],
                                    short_stop=P["short_stop"], sides=P["sides"])
    return parts, decide, longs, shorts


def staleness(tests=("E0", "TB", "T3", "T1", "T2"), stale: int = 200, fresh: int = 150):
    """Frozen gross profit (review R7): on every month-end decision bar of every OPEN window, the
    eligible names measured on gross profit / assets whose GP TTM level has not changed for more
    than ``stale`` days, how many have a fresh ROE instead (net income TTM changed within
    ``fresh`` days, equity > 0), and how many of the names the book holds after that decision."""
    strat, g = _helpers()
    recs = []
    for t in tests:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            parts, decide, longs, shorts = _book_parts(strat, g, p)
            gp_age = _age_since_change(p.features["fund_gp_ttm"])
            ni_age = _age_since_change(p.features["fund_ni_ttm"])
            idx = p.close.index
            inside = decide & (idx >= pd.Timestamp(te0, tz="UTC")) & (
                idx <= pd.Timestamp(te1, tz="UTC"))
            for i in np.flatnonzero(inside):
                el, m = parts["eligible"].iloc[i], parts["measure"].iloc[i]
                on_gp = el & (m == 0)
                old = on_gp & (gp_age.iloc[i] > stale)
                roe_ok = (ni_age.iloc[i] <= fresh) & (p.features["fund_equity"].iloc[i] > 0)
                held = longs.iloc[i] | shorts.iloc[i]
                recs.append({"test": t, "date": idx[i], "on gp": int(on_gp.sum()),
                             "stale": int(old.sum()), "stale with fresh roe":
                             int((old & roe_ok).sum()), "held": int(held.sum()),
                             "held stale": int((old & held).sum()),
                             "max age": float(gp_age.iloc[i][on_gp].max())})
            lab._PANELS.clear()
    df = pd.DataFrame(recs)
    agg = df.groupby("test", sort=False).agg(
        bars=("date", "size"), on_gp=("on gp", "mean"), stale=("stale", "mean"),
        stale_fresh_roe=("stale with fresh roe", "mean"), held=("held", "mean"),
        held_stale_mean=("held stale", "mean"), held_stale_max=("held stale", "max"),
        max_age=("max age", "max"))
    agg["stale share"] = agg["stale"] / agg["on_gp"]
    print(agg.round(3).to_string())
    return df


def floor(tests=("E0", "TB", "T3", "T1", "T2")):
    """The $5 floors on split-adjusted closes (review R5): per OPEN period, the names the
    strategy's ``min_price`` removes on its decision bars (a profitability ratio and a volatility,
    but an adjusted close under the floor), and for the top-500 tests the names the lab's universe
    filter drops only for a median adjusted price under $5 (ranked from the lab's cache)."""
    strat, g = _helpers()
    rows = []
    for t in tests:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            parts, decide, _, _ = _book_parts(strat, g, p)
            idx = p.close.index
            inside = decide & (idx >= pd.Timestamp(te0, tz="UTC")) & (
                idx <= pd.Timestamp(te1, tz="UTC"))
            ok = parts["raw"]["prof"].notna() & parts["raw"]["vol"].notna()
            cut = ok & (p.close < strat.params["min_price"])
            removed = cut[inside]
            row = {"test": t, "period": te0, "removed per decision": float(removed.sum(axis=1)
                                                                          .mean()),
                   "names": sorted(removed.columns[removed.any()])}
            if lab.TESTS[t][0] == "top":
                r = lab.ranked(te0)
                pool = r[(r["days"] >= lab.ROLL["min_days"])
                         & (r["dv"] >= lab.ROLL["min_dollar_volume"])]
                top = pool.nlargest(lab.ROLL["top_n"], "dv")
                row["lab floor drops"] = sorted(top.index[top["price"] < lab.ROLL["min_price"]])
            rows.append(row)
            lab._PANELS.clear()
            print(row, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "anatomy":
        anatomy(json.loads(a[1]) if len(a) > 1 else None,
                tuple(a[2].split(",")) if len(a) > 2 else ("T1",), hedge="--hedge" in a)
    elif a[0] == "coverage":
        coverage(tuple(a[1].split(",")) if len(a) > 1 else ("E0", "TB", "T3", "T1", "T2"))
    elif a[0] == "premise":
        premise(json.loads(a[1]) if len(a) > 1 else None,
                tuple(a[2].split(",")) if len(a) > 2 else ("E0", "TB", "T3"))
    elif a[0] == "boundary":
        boundary(a[1] if len(a) > 1 else "U0v2",
                 tuple(a[2].split(",")) if len(a) > 2 else ("E0", "TB", "T3", "T1", "T2"))
    elif a[0] == "v1v2":
        v1v2(a[1] if len(a) > 1 else "U0", a[2] if len(a) > 2 else "U0v2",
             engine="--engine" in a)
    elif a[0] == "boundary-report":
        boundary_report(a[1] if len(a) > 1 else "U0v2")
    elif a[0] == "bear":
        bear(tuple(a[1:]) or ("U0v2", "I1v2", "I4v2"))
    elif a[0] in ("staleness", "floor"):
        {"staleness": staleness, "floor": floor}[a[0]](
            tuple(a[1].split(",")) if len(a) > 1 else ("E0", "TB", "T3", "T1", "T2"))
    else:
        raise SystemExit(__doc__)
