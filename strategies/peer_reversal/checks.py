"""Review checks for peer_reversal (RESEARCH_LOG §13 and Round 2): the numbers behind the
review fixes and the lab-v2 re-evaluation.

Light reads of the saved lab results (``lab.RESULTS``: ``var/proposals_lab/results2/`` under lab
v2; the round-1 labels stay in ``results/``, read by ``v1v2`` and ``decompose``), the round-1
tuner's journal, and four heavier audits (``curve``, ``noise-luck``, ``fills``, ``decompose``).
Nothing here changes a saved label; the heavier audits cache to ``<lab.RESULTS>/peer_reversal/
fix/``. Run from the repo root:

    uv run python strategies/peer_reversal/checks.py stress U0v2 C2v2 I4v2  # P&L by stress state
    uv run python strategies/peer_reversal/checks.py noise U0v2 I4v2        # adoption Δ per copy
    uv run python strategies/peer_reversal/checks.py noise-luck U0v2        # copies, own band
    uv run python strategies/peer_reversal/checks.py curve '{}'             # reversal curve t
    uv run python strategies/peer_reversal/checks.py tuner                  # round-1 T surface
    uv run python strategies/peer_reversal/checks.py exposure U0v2 C2v2     # realized exposure
    uv run python strategies/peer_reversal/checks.py fills I4v2 T1          # under-filled orders
    uv run python strategies/peer_reversal/checks.py pairs I1v2 I4v2        # paired Δ, any base
    uv run python strategies/peer_reversal/checks.py v1v2 U0 I4             # round 1 vs round 2
    uv run python strategies/peer_reversal/checks.py decompose U0           # what each fix did
    uv run python strategies/peer_reversal/checks.py attribute U0 I4        # ... to an adoption
    uv run python strategies/peer_reversal/checks.py period I4 T1 2026-01-01  # one period, v1/v2
    uv run python strategies/peer_reversal/checks.py borrow U0v2            # hair re-trims?
    uv run python strategies/peer_reversal/checks.py holidays U0 I4         # the 2026 rows

Every t-statistic is Newey-West (Bartlett weights): daily book returns with 5 lags, per
entry-date trade means with ``k`` lags at curve day ``k``.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import pickle
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

REPO = "/Users/janmarti/Documents/code/stock-trading-simple"
KEY = "peer_reversal"
GROUPS = {"TRAIN": ["E0", "TB", "T3"], "T1": ["T1"], "T2": ["T2"], "T4": ["T4"]}


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lab = _load("lab", f"{REPO}/strategies/proposals_lab/lab.py")
S = _load("peer_reversal_strategy", f"{REPO}/strategies/peer_reversal/strategy.py")
FIX = lab.RESULTS / KEY / "fix"
V1 = lab.VAR / "results" / KEY          # round 1, lab v1 (kept for the record)
COSTS = ("returns", "late", "2x costs")


def nw_var(x: np.ndarray, lags: int) -> float:
    """Newey-West long-run variance of a demeaned series (Bartlett weights)."""
    n = len(x)
    var = x @ x / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return float(var)


def nw_t(x, lags: int = 5) -> float:
    """Newey-West t-statistic of the mean of ``x`` (NaNs dropped)."""
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) < 3:
        return float("nan")
    return float(x.mean() / np.sqrt(nw_var(x - x.mean(), lags) / len(x)))


def nw_diff_t(y, flag, lags: int = 5) -> float:
    """t of ``mean(y | flag) - mean(y | not flag)`` (OLS on a dummy, Newey-West errors)."""
    y, f = np.asarray(y, float), np.asarray(flag, float)
    ok = np.isfinite(y)
    y, f = y[ok], f[ok]
    x = np.column_stack([np.ones_like(f), f])
    xtx_inv = np.linalg.inv(x.T @ x)
    b = xtx_inv @ x.T @ y
    g = x * (y - x @ b)[:, None]
    s = g.T @ g
    for k in range(1, min(lags, len(y) - 1) + 1):
        c = g[k:].T @ g[:-k]
        s += (1 - k / (lags + 1)) * (c + c.T)
    return float(b[1] / np.sqrt((xtx_inv @ s @ xtx_inv)[1, 1]))


def _stitch(res: dict, group: str, key: str = "returns") -> dict:
    return lab.stitched([(r[key], r["bench"]) for t in GROUPS[group] if t in res
                         for r in res[t]])


def _compose(p, params: dict, **over) -> dict:
    keys = inspect.signature(S.compose).parameters
    return S.compose(p, **{k: v for k, v in {**params, **over}.items() if k in keys})


def stress_flags(group: str) -> pd.Series:
    """Stress state at the PREVIOUS close (the state the day's fills were decided in), per
    day of the group's periods, each period read on its own panel (peers do not matter)."""
    params = lab.load_strategy(KEY).params
    out = []
    for t in GROUPS[group]:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, [])
            st = _compose(p, params, peers="market")["stress"].shift(1, fill_value=False)
            idx = st.index
            out.append(st[(idx >= pd.Timestamp(te0, tz="UTC")) & (idx <= pd.Timestamp(te1,
                                                                                   tz="UTC"))])
    s = pd.concat(out)
    return s[~s.index.duplicated()]


def stress(labels) -> None:
    """Market-adjusted P&L (book − beta × market) split by the stress state at the previous
    close: %/yr of all days and its t, bp/day on each regime and the t of their difference."""
    flags = {g: stress_flags(g) for g in ("TRAIN", "T1", "T2")}
    rows = []
    for label in labels:
        res = lab.load_results(KEY, label)
        for g, fl in flags.items():
            s = _stitch(res, g)
            resid = s["returns"] - s["beta"] * s["bench"]
            on = fl.reindex(resid.index).fillna(False).to_numpy(dtype=bool)
            y = resid.to_numpy(dtype=float)
            rows.append({"label": label, "test": g, "stress days": f"{on.mean():.0%}",
                         "stress %/yr": 100 * y[on].sum() / s["years"],
                         "t(stress)": nw_t(y * on), "calm %/yr": 100 * y[~on].sum() / s["years"],
                         "t(calm)": nw_t(y * ~on), "stress bp/day": 1e4 * y[on].mean(),
                         "calm bp/day": 1e4 * y[~on].mean(), "t(diff)": nw_diff_t(y, on)})
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


def _paired(a: dict, b: dict) -> dict:
    return lab.paired(a["returns"], b["returns"].reindex(a["returns"].index).fillna(0.0),
                      b["bench"].reindex(a["returns"].index).fillna(0.0), b["years"])


def pairs(base: str, label: str) -> None:
    """``label`` minus ``base``: paired Δ alpha (P) on TRAIN (stitched), T1, T2, T4."""
    a, b = lab.load_results(KEY, label), lab.load_results(KEY, base)
    for g in GROUPS:
        if all(any(t in r for t in GROUPS[g]) for r in (a, b)):
            pr = _paired(_stitch(a, g), _stitch(b, g))
            print(f"{label} - {base} {g}: Δ {100 * pr['delta alpha']:+.2f} pts "
                  f"(P {pr['P(delta > 0)']:.2f})")


def noise(base: str, label: str, copies: int = 3) -> None:
    """The adoption Δ (``label`` − ``base``) re-read on every noise copy (both copies drop the
    same names: same seeds, same columns), the bar each copy implies, and where the copies
    sit against their own label (sd as in ``lab.noise``, and the RMS distance)."""
    ra, rb = lab.load_results(KEY, label), lab.load_results(KEY, base)
    for name, res in ((base, rb), (label, ra)):
        print(f"\n{name}: copy − label per test (pts)")
        for t in lab.OPEN:
            d = np.array([lab._stitch(lab.load_results(KEY, f"{name}-noise5-{c}")[t])["alpha"]
                          - lab._stitch(res[t])["alpha"] for c in range(copies)])
            print(f"  {t}: {np.round(100 * d, 2).tolist()}  sd {100 * d.std(ddof=1):.2f}  "
                  f"rms {100 * np.sqrt(np.mean(d * d)):.2f}  below the label {int((d < 0).sum())}"
                  f"/{copies}")
    sd = np.std([lab._stitch(lab.load_results(KEY, f"{base}-noise5-{c}")["T1"])["alpha"]
                 - lab._stitch(rb["T1"])["alpha"] for c in range(copies)], ddof=1)
    print(f"\n{label} − {base}, paired (the T1 bar is 2 × {base}'s T1 noise sd = "
          f"{200 * sd:.2f} pts):")
    for c in ["label", *range(copies)]:
        a = ra if c == "label" else lab.load_results(KEY, f"{label}-noise5-{c}")
        b = rb if c == "label" else lab.load_results(KEY, f"{base}-noise5-{c}")
        d = {g: _paired(_stitch(a, g), _stitch(b, g)) for g in ("TRAIN", "T1", "T2")}
        ok = d["T1"]["delta alpha"] > 2 * sd and all(v["delta alpha"] >= 0 for v in d.values())
        print(f"  {c!s:6} " + ", ".join(f"{g} {100 * v['delta alpha']:+.2f} (P "
                                        f"{v['P(delta > 0)']:.2f})" for g, v in d.items())
              + f" -> passes the bar: {ok}")


def _dropped_panels(label: str, copy: int, k: int = 5, want: str = "T1"):
    """Replay ``lab.run``'s name drops for noise copy ``copy`` (seed 100 + copy, tests in
    OPEN order) and yield ``(te0, panel without the dropped names)`` for test ``want``."""
    strat = lab.load_strategy(KEY, lab.load_results(KEY, label)["meta"]["overrides"])
    rng = np.random.default_rng(100 + copy)
    for t in lab.OPEN:
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            keep = sorted(set(p.close.columns) - set(rng.choice(p.close.columns, k,
                                                                replace=False)))
            if t == want:
                yield te0, replace(p, close=p.close[keep], open_=p.open_[keep],
                                   high=p.high[keep], low=p.low[keep], volume=p.volume[keep],
                                   features={f: v[keep] for f, v in p.features.items()})
        if t == want:
            return


def noise_luck(label: str, copies: int = 3, n: int = 100, seed: int = 29) -> None:
    """Each noise copy against ITS OWN shuffled-name band: the luck test of ``lab.luck``
    (same seed, names permuted within dollar-volume tenths) run on the copy's panels."""
    strat = lab.load_strategy(KEY, lab.load_results(KEY, label)["meta"]["overrides"])
    settings = lab._settings()
    FIX.mkdir(parents=True, exist_ok=True)
    for c in range(copies):
        saved = lab.load_results(KEY, f"{label}-noise5-{c}")["T1"]
        cache = FIX / f"noise-luck-{label}-{c}.pkl"
        if cache.exists():
            books = pickle.loads(cache.read_bytes())
        else:
            rng, books = np.random.default_rng(seed), []
            for i, (te0, p) in enumerate(_dropped_panels(label, c)):
                sig = strat.signals(p, **strat.params)
                real = lab.book(strat, p, settings, te0, signals=sig)["returns"]
                gap = float(np.abs(real.to_numpy() - saved[i]["returns"].to_numpy()).max())
                assert gap < 1e-12, f"replayed copy {c} {te0} differs from the saved copy: {gap}"
                tenth = lab._tenths(p, te0).to_numpy()
                cols = p.close.columns.to_numpy()
                per = []
                for _ in range(n):
                    perm = cols.copy()
                    for g in np.unique(tenth):
                        inside = np.flatnonzero(tenth == g)
                        perm[inside] = cols[rng.permutation(inside)]
                    mapping = dict(zip(cols, perm, strict=True))
                    shuf = {k: (v.rename(columns=mapping)[v.columns]
                                if isinstance(v, pd.DataFrame) and k != "borrow" else v)
                            for k, v in sig.items()}
                    per.append(lab.book(strat, p, settings, te0, signals=shuf)["returns"])
                books.append(per)
                print(f"noise-luck {label} copy {c} {te0}: {n} books", flush=True)
            cache.write_bytes(pickle.dumps(books))
        bench = [r["bench"] for r in saved]
        alphas = np.array([lab.stitched([(books[i][k], bench[i]) for i in range(len(books))])
                           ["alpha"] for k in range(n)])
        real = lab._stitch(saved)["alpha"]
        print(f"{label} copy {c}: real {100 * real:+.2f}%, band median "
              f"{100 * np.median(alphas):+.2f}%, p95 {100 * np.percentile(alphas, 95):+.2f}%, "
              f"max {100 * alphas.max():+.2f}% -> percentile {(alphas < real).mean() * 100:.0f}",
              flush=True)


def curve(over: dict) -> None:
    """The anatomy's reversal curve (TRAIN, every entry of the config) with Newey-West t over
    entry dates, by side and stress state, and the stress − calm difference."""
    A = _load("peer_reversal_anatomy", f"{REPO}/strategies/peer_reversal/anatomy.py")
    params = lab.load_strategy(KEY, over).params
    parts = [A.trades(t, te0, params) for t in lab.TRAIN for te0, _ in lab.periods(t)]
    e = pd.concat([x for x, _ in parts], ignore_index=True)
    cv = pd.concat([y for _, y in parts], ignore_index=True)
    print(f"{over}: {len(e)} TRAIN trades, gross {1e4 * e['gross'].mean():+.1f} bp a trade "
          f"(t over entry dates {nw_t(e.groupby('entry')['gross'].mean(), 10):+.2f})")
    groups = {"all": e.index == e.index, "long": e["side"] > 0, "short": e["side"] < 0,
              "stress": e["stress"], "calm": ~e["stress"]}
    print("  per trade bp | per entry date bp (its t) — the t is the per-date mean's")
    for name, mask in groups.items():
        m = np.asarray(mask)
        if not m.any():
            continue
        out = []
        for k in (5, 10, 21):
            x = pd.Series(cv[k].to_numpy()[m], index=e.loc[m, "entry"].to_numpy())
            daily = x.groupby(level=0).mean().sort_index()
            out.append(f"day {k} {1e4 * x.mean():+6.1f} | {1e4 * daily.mean():+6.1f} "
                       f"(t {nw_t(daily, k):+.2f})")
        g = e.loc[m, "gross"].groupby(e.loc[m, "entry"]).mean()
        print(f"  {name:6} n {int(m.sum()):5}: " + "; ".join(out)
              + f"; held trade {1e4 * e.loc[m, 'gross'].mean():+.1f} | {1e4 * g.mean():+.1f} "
              f"(t {nw_t(g, 10):+.2f})")
    if e["stress"].all() or not e["stress"].any():
        return
    for k in (10, 21):
        x = pd.DataFrame({"y": cv[k].to_numpy(), "entry": e["entry"].to_numpy(),
                          "stress": e["stress"].to_numpy()}).groupby("entry").mean()
        print(f"  stress − calm at day {k} (per entry date): "
              f"{1e4 * (x.y[x.stress > 0.5].mean() - x.y[x.stress < 0.5].mean()):+.1f} bp "
              f"(t {nw_diff_t(x['y'].to_numpy(), x['stress'].to_numpy() > 0.5, k):+.2f})")


def tuner() -> None:
    """The T study: trials per formation bin, the 20 random start-up trials alone (TPE's
    adaptive sampling confounds the bin means), and the top 10."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    journal = lab.VAR / "studies" / f"{KEY}-T.log"            # round 1's study (lab v1)
    st = optuna.load_study(study_name=f"{KEY}-T", storage=optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal))))
    df = st.trials_dataframe(attrs=("number", "value", "params", "state"))
    df = df[df.state == "COMPLETE"]
    print(f"complete {len(df)}, robust > 0: {int((df.value > 0).sum())}")
    for lo, hi in ((3, 8), (9, 12), (13, 16), (17, 21)):
        m = df[(df.params_formation >= lo) & (df.params_formation <= hi)]
        r = m[m.number < 20]
        print(f"  formation {lo:2}-{hi:2}: {len(m):3} trials, mean robust "
              f"{100 * m.value.mean():+.2f}%; random start-up trials {len(r)}, mean "
              f"{100 * r.value.mean():+.2f}%")
    r = df[df.number < 20]
    print(f"start-up trials: Spearman(formation, robust) "
          f"{r.params_formation.corr(r.value, method='spearman'):+.2f} (n {len(r)})")
    top = df.sort_values("value", ascending=False).head(10)
    print(f"top 10: q_in 0.05 in {int((top.params_q_in < 0.06).sum())}, slots 10 in "
          f"{int((top.params_slots == 10).sum())}, max_hold {int(top.params_max_hold.min())}-"
          f"{int(top.params_max_hold.max())}, formation {int(top.params_formation.min())}-"
          f"{int(top.params_formation.max())}")


def exposure(labels) -> None:
    """Realized mean long, short, gross and net exposure (share of the book) per test."""
    for label in labels:
        res = lab.load_results(KEY, label)
        for g in GROUPS:
            rows = [r for t in GROUPS[g] if t in res for r in res[t]]
            if rows:
                lo, sh = np.mean([r["long"] for r in rows]), np.mean([r["short"] for r in rows])
                print(f"{label} {g}: long {lo:.3f} short {sh:.3f} gross {lo - sh:.3f} "
                      f"net {lo + sh:+.3f}")


def fills(label: str, test: str = "T1") -> None:
    """Every opening order of ``label``'s book on ``test``: filled value ÷ wanted value
    (target × book value at the prior close; no order at all = 0), the realized gross, and
    how often the long and short slot counts differ (the book is dollar-neutral only then)."""
    from trading_simple import backtest as bt
    strat = lab.load_strategy(KEY, lab.load_results(KEY, label)["meta"]["overrides"])
    settings = lab._settings()
    tot = {"open": 0, "under": 0, "none": 0, "short under": 0}
    for te0, _ in lab.periods(test):
        p = lab.panel(test, te0, lab.strategy_features(strat))
        sig = strat.signals(p, **strat.params)
        res = lab.book(strat, p, settings, te0, signals=sig)
        pf, w = res["pf"], bt._score_window(p, sig, te0)[1]["weights"]
        val = pf.value()
        prev_val = val.shift(1).fillna(settings.backtest.init_cash)
        rec = pf.orders.records_readable
        filled = (rec.assign(v=rec["Size"] * rec["Price"]).groupby(["Timestamp", "Column"])["v"]
                  .sum())
        prev = w.shift(1).fillna(0.0)
        opens = np.argwhere(((w != 0) & (prev == 0)).to_numpy())
        ratio, short = [], []
        for i, j in opens:
            ts, col = w.index[i], w.columns[j]
            want = abs(w.iat[i, j]) * prev_val.iat[i]
            ratio.append(filled.get((ts, col), 0.0) / want)
            short.append(w.iat[i, j] < 0)
        ratio, short = np.array(ratio), np.array(short)
        av = pf.asset_value(group_by=False)
        gross = av.abs().sum(axis=1) / val
        uneven = ((w > 0).sum(axis=1) != (w < 0).sum(axis=1)) & (w != 0).any(axis=1)
        net = w.sum(axis=1)
        tot["open"] += len(ratio)
        tot["under"] += int((ratio < 0.9).sum())
        tot["none"] += int((ratio == 0).sum())
        tot["short under"] += int(((ratio < 0.9) & short).sum())
        print(f"{label} {test} {te0}: openings {len(ratio)}, filled < 90% {(ratio < 0.9).sum()}"
              f" (shorts {((ratio < 0.9) & short).sum()}, none {(ratio == 0).sum()}); realized "
              f"gross max {gross.max():.3f}, > 1 on {(gross > 1).mean():.0%} of bars; uneven "
              f"slots {uneven.mean():.0%} of bars, |net target| mean {net.abs().mean():.3f} "
              f"max {net.abs().max():.3f}", flush=True)
    print(f"{label} {test} total: {json.dumps(tot)}")


def _v1(label: str) -> dict:
    """A round-1 (lab v1) result, from ``var/proposals_lab/results/peer_reversal``."""
    return pickle.loads((V1 / f"{label}.pkl").read_bytes())


def _line(res: dict, group: str) -> str:
    s = _stitch(res, group)
    rows = [r for t in GROUPS[group] for r in res[t]]
    extra = " ".join(f"{k.split()[0]} {100 * _stitch(res, group, k)['alpha']:+.2f}"
                     for k in COSTS[1:] if k in rows[0])
    return (f"{group} {100 * s['alpha']:+.2f} (P {s['P(alpha > 0)']:.2f}) {extra} "
            f"beta {s['beta']:+.3f} orders/yr {sum(r['orders'] for r in rows) / s['years']:.0f}")


def v1v2(labels) -> None:
    """Round 1 (lab v1) against round 2 (lab v2, the same label + ``v2``): alpha (P), late-fill
    and 2×-cost alpha, beta and orders a year on TRAIN (stitched), T1, T2 and T4."""
    for label in labels:
        for tag, res in (("v1", _v1(label)), ("v2", lab.load_results(KEY, f"{label}v2"))):
            groups = [g for g in GROUPS if all(t in res for t in GROUPS[g])]
            print(f"{label:7} {tag}: " + " | ".join(_line(res, g) for g in groups))


def _engine_book(strat, p, settings, te0, sig) -> tuple[pd.Series, pd.Series, int]:
    """One period the way lab v1 ran it: the ENGINE's portfolio runner (``call_seq="auto"``,
    the engine's borrow) and day 0 dropped from the book and the benchmark."""
    from trading_simple import backtest as bt
    bt.configure()
    w, s = bt._score_window(p, sig, te0)
    fees, slip = bt.strategy_costs(strat, w, settings)
    pf = bt.run(w, s, fees=fees, slippage=slip, init_cash=settings.backtest.init_cash,
                fill=strat.fill, whole_shares=strat.whole_shares)
    _, bench = bt.benchmark(w, fees=fees, slippage=slip, fill=strat.fill)
    r = bt.equity_curve(pf)["value"].pct_change().fillna(0.0)
    return r, bench.reindex(r.index).fillna(0.0), int(pf.orders.count().sum())


def _day0_dropped(rows, key: str):
    """v1's accounting on lab-v2 rows: day 0 set to 0 in the book and the benchmark."""
    out = []
    for r in rows:
        x, b = r[key].copy(), r["bench"].copy()
        x.iloc[0], b.iloc[0] = 0.0, 0.0
        out.append((x, b))
    return out


def decompose(label: str, tests=None) -> None:
    """What each lab-v2 fix did to one config (RESEARCH_LOG R2.0): v1 → v2 split into the
    panels (holiday rows, ETFs, DB: v1 saved vs the engine's runner on the v2 panels, day 0
    dropped), the runner (call order and borrow: the engine's runner vs lab v2's, both with
    day 0 dropped) and day 0 (the rest), per cost setting. ``label`` is the round-1 label;
    its v2 twin is ``label`` + ``v2``. The engine books cache to ``fix/``."""
    tests = tests or lab.OPEN
    res1, res2 = _v1(label), lab.load_results(KEY, f"{label}v2")
    assert res1["meta"]["overrides"] == res2["meta"]["overrides"]
    strat = lab.load_strategy(KEY, res2["meta"]["overrides"])
    base = lab._settings()
    settings = {"returns": base, "late": lab.with_costs(base, 1.0, 50.0),
                "2x costs": lab.with_costs(base, 2.0)}
    FIX.mkdir(parents=True, exist_ok=True)
    cache = FIX / f"decompose-{label}.pkl"
    eng = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    for t in tests:
        if t in eng:
            continue
        rows = []
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            sig = strat.signals(p, **strat.params)
            row = {}
            for key, s in settings.items():
                r, b, n = _engine_book(strat, p, s, te0, sig)
                row[key] = r
                if key == "returns":
                    row["bench"], row["orders"] = b, n
            rows.append(row)
        eng[t] = rows
        cache.write_bytes(pickle.dumps(eng))
        print(f"decompose {label} {t}: engine books done", flush=True)
    for t in lab.TRAIN:                      # no holiday rows, no ETFs: v1 should reproduce
        gap = max(float(np.abs(a["returns"].to_numpy() - b["returns"].to_numpy()).max())
                  if len(a["returns"]) == len(b["returns"]) else np.inf
                  for a, b in zip(res1[t], eng[t], strict=True))
        print(f"{label} {t}: v1 saved vs the engine's runner on the v2 panels, max daily gap "
              f"{gap:.2e}")
    print(f"\n{label}: stitched alpha, pts/yr — v1 saved | engine on v2 panels (v1 accounting) "
          f"| lab-v2 runner, day 0 dropped | lab v2")
    for g in ("TRAIN", "T1", "T2"):
        for key in COSTS:
            rows1 = [r for t in GROUPS[g] for r in res1[t]]
            rows2 = [r for t in GROUPS[g] for r in res2[t]]
            rowse = [r for t in GROUPS[g] for r in eng[t]]
            a = [lab.stitched([(r[key], r["bench"]) for r in rows1])["alpha"],
                 lab.stitched([(r[key], r["bench"]) for r in rowse])["alpha"],
                 lab.stitched(_day0_dropped(rows2, key))["alpha"],
                 lab.stitched([(r[key], r["bench"]) for r in rows2])["alpha"]]
            a = 100 * np.array(a)
            print(f"  {g:5} {key:8}: {a[0]:+6.2f} | {a[1]:+6.2f} | {a[2]:+6.2f} | {a[3]:+6.2f}"
                  f"   panels {a[1] - a[0]:+5.2f}, runner {a[2] - a[1]:+5.2f}, "
                  f"day 0 {a[3] - a[2]:+5.2f}")
    print("\nper period, base costs (alpha: v1 saved -> engine on v2 panels -> lab v2; "
          "orders v1 -> v2; names only in the v1 / only in the v2 panel):")
    for t in ("T1", "T2"):
        for r1, en, r2 in zip(res1[t], eng[t], res2[t], strict=True):
            c1, c2 = set(r1["contrib total"].index), set(r2["contrib total"].index)
            print(f"  {t} {r2['period']}: {lab.period_alpha(r1['returns'], r1['bench']):+.3f} -> "
                  f"{lab.period_alpha(en['returns'], en['bench']):+.3f} -> "
                  f"{lab.period_alpha(r2['returns'], r2['bench']):+.3f}; orders {r1['orders']} "
                  f"-> {r2['orders']}; swapped {sorted(c1 - c2)} -> {sorted(c2 - c1)}")


def attribute(base: str, label: str) -> None:
    """The adoption Δ (``label`` − ``base``, paired) at each stage of :func:`decompose`: round 1
    saved, v1's accounting on the v2 panels, lab v2. Needs both labels' decompose caches."""
    eng = {k: pickle.loads((FIX / f"decompose-{k}.pkl").read_bytes()) for k in (base, label)}
    stages = (("v1 saved", {k: _v1(k) for k in (base, label)}),
              ("v1 accounting on the v2 panels", eng),
              ("lab v2", {k: lab.load_results(KEY, f"{k}v2") for k in (base, label)}))
    for name, src in stages:
        d = {g: _paired(lab.stitched([(r["returns"], r["bench"]) for t in GROUPS[g]
                                      for r in src[label][t]]),
                        lab.stitched([(r["returns"], r["bench"]) for t in GROUPS[g]
                                      for r in src[base][t]])) for g in ("TRAIN", "T1", "T2")}
        print(f"{label} − {base}, {name}: " + ", ".join(
            f"{g} {100 * v['delta alpha']:+.2f} (P {v['P(delta > 0)']:.2f})" for g, v in d.items()))


def period(label: str, test: str, te0: str, k: int = 6) -> None:
    """One period, round 1 against round 2: the v2 − v1 gap of the cumulative book return
    every 10 bars, the ``k`` names whose P&L contribution changed most, and each one's side
    and bars held in the v2 book (from the strategy's own signals)."""
    a = next(r for r in _v1(label)[test] if r["period"] == te0)
    b = next(r for r in lab.load_results(KEY, f"{label}v2")[test] if r["period"] == te0)
    gap = (1 + b["returns"]).cumprod() - (1 + a["returns"].reindex(b["returns"].index)
                                          .fillna(0.0)).cumprod()
    print(f"{label} {test} {te0}, cumulative book v2 − v1: " + ", ".join(
        f"{d.date()} {100 * v:+.1f}" for d, v in gap.iloc[::10].items()))
    c = pd.DataFrame({"v1": a["contrib total"], "v2": b["contrib total"]}).fillna(0.0)
    top = (c.v2 - c.v1).abs().nlargest(k).index
    strat = lab.load_strategy(KEY, lab.load_results(KEY, f"{label}v2")["meta"]["overrides"])
    p = lab.panel(test, te0, lab.strategy_features(strat))
    w = strat.signals(p, **strat.params)["weights"]
    w = w[w.index >= pd.Timestamp(te0, tz="UTC")]
    for name in top:
        held = w[name][w[name] != 0] if name in w else pd.Series(dtype=float)
        side = "none" if held.empty else ("short" if held.iloc[0] < 0 else "long")
        since = "" if held.empty else f" from {held.index[0].date()}"
        print(f"  {name}: contribution v1 {100 * c.v1[name]:+.2f} -> v2 {100 * c.v2[name]:+.2f} "
              f"pts; v2 {side}, {len(held)} bars held{since}")


def borrow(label: str, tests=None) -> None:
    """Lab v2's fix 4 (borrow on hair re-trims) against this book: per test, the short
    re-sizes (a held short's target changed to another negative value) and the largest gap
    between ``lab.borrow_fees`` and the engine's ``backtest._borrow_fees``."""
    from trading_simple import backtest as bt
    strat = lab.load_strategy(KEY, lab.load_results(KEY, label)["meta"]["overrides"])
    fees = 0.0                        # both add the same per-leg fee: compare the surcharge
    for t in tests or lab.OPEN:
        resized, gap = 0, 0.0
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            _, sig = bt._score_window(p, strat.signals(p, **strat.params), te0)
            tg = bt.rotation_targets(sig["weights"])
            prev = tg.ffill().shift(1).fillna(0.0)
            resized += int((tg.notna() & (tg < 0) & (prev < 0) & (tg != prev)).sum().sum())
            gap = max(gap, float((lab.borrow_fees(tg, sig["borrow"], fees)
                                  - bt._borrow_fees(tg, sig["borrow"], fees)).abs().max().max()))
        print(f"{label} {t}: short re-sizes {resized}, max fee-frame gap {gap:.1e}", flush=True)


def holidays(labels) -> None:
    """The v1 holiday rows (2026-06-19, 07-03, 09-07) and this book: (a) stressed bars in each
    2026 period, on the v2 panel and with those rows put back as empty rows (every name NaN,
    as for a panel without Swiss listings); (b) Jul-Sep 2026, book vs market, v1 and v2."""
    params = lab.load_strategy(KEY).params
    rows = pd.to_datetime(["2026-06-19", "2026-07-03", "2026-09-07"]).tz_localize("UTC")
    for t in ("T1", "T2", "T4"):
        for te0, te1 in lab.periods(t)[-2:]:
            p = lab.panel(t, te0, [])
            a, z = pd.Timestamp(te0, tz="UTC"), pd.Timestamp(te1, tz="UTC")
            idx = p.close.index.union(rows[rows <= z])
            q = replace(p, close=p.close.reindex(idx), open_=p.open_.reindex(idx),
                        high=p.high.reindex(idx), low=p.low.reindex(idx),
                        volume=p.volume.reindex(idx), features={})
            on = [_compose(x, params, peers="market")["stress"] for x in (p, q)]
            on = [s[(s.index >= a) & (s.index <= z) & ~s.index.isin(rows)] for s in on]
            print(f"{t} {te0}: stressed bars {int(on[0].sum())}/{len(on[0])} on the v2 panel, "
                  f"{int(on[1].sum())}/{len(on[1])} with the holiday rows put back")
    a, z = pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-09-30", tz="UTC")
    for label in labels:
        for tag, res in (("v1", _v1(label)), ("v2", lab.load_results(KEY, f"{label}v2"))):
            out = []
            for t in [t for t in ("T1", "T2", "T4") if t in res]:
                r = pd.concat([x["returns"] for x in res[t]])
                b = pd.concat([x["bench"] for x in res[t]])
                m = (r.index >= a) & (r.index <= z)
                out.append(f"{t} {100 * ((1 + r[m]).prod() - 1):+.1f}% (market "
                           f"{100 * ((1 + b[m]).prod() - 1):+.1f}%)")
            print(f"Jul-Sep 2026 {label} {tag}: " + ", ".join(out))


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == "stress":
        stress(a[1:])
    elif cmd == "noise":
        noise(a[1], a[2])
    elif cmd == "noise-luck":
        noise_luck(a[1])
    elif cmd == "curve":
        curve(json.loads(a[1]) if len(a) > 1 else {})
    elif cmd == "tuner":
        tuner()
    elif cmd == "exposure":
        exposure(a[1:])
    elif cmd == "fills":
        fills(a[1], a[2] if len(a) > 2 else "T1")
    elif cmd == "pairs":
        pairs(a[1], a[2])
    elif cmd == "v1v2":
        v1v2(a[1:])
    elif cmd == "decompose":
        decompose(a[1], a[2].split(",") if len(a) > 2 else None)
    elif cmd == "attribute":
        attribute(a[1], a[2])
    elif cmd == "period":
        period(a[1], a[2], a[3])
    elif cmd == "borrow":
        borrow(a[1], a[2].split(",") if len(a) > 2 else None)
    elif cmd == "holidays":
        holidays(a[1:])
    else:
        raise SystemExit(__doc__)
