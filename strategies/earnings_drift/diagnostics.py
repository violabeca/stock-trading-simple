"""earnings_drift diagnostics: every number RESEARCH_LOG quotes comes from committed code.

Review #6: numbers that lived only in a session transcript are recomputed here.

    uv run python strategies/earnings_drift/diagnostics.py <command> [args]

Commands (each prints a table). The outputs the log quotes are saved as ``diag-*.log``: round 1
(lab v1) in ``var/proposals_lab/results/earnings_drift/``, round 2 (lab v2, the version this
file now runs on: ``lab.RESULTS`` / ``lab.STUDIES``) in ``results2/earnings_drift/``.

    precision                 TRAIN only: the EPS cent quantum, the coarse-surprise rule, and the
                              forward returns of coarse vs precise events, old vs fixed code
    tiny                      TRAIN only, no returns: events with |estimate| < $0.05 against the
                              cent rule, per surprise measure (round 2, V0)
    corr <label> [p1 label]   daily / monthly return correlation with informed_flow_ls_p1's lab
                              book (default v6_si), per test both hold
    timing                    TRAIN only: volume and |abnormal return| by offset from earn_day
    trials <study> [dir]      TPE marginal means of a lab tuning journal (mean period alpha);
                              ``dir`` = the journal folder (default ``lab.STUDIES``)
    rescore <study> [k] [dir] the journal's k best distinct trials (default 10) re-scored on the
                              current lab's TRAIN books: robust score and rank, journal vs now
    mechanics ['<overrides>'] qualifying shares, holdings, entries a year, signals() time
    book <label>...           orders, exposure, cost drag and alpha per unit of gross, from the
                              saved results
    paired <base> <label>... [--tests E0,TB,T3]   each label's alpha minus the base's on the
                              tests STITCHED into one record (default the TRAIN era), paired
                              bootstrap P(Δ > 0), from the saved results
    anatomy <label> [tests]   re-runs the base books: realized gross, rejected / partial entries,
                              first-bar (carried-in) orders and the orders whose cost is charged
    warmup [tests]            the coarse rule with its unjudged warm-up counted coarse (the
                              shipped code, round 2 V0a) against the R0 code, which kept it
    v1acct <v2 label> <v1 label> [tests]   each period under lab v1's accounting on the v2
                              panels, beside the round-1 record and the v2 result: the
                              v1 -> v2 change split into the panels and the accounting
    dropped [tests] ['<overrides>']   coarse events on their reaction alone vs not scored vs
                              no rule, and the coarse entries by side
    placebo [n] [tests] ['<overrides>']   random events re-scored on their reaction alone, as
                              many a quarter as the coarse rule flags: the rule's Δ against
                              the spread of n placebo Δs (default 20, T1 and T2)
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ed_premise", HERE / "premise.py")
premise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(premise)
lab = premise.lab
KEY = "earnings_drift"
#: The tuned config of §4 (old code); ``cent_max`` inf switches the coarse rule off.
T0 = {"surprise": "pct", "w_ear": 0.0, "ear_end": 2, "trail": 126, "q": 0.4, "hold": 80,
      "slots": 20}
OFF = {"cent_max": float("inf")}
EVENT_KEYS = ("surprise", "w_ear", "ear_end", "trail", "min_events", "min_price", "cent_max")


def _events_fn():
    return lab.load_strategy(KEY).signals.__globals__["events"]


def _window(ev: pd.DataFrame, te0: str, te1: str) -> pd.DataFrame:
    d = ev["date"]
    return ev[(d >= pd.Timestamp(te0, tz="UTC")) & (d <= pd.Timestamp(te1, tz="UTC"))]


def precision() -> None:
    """The EPS precision issue on TRAIN: data properties, then coarse vs precise events."""
    events = _events_fn()
    strat = lab.load_strategy(KEY)
    feats = lab.strategy_features(strat)
    late = lab.panel("T1", "2026-07-01", ["fund_shares"]).features["fund_shares"]
    last_sh = late.ffill().iloc[-1]
    rows = []
    for t in lab.TRAIN:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, feats)
            f = p.features
            ev = _window(events(p), te0, te1)
            b, j = ev["bar"].to_numpy(), ev["col"].to_numpy()
            est, act = f["earn_estimate"].to_numpy(float), f["earn_actual"].to_numpy(float)
            pc = events(p, surprise="pct")
            pc = _window(pc, te0, te1)
            rows.append(pd.DataFrame({
                "test": t, "date": ev["date"].to_numpy(), "symbol": ev["symbol"].to_numpy(),
                "est": est[b, j], "act": act[b, j],
                "sur": f["earn_surprise"].to_numpy(float)[b, j],
                "close": p.close.to_numpy(float)[b - 3, j], "sue": ev["sue"].to_numpy(),
                "coarse_price": ev["coarse"].to_numpy(),
                "coarse_pct": pc["coarse"].to_numpy(),
                "F": last_sh.reindex(ev["symbol"]).to_numpy()
                / f["fund_shares"].to_numpy(float)[b, j]}))
    df = pd.concat(rows, ignore_index=True)
    df = df[df["est"].notna() & df["act"].notna()]
    cents = lambda x: float(((x * 100).round(6) % 1 == 0).mean())  # noqa: E731
    calc = (df["act"] - df["est"]) / df["est"].abs()
    ok = df["est"].abs() > 0
    print(f"== data: {len(df)} TRAIN events with actual and estimate")
    print(f"  whole-cent share: actual {cents(df['act']):.4f}, estimate {cents(df['est']):.4f}; "
          f"max |earn_surprise - (a - e)/|e||: {(df.loc[ok, 'sur'] - calc[ok]).abs().max():.2e}")
    q_p, q_e = 0.01 / df["close"], 0.01 / df["est"].abs()
    sue, pct = df["sue"], df["sur"]
    print(f"  price SUE P20/P50/P80 {sue.quantile(.2):.5f} / {sue.quantile(.5):.5f} / "
          f"{sue.quantile(.8):.5f}; quantum 0.01/close median {q_p.median():.5f} p90 "
          f"{q_p.quantile(.9):.5f} p99 {q_p.quantile(.99):.5f}")
    print(f"  pct P20/P50/P80 {pct.quantile(.2):.4f} / {pct.quantile(.5):.4f} / "
          f"{pct.quantile(.8):.4f}; quantum 0.01/|e| median {q_e.median():.4f} p90 "
          f"{q_e.quantile(.9):.4f} p99 {q_e.quantile(.99):.4f}")
    known = df["F"].notna()
    df["splitter"] = df["F"] >= 1.8
    print("  coarse share per test (rule cent_max 0.25, trailing spread):",
          df.groupby("test")[["coarse_price", "coarse_pct"]].mean().round(3).to_dict("index"))
    for m in ("coarse_price", "coarse_pct"):
        s = df[known].groupby(m)["splitter"].mean()
        print(f"  future splitters (SEC shares >= 1.8x by 2026) among {m} True / False: "
              f"{s.get(True, np.nan):.3f} / {s.get(False, np.nan):.3f}")
    top = df[df["coarse_price"]].groupby("symbol").size().sort_values(ascending=False).head(15)
    print("  most coarse (price):", top.to_dict())
    top = df[df["coarse_pct"]].groupby("symbol").size().sort_values(ascending=False).head(15)
    print("  most coarse (pct):", top.to_dict())
    sp = df[["date", "symbol", "splitter"]]
    for name, over in (("U0 old code", OFF), ("U0f", {}), ("T0 old code", {**T0, **OFF}),
                       ("T0q (T0 on the fixed code)", T0)):
        ev = premise.train_events(over, horizons=(20, 60, 80))
        qq = lab.load_strategy(KEY, over).params["q"]
        ev["side"] = np.where(ev["pct"] >= 1 - qq - 1e-9, "long",
                              np.where(ev["pct"] <= qq + 1e-9, "short", "mid"))
        # coarse under the fixed rule, whatever this config's own cent_max
        rule = lab.load_strategy(KEY, {k: v for k, v in over.items() if k != "cent_max"})
        tag = premise_coarse(rule)
        ev = ev.merge(tag, on=["date", "symbol"], how="left").merge(sp, on=["date", "symbol"],
                                                                    how="left")
        g = ev.groupby(["side", "rule_coarse"]).agg(
            n=("pct", "size"), fwd20=("fwd20", "mean"), fwd60=("fwd60", "mean"),
            fwd80=("fwd80", "mean"), splitters=("splitter", "mean"))
        print(f"\n== {name}: events by side x coarse-under-the-rule (beta-adjusted fwd returns "
              f"from the open after the usable bar, prices <= 2021-12-31)")
        print(g.round(4).to_string())
        c = ev[ev["rule_coarse"] == True]  # noqa: E712
        print(f"  coarse events: {len(c)}, long {int((c['side'] == 'long').sum())}, short "
              f"{int((c['side'] == 'short').sum())} (all events: long share "
              f"{(ev['side'] == 'long').mean():.3f}, coarse: {(c['side'] == 'long').mean():.3f})")


def tiny() -> None:
    """Round 2 V0: do the review's tiny estimates (|estimate| < $0.05) escape the cent rule?
    TRAIN windows, the shipped rule, both surprise measures; no returns. The quantum share is
    the cent quantum over the event's trailing P20-P80 spread (the rule flags above 0.25);
    future splitters = SEC shares >= 1.8x by 2026 (a diagnostic proxy only)."""
    strat = lab.load_strategy(KEY)
    g = strat.signals.__globals__
    feats = lab.strategy_features(strat)
    last = lab.panel("T1", "2026-07-01", ["fund_shares"]).features["fund_shares"].ffill().iloc[-1]
    rows = []
    for t in lab.TRAIN:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, feats)
            f = p.features
            for meas in ("price", "pct"):
                ev = g["events"](p, surprise=meas)
                spread = g["_trailing_spread"](ev["bar"].to_numpy(), ev["sue"].to_numpy(float),
                                               strat.params["trail"], strat.params["min_events"])
                ev = ev.assign(spread=spread)
                ev = _window(ev, te0, te1)
                b, j = ev["bar"].to_numpy(), ev["col"].to_numpy()
                est = f["earn_estimate"].to_numpy(float)[b, j]
                close = p.close.to_numpy(float)[b - 3, j]
                quantum = 0.01 / close if meas == "price" else 0.01 / np.abs(est)
                rows.append(pd.DataFrame({
                    "meas": meas, "symbol": ev["symbol"].to_numpy(), "est": est,
                    "sue": ev["sue"].to_numpy(), "coarse": ev["coarse"].to_numpy(),
                    "share": quantum / ev["spread"].to_numpy(), "pct": ev["pct"].to_numpy(),
                    "F": last.reindex(ev["symbol"]).to_numpy()
                    / f["fund_shares"].to_numpy(float)[b, j]}))
    df = pd.concat(rows, ignore_index=True)
    df = df[np.isfinite(df["est"])]
    df["tiny"] = df["est"].abs() < 0.05
    df["side"] = np.where(df["pct"] >= 0.8 - 1e-9, "long",
                          np.where(df["pct"] <= 0.2 + 1e-9, "short", "mid"))
    for meas in ("price", "pct"):
        m = df[df["meas"] == meas]
        tn = m[m["tiny"]]
        un = tn[~tn["coarse"]]
        known = un["F"].notna()
        print(f"{meas}: {len(m)} TRAIN events with an estimate, |estimate| < 0.05: {len(tn)} "
              f"({len(tn) / len(m):.1%}); neutralized {tn['coarse'].mean():.1%} (all other events "
              f"{m.loc[~m['tiny'], 'coarse'].mean():.1%})")
        with_sue = int(un["sue"].notna().sum())
        print(f"   tiny and kept: {len(un)}, of them with a surprise {with_sue};"
              f" quantum share median {un['share'].median():.3f}; sides "
              f"{un['side'].value_counts().to_dict()}; future splitters "
              f"{(un.loc[known, 'F'] >= 1.8).mean():.1%}")


def premise_coarse(strat) -> pd.DataFrame:
    """(date, symbol, rule_coarse) of every TRAIN event under ``strat``'s surprise measure and
    the shipped cent_max (the fixed rule)."""
    events = _events_fn()
    keys = {k: strat.params[k] for k in EVENT_KEYS}
    out = []
    for t in lab.TRAIN:
        for te0, te1 in lab.periods(t):
            ev = _window(events(lab.panel(t, te0, lab.strategy_features(strat)), **keys),
                         te0, te1)
            out.append(ev[["date", "symbol", "coarse"]])
    return pd.concat(out).rename(columns={"coarse": "rule_coarse"})


def corr(label: str, other: str = "v6_si") -> None:
    """Daily and monthly return correlation of ``label`` with p1's lab book, per test."""
    a = lab.load_results(KEY, label)
    b = lab.load_results("informed_flow_ls_p1", other)
    for t in lab.TESTS:
        if t not in a or t not in b:
            continue
        ra = pd.concat([r["returns"] for r in a[t]])
        rb = pd.concat([r["returns"] for r in b[t]])
        ra, rb = ra[~ra.index.duplicated()], rb[~rb.index.duplicated()]
        both = ra.index.intersection(rb.index)
        ra, rb = ra[both], rb[both]
        key = both.tz_convert(None).to_period("M")
        ma, mb = (1 + ra).groupby(key).prod() - 1, (1 + rb).groupby(key).prod() - 1
        print(f"{t}: {label} vs informed_flow_ls_p1/{other}: daily {ra.corr(rb):+.2f} "
              f"(n {len(both)}), monthly {ma.corr(mb):+.2f} (n {len(ma)})")


def timing() -> None:
    """Volume and |abnormal return| by offset from earn_day, relative to the name's normal
    (median over sessions p-65 .. p-6), TRAIN periods' events only."""
    strat = lab.load_strategy(KEY)
    offs = range(-3, 4)
    vol_r, ar_r = {k: [] for k in offs}, {k: [] for k in offs}
    for t in lab.TRAIN:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            v = p.volume.to_numpy(float)
            r = p.close.pct_change(fill_method=None).to_numpy(float)
            with warnings.catch_warnings():                 # all-NaN first row
                warnings.simplefilter("ignore", RuntimeWarning)
                ar = np.abs(r - np.nanmean(r, axis=1, keepdims=True))
            day = p.features["earn_day"]
            idx = p.close.index
            win = (idx >= pd.Timestamp(te0, tz="UTC")) & (idx <= pd.Timestamp(te1, tz="UTC"))
            for i, j in zip(*np.nonzero(day.notna().to_numpy() & win[:, None]), strict=True):
                if i - 65 < 0 or i + 3 >= len(idx):
                    continue
                nv, na = np.nanmedian(v[i - 65:i - 5, j]), np.nanmedian(ar[i - 65:i - 5, j])
                if not (nv > 0 and na > 0):
                    continue
                for k in offs:
                    vol_r[k].append(v[i + k, j] / nv)
                    ar_r[k].append(ar[i + k, j] / na)
    print(f"{len(vol_r[0])} TRAIN report sessions; ratio to the name's normal (median over "
          "sessions p-65 .. p-6), median / mean across events:")
    for k in offs:
        tag = {0: "  (report session p)", 2: "  (usable bar u = p+2: the decision)",
               3: "  (fill day)"}.get(k, "")
        print(f"  p{k:+d}: volume x{np.nanmedian(vol_r[k]):.2f} / x{np.nanmean(vol_r[k]):.2f}"
              f"   |abnormal return| x{np.nanmedian(ar_r[k]):.2f} / x{np.nanmean(ar_r[k]):.2f}"
              f"{tag}")


def _journal(study: str, folder: str | None = None):
    """The complete trials of a lab tuning journal (``folder`` default ``lab.STUDIES``)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    path = Path(folder) if folder else lab.STUDIES
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(
        str(path / f"{study}.log")))
    st = optuna.load_study(study_name=study, storage=storage)
    return [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]


def trials(study: str, folder: str | None = None) -> None:
    """Marginal mean of the trials' mean period alpha per knob value (a TPE journal)."""
    done = _journal(study, folder)
    df = pd.DataFrame([{**t.params, "alpha": t.user_attrs.get("mean alpha"), "score": t.value}
                       for t in done])
    knobs = [c for c in df.columns if c not in ("alpha", "score")]
    print(f"{study}: {len(df)} complete trials, {len(df[knobs].drop_duplicates())} distinct; "
          f"best score {df['score'].max():+.4f}")
    for k in knobs:
        g = df.groupby(k)["alpha"].agg(["mean", "size"])
        print(f"  {k}: " + ", ".join(
            f"{round(v, 4) if isinstance(v, float) else v}: {m:+.3f} (n {int(n)})"
            for v, (m, n) in zip(g.index, g.to_numpy(), strict=True)))


def rescore(study: str, k: int = 10, folder: str | None = None) -> None:
    """The journal's ``k`` best distinct trials re-scored on the current lab's TRAIN books
    (:func:`lab.robust` over the 11 period alphas, as ``lab.tune`` scores them): did a change
    of the lab reorder what the tuner picked? Selects nothing; reads TRAIN only."""
    settings = lab._settings()
    done = sorted(_journal(study, folder), key=lambda t: -t.value)
    seen, picks = set(), []
    for t in done:
        key = tuple(sorted(t.params.items()))
        if key not in seen:
            seen.add(key)
            picks.append(t)
        if len(picks) == k:
            break
    rows = []
    for rank, t in enumerate(picks, 1):
        strat = lab.load_strategy(KEY, t.params)
        alphas = []
        for x in lab.TRAIN:
            for te0, _ in lab.periods(x):
                r = lab.book(strat, lab.panel(x, te0, lab.strategy_features(strat)), settings,
                             te0)
                alphas.append(lab.period_alpha(r["returns"], r["bench"]))
        rows.append({"journal rank": rank, "journal score": t.value, "now": lab.robust(alphas),
                     "now mean alpha": float(np.mean(alphas)), **t.params})
        print(f"  {rank}: journal {t.value:+.4f} -> now {rows[-1]['now']:+.4f}  {t.params}",
              flush=True)
    df = pd.DataFrame(rows)
    df["now rank"] = df["now"].rank(ascending=False).astype(int)
    with pd.option_context("display.width", 200):
        print(df.round(4).to_string(index=False))
    print(f"rank correlation journal vs now: "
          f"{df['journal score'].corr(df['now'], method='spearman'):+.2f}; the journal's pick is "
          f"now #{int(df.loc[0, 'now rank'])} of {len(df)} (lab version {lab.LAB_VERSION})")


def mechanics(overrides: dict | None = None) -> None:
    """Qualifying shares, holdings, gross, net and entries a year on every OPEN period."""
    strat = lab.load_strategy(KEY, overrides)
    events = strat.signals.__globals__["events"]
    keys = {k: strat.params[k] for k in EVENT_KEYS}
    q = strat.params["q"]
    for t in lab.OPEN:
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            t0 = time.time()
            w = strat.signals(p, **strat.params)["weights"]
            secs = time.time() - t0
            ev = _window(events(p, **keys), te0, te1)
            sc = ev[ev["pct"].notna()]
            idx = w.index
            win = (idx >= pd.Timestamp(te0, tz="UTC")) & (idx <= pd.Timestamp(te1, tz="UTC"))
            ww = w[win]
            nl, ns = (ww > 0).sum(axis=1), (ww < 0).sum(axis=1)
            prev = w.shift(1, fill_value=0.0)[win]
            entries = int(((prev == 0) & (ww != 0)).to_numpy().sum())
            yrs = lab.years_of(ww.index)
            ql, qs = (sc["pct"] >= 1 - q - 1e-9).mean(), (sc["pct"] <= q + 1e-9).mean()
            print(f"{t} {te0}: scored {len(sc)}, coarse {ev['coarse'].mean():.3f}, qualify long "
                  f"{ql:.3f} short {qs:.3f}; held long {nl.mean():.1f} ({nl.min()}-{nl.max()}) "
                  f"short {ns.mean():.1f} ({ns.min()}-{ns.max()}); gross max "
                  f"{ww.abs().sum(axis=1).max():.3f}, net mean {ww.sum(axis=1).mean():+.4f}; "
                  f"entries/yr {entries / yrs:.0f}; signals {secs:.2f}s")


def book(labels: list[str], tests=None) -> None:
    """Orders, exposure, cost drag and alpha per unit of gross from saved results."""
    for label in labels:
        res = lab.load_results(KEY, label)
        for t in tests or [t for t in lab.TESTS if t in res]:
            if t not in res:
                continue
            rows = res[t]
            s = lab._stitch(rows)
            late, c2 = lab._stitch(rows, "late")["alpha"], lab._stitch(rows, "2x costs")["alpha"]
            lo, sh = np.mean([r["long"] for r in rows]), np.mean([r["short"] for r in rows])
            gross = lo - sh
            print(f"{label:6s} {t}: alpha {s['alpha']:+.4f}  orders/yr "
                  f"{sum(r['orders'] for r in rows) / s['years']:.0f}  long {lo:.3f} short "
                  f"{sh:.3f} gross {gross:.3f}  1x cost drag {s['alpha'] - c2:.4f}  late penalty "
                  f"{s['alpha'] - late:.4f}  alpha per unit of gross {s['alpha'] / gross:+.4f}")


def paired(base: str, labels: list[str], tests=None) -> None:
    """Stitched paired Δ of each label against ``base`` over ``tests`` (default TRAIN): the
    adoption bar's "Δ ≥ 0 on TRAIN" reads the E0 + TB + T3 periods as ONE record."""
    tests = tests or lab.TRAIN
    ref = lab.load_results(KEY, base)
    b = lab.stitched([(r["returns"], r["bench"]) for t in tests for r in ref[t]])
    print(f"{base} {'+'.join(tests)} stitched: alpha {b['alpha']:+.4f} (P {b['P(alpha > 0)']:.3f})")
    for label in labels:
        res = lab.load_results(KEY, label)
        s = lab.stitched([(r["returns"], r["bench"]) for t in tests for r in res[t]])
        pr = lab.paired(s["returns"], b["returns"].reindex(s["returns"].index).fillna(0.0),
                        b["bench"].reindex(s["returns"].index).fillna(0.0), b["years"])
        print(f"  {label}: alpha {s['alpha']:+.4f} (P {s['P(alpha > 0)']:.3f}), Δ vs {base} "
              f"{pr['delta alpha']:+.4f} (P(Δ > 0) {pr['P(delta > 0)']:.3f})")


def anatomy(label: str, tests=None) -> None:
    """Re-run ``label``'s base books: realized gross, entries the engine rejected or filled in
    part, the carried-in first-bar orders and the orders whose cost reaches the returns (lab v1
    dropped each period's first-bar return, so the first bar's orders were never charged; lab
    v2 counts day 0 from the initial cash, so every order is)."""
    from trading_simple import backtest as bt
    res = lab.load_results(KEY, label)
    strat = lab.load_strategy(KEY, res["meta"]["overrides"])
    settings = lab._settings()
    for t in tests or ["T1", "T2"]:
        agg = {"orders": 0, "first": 0, "entries": 0, "rejected": 0, "partial": 0,
               "rej_short": 0, "years": 0.0, "bars": 0, "over1": 0, "gmax": 0.0, "gsum": 0.0}
        for te0, _ in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            sig = strat.signals(p, **strat.params)
            out = lab.book(strat, p, settings, te0, signals=sig)
            pf = out["pf"]
            _, wsig = bt._score_window(p, sig, te0)
            w = wsig["weights"]
            o = pf.orders.records_readable
            real = pf.asset_value(group_by=False).div(pf.value(), axis=0)
            g = real.abs().sum(axis=1)
            prev = w.shift(1)
            prev.iloc[0] = 0.0
            ent = np.argwhere(((prev == 0) & (w != 0)).to_numpy())
            got = np.array([real.iat[i, j] / w.iat[i, j] for i, j in ent])
            agg["orders"] += len(o)
            agg["first"] += int((o["Timestamp"] == w.index[0]).sum())  # carried-in book
            agg["entries"] += len(ent)
            agg["rejected"] += int((got < 0.05).sum())
            agg["partial"] += int(((got >= 0.05) & (got < 0.9)).sum())
            agg["rej_short"] += int(sum(w.iat[i, j] < 0 for (i, j), x in zip(ent, got, strict=True)
                                        if x < 0.9))
            agg["years"] += lab.years_of(w.index)
            agg["bars"] += len(g)
            agg["over1"] += int((g > 1.0).sum())
            agg["gmax"] = max(agg["gmax"], float(g.max()))
            agg["gsum"] += float(g.sum())
            first = int((o["Timestamp"] == w.index[0]).sum())
            print(f"  {t} {te0}: orders {len(o)} (first bar {first}), realized gross mean "
                  f"{g.mean():.3f} max {g.max():.3f} (> 1.0 on "
                  f"{int((g > 1.0).sum())} bars), entries {len(ent)}: rejected "
                  f"{int((got < 0.05).sum())}, partial {int(((got >= 0.05) & (got < 0.9)).sum())}",
                  flush=True)
        y = agg["years"]
        v2 = getattr(lab, "LAB_VERSION", 1) >= 2
        charged = (agg["orders"] - (0 if v2 else agg["first"])) / y
        where = "charged since lab v2" if v2 else "cost not in the returns"
        print(f"{label} {t}: orders/yr {agg['orders'] / y:.0f}, first-bar (carried-in, {where}) "
              f"{agg['first'] / y:.0f}/yr, charged {charged:.0f}/yr; "
              f"realized gross mean {agg['gsum'] / agg['bars']:.3f} max {agg['gmax']:.3f}, "
              f"> 1.0 on {agg['over1'] / agg['bars']:.1%} of bars; entries {agg['entries']}: "
              f"rejected {agg['rejected']}, partial (< 90%) {agg['partial']}, of them shorts "
              f"{agg['rej_short']}")


def warmup(tests=("E0",)) -> None:
    """Round 2 V0a: the shipped rule counts an unjudged event (a trailing window holding fewer
    than ``min_events`` surprises, the panel's first ~90 events) as coarse; the R0 code kept
    its raw surprise in later events' reference distributions. Base books of both, per period
    and stitched, and how many in-window event ranks differ."""
    strat = lab.load_strategy(KEY)
    settings = lab._settings()
    keys = {k: strat.params[k] for k in EVENT_KEYS}
    g = strat.signals.__globals__
    for t in tests:
        books, bench = {"shipped (V0a)": [], "R0 code": []}, []
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            ev = g["events"](p, **keys)
            spread = g["_trailing_spread"](ev["bar"].to_numpy(), ev["sue"].to_numpy(float),
                                           strat.params["trail"], strat.params["min_events"])
            shipped = ev["coarse"].to_numpy(bool)
            r0 = shipped & np.isfinite(spread)          # R0: an unjudged event is not coarse
            sigs = {"shipped (V0a)": _signals_with_mask(strat, p, shipped),
                    "R0 code": _signals_with_mask(strat, p, r0)}
            for name, sig in sigs.items():
                out = lab.book(strat, p, settings, te0, signals=sig)
                books[name].append(out["returns"])
                if name == "R0 code":
                    bench.append(out["bench"])
            d = sigs["shipped (V0a)"]["weights"].ne(sigs["R0 code"]["weights"])
            win = (p.close.index >= pd.Timestamp(te0, tz="UTC")) & (
                p.close.index <= pd.Timestamp(te1, tz="UTC"))
            a0 = lab.period_alpha(books["shipped (V0a)"][-1], bench[-1])
            a1 = lab.period_alpha(books["R0 code"][-1], bench[-1])
            print(f"  {t} {te0}: unjudged events {int((~np.isfinite(spread)).sum())}; weight "
                  f"cells differing in the window {int(d[win].to_numpy().sum())}; alpha shipped "
                  f"{a0:+.4f} vs R0 {a1:+.4f}", flush=True)
        alpha = {k: lab.stitched(list(zip(v, bench, strict=True)))["alpha"]
                 for k, v in books.items()}
        print(f"{t}: " + ", ".join(f"{k} {v:+.4f}" for k, v in alpha.items()), flush=True)


def v1acct(label: str, v1label: str, tests=None, r0: bool = False) -> None:
    """Split each period's v1 -> v2 change into the panels and the accounting.

    Re-runs ``label``'s config on the v2 panels under lab v1's accounting: the engine's own
    portfolio runner (``backtest.run``: ``call_seq="auto"``, the engine's borrow fees) and each
    period's day 0 dropped (``pct_change().fillna(0)``, book and benchmark). Printed beside the
    round-1 record ``v1label`` (v1 panels, v1 accounting) and the saved v2 result ``label``.
    ``r0``: the v1-accounting book runs the R0 code's coarse rule (an unjudged warm-up event
    keeps its surprise, :func:`warmup`), so "panels" is the data alone."""
    from trading_simple import backtest as bt
    new = lab.load_results(KEY, label)
    old = pickle.loads((lab.VAR / "results" / KEY / f"{v1label}.pkl").read_bytes())
    strat = lab.load_strategy(KEY, new["meta"]["overrides"])
    settings = lab._settings()
    init = settings.backtest.init_cash
    for t in tests or [t for t in lab.OPEN if t in new and t in old]:
        pairs = {"v1": [], "v1 accounting, v2 panels": [], "v2": []}
        for i, (te0, _) in enumerate(lab.periods(t)):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            sig = strat.signals(p, **strat.params)
            if r0:
                g = strat.signals.__globals__
                ev = g["events"](p, **{k: strat.params[k] for k in EVENT_KEYS})
                judged = np.isfinite(g["_trailing_spread"](
                    ev["bar"].to_numpy(), ev["sue"].to_numpy(float), strat.params["trail"],
                    strat.params["min_events"]))
                sig = _signals_with_mask(strat, p, ev["coarse"].to_numpy(bool) & judged)
            w, s = bt._score_window(p, sig, te0)
            fees, slip = bt.strategy_costs(strat, w, settings)
            pf = bt.run(w, s, fees=fees, slippage=slip, init_cash=init, fill=strat.fill)
            _, bench = bt.benchmark(w, fees=fees, slippage=slip, fill=strat.fill)
            r = bt.equity_curve(pf)["value"].pct_change().fillna(0.0)
            o, n = old[t][i], new[t][i]
            pairs["v1"].append((o["returns"], o["bench"]))
            pairs["v1 accounting, v2 panels"].append((r, bench.reindex(r.index).fillna(0.0)))
            pairs["v2"].append((n["returns"], n["bench"]))
            a = [lab.period_alpha(*v[-1]) for v in pairs.values()]
            print(f"  {t} {te0}: v1 {a[0]:+.4f} | v1 accounting on v2 panels {a[1]:+.4f} "
                  f"(panels {a[1] - a[0]:+.4f}) | v2 {a[2]:+.4f} (accounting {a[2] - a[1]:+.4f});"
                  f" orders {int(o['orders'])} / {int(pf.orders.count().sum())} / "
                  f"{int(n['orders'])}", flush=True)
        st = {k: lab.stitched(v)["alpha"] for k, v in pairs.items()}
        print(f"{t}: " + ", ".join(f"{k} {v:+.4f}" for k, v in st.items())
              + f"  (panels {st['v1 accounting, v2 panels'] - st['v1']:+.4f}, accounting "
              f"{st['v2'] - st['v1 accounting, v2 panels']:+.4f})", flush=True)


def _signals_with_mask(strat, p, mask: np.ndarray, drop: bool = False) -> dict:
    """``signals()`` with the coarse set replaced by ``mask`` (one bool per event of
    :func:`strategy.events`): the masked events lose their surprise exactly as the rule's
    coarse events do (``drop``: they are not scored at all instead). ``mask`` = the rule's own
    ``coarse`` reproduces ``signals()``."""
    g = strat.signals.__globals__
    prm = strat.params
    keys = {k: prm[k] for k in EVENT_KEYS}
    ev = g["events"](p, **{**keys, "cent_max": float("inf")})
    u, sue, r_ear = ev["bar"].to_numpy(), ev["sue"].to_numpy(float), ev["r_ear"].to_numpy(float)
    r_sue = g["_trailing_pct"](u, np.where(mask, np.nan, sue), prm["trail"], prm["min_events"])
    w0 = np.where(np.isnan(r_sue), 0.0, 1.0 - prm["w_ear"])
    w1 = np.where(np.isnan(r_ear), 0.0, prm["w_ear"])
    with np.errstate(invalid="ignore", divide="ignore"):
        score = (w0 * np.nan_to_num(r_sue) + w1 * np.nan_to_num(r_ear)) / (w0 + w1)
    score[w0 + w1 <= 0] = np.nan
    if drop:
        score[mask] = np.nan
    pct = g["_trailing_pct"](u, score, prm["trail"], prm["min_events"])
    borrow, ok = g["crowding"](p, **{k: prm[k] for k in ("dtc_mid", "dtc_high", "short_max_dtc",
                                                         "borrow_gc", "borrow_mid",
                                                         "borrow_high")})
    n, m = p.close.shape
    longs, shorts = g["_select"](u, ev["col"].to_numpy(), pct, n, m, q=prm["q"],
                                 hold=prm["hold"], slots=prm["slots"], sides=prm["sides"],
                                 ok_short=ok.to_numpy(),
                                 allow=g["_crowd_filter"](p, ev.assign(pct=pct), prm["crowd"]))
    size_l = prm["gross"] * prm["long_share"] / prm["slots"]
    size_s = prm["gross"] * (1 - prm["long_share"]) / prm["slots"]
    w = pd.DataFrame(longs * size_l - shorts * size_s, index=p.close.index,
                     columns=p.close.columns)
    return {"weights": w.shift(1, fill_value=0.0), "borrow": borrow,
            "retrim_k": prm["retrim_k"]}


def placebo(n: int = 20, tests=("T1", "T2"), overrides: dict | None = None) -> None:
    """Re-score RANDOM events on their reaction alone, as many per calendar quarter as the
    rule flags coarse, ``n`` times: how far does alpha move when that share of events loses
    its surprise for no reason? Compares the rule (U0f) and no rule (old U0)."""
    strat = lab.load_strategy(KEY, overrides)
    settings = lab._settings()
    keys = {k: strat.params[k] for k in EVENT_KEYS}
    g = strat.signals.__globals__
    rng = np.random.default_rng(2026)
    for t in tests:
        books = {"no rule": [], "rule": [], **{f"placebo {k}": [] for k in range(n)}}
        bench = []
        for i, (te0, _) in enumerate(lab.periods(t)):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            ev = g["events"](p, **keys)
            real = ev["coarse"].to_numpy(bool)
            if i == 0:                                  # the mask path reproduces signals()
                ref = strat.signals(p, **strat.params)["weights"]
                pd.testing.assert_frame_equal(_signals_with_mask(strat, p, real)["weights"], ref)
            quarter = ev["date"].dt.tz_convert(None).dt.to_period("Q").to_numpy()
            rate = pd.Series(real).groupby(quarter).transform("mean").to_numpy()
            masks = {"no rule": np.zeros(len(ev), bool), "rule": real}
            for k in range(n):
                masks[f"placebo {k}"] = rng.random(len(ev)) < rate
            for name, mask in masks.items():
                out = lab.book(strat, p, settings, te0, signals=_signals_with_mask(strat, p, mask))
                books[name].append(out["returns"])
                if name == "no rule":
                    bench.append(out["bench"])
            print(f"  {t} {te0}: {len(masks)} books (coarse share {real.mean():.3f})", flush=True)
        alpha = {k: lab.stitched(list(zip(v, bench, strict=True)))["alpha"]
                 for k, v in books.items()}
        pl = np.array([alpha[f"placebo {k}"] for k in range(n)])
        d = pl - alpha["no rule"]
        print(f"{t}: no rule (old U0) {alpha['no rule']:+.4f}, rule (U0f) {alpha['rule']:+.4f} "
              f"(Δ {alpha['rule'] - alpha['no rule']:+.4f}); {n} placebos: alpha mean "
              f"{pl.mean():+.4f} sd {pl.std(ddof=1):.4f}, Δ vs no rule mean {d.mean():+.4f} "
              f"sd {d.std(ddof=1):.4f} min {d.min():+.4f} max {d.max():+.4f}; the rule's Δ "
              f"beats {(d < alpha['rule'] - alpha['no rule']).mean():.0%} of the placebo Δs",
              flush=True)


def dropped(tests=None, overrides: dict | None = None) -> None:
    """The rule's coarse events scored on their reaction alone (the shipped rule) against the
    same events not scored at all, and against no rule, per test (base books only). Also the
    coarse events that entered the book, by side, and the future splitters among them (SEC
    shares >= 1.8x from the test's first period to 2026: a diagnostic proxy only)."""
    strat = lab.load_strategy(KEY, overrides)
    settings = lab._settings()
    keys = {k: strat.params[k] for k in EVENT_KEYS}
    g = strat.signals.__globals__
    last = lab.panel("T1", "2026-07-01", ["fund_shares"]).features["fund_shares"].ffill().iloc[-1]
    for t in tests or lab.OPEN:
        books, bench = {"no rule": [], "reaction alone": [], "dropped": []}, []
        sh0 = lab.panel(t, lab.periods(t)[0][0], ["fund_shares"]).features["fund_shares"]
        split = (last / sh0.ffill().iloc[-1]) >= 1.8
        side = {"long": [0, 0], "short": [0, 0]}
        for te0, te1 in lab.periods(t):
            p = lab.panel(t, te0, lab.strategy_features(strat))
            ev = g["events"](p, **keys)
            real = ev["coarse"].to_numpy(bool)
            for name, (mask, drop) in {"no rule": (np.zeros(len(ev), bool), False),
                                       "reaction alone": (real, False),
                                       "dropped": (real, True)}.items():
                sig = _signals_with_mask(strat, p, mask, drop)
                out = lab.book(strat, p, settings, te0, signals=sig)
                books[name].append(out["returns"])
                if name == "no rule":
                    bench.append(out["bench"])
                if name == "reaction alone":
                    w = sig["weights"]
                    inside = _window(ev, te0, te1)
                    for _, e in inside[inside["coarse"]].iterrows():
                        if e["bar"] + 1 >= len(w):
                            continue
                        x = w.iat[e["bar"] + 1, e["col"]]
                        if x != 0 and w.iat[e["bar"], e["col"]] == 0:      # it entered
                            k = "long" if x > 0 else "short"
                            side[k][0] += 1
                            side[k][1] += bool(split.get(e["symbol"], False))
        alpha = {k: lab.stitched(list(zip(v, bench, strict=True)))["alpha"]
                 for k, v in books.items()}
        print(f"{t}: " + ", ".join(f"{k} {v:+.4f}" for k, v in alpha.items())
              + f"; coarse entries long {side['long'][0]} (future splitters "
              f"{side['long'][1]}), short {side['short'][0]} ({side['short'][1]})", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "precision":
        precision()
    elif cmd == "tiny":
        tiny()
    elif cmd == "corr":
        corr(a[1], a[2] if len(a) > 2 else "v6_si")
    elif cmd == "timing":
        timing()
    elif cmd == "trials":
        trials(a[1], a[2] if len(a) > 2 else None)
    elif cmd == "rescore":
        rescore(a[1], int(a[2]) if len(a) > 2 else 10, a[3] if len(a) > 3 else None)
    elif cmd == "v1acct":
        r0, a = "--r0" in a, [x for x in a if x != "--r0"]
        v1acct(a[1], a[2], a[3].split(",") if len(a) > 3 else None, r0=r0)
    elif cmd == "warmup":
        warmup(a[1].split(",") if len(a) > 1 else ("E0",))
    elif cmd == "mechanics":
        mechanics(json.loads(a[1]) if len(a) > 1 else {})
    elif cmd == "book":
        tests = None
        if "--tests" in a:
            i = a.index("--tests")
            tests, a = a[i + 1].split(","), a[:i] + a[i + 2:]
        book(a[1:], tests)
    elif cmd == "paired":
        tests = None
        if "--tests" in a:
            i = a.index("--tests")
            tests, a = a[i + 1].split(","), a[:i] + a[i + 2:]
        paired(a[1], a[2:], tests)
    elif cmd == "anatomy":
        anatomy(a[1], a[2].split(",") if len(a) > 2 else None)
    elif cmd == "dropped":
        dropped(a[1].split(",") if len(a) > 1 else None, json.loads(a[2]) if len(a) > 2 else None)
    elif cmd == "placebo":
        placebo(int(a[1]) if len(a) > 1 else 20, a[2].split(",") if len(a) > 2 else ("T1", "T2"),
                json.loads(a[3]) if len(a) > 3 else None)
    else:
        raise SystemExit(__doc__)
