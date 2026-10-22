import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # informed_flow_ls · the nested walk-forward and the verdict

    The old `informed_flow` learned that **an optimizer tuned on backtest profit finds luck**
    (v26: 4,000 trials, +12.6%/yr in sample, −10.8%/yr out of sample). So here the optimizer is
    judged by what it does UNSEEN, never by what it finds:

    1. **The plan**: the periods, the arms, the gates — written down in `IDEAS.md`
       ("Pre-registration v1") before this notebook ever ran
    2. **The walk-forward**: every 6 months from January 2022 an Optuna study tunes the four
       searched knobs on that period's PAST only (point-in-time universe, scored on its worst
       stretches, not its total), and the pick trades the next 6 months unseen — beside the
       untuned config and the controls. Cached per period: rerunning is instant
    3. **What tuning chose**, period by period, and how much each study was overfitting
       (the probability of backtest overfitting)
    4. **The stitched out-of-sample record** of every arm, and the SPY-hedged long book
    5. **Luck**: the same rules holding random names
    6. **The verdict**: every pre-registered gate, and one `SCORECARD` line
    7. **Paper trading**: the book to hold from 2026-10-20 — the only data no version has read

    §2 takes about an hour the first time (10 studies of 80 trials); §5 another 15 minutes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def table(df, rows=15):
        """A DataFrame as a marimo table; a named index becomes a column."""
        return mo.ui.table(
            df.reset_index() if df.index.name else df, selection=None, page_size=rows
        )

    return (table,)


@app.cell
def _(mo):
    reload_helpers = mo.ui.run_button(label="reload research.py (after it changed)")
    mo.hstack([reload_helpers], justify="start")
    return (reload_helpers,)


@app.cell
def _(reload_helpers):
    import importlib.util
    import json

    import numpy as np
    import pandas as pd
    from plotly.subplots import make_subplots

    from trading_simple import backtest as bt
    from trading_simple import strategies
    from trading_simple.config import PROJECT_ROOT, load_settings
    from trading_simple.data import load_panel
    from trading_simple.db import make_engine
    from trading_simple.plots import SERIES
    from trading_simple.plots import _base as layout
    from trading_simple.plots import _scatter as scatter

    _ = reload_helpers.value
    _spec = importlib.util.spec_from_file_location(
        "informed_flow_ls_research", strategies.STRATEGIES_DIR / "informed_flow_ls" / "research.py")
    rs = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rs)
    strat = strategies.load("informed_flow_ls")
    settings = load_settings()
    engine = make_engine(settings)
    return (PROJECT_ROOT, SERIES, bt, engine, json, layout, load_panel, make_subplots, np, pd,
            rs, scatter, settings, strat)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · The plan

    **Periods** (Table 1): train on everything before the test start (minus 21 sessions:
    the purge), test on the next 6 months. **Arms** (Table 2): the tuned pick is the PRIMARY —
    the procedure chosen for this version; the others are controls and stresses on the same
    days. **Gates** (Table 3): copied from `research.GATES`; the text is in `IDEAS.md`.
    """)
    return


@app.cell
def _(pd, rs, settings, strat, table, mo):
    LAST = "2026-10-16"
    N_TRIALS = strat.optuna["n_trials"]
    N_LUCK = 100
    FOLDS = rs.folds(LAST)
    _p = strat.params
    ARMS = {
        "tuned": ("tuned", {}, None),
        "untuned": ("config", {}, None),
        "long side": ("config", {"sides": "long"}, None),
        "no options": ("tuned", {"w_skew": 0.0}, None),
        "2x costs": ("tuned", {}, rs.with_costs(settings, rs.GATES["cost_multiple"])),
        "2x borrow": ("tuned", {k: _p[k] * rs.GATES["borrow_multiple"]
                                for k in ("borrow_gc", "borrow_mid", "borrow_high")}, None),
        "late fills": ("tuned", {}, rs.with_costs(settings, 1.0, rs.GATES["late_fill_bps"])),
        "untuned 2x costs": ("config", {}, rs.with_costs(settings, rs.GATES["cost_multiple"])),
        "untuned 2x borrow": ("config", {k: _p[k] * rs.GATES["borrow_multiple"]
                                         for k in ("borrow_gc", "borrow_mid", "borrow_high")},
                              None),
        "untuned late fills": ("config", {}, rs.with_costs(settings, 1.0,
                                                          rs.GATES["late_fill_bps"])),
    }
    _folds = pd.DataFrame(FOLDS, columns=["train ends", "test starts", "test ends"])
    _folds.index.name = "period"
    _arms = pd.DataFrame([(k, b, ", ".join(f"{x}={y}" for x, y in o.items()) or "-",
                           "costs changed" if c is not None else "-")
                          for k, (b, o, c) in ARMS.items()],
                         columns=["arm", "starts from", "overrides", "settings"])
    _gates = pd.DataFrame(list(rs.GATES.items()), columns=["gate", "bar"])
    mo.vstack([mo.md(f"**Table 1 · {len(FOLDS)} periods** · {N_TRIALS} trials each · search "
                     f"space `{strat.optuna['space']}`"), table(_folds),
               mo.md("**Table 2 · Arms**"), table(_arms),
               mo.md("**Table 3 · Gate settings**"), table(_gates)])
    return ARMS, FOLDS, LAST, N_LUCK, N_TRIALS


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · The walk-forward

    Each period: load the training panel (every past membership, point in time), run the
    study (resumable: `studies/<study>-wf-<date>.log`), then trade the test period for every
    arm. Results are cached in `studies/wf-<study>-<date>.pkl`; an arm added later runs alone.
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="run (or load) the walk-forward")
    mo.hstack([run_button], justify="start")
    return (run_button,)


@app.cell
def _(ARMS, FOLDS, N_TRIALS, engine, mo, rs, run_button, settings, strat):
    mo.stop(not run_button.value, mo.md("_Click the button to run or load the walk-forward._"))
    RESULTS = []
    with mo.status.progress_bar(total=len(FOLDS), title="walk-forward periods") as _bar:
        for _fold in FOLDS:
            RESULTS.append(rs.period(engine, settings, strat, _fold, n_trials=N_TRIALS,
                                     arms=ARMS))
            _bar.update(subtitle=f"done: test from {_fold[1]}")
    return (RESULTS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · What tuning chose

    Table 4: per period, the tuned values, the study's best training score (`robust`: mean
    stretch alpha minus half their spread), its **probability of backtest overfitting** (PBO:
    how often the training winner of one half of the training window lands in the bottom half
    on the other half; 0.5 = a coin flip), and every arm's alpha in the unseen test period.
    Values that jump from period to period are what the old version's refits did (v26: "the
    refits agreed on nothing").
    """)
    return


@app.cell
def _(RESULTS, bt, pd, rs):
    def alpha(pair):
        r, b = pair
        return float(rs.alpha_of(r.to_numpy(), b.to_numpy(), bt._years(r.index))[3][0])

    _rows = []
    for _res in RESULTS:
        _row = {"test": f"{_res['fold'][1][:7]} → {_res['fold'][2][:7]}", **_res["tuned"],
                "train score": _res["score"], "PBO": _res["pbo"]}
        _row |= {f"alpha · {k}": alpha(v) for k, v in _res["returns"].items()}
        _rows.append(_row)
    PERIODS = pd.DataFrame(_rows).set_index("test")
    return PERIODS, alpha


@app.cell
def _(PERIODS, mo, table):
    mo.vstack([mo.md("**Table 4 · Period by period** (alpha = yearly, after beta, in the "
                     "unseen test period)"), table(PERIODS.round(3), rows=12)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · The stitched out-of-sample record

    Every arm's test periods joined into one record (about 4.8 years), against buying and
    holding the same names. Alpha = CAGR − beta × buy & hold CAGR; P(alpha > 0) from a block
    bootstrap (21-day blocks, book and market resampled together). The **long side + SPY
    hedge** control holds the long side alone (half the book) and sells SPY short for the same
    amount, every day: the cheapest way to be market-neutral without shorting single names.
    """)
    return


@app.cell
def _(LAST, RESULTS, engine, load_panel, pd, rs, settings, strat):
    STITCHED = {k: rs.stitched([res["returns"][k] for res in RESULTS])
                for k in RESULTS[0]["returns"]}
    _spy = load_panel(engine, ["SPY"], "1d", "2021-06-01", LAST,
                      exchanges=settings.exchanges).close["SPY"].pct_change()
    _long = STITCHED["long side"]["returns"]
    _hedge = _long - strat.params["gross"] / 2 * _spy.reindex(_long.index).fillna(0.0)
    STITCHED["long side + SPY hedge"] = rs.stitched([(_hedge, STITCHED["long side"]["bench"])])
    _keys = ("years", "cagr", "buy & hold", "beta", "alpha", "P(alpha > 0)", "alpha p5",
             "alpha p95", "volatility", "max drawdown")
    SUMMARY = pd.DataFrame({k: {c: v[c] for c in _keys} for k, v in STITCHED.items()}).T
    SUMMARY.index.name = "arm"
    return STITCHED, SUMMARY


@app.cell
def _(SERIES, STITCHED, SUMMARY, layout, make_subplots, mo, scatter, table):
    _fig = make_subplots(rows=1, cols=1)
    _show = ("tuned", "untuned", "long side + SPY hedge", "no options")
    for _i, _k in enumerate(_show):
        _fig.add_trace(scatter((1 + STITCHED[_k]["returns"]).cumprod(), mode="lines", name=_k,
                               line={"color": SERIES[_i], "width": 2 if _i == 0 else 1.5}))
    _fig.add_trace(scatter((1 + STITCHED["tuned"]["bench"]).cumprod(), mode="lines",
                           name="buy & hold", line={"color": SERIES[5], "width": 1, "dash": "dot"}))
    _fig.update_xaxes(type="date")
    mo.vstack([mo.md("**Table 5 · Every arm, stitched**"), table(SUMMARY.round(3), rows=12),
               mo.md("**Figure 1 · Growth of $1 out of sample** (each period's test months only)"),
               layout(_fig, title="Stitched walk-forward", height=460)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Luck

    100 books per period that keep EVERY rule of the primary (its tuned values, bands, slots,
    borrow, costs) but hold random names: the composite's columns relabelled at random within
    dollar-volume tenths, so turnover and liquidity match and only the choice of names is
    random. Stitched like the primary. Cached per period (`studies/luck-*.pkl`).
    """)
    return


@app.cell
def _(mo):
    luck_button = mo.ui.run_button(label="run (or load) the luck band")
    mo.hstack([luck_button], justify="start")
    return (luck_button,)


@app.cell
def _(FOLDS, N_LUCK, RESULTS, bt, engine, luck_button, mo, np, rs, settings, strat):
    mo.stop(not luck_button.value, mo.md("_Click the button to run or load the luck band._"))
    _per = []
    with mo.status.progress_bar(total=len(FOLDS), title="random-name books") as _bar:
        for _fold, _res in zip(FOLDS, RESULTS, strict=True):
            _per.append(rs.shuffled_returns(strat, settings, engine, _fold, _res["tuned"],
                                            N_LUCK))
            _bar.update()
    luck = []
    for _k in range(N_LUCK):
        _r = [(_per[_i][_k], RESULTS[_i]["returns"]["tuned"][1]) for _i in range(len(FOLDS))]
        _s = rs.stitched(_r)
        luck.append(_s["alpha"])
    luck = np.array(luck)
    _ = bt
    return (luck,)


@app.cell
def _(luck, SERIES, STITCHED, layout, mo, np):
    import plotly.graph_objects as _go
    _fig = _go.Figure(_go.Histogram(x=luck, nbinsx=30, marker={"color": SERIES[0]},
                                    name="random-name books"))
    _a = STITCHED["tuned"]["alpha"]
    _fig.add_vline(x=_a, line={"color": SERIES[5], "width": 2},
                   annotation_text=f"tuned book {_a:+.1%}")
    _fig.update_xaxes(tickformat=".0%", title="stitched alpha per year")
    mo.vstack([mo.md(f"**Figure 2 · The primary against {len(luck)} random-name books**: it "
                     f"beats {np.mean(luck < _a):.0%} of them."),
               layout(_fig, title="Luck band", height=380)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · The verdict

    Every gate of `IDEAS.md` ("Pre-registration v1"). Hard gates decide; warnings are shown.
    The `SCORECARD` line is what to paste into the README's study history.
    """)
    return


@app.cell
def _(luck, PERIODS, RESULTS, STITCHED, json, mo, rs, strat, table):
    _stress = {k: STITCHED[k] for k in ("2x costs", "2x borrow", "late fills")}
    _controls = {k: STITCHED[k] for k in ("long side + SPY hedge", "no options")}
    VERDICT_TABLE, VERDICT = rs.verdict(
        STITCHED["tuned"], STITCHED["untuned"], luck, [r["pbo"] for r in RESULTS], _stress,
        list(PERIODS["alpha · tuned"]), _controls)
    _score = {"study": strat.optuna["study"], "verdict": VERDICT,
              "wf": {k: {c: round(float(v[c]), 4) for c in ("cagr", "beta", "alpha",
                                                             "P(alpha > 0)", "max drawdown")}
                     for k, v in STITCHED.items()},
              "luck_percentile": round(float((luck < STITCHED["tuned"]["alpha"]).mean()), 3),
              "median_pbo": round(float(PERIODS["PBO"].median()), 3),
              "tuned": [r["tuned"] for r in RESULTS]}
    SCORECARD = "SCORECARD " + json.dumps(_score, default=str)
    _kind = "success" if VERDICT == "PASS" else "danger"
    mo.vstack([mo.callout(mo.md(f"## Verdict: **{VERDICT}**"), kind=_kind),
               table(VERDICT_TABLE.round(4), rows=20),
               mo.md(f"```\n{SCORECARD}\n```")])
    return SCORECARD, VERDICT


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6b · The untuned config as the book

    Every nested run of v2-v4 found the tuned picks no better than the config's own values
    (RESEARCH_LOG.md §23), so the book that matters is the untuned one. Its gates (the two
    procedure gates — tuning beats not tuning, overfitting — do not apply to a book nothing
    tunes; its luck band is `untuned_verdict.py`'s, recorded in the log).
    """)
    return


@app.cell
def _(STITCHED, mo, np, pd, table):
    _u = STITCHED["untuned"]
    _rows = [("walk-forward alpha > 0", _u["alpha"], _u["alpha"] > 0),
             ("bootstrap P(alpha > 0) >= 0.90", _u["P(alpha > 0)"], _u["P(alpha > 0)"] >= 0.9),
             ("|beta| <= 0.3", abs(_u["beta"]), abs(_u["beta"]) <= 0.3),
             *[(f"alpha > 0 with {k[8:]}", STITCHED[k]["alpha"], STITCHED[k]["alpha"] > 0)
               for k in ("untuned 2x costs", "untuned 2x borrow", "untuned late fills")],
             ("(warning) max drawdown >= -25%", _u["max drawdown"], _u["max drawdown"] >= -0.25),
             ("(warning) beats long side + SPY hedge",
              _u["alpha"] - STITCHED["long side + SPY hedge"]["alpha"],
              _u["alpha"] > STITCHED["long side + SPY hedge"]["alpha"])]
    _t = pd.DataFrame(_rows, columns=["gate", "value", "passed"])
    _ = np
    mo.vstack([mo.md("**Table 5b · The untuned config's gates**"), table(_t.round(4), rows=10)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7 · Paper trading from 2026-10-20

    The only data no version has read is the future. Pick which book to paper trade: the
    **config's own values** (the default since 2026-10-20: in every nested run of v2-v4 the
    tuned picks failed to beat the untuned config, RESEARCH_LOG.md §19-§23) or the latest
    period's **tuned** values (the pre-registered primary; the next re-tuning would be due in
    January 2027). Either way on that period's members, signals through the last stored
    session; orders market-on-open for the next session. Shorts are charged Interactive
    Brokers' own borrow fee when `scripts/ibkr_shortstock.py` saved this morning's file (run it
    daily before the open), the calibrated tiers otherwise. **Save** writes the book to
    `var/informed_flow_ls/paper/book-<date>.csv`, so the paper record can be compared with the
    backtest's own trades later.
    """)
    return


@app.cell
def _(mo):
    paper_choice = mo.ui.dropdown(["config values", "latest tuned values"],
                                  value="config values", label="book to paper trade")
    mo.hstack([paper_choice], justify="start")
    return (paper_choice,)


@app.cell
def _(LAST, RESULTS, engine, mo, paper_choice, pd, rs, settings, strat):
    _tuned = paper_choice.value == "latest tuned values"
    _params = {**strat.params, **(RESULTS[-1]["tuned"] if _tuned else {})}
    _te0 = RESULTS[-1]["fold"][1]
    _panel = rs.load(engine, settings, strat, LAST, rs.members(engine, _te0))
    _mod = strat.signals.__globals__
    _parts = _mod["compose"](_panel, **rs.compose_params(_mod, _params))
    _longs, _shorts, _ = _mod["_select"](_parts["score"], _panel.close, _parts["shortable"],
                                         q_in=_params["q_in"], q_out=_params["q_out"],
                                         slots=(_params["long_slots"], _params["short_slots"]),
                                         max_hold=_params["max_hold"],
                                         short_stop=_params["short_stop"])
    _day = _panel.close.index[-1]
    _gl = _params["gross"] * _params["long_share"] / _params["long_slots"]
    _gs = _params["gross"] * (1 - _params["long_share"]) / _params["short_slots"]
    _held = [(s, "long", _gl) for s in _longs.columns[_longs.loc[_day]]] + \
            [(s, "short", -_gs) for s in _shorts.columns[_shorts.loc[_day]]]
    BOOK = pd.DataFrame(_held, columns=["symbol", "side", "weight"]).set_index("symbol")
    BOOK["rank"] = _parts["score"].loc[_day, BOOK.index]
    BOOK["days to cover"] = _panel.features["short_interest"].loc[_day, BOOK.index]
    BOOK["short interest / shares"] = _parts["si_share"].loc[_day, BOOK.index]
    BOOK["borrow / yr"] = _parts["borrow"].loc[_day, BOOK.index].where(BOOK["side"] == "short")
    _ibkr = _panel.features.get("borrow_fee")         # IBKR's own rate (scripts/ibkr_shortstock.py)
    BOOK["borrow source"] = [
        "" if side == "long" else
        "IBKR" if _ibkr is not None and pd.notna(_ibkr.at[_day, sym]) else "tier"
        for sym, side in zip(BOOK.index, BOOK["side"], strict=True)]
    BOOK["close"] = _panel.close.loc[_day, BOOK.index]
    BOOK = BOOK.sort_values(["side", "rank"], ascending=[True, False])
    mo.md(f"**Table 6 · The book after the close of {_day.date()}** — {len(BOOK)} positions, "
          f"{paper_choice.value}" + (f" `{RESULTS[-1]['tuned']}`" if _tuned else ""))
    return (BOOK,)


@app.cell
def _(BOOK, mo, table):
    save_button = mo.ui.run_button(label="save this book for the paper record")
    mo.vstack([table(BOOK.round(4), rows=30), mo.hstack([save_button], justify="start")])
    return (save_button,)


@app.cell
def _(BOOK, LAST, PROJECT_ROOT, mo, save_button):
    mo.stop(not save_button.value)
    _path = PROJECT_ROOT / "var" / "informed_flow_ls" / "paper" / f"book-{LAST}.csv"
    _path.parent.mkdir(parents=True, exist_ok=True)
    BOOK.to_csv(_path)
    mo.md(f"Saved `{_path}`.")
    return


if __name__ == "__main__":
    app.run()
