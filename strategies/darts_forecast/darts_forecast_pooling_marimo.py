import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Pooling: one model for all stocks, or one model per stock?

    One question only: **on the same stocks, days and inputs, does a model learned on one stock
    tell that stock's days apart better than one model learned on all stocks?** No calls, no
    profit, no luck bars: the three ways of learning are simply compared, stock by stock.

    - **pooled**: one model for all stocks, blind to which stock a row is
    - **pooled + stock**: one model that also reads the stock as a category, so a tree can
      split on a stock wherever enough of its rows say it behaves differently (the in-between)
    - **per stock**: one model per stock, learned from that stock's rows alone

    Every model is the same kind of small boosted tree, and each picks its own leaf size and
    number of rounds on its own held-out weeks, so none runs on settings sized for another.

    **The measure** is the **within-stock AUC** on VALIDATION: for one stock, how often the
    model rates a day that turned out positive above a day that did not. 0.5: it cannot tell
    them apart; 1.0: it always can. It is read inside each stock, so knowing that one stock
    moves more than another earns nothing; only telling a stock's own days apart counts.

    **Two targets**, at 13:30 for the move to 15:30 (the Hurdle and Movers question):

    - **moved**: did the stock move at least `THETA` percent, either way?
    - **up**: did it go up? Every day counts, no threshold
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
def _():
    import importlib.util
    import logging
    import warnings

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from trading_simple import strategies
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    warnings.filterwarnings("ignore")
    logging.getLogger("darts").setLevel(logging.ERROR)  # Darts' notes about optional models

    def _helpers(name):
        """One of the strategy folder's helper modules, loaded by path."""
        path = strategies.STRATEGIES_DIR / "darts_forecast" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"darts_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    lab, charts = _helpers("tsmixer_lab"), _helpers("charts")
    ml, hl, pl = _helpers("movers_lab"), _helpers("hurdle_lab"), _helpers("pooling_lab")
    ds = lab.ds  # daystart.py, as loaded by tsmixer_lab
    settings = load_settings()
    engine = make_engine(settings)
    return (
        charts,
        ds,
        engine,
        go,
        hl,
        lab,
        make_subplots,
        ml,
        np,
        pd,
        pl,
        settings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Settings

    1. **The question**: when the prediction is made, how far ahead, what counts as "moved"
    2. **The stocks and the data**: the same 40 active stocks and date ranges as the Hurdle
       notebook. TEST is never loaded here
    3. **The inputs and the models**: the Hurdle notebook's input table, and the held-out weeks
       that choose each model's leaf size and rounds
    4. **The planted check**: which input carries a planted effect, and how strong it is
    """)
    return


@app.cell
def _():
    # ── 1 · The question ──────────────────────────────────────────────
    DECISION = 240  # when the prediction is made: minutes after 09:30 (240 = 13:30)
    HORIZON = 120  # how far ahead, in minutes (120: to 15:30)
    THETA = 0.5  # "moved": at least this many percent, up or down ("up" needs no threshold)
    # ── 2 · The stocks and the data (dates: (first day, last day), both included) ─────────
    SYMBOLS = None  # None: the most active stocks of the screen below; or a list, e.g. ["DOCN"]
    UNIVERSE_N = 40  # how many stocks the screen keeps
    UNIVERSE_FROM = "2025-01-01"  # the stocks must have 1-minute bars since this day
    UNIVERSE_THETA = 1.0  # the screen keeps the stocks that most often moved this much (as Hurdle)
    TRAIN = ("2025-01-01", "2026-01-31")  # the models learn here
    VALIDATION = ("2026-02-01", "2026-07-30")  # every model is measured here
    READ_LOCKED = True  # allow the months the darts study series reserves (Feb 2026 on)
    # ── 3 · The inputs and the models ─────────────────────────────────
    HISTORY = 240  # minutes read before the decision (at most DECISION)
    PREV_DAYS = 1  # the sessions behind the "prev" day input
    GROUPS = ("activity", "day", "morning", "descriptors")  # hurdle_lab.GROUPS: the inputs
    HELD_EVERY = 4  # every 4th week of TRAIN chooses each model's leaf size and rounds
    # ── 4 · The planted check ─────────────────────────────────────────
    PLANT_INPUT = "ret30"  # the input that carries the planted effect (a column of Table 3)
    PLANT_STRENGTH = 0.5  # log-odds per standard deviation of that input
    return (
        DECISION,
        GROUPS,
        HELD_EVERY,
        HISTORY,
        HORIZON,
        PLANT_INPUT,
        PLANT_STRENGTH,
        PREV_DAYS,
        READ_LOCKED,
        SYMBOLS,
        THETA,
        TRAIN,
        UNIVERSE_FROM,
        UNIVERSE_N,
        UNIVERSE_THETA,
        VALIDATION,
    )


@app.cell(hide_code=True)
def _(DECISION, GROUPS, HISTORY, HORIZON, TRAIN, VALIDATION, ds, hl):
    _days = [*TRAIN, *VALIDATION]
    if _days != sorted(_days) or len(set(_days)) != len(_days):
        raise ValueError("TRAIN and VALIDATION must be (first, last) dates, in that order, "
                         "without overlap")
    if DECISION + HORIZON > ds.SESSION:
        raise ValueError(f"deciding at {ds.clock(DECISION)} + {HORIZON} min ends after the close")
    if not 0 < HISTORY <= DECISION:
        raise ValueError(f"HISTORY {HISTORY}: between 1 and {DECISION} minutes")
    _bad = sorted(set(GROUPS) - set(hl.GROUPS))
    if _bad:
        raise ValueError(f"GROUPS: unknown input groups {_bad}; known: {list(hl.GROUPS)}")
    CELL = (1, DECISION - HISTORY, HISTORY, HORIZON)  # the cell the input table uses
    return (CELL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The stocks

    The Movers / Hurdle screen: every stock with 1-minute bars since `UNIVERSE_FROM`, read on
    the TRAIN days only; stocks with gaps or under $1 million traded in the window are dropped,
    and the `UNIVERSE_N` that most often moved `UNIVERSE_THETA` percent stay (**Table 1**).
    """)
    return


@app.cell(hide_code=True)
def _(
    DECISION,
    HORIZON,
    SYMBOLS,
    TRAIN,
    UNIVERSE_FROM,
    UNIVERSE_N,
    UNIVERSE_THETA,
    engine,
    ml,
    mo,
    table,
):
    if SYMBOLS is None:
        _u = ml.universe(engine, start=UNIVERSE_FROM, train_end=TRAIN[1], decision=DECISION,
                         horizon=HORIZON, theta=UNIVERSE_THETA, n=UNIVERSE_N)
        STOCKS = list(_u.index)
        _out = mo.vstack([
            mo.md(f"**{len(STOCKS)}** stocks."),
            mo.md("### Table 1 · The stocks, most active first"), table(_u.round(3), rows=10)])
    else:
        STOCKS = list(SYMBOLS)
        _out = mo.md(f"`SYMBOLS` is set: **{len(STOCKS)}** stocks, no screen. {STOCKS}")
    mo.vstack([_out])
    return (STOCKS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The data, the inputs and the targets

    The 1-minute bars of every stock from TRAIN's first day to VALIDATION's last, and the Hurdle
    notebook's input table (`hurdle_lab.table`): one row per stock-day, every input known at the
    decision. The first run builds the market basket for these dates (a few minutes; saved).

    - **TRAIN**: every model learns here. Every `HELD_EVERY`-th week is held out first to choose
      the model's leaf size and number of rounds; then the model learns from all of TRAIN
    - **VALIDATION**: every model is measured here, stock by stock

    **Table 3** lists the inputs; **Table 4** the targets: how often each is 1, and how many
    stock-days each stock has (a per-stock model learns from its own row count only).
    """)
    return


@app.cell
def _(
    CELL,
    GROUPS,
    HELD_EVERY,
    PREV_DAYS,
    READ_LOCKED,
    STOCKS,
    THETA,
    TRAIN,
    VALIDATION,
    ds,
    engine,
    hl,
    lab,
    ml,
    mo,
    np,
    pd,
    pl,
    settings,
    table,
):
    _tune_end = ds.rs.WINDOWS["tune"][1]  # later months: reserved by the darts study series
    if VALIDATION[1] > _tune_end and not READ_LOCKED:
        raise ValueError(f"VALIDATION ends {VALIDATION[1]}, after {_tune_end}: set READ_LOCKED")
    _panel = ds.rs.load(settings, engine, start=TRAIN[0], end=VALIDATION[1], options=False,
                        symbols=STOCKS, adjust=True)
    _missing = sorted(set(STOCKS) - set(_panel.symbols))
    if _missing:
        raise ValueError(f"no 1-minute data for {_missing}: drop them from SYMBOLS")
    _F = lab.build(_panel, engine, basket_from=TRAIN[0], basket_to=VALIDATION[1], options=False)
    ml.with_opens(_F, ds.full_clock(_panel).open_)  # the move starts at the first trade
    _dates = np.array(_F["dates"], dtype=object)
    _sp = {"sessions": list(_dates)}
    for _k, (_a, _b) in {"train": TRAIN, "validation": VALIDATION}.items():
        _in = [x for x in _dates if _a <= x <= _b]
        if not _in:
            raise ValueError(f"no stored session in {_k.upper()} {_a} → {_b}")
        _sp[_k] = (_in[0], _in[-1])
    _f, _h, val_d = ds.split_days(_F["pp"], _sp, "validation", TRAIN[0])
    _train = np.r_[_f, _h]
    fit_d, held_d = hl.spread_held(_F["dates"], _train, HELD_EVERY)
    X, COLUMNS, _, ROW_DAY, ROW_STOCK, _ = hl.table(
        lab, _F, CELL, np.r_[_train, val_d], GROUPS, prev_days=PREV_DAYS, theta=THETA)
    Y = pl.targets(ml.moves(_F, CELL)[ROW_DAY, ROW_STOCK], THETA)
    NAMES = list(_F["names"])
    _span = lambda i: (_dates[i[0]], _dates[i[-1]])  # noqa: E731
    _split = pd.DataFrame([
        {"part": p, "from": _span(i)[0], "to": _span(i)[1], "sessions": len(i),
         "stock-days": int(np.isin(ROW_DAY, i).sum()), "use": u}
        for p, i, u in (("history only", np.arange(_train[0]), "typical sizes, never learned"),
                        ("TRAIN", _train, "every model learns"),
                        ("TRAIN, held-out weeks", held_d,
                         f"every {HELD_EVERY}th week: each model's leaf size and rounds"),
                        ("VALIDATION", val_d, "every model is measured"))])
    _in_train, _in_val = np.isin(ROW_DAY, _train), np.isin(ROW_DAY, val_d)
    _per_stock = np.bincount(ROW_STOCK[_in_train], minlength=len(NAMES))
    _targets = pd.DataFrame([
        {"target": t, "what": w, "share of 1s, TRAIN": float(np.nanmean(Y[t][_in_train])),
         "share of 1s, VALIDATION": float(np.nanmean(Y[t][_in_val])),
         "TRAIN stock-days per stock (median)": int(np.median(_per_stock)),
         "TRAIN 1s per stock (median)": int(np.median(np.bincount(
             ROW_STOCK[_in_train], weights=np.nan_to_num(Y[t][_in_train]),
             minlength=len(NAMES))))}
        for t, w in (("moved", f"moved at least {THETA:g}% either way"),
                     ("up", "went up (every day)"))])
    mo.vstack([
        mo.md(f"**{len(NAMES)}** stocks, **{len(_dates)}** sessions loaded "
              f"({_dates[0]} → {_dates[-1]}); **{len(X):,}** stock-days."),
        mo.md("### Table 2 · The data split"), table(_split),
        mo.md(f"### Table 3 · The inputs ({len(COLUMNS)} columns)"),
        mo.md(", ".join(f"`{c}`" for c in COLUMNS)),
        mo.md("### Table 4 · The targets"), table(_targets.round(3))])
    return COLUMNS, NAMES, ROW_DAY, ROW_STOCK, X, Y, fit_d, held_d, val_d


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Head to head

    Each approach learns each target on TRAIN, then rates every VALIDATION stock-day. **Table
    5** gives, per target and approach, the within-stock AUC averaged over the stocks, its
    median, and on how many stocks the approach is the best of the three. **Figure 1** puts
    every stock on one dot: across, its AUC from the pooled model; up, from the per-stock model
    (and from pooled + stock). A dot above the grey line is a stock where that approach beats
    the pooled model. About a minute per target.
    """)
    return


@app.cell
def _(mo):
    run_compare = mo.ui.run_button(label="compare the approaches")
    mo.hstack([run_compare], justify="start")
    return (run_compare,)


@app.cell(hide_code=True)
def _(NAMES, ROW_DAY, ROW_STOCK, X, Y, fit_d, held_d, mo, np, pd, pl, run_compare, table,
      val_d):
    mo.stop(not run_compare.value, mo.md("_Press **compare the approaches**._"))
    RES = {}
    with mo.status.spinner(title="Fitting the three approaches (about a minute per target) ..."):
        for _t in pl.TARGETS:
            RES[_t] = pl.compare(X, Y[_t], ROW_STOCK, ROW_DAY, fit_d, held_d, val_d, len(NAMES))

    def _chosen(info):
        """The leaf size and rounds a model chose (per stock: the medians)."""
        if isinstance(info, dict):
            return f"leaf {info['leaf']}, {info['rounds']} rounds"
        fits = [i for i in info if i is not None]
        return (f"median leaf {np.median([i['leaf'] for i in fits]):.0f}, "
                f"median {np.median([i['rounds'] for i in fits]):.0f} rounds")

    _rows, _words = [], []
    for _t, _r in RES.items():
        _auc = _r["auc"]
        _full = _auc.dropna()
        _best = _full.idxmax(axis=1).value_counts()
        for _a in pl.APPROACHES:
            _rows.append({"target": _t, "approach": _a, "mean AUC": _auc[_a].mean(),
                          "median AUC": _auc[_a].median(),
                          "best of the three on": f"{_best.get(_a, 0)} of {len(_full)} stocks",
                          "chosen": _chosen(_r["info"][_a])})
        _m = _auc.mean()
        _words.append(f"- **{_t}**: pooled **{_m['pooled']:.3f}**, pooled + stock "
                      f"**{_m['pooled + stock']:.3f}**, per stock **{_m['per stock']:.3f}**; "
                      f"per stock beats pooled on **{(_full['per stock'] > _full['pooled']).sum()}"
                      f"** of {len(_full)} stocks")
    SUMMARY = pd.DataFrame(_rows)
    mo.vstack([
        mo.md("Mean within-stock AUC on VALIDATION (0.5: cannot tell a stock's days apart):\n\n"
              + "\n".join(_words)),
        mo.md("### Table 5 · The three approaches, per target"), table(SUMMARY.round(3))])
    return (RES,)


@app.cell(hide_code=True)
def _(NAMES, RES, charts, go, make_subplots, mo, np, pd, pl, table):
    _all = np.concatenate([r["auc"].to_numpy().ravel() for r in RES.values()])
    _lo = min(0.3, float(np.nanmin(_all)) - 0.02)
    _hi = max(0.7, float(np.nanmax(_all)) + 0.02)
    _fig = make_subplots(rows=1, cols=len(RES), subplot_titles=[f"target: {t}" for t in RES],
                         horizontal_spacing=0.08)
    for _c, (_t, _r) in enumerate(RES.items(), start=1):
        _auc = _r["auc"]
        _fig.add_trace(go.Scatter(x=[_lo, _hi], y=[_lo, _hi], mode="lines", showlegend=False,
                                  line={"color": charts._MUTED, "dash": "dash", "width": 1},
                                  hoverinfo="skip"), row=1, col=_c)
        for _a, _col in (("per stock", charts.SERIES[2]), ("pooled + stock", charts.SERIES[1])):
            _fig.add_trace(go.Scatter(
                x=_auc["pooled"], y=_auc[_a], mode="markers", name=_a, legendgroup=_a,
                showlegend=_c == 1, text=NAMES,
                marker={"color": _col, "size": 9, "line": {"color": "white", "width": 1}},
                hovertemplate=f"%{{text}}<br>pooled %{{x:.3f}}<br>{_a} %{{y:.3f}}"
                              "<extra></extra>"), row=1, col=_c)
        _fig.update_xaxes(range=[_lo, _hi], title_text="AUC, pooled", row=1, col=_c)
        _fig.update_yaxes(range=[_lo, _hi], title_text="AUC, the other approach", row=1, col=_c)
    charts._base(_fig, title="Figure 1 · Each stock's AUC: pooled (across) against per stock and "
                 "pooled + stock (up); above the grey line = better than pooled", height=480)
    charts.below(_fig)
    _detail = pd.DataFrame({"stock": NAMES, "VALIDATION stock-days": RES["moved"]["rows"],
                            "moved share": RES["moved"]["rate"]})
    for _t, _r in RES.items():
        for _a in pl.APPROACHES:
            _detail[f"{_t}: {_a}"] = _r["auc"][_a].to_numpy()
    mo.vstack([_fig, mo.md("### Table 6 · Every stock's AUC, per target and approach"),
               table(_detail.round(3), rows=10)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · Transfer: does a stock's own model know it best?

    Every per-stock model rates the VALIDATION days of **every** stock. **Figure 2** (one map per
    target): a row is the stock whose model is used, a column the stock it rates; blue is better
    than guessing (AUC above 0.5), red worse. The diagonal is each stock's own model on its own
    days.

    - If the stocks are **different**, the diagonal stands out: a stock's own model is the best
      one for it, and the other stocks' models do worse there
    - If they **share** their patterns, the diagonal does not stand out: other stocks' models
      rate a stock about as well as its own, which is what a pooled model uses

    **Table 7** puts it in numbers per stock: its own model, the average of the other stocks'
    models on its days, and the pooled model.
    """)
    return


@app.cell(hide_code=True)
def _(NAMES, RES, charts, go, mo, np, pd, table):
    _figs, _cols, _words = [], {"stock": NAMES}, []
    _off = ~np.eye(len(NAMES), dtype=bool)
    for _t, _r in RES.items():
        _m = _r["transfer"]
        _lim = max(0.1, float(np.nanmax(np.abs(_m - 0.5))))
        _fig = go.Figure(go.Heatmap(
            z=_m, x=NAMES, y=NAMES, colorscale=charts.DIVERGING, zmin=0.5 - _lim,
            zmax=0.5 + _lim, xgap=1, ygap=1, colorbar={"title": {"text": "AUC"}},
            hovertemplate="model of %{y}, rating %{x}'s days: AUC %{z:.3f}<extra></extra>"))
        charts._base(_fig, title=f"Figure 2 · Target {_t}: every stock's model (row) rating every "
                     "stock's VALIDATION days (column)", height=760)
        _fig.update_xaxes(type="category", tickangle=-60, title_text="the stock rated")
        _fig.update_yaxes(type="category", autorange="reversed",
                          title_text="the stock whose model rates")
        _figs.append(_fig)
        _own = np.diag(_m)
        _others = np.array([np.nanmean(_m[_off[:, j], j]) for j in range(len(NAMES))])
        _cols |= {f"{_t}: own model": _own, f"{_t}: other stocks' models": _others,
                  f"{_t}: pooled": _r["auc"]["pooled"].to_numpy()}
        _ok = np.isfinite(_own) & np.isfinite(_others)
        _words.append(f"- **{_t}**: a stock's own model beats the other stocks' models on its "
                      f"days on **{(_own[_ok] > _others[_ok]).sum()} of {_ok.sum()}** stocks, by "
                      f"**{np.mean(_own[_ok] - _others[_ok]):+.3f}** AUC on average")
    mo.vstack([*_figs, mo.md("\n".join(_words)),
               mo.md("### Table 7 · Own model, other stocks' models, pooled model"),
               table(pd.DataFrame(_cols).round(3), rows=10)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · The planted check: what each answer looks like

    The same real stocks, days and inputs, with **planted** labels in place of the real ones, so
    the right answer is known. The labels follow the input `PLANT_INPUT` (standardized inside
    each stock) with strength `PLANT_STRENGTH`, at the real "moved" share:

    - **none**: the labels ignore every input. All three approaches should sit near 0.5; the
      spread of the per-stock AUCs over the stocks shows how far apart two numbers land when
      there is nothing to find
    - **shared**: every stock follows the input the same way. Pooling should win: the same
      pattern, learned from all stocks' rows
    - **per stock**: half of the stocks follow the input, the other half the opposite way. The
      blind pooled model should lose (the halves cancel, unless other inputs tell them apart);
      per stock and pooled + stock should find it, and the transfer diagonal should stand out

    If the real targets in sections 1 and 2 look like the **shared** row, one model for all
    stocks is the better focus; if they look like the **per stock** row, models per stock are.
    About three minutes.
    """)
    return


@app.cell
def _(mo):
    run_plant = mo.ui.run_button(label="run the planted check")
    mo.hstack([run_plant], justify="start")
    return (run_plant,)


@app.cell(hide_code=True)
def _(
    COLUMNS,
    NAMES,
    PLANT_INPUT,
    PLANT_STRENGTH,
    ROW_DAY,
    ROW_STOCK,
    X,
    Y,
    fit_d,
    held_d,
    mo,
    np,
    pd,
    pl,
    run_plant,
    table,
    val_d,
):
    mo.stop(not run_plant.value, mo.md("_Press **run the planted check**._"))
    if PLANT_INPUT not in COLUMNS:
        raise ValueError(f"PLANT_INPUT {PLANT_INPUT!r} is not an input; one of {COLUMNS}")
    _n, _off = len(NAMES), ~np.eye(len(NAMES), dtype=bool)
    _rows = []
    with mo.status.spinner(title="Planting and fitting (about a minute each) ..."):
        for _kind in pl.PLANTS:
            _y = pl.plant(X, ROW_STOCK, COLUMNS.index(PLANT_INPUT), _kind,
                          strength=PLANT_STRENGTH, rate=float(np.nanmean(Y["moved"])),
                          n_stocks=_n)
            _r = pl.compare(X, _y, ROW_STOCK, ROW_DAY, fit_d, held_d, val_d, _n)
            _auc, _m = _r["auc"], _r["transfer"]
            _own = np.diag(_m)
            _others = np.array([np.nanmean(_m[_off[:, j], j]) for j in range(_n)])
            _rows.append({"planted": _kind, **{_a: _auc[_a].mean() for _a in pl.APPROACHES},
                          "per stock beats pooled on": int((_auc["per stock"]
                                                            > _auc["pooled"]).sum()),
                          "own model minus others'": float(np.nanmean(_own - _others)),
                          "per-stock AUC spread over stocks (SD)": _auc["per stock"].std()})
    mo.vstack([
        mo.md("### Table 8 · Planted labels: mean within-stock AUC per approach"),
        table(pd.DataFrame(_rows).round(3))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reading the answer

    Per target, set sections 1 and 2 beside the planted rows of Table 8:

    - **One model for all stocks**: pooled is as good as or better than per stock on most
      stocks, pooled + stock adds nothing, and the transfer diagonal does not stand out. The
      stocks share what is learnable; one model sees all of it
    - **One model per stock**: per stock (and pooled + stock) clearly above pooled on most
      stocks, and each stock's own model beats the other stocks' models on its days
    - **In between**: pooled + stock the best of the three. Shared patterns plus stock-specific
      adjustments: keep one model and let it read the stock
    - **Nothing either way**: all three near 0.5, like the planted "none" row. For that target
      there is nothing to pool or to split; the choice does not matter until an input that
      knows something comes along

    The per-stock models learn from a few hundred rows each, the pooled ones from all of them:
    the planted "shared" row shows what that difference costs when the stocks really are alike.
    """)
    return


if __name__ == "__main__":
    app.run()
