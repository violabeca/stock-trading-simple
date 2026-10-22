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
    # Hurdle: the same question as Movers, answered by small trees

    At **13:30** every day, for each stock: will it move by the end of the horizon (**15:30**) by at
    least `THETA` percent, and up or down? The **Movers** notebook answers with one TSMixer that
    reads the raw morning. This notebook answers the *same* question on the *same* stock-days, with
    the *same* labels, calls and scoring, using **gradient-boosted trees on a short table of
    numbers**. It is the control for the TSMixer: if trees on a few summaries cannot beat the
    activity rule, a network on the raw path has to show it can do more.

    The model is a **hurdle**, two small questions instead of one:

    1. **Will it move?** `P(move)`: a yes / no tree trained on every stock-day
    2. **Which way, if it moves?** `P(up | move)`: trained only on the stock-days that moved

    Each day the `K` stocks with the highest `P(move)` are called, each long if `P(up | move)` is
    above one half, else short. A **direct** three-class tree (down / flat / up in one model) runs
    beside it as a control.

    - **every stock-day counts once**: the labels are classes, so a 5% move weighs as much as a 1%
      move
    - **no search**: the trees' knobs are fixed (shallow, slow, regularized); only the *inputs*
      change, in a handful of steps, and the best of those few is compared with the luck bar
    - **rules that need no model**: random picks, and the `K` stocks with the biggest average move
      over the last 20 days, all called on the side TRAIN's moves leaned to (the hurdle that every
      model has to clear: when the market rises, "always long" beats a coin without knowing
      anything)
    - **a luck band**: the labels shuffled inside each day, with the models' calls kept; one
      shuffle serves every row, so the best row is held against the best of the shuffled rows
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
    ml, hl = _helpers("movers_lab"), _helpers("hurdle_lab")
    effects = _helpers("stock_effects")  # "Do the stocks differ?"
    ds = lab.ds  # daystart.py, as loaded by tsmixer_lab
    settings = load_settings()
    engine = make_engine(settings)
    return charts, ds, effects, engine, hl, lab, ml, np, pd, settings


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Settings

    The next cell holds every setting. Keep the first three blocks the same as in the Movers
    notebook and both notebooks score the same stock-days.

    1. **The question**: when the prediction is made, how far ahead it looks, what counts as a move
    2. **The book**: cash and the size of a position, which give the number of calls per day
    3. **The stocks and the data**: how many stocks, and three consecutive date ranges
       **TRAIN** → **VALIDATION** → **TEST**
    4. **The inputs**: the morning read before the decision, and the four input steps compared;
       the relabellings behind "Do the stocks differ?"
    """)
    return


@app.cell
def _():
    # ── 1 · The question ──────────────────────────────────────────────
    DECISION = 240  # when the prediction is made: minutes after 09:30 (240 = 13:30)
    HORIZON = 120  # how far ahead, in minutes (120: to 15:30, before the closing chaos)
    THETA = 1.0  # a "move" is at least this many percent, up or down; under it the stock is flat
    # ── 2 · The book ───────────────────────────────────────────────
    CAPITAL = 50_000  # cash to invest
    POSITION = 10_000  # dollars in one position: CAPITAL // POSITION calls a day
    # ── 3 · The stocks and the data (dates: (first day, last day), both included) ─────────
    SYMBOLS = None  # None: the most active stocks of the screen below; or a list, e.g. ["DOCN"]
    UNIVERSE_N = 40  # how many stocks the screen keeps
    UNIVERSE_FROM = "2025-01-01"  # the stocks must have 1-minute bars since this day
    TRAIN = ("2025-01-01", "2026-01-31")  # the trees learn here (its held-out weeks: below)
    VALIDATION = ("2026-02-01", "2026-07-30")  # every input step is scored here
    TEST = ("2026-08-01", "2026-09-30")  # read ONCE, at the very end
    READ_LOCKED = True  # allow the months the darts study series reserves (Feb 2026 on)
    # ── 4 · The inputs ────────────────────────────────────────────────
    HISTORY = 240  # minutes read before the decision (at most DECISION)
    PREV_DAYS = 1  # the sessions behind the "prev" day input
    # Each step adds columns (hurdle_lab.GROUPS): activity = what the last 20 sessions did in this
    # window; day = gap, previous move, same-hour move; morning = returns, volatility, range,
    # volume, flow, market and sector over the last 15-240 minutes; descriptors = the stock's
    # typical move, price and dollar volume (never its name).
    STEPS = {
        "activity only": ("activity",),
        "+ day inputs": ("activity", "day"),
        "+ morning": ("activity", "day", "morning"),
        "+ descriptors": ("activity", "day", "morning", "descriptors"),
    }
    KINDS = ("hurdle", "direct")  # two trees (move, then side) | one three-class tree
    HELD_EVERY = 4  # every 4th week of TRAIN chooses the boosting rounds; then all of it learns
    SHUFFLES = 1000  # label shuffles behind the luck band (a 97.5th percentile needs ~1000)
    RELABELS = 500  # stock relabellings inside each day behind every "Q p" of Table 4b
    return (
        CAPITAL,
        DECISION,
        HELD_EVERY,
        HISTORY,
        HORIZON,
        KINDS,
        POSITION,
        PREV_DAYS,
        READ_LOCKED,
        RELABELS,
        SHUFFLES,
        STEPS,
        SYMBOLS,
        TEST,
        THETA,
        TRAIN,
        UNIVERSE_FROM,
        UNIVERSE_N,
        VALIDATION,
    )


@app.cell(hide_code=True)
def _(
    CAPITAL,
    DECISION,
    HISTORY,
    HORIZON,
    POSITION,
    STEPS,
    TEST,
    TRAIN,
    VALIDATION,
    ds,
    hl,
):
    _days = [*TRAIN, *VALIDATION, *TEST]
    if _days != sorted(_days) or len(set(_days)) != len(_days):
        raise ValueError("TRAIN, VALIDATION and TEST must be (first, last) dates, in that "
                         "order, without overlap")
    if DECISION + HORIZON > ds.SESSION:
        raise ValueError(f"deciding at {ds.clock(DECISION)} + {HORIZON} min ends after the close")
    if not 0 < HISTORY <= DECISION:
        raise ValueError(f"HISTORY {HISTORY}: between 1 and {DECISION} minutes")
    _bad = sorted({g for groups in STEPS.values() for g in groups} - set(hl.GROUPS))
    if _bad:
        raise ValueError(f"STEPS: unknown input groups {_bad}; known: {list(hl.GROUPS)}")
    K = max(1, int(CAPITAL // POSITION))  # calls (positions) a day
    CELL = (1, DECISION - HISTORY, HISTORY, HORIZON)  # the cell the tables below use
    return CELL, K


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The stocks

    The screen looks at every stock with 1-minute bars since `UNIVERSE_FROM`, on the **TRAIN**
    days only: the share of days it moved at least `THETA` between the decision and the horizon's
    end, its typical move, price, and the dollars traded in that window. Stocks with gaps or
    under $1 million traded in the window are dropped; the most active `UNIVERSE_N` stay
    (**Table 1**). The result is saved in `var/movers/` and shared with the Movers notebook.
    """)
    return


@app.cell(hide_code=True)
def _(
    DECISION,
    HORIZON,
    SYMBOLS,
    THETA,
    TRAIN,
    UNIVERSE_FROM,
    UNIVERSE_N,
    engine,
    ml,
    mo,
    table,
):
    if SYMBOLS is None:
        _u = ml.universe(engine, start=UNIVERSE_FROM, train_end=TRAIN[1], decision=DECISION,
                         horizon=HORIZON, theta=THETA, n=UNIVERSE_N)
        STOCKS = list(_u.index)
        _out = mo.vstack([
            mo.md(f"**{len(STOCKS)}** stocks. Share of TRAIN days with a move of at least "
                  f"{THETA:g}%: from **{_u['big'].iloc[-1]:.0%}** to **{_u['big'].iloc[0]:.0%}** "
                  f"(median {_u['big'].median():.0%})."),
            mo.md("### Table 1 · The stocks, most active first"), table(_u.round(3), rows=12)])
    else:
        STOCKS = list(SYMBOLS)
        _out = mo.md(f"`SYMBOLS` is set: **{len(STOCKS)}** stocks, no screen. {STOCKS}")
    mo.vstack([_out])
    return (STOCKS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The data and the split

    The 1-minute bars of every stock from TRAIN's first day to TEST's last, and every input built
    from them (`tsmixer_lab.build`, the same inputs as the TSMixer notebooks). Table 2 shows the
    days:

    - **history only**: the first 10 sessions, read for the typical sizes, never learned
    - **TRAIN**: the trees learn. **Every `HELD_EVERY`-th week is held out** first, to choose the
      number of boosting rounds (the trees' early stopping); then the trees learn from all of
      TRAIN with that number. Spread over the year, the held-out weeks see calm and busy months
      alike (the last fortnight of 2025 was the holidays: 13% of the stock-days moved there,
      against 28% over the year)
    - **VALIDATION**: every input step is scored here; **TEST**: read once, at the very end

    **Table 3** counts the classes on TRAIN and VALIDATION (TEST's stay unread until the end). A
    model that always says "flat" is right on the biggest class, which is why only the calls
    count. It also gives the **prior side**: the side TRAIN's moves leaned to, which is what a
    model that knows nothing about the direction calls.
    """)
    return


@app.cell
def _(
    CELL,
    HELD_EVERY,
    READ_LOCKED,
    STOCKS,
    TEST,
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
    settings,
    table,
):
    _tune_end = ds.rs.WINDOWS["tune"][1]  # later months: reserved by the darts study series
    if TEST[1] > _tune_end and not READ_LOCKED:
        raise ValueError(f"TEST ends {TEST[1]}, after {_tune_end}: set READ_LOCKED = True")
    _panel = ds.rs.load(settings, engine, start=TRAIN[0], end=TEST[1], options=False,
                        symbols=STOCKS, adjust=True)
    _missing = sorted(set(STOCKS) - set(_panel.symbols))
    if _missing:
        raise ValueError(f"no 1-minute data for {_missing}: drop them from SYMBOLS")
    F = lab.build(_panel, engine, basket_from=TRAIN[0], basket_to=TEST[1], options=False)
    ml.with_opens(F, ds.full_clock(_panel).open_)  # the fill: the first trade after 13:30
    _dates = np.array(F["dates"], dtype=object)
    SP = {"sessions": list(_dates)}
    for _k, (_a, _b) in {"train": TRAIN, "validation": VALIDATION, "test": TEST}.items():
        _in = [d for d in _dates if _a <= d <= _b]
        if not _in:
            raise ValueError(f"no stored session in {_k.upper()} {_a} → {_b}")
        SP[_k] = (_in[0], _in[-1])
    _f, _h, val_d = ds.split_days(F["pp"], SP, "validation", TRAIN[0])
    _ft, _ht, test_d = ds.split_days(F["pp"], SP, "test", TRAIN[0])
    _train, _learn_t = np.r_[_f, _h], np.r_[_ft, _ht]  # TRAIN; TRAIN + VALIDATION (for TEST)
    fit_d, held_d = hl.spread_held(F["dates"], _train, HELD_EVERY)
    fit_t, held_t = hl.spread_held(F["dates"], _learn_t, HELD_EVERY)
    PRIOR = ml.prior_side(F, CELL, _train, THETA)  # VALIDATION's calls: TRAIN's lean
    PRIOR_T = ml.prior_side(F, CELL, _learn_t, THETA)  # TEST's: TRAIN + VALIDATION's lean
    _hist = np.arange(_train[0])
    _span = lambda i: (_dates[i[0]], _dates[i[-1]])  # noqa: E731
    _split = pd.DataFrame([
        {"part": p, "from": _span(i)[0], "to": _span(i)[1], "sessions": len(i), "use": u}
        for p, i, u in (("history only", _hist, "typical sizes: read, never learned"),
                        ("TRAIN", _train, "the trees learn"),
                        ("TRAIN, held-out weeks", held_d,
                         f"every {HELD_EVERY}th week: the number of boosting rounds"),
                        ("VALIDATION", val_d, "every input step is scored"),
                        ("TEST", test_d, "read ONCE, at the very end"))])
    _cls = ml.labels(ml.moves(F, CELL), THETA)
    _ok = lab.valid(F, CELL)
    _rows = []
    for _name, _days in (("TRAIN", _train), ("VALIDATION", val_d)):
        _c = _cls[_days][_ok[_days] & (_cls[_days] >= 0)]
        _rows.append({"part": _name, "stock-days": len(_c),
                      **{f"{n}": float(np.mean(_c == i)) for i, n in enumerate(ml.CLASSES)},
                      "up, of the moves": float(np.sum(_c == 2) / np.sum(_c != 1))})
    _lean = _rows[0]["up, of the moves"]
    _side = "long" if PRIOR == 2 else "short"
    mo.vstack([
        mo.md(f"**{len(F['names'])}** stocks, **{len(_dates)}** sessions loaded "
              f"({_dates[0]} → {_dates[-1]})."),
        mo.md("### Table 2 · The data split"), table(_split),
        mo.md(f"### Table 3 · The classes (a move is at least {THETA:g}% up or down)"),
        table(pd.DataFrame(_rows).round(3)),
        mo.md(f"On TRAIN, **{_lean:.0%}** of the moves went up: the prior side is **{_side}**. "
              f"A model that knows nothing about the direction calls every stock {_side}, and is "
              "right on that share of its picks that move.")])
    return PRIOR, PRIOR_T, F, fit_d, fit_t, held_d, held_t, test_d, val_d


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The hurdle: what needs no model

    On each day the rules call `K` stocks. A pick is right when the stock moved at least `THETA`
    the called way. **Table 4** is the bar every model has to clear, on TRAIN and on VALIDATION:

    - **random picks**, random side: any `K` stocks of the day, right on half of those that moved
    - **activity picks**, random side: the `K` stocks with the biggest average move over the last
      20 days. No model, no training
    - **activity picks, prior side**: the same picks, all called on the side TRAIN's moves leaned
      to. **This is the hurdle.** A coin is the wrong bar for the side: when the market rises,
      "always long" is right on more than half of the moves without knowing anything. A tree that
      only learned which stocks are active and which way TRAIN leaned ties it; only knowing
      something about *today* or the *direction* beats it
    """)
    return


@app.cell(hide_code=True)
def _(CELL, F, K, PRIOR, THETA, fit_d, held_d, lab, ml, mo, np, pd, table, val_d):
    _cls = ml.labels(ml.moves(F, CELL), THETA)
    _act = ml.activity(F, CELL, THETA)["mean"]
    _ok = lab.valid(F, CELL) & (_cls >= 0) & np.isfinite(_act)
    _rows = []
    for _name, _days in (("TRAIN", np.sort(np.r_[fit_d, held_d])), ("VALIDATION", val_d)):
        _d, _k = np.nonzero(_ok[_days])
        _d = _days[_d]
        _s = ml.ladder(_cls[_d, _k], _d, _act[_d, _k], K=K, prior=PRIOR)
        _rows.append({
            "part": _name, "days": _s["days"],
            "random picks that moved": _s["move_random"],
            "activity picks that moved": _s["move_naive"],
            "random calls right": _s["call_random"],
            "activity calls right, random side": _s["call_naive"],
            "activity calls right, prior side": _s["call_prior"]})
    mo.vstack([
        mo.md(f"### Table 4 · The hurdle: {K} calls a day, no model"),
        table(pd.DataFrame(_rows).round(3))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Do the stocks differ?

    One model per stock can only beat one model for all stocks if an input acts differently on
    different stocks: a morning surge that carries on in one stock and reverses in another, say.
    Before anyone trains forty models, this asks the table directly, and trains nothing. For every
    input of the steps below (activity, day and morning; the descriptors barely move inside a
    stock) and both outcomes (**moved**: at least `THETA` either way; **up**: went up, among the
    days it moved), each stock gets its own slope: the outcome on the input standardized inside
    the stock, in log-odds per standard deviation, so a stock that moves on 45% of its days and
    one that moves on 15% are on one scale. **Table 4b**, one row per input and outcome:

    - **Q p** (TRAIN): do the stocks' slopes spread more than their noise allows? The share of
      `RELABELS` relabellings (the stocks shuffled inside each day: a day keeps its shared mood,
      a stock its days) whose slopes spread as much. **tau** is the spread of the true slopes
      beyond that luck, **I2** its share of the whole spread
    - **carry r** and **carry p** (VALIDATION, other days): does a stock's TRAIN slope come back?
      The correlation over stocks of every stock's TRAIN slope with its VALIDATION slope, and the
      share of stock shuffles that reach it. Only a difference that comes back can be traded
    - **pooled**: the slope the stocks share; **typical se**: how well one stock's own slope is
      known (the median stock's standard error); **stocks**: those with enough rows for a slope

    A model per stock has something to learn only in a row where **Q p** and **carry p** are both
    small. With this many rows a few small p-values are luck: the line above the table counts
    them against what luck gives. It tests one input at a time, as a straight line: a habit that
    shows only in a combination of inputs would slip through. **Figure 4b** shows the stocks of
    the row whose differences come back best.
    """)
    return


@app.cell(hide_code=True)
def _(
    CELL,
    F,
    PREV_DAYS,
    RELABELS,
    THETA,
    charts,
    effects,
    fit_d,
    held_d,
    hl,
    lab,
    mo,
    np,
    table,
    val_d,
):
    import plotly.graph_objects as pgo

    _kw = dict(prev_days=PREV_DAYS, theta=THETA)
    _groups = ("activity", "day", "morning")
    _Xa, _cols, _ca, _da, _ka, _ = hl.table(lab, F, CELL, np.sort(np.r_[fit_d, held_d]),
                                            _groups, **_kw)
    _Xb, _, _cb, _db, _kb, _ = hl.table(lab, F, CELL, val_d, _groups, **_kw)
    with mo.status.spinner(title="Every stock's slopes, and the relabellings ..."):
        DIFFER, _per = effects.differ((_Xa, _ca, _da, _ka), (_Xb, _cb, _db, _kb), _cols,
                                      len(F["names"]), perms=RELABELS)
    _m = len(DIFFER)
    _q, _c = DIFFER["Q p"] < 0.05, DIFFER["carry p"] < 0.05
    _strict = DIFFER[(DIFFER["Q p"] < 0.05) & (DIFFER["carry p"] < 0.05 / _m)]
    _names = ", ".join(_strict["input"] + " → " + _strict["outcome"])
    _verdict = (
        f"**{len(_strict)} row(s) stand out** ({_names}): the stocks differ there and the "
        "differences come back on other days. A model per stock, or one for all stocks "
        "fine-tuned on each, has something to learn from them."
        if len(_strict) else
        "**No row stands out**: no input acts measurably differently on different stocks in a way "
        "that comes back. A model per stock would fit noise on these inputs; one model for all "
        "stocks loses nothing visible by pooling them.")
    _top = DIFFER.sort_values("carry p").iloc[0]
    _a, _b = _per[(_top["outcome"], _top["input"])]
    _fig = pgo.Figure(pgo.Scatter(x=_a, y=_b, mode="markers+text", text=F["names"],
                                  textposition="top center", marker={"color": charts.SERIES[0]},
                                  showlegend=False))
    _fig.add_hline(y=0, line_color=charts._MUTED)
    _fig.add_vline(x=0, line_color=charts._MUTED)
    _fig.update_xaxes(title="TRAIN slope (log-odds per standard deviation)")
    _fig.update_yaxes(title="VALIDATION slope")
    charts._base(_fig, title=f"Figure 4b · {_top['input']} → {_top['outcome']}: every stock's "
                 f"TRAIN slope against its VALIDATION slope (carry r {_top['carry r']:+.2f}, "
                 f"p {_top['carry p']:.3f})", height=440)
    mo.vstack([
        mo.md(f"**{int(_q.sum())} of {_m}** rows have Q p < 0.05 and **{int(_c.sum())} of {_m}** "
              f"carry p < 0.05; luck alone gives about **{0.05 * _m:.1f}** of each, and "
              f"{0.0025 * _m:.2f} rows with both. A row stands on its own with carry p < "
              f"{0.05 / _m:.4f} (Bonferroni over the {_m} rows). One stock's own slope is known to "
              f"about ±{DIFFER['typical se'].median():.2f} log-odds per standard deviation."),
        mo.md(_verdict),
        mo.md("### Table 4b · Do the stocks differ? (one row per input and outcome; TRAIN for "
              "Q, TRAIN → VALIDATION for carry)"),
        table(DIFFER.sort_values(["carry p", "Q p"]).round(4), rows=12), _fig])
    return (DIFFER,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The input steps

    Each row of **Table 5** is one tree model (`hurdle` or `direct`) fit on the TRAIN days with one
    set of inputs, then asked for its `K` calls on every VALIDATION day. The columns:

    - **excess z**: per day, the share of the model's calls that are right minus that of the
      activity picks on the prior side, averaged over days, divided by its standard error over
      days. It is 0 when the trees know what the activity rule and the prior side know and nothing
      more
    - **luck 97.5%**: the same z with the labels shuffled inside each day (`SHUFFLES` times), the
      97.5th percentile for that row. A row that calls some stocks against the prior side starts
      below 0: with shuffled labels those calls lose to the day's lean. The best row is held
      against the bar below the table: the best of ALL rows on the same shuffles
    - **picking z**: the picks alone, the share of them that moved against the activity picks
    - **side z**: the sides alone, the model's sides against the prior side on its own picks;
      **dir right** and **prior side right**: among its picks that moved, the share its sides got
      right and the share the prior side would have got
    - **AP per day**: the average precision of "will it move" inside each day, averaged over days
      (the model, the activity rule). Pooled over all stock-days it would also credit knowing
      which *days* are busy, which `K` calls a day cannot use
    - **rounds** and **held / prior loss**: per tree (`moved` / `up`, or the one `direct` tree),
      the boosting rounds the held-out weeks chose, and the held-out log loss against always
      predicting TRAIN's class shares. An `up` tree that stops after a few rounds has learned
      nothing about the direction
    """)
    return


@app.cell
def _(mo):
    run_steps = mo.ui.run_button(label="run the input steps")
    mo.hstack([run_steps], justify="start")
    return (run_steps,)


@app.cell(hide_code=True)
def _(
    CELL,
    F,
    K,
    KINDS,
    PREV_DAYS,
    PRIOR,
    SHUFFLES,
    STEPS,
    THETA,
    fit_d,
    held_d,
    hl,
    lab,
    ml,
    mo,
    np,
    pd,
    run_steps,
    table,
    val_d,
):
    mo.stop(not run_steps.value, mo.md("_Press **run the input steps**._"))
    _kw = dict(prev_days=PREV_DAYS, theta=THETA)
    RUNS = {}
    with mo.status.spinner(title="Fitting the trees ..."):
        for _step, _groups in STEPS.items():
            _Xf, _cols, _cf, *_ = hl.table(lab, F, CELL, fit_d, _groups, **_kw)
            _Xh, _, _ch, *_ = hl.table(lab, F, CELL, held_d, _groups, **_kw)
            # every step has the same rows (stock-days); only the columns differ
            _Xv, _, _cv, _dv, _, _av = hl.table(lab, F, CELL, val_d, _groups, **_kw)
            for _kind in KINDS:
                _model = hl.fit(_Xf, _cf, _Xh, _ch, kind=_kind)
                _p = hl.proba(_model, _Xv)
                RUNS[(_step, _kind)] = {
                    "p": _p, "groups": _groups, "columns": _cols, "info": _model["info"],
                    "scores": ml.ladder(_cv, _dv, _av, _p, K=K, prior=PRIOR)}
    with mo.status.spinner(title=f"{SHUFFLES} label shuffles ..."):
        BAND = hl.shuffle_band(_cv, _dv, _av, {k: r["p"] for k, r in RUNS.items()}, K=K,
                               prior=PRIOR, draws=SHUFFLES)
    _bar = float(np.quantile(BAND["best"], 0.975))
    _luck = np.quantile(BAND["z"], 0.975, axis=0)
    _rows = []
    for _j, ((_step, _kind), _r) in enumerate(RUNS.items()):
        _s, _i = _r["scores"], _r["info"]
        _rows.append({
            "inputs": _step, "model": _kind, "columns": len(_r["columns"]),
            "excess z": _s["excess_z_prior"], "luck 97.5%": float(_luck[_j]),
            "picking z": _s["move_z_naive"], "side z": _s["side_z"],
            "calls right": _s["call_prec"], "activity, prior side": _s["call_prior"],
            "picks that moved": _s["move_rate"], "activity picks that moved": _s["move_naive"],
            "dir right": _s["dir_acc"], "prior side right": _s["dir_prior"],
            "AP per day": _s["ap"], "activity AP per day": _s["ap_naive"],
            "rounds": " / ".join(str(v["rounds"]) for v in _i.values()),
            "held / prior loss": "  ".join(f"{v['held_loss']:.3f} / {v['prior_loss']:.3f}"
                                           for v in _i.values())})
    RESULTS = pd.DataFrame(_rows)
    BEST_KEY = max(RUNS, key=lambda k: RUNS[k]["scores"]["excess_z_prior"])
    _best = RUNS[BEST_KEY]["scores"]
    _verdict = ("**above luck**" if _best["excess_z_prior"] >= _bar else
                "**within luck**: what the best of this many tries looks like by chance")
    _side = "long" if PRIOR == 2 else "short"
    mo.vstack([
        mo.md(f"**Best: {BEST_KEY[0]}, {BEST_KEY[1]}.** Excess z "
              f"**{_best['excess_z_prior']:+.2f}** over the activity picks called {_side}; "
              f"{_best['calls']} calls over {_best['days']} days, **{_best['call_prec']:.1%}** "
              f"right against {_best['call_prior']:.1%} for that rule "
              f"({_best['call_naive']:.1%} with a random side). On {SHUFFLES} within-day label "
              f"shuffles, the best of the {len(RUNS)} rows reaches z ≥ **{_bar:.2f}** only 2.5% "
              f"of the time → {_verdict}."),
        mo.md(f"### Table 5 · Every input step and model ({K} calls a day, VALIDATION)"),
        table(RESULTS.round(4), rows=10)])
    return BAND, BEST_KEY, RESULTS, RUNS


@app.cell(hide_code=True)
def _(BAND, BEST_KEY, K, RESULTS, RUNS, charts, mo, np):
    import plotly.graph_objects as go

    _best = RUNS[BEST_KEY]
    _s = _best["scores"]
    _fig = go.Figure()
    for _kind, _col in (("hurdle", charts.SERIES[0]), ("direct", charts.SERIES[1])):
        _part = RESULTS[RESULTS["model"] == _kind]
        _fig.add_trace(go.Bar(x=_part["inputs"], y=_part["calls right"], name=_kind,
                              marker={"color": _col}))
    _fig.add_hline(y=_s["call_prior"], line_dash="dash", line_color=charts.SERIES[3],
                   annotation_text="activity picks, prior side (the hurdle)")
    _fig.add_hline(y=_s["call_naive"], line_dash="dot", line_color=charts.SERIES[3],
                   annotation_text="activity picks, random side")
    _fig.add_hline(y=_s["call_random"], line_dash="dot", line_color=charts._MUTED,
                   annotation_text="random picks")
    _fig.update_layout(barmode="group", yaxis_tickformat=".0%")
    charts._base(_fig, title=f"Figure 1 · Share of the calls right, by input step "
                 f"({K} calls a day, VALIDATION)", height=420)
    _fig2 = go.Figure(go.Histogram(x=BAND["best"], nbinsx=30, marker={"color": charts._MUTED},
                                   name="best row, shuffled labels"))
    _fig2.add_vline(x=_s["excess_z_prior"], line_color=charts.SERIES[0],
                    annotation_text="the best row, real labels")
    _fig2.add_vline(x=float(np.quantile(BAND["best"], 0.975)), line_dash="dash",
                    line_color=charts.SERIES[3], annotation_text="luck bar (97.5%)")
    charts._base(_fig2, title=f"Figure 2 · The best row's excess z against the best of all "
                 f"{len(RUNS)} rows on {len(BAND['best'])} within-day label shuffles",
                 height=360)
    _imp = mo.md(
        "The best row uses **" + str(len(_best["columns"])) + " columns**: "
        + ", ".join(_best["columns"]) + ".")
    mo.vstack([_fig, _fig2, _imp])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## TEST: read once

    The best row (its inputs and its model) is fit again on TRAIN and VALIDATION (every
    `HELD_EVERY`-th week of both chooses the rounds, then all of it learns), with the same fixed
    knobs, and calls the TEST days; the prior side is the one TRAIN and VALIDATION leaned to.
    **Press once**: whatever it says, the inputs must not be tuned on it afterwards. If the best
    row was within luck on VALIDATION, TEST is the only honest look at whether it knows anything.
    The Movers notebook calls the same TEST days: decide before pressing which notebook reads them.
    """)
    return


@app.cell
def _(mo):
    read_test = mo.ui.run_button(label="read TEST (once)")
    mo.hstack([read_test], justify="start")
    return (read_test,)


@app.cell(hide_code=True)
def _(
    BEST_KEY,
    CELL,
    F,
    K,
    PREV_DAYS,
    PRIOR_T,
    RUNS,
    SHUFFLES,
    THETA,
    fit_t,
    held_t,
    hl,
    lab,
    ml,
    mo,
    np,
    pd,
    read_test,
    table,
    test_d,
):
    mo.stop(not read_test.value, mo.md("_Press the button to read TEST (once)._"))
    _groups = RUNS[BEST_KEY]["groups"]
    _kw = dict(prev_days=PREV_DAYS, theta=THETA)
    _Xf, _, _cf, *_ = hl.table(lab, F, CELL, fit_t, _groups, **_kw)
    _Xh, _, _ch, *_ = hl.table(lab, F, CELL, held_t, _groups, **_kw)
    _Xt, _, _ct, _dt, _, _at = hl.table(lab, F, CELL, test_d, _groups, **_kw)
    _model = hl.fit(_Xf, _cf, _Xh, _ch, kind=BEST_KEY[1])
    _pt = hl.proba(_model, _Xt)
    TEST_SCORES = ml.ladder(_ct, _dt, _at, _pt, K=K, prior=PRIOR_T)
    _band = hl.shuffle_band(_ct, _dt, _at, {"best": _pt}, K=K, prior=PRIOR_T, draws=SHUFFLES)
    _t, _v = TEST_SCORES, RUNS[BEST_KEY]["scores"]
    _side = "long" if PRIOR_T == 2 else "short"
    _rows = [
        ("share of the picks that moved", _t["move_random"], _t["move_naive"], _t["move_rate"],
         f"z {_t['move_z_naive']:+.2f} vs the activity picks"),
        ("share of the calls right", _t["call_random"], _t["call_prior"], _t["call_prec"],
         f"z {_t['excess_z_prior']:+.2f} vs the activity picks called {_side}"),
        ("direction right, among the picks that moved", 0.5, _t["dir_prior"], _t["dir_acc"],
         f"side z {_t['side_z']:+.2f} vs calling them all {_side}"),
        ("average precision of 'will it move', per day", _t["ap_base"], _t["ap_naive"],
         _t["ap"], "chance / activity / model")]
    _mix = ", ".join(f"{n} {np.mean(_ct == i):.1%}" for i, n in enumerate(ml.CLASSES))
    mo.vstack([
        mo.md(f"TEST, {_t['calls']} calls over {_t['days']} days: **{_t['call_prec']:.1%}** "
              f"right against {_t['call_prior']:.1%} for the activity picks called {_side} "
              f"(VALIDATION: {_v['call_prec']:.1%} against {_v['call_prior']:.1%}). Excess z "
              f"**{_t['excess_z_prior']:+.2f}**; the 97.5th percentile of {SHUFFLES} within-day "
              f"label shuffles is {np.quantile(_band['z'][:, 0], 0.975):+.2f}. TEST's classes: "
              f"{_mix}."),
        mo.md("### Table 6 · TEST, once"),
        table(pd.DataFrame(_rows, columns=["rung", "random", "the rule", "the model",
                                           "test"]).round(3))])
    return


if __name__ == "__main__":
    app.run()
