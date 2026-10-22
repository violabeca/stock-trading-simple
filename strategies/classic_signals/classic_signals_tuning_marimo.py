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
    # Classic signals · tuning

    This notebook finds settings for the strategy of `classic_signals_marimo.py`: **one setting
    for one stock or for several stocks together** (pick them in section 1), checks whether it
    holds up, and exports it, so that notebook can start from it.

    1. **The data**: pick the stocks to tune; their candles of the training window (Jan 2025 –
       Jun 2026) and of the validation window (Jul 2026 – 16 Oct 2026). Everything from 17 Oct
       2026 on is the final check: never loaded here
    2. **The search space**: what the optimizer may choose. By default it decides everything
       itself (every rule's switch on or off, every number in its range); you only override
       what you want fixed
    3. **The objective**: the daily Sharpe ratio of the stocks' combined profit on the training
       window, the luck line it must clear, and the validation score recorded beside it
    4. **A quick check**: one run with the optimizer's first trial
    5. **The optimization**: a new run, or more trials for an earlier one; watch it live in
       Optuna's dashboard
    6. **Which hyperparameters matter**: each one moved across its range, the rest held: what it
       does to the score, and which ones do nothing at all
    7. **Does the procedure hold up?** The walk forward inside the training window: the
       optimization repeated every few sessions, each block traded with settings chosen on
       earlier sessions only
    8. **The export**: the best settings, one complete set for every stock, as a file the main
       notebook loads under *Starting values*

    The heavy steps (5, 6, 7) only run when you click their button.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def table(df, rows=12):
        """A DataFrame as a marimo table; a named index becomes a column."""
        return mo.ui.table(
            df.reset_index() if df.index.name else df, selection=None, page_size=rows
        )

    return (table,)


@app.cell
def _(mo):
    reload_helpers = mo.ui.run_button(label="reload the helper files (after they changed)")
    mo.hstack([reload_helpers], justify="start")
    return (reload_helpers,)


@app.cell
def _(reload_helpers):
    import importlib.util
    import math
    import time

    import numpy as np
    import pandas as pd

    from trading_simple import strategies
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    _ = reload_helpers.value     # a click reruns this cell: the helper files are read again

    def _helper(name):
        """One of this strategy's helper files, loaded by path."""
        path = strategies.STRATEGIES_DIR / "classic_signals" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"classic_signals_t_{name}", path)
        assert spec is not None and spec.loader is not None, path
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    data, charts, results = _helper("data"), _helper("charts"), _helper("results")
    combine, hyperparams, tuning = _helper("combine"), _helper("hyperparams"), _helper("tuning")
    simulate, pipeline = _helper("simulate"), _helper("pipeline")
    RULES = combine.load_rules()
    settings = load_settings()
    engine = make_engine(settings)
    return (
        RULES,
        charts,
        data,
        engine,
        hyperparams,
        math,
        np,
        pd,
        pipeline,
        results,
        settings,
        simulate,
        time,
        tuning,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · The data

    **Pick the stocks to tune together** below: everything in this notebook (the quick check,
    the optimization, the effects, the walk forward and the export) then looks for **one
    setting for all of them**, judged on their combined profit. One stock works too. Changing
    the pick reloads the candles and starts over. The validation window ends on the last day
    **every** picked stock has 1-minute candles (some stocks' candles stop at the end of
    September 2026), so no stock is scored on fewer validation sessions than the others.

    **Every trade risks the same dollars at its first stop** (`simulate.costs`: $100, at most
    $20,000 worth), so a calm and a jumpy stock, or a calm and a wild quarter, count alike, and
    the stocks' daily profits add up on one scale. Distances are measured in each stock's own
    moves (standard deviations, typical moves, typical daily moves), so one setting means the
    same on every stock.

    Loaded: every stock's one-minute candles (every stored hour: the bands and the moving
    average may read the hours outside the regular session) and the daily candles of the stock,
    its sector fund and the market fund, for the **training window** (the optimizer chooses on
    it) and the **validation window** (every trial is also scored on it, to watch, never to
    choose). The months after the validation window are the final check and are never loaded
    here. Each stock gets a **lab**: the strategy for any settings, its parts (candles, bands,
    trends, levels, ...) built once and kept, so trying many settings mostly reuses them.
    """)
    return


@app.cell
def _(data):
    STOCKS = data.SECTOR_FUND  # every stock that can be picked, and its sector fund
    MARKET = "SPY"  # the S&P 500 fund: the market as a whole
    TRAINING = ("2025-01-01", "2026-06-30")  # the optimizer chooses on these months
    VALIDATION = ("2026-07-01", "2026-10-16")  # scored on every trial, never to choose; it ends
    # earlier when a picked stock's 1-minute candles do (section 1)
    DAILY_FROM = "2024-01-01"  # daily candles for the trends and levels start here
    return DAILY_FROM, MARKET, STOCKS, TRAINING, VALIDATION


@app.cell
def _(STOCKS, mo):
    stocks = mo.ui.multiselect(list(STOCKS), value=["NOW", "GEV"],
                               label="**stocks to tune together**")
    mo.hstack([stocks], justify="start")
    return (stocks,)


@app.cell(hide_code=True)
def _(
    DAILY_FROM,
    MARKET,
    RULES,
    STOCKS,
    TRAINING,
    VALIDATION,
    data,
    engine,
    mo,
    pd,
    pipeline,
    settings,
    simulate,
    stocks,
    table,
    tuning,
):
    mo.stop(not stocks.value, mo.md("_Pick at least one stock._"))
    PICKED = [_s for _s in STOCKS if _s in stocks.value]      # the stocks tuned together
    with mo.status.spinner(title=f"loading {len(PICKED)} stocks"):
        LABS, VALID_END = tuning.labs(engine, PICKED, training=TRAINING, validation=VALIDATION,
                                      sectors=STOCKS, rules=RULES,
                                      costs=simulate.costs(settings), market=MARKET,
                                      daily_from=DAILY_FROM)
    _reports = LABS[0].reports
    GROUP = tuning.group(LABS)                 # "NOW + GEV": the name of the runs and walks
    TRAIN = int((LABS[0].sessions()[1] < pd.Timestamp(VALIDATION[0])).sum())  # training sessions
    _rows = []
    for _lab in LABS:
        _s, _all, _opt = _lab.names["stock"], _lab.sessions()[1], _lab.options.index
        for _w, _part in (("training", _all[_all < pd.Timestamp(VALIDATION[0])]),
                          ("validation", _all[_all >= pd.Timestamp(VALIDATION[0])])):
            _rows.append({"stock": _s, "sector fund": STOCKS[_s], "window": _w,
                          "sessions": len(_part), "first": _part[0].date(),
                          "last": _part[-1].date(),
                          "reports": int(((_reports["symbol"] == _s) & _reports["reaction day"]
                                          .between(_part[0], _part[-1])).sum())
                          if len(_reports) else 0,
                          "days with option data": int(((_opt >= _part[0])
                                                        & (_opt <= _part[-1])).sum())})
    mo.vstack([mo.md(f"### Table 1 · What the labs of {GROUP} trade (validation until "
                     f"{VALID_END}: the last day every picked stock has 1-minute candles)"),
               table(pd.DataFrame(_rows), rows=24)])
    return GROUP, LABS, PICKED, TRAIN, VALID_END


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · The search space: what the optimizer may choose

    **By default the optimizer decides everything itself**: every rule's switch (on or off),
    every number within its range, every choice. You only step in where you want a say:

    - **a switch**: *optimizer decides*, *always on* or *always off*
    - **a number**: the optimizer picks a value *from* … *to*, in steps of *step*. Set *from*
      and *to* to the same value to fix it there
    - **a choice** (candle size, stop, holding limit, profit target, ...): the values the
      optimizer may pick from. Leave just one to fix it

    The grey text says what each one does and what a change rebuilds: a change of the candles
    or of an indicator makes a trial slower than a change of a rule or of the exit. A rule's
    lever only matters while its rule is on, a stop's distance only with its kind of stop, the
    daily-move target only with that target: a trial that leaves one out does not draw it.
    The starting ranges come from `hyperparams.py` (a lever's: from its rule's file in
    `rules/`): change them there to make a range the default for good, or here for this
    session. **Reset** puts every control back. **Table 2** sums up the space as it stands.

    The optimizer's **first trial** uses the values of *first trial*: the built-in ones, or a
    stock's values from a file this notebook exported before.

    One hint from a first short run: free, the optimizer switched the **earnings guard** off for
    two stocks (only a handful of reports in the history, so it saw no reason to keep it),
    although on its own it raised NOW's score. Set it to *always on* if you want it kept.
    """)
    return


@app.cell
def _(hyperparams, mo):
    reset_space = mo.ui.run_button(label="reset: the optimizer decides everything")
    _options = {"built-in values (hyperparams.py)": None}
    for _f in hyperparams.files():
        for _s in hyperparams.load(_f)["stocks"]:
            _options[f"{_f.stem} · tuned for {_s}"] = (str(_f), _s)
    start_from = mo.ui.dropdown(_options, value="built-in values (hyperparams.py)",
                                label="first trial")
    mo.hstack([reset_space, start_from], justify="start", gap=2)
    return reset_space, start_from


@app.cell
def _(RULES, hyperparams, mo, reset_space):
    _ = reset_space.value     # a click rebuilds the controls below: the optimizer free again
    HP = hyperparams.table(RULES)
    _default = hyperparams.default_space(RULES)

    def _control(_n):
        _r = HP[_n]
        if _n.startswith("on: "):
            return mo.ui.dropdown({"optimizer decides": "free", "always on": "on",
                                   "always off": "off"}, value="optimizer decides")
        if "choices" in _r:
            _opts = {str(_l): _v for _v, _l in _r["choices"].items()}
            return mo.ui.multiselect(_opts, value=list(_opts))
        _lo, _hi, _st = _default[_n]
        _kw = {"start": _r["low"], "stop": _r["high"], "step": _r["step"]}
        return mo.ui.array([
            mo.ui.number(value=_lo, label="from", **_kw),
            mo.ui.number(value=_hi, label="to", **_kw),
            mo.ui.number(start=_r["step"], stop=max(_r["high"] - _r["low"], _r["step"]),
                         step=_r["step"], value=_st, label="step")])

    space_ui = mo.ui.dictionary({_n: _control(_n) for _n in HP})

    def _row(_n):
        _r = HP[_n]
        _c = space_ui[_n]
        _parts = [_c[0], _c[1], _c[2]] if "choices" not in _r else [_c]
        _when = (f" · only with {hyperparams.needs_words(_r)}"
                 if isinstance(_r.get("needs"), tuple) else "")
        _info = mo.md(f"<span style='color:#77756f'>{_r['doc']} · a change rebuilds "
                      f"{hyperparams.STAGES[_r['stage']]}{_when}</span>")
        _name = mo.md(f"**{_r['label'] if not _n.startswith('on: ') else _n[4:]}**")
        return mo.hstack([_name, *_parts, _info], justify="start", gap=1, align="center")

    _sections = {"The rules: on or off": [_n for _n in HP if _n.startswith("on: ")],
                 "The rules' levers": [_n for _n, _r in HP.items()
                                       if isinstance(_r.get("needs"), str)],
                 "The exit": [_n for _n, _r in HP.items() if _r["stage"] == "exit"],
                 "Candles and indicators": [_n for _n, _r in HP.items()
                                            if _r["stage"] in ("candles", "indicators")]}
    _blocks = []
    for _title, _names in _sections.items():
        _blocks += [mo.md(f"#### {_title}"), *[_row(_n) for _n in _names]]
    mo.vstack(_blocks, gap=0.3)
    return HP, space_ui


@app.cell(hide_code=True)
def _(HP, RULES, hyperparams, math, mo, space_ui, start_from, table):
    _raw = {}
    for _n, _v in space_ui.value.items():
        if _n.startswith("on: "):
            _raw[_n] = {"free": [True, False], "on": [True], "off": [False]}[_v]
        elif "choices" in HP[_n]:
            _raw[_n] = list(_v)
        else:
            _raw[_n] = tuple(_v)
    SPACE = hyperparams.clean_space(_raw, RULES)          # what the optimizer may choose
    START = hyperparams.starting(RULES, *(start_from.value or ()))   # the first trial
    FREE = [_n for _n in HP if hyperparams.is_free(SPACE[_n])]
    _combos = math.prod(hyperparams.values(SPACE[_n]) for _n in FREE) if FREE else 1
    _levers = [_n for _n in FREE if "needs" in HP[_n]]
    _slow = [_n for _n in FREE if HP[_n]["stage"] in ("candles", "indicators")]
    mo.vstack([
        mo.md(f"""
        **The optimizer chooses {len(FREE)} of the {len(HP)} hyperparameters; {len(HP) -
        len(FREE)} are fixed.** Together they span about **10^{math.log10(_combos):.0f}
        combinations**: far more than any tuning can try, so Optuna samples the promising part.
        {len(_levers)} of them only count while a switch or a choice allows them, and
        {len(_slow)} rebuild an indicator when they change (slower trials).
        """),
        mo.md("### Table 2 · The search space"),
        table(hyperparams.describe_space(SPACE, RULES), rows=40)])
    return FREE, SPACE, START


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The objective

    **The score** of a setting is its **daily Sharpe ratio on the training window**: the profit
    of every session after costs (an open trade valued at the session's last close; with
    several stocks, their profits added up), its average divided by how much it swings from
    session to session, scaled to a year (× √252). 1 means a year's profit about as large as a
    year's swings; 2 is what traders call very good. If any stock trades fewer than the minimum
    trades per session, the setting has no score, so "never trade" cannot win. The profit, the
    trades, the worst drawdown, the worst stock's own score and every stock's own score and
    profit are recorded beside it.

    Why this score: a test on 500 random settings of NOW (2025 against 2026 H1) found no score
    that picks settings that keep working better than another (profit ÷ worst drawdown, total
    profit, profitable months: all about alike). This one has two things the others lack. It
    does not grow with the window's length, so the training and the validation scores compare
    directly. And it has a known scale of luck:

    Several stocks put more evidence behind every trial: on one stock a real edge of the size
    good strategies have (a Sharpe of about 1) is lost in the luck of many trials; summed over
    many stocks it stands out (the stocks move together a little, so a bit less than the count
    suggests).

    **The luck line.** Without any edge, the best of $n$ settings still reaches about
    $\Phi^{-1}(0.5^{1/n}) \times \sqrt{252 / \text{sessions}}$ by chance (Φ⁻¹: the inverse of the
    normal curve). Trying more settings raises the line: about 0.24 for 10 trials on 373
    sessions, 0.42 for 100, 0.61 for 5,000. A best score that does not clearly clear it has
    found nothing. (Settings that resemble each other are not independent, so the true line is
    somewhat lower; on NOW it came out 0.1 lower.)

    **The validation score**: every trial is also scored on the validation window, beside the
    training score, never used to choose. Section 5 draws both: if the training score does not
    predict the validation score across trials, the optimizer is fitting noise.
    """)
    return


@app.cell
def _(mo):
    trials = mo.ui.number(start=1, stop=20000, step=1, value=200, label="trials")
    min_trades = mo.ui.slider(0.0, 3.0, step=0.1, value=0.5, show_value=True,
                              label="minimum trades per session")
    mo.hstack([trials, min_trades], justify="start", gap=2)
    return min_trades, trials


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · A quick check

    One run of the stock with the optimizer's first trial (the *first trial* values, moved into
    the search space: a fixed hyperparameter at its value), on the training window and on the
    validation window, and how long it took (the first run builds the parts, later ones mostly
    reuse them). With the same values, the main notebook shows the same trades: both run
    `pipeline.py`.
    """)
    return


@app.cell(hide_code=True)
def _(
    LABS,
    PICKED,
    RULES,
    SPACE,
    START,
    VALIDATION,
    hyperparams,
    min_trades,
    mo,
    table,
    time,
    tuning,
):
    _t0 = time.time()
    _j = tuning.judge(LABS, hyperparams.project(START, SPACE, RULES), VALIDATION[0],
                      min_per_day=min_trades.value, validate=True)
    mo.vstack([mo.md(f"### Table 3 · One run with the optimizer's first trial, every stock and "
                     f"together (score: none = too few trades; {time.time() - _t0:.1f} s)"),
               table(tuning.by_stock(_j, PICKED).round(2))])
    return


@app.cell(hide_code=True)
def _(mo, tuning):
    mo.md(f"""
    ## 5 · The optimization

    Optuna tries settings from the search space and keeps the one with the best training score.
    Each run first tries a few settings at random (a third of its trials, at most 10) before it
    models which ones score well. **Figure 1** shows every trial: its training score, the best
    so far, the luck line and the validation score of the best so far; and every trial's
    training against its validation score. **Table 4** compares, for every hyperparameter the
    optimizer chooses, the first trial's value with the best; **Table 5** the scores against
    the luck line; **Table 5b** the best trial stock by stock (does it work on most of them,
    or only on one?); **Table 5c** its profit from longs and from shorts, quarter by quarter
    (a setting that only leans with a trend makes its money on one side).

    **Runs are kept and can be continued.** Every run is written, trial by trial, to one journal
    file under its name. To add trials later, pick the run under *run* (it continues with its
    own search space, first trial and minimum trades: sections 2 and 3 do not apply to it) and
    click the button again. A run whose inputs changed since (the data, the windows or the code)
    cannot be continued: start a new one. **Faster, from a terminal, two workers or more at
    once** on one run (each its own seed): `uv run python strategies/classic_signals/tuning.py
    continue "<run name>" <trials> <worker>` (worker 0, 1, ...; about 2 GB each for 11
    stocks).

    **Watch a run live in Optuna's dashboard.** Run this in a terminal and leave it running:

    `uvx optuna-dashboard {tuning.JOURNAL}`

    then open <http://127.0.0.1:8080>: the start page lists every run; click one. *History*
    draws each trial's score and the best so far, *Hyperparameter Importance* ranks the settings
    by how much they move the score, *Slice* plots each setting against the score, and *Trials*
    lists every trial with its settings, its profit and trades and its validation score. A
    trial with too few trades shows as *pruned* (no score). The pages update as trials finish.
    """)
    return


@app.cell
def _(GROUP, mo, time, tuning):
    refresh_runs = mo.ui.run_button(label="refresh the list of runs")
    _runs = tuning.runs(tuning.JOURNAL, GROUP)
    pick_run = mo.ui.dropdown({"a new run (its name below)": "", **{_n: _n for _n in _runs}},
                              value="a new run (its name below)", label="run")
    run_name = mo.ui.text(value=f"{GROUP} {time.strftime('%Y-%m-%d %H:%M')}",
                          label="a new run's name")
    run_now = mo.ui.run_button(label="run the optimization")
    mo.vstack([mo.hstack([pick_run, run_name, refresh_runs], justify="start", gap=2),
               mo.hstack([run_now], justify="start")])
    return pick_run, refresh_runs, run_name, run_now


@app.cell(hide_code=True)
def _(
    GROUP,
    LABS,
    RULES,
    SPACE,
    START,
    VALIDATION,
    min_trades,
    mo,
    pick_run,
    run_name,
    run_now,
    trials,
    tuning,
):
    RUN = {}     # GROUP -> tuning.optimize; empty until the button is clicked
    if run_now.value:
        _name = pick_run.value or run_name.value.strip()
        if pick_run.value:               # continue a run: with its own space and first trial
            _cfg = tuning.run_settings(tuning.JOURNAL, _name, RULES)
            _space, _start, _min = _cfg["space"], _cfg["start"], _cfg["min_per_day"]
        else:
            _space, _start, _min = SPACE, START, min_trades.value
        with mo.status.spinner(title=f"'{_name}': {int(trials.value)} trials on {GROUP}"):
            RUN[GROUP] = tuning.optimize(LABS, VALIDATION[0], space=_space, start=_start,
                                         trials=int(trials.value), min_per_day=_min,
                                         validate=True, storage=tuning.JOURNAL, name=_name)
        RUN[GROUP]["space"] = _space
    return (RUN,)


@app.cell(hide_code=True)
def _(GROUP, LABS, PICKED, RUN, charts, hyperparams, mo, pd, table, tuning):
    mo.stop(not RUN, mo.md("_Pick a run or name a new one, then click **run the optimization** "
                           "(about a second per trial and stock; the first one per indicator "
                           "is slower)._"))
    _r = RUN[GROUP]
    _log = _r["trials"]
    _rows = [{"hyperparameter": _n, "first trial": _r["start"][_n], "best": _r["best"][_n]}
             for _n in _r["space"] if hyperparams.is_free(_r["space"][_n])]
    _best = _log[_log["trial"] == _r["best trial"]].iloc[0] if _r["best trial"] is not None \
        else pd.Series(dtype=object)
    _scores = pd.DataFrame([{
        "trials": len(_log), "too few trades": int(_log["score"].isna().sum()),
        "best training score": _r["score"] if _r["score"] > -1e8 else None,
        "luck line (typical best with no edge)": _r["luck"],
        "luck line, 1 time in 20": tuning.luck(int(_log["score"].notna().sum()),
                                               _r["sessions"], chance=0.95),
        "worst stock": _best.get("worst stock"),
        "training P&L $": _best.get("P&L $"), "training trades": _best.get("trades"),
        "worst drawdown $": _best.get("worst drawdown $"),
        "validation score": _best.get("validation score"),
        "validation P&L $": _best.get("validation P&L $"),
        "stocks profitable on validation": _best.get("stocks profitable on validation")}])
    mo.vstack([
        charts.trials_figure(_r, title="Figure 1 · Every trial of the run: training score, best "
                             "so far, luck line, validation"),
        mo.md("### Table 4 · What the optimizer chose: first trial → best"),
        table(pd.DataFrame(_rows), rows=40),
        mo.md("### Table 5 · The best trial's scores, and the luck line it must clear"),
        table(_scores.round(3)),
        mo.md("### Table 5b · The best trial, stock by stock"),
        table(tuning.by_stock(_best.to_dict(), PICKED).round(2)),
        mo.md("### Table 5c · The best trial's profit from longs and from shorts, by quarter of "
              "the entry (the last two quarters: the validation window)"),
        table(tuning.sides(LABS, _r["best"]).round(0), rows=24)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · Which hyperparameters move the score?

    Around one set of values (the best of the run if one ran, else the optimizer's first
    trial), every hyperparameter is moved across its range on its own, everything else held, and
    the strategy is scored on the training window (every stock, their profits added up).
    **Table 6** ranks them by their **effect**: the highest score minus the lowest, among the
    values that traded enough to be scored;
    *values trading too little* counts the others (a sign the range reaches settings that hardly
    trade). An effect of 0 means the hyperparameter does nothing here, and the last column says
    why (it is not in use: its rule is off, or another stop or target kind is chosen; or no
    trade changed at all). **Figure 2** draws the score against the value for the strongest
    ones (the black dot: the value as set; a red cross: a value that traded too little to be
    scored).

    This is a look around one point: a hyperparameter can matter more (or less) once others
    change. A best setting whose neighbours all score far lower is a lone spike: luck, most
    likely.
    """)
    return


@app.cell
def _(mo):
    effects_points = mo.ui.slider(2, 9, step=1, value=4, show_value=True,
                                  label="values per number")
    run_effects = mo.ui.run_button(label="measure the effects")
    mo.hstack([effects_points, run_effects], justify="start", gap=2)
    return effects_points, run_effects


@app.cell(hide_code=True)
def _(
    GROUP,
    LABS,
    RULES,
    RUN,
    SPACE,
    START,
    VALIDATION,
    effects_points,
    hyperparams,
    min_trades,
    mo,
    run_effects,
    tuning,
):
    mo.stop(not run_effects.value, mo.md("_Click **measure the effects** (one run per value "
                                          "and stock: a minute or two)._"))
    _around = RUN[GROUP]["best"] if RUN else hyperparams.project(START, SPACE, RULES)
    EFFECTS = tuning.effects(LABS, _around, last=VALIDATION[0], min_per_day=min_trades.value,
                             points=effects_points.value)
    EFFECT_SUMMARY = tuning.effect_summary(EFFECTS, _around, RULES)
    return EFFECTS, EFFECT_SUMMARY


@app.cell(hide_code=True)
def _(EFFECTS, EFFECT_SUMMARY, GROUP, charts, mo, table):
    mo.vstack([
        mo.md("### Table 6 · Which hyperparameters move the score (the strongest first)"),
        table(EFFECT_SUMMARY.round(3), rows=40),
        charts.effects_figure(EFFECTS, EFFECT_SUMMARY, title=f"Figure 2 · {GROUP}: the score "
                              "across each hyperparameter's range, the rest held")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7 · Does the procedure hold up? The walk forward

    The optimization, repeated every *refit* sessions from a first session on, inside the
    training window: each block of sessions traded with the setting chosen on the sessions
    before it (one setting for all the stocks), stitched into one record. Against it, **no
    tuning** (the optimizer's first trial throughout): the control the tuned walk must beat, on
    the same sessions. The candle size
    must be fixed in the search space for a walk (a walk trades one size).

    **Figure 3** draws the profit curves, **Table 7** the bottom line, **Figure 4** the settings
    the walk chose over time (values that jump from one tuning to the next mean the tuning
    follows noise). The walks run in separate processes and are kept on disk
    (`var/classic_signals/tuning/`): a repeat with the same inputs is instant.
    """)
    return


@app.cell
def _(TRAIN, mo):
    walk_first = mo.ui.slider(20, TRAIN - 10, step=10, value=120,
                              show_value=True, label="first traded session")
    walk_refit = mo.ui.slider(1, 120, step=1, value=20, show_value=True,
                              label="refit: sessions between two tunings")
    walk_trials = mo.ui.number(start=1, stop=5000, step=1, value=50,
                               label="trials per tuning")
    walk_workers = mo.ui.slider(1, 8, step=1, value=2, show_value=True,
                                label="walks at once (processes)")
    run_walk = mo.ui.run_button(label="run the walk forward")
    mo.vstack([mo.hstack([walk_first, walk_refit, walk_trials], justify="start", gap=2),
               mo.hstack([walk_workers, run_walk], justify="start", gap=2)])
    return run_walk, walk_first, walk_refit, walk_trials, walk_workers


@app.cell(hide_code=True)
def _(
    LABS,
    SPACE,
    START,
    TRAIN,
    data,
    hyperparams,
    min_trades,
    mo,
    run_walk,
    tuning,
    walk_first,
    walk_refit,
    walk_trials,
    walk_workers,
):
    mo.stop(hyperparams.is_free(SPACE["candle_size"]), mo.md(
        "_The walk forward trades one candle size: leave just one **candle size** in the search "
        "space (section 2; one walk per size)._"))
    _base = {"first": walk_first.value, "refit": walk_refit.value, "space": SPACE,
             "start": START, "trials": int(walk_trials.value), "min_per_day": min_trades.value,
             "last": TRAIN}
    _jobs = [(LABS, "no tuning", {**_base, "tune": False}), (LABS, "tuned", _base)]
    _cache = data.VAR / "tuning"
    _missing = sum(not tuning.is_cached(*_j, cache=_cache) for _j in _jobs)
    mo.stop(bool(_missing) and not run_walk.value, mo.md(
        f"_**{_missing} of {len(_jobs)} walks** are not computed yet for these inputs: click "
        "**run the walk forward**._"))
    if _missing:
        with mo.status.progress_bar(total=_missing, title="walks forward") as _bar:
            WALKS = tuning.run_walks(_jobs, cache=_cache, workers=walk_workers.value,
                                     progress=lambda _d, _t: _bar.update())
    else:
        WALKS = tuning.run_walks(_jobs, cache=_cache)
    return (WALKS,)


@app.cell(hide_code=True)
def _(WALKS, charts, mo, results, table):
    mo.vstack([
        charts.walk_curves_figure(WALKS, title="Figure 3 · The walk forward: profit after costs "
                                  "over the traded sessions, tuned against no tuning"),
        mo.md("### Table 7 · The walk forward's bottom line"),
        table(results.walk_table(WALKS).round(2), rows=24)])
    return


@app.cell(hide_code=True)
def _(GROUP, WALKS, charts):
    charts.settings_figure(WALKS, GROUP, title=f"Figure 4 · {GROUP}: the settings the walk "
                           "chose, tuning after tuning")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8 · The export

    Writes the best settings of the run (section 5), one complete set of every hyperparameter
    for every stock of the run, with what made them (the run, the search space, the scores, the
    data), to
    `var/classic_signals/settings/<date>-<name>.json`. In the main notebook, pick the file under
    **Starting values**: every control there then starts at these values, for further
    inspection by hand.
    """)
    return


@app.cell
def _(GROUP, mo):
    export_name = mo.ui.text(value=f"{GROUP.replace(' + ', '-')}-tuned", label="file name")
    export_note = mo.ui.text(value="", label="a note (what you tried)", full_width=True)
    export_now = mo.ui.run_button(label="export the best settings")
    mo.vstack([mo.hstack([export_name, export_now], justify="start", gap=2), export_note])
    return export_name, export_note, export_now


@app.cell(hide_code=True)
def _(
    GROUP,
    PICKED,
    RUN,
    TRAINING,
    VALIDATION,
    VALID_END,
    export_name,
    export_note,
    export_now,
    hyperparams,
    mo,
):
    mo.stop(not (RUN and export_now.value), mo.md(
        "_Run the optimization (section 5), then click **export the best settings**._"))
    _r = RUN[GROUP]
    _best = _r["trials"][_r["trials"]["trial"] == _r["best trial"]]
    _meta = {"note": export_note.value, "training": list(TRAINING),
             "validation": [VALIDATION[0], VALID_END], "run": _r.get("name"),
             "the optimizer chose": [_n for _n, _v in _r["space"].items()
                                     if hyperparams.is_free(_v)],
             "search space": {_k: list(_v) for _k, _v in _r["space"].items()},
             "objective": "daily Sharpe ratio on the training window",
             "scores": {"training score": _r["score"], "luck line": _r["luck"],
                        "trials": len(_r["trials"]),
                        **{_k: _best[_k].iloc[0] for _k in ("validation score",
                                                             "validation P&L $")
                           if _k in _best and len(_best)}}}
    _path = hyperparams.save({_s: _r["best"] for _s in PICKED}, meta=_meta,
                             name=export_name.value.strip() or "tuned")
    mo.vstack([mo.md(f"**Exported:** `{_path}`. In `classic_signals_marimo.py`, pick it under "
                     "*Starting values* (restart its kernel or click *reload the helper "
                     "files* to see a new file)."),
               mo.md(f"```json\n{_path.read_text()[:4000]}\n```")])
    return


if __name__ == "__main__":
    app.run()
