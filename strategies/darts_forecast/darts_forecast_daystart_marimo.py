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
    # Day-start forecast

    Does a Darts model have an **edge over pure luck** at one question: at a fixed time of
    day, will the stock be **up or down at the end of the horizon**, and by roughly how much?

    - one example = one stock-day: the model reads the history before the decision and
      predicts the bars to the end of the horizon; the tuning grades only the move to the end
    - one stock at a time (`SYMBOLS`): now DOCN (DigitalOcean), the most movement among the
      names with full 1-minute data since January 2025; the first runs used COLB
    - the optimizer searches the history length, the inputs and the model (Tables 1 and 7);
      everything else is fixed (Table 2)
    - the luck check: every score is compared with random up/down outcomes of the same sizes
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

    ds, charts = _helpers("daystart"), _helpers("charts")
    ex = ds.ex  # explorer.py, as loaded by daystart
    settings = load_settings()
    engine = make_engine(settings)
    return charts, ds, engine, ex, np, pd, settings, strategies


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Settings

    The next cell holds every setting, in six blocks. Edit a value and everything below re-runs.

    1. **The question** (fixed): when the prediction is made, and how far ahead it looks
    2. **The data**: the stock, `TEST` (read once, at the very end, in both modes) and
       `WALK_FORWARD`, which picks ONE of two ways to learn and score:
       - **2a · single split** (`WALK_FORWARD = False`): one model learns on `TRAIN` and
         scores all of `VALIDATION`
       - **2b · walk-forward** (`WALK_FORWARD = True`): `WF_SCORED` is cut into steps (a month
         or a week, `WF_STEP`); each step is scored by its own model, which learns on the
         `WF_WINDOW_MONTHS` months just before that step (a rolling window, as a model retrained
         live; `None`: every earlier day), then everything moves one step on

       Only the chosen block is used; Table 4 and Figure 0 show its exact days
    3. **What the optimizer searches**: the history length and the inputs (Table 1); the
       model types and their knobs are in Table 7
    4. **Training** (fixed): how many passes, when to stop
    5. **The luck check**: how many random outcomes every score is compared with
    6. **The worked example**: one configuration for "What goes in" and "Train one model"
    """)
    return


@app.cell
def _():
    # ── 1 · The question (fixed) ──────────────────────────────────────────────────────────
    DECISION = 150  # when the prediction is made: minutes after 09:30 (150 = 12:00)
    HORIZONS = [60]  # how far ahead, in minutes; each horizon is its own question: [30, 60]

    # ── 2 · The data (dates: (first day, last day), both included) ────────────────────────
    # One stock: DOCN (DigitalOcean) moves most of the names with full 1-minute data since
    # Jan 2025 (12:00 → 13:00: 79 bp on average, COLB 32 bp). The first runs: ["COLB"].
    SYMBOLS = ["COLB"]  # None: the explorer's 12 picks; TFT/NHiTS/BlockRNN: COLB, then DOCN
    WALK_FORWARD = False  # how models learn and score: False = block 2a, True = block 2b
    TEST = ("2026-07-01", "2026-09-30")  # both modes: read ONCE, at the very end
    READ_LOCKED = True  # allow the months the darts study series reserves (Feb 2026 on)

    # ── 2a · Single split (WALK_FORWARD = False): ONE model scores all of VALIDATION ──────
    TRAIN = ("2025-01-01", "2025-12-31")  # the model learns here; its last 10 days are held out
    VALIDATION = ("2026-01-01", "2026-06-30")  # every trial is scored here

    # ── 2b · Walk-forward (WALK_FORWARD = True): retrained every step, as it would be live ──
    WF_SCORED = ("2025-10-01", "2026-06-30")  # scored step by step, each step by its own model
    WF_STEP = "month"  # how often a new model is trained: "month" or "week" (~4x more fits)
    WF_WINDOW_MONTHS = 8  # each model learns on the months just before its step (rolling);
    #                       None: on every day since WF_LEARN_FROM (expanding)
    WF_LEARN_FROM = "2025-01-01"  # the first day any model may learn on

    # ── 3 · What the optimizer searches (the model types and their knobs: Table 7) ────────
    # name: ([values] | (low, high, step), what it is); a list with one value = fixed
    SPACE = {
        "bar": ([1], "bar size: minutes per bar (fixed to 1 for now)"),
        "input": ((15, 240, 15), "minutes of history read before the decision (at least the "
                  "horizon, at most since 09:30)"),
        "volume": ([True, False], "each bar's volume as an input"),
        "prev": ([True, False], "the previous-day inputs: overnight gap, previous days' move"),
        "prev_days": ([1, 2, 5], "sessions behind the previous days' move (only with prev)"),
    }
    MODEL_TYPES = ["tft", "nhits", "blockrnn"]  # searched, one study each (all: Table 7)
    MAX_TRIALS = 60  # trials per study: one study per model type and horizon

    # ── 4 · Training (fixed) ──────────────────────────────────────────────────────────────
    EPOCHS = 60  # at most this many passes over the TRAIN days
    PATIENCE = 5  # stop after this many passes without a better held-out loss (best kept)

    # ── 5 · The luck check ────────────────────────────────────────────────────────────────
    DRAWS = 200  # random up/down outcomes behind every luck band

    # ── 6 · The worked example ("What goes in" and "Train one model") ─────────────────────
    DEFAULT = (1, DECISION - 120, 120, HORIZONS[0])  # (bar, offset, input, output): 2 h history
    VOLUME, PREV, PREV_DAYS = True, True, 1  # its inputs
    MODEL = "tsmixer"  # its model type: any of Table 7
    return (
        DECISION,
        DEFAULT,
        DRAWS,
        EPOCHS,
        HORIZONS,
        MAX_TRIALS,
        MODEL,
        MODEL_TYPES,
        PATIENCE,
        PREV,
        PREV_DAYS,
        READ_LOCKED,
        SPACE,
        SYMBOLS,
        TEST,
        TRAIN,
        VALIDATION,
        VOLUME,
        WALK_FORWARD,
        WF_LEARN_FROM,
        WF_SCORED,
        WF_STEP,
        WF_WINDOW_MONTHS,
    )


@app.cell(hide_code=True)
def _(
    DECISION,
    DEFAULT,
    HORIZONS,
    PREV,
    PREV_DAYS,
    SPACE,
    TEST,
    TRAIN,
    VALIDATION,
    VOLUME,
    WALK_FORWARD,
    WF_LEARN_FROM,
    WF_SCORED,
    WF_STEP,
    WF_WINDOW_MONTHS,
    ds,
):
    VALUES = {k: v for k, (v, _) in SPACE.items()}
    _lo, _hi, STEP = VALUES["input"]
    TASKS = {}  # horizon -> (shortest, longest) history: at least the horizon, at most since 09:30
    for _h in HORIZONS:
        TASKS[_h] = (max(_lo, _h), min(_hi, DECISION))
        _whole = all(not (DECISION % g or _h % g or STEP % g) for g in VALUES["bar"])
        if DECISION + _h > ds.SESSION or TASKS[_h][0] > TASKS[_h][1] or not _whole:
            raise ValueError(
                f"horizon {_h} min: deciding at {ds.clock(DECISION)} needs whole bars, the end "
                f"before the close and at least {_h} min of history since 09:30"
            )
    if ds.cells(*([v] for v in DEFAULT)) != [DEFAULT]:
        raise ValueError(f"DEFAULT {DEFAULT} needs whole bars inside one session")
    EXAMPLE = {"bar": DEFAULT[0], "input": DEFAULT[2], "volume": VOLUME, "prev": PREV,
               "prev_days": PREV_DAYS}
    for _k, _v in EXAMPLE.items():  # the worked example must be a point of SPACE
        _ok = (TASKS[DEFAULT[3]][0] <= _v <= TASKS[DEFAULT[3]][1] if _k == "input"
               else _v in VALUES[_k])
        if not _ok:
            raise ValueError(f"the worked example's {_k} = {_v} is not a value of SPACE['{_k}']")
    # The chosen mode's days: where learning starts, which days are scored (block 2a or 2b)
    MODE = "walk-forward" if WALK_FORWARD else "single split"
    LEARN_FROM, SCORED = (WF_LEARN_FROM, WF_SCORED) if WALK_FORWARD else (TRAIN[0], VALIDATION)
    _days = [WF_LEARN_FROM, *WF_SCORED, *TEST] if WALK_FORWARD else [*TRAIN, *VALIDATION, *TEST]
    if _days != sorted(_days) or len(set(_days)) != len(_days):
        raise ValueError(f"{MODE}: the dates of block 2{'b' if WALK_FORWARD else 'a'} and TEST "
                         "must be in order, without overlap")
    _window_ok = WF_WINDOW_MONTHS is None or (isinstance(WF_WINDOW_MONTHS, int)
                                              and WF_WINDOW_MONTHS >= 1)
    if WALK_FORWARD and (WF_STEP not in ("month", "week") or not _window_ok):
        raise ValueError('WF_STEP must be "month" or "week"; WF_WINDOW_MONTHS a whole number of '
                         "months (>= 1) or None")
    return EXAMPLE, LEARN_FROM, MODE, SCORED, STEP, TASKS, VALUES


@app.cell(hide_code=True)
def _(
    DECISION,
    DEFAULT,
    DRAWS,
    EPOCHS,
    EXAMPLE,
    HORIZONS,
    MAX_TRIALS,
    MODE,
    MODEL_TYPES,
    PATIENCE,
    READ_LOCKED,
    SPACE,
    STEP,
    SYMBOLS,
    TASKS,
    TEST,
    TRAIN,
    VALIDATION,
    VALUES,
    WALK_FORWARD,
    WF_LEARN_FROM,
    WF_SCORED,
    WF_STEP,
    WF_WINDOW_MONTHS,
    ds,
    ex,
    mo,
    pd,
    table,
):
    def _values(v):
        """A knob's values in words: a list, or a range per horizon."""
        if isinstance(v, list):
            return ", ".join(map(str, v))
        return "; ".join(f"{a}–{b} min, step {STEP} (horizon {h})" for h, (a, b) in TASKS.items())

    _hyper = pd.DataFrame(
        [
            (k, doc, _values(v), str(EXAMPLE[k]),
             "fixed" if isinstance(v, list) and len(v) == 1 else "searched")
            for k, (v, doc) in SPACE.items()
        ],
        columns=["hyperparameter", "what it is", "values", "worked example", "status"],
    )
    _inputs = len(VALUES["volume"]) * sum(len(VALUES["prev_days"]) if p else 1
                                          for p in VALUES["prev"])
    _types = MODEL_TYPES
    _held = ds.HELD_DAYS["validation"]
    _a, _b = ("not used now", "✓") if WALK_FORWARD else ("✓", "not used now")
    _fixed = pd.DataFrame(
        [
            ("1 · question", "decision time", ds.clock(DECISION), "when the prediction is made"),
            ("1 · question", "horizons", ", ".join(f"{h} min" for h in HORIZONS),
             "how far ahead; each horizon is its own question (own studies)"),
            ("2 · data", "stocks", ", ".join(SYMBOLS) if SYMBOLS else "the explorer's 12 picks",
             "one model learns them all"),
            ("2 · data", "WALK_FORWARD", f"{WALK_FORWARD}: {MODE}",
             "False: block 2a (single split) · True: block 2b (walk-forward)"),
            ("2 · data", "TEST", f"{TEST[0]} → {TEST[1]}", "read ONCE, at the very end"),
            ("2 · data", "READ_LOCKED", "read" if READ_LOCKED else "not read",
             "the months the darts study series reserves for itself (Feb 2026 on)"),
            ("2a · single split", "TRAIN", f"{TRAIN[0]} → {TRAIN[1]}" + f" ({_a})",
             f"ONE model learns here; its last {_held} days are held out (early stop, β)"),
            ("2a · single split", "VALIDATION", f"{VALIDATION[0]} → {VALIDATION[1]} ({_a})",
             "that model scores every trial here"),
            ("2b · walk-forward", "WF_SCORED", f"{WF_SCORED[0]} → {WF_SCORED[1]} ({_b})",
             "scored step by step, each step by its own model"),
            ("2b · walk-forward", "WF_STEP", f"{WF_STEP} ({_b})",
             "how often a new model is trained"),
            ("2b · walk-forward", "WF_WINDOW_MONTHS",
             f"{WF_WINDOW_MONTHS if WF_WINDOW_MONTHS else 'None: expanding'} ({_b})",
             f"each model learns on the months just before its step (last {_held} days held "
             "out); None: on every earlier day"),
            ("2b · walk-forward", "WF_LEARN_FROM", f"{WF_LEARN_FROM} ({_b})",
             "the first day any model may learn on"),
            ("4 · training", "epochs", EPOCHS, "at most this many passes over the learning days"),
            ("4 · training", "early stopping", f"{PATIENCE} passes",
             "without a better held-out loss; the best pass is kept"),
            ("5 · luck", "luck draws", DRAWS, "random up/down outcomes behind every luck band"),
            ("in the code", "typical-size window", f"{ds.VOL_DAYS} sessions",
             "every typical size reads earlier sessions only (daystart.py)"),
            ("in the code", "move clip", f"±{ds.CLIP:g} typical moves",
             "a bigger bar move counts as this (daystart.py)"),
            ("in the code", "learning rate", ex.SETTINGS["lr"],
             "the worked example's; searched as `lr` (Table 7; explorer.py)"),
        ],
        columns=["block", "setting", "value", "what it means"],
    ).astype(str)
    mo.vstack(
        [
            mo.md(
                f"**The question:** decide at **{ds.clock(DECISION)}** and predict the move to "
                f"the end of {', '.join(f'**{h} min**' for h in HORIZONS)} ahead · "
                f"**{_inputs}** input sets · budget **MAX_TRIALS = {MAX_TRIALS} per study** "
                f"({len(HORIZONS)} horizon(s) × {len(_types)} model types) · worked example: "
                f"{ds.label(DEFAULT)}"
            ),
            mo.md("### Table 1 · What the optimizer searches: history length and inputs"),
            table(_hyper),
            mo.md("### Table 2 · Fixed settings, by block of the settings cell"),
            table(_fixed, rows=20),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Words**

    - **decision / horizon**: when the prediction is made, and how far ahead it looks (both
      fixed: they define the question)
    - **input**: the minutes of history the model reads before the decision (searched)
    - **bar**: minutes of trading, built from the stored 1-minute bars
    - **typical move**: a bar's move ÷ the usual size of that clock time's move (last 20
      sessions), so calm and wild stocks are comparable
    - **end move**: the stock's total move from the decision to the end of the horizon, in
      typical moves: what the tuning grades
    - **end-move correlation**: how well predicted and actual end moves line up over all
      VALIDATION stock-days (the tuning's score; 0 = no skill)
    - **luck z**: how many luck-widths the end-move correlation sits above random up/down
      outcomes of the same sizes (about 2 = rare by chance for ONE try; the best of many tries
      needs more)
    - **R² vs "no change"**: 0 = as good as saying nothing, above 0 = better (the path grade of
      the worked example)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Stocks and the data split

    The days are loaded and cut as set in block 2. **Table 4** and **Figure 0** show, for the
    chosen mode, one row per model (a single split has one, walk-forward one per scored
    month) and TEST:

    - **history only**: the first 10 sessions; the typical sizes need earlier sessions, so
      these days are read, never learned
    - **learns**: the model's training days; its **last 10 are held out** to stop the
      training and to set the size scale β
    - **scores**: the days its predictions are graded on; all models' scored days together
      give a trial its score
    - **TEST**: read once, at the very end

    With walk-forward, each model learns on the `WF_WINDOW_MONTHS` months just before its step
    and scores that step; then everything moves one step on, as a model retrained live (with
    `None`, each model learns on every earlier day). Half-days (13:00 close) are never used.
    """)
    return


@app.cell
def _(
    LEARN_FROM,
    READ_LOCKED,
    SCORED,
    SYMBOLS,
    TEST,
    TRAIN,
    WALK_FORWARD,
    WF_STEP,
    WF_WINDOW_MONTHS,
    ds,
    engine,
    ex,
    mo,
    np,
    pd,
    settings,
    table,
):
    _cands = ex.candidates(engine, (LEARN_FROM, TEST[1]), pool=SYMBOLS)  # SYMBOLS (None: all)
    picks = ex.pick(_cands)
    NAMES = SYMBOLS or list(picks["symbol"])
    _tune_end = ds.rs.WINDOWS["tune"][1]  # later months: reserved by the darts study series
    if TEST[1] > _tune_end and not READ_LOCKED:
        raise ValueError(f"TEST ends {TEST[1]}, after {_tune_end}: those months are reserved by "
                         "the darts study series. Set READ_LOCKED = True to read them.")
    # the 1-minute bars (dividend-adjusted) from the first learning day to TEST's last
    panel = ds.rs.load(settings, engine, start=LEARN_FROM, end=TEST[1], options=False,
                       symbols=NAMES, adjust=True)
    _missing = sorted(set(NAMES) - set(panel.symbols))
    if _missing:
        raise ValueError(f"no 1-minute data for {_missing}: drop them from SYMBOLS")
    pp = ds.prepared(panel)
    _dates = np.array([str(x) for x in pp["dates"]], dtype=object)  # plain strings

    def _model(learn_from, learn_to, scored):
        """One model's days: it learns from ``learn_from`` to ``learn_to`` (its last 10 days,
        at most a third, held out) and scores the days of ``scored`` (first, last)."""
        fit, held, graded = ds.split_days(
            pp, {"train": (learn_from, learn_to), "validation": scored}, "validation",
            learn_from)
        if len(fit) < 10:
            raise ValueError(f"the model scoring {scored[0]} learns on only {len(fit)} days: "
                             "widen WF_WINDOW_MONTHS or start WF_SCORED later")
        return fit, held, graded

    def _step(d):
        """The step a scored day belongs to: its month, or its ISO week."""
        if WF_STEP == "month":
            return d[:7]
        year, week, _ = pd.Timestamp(d).isocalendar()
        return f"{year}-W{week:02d}"

    _scored = np.flatnonzero(pp["usable"] & (_dates >= SCORED[0]) & (_dates <= SCORED[1]))
    if not len(_scored):
        raise ValueError(f"no stored session in the scored days {SCORED[0]} → {SCORED[1]}")
    if WALK_FORWARD:  # one model per step; it learns on the window just before its step
        FOLDS = []
        _keys = [_step(d) for d in _dates[_scored]]
        for _k in dict.fromkeys(_keys):  # the steps, in time order
            _in = [i for i, k in zip(_scored, _keys, strict=True) if k == _k]
            _start = _dates[_in[0]]
            _from = LEARN_FROM if WF_WINDOW_MONTHS is None else max(LEARN_FROM, str(
                (pd.Timestamp(_start) - pd.DateOffset(months=WF_WINDOW_MONTHS)).date()))
            FOLDS.append(_model(_from, _dates[_in[0] - 1], (_start, _dates[_in[-1]])))
    else:  # one model: it learns on TRAIN and scores all of VALIDATION
        FOLDS = [_model(LEARN_FROM, TRAIN[1], SCORED)]
    fit_d, held_d, _ = FOLDS[0]  # the worked example learns like the first model
    val_d = np.concatenate([v for *_, v in FOLDS])  # every scored day
    _test = [d for d in _dates if TEST[0] <= d <= TEST[1]]
    if not _test:
        raise ValueError(f"no stored session in TEST {TEST[0]} → {TEST[1]}")
    SP = {"sessions": list(_dates), "train": (_dates[fit_d[0]], _dates[held_d[-1]]),
          "validation": (_dates[val_d[0]], _dates[val_d[-1]]), "test": (_test[0], _test[-1])}
    mo.vstack(
        [
            mo.md(f"**{len(panel.symbols)}** stock(s), **{len(_dates)}** sessions loaded "
                  f"({_dates[0]} → {_dates[-1]})."),
            mo.md("### Table 3 · The stocks"),
            table(ex.pick_table(picks[picks["symbol"].isin(NAMES)])),
        ]
    )
    return FOLDS, NAMES, SP, fit_d, held_d, picks, pp


@app.cell(hide_code=True)
def _(FOLDS, MODE, SP, charts, mo, np, pd, pp, table):
    _dates = np.array([str(x) for x in pp["dates"]], dtype=object)  # plain strings
    _span = lambda i: (_dates[i[0]], _dates[i[-1]])  # noqa: E731
    _words = lambda i: f"{_dates[i[0]]} → {_dates[i[-1]]}"  # noqa: E731
    _hist = np.arange(FOLDS[0][0][0])  # the sessions before the first learning day
    _test = np.flatnonzero(pp["usable"] & (_dates >= SP["test"][0]) & (_dates <= SP["test"][1]))
    _rows = [{"model": str(n + 1), "learns": _words(f), "learning days": str(len(f)),
              "held out (early stop, β)": _words(h), "scores": _words(v),
              "scored days": len(v)} for n, (f, h, v) in enumerate(FOLDS)]
    _rows.append({"model": "TEST", "learns": "–", "learning days": "–",
                  "held out (early stop, β)": "–", "scores": f"{_words(_test)} (read once)",
                  "scored days": len(_test)})
    _bars = [(f"model {n + 1}", [("history only", *_span(_hist)), ("learns", *_span(f)),
                                 ("held out (early stop, β)", *_span(h)),
                                 ("scores", *_span(v))])
             for n, (f, h, v) in enumerate(FOLDS)]
    _bars.append(("TEST", [("TEST (read once)", *_span(_test))]))
    mo.vstack(
        [
            mo.md(f"### Table 4 · The data split: **{MODE}**, {len(FOLDS)} model(s)"),
            mo.md(f"History only (read, never learned): {_words(_hist)} ({len(_hist)} "
                  "sessions). Each row is one model: the days it learns on, the days held "
                  "out, the days it scores."),
            table(pd.DataFrame(_rows)),
            charts.split_calendar(_bars, title=f"Figure 0 · How the days are used: {MODE}, "
                                  f"one row per model, and TEST"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## What goes in, what comes out

    One example = one stock-day. Pick a stock and a day. The model's own output appears in
    Figure 1 once it is trained (next section).

    - **Figure 1**: 1a the day's price and the windows · 1b the target bars: input (blue),
      truth (green), model output (red) · 1c the volume input
    - **Figure 2**: the previous-day inputs
    - **Table 5 / 6**: what the model takes in and gives out, and its hyperparameters
    """)
    return


@app.cell
def _(DEFAULT, NAMES, SP, ds, ex, mo, picks, pp):
    _name, _ts = ex.example(picks, NAMES, SP, clock=ds.clock(DEFAULT[1] + DEFAULT[2]))
    _days = [  # TEST days are never offered: TEST is read once
        str(d)
        for d, ok in zip(pp["dates"], pp["usable"], strict=True)
        if ok and SP["train"][0] <= d <= SP["validation"][1]
    ]
    stock_pick = mo.ui.dropdown(NAMES, value=_name, label="stock")
    day_pick = mo.ui.dropdown(
        _days, value=_ts[:10] if _ts[:10] in _days else _days[0], label="day", searchable=True
    )
    mo.hstack([stock_pick, day_pick], justify="start")
    return day_pick, stock_pick


@app.cell
def _(
    DEFAULT,
    PREV,
    PREV_DAYS,
    SP,
    VOLUME,
    charts,
    day_pick,
    ds,
    fit_d,
    held_d,
    mo,
    np,
    pd,
    pp,
    stock_pick,
    trained,
):
    g0, o0, l0, h0 = DEFAULT
    NAME_IO, DAY_IO = stock_pick.value, day_pick.value
    _d, _k = list(pp["dates"]).index(DAY_IO), pp["names"].index(NAME_IO)
    _b = ds.bars(pp, g0)
    _kw = {"volume": VOLUME, "prev": PREV, "prev_days": PREV_DAYS}
    # exactly what Darts is given for this stock-day: the input bars, then the actual output bars
    _ser, _vol, _, _ = ds.examples(_b, pp, DEFAULT, [_d], full=True, **_kw)
    _y = _ser[_k].values()[:, 0]
    static = _ser[_k].static_covariates.iloc[0]
    IO = {
        "name": NAME_IO, "day": DAY_IO, "cell": DEFAULT,
        "times": pd.date_range(f"{DAY_IO} 09:31", periods=ds.SESSION, freq="min"),
        "price": pp["cf"][_d, :, _k], "y_in": _y[: l0 // g0], "y_out": _y[l0 // g0 :],
        "volume": _vol[_k].values()[: l0 // g0, 0] if VOLUME else None,
        "pred": None, "pred_price": None,
    }
    if trained is not None:  # the network's own output for this day, from its input bars only
        _raw, _, _j = ds.predict(trained, _b, pp, DEFAULT, [_d], **_kw)
        IO["pred"] = _raw[_j == _k][0]
        _p0 = pp["cf"][_d, o0 + l0 - 1, _k]  # the decision's last traded close
        if not np.isfinite(_p0):  # no trade yet today: the previous session's close
            _p0 = pp["close"][_d - 1, _k]
        _u = np.nan_to_num(_b["u"][_d, (o0 + l0) // g0 : (o0 + l0 + h0) // g0, _k])
        IO["pred_price"] = np.r_[_p0, _p0 * np.exp(np.cumsum(IO["pred"] * _u))]
    _where = (
        "a learning day of model 1 (Table 4): it learned from this day's bars, so its "
        "output is in-sample"
        if _d in fit_d
        else "held out by model 1: it shaped when the training stopped"
        if _d in held_d
        else "a scored day: model 1 has not seen it"
        if SP["validation"][0] <= DAY_IO <= SP["validation"][1]
        else "neither learned nor scored by model 1"
    )
    mo.vstack(
        [
            mo.md(
                f"**{NAME_IO}** on **{DAY_IO}** · {ds.label(DEFAULT)} · this day is {_where}"
            ),
            charts.model_io(IO),
        ]
    )
    return IO, NAME_IO, g0, h0, l0, static


@app.cell(hide_code=True)
def _(IO, PREV, PREV_DAYS, charts, ds, mo, static):
    mo.vstack(
        [
            mo.md(
                "**Figure 2** shows the two previous-day numbers the model gets once for the "
                "whole day: the overnight **gap** (today's first trade against yesterday's "
                "last) and the **previous move** (the last sessions' close-to-close move). "
                "The gap is 0 until the stock has traded."
            ),
            charts.static_inputs(
                {
                    "name": IO["name"], "day": IO["day"], "gap": static["gap"],
                    "prev_move": static["prev_move"], "prev_days": PREV_DAYS,
                    "limit": ds.CLIP_IN["prev"],
                }
            )
            if PREV
            else mo.md("Previous-day inputs are off (`PREV = False`)."),
        ]
    )
    return


@app.cell
def _(
    IO,
    MODEL,
    NAMES,
    NAME_IO,
    PREV,
    VOLUME,
    fit_d,
    g0,
    h0,
    l0,
    make_model,
    mo,
    pd,
    static,
    table,
):
    # the model as ds.fit builds it; this one is not trained
    est = make_model(MODEL, l0 // g0, h0 // g0, len(fit_d) * len(NAMES))
    if (est.input_chunk_length, est.output_chunk_length) != (len(IO["y_in"]), len(IO["y_out"])):
        raise ValueError("the model's input / output length differs from the example's")
    _static = f"the stock (one-hot: is_{NAME_IO} = 1)" + (
        f", the gap ({static['gap']:+.2f} typical nights) and the previous days' move "
        f"({static['prev_move']:+.2f} typical moves)" if PREV else ""
    )
    _interface = pd.DataFrame(
        [
            ("target (past)", "the input bars' moves, in typical moves",
             f"{est.input_chunk_length} bars × 1"),
            ("volume (past covariate)", "the input bars' volume, log against the usual bar",
             f"{len(IO['volume'])} bars × 1" if VOLUME else "off"),
            ("static", _static, f"{len(static)} numbers"),
            ("output", "the next bar moves, all at once", f"{est.output_chunk_length} bars × 1"),
            ("truth", "what the stock actually did: the grade compares it with the output",
             f"{len(IO['y_out'])} bars × 1"),
        ],
        columns=["role", "what", "size"],
    )
    _arch = (
        "hidden_size", "num_encoder_layers", "num_decoder_layers", "decoder_output_dim",
        "temporal_decoder_hidden", "ff_size", "num_blocks", "kernel_size", "lstm_layers",
        "num_attention_heads", "hidden_continuous_size", "num_stacks", "num_layers",
        "layer_widths", "model", "hidden_dim", "n_rnn_layers", "dropout", "batch_size",
        "n_epochs",
    )  # whichever of these the chosen model has
    _params = (
        pd.Series({k: est.model_params[k] for k in _arch if k in est.model_params})
        .rename_axis("parameter")
        .to_frame("value")
    )
    mo.vstack(
        [
            mo.md(f"### Table 5 · The model `{type(est).__name__}`: what goes in, what comes out"),
            table(_interface),
            mo.md("### Table 6 · The model's hyperparameters (its defaults: `make_model`)"),
            table(_params),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train one model

    Switch **train the model** on to fit `MODEL` on `DEFAULT` (Table 5) on the learning days
    of model 1 (Table 4; with a single split, TRAIN):
    at most `EPOCHS` passes, stopped when the held-out loss stops improving, best pass kept.

    - **Figure 3**, redrawn after every pass: 3a the training and held-out loss against
      "no change" (dashed: the loss to beat), 3b the held-out R² against "no change"
    - **Figure 1** (above) then shows the model's output for the selected stock-day
    """)
    return


@app.cell
def _(mo):
    train_switch = mo.ui.switch(label="train the model")
    mo.hstack([train_switch], justify="start")
    return (train_switch,)


@app.cell(hide_code=True)
def _(np):
    import pytorch_lightning as pl
    import torch

    class TrainingCurve(pl.Callback):
        """Records every pass: mean training loss, held-out loss, and held-out R² against "no
        change" (1 - loss / baseline). Calls ``on_pass(self)`` after each pass; a True return
        stops the training and sets ``stopped``."""

        def __init__(self, baseline, on_pass=None):
            self.baseline, self.on_pass = baseline, on_pass
            self.rows, self._losses, self.stopped = [], [], False

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self._losses.append(float(outputs["loss"] if isinstance(outputs, dict) else outputs))

        def on_validation_epoch_end(self, trainer, pl_module):
            val = trainer.callback_metrics.get("val_loss")
            if trainer.sanity_checking or val is None:
                return
            self.rows.append({"pass": len(self.rows) + 1, "train": float(np.mean(self._losses)),
                              "validation": float(val), "r2": 1 - float(val) / self.baseline})
            self._losses = []
            if self.on_pass is not None and self.on_pass(self):
                trainer.should_stop = self.stopped = True

    class EndpointLoss(torch.nn.Module):
        """The end-move loss: the squared error of the SUM of the predicted bar moves over the
        horizon, so only the move to the end counts, not the path's shape. Darts calls it with
        (batch, bars, 1) predictions and targets."""

        def __init__(self):
            super().__init__()
            self.reduction = "mean"  # Darts expects the attribute

        def forward(self, pred, target):
            err = (pred - target).sum(dim=1).pow(2)
            return err.mean() if self.reduction == "mean" else err

    return EndpointLoss, TrainingCurve, torch


@app.cell(hide_code=True)
def _(
    EPOCHS,
    EndpointLoss,
    PREV,
    PREV_DAYS,
    TrainingCurve,
    VOLUME,
    ds,
    ex,
    fit_d,
    held_d,
    np,
    pp,
    torch,
):
    from darts.models import BlockRNNModel, NHiTSModel, TFTModel

    #: Model types built here (``explorer.make_path_model`` knows tide, tsmixer, dlinear only).
    MORE_TYPES = {"tft": TFTModel, "nhits": NHiTSModel, "blockrnn": BlockRNNModel}

    def make_model(kind, L, H, n_samples, seed=0, hp=None):
        """An untrained Darts model of ``kind`` reading ``L`` bars and predicting the next ``H``
        at once, with ``explorer.make_path_model``'s training setup (early stopping on the
        held-out days, the best pass kept, mean-squared-error point forecasts, a batch size by
        sample count) and the knobs ``hp`` on top of its defaults. TFT, NHiTS and BlockRNN take
        that setup from a TiDE build; NHiTS reads no static inputs (the stock, the previous
        days); TFT gets relative positions in place of calendar inputs."""
        base = ex.make_path_model("tide" if kind in MORE_TYPES else kind, L, H, epochs=EPOCHS,
                                  n_samples=n_samples, seed=seed)
        if kind not in MORE_TYPES:
            return type(base)(**{**base.model_params, **(hp or {})})
        shared = ("input_chunk_length", "output_chunk_length", "n_epochs", "batch_size",
                  "random_state", "pl_trainer_kwargs", "optimizer_kwargs")
        params = {k: base.model_params[k] for k in shared}
        if kind != "nhits":
            params["use_static_covariates"] = True
        if kind == "tft":  # point forecasts (its default is a quantile likelihood)
            params |= {"likelihood": None, "loss_fn": torch.nn.MSELoss(),
                       "add_relative_index": True}
        return MORE_TYPES[kind](**{**params, **(hp or {})})

    def train_model(cell, kind, hp=None, on_pass=None, inputs=None, seed=0, days=None):
        """Fit one Darts torch model on its learning days as ``ds.fit`` does: one model for all
        stocks, early stopping on the held-out days, the best pass's weights kept.

        Args:
            cell: (bar, offset, input, output) minutes.
            kind: tsmixer | tide | dlinear | tft | nhits | blockrnn (:func:`make_model`).
            hp: Knobs that replace its defaults: Darts arguments, plus "lr" and
                "weight_decay" (AdamW), "patience" (the early stopping) and "loss" ("bar":
                every bar's move, Darts' default; "endpoint": only the move to the end).
            on_pass: Called with the TrainingCurve after every pass; True stops training.
            inputs: {"volume", "prev", "prev_days"} (default: the worked example's).
            seed: The model's random seed.
            days: (learning days, held-out days) as session indices (default: model 1's,
                Table 4; every walk-forward model passes its own).

        Returns:
            (model, curve): the fitted model and its TrainingCurve.
        """
        g, _, L, H = cell
        inputs = inputs or {"volume": VOLUME, "prev": PREV, "prev_days": PREV_DAYS}
        fit, held = days if days is not None else (fit_d, held_d)
        b = ds.bars(pp, g)
        ser, vol, _, _ = ds.examples(b, pp, cell, fit, full=True, **inputs)
        vser, vvol, _, _ = ds.examples(b, pp, cell, held, full=True, **inputs)
        hp = dict(hp or {})
        optim = {k: hp.pop(k) for k in ("lr", "weight_decay") if k in hp}
        if optim:
            hp["optimizer_kwargs"] = {"lr": ex.SETTINGS["lr"], **optim}
        if "weight_decay" in optim:  # decoupled weight decay (AdamW), not Adam's L2
            hp["optimizer_cls"] = torch.optim.AdamW
        endpoint = hp.pop("loss", "bar") == "endpoint"
        if endpoint:
            hp["loss_fn"] = EndpointLoss()
        patience = hp.pop("patience", None)
        model = make_model(kind, L // g, H // g, len(ser), seed, hp)
        for c in model.trainer_params["callbacks"]:  # the early stopping's patience
            if patience is not None and hasattr(c, "patience"):
                c.patience = patience
        y_out = [s.values()[L // g :, 0] for s in vser]  # the held-out days' actual bars
        no_change = np.mean([y.sum() ** 2 if endpoint else np.mean(y ** 2) for y in y_out])
        curve = TrainingCurve(float(no_change), on_pass)  # the held-out loss to beat
        model.trainer_params["callbacks"].append(curve)
        cov = {"past_covariates": vol, "val_past_covariates": vvol} if inputs["volume"] else {}
        model.fit(ser, val_series=vser, verbose=False, **cov)
        ex.SM["_quiet"]()
        keep = next(c for c in model.trainer_params["callbacks"] if hasattr(c, "state"))
        if keep.state is not None:  # back to the weights of the best pass
            model.model.load_state_dict(keep.state)
        return model, curve

    return make_model, train_model


@app.cell(hide_code=True)
def _(
    DEFAULT,
    EPOCHS,
    MODEL,
    PATIENCE,
    charts,
    fit_d,
    held_d,
    mo,
    pp,
    train_model,
    train_switch,
):
    import time

    trained = fit_info = None
    if train_switch.value:
        _t0 = time.time()
        trained, _curve = train_model(
            DEFAULT, MODEL, hp={"patience": PATIENCE}, on_pass=lambda c: mo.output.replace(
                charts.training_curve(c.rows, c.baseline, epochs=EPOCHS)))
        fit_info = {"curve": _curve.rows, "baseline": _curve.baseline,
                    "seconds": time.time() - _t0}
    if fit_info is None:
        _out = [mo.md("_Not trained yet: switch **train the model** on to fit it._")]
    else:
        _best = max(fit_info["curve"], key=lambda r: r["r2"])
        _d = pp["dates"]
        _out = [
            mo.md(
                f"**Trained** in {fit_info['seconds']:.1f} s on {len(fit_d)} learning days "
                f"({_d[fit_d[0]]} → {_d[fit_d[-1]]}; held out {_d[held_d[0]]} → "
                f"{_d[held_d[-1]]}): {len(fit_info['curve'])} passes, best pass "
                f"**{_best['pass']}**, held-out loss {_best['validation']:.3f} against "
                f"{fit_info['baseline']:.3f} for \"no change\" (R² {_best['r2']:+.3f}). "
                "**Figure 1** above now shows the model's output."
            ),
            charts.training_curve(fit_info["curve"], fit_info["baseline"], epochs=EPOCHS),
        ]
    mo.vstack(_out)
    return (trained,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Tune: Optuna searches, W&B records

    **run the studies** runs one study per model type of `MODEL_TYPES` (Table 7) and horizon
    (Table 2), up to
    `MAX_TRIALS` trials each. Every trial answers the same question: decide at the fixed time,
    predict the stock's move to the end of the horizon. A trial draws the history length and
    the inputs (Table 1) and the model type's knobs (Table 7), trains, and is graded on
    VALIDATION.

    - **score** (maximized): the **end-move correlation**: how well predicted and actual moves
      to the end of the horizon line up over all VALIDATION stock-days. It is scale-free: a
      confident model is not punished for its size, a quiet line earns nothing
    - **luck check**: every trial's correlation against `DRAWS` make-believe outcomes with the
      same sizes and random up/down (`luck_z`, `luck_p`; `hit_p` for the hit rate). The best
      of many trials needs a higher z: the line above Table 8 gives the bar
    - **also per trial**: hit rate (right up/down), the size error after a scale β learned on
      the held-out days (`mae_skill`: 1 − error ÷ the "no change" error), the end R², and
      every stock's own correlation (the worst stock, the share of stocks above 0)
    - **loss**: a searched knob: `bar` (every bar's move, Darts' default) or `endpoint` (only
      the move to the end, what the score cares about)
    - **walk-forward** (`WALK_FORWARD = True`, block 2b): each step of `WF_SCORED` is scored
      by its own model, which learns on the window just before that step (Table 4, Figure 0):
      one fit per step per trial, its own studies ("wf" in the name). The section **Compare**
      sets both modes side by side, on the days both score
    - **no pruning**: every trial trains until its own early stopping (`PATIENCE`); each
      pass's held-out R² is still reported (Figure 5, Optuna Dashboard)
    - **studies**: journals in `studies/`, one per model type and horizon, named by a
      fingerprint of every setting; re-running tops each up to `MAX_TRIALS`. Each trial's
      predictions are saved in `var/daystart_trials/` (re-scoring needs no retraining)
    - **W&B**: one run per trial through OptunaHub's W&B callback, grouped by study: the
      knobs, the scores, every stock's correlation and the per-pass curve. Offline in
      `var/wandb/` (`WANDB_MODE=offline` in `.env`)
    - **compare**: Table 8 (the best of each model type and horizon), Table 9 (all trials),
      Table 10 and Figure 7 (the best trial, per stock and per stock-day), Figures 4–6 (the
      study of the best trial)
    - **caveat**: VALIDATION is one month; the best of many trials is partly luck. Read the
      winner on TEST once, at the end
    """)
    return


@app.cell
def _(MODEL_TYPES, mo, pd, table):
    # The model types (one study each) and their knobs: [choices] | (low, high) integers |
    # (low, high) floats | (low, high, "log") on a log scale. "all": every model type: the
    # learning rate, weight decay (AdamW) and the loss ("bar" | "endpoint"); the rest are the
    # Darts model's own arguments.
    SEARCH = {
        "all": {"lr": (1e-4, 1e-2, "log"), "weight_decay": (1e-7, 1e-2, "log"),
                "loss": ["bar", "endpoint"]},
        "tsmixer": {"hidden_size": (16, 256, "log"), "ff_size": (16, 256, "log"),
                    "num_blocks": (1, 4), "activation": ["ReLU", "GELU", "SELU"],
                    "norm_type": ["LayerNorm", "LayerNormNoBias", "TimeBatchNorm2d"],
                    "normalize_before": [False, True], "dropout": (0.0, 0.3)},
        "tide": {"hidden_size": (32, 256, "log"), "num_encoder_layers": (1, 3),
                 "num_decoder_layers": (1, 3), "decoder_output_dim": (4, 32, "log"),
                 "temporal_decoder_hidden": (8, 64, "log"), "temporal_width_past": (2, 8),
                 "use_layer_norm": [False, True], "dropout": (0.0, 0.3)},
        "dlinear": {"kernel_size": [5, 13, 25], "const_init": [True, False]},
        "tft": {"hidden_size": [16, 32, 64], "lstm_layers": (1, 2),
                "num_attention_heads": [1, 2, 4], "full_attention": [False, True],
                "hidden_continuous_size": [4, 8, 16, 32], "dropout": (0.0, 0.3)},
        "nhits": {"num_stacks": (1, 3), "num_blocks": (1, 2), "num_layers": (1, 3),
                  "layer_widths": [64, 128, 256, 512], "MaxPool1d": [True, False],
                  "dropout": (0.0, 0.3)},
        "blockrnn": {"model": ["RNN", "LSTM", "GRU"], "hidden_dim": (8, 128, "log"),
                     "n_rnn_layers": (1, 3), "dropout": (0.0, 0.3)},
    }
    run_study = mo.ui.run_button(label="run the studies")
    _unknown = sorted(set(MODEL_TYPES) - set(SEARCH) - {"all"})
    if _unknown or not MODEL_TYPES:
        raise ValueError(f"MODEL_TYPES must name model types of SEARCH; unknown: {_unknown}")
    _rows = [("all model types" if m == "all" else m, k, str(v),
              "✓" if m == "all" or m in MODEL_TYPES else "not in MODEL_TYPES")
             for m, knobs in SEARCH.items() for k, v in knobs.items()]
    mo.vstack(
        [
            mo.md("### Table 7 · The model types (one study each) and their searched knobs"),
            table(pd.DataFrame(_rows, columns=["model type", "knob", "range", "searched now"]),
                  rows=30),
            run_study,
        ]
    )
    return SEARCH, run_study


@app.cell(hide_code=True)
def _(
    DECISION,
    DRAWS,
    FOLDS,
    PATIENCE,
    SEARCH,
    STEP,
    TASKS,
    VALUES,
    ds,
    ex,
    np,
    pd,
    pp,
    strategies,
    train_model,
):
    import os

    import optuna
    import optunahub
    import wandb
    from dotenv import dotenv_values
    from scipy.stats import binomtest

    _root = strategies.STRATEGIES_DIR.parent
    for _k, _v in dotenv_values(_root / ".env").items():  # the W&B key; real env vars win
        if _k.startswith("WANDB_") and _v:
            os.environ.setdefault(_k, _v)
    WANDB_MODE = os.environ.setdefault(  # online with a key, offline without (or if .env says so)
        "WANDB_MODE", "online" if os.environ.get("WANDB_API_KEY") else "offline")
    os.environ.setdefault("WANDB_SILENT", "true")
    WANDB_DIR = _root / "var" / "wandb"  # git-ignored: offline runs, local copies of online ones
    TRIALS_DIR = _root / "var" / "daystart_trials"  # every trial's predictions (git-ignored)
    # Optuna's W&B callback from the OptunaHub registry, pinned to one commit (cached locally)
    wandb_hub = optunahub.load_module("callbacks/wandb",
                                      ref="618066c693941399e0890b03fd3a516aec141bfe")

    def suggest(trial, name, spec):
        """One knob, drawn as its SEARCH declaration says (a range is drawn as a number)."""
        if isinstance(spec, list):
            return trial.suggest_categorical(name, spec)
        lo, hi, *log = spec
        draw = trial.suggest_int if isinstance(lo, int) else trial.suggest_float
        return draw(name, lo, hi, log=bool(log))

    def draw(trial, horizon, kind):
        """Every knob of one trial for one question (DECISION, horizon) and one model type: the
        history length (a number, so its order counts), the inputs, the knobs every model type
        shares and the type's own. Every draw can exist: no rejections."""
        lo, hi = TASKS[horizon]
        g = trial.suggest_categorical("bar", VALUES["bar"])
        length = trial.suggest_int("input", lo, hi, step=STEP)
        prev = (trial.suggest_categorical("prev", VALUES["prev"]) if kind != "nhits"
                else False)  # NHiTS reads no static inputs, so no previous-day inputs
        inputs = {"volume": trial.suggest_categorical("volume", VALUES["volume"]), "prev": prev,
                  "prev_days": 1}
        if inputs["prev"]:
            inputs["prev_days"] = trial.suggest_categorical("prev_days", VALUES["prev_days"])
        hp = {n: suggest(trial, n, s) for n, s in {**SEARCH["all"], **SEARCH[kind]}.items()}
        return (g, DECISION - length, length, horizon), inputs, {**hp, "patience": PATIENCE}

    def end_moves(model, cell, inputs, days):
        """The model's paths on ``days``, and the predicted and actual move to the end of the
        horizon (typical moves) of every stock-day."""
        g, _, _, H = cell
        b = ds.bars(pp, g)
        raw, s, j = ds.predict(model, b, pp, cell, days, **inputs)
        act = ex.actual_paths(b["pin"], s, j, H // g)
        return raw, s, j, raw.sum(axis=1), act.sum(axis=1)

    def correlation(p, a):
        """Pearson correlation; 0 when either side does not move (no skill to measure)."""
        return float(np.corrcoef(p, a)[0, 1]) if np.std(p) > 0 and np.std(a) > 0 else 0.0

    def end_grade(pe, ae, day, j, beta):
        """The end-move grade of predicted ``pe`` vs actual ``ae`` end moves: correlation (the
        score), hit rate, size error after the scale ``beta`` (one per stock-day, or one for
        all), end R², loudness, per stock, per month, and the luck check: the same predictions
        against ``DRAWS`` make-believe outcomes that keep every move's size but flip its sign at
        random (one coin per day for all stocks). ``luck_z`` = how many luck-SDs the
        correlation sits above luck, ``luck_p`` = the share of coins that do at least as well,
        ``hit_p`` = the chance of the hit rate by coin-tossing.

        Returns:
            (scores, per-stock correlation, per-month correlation)
        """
        beta = np.broadcast_to(np.asarray(beta, float), pe.shape)
        hits = np.sign(pe) == np.sign(ae)
        _, by_day = np.unique(day, return_inverse=True)
        coins = np.random.default_rng(0).choice([-1.0, 1.0], size=(DRAWS, by_day.max() + 1))
        luck = np.array([correlation(pe, c[by_day] * ae) for c in coins])
        out = {"end_corr": correlation(pe, ae),
               "hit_rate": float(hits.mean()),
               "mae_skill": float(1 - np.mean(np.abs(beta * pe - ae)) / np.mean(np.abs(ae))),
               "end_r2": float(1 - np.sum((beta * pe - ae) ** 2) / np.sum(ae ** 2)),
               "end_r2_raw": float(1 - np.sum((pe - ae) ** 2) / np.sum(ae ** 2)),
               "loudness": float(np.sum(pe ** 2) / np.sum(ae ** 2)),
               "beta": float(beta.mean())}
        per = {pp["names"][k]: correlation(pe[j == k], ae[j == k]) for k in np.unique(j)}
        out |= {"worst_stock_corr": min(per.values()),
                "stocks_positive": float(np.mean([v > 0 for v in per.values()])),
                "luck_z": float((out["end_corr"] - luck.mean()) / luck.std()) if luck.std() > 0
                else 0.0,
                "luck_p": float((1 + np.sum(luck >= out["end_corr"])) / (1 + DRAWS)),
                "hit_p": float(binomtest(int(hits.sum()), len(hits), 0.5,
                                         alternative="greater").pvalue)}
        month = np.array([str(pp["dates"][d])[:7] for d in day])
        by_month = {m: correlation(pe[month == m], ae[month == m]) for m in np.unique(month)}
        out["months_positive"] = float(np.mean([v > 0 for v in by_month.values()]))
        return out, per, by_month

    def fit_and_score(trial, cell, inputs, kind, hp):
        """Inside the trial's W&B run, for every fold of FOLDS (one with a single split, one
        per VALIDATION month with walk-forward): train on the fold's days until its own early
        stopping, take the size scale β from its held-out days (never from VALIDATION), and
        predict the fold's VALIDATION days. All folds' stock-days are then graded together,
        on the same days in both modes. Optuna's per-pass curve is the last fold's (the most
        training days). The curves and scores are logged once at the trial's step; the
        callback then adds the knobs at that step."""

        def report(curve):  # each pass's held-out R²: the trial's learning curve in Optuna
            r = curve.rows[-1]
            trial.report(r["r2"], r["pass"])

        parts, curves = [], []
        for f, (fit, held, graded) in enumerate(FOLDS):
            model, curve = train_model(cell, kind, hp, inputs=inputs, days=(fit, held),
                                       on_pass=report if f == len(FOLDS) - 1 else None)
            *_, ph, ah = end_moves(model, cell, inputs, held)
            raw, s, j, pe, ae = end_moves(model, cell, inputs, graded)
            beta = ex.calibration(ph[:, None], ah[:, None])
            parts.append((raw, s, j, pe, ae, np.full(len(pe), beta)))
            curves += [{"fold": f + 1, **r} for r in curve.rows]
        raw, s, j, pe, ae, beta = (np.concatenate(x) for x in zip(*parts, strict=True))
        result, per, by_month = end_grade(pe, ae, s // (ds.SESSION // cell[0]), j, beta)
        folder = TRIALS_DIR / trial.study.study_name
        folder.mkdir(parents=True, exist_ok=True)
        date = np.array([str(pp["dates"][x]) for x in s // (ds.SESSION // cell[0])])
        np.savez_compressed(folder / f"{trial.number}.npz", raw=raw, s=s, j=j, beta=beta,
                            date=date, act_end=ae)
        wandb.run.log({"curve": wandb.Table(dataframe=pd.DataFrame(curves)), **result,
                       **{f"stock_corr/{k}": v for k, v in per.items()},
                       **{f"month_corr/{k}": v for k, v in by_month.items()}},
                      step=trial.number)
        for k, v in {**result, "passes": len(curves), "stock_corr": per,
                     "month_corr": by_month}.items():
            if k != "end_corr":  # the trial's value
                trial.set_user_attr(k, v)
        return result["end_corr"]

    def make_objective(wandbc, horizon, kind):
        """The trial function of one study (one horizon, one model type): draw the knobs, then
        train and grade inside the trial's own W&B run (``wandbc.track_in_wandb``)."""

        def objective(trial):
            cell, inputs, hp = draw(trial, horizon, kind)
            return wandbc.track_in_wandb()(
                lambda t: fit_and_score(t, cell, inputs, kind, hp))(trial)

        return objective

    return TRIALS_DIR, WANDB_DIR, WANDB_MODE, make_objective, optuna, wandb_hub


@app.cell
def _(
    DECISION,
    EPOCHS,
    HORIZONS,
    MAX_TRIALS,
    MODEL_TYPES,
    PATIENCE,
    SEARCH,
    SYMBOLS,
    TEST,
    TRAIN,
    VALIDATION,
    VALUES,
    WALK_FORWARD,
    WANDB_DIR,
    WANDB_MODE,
    WF_LEARN_FROM,
    WF_SCORED,
    WF_STEP,
    WF_WINDOW_MONTHS,
    ds,
    make_objective,
    mo,
    optuna,
    pd,
    run_study,
    strategies,
    table,
    wandb_hub,
):
    import hashlib
    import json

    from trading_simple import optimize as opt

    _strat = strategies.load("darts_forecast")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def study_name(kind, horizon, walk_forward):
        """A study's name: model type, horizon, "wf" for walk-forward, and a fingerprint of
        every setting that makes it one experiment (a changed setting starts a new study)."""
        days = ([("walk-forward", WF_SCORED, WF_STEP, WF_WINDOW_MONTHS, WF_LEARN_FROM)]
                if walk_forward else [TRAIN, VALIDATION])  # block 2b or 2a
        setup = [VALUES, SEARCH["all"], SEARCH[kind], kind, DECISION, horizon, *days, TEST,
                 SYMBOLS, EPOCHS, PATIENCE, "end-move correlation, no pruning",
                 ds.DAYSTART_VERSION]
        tag = "wf-" if walk_forward else ""
        return (f"daystart-{kind}-h{horizon}-{tag}"
                + hashlib.sha1(json.dumps(setup, default=str).encode()).hexdigest()[:8])

    def open_study(name, create=False):
        """The study ``name`` from its journal; None if it has none yet (unless ``create``)."""
        if not create and not opt.study_path(_strat, name).exists():
            return None
        return optuna.create_study(
            study_name=name, storage=opt.storage_for(_strat, name), load_if_exists=True,
            direction="maximize",  # TPE, joint within each group of knobs drawn together
            sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True),
            pruner=optuna.pruners.NopPruner(),  # every trial trains to its own early stop
        )

    studies = {  # (horizon, model type) -> its study, in the mode WALK_FORWARD sets
        (_h, _kind): open_study(study_name(_kind, _h, WALK_FORWARD), create=True)
        for _h in HORIZONS for _kind in MODEL_TYPES
    }
    if run_study.value:
        for (_h, _kind), _study in studies.items():
            _todo = MAX_TRIALS - len(_study.trials)  # top up to MAX_TRIALS, never past it
            if _todo <= 0:
                continue
            _wandbc = wandb_hub.WeightsAndBiasesCallback(  # one W&B run per trial
                metric_name="end_corr", as_multirun=True,
                wandb_kwargs={"project": "darts-daystart", "group": _study.study_name,
                              "dir": WANDB_DIR},
            )
            _study.optimize(
                make_objective(_wandbc, _h, _kind), n_trials=_todo,
                catch=(RuntimeError, ValueError),  # a knob combination that breaks: FAIL, go on
                callbacks=[_wandbc, lambda s, t, k=_kind, h=_h: mo.output.replace(mo.md(
                    f"{k}, {h} min: trial {t.number + 1} of {MAX_TRIALS}: "
                    f"{t.state.name.lower()}"))],
            )
    _rows = []
    for (_h, _kind), _study in studies.items():
        _done = [t.value for t in _study.trials if t.state.name == "COMPLETE"]
        _rows.append({"model type": _kind, "horizon (min)": _h, "complete": len(_done),
                      "failed": sum(t.state.name == "FAIL" for t in _study.trials),
                      "best end corr": round(max(_done), 4) if _done else None,
                      "study": _study.study_name})
    _wb = ("live at wandb.ai, project `darts-daystart`" if WANDB_MODE == "online"
           else "offline in `var/wandb/` (`wandb sync` uploads them)")
    mo.vstack(
        [
            mo.md(f"**Studies**, VALIDATION scored by "
                  f"**{'walk-forward' if WALK_FORWARD else 'single split'}** "
                  f"(`MAX_TRIALS = {MAX_TRIALS}` each). Journals: "
                  "`strategies/darts_forecast/studies/`; predictions: `var/daystart_trials/`; "
                  f"W&B runs: {_wb}."),
            table(pd.DataFrame(_rows)),
        ]
    )
    return open_study, studies, study_name


@app.cell(hide_code=True)
def _(
    DECISION,
    TRIALS_DIR,
    charts,
    ds,
    ex,
    mo,
    np,
    optuna,
    pd,
    pp,
    studies,
    table,
):
    import contextlib

    _frames = [s.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state",
                                         "duration")).assign(type=k, horizon=h,
                                                             study=s.study_name)
               for (h, k), s in studies.items() if s.trials]
    _all = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    if _all.empty or not (_all["state"] == "COMPLETE").any():
        _out = [mo.md("_No completed trial yet: press **run the studies**._")]
    else:
        _done = (_all[_all["state"] == "COMPLETE"]
                 .assign(duration=lambda x: x["duration"].dt.total_seconds())
                 .rename(columns=lambda c: c.removeprefix("params_").removeprefix("user_attrs_"))
                 .rename(columns={"value": "end_corr"})
                 .sort_values("end_corr", ascending=False).reset_index(drop=True))
        _show = [c for c in ["type", "horizon", "number", "end_corr", "luck_z", "luck_p",
                             "hit_rate", "hit_p", "mae_skill",
                             "months_positive", "beta", "end_r2", "loudness",
                             "worst_stock_corr", "stocks_positive",
                             "input", "volume", "prev", "prev_days", "loss", "lr",
                             "weight_decay", "passes", "duration"] if c in _done]
        _best = _done.iloc[0]
        _study = studies[(_best["horizon"], _best["type"])]
        # the best trial's predictions, per stock-day (saved by the trial)
        _npz = np.load(TRIALS_DIR / _best["study"] / f"{int(_best['number'])}.npz")
        _g, _H = int(_best["bar"]), int(_best["horizon"])
        _cell = (_g, DECISION - int(_best["input"]), int(_best["input"]), _H)
        _act = ex.actual_paths(ds.bars(pp, _g)["pin"], _npz["s"], _npz["j"], _H // _g)
        _beta = _npz["beta"] if "beta" in _npz else _best["beta"]  # walk-forward: per fold
        _pe, _ae = _npz["raw"].sum(axis=1) * _beta, _act.sum(axis=1)
        _names = np.array(pp["names"])[_npz["j"]]
        _stock = pd.DataFrame(
            [{"stock": n, "stock-days": int((_names == n).sum()),
              "end corr": _best["stock_corr"][n],
              "hit rate": float(np.mean(np.sign(_pe[_names == n]) == np.sign(_ae[_names == n]))),
              "mean |actual end move|": float(np.mean(np.abs(_ae[_names == n])))}
             for n in sorted(set(_names))]).sort_values("end corr").round(3)
        _vis = optuna.visualization
        _figs = [("Figure 4 · End-move correlation per trial (the best trial's study)",
                  _vis.plot_optimization_history(_study)),
                 ("Figure 5 · Held-out R² per pass, every trial (its learning curve)",
                  _vis.plot_intermediate_values(_study))]
        with contextlib.suppress(RuntimeError, ValueError, ZeroDivisionError):  # needs a few
            _figs.append(("Figure 6 · Which knobs move the score (importance)",
                          _vis.plot_param_importances(_study)))
        _bar = ex.best_of_bar(len(_done))  # the z the best of this many tries reaches by luck
        _verdict = ("**above luck**" if _best["luck_z"] >= _bar else
                    "**within luck**: what the best of this many tries looks like by chance")
        _out = [
            mo.md(f"**Luck check:** the best trial's end correlation {_best['end_corr']:+.3f} "
                  f"sits **z = {_best['luck_z']:+.2f}** above random up/down outcomes (hit rate "
                  f"{_best['hit_rate']:.0%}). The best of {len(_done)} trials reaches z ≥ "
                  f"{_bar:.2f} by luck alone only 2.5% of the time → {_verdict}. TEST (read "
                  "once) decides."),
            mo.md("### Table 8 · The best trial of each model type and horizon"),
            table(_done.groupby(["type", "horizon"]).head(1)[_show].round(4)),
            mo.md("### Table 9 · All completed trials, best first (end corr: 0 = no skill)"),
            table(_done[_show].round(4), rows=15),
            mo.md(f"### Table 10 · The best trial ({_best['type']}, {_H} min, trial "
                  f"{int(_best['number'])}: {ds.label(_cell)}), per stock"),
            table(_stock),
            charts.endpoint_scatter(
                _pe, _ae, _names,
                title=f"Figure 7 · The best trial: predicted (× β {_best['beta']:.2f}) vs actual "
                      f"end move, one dot per VALIDATION stock-day (corr "
                      f"{_best['end_corr']:+.3f}, hit rate {_best['hit_rate']:.0%})"),
        ]
        _out += [charts._base(f, title=t, height=420) for t, f in _figs]
    mo.vstack(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare: single split vs walk-forward

    Table 11 puts each model type's study in both modes side by side (a mode that has not run
    yet is missing). The two modes may score different days (`VALIDATION`, `WF_SCORED`), so
    every trial is also re-scored, from its saved predictions, on the days **both** modes
    score: those columns and Figure 8 compare like for like.

    - **the best trial**: its end correlation and luck z, against the bar the best of that many
      trials reaches by luck alone; above the bar = more than luck
    - **all trials**: the median end correlation and the share of trials above luck (z ≥ 2)
      say more than one lucky best
    - **months positive**: in how many VALIDATION months the best trial's correlation is
      above 0; an edge that lives in one month is fragile (trials scored before this column
      existed show "–")
    """)
    return


@app.cell(hide_code=True)
def _(
    HORIZONS,
    LEARN_FROM,
    MODEL_TYPES,
    TRAIN,
    TRIALS_DIR,
    VALIDATION,
    WF_LEARN_FROM,
    WF_SCORED,
    charts,
    ds,
    ex,
    mo,
    np,
    open_study,
    pd,
    pp,
    study_name,
    table,
):
    import plotly.graph_objects as go

    _lo, _hi = max(VALIDATION[0], WF_SCORED[0]), min(VALIDATION[1], WF_SCORED[1])  # both score
    _dates = np.array([str(x) for x in pp["dates"]], dtype=object)  # plain strings

    def _common(study, trial, horizon, start):
        """The trial's end correlation on the days both modes score, from its saved predictions
        (a file without dates only if its data started on this panel's first day)."""
        f = TRIALS_DIR / study.study_name / f"{trial.number}.npz"
        if _lo > _hi or not f.exists():
            return np.nan
        z = np.load(f)
        if "date" in z:
            day, act = z["date"], z["act_end"]
        elif start == LEARN_FROM:  # the same first day: its indices are this panel's
            g = int(trial.params["bar"])
            day = _dates[z["s"] // (ds.SESSION // g)]
            act = ex.actual_paths(ds.bars(pp, g)["pin"], z["s"], z["j"], horizon // g).sum(1)
        else:
            return np.nan
        m = (day >= _lo) & (day <= _hi)
        p, a = z["raw"].sum(axis=1)[m], act[m]
        ok = m.sum() > 2 and p.std() > 0 and a.std() > 0
        return float(np.corrcoef(p, a)[0, 1]) if ok else np.nan

    _rows, _fig = [], go.Figure()
    for _h in HORIZONS:
        for _kind in MODEL_TYPES:
            for _wf in (False, True):
                _st = open_study(study_name(_kind, _h, _wf))
                _done = [t for t in _st.trials if t.state.name == "COMPLETE"] if _st else []
                if not _done:
                    continue
                _mode = "walk-forward" if _wf else "single split"
                _z = np.array([t.user_attrs.get("luck_z", np.nan) for t in _done])
                _best = max(_done, key=lambda t: t.value)
                _bar = ex.best_of_bar(len(_done))  # the z the best of this many reaches by luck
                _months = _best.user_attrs.get("month_corr")
                _start = WF_LEARN_FROM if _wf else TRAIN[0]
                _both = np.array([_common(_st, t, _h, _start) for t in _done])
                _rows.append({
                    "model type": _kind, "horizon": _h, "mode": _mode,
                    "scored days": " → ".join(WF_SCORED if _wf else VALIDATION),
                    "trials": len(_done), "best end corr": round(_best.value, 3),
                    "best luck z": round(_best.user_attrs.get("luck_z", np.nan), 2),
                    "bar for the best": round(_bar, 2),
                    "best above luck": bool(_best.user_attrs.get("luck_z", -np.inf) >= _bar),
                    "best hit rate": f"{_best.user_attrs.get('hit_rate', np.nan):.0%}",
                    "best months positive": (f"{sum(v > 0 for v in _months.values())} of "
                                             f"{len(_months)}" if _months else "–"),
                    "trials z ≥ 2": f"{np.mean(_z >= 2):.0%}",
                    "common days: best trial": round(_both[_done.index(_best)], 3),
                    "common days: median": round(float(np.nanmedian(_both)), 3)
                    if np.isfinite(_both).any() else np.nan,
                    "study": _st.study_name})
                _fig.add_trace(go.Box(
                    y=_both, name=f"{_kind} · {_mode}", boxpoints="all", jitter=0.4, pointpos=0,
                    marker={"size": 4, "color": charts.SERIES[1 if _wf else 0]},
                    line={"color": charts.SERIES[1 if _wf else 0]}, showlegend=False))
    if not _rows:
        _out = [mo.md("_No study with completed trials yet._")]
    else:
        _fig.add_hline(y=0, line={"color": charts._MUTED, "width": 1})  # after the traces
        charts._base(_fig, title=f"Figure 8 · Every trial's end correlation on the days both "
                     f"modes score ({_lo} → {_hi}), per model type and mode", height=460)
        _fig.update_yaxes(title_text="end-move correlation (common days)")
        _out = [mo.md("### Table 11 · Single split vs walk-forward, per model type"),
                mo.md(f"Each study's own score uses its own scored days; the **common days** "
                      f"columns and Figure 8 re-score every trial on the days both modes score, "
                      f"**{_lo} → {_hi}**, so the modes compare like for like."
                      if _lo <= _hi else "**The two modes score no common day.**"),
                table(pd.DataFrame(_rows)), _fig]
    mo.vstack(_out)
    return


if __name__ == "__main__":
    app.run()
