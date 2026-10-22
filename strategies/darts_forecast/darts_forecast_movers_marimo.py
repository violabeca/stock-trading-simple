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
    # Movers: who moves this afternoon, and which way?

    At **13:30** every day, for each stock: will it move by the end of the horizon (**15:30**),
    up or down, by at least `THETA` percent? One TSMixer learns **many stocks** at once, reads the
    morning, and answers with three probabilities: **down / flat / up**.

    This notebook replaces the end-correlation score of the other TSMixer notebooks:

    - **every stock-day counts once**: the outcome is a class, so a 5% move weighs exactly as
      much as a 1% move, and "flat, and it was flat" is a right answer
    - **the outcome is in percent**, not in the stock's own typical moves, so a move is a move
      you could trade: from the fill (the first trade after 13:30) to the last trade before 15:30
    - **what is scored is the CALL**: with your cash and the size of a position the book holds
      `K` positions, so each day the model calls its `K` likeliest movers, each long or short.
      Every call is one vote
    - **the rules need no model**: random picks, and the `K` stocks with the biggest average move
      over the last 20 days, called on the side TRAIN's moves leaned to. That is the hurdle:
      knowing which stocks are active is cheap and already triples the hit rate, and in a rising
      market "always long" beats a coin. The model has to know something on top of both
    - **the stocks are an active set**: those with 1-minute bars since 2025-01-01 that move most
      in this window on the TRAIN days (VALIDATION and TEST are never used to pick them)
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

    lab, charts, ml = _helpers("tsmixer_lab"), _helpers("charts"), _helpers("movers_lab")
    hl = _helpers("hurdle_lab")  # the held-out weeks and the luck band, shared with Hurdle
    ds, ex = lab.ds, lab.ex  # daystart.py and explorer.py, as loaded by tsmixer_lab
    settings = load_settings()
    engine = make_engine(settings)
    return charts, ds, engine, ex, hl, lab, ml, np, pd, settings, strategies


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Settings

    The next cell holds every setting, in six blocks. Edit a value and everything below re-runs.

    1. **The question**: when the prediction is made, how far ahead it looks, what counts as a move
    2. **The book**: cash and the size of a position, which give the number of calls per day
    3. **The stocks and the data**: how many stocks, and three consecutive date ranges
       **TRAIN** → **VALIDATION** → **TEST** (Table 2 shows the exact days)
    4. **The inputs**: each one `True` (read), `False` (not read) or `"search"` (the optimizer
       decides, trial by trial); the stock's own path is always read
    5. **What the optimizer searches**: the history length and TSMixer's knobs (Table 5)
    6. **Training and the checks**: how many passes, when to stop, which weeks are held out; how
       many trials; the best trial's luck band and its re-runs with other seeds
    """)
    return


@app.cell
def _():
    # ── 1 · The question ──────────────────────────────────────────────────────────────────
    DECISION = 240  # when the prediction is made: minutes after 09:30 (240 = 13:30)
    HORIZON = 120  # how far ahead, in minutes (120: to 15:30, before the closing chaos)
    THETA = 1.0  # a "move" is at least this many percent, up or down; under it the stock is flat

    # ── 2 · The book ──────────────────────────────────────────────────────────────────────
    CAPITAL = 50_000  # cash to invest
    POSITION = 10_000  # dollars in one position: CAPITAL // POSITION calls a day

    # ── 3 · The stocks and the data (dates: (first day, last day), both included) ─────────
    SYMBOLS = None  # None: the most active stocks of the screen below; or a list, e.g. ["DOCN"]
    UNIVERSE_N = 40  # how many stocks the screen keeps
    UNIVERSE_FROM = "2025-01-01"  # the stocks must have 1-minute bars since this day
    TRAIN = ("2025-01-01", "2025-12-31")  # the model learns here; some weeks are held out
    VALIDATION = ("2026-01-01", "2026-06-30")  # every trial is scored here
    TEST = ("2026-07-01", "2026-09-30")  # read ONCE, at the very end
    READ_LOCKED = True  # allow the months the darts study series reserves (Feb 2026 on)

    # ── 4 · The inputs: True = read, False = not read, "search" = optimizer ───────────────
    INPUTS = {
        "volume": True, "range": True, "flow": True, "market": True, "sector": True,
        "gap": True, "prev": True, "same_hour": True,
    }
    PREV_DAYS = [1, 2, 5]  # the sessions behind "prev" (searched when it is read)
    STOCK_ID = True  # give the model the stock's one-hot; False: only its activity statistics

    # ── 5 · What the optimizer searches ───────────────────────────────────────────────────
    HISTORY = (60, 240, 30)  # minutes read before the decision: (fewest, most, step)

    # ── 6 · Training and the study ────────────────────────────────────────────────────────
    EPOCHS = 60  # at most this many passes over the TRAIN days
    PATIENCE = 5  # stop after this many passes without a better held-out loss (best kept)
    HELD_EVERY = 4  # every 4th calendar week of the days learned on is held out to stop on
    MAX_TRIALS = 40  # trials in the study
    SHUFFLES = 1000  # within-day label shuffles behind the best trial's luck band (Figure 3)
    SEEDS = (1, 2, 3)  # the best trial trained again with these seeds (Table 7b)
    return (
        CAPITAL,
        DECISION,
        EPOCHS,
        HELD_EVERY,
        HISTORY,
        HORIZON,
        INPUTS,
        MAX_TRIALS,
        PATIENCE,
        POSITION,
        PREV_DAYS,
        READ_LOCKED,
        SEEDS,
        SHUFFLES,
        STOCK_ID,
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
    INPUTS,
    POSITION,
    TEST,
    TRAIN,
    VALIDATION,
    ds,
    lab,
):
    _days = [*TRAIN, *VALIDATION, *TEST]
    if _days != sorted(_days) or len(set(_days)) != len(_days):
        raise ValueError("TRAIN, VALIDATION and TEST must be (first, last) dates, in that "
                         "order, without overlap")
    if DECISION + HORIZON > ds.SESSION:
        raise ValueError(f"deciding at {ds.clock(DECISION)} + {HORIZON} min ends after the close")
    _bad = sorted(k for k, v in INPUTS.items() if v not in (True, False, "search"))
    _unknown = sorted(set(INPUTS) - (set(lab.OPTIONAL) - {"options"}))
    if _unknown or _bad:
        raise ValueError(f"INPUTS: unknown {_unknown}; not True/False/'search': {_bad}")
    _lo, _hi, STEP = HISTORY
    LENGTHS = (_lo, min(_hi, DECISION))
    if LENGTHS[0] > LENGTHS[1]:
        raise ValueError(f"HISTORY {HISTORY}: no length up to {DECISION} minutes")
    K = max(1, int(CAPITAL // POSITION))  # calls (positions) a day
    DEFAULT = (1, DECISION - LENGTHS[1], LENGTHS[1], HORIZON)  # the cell the tables below use
    return DEFAULT, K, LENGTHS, STEP


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The stocks

    The screen looks at every stock with 1-minute bars since `UNIVERSE_FROM`, on the **TRAIN**
    days only: the share of days it moved at least `THETA` between the decision and the horizon's
    end, its typical move, price, and the dollars traded in that window. Stocks with gaps or
    under $1 million traded in the window are dropped; the most active `UNIVERSE_N` stay
    (**Table 1**). The first run takes about a minute; the result is saved in `var/movers/`.

    The stocks with 1-minute bars that far back are the part of the backfill done first, and the
    backfill went **alphabetically**: since 2025-01-01 that is only tickers B to G (244 names; by
    June 2026 there are 1,757, A to Z). So the screen picks the most active stocks of that slice,
    not of the market, and the "market" and "sector" inputs are baskets of the same slice. A later
    `UNIVERSE_FROM` opens the alphabet and shortens TRAIN.

    Keep `UNIVERSE_N` well above the `K` calls a day: with as many stocks as calls, every stock is
    called every day, and only the sides are tested.
    """)
    return


@app.cell(hide_code=True)
def _(
    DECISION,
    HORIZON,
    K,
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
            mo.md(f"**{len(STOCKS)}** stocks, tickers **{min(STOCKS)[0]}** to "
                  f"**{max(STOCKS)[0]}**. Share of TRAIN days with a move of at least "
                  f"{THETA:g}%: from **{_u['big'].iloc[-1]:.0%}** to **{_u['big'].iloc[0]:.0%}** "
                  f"(median {_u['big'].median():.0%})."),
            mo.md("### Table 1 · The stocks, most active first"), table(_u.round(3), rows=12)])
    else:
        STOCKS = list(SYMBOLS)
        _out = mo.md(f"`SYMBOLS` is set: **{len(STOCKS)}** stocks, no screen. {STOCKS}")
    _few = mo.callout(mo.md(
        f"**{len(STOCKS)} stocks for {K} calls a day: every stock is called every day.** The "
        "model picks nothing, so the share of its picks that moved equals the rules' (picking z "
        "0.00) and only its sides are tested. Raise `UNIVERSE_N` (or add `SYMBOLS`) to test the "
        "picking too."), kind="warn") if len(STOCKS) <= K else None
    mo.vstack([_few, _out] if _few is not None else [_out])
    return (STOCKS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The data and the split

    The 1-minute bars of every stock from TRAIN's first day to TEST's last, and every input built
    from them (`tsmixer_lab.build`, the same inputs as the other TSMixer notebooks). Table 2 shows
    the days:

    - **history only**: the first 10 sessions, read for the typical sizes, never learned
    - **TRAIN**: the model learns; **every `HELD_EVERY`-th calendar week is held out** to stop
      the training. Spread over the year, the held-out weeks see calm and busy months alike: the
      last 10 days of 2025, the old held-out set, were the holidays (50 stock-days of 5 stocks,
      7 up and 7 down: too few to tell one pass from the next)
    - **VALIDATION**: every trial is scored here; **TEST**: read once, at the very end

    Every move starts at the **fill**, the first trade after 13:30, not at the last trade of the
    13:29 minute: that one sits on the bid or the ask, and no call can trade at it.

    **Table 3** counts the classes: how many stock-days went down, stayed flat, went up. A model
    that always says "flat" is right on the biggest class, which is why only the calls count. It
    also gives the **prior side**: the side TRAIN's moves leaned to, which is what a model that
    knows nothing about the direction calls.
    """)
    return


@app.cell
def _(
    DEFAULT,
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
    _hist = np.arange(_train[0])
    _span = lambda i: (_dates[i[0]], _dates[i[-1]])  # noqa: E731
    _split = pd.DataFrame([
        {"part": p, "from": _span(i)[0], "to": _span(i)[1], "sessions": len(i), "use": u}
        for p, i, u in (("history only", _hist, "typical sizes: read, never learned"),
                        ("TRAIN", _train, "the model learns"),
                        ("TRAIN, held-out weeks", held_d,
                         f"every {HELD_EVERY}th week: stops the training"),
                        ("VALIDATION", val_d, "every trial is scored"),
                        ("TEST", test_d, "read ONCE, at the very end"))])
    PRIOR = ml.prior_side(F, DEFAULT, _train, THETA)  # VALIDATION's calls: TRAIN's lean
    PRIOR_T = ml.prior_side(F, DEFAULT, _learn_t, THETA)  # TEST's: TRAIN + VALIDATION's lean
    _cls = ml.labels(ml.moves(F, DEFAULT), THETA)
    _ok = lab.valid(F, DEFAULT)
    _rows = []
    for _name, _days in (("TRAIN", _train), ("VALIDATION", val_d)):
        _c = _cls[_days][_ok[_days] & (_cls[_days] >= 0)]
        _rows.append({"part": _name, "stock-days": len(_c),
                      **{f"{n}": float(np.mean(_c == i)) for i, n in enumerate(ml.CLASSES)},
                      "up, of the moves": float(np.sum(_c == 2) / np.sum(_c != 1))})
    _side = "long" if PRIOR == 2 else "short"
    mo.vstack([
        mo.md(f"**{len(F['names'])}** stocks, **{len(_dates)}** sessions loaded "
              f"({_dates[0]} → {_dates[-1]})."),
        mo.md("### Table 2 · The data split"), table(_split),
        mo.md(f"### Table 3 · The classes (a move is at least {THETA:g}% up or down; TEST's "
              "stay unread until the end)"),
        table(pd.DataFrame(_rows).round(3)),
        mo.md(f"On TRAIN, **{_rows[0]['up, of the moves']:.0%}** of the moves went up: the prior "
              f"side is **{_side}**. A model that knows nothing about the direction calls every "
              f"stock {_side}, and is right on that share of its picks that move.")])
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
      to. **This is the rule to beat.** A coin is the wrong bar for the side: when the market
      rises, "always long" is right on more than half of the moves without knowing anything. A
      model that only learned which stocks are active and which way TRAIN leaned ties it; only
      knowing something about *today* or the *direction* beats it
    """)
    return


@app.cell(hide_code=True)
def _(DEFAULT, F, K, PRIOR, THETA, fit_d, held_d, lab, ml, mo, np, pd, table, val_d):
    _cls = ml.labels(ml.moves(F, DEFAULT), THETA)
    _act = ml.activity(F, DEFAULT, THETA)["mean"]
    _ok = lab.valid(F, DEFAULT) & (_cls >= 0) & np.isfinite(_act)
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
    _stock = pd.DataFrame({
        "stock": F["names"],
        "VALIDATION days that moved": [float((_cls[val_d, k][_ok[val_d, k]] != 1).mean())
                                       for k in range(len(F["names"]))]}
    ).sort_values("VALIDATION days that moved", ascending=False)
    mo.vstack([
        mo.md(f"### Table 4 · The hurdle: {K} calls a day, no model"),
        table(pd.DataFrame(_rows).round(3)),
        mo.md("### Table 4b · How often each stock moves (VALIDATION)"),
        table(_stock.round(3), rows=10)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The study

    **Table 5** lists the knobs the optimizer searches. A trial draws them, trains one TSMixer on
    every stock's TRAIN days (the loss is the cross-entropy of the three classes, every stock-day
    weighing the same; only `flat_weight` can turn the flat class down), stops on the held-out
    weeks, and calls `K` stocks a day on VALIDATION.

    The trial's score is the **z of its excess over the activity rule on the prior side**: per
    day, the share of its calls that are right minus that of the activity picks all called on
    the side TRAIN leaned to, averaged over days, divided by its standard error over days. It is
    0 when the model knows what the activity rule and the prior side know and nothing more; above
    0, the model adds something. The best of many trials is partly luck: the line above Table 6
    gives the z the best of this many trials reaches by chance alone (2.5% of the time), and
    Figure 3 holds the best trial against its own luck band.
    """)
    return


@app.cell
def _(mo, pd, table):
    # TSMixer's knobs: [choices] | (low, high) integers | (low, high) floats |
    # (low, high, "log") on a log scale; "lr" / "weight_decay" (AdamW) and "flat_weight" (the
    # loss weight of the flat class) are training knobs.
    SEARCH = {
        "lr": (1e-4, 1e-2, "log"), "weight_decay": (1e-7, 1e-2, "log"),
        "flat_weight": (0.1, 1.0, "log"),
        "hidden_size": (16, 256, "log"), "ff_size": (16, 256, "log"), "num_blocks": (1, 4),
        "activation": ["ReLU", "GELU", "SELU"],
        "norm_type": ["LayerNorm", "LayerNormNoBias", "TimeBatchNorm2d"],
        "normalize_before": [False, True], "dropout": (0.0, 0.3),
    }
    run_study = mo.ui.run_button(label="run the study")
    mo.vstack([
        mo.md("### Table 5 · TSMixer's knobs the optimizer searches"),
        table(pd.DataFrame([(k, str(v)) for k, v in SEARCH.items()],
                           columns=["knob", "range"]), rows=12),
        run_study,
    ])
    return SEARCH, run_study


@app.cell(hide_code=True)
def _(
    DECISION,
    EPOCHS,
    F,
    HORIZON,
    INPUTS,
    K,
    LENGTHS,
    PATIENCE,
    PREV_DAYS,
    PRIOR,
    SEARCH,
    STEP,
    STOCK_ID,
    THETA,
    fit_d,
    held_d,
    lab,
    ml,
    np,
    val_d,
):
    import optuna

    TRIALS_DIR = ml.VAR / "trials"  # every trial's probabilities (git-ignored)

    def suggest(trial, name, spec):
        """One knob, drawn as its SEARCH declaration says (a range is drawn as a number)."""
        if isinstance(spec, list):
            return trial.suggest_categorical(name, spec)
        lo, hi, *log = spec
        draw = trial.suggest_int if isinstance(lo, int) else trial.suggest_float
        return draw(name, lo, hi, log=bool(log))

    def draw(trial):
        """Every knob of one trial: the history length, the inputs set to "search", the
        previous days' length and TSMixer's knobs."""
        length = trial.suggest_int("history", LENGTHS[0], LENGTHS[1], step=STEP)
        feats = [f for f, v in INPUTS.items()
                 if v is True or (v == "search" and trial.suggest_categorical(f"use_{f}",
                                                                              [True, False]))]
        prev_days = (trial.suggest_categorical("prev_days", PREV_DAYS)
                     if "prev" in feats else PREV_DAYS[0])
        hp = {n: suggest(trial, n, s) for n, s in SEARCH.items()}
        return (1, DECISION - length, length, HORIZON), feats, prev_days, hp

    def run(trial):
        """The trial function: draw, train ONE TSMixer on every stock's TRAIN days, call the
        K likeliest movers of every VALIDATION day, score the calls."""
        cell, feats, prev_days, hp = draw(trial)
        kw = dict(prev_days=prev_days, stock_id=STOCK_ID, theta=THETA)
        model, info = ml.train(lab, F, cell, fit_d, held_d, feats, {**hp, "patience": PATIENCE},
                               epochs=EPOCHS, **kw)
        p, d, k = ml.predict_proba(lab, model, F, cell, val_d, feats, **kw)
        cls, act = ml.targets(F, cell, d, k, THETA)
        scores = ml.ladder(cls, d, act, p, K=K, prior=PRIOR)
        folder = TRIALS_DIR / trial.study.study_name
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"{trial.number}.npz", p=p, cls=cls, d=d, k=k, act=act)
        for key, val in {**scores, **info, "inputs": feats}.items():
            trial.set_user_attr(key, val if isinstance(val, (list, str)) else float(val))
        return float(np.nan_to_num(scores["excess_z_prior"], nan=-10.0))

    return TRIALS_DIR, optuna, run


@app.cell
def _(
    CAPITAL,
    DECISION,
    EPOCHS,
    HELD_EVERY,
    HISTORY,
    HORIZON,
    INPUTS,
    MAX_TRIALS,
    PATIENCE,
    POSITION,
    PREV_DAYS,
    SEARCH,
    STOCKS,
    STOCK_ID,
    TEST,
    THETA,
    TRAIN,
    VALIDATION,
    lab,
    ml,
    mo,
    optuna,
    run,
    run_study,
    strategies,
):
    import hashlib
    import json
    from pathlib import Path

    from trading_simple import optimize as opt

    _strat = strategies.load("darts_forecast")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _source = b"".join(Path(m.__file__).read_bytes() for m in (lab, ml))
    _code = hashlib.sha1(_source).hexdigest()[:10]
    _setup = [STOCKS, TRAIN, VALIDATION, TEST, DECISION, HORIZON, THETA, CAPITAL, POSITION, INPUTS,
              PREV_DAYS, STOCK_ID, HISTORY, SEARCH, EPOCHS, PATIENCE, HELD_EVERY,
              "excess z over activity, prior side", _code]
    STUDY = ("movers-h" + str(HORIZON) + "-"
             + hashlib.sha1(json.dumps(_setup, default=str).encode()).hexdigest()[:8])
    study = optuna.create_study(
        study_name=STUDY, storage=opt.storage_for(_strat, STUDY), load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True),
        pruner=optuna.pruners.NopPruner(),  # every trial trains to its own early stop
    )
    if "stocks" not in study.user_attrs:  # what the comparison table shows
        study.set_user_attr("stocks", STOCKS)
        study.set_user_attr("inputs", INPUTS)
    _todo = MAX_TRIALS - len(study.trials)  # top up to MAX_TRIALS, never past it
    if run_study.value and _todo > 0:
        study.optimize(run, n_trials=_todo, catch=(RuntimeError, ValueError),
                       callbacks=[lambda s, t: mo.output.replace(mo.md(
                           f"trial {t.number + 1} of {MAX_TRIALS}: {t.state.name.lower()}"))])
    _n = sum(t.state.name == "COMPLETE" for t in study.trials)
    mo.md(f"Study **{STUDY}**: {_n} of `MAX_TRIALS = {MAX_TRIALS}` trials complete. Journal: "
          f"`strategies/darts_forecast/studies/{STUDY}.log`; live: `uvx optuna-dashboard "
          f"strategies/darts_forecast/studies/{STUDY}.log`.")
    return (study,)


@app.cell(hide_code=True)
def _(K, PRIOR, SHUFFLES, TRIALS_DIR, charts, ex, hl, ml, mo, np, optuna, pd, study, table):
    import plotly.graph_objects as go

    _done = [t for t in study.trials if t.state.name == "COMPLETE"]
    mo.stop(not _done, mo.md("_No completed trial yet: press **run the study**._"))
    _keys = ("calls", "call_prec", "call_prior", "move_rate", "move_z_naive", "dir_acc",
             "dir_prior", "side_z", "excess_prior", "excess_z_random", "ap", "val_loss",
             "prior_loss", "epochs")
    _df = pd.DataFrame([{
        "trial": t.number, "excess z (vs activity, prior side)": t.value,
        **{k: t.user_attrs.get(k) for k in _keys},
        "inputs": ", ".join(t.user_attrs.get("inputs", [])), **t.params,
        "seconds": t.duration.total_seconds()} for t in _done])
    _df = _df.sort_values("excess z (vs activity, prior side)", ascending=False).reset_index(
        drop=True)
    for _c in ("lr", "weight_decay"):  # log-scale knobs: 4 decimals would show 0.0000
        if _c in _df:
            _df[_c] = _df[_c].map(lambda v: f"{v:.1e}")
    _best = max(_done, key=lambda t: t.value)
    _bar = ex.best_of_bar(len(_done))
    _first10 = [t.value for t in sorted(_done, key=lambda t: t.number)[:10]]
    _npz = np.load(TRIALS_DIR / study.study_name / f"{_best.number}.npz")
    BEST = ml.ladder(_npz["cls"], _npz["d"], _npz["act"], _npz["p"], K=K, prior=PRIOR)
    _band = hl.shuffle_band(_npz["cls"], _npz["d"], _npz["act"], {"best": _npz["p"]}, K=K,
                            prior=PRIOR, draws=SHUFFLES)["z"][:, 0]
    _luck = float(np.quantile(_band, 0.975))
    _side = "long" if PRIOR == 2 else "short"
    _rows = [
        ("share of the picks that moved (at least THETA)", BEST["move_random"],
         BEST["move_naive"], BEST["move_rate"], f"z {BEST['move_z_naive']:+.2f} vs activity"),
        ("share of the calls right (moved the called way)", BEST["call_random"],
         BEST["call_prior"], BEST["call_prec"],
         f"z {BEST['excess_z_prior']:+.2f} vs the activity picks called {_side}"),
        ("direction right, among the picks that moved", 0.5, BEST["dir_prior"], BEST["dir_acc"],
         f"side z {BEST['side_z']:+.2f} vs calling them all {_side}"),
        ("average precision of 'will it move', per day", BEST["ap_base"], BEST["ap_naive"],
         BEST["ap"], "chance / activity / model")]
    _ladder = pd.DataFrame(_rows, columns=["rung", "random", "the rule", "the model", "test"])
    _verdict = ("**above luck**" if _best.value >= _bar else
                "**within luck**: what the best of this many tries looks like by chance")
    _fig = go.Figure()
    for _name, _col, _vals in (("random picks", charts._MUTED, (BEST["move_random"],
                                                                BEST["call_random"])),
                               (f"activity picks, called {_side} (the hurdle)",
                                charts.SERIES[3], (BEST["move_naive"], BEST["call_prior"])),
                               ("the model", charts.SERIES[0],
                                (BEST["move_rate"], BEST["call_prec"]))):
        _fig.add_trace(go.Bar(x=["share of picks that moved", "share of calls right"],
                              y=list(_vals), name=_name, marker={"color": _col}))
    _fig.add_trace(go.Scatter(x=["share of calls right"], y=[BEST["call_prec"]], mode="markers",
                              showlegend=False, marker={"opacity": 0},
                              error_y={"type": "data", "array": [BEST["call_se"]],
                                       "visible": True}))
    _fig.update_layout(barmode="group")
    charts._base(_fig, title=f"Figure 1 · The best trial's calls against the two rules "
                 f"({K} calls a day, VALIDATION; bar: ± standard error over days)", height=420)
    _hist = optuna.visualization.plot_optimization_history(study)
    _fig3 = go.Figure(go.Histogram(x=_band, nbinsx=30, marker={"color": charts._MUTED},
                                   name="shuffled labels"))
    _fig3.add_vline(x=_best.value, line_color=charts.SERIES[0], annotation_text="the real labels")
    _fig3.add_vline(x=_luck, line_dash="dash", line_color=charts.SERIES[3],
                    annotation_text="97.5%")
    charts._base(_fig3, title=f"Figure 3 · The best trial's excess z against {SHUFFLES} "
                 "within-day label shuffles (its calls kept)", height=360)
    _own = "above" if _best.value > _luck else "inside"
    mo.vstack([
        mo.md(f"**Best trial #{_best.number}:** excess z **{_best.value:+.2f}** over the activity "
              f"picks called {_side}; {BEST['calls']} calls over {BEST['days']} days, "
              f"**{BEST['call_prec']:.1%}** right against {BEST['call_prior']:.1%} for that rule "
              f"({BEST['call_naive']:.1%} with a random side). The best of "
              f"{len(_done)} trials reaches z ≥ **{_bar:.2f}** by luck only 2.5% of the time → "
              f"{_verdict}. The first 10 trials (random draws, no selection) average "
              f"z **{np.mean(_first10):+.2f}**."),
        mo.md(f"Its own luck band (Figure 3: the labels shuffled inside each day, every day "
              f"keeping its number of down / flat / up) reaches **{_luck:+.2f}** 2.5% of the "
              f"time: the real z is **{_own}** it. The band reads one trial as if it were the "
              "only one; the "
              "bar above also pays for picking the best of all of them. A trial that calls some "
              "stocks against the prior side starts below 0 there: with shuffled labels those "
              "calls lose to the day's lean."),
        mo.md("### Table 6 · All completed trials, best first"), table(_df.round(4), rows=15),
        mo.md(f"### Table 7 · The best trial's ladder ({K} calls a day, VALIDATION)"),
        table(_ladder.round(3)), _fig,
        charts._base(_hist, title="Figure 2 · The score of every trial", height=380), _fig3])
    return (BEST,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The same trial, other seeds

    A trial's score carries the luck of its seed (its starting weights, the order of its
    batches). In the first 5-stock study, the side the trials leaned to swung from 3% to 100%
    long calls, a lean set more by where training started than by the data. **Table 7b** trains
    the best trial again with each of `SEEDS` (same knobs, same days) and scores it the same way.
    The study's own seed is the one the search picked, so it sits at the top by construction;
    the other seeds' mean is the trial's honest VALIDATION score. If the z moves by about 1 from
    seed to seed, the trial's rank in Table 6 is mostly its seed.
    """)
    return


@app.cell
def _(mo):
    run_seeds = mo.ui.run_button(label="re-run the best trial with SEEDS")
    mo.hstack([run_seeds], justify="start")
    return (run_seeds,)


@app.cell(hide_code=True)
def _(
    DECISION,
    EPOCHS,
    F,
    HORIZON,
    K,
    PATIENCE,
    PREV_DAYS,
    PRIOR,
    SEARCH,
    SEEDS,
    STOCK_ID,
    THETA,
    TRIALS_DIR,
    fit_d,
    held_d,
    lab,
    ml,
    mo,
    np,
    pd,
    run_seeds,
    study,
    table,
    val_d,
):
    mo.stop(not run_seeds.value, mo.md("_Press **re-run the best trial with SEEDS**._"))
    _done = [t for t in study.trials if t.state.name == "COMPLETE"]
    mo.stop(not _done, mo.md("_No completed trial yet._"))
    _best = max(_done, key=lambda t: t.value)
    _p, _feats = _best.params, list(_best.user_attrs["inputs"])
    _cell = (1, DECISION - _p["history"], _p["history"], HORIZON)
    _kw = dict(prev_days=_p.get("prev_days", PREV_DAYS[0]), stock_id=STOCK_ID, theta=THETA)

    def _row(seed, p, d, act, cls, epochs):
        """One seed's line: its score, its picking and its sides."""
        s = ml.ladder(cls, d, act, p, K=K, prior=PRIOR)
        return {"seed": seed, "excess z": s["excess_z_prior"], "calls right": s["call_prec"],
                "picking z": s["move_z_naive"], "side z": s["side_z"],
                "long calls": float(np.mean(ml.calls(p, d, act, K)[1] == 2)), "epochs": epochs}

    _npz = np.load(TRIALS_DIR / study.study_name / f"{_best.number}.npz")
    _rows = [_row("0 (the study)", _npz["p"], _npz["d"], _npz["act"], _npz["cls"],
                  _best.user_attrs["epochs"])]
    with mo.status.spinner(title=f"Training trial #{_best.number} with seeds {list(SEEDS)} ..."):
        for _s in SEEDS:
            _model, _info = ml.train(lab, F, _cell, fit_d, held_d, _feats,
                                     {**{n: _p[n] for n in SEARCH}, "patience": PATIENCE},
                                     epochs=EPOCHS, seed=_s, **_kw)
            _pv, _d, _k = ml.predict_proba(lab, _model, F, _cell, val_d, _feats, **_kw)
            _cls, _act = ml.targets(F, _cell, _d, _k, THETA)
            _rows.append(_row(str(_s), _pv, _d, _act, _cls, _info["epochs"]))
    SEED_RUNS = pd.DataFrame(_rows)
    _z = SEED_RUNS["excess z"]
    mo.vstack([
        mo.md(f"Trial #{_best.number} over {len(_z)} seeds: excess z from **{_z.min():+.2f}** to "
              f"**{_z.max():+.2f}** (SD {_z.std(ddof=1):.2f}). Without the study's own seed the "
              f"mean is **{_z.iloc[1:].mean():+.2f}**: the trial's honest VALIDATION score."),
        mo.md("### Table 7b · The best trial, trained again with other seeds (VALIDATION)"),
        table(SEED_RUNS.round(3))])
    return (SEED_RUNS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## TEST: read once

    The best trial is trained again on TRAIN and VALIDATION (every `HELD_EVERY`-th week of both
    held out to stop the training) with the same settings and seed, and calls the TEST days.
    **Press once**: whatever it says, the study must not be tuned on it afterwards. If the best
    trial was within luck on VALIDATION, TEST is the only honest look at whether it knows
    anything.
    """)
    return


@app.cell
def _(mo):
    read_test = mo.ui.run_button(label="read TEST (once)")
    mo.hstack([read_test], justify="start")
    return (read_test,)


@app.cell(hide_code=True)
def _(
    BEST,
    DECISION,
    EPOCHS,
    F,
    HORIZON,
    K,
    PATIENCE,
    PREV_DAYS,
    PRIOR_T,
    SEARCH,
    STOCK_ID,
    THETA,
    fit_t,
    held_t,
    lab,
    ml,
    mo,
    pd,
    read_test,
    study,
    table,
    test_d,
):
    mo.stop(not read_test.value, mo.md("_Press the button to read TEST (once)._"))
    _done = [t for t in study.trials if t.state.name == "COMPLETE"]
    mo.stop(not _done, mo.md("_No completed trial yet._"))
    _best = max(_done, key=lambda t: t.value)
    _p, _feats = _best.params, list(_best.user_attrs["inputs"])
    _cell = (1, DECISION - _p["history"], _p["history"], HORIZON)
    _kw = dict(prev_days=_p.get("prev_days", PREV_DAYS[0]), stock_id=STOCK_ID, theta=THETA)
    with mo.status.spinner(title=f"Training trial #{_best.number} again on TRAIN + VALIDATION ..."):
        _model, _ = ml.train(lab, F, _cell, fit_t, held_t, _feats,
                             {**{n: _p[n] for n in SEARCH}, "patience": PATIENCE},
                             epochs=EPOCHS, **_kw)
        _pt, _d, _k = ml.predict_proba(lab, _model, F, _cell, test_d, _feats, **_kw)
    _cls, _act = ml.targets(F, _cell, _d, _k, THETA)
    TEST_SCORES = ml.ladder(_cls, _d, _act, _pt, K=K, prior=PRIOR_T)
    _t = TEST_SCORES
    _side = "long" if PRIOR_T == 2 else "short"
    _rows = [
        ("share of the picks that moved", _t["move_random"], _t["move_naive"], _t["move_rate"],
         f"z {_t['move_z_naive']:+.2f} vs activity"),
        ("share of the calls right", _t["call_random"], _t["call_prior"], _t["call_prec"],
         f"z {_t['excess_z_prior']:+.2f} vs the activity picks called {_side}"),
        ("direction right, among the picks that moved", 0.5, _t["dir_prior"], _t["dir_acc"],
         f"side z {_t['side_z']:+.2f} vs calling them all {_side}")]
    _mix = ", ".join(f"{n} {(_cls == i).mean():.1%}" for i, n in enumerate(ml.CLASSES))
    mo.vstack([
        mo.md(f"TEST, {_t['calls']} calls over {_t['days']} days: **{_t['call_prec']:.1%}** "
              f"right against {_t['call_prior']:.1%} for the activity picks called {_side} "
              f"(VALIDATION: {BEST['call_prec']:.1%} against {BEST['call_prior']:.1%}). TEST's "
              f"classes: {_mix}."),
        mo.md("### Table 8 · TEST, once"),
        table(pd.DataFrame(_rows, columns=["rung", "random", "the rule", "the model",
                                           "test"]).round(3))])
    return


if __name__ == "__main__":
    app.run()
