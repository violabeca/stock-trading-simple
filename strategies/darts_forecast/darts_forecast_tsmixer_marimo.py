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
    # TSMixer deep dive: one model per stock

    The question stays the one of the day-start notebook: at a fixed time of day, will a stock be
    **up or down at the end of the horizon**, and by roughly how much? Stocks behave differently,
    and one TSMixer for seven stocks at once found nothing (best +0.05, z 1.3), so here **every
    stock gets its own study and its own model**, and the stocks are compared afterwards.

    - **40 stocks** in six groups: market cap (large, mid, small) × trading volume (high, low),
      spread over the sectors (Table 1)
    - **ten inputs** (Table 2), each switchable, chosen from the literature on intraday return
      predictability and available at least daily
    - **one split**: TRAIN (the model learns), VALIDATION (every trial is scored), TEST (read
      once, at the very end)
    - **the score**: the correlation of the predicted and the actual move to the end of the
      horizon, checked against luck (random up/down outcomes of the same sizes)
    - **run all stocks**: one study per stock with a progress bar; each stock's best model is
      saved; the leaderboard (Table 5) and the groups (Table 6) compare the stocks
    - **one stock in detail**: its trials, and which inputs its best model leans on
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
    imp = _helpers("tsmixer_importance")
    ds, ex = lab.ds, lab.ex  # daystart.py and explorer.py, as loaded by tsmixer_lab
    settings = load_settings()
    engine = make_engine(settings)
    return charts, ds, engine, ex, imp, lab, np, pd, settings, strategies


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Settings

    The next cell holds every setting, in six blocks. Edit a value and everything below re-runs.

    1. **The question** (fixed): when the prediction is made, and how far ahead it looks
    2. **The data**: the stocks with their group (each gets its own model) and three consecutive
       date ranges: **TRAIN** → **VALIDATION** → **TEST** (Table 3 and Figure 0 show the days)
    3. **The inputs** (Table 2): each one `True` (read), `False` (not read) or `"search"` (the
       optimizer decides, trial by trial); the stock's own path is always read
    4. **What the optimizer searches**: the history length and the trials per stock; TSMixer's
       own knobs are in Table 4
    5. **Training** (fixed): how many passes, when to stop
    6. **Luck check and records**: random outcomes per score; W&B on or off
    """)
    return


@app.cell
def _():
    # ── 1 · The question (fixed) ──────────────────────────────────────────────────────────
    DECISION = 150  # when the prediction is made: minutes after 09:30 (150 = 12:00)
    HORIZON = 60  # how far ahead, in minutes

    # ── 2 · The data (dates: (first day, last day), both included) ────────────────────────
    # One model per stock. Groups: market cap (large >= $10B, mid $2-10B, small < $2B) x
    # dollar volume (high / low: above / below the median of their cap group). All have full
    # 1-minute data since January 2025 and a trade in at least half of the minutes.
    STOCKS = {
        # large cap, high volume
        "COST": "large · high", "CRM": "large · high", "C": "large · high",
        "CMCSA": "large · high", "CAT": "large · high", "COP": "large · high",
        "CI": "large · high",
        # large cap, low volume
        "DOCN": "large · low", "E": "large · low", "CNA": "large · low", "CACI": "large · low",
        "EGP": "large · low", "EHC": "large · low", "CLH": "large · low",
        # mid cap, high volume
        "COLB": "mid · high", "CE": "mid · high", "CLF": "mid · high", "CROX": "mid · high",
        "CAG": "mid · high", "CNX": "mid · high", "CRUS": "mid · high", "CBSH": "mid · high",
        # mid cap, low volume
        "EFXT": "mid · low", "ECO": "mid · low", "CGAU": "mid · low", "CAE": "mid · low",
        "CON": "mid · low", "CATY": "mid · low", "CLMT": "mid · low", "DRH": "mid · low",
        # small cap, high volume
        "CRI": "small · high", "CNMD": "small · high", "DXC": "small · high",
        "EFC": "small · high", "COLL": "small · high",
        # small cap, low volume
        "CMRE": "small · low", "DOLE": "small · low", "CIM": "small · low", "DLX": "small · low",
        "CSWC": "small · low",
    }
    TRAIN = ("2025-01-01", "2025-12-31")  # the model learns here; its last 10 days are held out
    VALIDATION = ("2026-01-01", "2026-06-30")  # every trial is scored here
    TEST = ("2026-07-01", "2026-09-30")  # read ONCE, at the very end
    READ_LOCKED = True  # allow the months the darts study series reserves (Feb 2026 on)

    # ── 3 · The inputs (Table 2): True = read, False = not read, "search" = optimizer ─────
    INPUTS = {
        "volume": True, "range": True, "flow": True, "market": True, "sector": True,
        "gap": True, "prev": True, "same_hour": True,
        "options": False,  # option data exists only for MU, CSCO, ANET, AAL, COIN
    }
    PREV_DAYS = [1, 2, 5]  # the sessions behind "prev" (searched when it is read)

    # ── 4 · What the optimizer searches (TSMixer's knobs: Table 4) ────────────────────────
    HISTORY = (60, 150, 15)  # minutes read before the decision: (fewest, most, step)
    MAX_TRIALS = 40  # trials PER STOCK; raising it later tops every study up (10 = random only)

    # ── 5 · Training (fixed) ──────────────────────────────────────────────────────────────
    EPOCHS = 60  # at most this many passes over the TRAIN days
    PATIENCE = 5  # stop after this many passes without a better held-out loss (best kept)

    # ── 6 · Luck check and records ────────────────────────────────────────────────────────
    DRAWS = 200  # random up/down outcomes behind every luck band
    WANDB = False  # one offline W&B run per trial (adds about a second to every trial)
    return (
        DECISION,
        DRAWS,
        EPOCHS,
        HISTORY,
        HORIZON,
        INPUTS,
        MAX_TRIALS,
        PATIENCE,
        PREV_DAYS,
        READ_LOCKED,
        STOCKS,
        TEST,
        TRAIN,
        VALIDATION,
        WANDB,
    )


@app.cell(hide_code=True)
def _(DECISION, HISTORY, HORIZON, INPUTS, TEST, TRAIN, VALIDATION, ds, lab):
    _days = [*TRAIN, *VALIDATION, *TEST]
    if _days != sorted(_days) or len(set(_days)) != len(_days):
        raise ValueError("TRAIN, VALIDATION and TEST must be (first, last) dates, in that "
                         "order, without overlap")
    if DECISION + HORIZON > ds.SESSION:
        raise ValueError(f"deciding at {ds.clock(DECISION)} + {HORIZON} min ends after the close")
    _unknown = sorted(set(INPUTS) - set(lab.OPTIONAL))
    _bad = sorted(k for k, v in INPUTS.items() if v not in (True, False, "search"))
    if _unknown or _bad:
        raise ValueError(f"INPUTS: unknown {_unknown}; not True/False/'search': {_bad}")
    _lo, _hi, STEP = HISTORY
    LENGTHS = (max(_lo, HORIZON), min(_hi, DECISION))  # history: at least the horizon
    if LENGTHS[0] > LENGTHS[1]:
        raise ValueError(f"HISTORY {HISTORY}: no length between {HORIZON} and {DECISION} minutes")
    DEFAULT = (1, DECISION - LENGTHS[1], LENGTHS[1], HORIZON)  # the overview's cell (Figure 1)
    return DEFAULT, LENGTHS, STEP


@app.cell(hide_code=True)
def _(READ_LOCKED, TEST, TRAIN, VALIDATION, ds, engine, lab, np, settings):
    def stock_data(symbol):
        """One stock's 1-minute bars from TRAIN's first day to TEST's last (dividend-adjusted),
        every input built from them (``tsmixer_lab.build``) and its split: the session indices
        it learns on (``fit``), holds out (``held``: the last 10 TRAIN days), is scored on
        (``val``) and reads once at the end (``test``)."""
        tune_end = ds.rs.WINDOWS["tune"][1]  # later months: reserved by the darts study series
        if TEST[1] > tune_end and not READ_LOCKED:
            raise ValueError(f"TEST ends {TEST[1]}, after {tune_end}: set READ_LOCKED = True")
        panel = ds.rs.load(settings, engine, start=TRAIN[0], end=TEST[1], options=False,
                           symbols=[symbol], adjust=True)
        if symbol not in panel.symbols:
            raise ValueError(f"no 1-minute data for {symbol}")
        F = lab.build(panel, engine, basket_from=TRAIN[0], basket_to=TEST[1], options=True)
        dates = np.array(F["dates"], dtype=object)
        sp = {"sessions": list(dates)}
        for part, (a, b) in {"train": TRAIN, "validation": VALIDATION, "test": TEST}.items():
            inside = [d for d in dates if a <= d <= b]
            if not inside:
                raise ValueError(f"{symbol}: no stored session in {part.upper()} {a} → {b}")
            sp[part] = (inside[0], inside[-1])
        fit, held, val = ds.split_days(F["pp"], sp, "validation", TRAIN[0])
        test = np.flatnonzero(F["pp"]["usable"] & (dates >= sp["test"][0])
                              & (dates <= sp["test"][1]))
        return {"F": F, "SP": sp, "fit": fit, "held": held, "val": val, "test": test}

    return (stock_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The stocks

    **Table 1**: each stock's group, sector, market cap (today), trading volume and volatility
    (the three months before TRAIN), and the share of minutes with a trade (the whole window).
    About a minute the first time, then saved. The low-volume stocks trade in fewer minutes: a
    stock-day only becomes an example if the stock traded in at least half of the minutes the
    model reads. Every stock with full 1-minute data since January 2025 has a ticker from C to
    E (the order of the backfill), so the 40 do too.
    """)
    return


@app.cell(hide_code=True)
def _(STOCKS, TEST, TRAIN, engine, ex, mo, pd, table):
    _c = ex.candidates(engine, (TRAIN[0], TEST[1]), pool=list(STOCKS)).reindex(list(STOCKS))
    _stocks = pd.DataFrame({
        "stock": list(STOCKS), "group": list(STOCKS.values()), "sector": _c["sector"].values,
        "market cap $B": (_c["market_cap"] / 1e9).round(1).values,
        "traded $M a day": (_c["dollar_volume"] / 1e6).round(1).values,
        "minutes with a trade %": (100 * _c["traded"]).round().values,
        "volatility %/yr": (100 * _c["vol"]).round().values,
        "company": _c["name"].astype(str).str.replace(" Common Stock", "").str[:40].values})
    mo.vstack([mo.md("### Table 1 · The 40 stocks"), table(_stocks, rows=40)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## One stock: its inputs and its split

    Pick a stock (the same pick drives the detail sections at the end; loading it takes about
    20 seconds). **Table 2** lists the ten inputs: what each one is, why it may carry
    information (the literature), where it comes from, its setting, and on what share of the
    stock's days it has a value. **Table 3** and **Figure 0** show the split:

    - **history only**: the first 10 sessions, read for the typical sizes, never learned
    - **TRAIN**: the model learns; its **last 10 days are held out** to stop the training and
      to set the size scale β
    - **VALIDATION**: every trial is scored here; **TEST**: read once, at the very end

    Every input is in the stock's own typical units, measured on earlier sessions only. The
    market and sector inputs come from a basket of every stock with full 1-minute data (equal
    weights; the sector basket leaves the stock itself out). Only data available at least daily
    is used: news, ratings and earnings dates are sporadic, short interest is twice a month.
    """)
    return


@app.cell
def _(STOCKS, mo):
    stock_pick = mo.ui.dropdown(list(STOCKS), value="COLB" if "COLB" in STOCKS else
                                list(STOCKS)[0], label="stock", searchable=True)
    mo.hstack([stock_pick], justify="start")
    return (stock_pick,)


@app.cell(hide_code=True)
def _(INPUTS, charts, lab, mo, np, pd, stock_data, stock_pick, table):
    with mo.status.spinner(title=f"Loading {stock_pick.value} ..."):
        one = stock_data(stock_pick.value)
    _F, _s = one["F"], stock_pick.value
    _rows = [{
        "input": f, "level": "per minute" if v["level"] == "bar" else "per day",
        "setting": "always" if f == "path" else str(INPUTS[f]), "what it is": v["what"],
        "why (literature)": v["why"], "source": v["source"],
        f"has a value ({_s}'s days)": f"{_F['available'].loc[_s, f]:.0%}"}
        for f, v in lab.FEATURES.items()]
    _dates = np.array(_F["dates"], dtype=object)
    _span = lambda i: (_dates[i[0]], _dates[i[-1]])  # noqa: E731
    _hist = np.arange(one["fit"][0])
    _parts = (("history only", _hist, "typical sizes: read, never learned"),
              ("TRAIN", one["fit"], "the model learns"),
              ("TRAIN, held out", one["held"], "early stopping and the size scale β"),
              ("VALIDATION", one["val"], "every trial is scored"),
              ("TEST", one["test"], "read ONCE, at the very end"))
    _split = pd.DataFrame([{"part": p, "from": _span(i)[0], "to": _span(i)[1],
                            "sessions": len(i), "use": u} for p, i, u in _parts])
    _fig = charts.split_calendar(
        [("the model", [("history only", *_span(_hist)), ("learns", *_span(one["fit"])),
                        ("held out (early stop, β)", *_span(one["held"])),
                        ("scores", *_span(one["val"]))]),
         ("TEST", [("TEST (read once)", *_span(one["test"]))])],
        title=f"Figure 0 · {_s}: the data split on the calendar")
    mo.vstack([
        mo.md("### Table 2 · The ten inputs"), table(pd.DataFrame(_rows), rows=10),
        mo.md(f"### Table 3 · {_s}: the data split"), table(_split), _fig,
    ])
    return (one,)


@app.cell
def _(mo, one):
    _days = [d for d in one["F"]["dates"]
             if one["SP"]["train"][0] <= d <= one["SP"]["validation"][1]]
    day_pick = mo.ui.dropdown(_days, value=one["SP"]["validation"][0], label="day",
                              searchable=True)
    mo.hstack([day_pick], justify="start")
    return (day_pick,)


@app.cell(hide_code=True)
def _(DEFAULT, PREV_DAYS, charts, day_pick, ds, lab, mo, np, one, pd, stock_pick):
    _F, _s = one["F"], stock_pick.value
    _, _off, _L, _H = DEFAULT
    _D = _off + _L
    _d, _k = list(_F["dates"]).index(day_pick.value), 0
    _bars = {"path (typical moves)": _F["y"][_d, _off:_D, _k]}
    _bars |= {f: _F["bar"][f][_d, _off:_D, _k] for f in lab.OPTIONAL
              if lab.FEATURES[f]["level"] == "bar"}
    _io = {"name": _s, "day": day_pick.value, "cell": DEFAULT,
           "times": pd.date_range(f"{day_pick.value} 09:31", periods=ds.SESSION, freq="min"),
           "price": _F["pp"]["cf"][_d, :, _k], "bars": _bars,
           "truth": _F["y"][_d, _D:_D + _H, _k]}
    _day = {"gap": lab.day_inputs(_F, DEFAULT, ["gap"])[_d, _k, 0]}
    _day |= {f"prev ({p} days)": _F["prev_move"][p][_d, _k] for p in PREV_DAYS}
    _day["same hour"] = lab.day_inputs(_F, DEFAULT, ["same_hour"])[_d, _k, 0]
    _day |= {f"option {s}": (_F["options"][_d, _k, i] if _F["options"] is not None else np.nan)
             for i, s in enumerate(lab.OPTION_SIGNALS)}
    _ok = bool(lab.valid(_F, DEFAULT)[_d, _k])
    mo.vstack([
        mo.md(f"**{_s}** on **{day_pick.value}**, reading {_L} minutes "
              f"({ds.clock(_off)}–{ds.clock(_D)}) to predict {_H} "
              f"({ds.clock(_D)}–{ds.clock(_D + _H)}): "
              + ("an example." if _ok else "**not an example** (no data, or too few trades).")),
        charts.inputs_picture(_io),
        charts.day_inputs_picture(_day, title=f"Figure 2 · {_s} on {day_pick.value}: the "
                                  "per-day inputs", limit=lab.DAY_CLIP),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Run all stocks: one study and one model per stock

    **run all stocks** runs one Optuna study per stock, `MAX_TRIALS` trials each, one stock
    after the other; the progress bar shows the stock, the trial, the stock's best so far and
    the time left. A trial draws the history length, the inputs set to `"search"`, the previous
    days' length and TSMixer's knobs (Table 4), trains ONE TSMixer on the stock's TRAIN days and
    scores it on its VALIDATION days.

    - **score** (maximized): the end-move correlation; the luck check flips each day's move at
      random; also: hit rate, size error after a scale β from the held-out days, per month
    - **resumable**: each stock's study is a journal in `studies/`
      (`tsmixer-lab-h<horizon>-<batch>-<stock>`; the batch is a fingerprint of every setting
      except the stocks and `MAX_TRIALS`, and of `tsmixer_lab.py`). Pressing the button again
      tops every study up to `MAX_TRIALS`: an interrupted run continues where it stopped, and a
      first pass with `MAX_TRIALS = 10` (random trials only) can be extended later
    - **the models**: whenever a trial beats its stock's best so far, its weights are saved to
      `var/tsmixer_lab/models/<batch>/<stock>.ckpt`, its settings and scores beside them in
      `<stock>.json` (`tsmixer_lab.load_model` reads them back); every trial's predictions go
      to `var/tsmixer_lab/trials/`
    - **time**: about 20 seconds to load a stock, then 2–4 seconds per trial: 40 stocks × 40
      trials is about 1–2 hours. Watch one stock live with
      `uvx optuna-dashboard strategies/darts_forecast/studies/<study>.log`
    """)
    return


@app.cell
def _(mo, pd, table):
    # TSMixer's knobs: [choices] | (low, high) integers | (low, high) floats |
    # (low, high, "log") on a log scale; "lr" / "weight_decay" (AdamW) and "loss" ("bar" |
    # "endpoint": only the move to the end counts) are training knobs.
    SEARCH = {
        "lr": (1e-4, 1e-2, "log"), "weight_decay": (1e-7, 1e-2, "log"),
        "loss": ["bar", "endpoint"],
        "hidden_size": (16, 256, "log"), "ff_size": (16, 256, "log"), "num_blocks": (1, 4),
        "activation": ["ReLU", "GELU", "SELU"],
        "norm_type": ["LayerNorm", "LayerNormNoBias", "TimeBatchNorm2d"],
        "normalize_before": [False, True], "dropout": (0.0, 0.3),
    }
    run_all = mo.ui.run_button(label="run all stocks")
    mo.vstack([
        mo.md("### Table 4 · TSMixer's knobs the optimizer searches"),
        table(pd.DataFrame([(k, str(v)) for k, v in SEARCH.items()],
                           columns=["knob", "range"]), rows=12),
        run_all,
    ])
    return SEARCH, run_all


@app.cell(hide_code=True)
def _(
    DECISION,
    EPOCHS,
    HISTORY,
    HORIZON,
    INPUTS,
    PATIENCE,
    PREV_DAYS,
    SEARCH,
    STOCKS,
    TEST,
    TRAIN,
    VALIDATION,
    lab,
    strategies,
):
    import hashlib
    import json
    from pathlib import Path

    import optuna

    from trading_simple import optimize as opt

    _strat = strategies.load("darts_forecast")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _code = hashlib.sha1(Path(lab.__file__).read_bytes()).hexdigest()[:10]
    _setup = [TRAIN, VALIDATION, TEST, DECISION, HORIZON, INPUTS, PREV_DAYS, HISTORY, SEARCH,
              EPOCHS, PATIENCE, "end-move correlation, one model per stock", _code]
    BATCH = "b" + hashlib.sha1(json.dumps(_setup, default=str).encode()).hexdigest()[:8]
    MODELS = lab.BASKET_DIR / "models" / BATCH  # each stock's best model (git-ignored)
    TRIALS_DIR = lab.BASKET_DIR / "trials"  # every trial's predictions (git-ignored)

    def open_study(symbol, create=False):
        """One stock's study of this batch, from its journal; None when it has none yet
        (unless ``create``)."""
        name = f"tsmixer-lab-h{HORIZON}-{BATCH}-{symbol}"
        if not create and not opt.study_path(_strat, name).exists():
            return None
        study = optuna.create_study(
            study_name=name, storage=opt.storage_for(_strat, name), load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, group=True),
            pruner=optuna.pruners.NopPruner(),  # every trial trains to its own early stop
        )
        if create and "batch" not in study.user_attrs:  # what the comparison shows
            for key, value in {"batch": BATCH, "stock": symbol, "group": STOCKS[symbol],
                               "inputs": INPUTS}.items():
                study.set_user_attr(key, value)
        return study

    return BATCH, MODELS, TRIALS_DIR, json, open_study, optuna


@app.cell(hide_code=True)
def _(
    BATCH,
    DECISION,
    DRAWS,
    EPOCHS,
    HORIZON,
    INPUTS,
    LENGTHS,
    MODELS,
    PATIENCE,
    PREV_DAYS,
    SEARCH,
    STEP,
    STOCKS,
    TRIALS_DIR,
    ex,
    json,
    lab,
    np,
    optuna,
    pd,
):
    def suggest(trial, name, spec):
        """One knob, drawn as its SEARCH declaration says (a range is drawn as a number)."""
        if isinstance(spec, list):
            return trial.suggest_categorical(name, spec)
        lo, hi, *log = spec
        draw = trial.suggest_int if isinstance(lo, int) else trial.suggest_float
        return draw(name, lo, hi, log=bool(log))

    def make_objective(symbol, data, log_wandb=False):
        """The trial function of one stock: draw the knobs, train ONE TSMixer on the stock's
        TRAIN days, scale on its held-out days, score on its VALIDATION days, and save the
        model whenever it beats the stock's best so far."""
        F = data["F"]
        if INPUTS["options"] is True and F["available"].loc[symbol, "options"] < 0.5:
            raise ValueError(f"{symbol} has no option data: set INPUTS['options'] = False")

        def objective(trial):
            length = trial.suggest_int("history", LENGTHS[0], LENGTHS[1], step=STEP)
            feats = [f for f, v in INPUTS.items()
                     if v is True or (v == "search" and trial.suggest_categorical(
                         f"use_{f}", [True, False]))]
            prev_days = (trial.suggest_categorical("prev_days", PREV_DAYS)
                         if "prev" in feats else PREV_DAYS[0])
            hp = {n: suggest(trial, n, s) for n, s in SEARCH.items()}
            cell = (1, DECISION - length, length, HORIZON)

            def report(curve):  # each pass's held-out R²: the trial's learning curve
                r = curve.rows[-1]
                trial.report(r["r2"], r["pass"])

            kw = {"prev_days": prev_days}
            model, curve = lab.train(F, cell, data["fit"], data["held"], feats,
                                     {**hp, "patience": PATIENCE}, epochs=EPOCHS,
                                     on_pass=report, **kw)
            ph, dh, kh = lab.predict(model, F, cell, data["held"], feats, **kw)
            beta = ex.calibration(ph.sum(1)[:, None], lab.end_moves(F, cell, dh, kh)[:, None])
            raw, d, k = lab.predict(model, F, cell, data["val"], feats, **kw)
            ae = lab.end_moves(F, cell, d, k)
            scores, _, by_month = lab.grade(raw.sum(1), ae, d, k, F, beta=beta, draws=DRAWS)
            folder = TRIALS_DIR / trial.study.study_name
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f"{trial.number}.npz", raw=raw, d=d, act_end=ae,
                                date=np.array([F["dates"][x] for x in d]))
            for key, val in {**scores, "days": len(d), "passes": len(curve.rows),
                             "inputs": feats, "month_corr": by_month}.items():
                if key != "end_corr":
                    trial.set_user_attr(key, val)
            earlier = [t.value for t in trial.study.get_trials(
                deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))]
            if not earlier or scores["end_corr"] > max(earlier):  # the stock's best so far
                MODELS.mkdir(parents=True, exist_ok=True)
                lab.save_model(model, MODELS / f"{symbol}.ckpt")
                (MODELS / f"{symbol}.json").write_text(json.dumps(
                    {"stock": symbol, "group": STOCKS[symbol], "batch": BATCH,
                     "study": trial.study.study_name, "trial": trial.number, "cell": cell,
                     "inputs": feats, "prev_days": prev_days, "knobs": hp, "beta": beta,
                     **scores, "month_corr": by_month}, indent=1, default=str))
            if log_wandb:
                import wandb

                wandb.run.log({"curve": wandb.Table(dataframe=pd.DataFrame(curve.rows)),
                               **scores, **{f"month_corr/{m}": v for m, v in by_month.items()}},
                              step=trial.number)
            return scores["end_corr"]

        return objective

    return (make_objective,)


@app.cell
def _(
    MAX_TRIALS,
    STOCKS,
    WANDB,
    lab,
    make_objective,
    mo,
    open_study,
    pd,
    run_all,
    stock_data,
    strategies,
    table,
):
    import os
    import time

    _studies = {s: open_study(s) for s in STOCKS}
    _todo = {s: MAX_TRIALS - (len(st.trials) if st else 0) for s, st in _studies.items()}
    _total = sum(max(n, 0) for n in _todo.values())
    _log = []
    if run_all.value and _total:
        if WANDB:
            import optunahub
            from dotenv import dotenv_values

            _root = strategies.STRATEGIES_DIR.parent
            for _k, _v in dotenv_values(_root / ".env").items():  # W&B settings; env wins
                if _k.startswith("WANDB_") and _v:
                    os.environ.setdefault(_k, _v)
            os.environ.setdefault("WANDB_MODE", "offline")
            os.environ.setdefault("WANDB_SILENT", "true")
            _hub = optunahub.load_module("callbacks/wandb",
                                         ref="618066c693941399e0890b03fd3a516aec141bfe")
        _t0 = time.time()
        with mo.status.progress_bar(total=_total, title="Running all stocks", show_eta=True,
                                    show_rate=True, completion_title="All stocks done") as _bar:
            for _i, (_s, _n) in enumerate(_todo.items(), start=1):
                if _n <= 0:
                    continue
                _where = f"{_s} ({STOCKS[_s]}), stock {_i} of {len(STOCKS)}"
                _bar.update(0, subtitle=f"{_where}: loading the data")
                _ts = time.time()
                try:
                    _objective = make_objective(_s, stock_data(_s), log_wandb=WANDB)
                except ValueError as _e:  # no data, or none in a part: skip the stock
                    _bar.update(_n, subtitle=f"{_where}: skipped")
                    _log.append({"stock": _s, "group": STOCKS[_s], "trials run": 0,
                                 "note": f"skipped: {_e}"})
                    continue
                _study = open_study(_s, create=True)
                _cbs = []
                if WANDB:
                    _wandbc = _hub.WeightsAndBiasesCallback(
                        metric_name="end_corr", as_multirun=True,
                        wandb_kwargs={"project": "darts-tsmixer-lab", "group": _study.study_name,
                                      "dir": lab.BASKET_DIR.parent / "wandb"})
                    _objective = _wandbc.track_in_wandb()(_objective)
                    _cbs.append(_wandbc)

                def _tick(study, trial, bar=_bar, where=_where):
                    done = [t.value for t in study.trials if t.state.name == "COMPLETE"]
                    best = f", best so far {max(done):+.3f}" if done else ""
                    bar.update(1, subtitle=f"{where}: trial {trial.number + 1} of "
                                           f"{MAX_TRIALS}{best}")

                _study.optimize(_objective, n_trials=_n, catch=(RuntimeError, ValueError),
                                callbacks=[*_cbs, _tick])
                _done = [t for t in _study.trials if t.state.name == "COMPLETE"]
                _best = max(_done, key=lambda t: t.value) if _done else None
                _failed = sum(t.state.name == "FAIL" for t in _study.trials[-_n:])
                _log.append({
                    "stock": _s, "group": STOCKS[_s], "trials run": _n,
                    "best end corr": round(_best.value, 3) if _best else None,
                    "best z": round(_best.user_attrs["luck_z"], 2) if _best else None,
                    "minutes": round((time.time() - _ts) / 60, 1),
                    "note": f"{_failed} trials failed" if _failed else ""})
                mo.output.append(mo.md(f"{_s}: done in {_log[-1]['minutes']} min, best end "
                                       f"corr {_log[-1]['best end corr']}"))
        _msg = (f"**This run:** {_total} trials over {len(_log)} stocks in "
                f"{(time.time() - _t0) / 60:.0f} min. The leaderboard (Table 5) is below.")
    elif _total:
        _msg = (f"**{_total} trials to go** over {sum(n > 0 for n in _todo.values())} stocks "
                f"(`MAX_TRIALS = {MAX_TRIALS}` per stock): press **run all stocks**.")
    else:
        _msg = f"**All {len(STOCKS)} stocks complete** (`MAX_TRIALS = {MAX_TRIALS}` each)."
    run_finished = time.time()  # the cells below re-read the studies after every run
    mo.vstack([mo.md(_msg), table(pd.DataFrame(_log), rows=40)] if _log else [mo.md(_msg)])
    return (run_finished,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The leaderboard: which stocks have an edge?

    **Table 5** ranks the stocks by their best trial's luck z. Two warnings:

    - **a stock's best is the best of its trials**: compare its z with **its bar**, the z the
      best of that many trials reaches by luck 2.5% of the time. The **first 10 trials** are
      random draws, not chosen by the optimizer: their mean shows the stock without selection
    - **picking the best stocks is another selection**: the best of 40 stocks, each the best
      of its trials, has to clear the bar for ALL their trials together (the line above
      Table 5). Whichever stocks look strongest here must be confirmed on TEST, read once

    **Table 6** sums up the six groups; **Figure 3** draws every stock's best z against its bar.
    """)
    return


@app.cell(hide_code=True)
def _(STOCKS, charts, ex, mo, np, open_study, pd, run_finished, table):
    import plotly.graph_objects as go

    _ = run_finished  # re-read after every run
    _rows = []
    for _s, _g in STOCKS.items():
        _st = open_study(_s)
        _done = [t for t in _st.trials if t.state.name == "COMPLETE"] if _st else []
        if not _done:
            continue
        _best = max(_done, key=lambda t: t.value)
        _m = _best.user_attrs.get("month_corr", {})
        _rows.append({
            "stock": _s, "group": _g, "trials": len(_done), "best end corr": _best.value,
            "best z": _best.user_attrs["luck_z"], "its bar": ex.best_of_bar(len(_done)),
            "hit rate": _best.user_attrs["hit_rate"],
            "months positive": f"{sum(v > 0 for v in _m.values())} of {len(_m)}",
            "first 10 (random)": float(np.mean(
                [t.value for t in sorted(_done, key=lambda t: t.number)[:10]])),
            "median trial": float(np.median([t.value for t in _done])),
            "VALIDATION days": _best.user_attrs.get("days"),
            "history": _best.params["history"], "loss": _best.params["loss"],
            "inputs of the best": ", ".join(_best.user_attrs.get("inputs", []))})
    if not _rows:
        _out = [mo.md("_No stock has a completed trial yet: press **run all stocks**._")]
    else:
        _lb = pd.DataFrame(_rows)
        _lb.insert(6, "above its bar", _lb["best z"] >= _lb["its bar"])
        _lb = _lb.sort_values("best z", ascending=False).reset_index(drop=True)
        _n_all = int(_lb["trials"].sum())
        _bar_all = ex.best_of_bar(_n_all)
        _top = _lb.iloc[0]
        _groups = _lb.groupby("group", sort=False).agg(**{
            "stocks": ("stock", "size"), "mean best end corr": ("best end corr", "mean"),
            "mean best z": ("best z", "mean"), "above their bar": ("above its bar", "sum"),
            "mean first 10 (random)": ("first 10 (random)", "mean"),
            "mean hit rate": ("hit rate", "mean")}).reset_index()
        _colors = {g: charts.SERIES[i % len(charts.SERIES)]
                   for i, g in enumerate(dict.fromkeys(STOCKS.values()))}
        _plot = _lb.iloc[::-1]  # the best on top
        _fig = go.Figure()
        for _g, _part in _plot.groupby("group", sort=False):
            _fig.add_trace(go.Bar(x=_part["best z"], y=_part["stock"], orientation="h",
                                  name=_g, marker={"color": _colors[_g]}))
        _fig.add_trace(go.Scatter(x=_plot["its bar"], y=_plot["stock"], mode="markers",
                                  name="its bar (best of its trials)",
                                  marker={"symbol": "line-ns-open", "size": 14,
                                          "color": charts._INK,
                                          "line": {"color": charts._INK, "width": 2}}))
        _fig.add_vline(x=_bar_all, line={"color": charts._MUTED, "dash": "dash"})
        charts._base(_fig, title=f"Figure 3 · Every stock's best trial: luck z against its bar "
                     f"(dashed: the bar for the best of all {_n_all} trials, {_bar_all:.2f})",
                     height=220 + 18 * len(_plot))
        _fig.update_layout(barmode="overlay")
        _fig.update_yaxes(categoryorder="array", categoryarray=list(_plot["stock"]))
        _fig.update_xaxes(title_text="luck z of the stock's best trial")
        _out = [
            mo.md(f"**{int(_lb['above its bar'].sum())} of {len(_lb)}** stocks have a best "
                  f"trial above their own bar. The best of all **{_n_all}** trials has to "
                  f"reach z ≥ **{_bar_all:.2f}**; the top stock, **{_top['stock']}** "
                  f"({_top['group']}), reaches z {_top['best z']:+.2f}. The first 10 trials "
                  f"of every stock (random draws) average "
                  f"**{_lb['first 10 (random)'].mean():+.3f}**."),
            mo.md("### Table 5 · The leaderboard, best luck z first"),
            table(_lb.round(3), rows=40),
            mo.md("### Table 6 · The six groups"), table(_groups.round(3)), _fig,
        ]
    mo.vstack(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## One stock in detail

    The stock picked above (the pick is repeated here): its trials, best first (**Table 7**),
    its best trial's predicted against actual end moves (**Figure 4**), the score trial by
    trial (**Figure 5**) and which knobs and inputs moved it (**Figure 6**).
    """)
    return


@app.cell(hide_code=True)
def _(
    MODELS,
    TRIALS_DIR,
    charts,
    mo,
    np,
    open_study,
    optuna,
    pd,
    run_finished,
    stock_pick,
    table,
):
    import contextlib

    _ = run_finished  # re-read after every run
    _s = stock_pick.value
    _st = open_study(_s)
    _done = [t for t in _st.trials if t.state.name == "COMPLETE"] if _st else []
    if not _done:
        _out = [stock_pick, mo.md(f"_{_s} has no completed trial yet._")]
    else:
        _df = pd.DataFrame([{
            "trial": t.number, "end_corr": t.value, **{k: t.user_attrs.get(k) for k in (
                "luck_z", "hit_rate", "hit_p", "mae_skill", "beta", "months_positive",
                "passes")}, "inputs": ", ".join(t.user_attrs.get("inputs", [])), **t.params,
            "seconds": t.duration.total_seconds()} for t in _done])
        _df = _df.sort_values("end_corr", ascending=False).reset_index(drop=True)
        _best = max(_done, key=lambda t: t.value)
        _npz = np.load(TRIALS_DIR / _st.study_name / f"{_best.number}.npz")
        _pe = _npz["raw"].sum(axis=1) * _best.user_attrs["beta"]
        _model = MODELS / f"{_s}.ckpt"
        _vis = optuna.visualization
        _figs = [("Figure 5 · End-move correlation per trial",
                  _vis.plot_optimization_history(_st))]
        with contextlib.suppress(RuntimeError, ValueError, ZeroDivisionError):  # needs a few
            _figs.append(("Figure 6 · Which knobs and inputs move the score (importance)",
                          _vis.plot_param_importances(_st)))
        _out = [
            stock_pick,
            mo.md(f"**{_s}**: best trial #{_best.number}, end correlation "
                  f"**{_best.value:+.3f}**, luck z {_best.user_attrs['luck_z']:+.2f}, hit rate "
                  f"{_best.user_attrs['hit_rate']:.0%}; month by month "
                  + ", ".join(f"{m[2:]} {v:+.2f}"
                              for m, v in _best.user_attrs.get("month_corr", {}).items())
                  + f". Its model: `{_model.relative_to(MODELS.parents[3])}`"
                  + ("" if _model.exists() else " (**missing**)")),
            mo.md(f"### Table 7 · {_s}: every trial, best first"),
            table(_df.round(4), rows=10),
            charts.endpoint_scatter(
                _pe, _npz["act_end"], np.full(len(_pe), _s),
                title=f"Figure 4 · {_s}, best trial: predicted (× β "
                      f"{_best.user_attrs['beta']:.2f}) vs actual end move per VALIDATION day"),
        ]
        _out += [charts._base(f, title=t, height=400) for t, f in _figs]
    mo.vstack(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Which inputs does the stock's best model use?

    Measured **for the picked stock's best trial only**, on its saved model (loaded, not
    trained again), in two ways: scrambling (Table 8, Figure 7) and SHAP (Table 9, Figures 8
    and 9). Both ask what happens when an input is **absent**, and absent means: read from
    another day of the same stock, so the input keeps its usual size and only its link to that
    day's outcome is cut. VALIDATION days only; TEST stays unread.

    **Scrambling** (about a minute) scores VALIDATION again with one input scrambled at a time,
    ten times each, and shows how far the end correlation falls:

    - **drop near 0** (inside its sd): the model does not use that input, or it adds nothing
    - **drop clearly above its sd, and every scramble hurt**: the model leans on it, and it
      helps on days the model never learned from
    - **drop below 0**: the model scores *better* without it: the input hurts out of sample
    - **market + sector** is also scrambled together: they share most of their information
    - a drop cannot be much bigger than the score the model had: if the stock's best trial is
      within luck, every drop is small and noisy, and the table says what this one model leans
      on, not what predicts
    """)
    return


@app.cell
def _(mo):
    run_importance = mo.ui.run_button(label="measure the inputs of the best trial")
    mo.hstack([run_importance], justify="start")
    return (run_importance,)


@app.cell(hide_code=True)
def _(MODELS, json, lab, mo, open_study, run_importance, stock_pick):
    mo.stop(not run_importance.value,
            mo.md("_Press the button to measure the picked stock's best trial._"))
    _s = stock_pick.value
    _st = open_study(_s)
    _done = [t for t in _st.trials if t.state.name == "COMPLETE"] if _st else []
    mo.stop(not _done, mo.md(f"_{_s} has no completed trial yet._"))
    _best = max(_done, key=lambda t: t.value)
    _path = MODELS / f"{_s}.json"
    mo.stop(not _path.exists(), mo.md(f"_No saved model for {_s} (`{_path}`)._"))
    _meta = json.loads(_path.read_text())
    mo.stop(_meta["trial"] != _best.number,
            mo.md(f"_The saved model is trial #{_meta['trial']}, the best is #{_best.number}._"))
    best_run = {"stock": _s, "trial": _best, "cell": tuple(_meta["cell"]),
                "feats": _meta["inputs"], "prev": _meta["prev_days"],
                "model": lab.load_model(MODELS / f"{_s}.ckpt", tuple(_meta["cell"]),
                                        _meta["knobs"])}
    return (best_run,)


@app.cell(hide_code=True)
def _(best_run, charts, imp, lab, mo, one, table):
    import plotly.graph_objects as pgo

    mo.stop(best_run["stock"] != one["F"]["names"][0], mo.md("_Press the button again._"))
    _best = best_run["trial"]
    with mo.status.spinner(title="Scrambling every input, ten times each ..."):
        _base, perm_scores = imp.importance(
            lab, best_run["model"], one["F"], best_run["cell"], one["val"], best_run["feats"],
            prev_days=best_run["prev"], repeats=10)
    _show = perm_scores.round(3).assign(hurt=perm_scores["hurt"].map("{:.0%}".format)).rename(
        columns={"drop": "drop in end corr", "sd": "sd over scrambles",
                 "hurt": "scrambles that hurt", "scrambled corr": "end corr when scrambled"})
    _fig = pgo.Figure(pgo.Bar(
        x=perm_scores["input"], y=perm_scores["drop"], showlegend=False,
        error_y={"type": "data", "array": perm_scores["sd"], "visible": True}))
    _fig.add_hline(y=0, line={"color": charts._MUTED, "width": 1})  # after the trace
    charts._base(_fig, height=420, title=f"Figure 7 · {best_run['stock']}: how far the end "
                 "correlation falls when one input is scrambled (best trial, ± sd)")
    _same = abs(_base - _best.value) < 1e-6
    mo.vstack([
        mo.md(f"**{best_run['stock']}**, trial **#{_best.number}**: the saved model scores "
              f"**{_base:+.3f}** on VALIDATION"
              + (" (the study's score)." if _same else
                 f", the study recorded {_best.value:+.3f}: the data changed since.")),
        mo.md(f"### Table 8 · {best_run['stock']}: what the best trial's end correlation "
              "loses when an input is scrambled"),
        table(_show, rows=12), _fig,
    ])
    return perm_scores, pgo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### SHAP: what each input did to each prediction

    The same saved model, now with **Shapley values** (the `shap` library, which `darts`
    already requires): for one day, how far each input moved the predicted move to the end of
    the horizon away from what the model predicts with every input absent (an average over ten
    other days). The values add up exactly: *prediction = average with every input absent + the
    inputs' values*. They are **signed**, so they also say the direction, which scrambling
    cannot.

    - **mean |SHAP|**: how much the input moves predictions on average, whether or not that
      moves them the right way. High here and near 0 in Table 8 means the model moves with the
      input but it does not help on VALIDATION
    - **Figure 8**: every explained day; a wide spread is an input that matters on some days
    - **Figure 9** (per-day inputs: gap, previous days, same hour): the input's value against
      its SHAP value; a rising line means a high value pushes the prediction up
    - correlated inputs share the credit (market and sector)
    - about 1.5 seconds per explained day
    """)
    return


@app.cell
def _(mo):
    shap_explain = mo.ui.number(start=5, stop=60, step=5, value=20, label="days explained")
    run_shap = mo.ui.run_button(label="compute SHAP values")
    mo.hstack([shap_explain, run_shap], justify="start", align="end")
    return run_shap, shap_explain


@app.cell(hide_code=True)
def _(
    best_run,
    charts,
    imp,
    lab,
    mo,
    np,
    one,
    pd,
    perm_scores,
    pgo,
    run_shap,
    shap_explain,
    table,
):
    from plotly.subplots import make_subplots

    mo.stop(not run_shap.value,
            mo.md("_Press the button to compute SHAP values for the same model._"))
    with mo.status.spinner(title="Shapley values: every input present or absent, many orders ..."):
        _r = imp.shap_values(lab, best_run["model"], one["F"], best_run["cell"], one["val"],
                             best_run["feats"], prev_days=best_run["prev"],
                             explain=int(shap_explain.value))
    _v, _names = _r["values"], _r["names"]
    _gap = float(np.abs(_r["base"] + _v.sum(axis=1) - _r["pred"]).max())
    _shap = pd.DataFrame({
        "input": _names, "mean |SHAP|": np.abs(_v).mean(axis=0), "mean SHAP": _v.mean(axis=0),
        "sd": _v.std(axis=0), "pushes up": (_v > 0).mean(axis=0)})
    _shap = _shap.merge(perm_scores[["input", "drop"]].rename(
        columns={"drop": "drop in end corr (Table 8)"}), on="input", how="left")
    _shap = _shap.sort_values("mean |SHAP|", ascending=False).reset_index(drop=True)
    _show = _shap.round(4).assign(**{"pushes up": _shap["pushes up"].map("{:.0%}".format)})
    _strip = pgo.Figure()
    for _name in _shap["input"][::-1]:  # the biggest on top of a horizontal chart
        _j = _names.index(_name)
        _strip.add_trace(pgo.Box(
            x=_v[:, _j], name=_name, orientation="h", boxpoints="all", jitter=0.4, pointpos=0,
            marker={"size": 4}, showlegend=False, text=list(_r["date"]),
            hovertemplate="%{x:+.3f} typical moves<br>%{text}<extra></extra>"))
    _strip.add_vline(x=0, line={"color": charts._MUTED, "width": 1})  # after the traces
    charts._base(_strip, height=120 + 46 * len(_names), title="Figure 8 · SHAP value of every "
                 "input on every explained day (typical moves of the predicted end move)")
    _out = [
        mo.md(f"**{len(_v)}** days of **{best_run['stock']}** explained. The values add up to "
              f"the model's own predictions to within **{_gap:.1e}**."),
        mo.md("### Table 9 · What each input does to the best trial's predictions"),
        table(_show, rows=12), _strip]
    _scalars = list(_r["x"].columns)
    if _scalars:
        _dep = make_subplots(rows=1, cols=len(_scalars), subplot_titles=_scalars,
                             shared_yaxes=True)
        for _i, _f in enumerate(_scalars, start=1):
            _dep.add_trace(pgo.Scatter(
                x=_r["x"][_f], y=_v[:, _names.index(_f)], mode="markers", showlegend=False,
                marker={"size": 5}, text=list(_r["date"]),
                hovertemplate=f"{_f} %{{x:+.2f}}<br>SHAP %{{y:+.3f}}<br>%{{text}}"
                              "<extra></extra>"), row=1, col=_i)
            _dep.add_hline(y=0, line={"color": charts._MUTED, "width": 1}, row=1, col=_i)
        charts._base(_dep, height=380, title="Figure 9 · The input's value (in its typical "
                     "units) against its SHAP value: a rising line = a high value pushes the "
                     "prediction up")
        _out.append(_dep)
    mo.vstack(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare batches

    Every batch of this notebook (one per combination of inputs and settings, all stocks) side
    by side: switch an input off (or to `"search"`), run all stocks again, and its row appears.
    Compare batches on the same stocks; the fairest comparison of two input sets is the **mean
    of the first 10 trials** (random draws, not chosen by the optimizer). The earlier pooled
    study (one model for seven stocks) is listed too.
    """)
    return


@app.cell(hide_code=True)
def _(charts, ex, mo, np, optuna, pd, run_finished, strategies, table):
    import plotly.graph_objects as _go
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    _ = run_finished  # re-read after every run
    _rows = []
    for _path in sorted((strategies.STRATEGIES_DIR / "darts_forecast" / "studies").glob(
            "tsmixer-lab-*.log")):
        _st = optuna.load_study(study_name=_path.stem,
                                storage=JournalStorage(JournalFileBackend(str(_path))))
        _done = [t for t in _st.trials if t.state.name == "COMPLETE"]
        if not _done:
            continue
        _best = max(_done, key=lambda t: t.value)
        _inputs = _st.user_attrs.get("inputs", {})
        _rows.append({
            "batch": _st.user_attrs.get("batch", "pooled: " + ", ".join(
                _st.user_attrs.get("stocks", []))),
            "inputs read": ", ".join(k for k, v in _inputs.items() if v is True) or "path only",
            "searched": ", ".join(k for k, v in _inputs.items() if v == "search") or "–",
            "trials": len(_done), "best z": _best.user_attrs.get("luck_z", np.nan),
            "above bar": _best.user_attrs.get("luck_z", -9) >= ex.best_of_bar(len(_done)),
            "first 10": float(np.mean([t.value for t in sorted(
                _done, key=lambda t: t.number)[:10]]))})
    if not _rows:
        _out = [mo.md("_No study with completed trials yet._")]
    else:
        _all = pd.DataFrame(_rows)
        _sum = _all.groupby(["batch", "inputs read", "searched"], sort=False).agg(**{
            "studies (stocks)": ("trials", "size"), "trials": ("trials", "sum"),
            "mean first 10 (random)": ("first 10", "mean"), "mean best z": ("best z", "mean"),
            "stocks above their bar": ("above bar", "sum")}).round(3).reset_index()
        _fig = _go.Figure()
        for _b, _part in _all.groupby("batch", sort=False):
            _fig.add_trace(_go.Box(y=_part["best z"], name=_b, boxpoints="all", jitter=0.4,
                                   pointpos=0, marker={"size": 5}, showlegend=False))
        _fig.add_hline(y=0, line={"color": charts._MUTED, "width": 1})  # after the traces
        charts._base(_fig, title="Figure 10 · Every stock's best luck z, per batch", height=420)
        _out = [mo.md("### Table 10 · Every batch of this notebook"), table(_sum), _fig]
    mo.vstack(_out)
    return


if __name__ == "__main__":
    app.run()
